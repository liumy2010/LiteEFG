"""Compare FOW rules with an independent python-chess movement oracle.

The oracle implements the published MIT rule differences and does not import
the environment's movement or visibility helpers.  Install chess==1.11.2 to
run this optional suite.  Rules and example observations are published at
https://www.mit.edu/~6.7980/fow/starter/original-rules.md and
https://www.mit.edu/~6.7980/fow/starter/example-transcript.txt.
"""

from collections import Counter
from functools import partial
import random

import numpy as np
import pytest

chess = pytest.importorskip("chess")
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from LiteEFG.drl.env import _dark_chess as core


def _moves(board, side=None):
    """Ordinary pseudo-legal moves, plus castling without attack restrictions."""
    side = board.turn if side is None else side
    view = board.copy(stack=False)
    view.turn = side
    if side != board.turn:
        view.ep_square = None
    moves = [move for move in view.generate_pseudo_legal_moves()
             if not view.is_castling(move)]
    rank = 0 if side else 7
    king_square = chess.square(4, rank)
    if view.piece_at(king_square) == chess.Piece(chess.KING, side):
        for rook_file, destination, intervening in (
            (0, 2, (1, 2, 3)), (7, 6, (5, 6))
        ):
            rook_square = chess.square(rook_file, rank)
            has_right = bool(view.clean_castling_rights() & (1 << rook_square))
            if (has_right and
                    view.piece_at(rook_square) == chess.Piece(chess.ROOK, side) and
                    all(view.piece_at(chess.square(file, rank)) is None
                        for file in intervening)):
                moves.append(chess.Move(king_square, chess.square(destination, rank)))
    return moves


def _observations(board):
    result = np.full((2, 64), 7, dtype=np.int32)
    for player, side in enumerate((chess.WHITE, chess.BLACK)):
        seen = set(board.pieces(chess.PAWN, side))
        seen.update(square for square, piece in board.piece_map().items()
                    if piece.color == side)
        for move in _moves(board, side):
            seen.add(move.to_square)
            if side == board.turn and board.is_en_passant(move):
                seen.add(chess.square(chess.square_file(move.to_square),
                                      chess.square_rank(move.from_square)))
        for square in seen:
            piece = board.piece_at(square)
            result[player, square] = (0 if piece is None else
                                     piece.piece_type * (1 if piece.color else -1))
    return result


def _key(board):
    ep = board.ep_square if any(board.generate_pseudo_legal_ep()) else None
    return board.board_fen(), board.turn, board.clean_castling_rights(), ep


def _outcome(board, repetitions, plies, max_plies):
    previous_king = board.king(not board.turn)
    if previous_king is None or board.is_attacked_by(board.turn, previous_king):
        return True, 0 if board.turn else 1
    if repetitions[_key(board)] >= 3 or board.halfmove_clock >= 100:
        return True, -1
    if not _moves(board):
        return True, 1 if board.turn else 0
    return (True, -1) if plies >= max_plies else (False, -1)


def _underpromotion_actions():
    actions = {}
    action = 4096
    for source_rank, target_rank in ((6, 7), (1, 0)):
        for source_file in range(8):
            for target_file in range(max(0, source_file - 1),
                                     min(7, source_file + 1) + 1):
                for promotion in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
                    actions[(chess.square(source_file, source_rank),
                             chess.square(target_file, target_rank), promotion)] = action
                    action += 1
    assert action == 4228
    return actions


_UNDERPROMOTIONS = _underpromotion_actions()


def _action(move):
    if move.promotion not in (None, chess.QUEEN):
        return _UNDERPROMOTIONS[(move.from_square, move.to_square, move.promotion)]
    return move.from_square * 64 + move.to_square


def _state(board):
    state = core.initial_state()
    pieces = np.zeros(64, dtype=np.int32)
    for square, piece in board.piece_map().items():
        pieces[square] = piece.piece_type * (1 if piece.color else -1)
    rights = [[board.has_queenside_castling_rights(side),
               board.has_kingside_castling_rights(side)]
              for side in (chess.WHITE, chess.BLACK)]
    state = state._replace(
        board=jnp.asarray(pieces), turn=jnp.asarray(0 if board.turn else 1),
        castling_rights=jnp.asarray(rights),
        ep_square=jnp.asarray(-1 if board.ep_square is None else board.ep_square),
        halfmove_clock=jnp.asarray(board.halfmove_clock), ply=jnp.asarray(0),
        terminated=jnp.asarray(False), winner=jnp.asarray(-1),
        repetition_count=jnp.asarray(1),
    )
    return state._replace(repetition_keys=jnp.zeros_like(state.repetition_keys)
                          .at[0].set(core.position_key(state)))


_snapshot = jax.jit(lambda state: (core.legal_moves(state), core.observed_boards(state)))
_advance = jax.jit(partial(core.transition, max_plies=1000))


def _assert_position(state, board):
    expected_board = np.zeros(64, dtype=np.int32)
    for square, piece in board.piece_map().items():
        expected_board[square] = piece.piece_type * (1 if piece.color else -1)
    np.testing.assert_array_equal(state.board, expected_board, err_msg=board.fen())
    assert int(state.turn) == (0 if board.turn else 1)
    assert int(state.ep_square) == (-1 if board.ep_square is None else board.ep_square)
    assert int(state.halfmove_clock) == board.halfmove_clock
    expected_rights = [[board.has_queenside_castling_rights(side),
                        board.has_kingside_castling_rights(side)]
                       for side in (chess.WHITE, chess.BLACK)]
    np.testing.assert_array_equal(state.castling_rights, expected_rights)
    mask, observations = _snapshot(state)
    np.testing.assert_array_equal(observations, _observations(board), err_msg=board.fen())
    for player, side in enumerate((chess.WHITE, chess.BLACK)):
        expected = {_action(move) for move in _moves(board, side)}
        actual = set(np.flatnonzero(np.asarray(mask[player])))
        assert actual == expected, (board.fen(), player, actual - expected, expected - actual)


@pytest.mark.parametrize("seed", range(6))
def test_random_play_matches_independent_oracle(seed):
    rng = random.Random(seed)
    board = chess.Board()
    state = core.initial_state()
    repetitions = Counter({_key(board): 1})
    for ply in range(1, 161):
        _assert_position(state, board)
        move = rng.choice(_moves(board))
        state = _advance(state, jnp.asarray(_action(move)))
        board.push(move)
        repetitions[_key(board)] += 1
        terminal, winner = _outcome(board, repetitions, ply, 1000)
        assert bool(state.terminated) == terminal, (seed, ply, board.fen())
        assert int(state.winner) == winner, (seed, ply, board.fen())
        assert int(state.ply) == ply
        if terminal:
            _assert_position(state, board)
            break


@pytest.mark.parametrize("fen,uci_moves", [
    ("k3r3/8/8/8/8/8/8/4K2R w K - 0 1", ["e1g1"]),
    ("k4r2/8/8/8/8/8/8/4K2R w K - 0 1", ["e1g1"]),
    ("k5r1/8/8/8/8/8/8/4K2R w K - 0 1", ["e1g1"]),
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", ["e1c1", "e8g8"]),
    ("k7/8/8/8/8/8/7K/4Q3 w - - 0 1", ["e1c1"]),
    ("4q3/7k/8/8/8/8/8/K7 b - - 0 1", ["e8c8"]),
    ("k7/8/8/8/8/8/7K/4R3 w - - 0 1", ["e1g1"]),
    ("4r3/7k/8/8/8/8/8/K7 b - - 0 1", ["e8g8"]),
    ("k3r3/3p4/8/4P3/8/8/8/4K3 b - - 0 1", ["d7d5", "e5d6"]),
    ("4k3/8/8/8/4p3/8/3P4/K3R3 w - - 0 1", ["d2d4", "e4d3"]),
    ("k7/3p4/8/2P1P3/8/8/8/7K b - - 0 1", ["d7d5", "c5d6"]),
    ("k7/8/8/8/8/4r3/4P3/7K w - - 0 1", ["h1g1"]),
    ("8/7K/8/8/8/8/pp6/kp6 w - - 0 1", ["h7h8"]),
    ("7k/8/8/8/8/8/8/K7 w - - 99 1", ["a1b1"]),
    ("7k/8/8/8/8/8/P7/K7 w - - 99 1", ["a2a3"]),
    ("7k/8/8/8/8/8/8/K7 w - - 0 1", ["a1b1", "h8g8"]),
    (chess.STARTING_FEN, ["f2f3", "e7e5", "g2g4", "d8h4", "a2a3"]),
])
def test_special_rules_match_independent_oracle(fen, uci_moves):
    board = chess.Board(fen)
    state = _state(board)
    repetitions = Counter({_key(board): 1})
    _assert_position(state, board)
    for ply, uci in enumerate(uci_moves, start=1):
        move = chess.Move.from_uci(uci)
        assert move in _moves(board)
        state = _advance(state, jnp.asarray(_action(move)))
        board.push(move)
        repetitions[_key(board)] += 1
        _assert_position(state, board)
        terminal, winner = _outcome(board, repetitions, ply, 1000)
        assert (bool(state.terminated), int(state.winner)) == (terminal, winner)
        if terminal:
            assert ply == len(uci_moves), "Test fixture must stop at the first terminal state"


@pytest.mark.parametrize("fen,source,target", [
    ("k6r/6P1/8/8/8/8/8/7K w - - 0 1", "g7", "g8"),
    ("k6r/6P1/8/8/8/8/8/7K w - - 0 1", "g7", "h8"),
    ("7k/8/8/8/8/8/6p1/K6R b - - 0 1", "g2", "g1"),
    ("7k/8/8/8/8/8/6p1/K6R b - - 0 1", "g2", "h1"),
])
@pytest.mark.parametrize("promotion", ["q", "r", "b", "n"])
def test_promotion_choices_have_distinct_actions_and_correct_boards(fen, source, target, promotion):
    board = chess.Board(fen)
    state = _state(board)
    _assert_position(state, board)
    move = chess.Move.from_uci(source + target + promotion)
    assert move in _moves(board)
    state = _advance(state, jnp.asarray(_action(move)))
    board.push(move)
    _assert_position(state, board)


def test_repetition_draw_occurs_on_third_complete_position():
    board = chess.Board()
    state = core.initial_state()
    cycle = ["g1f3", "g8f6", "f3g1", "f6g8"]
    for ply, uci in enumerate(cycle * 2, start=1):
        move = chess.Move.from_uci(uci)
        board.push(move)
        state = _advance(state, jnp.asarray(_action(move)))
        assert bool(state.terminated) == (ply == 8)
        assert int(state.winner) == -1
        _assert_position(state, board)


def test_repetition_key_counts_pseudo_legal_pinned_en_passant():
    board = chess.Board("k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    assert board.has_pseudo_legal_en_passant()
    assert not board.has_legal_en_passant()
    state = _state(board)
    without_ep = state._replace(ep_square=jnp.asarray(-1))
    assert not np.array_equal(core.position_key(state), core.position_key(without_ep))
    board.remove_piece_at(chess.E5)
    state = _state(board)
    np.testing.assert_array_equal(core.position_key(state),
                                  core.position_key(state._replace(ep_square=jnp.asarray(-1))))


@pytest.mark.parametrize("fen,plies,occurrences", [
    ("k3r3/8/8/8/8/8/8/4K3 b - - 100 1", 1000, 3),
    ("7K/8/8/8/8/8/pp6/kp6 b - - 100 1", 1000, 1),
    ("7K/8/8/8/8/8/pp6/kp6 b - - 99 1", 1000, 1),
    ("7k/8/8/8/8/8/8/K7 w - - 0 1", 1000, 1),
])
def test_terminal_rule_precedence_matches_oracle(fen, plies, occurrences):
    board = chess.Board(fen)
    state = _state(board)._replace(ply=jnp.asarray(plies),
                                  repetition_count=jnp.asarray(occurrences))
    state = jax.jit(core.evaluate_outcome)(state)
    expected = _outcome(board, Counter({_key(board): occurrences}), plies, 1000)
    assert (bool(state.terminated), int(state.winner)) == expected


def test_initial_and_first_move_observations_match_published_transcript():
    state = core.initial_state()
    initial = ("RNBQKBNRPPPPPPPP----------------????????????????????????????????",
               "????????????????????????????????----------------pppppppprnbqkbnr")
    after_c4 = ("RNBQKBNRPP-PPPPP----------P-----??-?????????????????????????????",
                "????????????????????????????????----------------pppppppprnbqkbnr")
    symbols = {0: "-", 7: "?"}
    for piece_type in range(1, 7):
        symbols[piece_type] = chess.piece_symbol(piece_type).upper()
        symbols[-piece_type] = chess.piece_symbol(piece_type)
    for index, expected in enumerate((initial, after_c4)):
        actual = tuple("".join(symbols[int(value)] for value in row)
                       for row in np.asarray(_snapshot(state)[1]))
        assert actual == expected
        if index == 0:
            state = _advance(state, jnp.asarray(_action(chess.Move.from_uci("c2c4"))))
