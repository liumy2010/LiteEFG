"""Pure JAX rules for MIT's ``fow-v1`` dark chess variant.

Squares use a1=0 through h8=63. Positive pieces are white, negative pieces
are black, and P/N/B/R/Q/K are numbered 1 through 6. Player 0 is white.
The rules follow https://www.mit.edu/~6.7980/fow/?tab=rules and its referee.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


UNSEEN = 7
REPETITION_WINDOW = 101

# The base action is from_square * 64 + to_square. A pawn reaching its last
# rank promotes to a queen; the additional actions provide N/B/R promotions.
_moves = [(source, target, 0) for source in range(64) for target in range(64)]
_moves.extend(
    (rank * 8 + file, target_rank * 8 + target_file, piece)
    for rank, target_rank in ((6, 7), (1, 0))
    for file in range(8)
    for target_file in range(max(0, file - 1), min(8, file + 2))
    for piece in (2, 3, 4)
)
ACTION_FROM, ACTION_TO, ACTION_PROMOTION = np.asarray(_moves, dtype=np.int32).T
NUM_ACTIONS = len(_moves)

_squares = np.arange(64, dtype=np.int32)
_file = _squares % 8
_rank = _squares // 8
_df = _file[None, :] - _file[:, None]
_dr = _rank[None, :] - _rank[:, None]
_diagonal = (np.abs(_df) == np.abs(_dr)) & (_df != 0)
_straight = ((_df == 0) | (_dr == 0)) & ((_df != 0) | (_dr != 0))
_knight = ((np.abs(_df) == 1) & (np.abs(_dr) == 2)) | (
    (np.abs(_df) == 2) & (np.abs(_dr) == 1)
)
_king = (np.maximum(np.abs(_df), np.abs(_dr)) == 1)
_between = np.zeros((64, 64, 6), dtype=np.int32)
_between_valid = np.zeros((64, 64, 6), dtype=np.bool_)
for _source in range(64):
    for _target in range(64):
        if _diagonal[_source, _target] or _straight[_source, _target]:
            _distance = max(abs(int(_df[_source, _target])), abs(int(_dr[_source, _target])))
            _stride = int(np.sign(_df[_source, _target]) + 8 * np.sign(_dr[_source, _target]))
            for _offset in range(1, _distance):
                _between[_source, _target, _offset - 1] = _source + _offset * _stride
                _between_valid[_source, _target, _offset - 1] = True


class CoreState(NamedTuple):
    """Hidden board state; castling rights are [player, queenside/kingside].

    Repetition keys are exact packed positions, rather than probabilistic
    hashes. The fifty-move rule bounds relevant history to 101 positions.
    ``repetition_count`` is the current position's number of occurrences.
    """

    board: jax.Array
    turn: jax.Array
    castling_rights: jax.Array
    ep_square: jax.Array
    halfmove_clock: jax.Array
    ply: jax.Array
    terminated: jax.Array
    winner: jax.Array
    repetition_keys: jax.Array
    repetition_count: jax.Array


def _nonpawn_attacks(board):
    clear = jnp.all(
        ~jnp.asarray(_between_valid) | (board[jnp.asarray(_between)] == 0), axis=-1
    )
    piece = jnp.abs(board)[:, None]
    return (
        ((piece == 2) & jnp.asarray(_knight))
        | ((piece == 3) & jnp.asarray(_diagonal) & clear)
        | ((piece == 4) & jnp.asarray(_straight) & clear)
        | ((piece == 5) & jnp.asarray(_diagonal | _straight) & clear)
        | ((piece == 6) & jnp.asarray(_king))
    )


def _ep_moves(state, player):
    sign = 1 - 2 * player
    board = state.board
    return (
        (board[:, None] == sign)
        & (jnp.asarray(_dr) == sign)
        & (jnp.abs(jnp.asarray(_df)) == 1)
        & (jnp.asarray(_rank)[:, None] == jnp.where(player == 0, 4, 3))
        & (jnp.arange(64)[None, :] == state.ep_square)
        & (board[None, :] == 0)
        & (board[jnp.clip(jnp.arange(64) - 8 * sign, 0, 63)][None, :] == -sign)
        & (player == state.turn)
    )


def _clean_castling_rights(state):
    kings = state.board[jnp.array([4, 60])] == jnp.array([6, -6])
    rooks = state.board[jnp.array([[0, 7], [56, 63]])] == jnp.array([[4], [-4]])
    return state.castling_rights & kings[:, None] & rooks


def legal_moves(state):
    """Return pseudo-legal actions for both sides, shape [2, 4228].

    Off-turn players cannot capture en passant. Terminal status is deliberately
    ignored so visibility can still be computed from the final board.
    """
    board = state.board
    nonpawn = _nonpawn_attacks(board)
    rights = _clean_castling_rights(state)

    def one_side(player):
        sign = 1 - 2 * player
        own = board * sign > 0
        target_free = board[None, :] * sign <= 0
        empty = board[None, :] == 0
        pawn = board[:, None] == sign
        single = (jnp.asarray(_df) == 0) & (jnp.asarray(_dr) == sign) & empty
        double = (
            (jnp.asarray(_df) == 0)
            & (jnp.asarray(_dr) == 2 * sign)
            & (jnp.asarray(_rank)[:, None] == jnp.where(player == 0, 1, 6))
            & empty
            & (board[jnp.clip(jnp.arange(64) + 8 * sign, 0, 63)][:, None] == 0)
        )
        capture = (
            (jnp.abs(jnp.asarray(_df)) == 1)
            & (jnp.asarray(_dr) == sign)
            & (board[None, :] * sign < 0)
        )
        moves = (nonpawn & own[:, None] & target_free) | (
            pawn & (single | double | capture)
        ) | _ep_moves(state, player)
        home = player * 56
        queenside = rights[player, 0] & jnp.all(board[home + jnp.array([1, 2, 3])] == 0)
        kingside = rights[player, 1] & jnp.all(board[home + jnp.array([5, 6])] == 0)
        moves = moves.at[home + 4, home + 2].set(moves[home + 4, home + 2] | queenside)
        moves = moves.at[home + 4, home + 6].set(moves[home + 4, home + 6] | kingside)
        extras = (
            moves[jnp.asarray(ACTION_FROM[4096:]), jnp.asarray(ACTION_TO[4096:])]
            & (board[jnp.asarray(ACTION_FROM[4096:])] == sign)
            & (jnp.asarray(ACTION_TO[4096:] // 8) == jnp.where(player == 0, 7, 0))
        )
        return jnp.concatenate((moves.reshape(-1), extras))

    return jax.vmap(one_side)(jnp.arange(2, dtype=jnp.int32))


def observed_boards(state, legal=None):
    """Return each player's visible board, with UNSEEN on hidden squares."""
    if legal is None:
        legal = legal_moves(state)
    # Promotion variants share destinations, so the base 4096 suffice.
    destinations = legal[:, :4096].reshape(2, 64, 64).any(axis=1)
    signs = jnp.array([1, -1])
    visible = destinations | (state.board[None, :] * signs[:, None] > 0)
    ep_captures = jax.vmap(lambda player: jnp.any(_ep_moves(state, player)))(
        jnp.arange(2, dtype=jnp.int32)
    )
    ep_pawn = state.ep_square - 8 * signs
    visible |= ep_captures[:, None] & (jnp.arange(64)[None, :] == ep_pawn[:, None])
    return jnp.where(visible, state.board[None, :], UNSEEN)


def square_attacked(board, square, attacker):
    """Whether a side attacks a square, without orthodox check restrictions."""
    sign = 1 - 2 * attacker
    pawn = (
        (board[:, None] == sign)
        & (jnp.asarray(_dr) == sign)
        & (jnp.abs(jnp.asarray(_df)) == 1)
    )
    attacks = _nonpawn_attacks(board) | pawn
    return jnp.any(attacks[:, jnp.clip(square, 0, 63)] & (board * sign > 0))


def position_key(state):
    """Pack the exact repetition position into nine uint32 values."""
    pieces = jnp.where(state.board < 0, 6 - state.board, state.board).astype(jnp.uint32)
    board_key = jnp.sum(
        pieces.reshape(8, 8) << (4 * jnp.arange(8, dtype=jnp.uint32)),
        axis=1, dtype=jnp.uint32,
    )
    ep = jnp.where(jnp.any(_ep_moves(state, state.turn)), state.ep_square + 1, 0)
    rights = _clean_castling_rights(state).reshape(-1).astype(jnp.uint32)
    metadata = (
        state.turn.astype(jnp.uint32)
        | (jnp.sum(rights << jnp.arange(1, 5, dtype=jnp.uint32), dtype=jnp.uint32))
        | (ep.astype(jnp.uint32) << jnp.uint32(5))
    )
    return jnp.concatenate((board_key, metadata[None]))


def initial_state():
    """Create the standard initial position, with white to move."""
    board = jnp.array(
        [4, 2, 3, 5, 6, 3, 2, 4] + [1] * 8 + [0] * 32
        + [-1] * 8 + [-4, -2, -3, -5, -6, -3, -2, -4],
        dtype=jnp.int32,
    )
    state = CoreState(
        board=board,
        turn=jnp.array(0, dtype=jnp.int32),
        castling_rights=jnp.ones((2, 2), dtype=jnp.bool_),
        ep_square=jnp.array(-1, dtype=jnp.int32),
        halfmove_clock=jnp.array(0, dtype=jnp.int32),
        ply=jnp.array(0, dtype=jnp.int32),
        terminated=jnp.array(False),
        winner=jnp.array(-1, dtype=jnp.int32),
        repetition_keys=jnp.zeros((REPETITION_WINDOW, 9), dtype=jnp.uint32),
        repetition_count=jnp.array(1, dtype=jnp.int32),
    )
    return state._replace(repetition_keys=state.repetition_keys.at[0].set(position_key(state)))


def evaluate_outcome(state, max_plies=1000):
    """Apply king-capture, draw, no-move, then arena horizon adjudication."""
    previous_king = (state.board == (1 - 2 * (1 - state.turn)) * 6)
    king_lost = ~jnp.any(previous_king) | square_attacked(
        state.board, jnp.argmax(previous_king), state.turn
    )
    rule_draw = (state.repetition_count >= 3) | (state.halfmove_clock >= 100)
    no_moves = ~jnp.any(legal_moves(state)[state.turn])
    cap = state.ply >= max_plies
    winner = jnp.where(
        king_lost, state.turn,
        jnp.where(rule_draw, -1, jnp.where(no_moves, 1 - state.turn, -1)),
    ).astype(jnp.int32)
    return state._replace(terminated=king_lost | rule_draw | no_moves | cap, winner=winner)


def transition(state, action, max_plies=1000):
    """Apply a legal action and adjudicate. Terminal states are absorbing."""
    action = jnp.asarray(action, dtype=jnp.int32)

    def advance(current):
        source = jnp.asarray(ACTION_FROM)[action]
        target = jnp.asarray(ACTION_TO)[action]
        promotion = jnp.asarray(ACTION_PROMOTION)[action]
        piece = current.board[source]
        pawn = jnp.abs(piece) == 1
        sign = 1 - 2 * current.turn
        ep_capture = pawn & (target == current.ep_square) & (current.board[target] == 0) & (
            source % 8 != target % 8
        )
        capture = (current.board[target] != 0) | ep_capture
        promote = pawn & ((target // 8 == 0) | (target // 8 == 7))
        placed = jnp.where(promote, sign * jnp.where(promotion == 0, 5, promotion), piece)
        board = current.board.at[source].set(0).at[target].set(placed)
        captured_square = jnp.clip(target - 8 * sign, 0, 63)
        board = board.at[captured_square].set(jnp.where(ep_capture, 0, board[captured_square]))

        castling = (jnp.abs(piece) == 6) & (jnp.abs(target - source) == 2)
        rook_source = current.turn * 56 + jnp.where(target > source, 7, 0)
        rook_target = current.turn * 56 + jnp.where(target > source, 5, 3)
        board = board.at[rook_source].set(jnp.where(castling, 0, board[rook_source]))
        board = board.at[rook_target].set(jnp.where(castling, sign * 4, board[rook_target]))

        corners = jnp.array([[0, 7], [56, 63]])
        rights = current.castling_rights & (source != corners) & (target != corners)
        rights &= ~((jnp.arange(2)[:, None] == current.turn) & (jnp.abs(piece) == 6))
        next_state = current._replace(
            board=board,
            turn=1 - current.turn,
            castling_rights=rights,
            ep_square=jnp.where(pawn & (jnp.abs(target - source) == 16), (source + target) // 2, -1),
            halfmove_clock=jnp.where(pawn | capture, 0, current.halfmove_clock + 1),
            ply=current.ply + 1,
        )
        key = position_key(next_state)
        history = current.repetition_keys.at[next_state.ply % REPETITION_WINDOW].set(key)
        valid = jnp.arange(REPETITION_WINDOW) <= next_state.ply
        count = jnp.sum(jnp.all(history == key[None, :], axis=1) & valid, dtype=jnp.int32)
        next_state = next_state._replace(repetition_keys=history, repetition_count=count)
        return evaluate_outcome(next_state, max_plies)

    return jax.lax.cond(state.terminated, lambda current: current, advance, state)
