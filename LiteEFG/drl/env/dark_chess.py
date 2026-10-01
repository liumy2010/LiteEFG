"""MIT Fog of War chess with pure JAX transitions and private observations.

Rules: https://www.mit.edu/~6.7980/fow/?tab=rules
Protocol: https://www.mit.edu/~6.7980/fow/starter/original-rules.md
"""

from dataclasses import dataclass
from numbers import Integral
import re
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..environment import Environment
from . import _dark_chess as core


BOARD_FEATURE_SIZE = 14 * 8 * 8
CRITIC_METADATA_SIZE = 73


class DarkChessState(NamedTuple):
    """Fixed-shape device state; ``core`` contains privileged game information.

    History is newest first, including the current observation. Each player
    receives a new observation after *every* ply. It uses the protocol's
    absolute square coordinates; feature encoding supplies player perspective.
    """

    core: core.CoreState
    observation_history: jax.Array


def action_from_uci(move: str) -> int:
    """Encode a host-side UCI move; legality is checked using the action mask.

    Queen promotions share the ordinary from/to action. Underpromotions have
    distinct actions. This function validates notation, not board legality.
    """
    if not isinstance(move, str) or not re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", move):
        raise ValueError("Expected a UCI move such as e2e4 or a7a8n")
    source = ord(move[0]) - ord("a") + 8 * (int(move[1]) - 1)
    target = ord(move[2]) - ord("a") + 8 * (int(move[3]) - 1)
    if source == target:
        raise ValueError("UCI source and destination must differ")
    if len(move) == 4:
        return source * 64 + target
    if ((source // 8, target // 8) not in ((6, 7), (1, 0))
            or abs(source % 8 - target % 8) > 1):
        raise ValueError("Promotion must move from the penultimate to the last rank")
    if move[4] == "q":
        return source * 64 + target
    piece = {"n": 2, "b": 3, "r": 4}[move[4]]
    matches = np.flatnonzero(
        (core.ACTION_FROM == source) & (core.ACTION_TO == target)
        & (core.ACTION_PROMOTION == piece)
    )
    return int(matches[0])


def action_to_uci(action: int, state: DarkChessState) -> str:
    """Decode a host-side action using the pre-move state for queen promotion.

    This inspection helper is not a JAX operation. No hidden board information
    is needed: the moved piece belongs to the acting player.
    """
    if isinstance(action, bool) or not isinstance(action, Integral) or not 0 <= action < core.NUM_ACTIONS:
        raise ValueError(f"action must be an integer in [0, {core.NUM_ACTIONS})")
    source, target = int(core.ACTION_FROM[action]), int(core.ACTION_TO[action])

    def square(index):
        return chr(ord("a") + index % 8) + str(index // 8 + 1)

    promotion = int(core.ACTION_PROMOTION[action])
    if promotion == 0 and abs(int(state.core.board[source])) == 1 and target // 8 in (0, 7):
        promotion = 5
    return square(source) + square(target) + {0: "", 2: "n", 3: "b", 4: "r", 5: "q"}[promotion]


@dataclass(frozen=True)
class DarkChess(Environment):
    """Two-player MIT Fog of War chess, with white at player index zero.

    ``max_plies`` is the automatic draw horizon (the arena uses 1,000).
    ``history_length`` counts observation frames, including the current one.
    The default eight-board input is a finite-memory abstraction, not a
    perfect-recall information set. It does not encode past actions.

    Actions 0..4095 encode ``64 * source + target``; a pawn reaching its last
    rank promotes to a queen. The remaining 132 actions encode knight, bishop,
    and rook promotions for each promotion edge, ordered by white then black,
    increasing source square, increasing target square, and N/B/R. Squares are
    a1=0 through h8=63. Actions remain in this absolute orientation for both
    players. ``step`` requires a legal action for the active player and ignores
    the inactive entry. Terminal states are absorbing.

    Actor features contain private board observations followed by the player's
    color flag. ``full_info`` contains the current complete board, both players'
    observation histories, validity masks, and rule metadata for a privileged
    critic. Its value is for the current mover and both player rows are equal.
    Only the current repetition count is encoded, so this compact input is
    not a complete Markov state.
    Actions and raw observations use absolute coordinates. Feature planes use
    own/opponent colors and flip black's ranks, preserving file order.
    """

    max_plies: int = 1000
    history_length: int = 8

    def __post_init__(self):
        if isinstance(self.max_plies, bool) or not isinstance(self.max_plies, Integral) or self.max_plies < 1:
            raise ValueError("max_plies must be a positive integer")
        if (isinstance(self.history_length, bool)
                or not isinstance(self.history_length, Integral)
                or self.history_length < 1):
            raise ValueError("history_length must be a positive integer")

    @property
    def num_players(self):
        return 2

    @property
    def num_actions(self):
        return core.NUM_ACTIONS

    @property
    def max_steps(self):
        return self.max_plies

    @property
    def critic_board_count(self):
        return 1 + 2 * self.history_length

    @property
    def critic_metadata_size(self):
        return CRITIC_METADATA_SIZE

    @property
    def full_info_size(self):
        return (
            self.critic_board_count * BOARD_FEATURE_SIZE
            + 2 * self.history_length + self.critic_metadata_size
        )

    def init(self, key):
        del key  # The standard chess starting position is deterministic.
        position = core.initial_state()
        history = jnp.full((self.history_length, 2, 64), core.UNSEEN, dtype=jnp.int32)
        history = history.at[0].set(core.observed_boards(position))
        return DarkChessState(position, history)

    def observation(self, state):
        """Return protocol boards ``[2,64]`` in absolute a1..h8 coordinates.

        White P/N/B/R/Q/K are 1..6, black pieces -1..-6, empty squares 0,
        and unseen squares 7. Own pieces are always visible.
        """
        return core.observed_boards(state.core)

    @staticmethod
    def _encode_perspective(boards, player):
        """Encode absolute boards [N,64] in one player's coordinate frame."""
        boards = boards.reshape(-1, 8, 8)
        boards = jnp.where(player == 1, boards[:, ::-1, :], boards)
        relative = boards * (1 - 2 * player)
        categories = jnp.where(relative > 0, relative - 1, -relative + 5)
        categories = jnp.where(boards == 0, 12, categories)
        categories = jnp.where(boards == core.UNSEEN, 13, categories)
        planes = jax.nn.one_hot(categories, 14, dtype=jnp.float32)
        return jnp.moveaxis(planes, -1, 1)

    @staticmethod
    def _encode_boards(boards):
        """Encode absolute signed boards [2,history_length,64] as relative piece planes."""
        return jax.vmap(DarkChess._encode_perspective)(boards, jnp.arange(2))

    def history_mask(self, state):
        """Return [2,history_length] validity flags for newest-first observation frames."""
        valid = jnp.arange(self.history_length) <= state.core.ply
        return jnp.broadcast_to(valid, (2, self.history_length))

    def board_features(self, state):
        """Return private float32 planes [2,history_length,14,8,8], newest frame first.

        Channels are own P/N/B/R/Q/K, opponent P/N/B/R/Q/K, visible empty,
        and unseen. Rank zero is the player's home rank; files remain a..h.
        Unused frames are zero, distinguished by ``history_mask(state)``.
        """
        planes = self._encode_boards(jnp.swapaxes(state.observation_history, 0, 1))
        return planes * self.history_mask(state)[:, :, None, None, None]

    def full_board_features(self, state):
        """Return current complete board planes [2,14,8,8] for a critic.

        Encoding and orientation match ``board_features``. The unseen channel
        is zero. No history or non-board game state is included.
        """
        boards = jnp.broadcast_to(state.core.board, (2, 1, 64))
        return self._encode_boards(boards)[:, 0]

    def critic_history_mask(self, state):
        """Return [2,history_length] validity flags, current mover first, then opponent."""
        players = jnp.array([state.core.turn, 1 - state.core.turn])
        return self.history_mask(state)[players]

    def critic_board_features(self, state):
        """Return [1+2*history_length,14,8,8] boards in the current mover's perspective.

        The sequence contains the complete current board, the mover's newest-
        first observation window, then the opponent's newest-first window.
        Opponent observations retain their original visibility and unseen
        squares. All boards share the mover's coordinates and color encoding.
        Padding frames contain zeros.
        """
        mover = state.core.turn
        boards = jnp.concatenate((
            state.core.board[None],
            state.observation_history[:, mover],
            state.observation_history[:, 1 - mover],
        ), axis=0)
        valid = jnp.concatenate((jnp.ones(1, jnp.bool_), self.critic_history_mask(state).reshape(-1)))
        return self._encode_perspective(boards, mover) * valid[:, None, None, None]

    def critic_metadata(self, state):
        """Return 73 float32 rule features, in this fixed order:

        * Mover color (white=1, black=0).
        * Mover then opponent queenside/kingside castling rights (four flags).
        * En passant target (65-way one-hot: 64 mover-oriented squares, none).
        * Halfmove clock / 100, remaining plies / max_plies, and the current
          position's repetition count / 3.

        The engine retains exact repetition records for adjudication; the
        compact count here does not encode all future repetition outcomes.
        """
        position = state.core
        mover = position.turn
        players = jnp.array([mover, 1 - mover])
        rights = position.castling_rights[players].reshape(-1).astype(jnp.float32)
        ep = position.ep_square
        oriented_ep = jnp.where(mover == 0, ep, (7 - ep // 8) * 8 + ep % 8)
        ep_index = jnp.where(ep >= 0, oriented_ep, 64)
        clocks = jnp.array([
            position.halfmove_clock / 100.0,
            jnp.maximum(self.max_plies - position.ply, 0) / self.max_plies,
            position.repetition_count / 3.0,
        ], dtype=jnp.float32)
        return jnp.concatenate((
            jnp.asarray([1 - mover], dtype=jnp.float32), rights,
            jax.nn.one_hot(ep_index, 65, dtype=jnp.float32), clocks,
        ))

    def features(self, state):
        """Return private actor inputs and a privileged current-mover input.

        ``information_set`` is float32 [2,history_length*896+1]: the flattened
        private observation window followed by the player's color (white=1,
        black=0). ``history_mask`` is bool [2,history_length]. Actor windows
        use each player's own perspective and include no opponent observations.
        ``full_info`` is float32 [2,D], with identical rows: flattened
        ``critic_board_features``, two history masks, then 73 rule features.
        D=(1+2*history_length)*896+2*history_length+73. Its scalar value predicts
        the current mover's return; inactive rows must not be used as that
        player's value.
        """
        critic = jnp.concatenate((
            self.critic_board_features(state).reshape(-1),
            self.critic_history_mask(state).reshape(-1).astype(jnp.float32),
            self.critic_metadata(state),
        ))
        return {
            "information_set": jnp.concatenate((
                self.board_features(state).reshape(2, -1),
                jnp.array([[1.0], [0.0]], dtype=jnp.float32),
            ), axis=-1),
            "full_info": jnp.broadcast_to(critic, (2, self.full_info_size)),
            "history_mask": self.history_mask(state),
        }

    def legal_action_mask(self, state):
        return core.legal_moves(state.core) & self.active_players(state)[:, None]

    def active_players(self, state):
        return (jnp.arange(2) == state.core.turn) & ~state.core.terminated

    def step(self, state, actions, key):
        del key
        actions = jnp.asarray(actions, dtype=jnp.int32)

        def advance(current):
            mover = current.core.turn
            action = actions[mover]
            position = core.transition(current.core, action, self.max_plies)
            after = DarkChessState(
                position,
                jnp.concatenate((core.observed_boards(position)[None], current.observation_history[:-1]), axis=0),
            )
            reward = jnp.where(
                position.terminated & (position.winner >= 0),
                jnp.where(jnp.arange(2) == position.winner, 1.0, -1.0), 0.0,
            ).astype(jnp.float32)
            return after, reward, position.terminated

        def absorb(current):
            return current, jnp.zeros(2, dtype=jnp.float32), jnp.array(True)

        return jax.lax.cond(state.core.terminated, absorb, advance, state)

    action_from_uci = staticmethod(action_from_uci)
    action_to_uci = staticmethod(action_to_uci)


__all__ = [
    "DarkChess", "DarkChessState", "action_from_uci", "action_to_uci",
    "BOARD_FEATURE_SIZE", "CRITIC_METADATA_SIZE",
]
