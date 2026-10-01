# Reference (game rules and point_difference rewards): OpenSpiel Goofspiel.
# https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/games/goofspiel/goofspiel.cc

"""Functional JAX Goofspiel for batched outcome sampling."""

from dataclasses import dataclass
from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..environment import Environment


class GoofspielState(NamedTuple):
    """Device state. ``point_cards`` includes private, unrevealed chance draws.

    Cards in ``point_cards`` are worth 1 through ``num_cards``; bid actions are
    indexed from zero. Unplayed bids and round outcomes use -1 as a sentinel.
    Player-visible features are obtained through :meth:`Goofspiel.features`.
    """

    round: jax.Array
    point_cards: jax.Array
    hands: jax.Array
    bids: jax.Array
    winners: jax.Array
    scores: jax.Array


@dataclass(frozen=True)
class Goofspiel(Environment):
    """Two-player simultaneous Goofspiel with a fixed, complete-game horizon.

    Each round both players bid one unused card for the public point card. The
    higher bid wins its points; a tie discards the point card. Actions are card
    values minus one. ``point_difference`` follows OpenSpiel: each player's
    score minus the players' mean score, or half the score difference for two
    players. Rewards are zero until the terminal transition and are not scaled.

    ``imp_info=False`` reveals both players' previous bids. ``imp_info=True``
    reveals only one's own bids and the winner of each round. Both variants
    preserve the full order of the visible history and hide future chance
    draws in ``information_set``. The separate ``full_info`` feature includes
    both players' hands and completed bids, and all future chance draws.
    Features are relative to the observing player, with one's own score,
    hand and bids preceding the opponent's.

    ``init``, ``features``, ``legal_action_mask``, ``active_players`` and ``step``
    are pure functions suitable for ``jax.jit`` and ``jax.vmap``. The final
    forced bid remains an explicit round. ``step`` requires legal actions at
    nonterminal states; terminal states are absorbing with zero further reward.
    """

    num_cards: int = 7
    imp_info: bool = False
    points_order: str = "random"
    returns_type: str = "point_difference"

    def __post_init__(self):
        if (isinstance(self.num_cards, bool)
                or not isinstance(self.num_cards, Integral)
                or self.num_cards < 1):
            raise ValueError("num_cards must be a positive integer")
        if not isinstance(self.imp_info, bool):
            raise ValueError("imp_info must be a bool")
        if self.points_order not in ("random", "ascending", "descending"):
            raise ValueError("points_order must be random, ascending or descending")
        if self.returns_type != "point_difference":
            raise ValueError("returns_type must be point_difference")

    @property
    def num_players(self):
        return 2

    @property
    def num_actions(self):
        return self.num_cards

    @property
    def max_steps(self):
        return self.num_cards

    def init(self, key):
        n = self.num_cards
        cards = jnp.arange(1, n + 1, dtype=jnp.int32)
        if self.points_order == "random":
            cards = jax.random.permutation(key, cards)
        elif self.points_order == "descending":
            cards = cards[::-1]
        return GoofspielState(
            round=jnp.array(0, dtype=jnp.int32),
            point_cards=cards,
            hands=jnp.ones((2, n), dtype=jnp.bool_),
            bids=jnp.full((n, 2), -1, dtype=jnp.int32),
            winners=jnp.full((n,), -1, dtype=jnp.int32),
            scores=jnp.zeros((2,), dtype=jnp.float32),
        )

    def features(self, state):
        """Return player-visible ``information_set`` and privileged ``full_info``.

        ``information_set`` is a float32 vector for each player, with shape
        ``[2, D]``. It contains only that player's visible information.
        ``full_info`` uses the public-bid encoding and reveals the complete
        pre-sampled prize order, both remaining hands, and both players'
        completed bids, including in imperfect-information games. Its shape
        is ``[2, 3 * num_cards**2 + 6 * num_cards + 3]``.
        """
        return {
            "information_set": self._encode_features(state, full_info=False),
            "full_info": self._encode_features(state, full_info=True),
        }

    def _encode_features(self, state, *, full_info):
        """Encode visible or complete state as float32 vectors ``[2, D]``.

        Feature blocks are round one-hot, own hand, visible prize one-hots,
        own bid one-hots, round winner one-hots (self/opponent/tie), and the two
        scores divided by the sum of all prize values. Public-bid games append
        the opponent hand and opponent bid one-hots. Unobserved history is zero.
        ``full_info`` reveals all prize cards and always includes the opponent
        hand and completed bids; unplayed bids remain zero.
        """
        n = self.num_cards
        rounds = jnp.arange(n)
        completed = rounds < state.round
        revealed = jnp.ones((n,), dtype=jnp.bool_) if full_info else rounds <= state.round
        round_feature = jax.nn.one_hot(state.round, n + 1, dtype=jnp.float32)
        point_history = jax.nn.one_hot(
            state.point_cards - 1, n, dtype=jnp.float32
        ) * revealed[:, None]
        bid_history = jax.nn.one_hot(state.bids, n, dtype=jnp.float32)
        bid_history = bid_history * completed[:, None, None]
        score_scale = n * (n + 1) / 2

        def player_observation(player):
            opponent = 1 - player
            # A tie has winner 2; -1 denotes an unplayed round.
            winners = jnp.where(
                state.winners == 2, 2,
                jnp.where(state.winners == player, 0, 1),
            )
            winner_history = jax.nn.one_hot(winners, 3, dtype=jnp.float32)
            winner_history = winner_history * completed[:, None]
            features = [
                round_feature,
                state.hands[player].astype(jnp.float32),
                point_history.reshape(-1),
                bid_history[:, player].reshape(-1),
                winner_history.reshape(-1),
                jnp.stack((state.scores[player], state.scores[opponent]))
                / score_scale,
            ]
            if full_info or not self.imp_info:
                features.extend((
                    state.hands[opponent].astype(jnp.float32),
                    bid_history[:, opponent].reshape(-1),
                ))
            return jnp.concatenate(features)

        return jax.vmap(player_observation)(jnp.arange(2))

    def legal_action_mask(self, state):
        return state.hands & (state.round < self.num_cards)

    def active_players(self, state):
        return jnp.full((2,), state.round < self.num_cards, dtype=jnp.bool_)

    def step(self, state, actions, key):
        """Apply both bids and return ``(state, reward[2], done)``.

        Chance is sampled by ``init`` and revealed one prize at a time. ``key``
        is accepted for the common stochastic-environment protocol.
        """
        del key
        actions = jnp.asarray(actions, dtype=jnp.int32)

        def advance(current):
            winner = jnp.where(
                actions[0] == actions[1], 2,
                jnp.where(actions[0] > actions[1], 0, 1),
            ).astype(jnp.int32)
            scores = current.scores + (
                jnp.arange(2) == winner
            ).astype(jnp.float32) * current.point_cards[current.round]
            next_state = current._replace(
                round=current.round + 1,
                hands=current.hands.at[jnp.arange(2), actions].set(False),
                bids=current.bids.at[current.round].set(actions),
                winners=current.winners.at[current.round].set(winner),
                scores=scores,
            )
            done = next_state.round == self.num_cards
            reward = jnp.where(done, scores - jnp.mean(scores), 0.0)
            return next_state, reward, done

        def absorb(current):
            return current, jnp.zeros((2,), dtype=jnp.float32), jnp.array(True)

        return jax.lax.cond(state.round < self.num_cards, advance, absorb, state)


__all__ = ["Goofspiel", "GoofspielState"]
