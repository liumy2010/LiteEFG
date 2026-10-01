"""Exact neural-policy adapter checks using independent public-history recursion."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("optax")
pyspiel = pytest.importorskip("pyspiel")

from open_spiel.python.algorithms import exploitability as openspiel_exploitability
from open_spiel.python.policy import TabularPolicy

from examples.goofspiel_exact import exact_exploitability
from LiteEFG.drl.env import Goofspiel
from LiteEFG.drl.graph import DRLGraph
from LiteEFG.drl.runtime import Policy


def policy(kind, num_cards):
    def apply(observations, mask):
        uniform = mask.astype(jnp.float32) / mask.sum(axis=-1, keepdims=True)
        low = jnp.argmax(mask, axis=-1)
        high = num_cards - 1 - jnp.argmax(mask[..., ::-1], axis=-1)
        if kind == "uniform":
            return uniform
        if kind in ("lowest", "highest"):
            action = low if kind == "lowest" else high
            return jax.nn.one_hot(action, num_cards)
        own_bid_offset = (num_cards + 1) + num_cards + num_cards * num_cards
        first_bid = jnp.argmax(observations[..., own_bid_offset:own_bid_offset + num_cards], axis=-1)
        if kind == "soft":
            turn = jnp.argmax(observations[..., :num_cards + 1], axis=-1)
            prize_offset = (num_cards + 1) + num_cards
            prize_history = observations[..., prize_offset:prize_offset + num_cards * num_cards]
            prize_history = prize_history.reshape(observations.shape[:-1] + (num_cards, num_cards))
            current_prize = (prize_history * jax.nn.one_hot(turn, num_cards)[..., :, None]
                             * jnp.arange(1, num_cards + 1)).sum(axis=(-2, -1))
            weights = mask * (1 + ((jnp.arange(1, num_cards + 1) * current_prize[..., None]
                                  + 3 * first_bid[..., None]) % 7))
            return weights / weights.sum(axis=-1, keepdims=True)
        two_rounds_played = observations[..., 2:num_cards + 1].sum(axis=-1) > 0
        action = jnp.where(first_bid % 2 == 0, low, high)
        return jnp.where(two_rounds_played[..., None], jax.nn.one_hot(action, num_cards), uniform)
    return apply


def independent_metrics(num_cards, kinds):
    """Solve the public game directly; maximize before observing the current bid.

    Returns utilities and both unilateral best-response values. This evaluator
    uses tuple histories, direct score differences and no LiteEFG/JAX operations.
    """
    def probabilities(kind, hand, history, player, prize):
        if kind == "soft":
            first_bid = history[0][player + 1] if history else 0
            weights = np.asarray([1 + ((action + 1) * prize + 3 * first_bid) % 7
                                  for action in hand], dtype=float)
            return weights / weights.sum()
        if kind == "uniform" or (kind == "history" and len(history) < 2):
            return np.full(len(hand), 1.0 / len(hand))
        if kind == "history":
            kind = "lowest" if history[0][player + 1] % 2 == 0 else "highest"
        selected = 0 if kind == "lowest" else len(hand) - 1
        values = np.zeros(len(hand))
        values[selected] = 1.0
        return values

    def solve(prizes, hands, history):
        if not prizes:
            return np.zeros(3)
        result = np.zeros(3)
        for prize in prizes:
            first = probabilities(kinds[0], hands[0], history, 0, prize)
            second = probabilities(kinds[1], hands[1], history, 1, prize)
            remaining = tuple(card for card in prizes if card != prize)
            values = np.zeros((len(hands[0]), len(hands[1]), 3))
            for i, bid0 in enumerate(hands[0]):
                for j, bid1 in enumerate(hands[1]):
                    after = (tuple(card for card in hands[0] if card != bid0),
                             tuple(card for card in hands[1] if card != bid1))
                    stage_value = prize * ((bid0 > bid1) - (bid0 < bid1)) / 2
                    continuation = solve(remaining, after, history + ((prize, bid0, bid1),))
                    values[i, j] = continuation + (stage_value, stage_value, -stage_value)
            result += (first @ values[:, :, 0] @ second,
                       np.max(values[:, :, 1] @ second),
                       np.max(first @ values[:, :, 2]))
        return result / len(prizes)

    value, best0, best1 = solve(tuple(range(1, num_cards + 1)),
                              (tuple(range(num_cards)),) * 2, ())
    utility = np.array([value, -value])
    best = np.array([best0, best1])
    return utility, best, best - utility


@pytest.mark.parametrize("kinds", [
    ("uniform", "uniform"), ("lowest", "highest"), ("uniform", "highest"),
])
def test_three_card_exact_metrics_match_independent_recursion_and_openspiel(kinds):
    env = Goofspiel(3, imp_info=False, points_order="random")
    result = exact_exploitability(env, *(policy(kind, 3) for kind in kinds), batch_size=16)
    assert result["size"]["information_sets"] == [57, 57]
    utility, best, gains = independent_metrics(3, kinds)
    np.testing.assert_allclose(result["utility"], utility, atol=2e-7, rtol=0)
    np.testing.assert_allclose(result["best_response_utility"], best, atol=2e-7, rtol=0)
    np.testing.assert_allclose(result["deviation_gain"], gains, atol=2e-7, rtol=0)
    assert result["nash_conv"] == pytest.approx(gains.sum(), abs=3e-7)
    assert result["exploitability"] == pytest.approx(gains.sum() / 2, abs=2e-7)

    game = pyspiel.convert_to_turn_based(pyspiel.load_game("goofspiel", {
        "num_cards": 3, "players": 2, "imp_info": False,
        "points_order": "random", "returns_type": "point_difference",
    }))
    tabular = TabularPolicy(game)
    for index, state in enumerate(tabular.states):
        kind = kinds[state.current_player()]
        legal = state.legal_actions()
        tabular.action_probability_array[index] = 0
        if kind == "uniform":
            tabular.action_probability_array[index, legal] = 1 / len(legal)
        else:
            action = min(legal) if kind == "lowest" else max(legal)
            tabular.action_probability_array[index, action] = 1
    oracle = openspiel_exploitability.nash_conv(
        game, tabular, return_only_nash_conv=False, use_cpp_br=True)
    np.testing.assert_allclose(result["deviation_gain"], oracle.player_improvements,
                               atol=2e-7, rtol=0)


def test_four_card_exact_preserves_distinct_ordered_bid_histories():
    env = Goofspiel(4, imp_info=False, points_order="random")
    history_policy = policy("history", 4)
    observations, masks = [], []
    for past_bids in (((0, 2), (1, 3)), ((1, 3), (0, 2))):
        state = env.init(jax.random.PRNGKey(0))._replace(point_cards=jnp.arange(1, 5))
        for actions in past_bids:
            state, _, _ = env.step(state, jnp.asarray(actions), jax.random.PRNGKey(0))
        observations.append(env.features(state)["information_set"][0])
        masks.append(env.legal_action_mask(state)[0])
    # Both histories have the same hands, scores, winner sequence and prizes.
    # The network-visible ordered bid history still determines different play.
    np.testing.assert_array_equal(masks[0], masks[1])
    probabilities = np.asarray(history_policy(jnp.stack(observations), jnp.stack(masks)))
    np.testing.assert_array_equal(probabilities[0], [0, 0, 1, 0])
    np.testing.assert_array_equal(probabilities[1], [0, 0, 0, 1])

    result = exact_exploitability(env, history_policy, history_policy, batch_size=127)
    assert result["size"]["information_sets"] == [3652, 3652]
    utility, best, gains = independent_metrics(4, ("history", "history"))
    np.testing.assert_allclose(result["utility"], utility, atol=2e-6, rtol=0)
    np.testing.assert_allclose(result["best_response_utility"], best, atol=2e-6, rtol=0)
    np.testing.assert_allclose(result["deviation_gain"], gains, atol=2e-6, rtol=0)


def test_seven_card_size_guard_runs_before_policy_inference():
    calls = []

    def unexpected_policy(observations, mask):
        calls.append((observations, mask))
        raise AssertionError("The oversized game must be rejected before neural inference")

    with pytest.raises(ValueError, match="size|nodes|limit|budget|max_nodes"):
        exact_exploitability(Goofspiel(7), unexpected_policy, unexpected_policy)
    assert calls == []


@pytest.mark.parametrize("num_cards,kinds", [
    (3, ("soft", "uniform")), (4, ("soft", "highest")),
    (4, ("soft", "history")),
])
def test_soft_current_prize_and_history_dependent_policy_matches_independent_oracle(num_cards, kinds):
    result = exact_exploitability(Goofspiel(num_cards),
                                  *(policy(kind, num_cards) for kind in kinds), batch_size=127)
    utility, best, gains = independent_metrics(num_cards, kinds)
    np.testing.assert_allclose(result["utility"], utility, atol=2e-6, rtol=0)
    np.testing.assert_allclose(result["best_response_utility"], best, atol=2e-6, rtol=0)
    np.testing.assert_allclose(result["deviation_gain"], gains, atol=2e-6, rtol=0)


@pytest.mark.parametrize("feature_player", [0, 1])
def test_exact_evaluation_passes_named_public_features_to_graph_policy(feature_player):
    class FeatureGoofspiel(Goofspiel):
        def features(self, state):
            features = super().features(state)
            # This feature depends only on the currently revealed prize and
            # each player's own ordered bids; it is legal actor information.
            features["bid_weights"] = policy("soft", self.num_cards)(
                features["information_set"], self.legal_action_mask(state))
            return features

    class FeaturePolicy(DRLGraph):
        def __init__(self):
            super().__init__()
            self.strategy = self.env.bid_weights / self.env.bid_weights.sum()
            self.value = self.env.bid_weights.sum() * 0

    env, graph = FeatureGoofspiel(3), FeaturePolicy()
    state = env.init(jax.random.key(0))
    local_features = {name: value[0] for name, value in env.features(state).items()}
    states = graph.init(jax.random.key(0), {
        **local_features, "legal_action_mask": env.legal_action_mask(state)[0],
    })
    wrapped = Policy(graph, tuple(state.params for state in states),
                     local_features["information_set"].shape[0], env.num_actions)
    policies = [policy("highest", 3), policy("highest", 3)]
    policies[feature_player] = wrapped
    kinds = ["highest", "highest"]
    kinds[feature_player] = "soft"
    result = exact_exploitability(env, *policies, batch_size=17)
    utility, best, gains = independent_metrics(3, kinds)
    np.testing.assert_allclose(result["utility"], utility, atol=2e-6, rtol=0)
    np.testing.assert_allclose(result["best_response_utility"], best, atol=2e-6, rtol=0)
    np.testing.assert_allclose(result["deviation_gain"], gains, atol=2e-6, rtol=0)
