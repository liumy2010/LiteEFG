"""Independent checks of trajectory semantics, model snapshots and match estimates."""

import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pytest
import LiteEFG as leg

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("optax")
nn = pytest.importorskip("flax.linen")

from LiteEFG.baselines.drl.PPO import graph
from LiteEFG.drl.environment import Environment
from LiteEFG.drl.env import Goofspiel
from LiteEFG.drl.graph import DRLGraph, aggregate, masked_softmax, model, operation_node, where
from LiteEFG.drl.runtime import (
    Policy,
    Trainer,
    average_utility,
    rollout,
    uniform_policy,
)


def networks(num_actions=3, hidden_size=8):
    return (
        [model(nn.Sequential([nn.Dense(hidden_size), nn.tanh, nn.Dense(num_actions)]))
         for _ in range(2)],
        [model(nn.Sequential([nn.Dense(hidden_size), nn.tanh, nn.Dense(1)]))
         for _ in range(2)],
    )


def lowest_bid(observations, mask):
    del observations
    return jax.nn.one_hot(jnp.argmax(mask, axis=-1), mask.shape[-1])


def highest_bid(observations, mask):
    del observations
    action = mask.shape[-1] - 1 - jnp.argmax(mask[..., ::-1], axis=-1)
    return jax.nn.one_hot(action, mask.shape[-1])


def with_value(policy):
    return lambda obs, mask: (policy(obs, mask), jnp.zeros(obs.shape[0]))


def parameters(states):
    return tuple(tuple(state.params for state in player) for player in states)


def assert_same_tree(left, right):
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    assert left_tree == right_tree
    for before, after in zip(left_leaves, right_leaves):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))


def test_graph_binding_owns_one_trainer_and_never_silently_resets_it(monkeypatch):
    algo = graph(*networks(), num_actions=3)
    env = Goofspiel(num_cards=3)
    with pytest.raises(RuntimeError, match="bind"):
        _ = algo.trainer
    assert algo.bind(env, batch_size=2, seed=7) is algo
    trainer = algo.trainer
    assert trainer.graph is algo and trainer.env is env
    assert trainer.batch_size == 2
    states, key = trainer.states, trainer.key

    def unexpected_init(*args, **kwargs):
        pytest.fail("Binding again must not reinitialize models")

    monkeypatch.setattr(algo, "init", unexpected_init)
    for bind in (lambda: algo.bind(env, batch_size=4, seed=99),
                 lambda: Trainer(algo, env, batch_size=4, seed=99)):
        with pytest.raises(RuntimeError, match="already bound"):
            bind()
    with pytest.raises(AttributeError):
        algo.trainer = object()
    assert algo.trainer is trainer
    assert trainer.states is states and trainer.key is key
    assert trainer.iteration == 0


def test_failed_binding_does_not_publish_a_partial_trainer_and_can_be_retried(monkeypatch):
    algo = graph(*networks(), num_actions=3)
    initialize = algo.init

    def incomplete_init(*args, **kwargs):
        raise RuntimeError("resource initialization failed")

    monkeypatch.setattr(algo, "init", incomplete_init)
    env = Goofspiel(num_cards=3)
    with pytest.raises(RuntimeError, match="resource initialization"):
        algo.bind(env, batch_size=2)
    with pytest.raises(RuntimeError, match="bind"):
        _ = algo.trainer
    monkeypatch.setattr(algo, "init", initialize)
    algo.bind(env, batch_size=2)
    assert len(algo.trainer.states) == env.num_players
    assert algo.trainer.iteration == 0


def test_low_level_trainer_creation_attaches_to_its_graph():
    algo = graph(*networks(), num_actions=3)
    trainer = Trainer(algo, Goofspiel(num_cards=3), batch_size=2)
    assert algo.trainer is trainer


def test_seven_round_rollouts_are_complete_legal_and_zero_sum():
    env = Goofspiel(num_cards=7)
    keys = jax.random.split(jax.random.PRNGKey(0), 11)
    records, complete = jax.jit(lambda k: rollout(
        env, (with_value(uniform_policy), with_value(uniform_policy)), k))(keys)
    records = jax.device_get(records)
    assert np.asarray(complete).all()
    assert records["_valid"].shape == (7, 11, 2)
    assert records["_valid"].all()
    np.testing.assert_array_equal(records["_done"][:-1], False)
    np.testing.assert_array_equal(records["_done"][-1], True)
    np.testing.assert_array_equal(records["reward"][:-1], 0)
    np.testing.assert_array_equal(records["reward"].sum(axis=-1), 0)
    selected_legal = np.take_along_axis(
        records["legal_action_mask"], records["action"][..., None], axis=-1)
    assert selected_legal.all()
    np.testing.assert_array_equal(
        np.sort(records["action"], axis=0),
        np.broadcast_to(np.arange(7)[:, None, None], (7, 11, 2)))
    assert np.isfinite(records["log_sampling_prob"]).all()
    assert records["_policy_valid"].all()


def test_hidden_chance_and_later_uniform_actions_are_independent():
    class HiddenCoin(Environment):
        num_players = 2
        num_actions = 2
        max_steps = 2

        def init(self, key):
            chance = jax.random.categorical(jax.random.split(key, 4)[1], jnp.zeros(2))
            return {"turn": jnp.array(0), "chance": chance}

        def features(self, state):
            # Both players see only the turn. Neither can condition on the coin.
            return {"information_set": jnp.full((2, 1), state["turn"], dtype=jnp.float32)}

        def legal_action_mask(self, state):
            return jnp.ones((2, 2), dtype=bool)

        def active_players(self, state):
            return jnp.full((2,), state["turn"] < 2)

        def step(self, state, actions, key):
            del key
            after = dict(state, turn=state["turn"] + 1)
            done = after["turn"] == 2
            payoff = jnp.where(done, jnp.where(actions[0] == state["chance"], 1.0, -1.0), 0.0)
            return after, jnp.stack((payoff, -payoff)), done

    # A policy that never observes the fair coin must win half the time. This
    # also tests independent chance/action randomness across multiple turns.
    # Reusing the environment initialization stream makes every guess correct.
    result = average_utility(HiddenCoin(), uniform_policy, uniform_policy,
                             episodes=8192, batch_size=1024, seed=0)
    assert abs(result["mean_utility"][0]) < 0.06
    assert sum(result["mean_utility"]) == 0
    assert abs(result["win_rate"][0] - 0.5) < 0.03


@pytest.mark.parametrize("lam,expected_first", [(1.0, 0.5), (0.0, 0.375)])
def test_graph_defined_gae_crosses_inactive_steps_and_collects_terminal_reward(lam, expected_first):
    # P0 acts at t=0,2; P1 acts only at t=1. Terminal rewards arrive at t=3.
    # gamma=.5: P0's Monte Carlo return at t=0 is .5*1 + .5**3*4 = 1.
    # With lambda=0 its target instead bootstraps at t=2: .5*1 + .5**2*1.5.
    records = {
        "sampling_value": jnp.asarray([[0.5, 90], [91, -0.25], [1.5, 92], [93, 94]])[:, None],
        "reward": jnp.asarray([[0.0, 0], [1, -1], [0, 0], [4, -4]])[:, None],
    }
    active = jnp.asarray([[True, False], [False, True], [True, False], [False, False]])[:, None]
    algo = DRLGraph()
    transition_reward = aggregate(algo.reward, "sum", object="segment", discount=0.5)
    next_value = aggregate(algo.sampling_value, object="children", discount=0.5)
    delta = transition_reward + next_value - algo.sampling_value
    advantage_node = aggregate(delta, "sum", object="descendants", discount=0.5,
                               decay=lam, include_self=True)
    target_node = advantage_node + algo.sampling_value
    algo.metrics = {"advantage": advantage_node, "target": target_node}
    algo.init(jax.random.key(0), {})

    def compute(inputs, valid):
        prepared = algo.update((), inputs, upd_color=[0], valid=valid,
                               alive=jnp.ones(valid.shape, bool))
        return algo.evaluate((), prepared, (advantage_node, target_node))

    calculated = [jax.jit(compute)(
        {name: value[:, :, player] for name, value in records.items()}, active[:, :, player])
        for player in (0, 1)]
    advantage, target = [jnp.stack([item[index] for item in calculated], axis=2)
                         for index in (0, 1)]
    expected_advantage = np.array(
        [[expected_first, 0], [0, -1.75], [0.5, 0], [0, 0]])[:, None]
    expected_target = np.array(
        [[expected_first + 0.5, 0], [0, -2], [2, 0], [0, 0]])[:, None]
    np.testing.assert_allclose(jnp.where(active, advantage, 0), expected_advantage, rtol=0, atol=1e-7)
    np.testing.assert_allclose(jnp.where(active, target, 0), expected_target, rtol=0, atol=1e-7)


@pytest.mark.parametrize("device", jax.devices("cpu") + [d for d in jax.devices() if d.platform != "cpu"],
                         ids=lambda device: device.platform)
def test_training_updates_both_players_and_keeps_frozen_policy(device):
    with jax.default_device(device):
        env = Goofspiel(num_cards=3)
        algo = graph(*networks(), num_actions=3)
        trainer = Trainer(algo, env, batch_size=4, seed=0)
        previous = parameters(trainer.states)
        frozen = trainer.policy(0)
        state = env.init(jax.random.PRNGKey(0))
        obs, mask = env.features(state)["information_set"][0:1], env.legal_action_mask(state)[0:1]
        frozen_prediction = np.asarray(frozen(obs, mask))
        data = trainer.update(trainer.collect(), upd_color=[0])
        size = data[0]["_valid"].size
        data = jax.tree.map(lambda x: x.reshape((size,) + x.shape[2:]), data)
        for _ in range(2):
            for start in range(0, size, 5):
                batch = jax.tree.map(lambda x: x[start:start + 5], data)
                result = trainer.optimize(batch, upd_color=[1])
        assert result["iteration"] == 1
        assert result["trajectories"] == 4
        np.testing.assert_allclose(sum(result["mean_utility"]), 0, atol=1e-7)
        for player, (before, after) in enumerate(zip(previous, parameters(trainer.states))):
            assert any(not np.array_equal(np.asarray(a), np.asarray(b))
                       for a, b in zip(jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves(after)))
            assert all(int(state.step) > 0 for state in trainer.states[player])
            assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree_util.tree_leaves(after))
        assert_same_tree(frozen.params, previous[0])
        np.testing.assert_array_equal(np.asarray(frozen(obs, mask)), frozen_prediction)
        assert all(np.isfinite(result[name]).all()
                   for name in ("loss", "policy_loss", "value_loss", "entropy"))


def test_two_pure_policies_have_hand_computed_average_utility():
    env = Goofspiel(num_cards=7, points_order="ascending")
    result = average_utility(env, lowest_bid, highest_bid, episodes=13, batch_size=5, seed=0)
    # Ascending vs descending bids: P0 wins prizes 5+6+7, P1 wins 1+2+3.
    # Goofspiel point_difference for two players is half their score difference.
    assert result["episodes"] == 13
    np.testing.assert_array_equal(result["mean_utility"], [6, -6])
    np.testing.assert_array_equal(result["standard_error"], [0, 0])
    np.testing.assert_array_equal(result["confidence_interval_95"], [[6, 6], [-6, -6]])
    np.testing.assert_array_equal(result["win_rate"], [1, 0])
    assert result["tie_rate"] == 0


def test_training_and_matches_supply_the_actual_player_seat():
    class SeatPolicy(DRLGraph):
        def __init__(self):
            super().__init__()
            low = operation_node(lambda mask: lowest_bid(None, mask), self.legal_action_mask)
            high = operation_node(lambda mask: highest_bid(None, mask), self.legal_action_mask)
            self.strategy = where(self.player == 0, low, high)
            critic = model(nn.Dense(1))
            target = aggregate(self.reward, "sum", object="segment")
            with leg.backward(color=1):
                self.value = critic(self.env.information_set).squeeze()
                self.minimize((self.value - target) ** 2, models=[critic])

    env = Goofspiel(num_cards=7, points_order="ascending")
    trainer = Trainer(SeatPolicy(), env, batch_size=2)
    data = trainer.update(trainer.collect(), upd_color=[0])
    size = data[0]["_valid"].size
    data = jax.tree.map(lambda x: x.reshape((size,) + x.shape[2:]), data)
    for start in range(0, size, 7):
        batch = jax.tree.map(lambda x: x[start:start + 7], data)
        metrics = trainer.optimize(batch, upd_color=[1])
    np.testing.assert_array_equal(metrics["mean_utility"], [6, -6])
    # Even when the snapshot is used in a different seat, formulas must receive
    # the playing seat rather than the seat its parameters were trained in.
    result = average_utility(env, trainer.policy(1), trainer.policy(0), episodes=3, batch_size=2)
    np.testing.assert_array_equal(result["mean_utility"], [6, -6])


def test_evaluation_does_not_validate_unrequested_padding_episodes():
    env = Goofspiel(num_cards=7)
    key = jax.random.PRNGKey(0)
    initial = [env.init(jax.random.fold_in(jax.random.fold_in(key, index), 0))
               for index in range(2)]
    first_prize, padding_prize = [int(state.point_cards[0]) for state in initial]
    assert first_prize != padding_prize

    def policy(obs, mask):
        # The first revealed prize remains visible in every later observation.
        # Only the unused second episode is invalid; the requested one is legal.
        first_prize_offset = 2 * env.num_cards + 1
        invalid = obs[..., first_prize_offset + padding_prize - 1] > 0
        return jnp.where(invalid[..., None], 0, uniform_policy(obs, mask))

    result = average_utility(env, policy, lowest_bid, episodes=1, batch_size=2, seed=0)
    assert result["episodes"] == 1
    assert np.isfinite(result["mean_utility"]).all()


def test_match_seeds_chunking_and_uncertainty_match_independent_episode_utilities():
    env = Goofspiel(num_cards=7)
    episodes, seed = 17, 0
    first = average_utility(env, lowest_bid, highest_bid, episodes=episodes, batch_size=5, seed=seed)
    second = average_utility(env, lowest_bid, highest_bid, episodes=episodes, batch_size=8, seed=seed)
    assert first == second

    # Compute each payoff directly from its chance permutation and fixed bid order,
    # independently of the rollout evaluator and its running moments.
    key = jax.random.PRNGKey(seed)
    prizes = np.asarray(jax.vmap(lambda i: env.init(
        jax.random.fold_in(jax.random.fold_in(key, i), 0)).point_cards)(
            jnp.arange(episodes, dtype=jnp.uint32)))
    payoffs0 = (prizes[:, 4:].sum(axis=1) - prizes[:, :3].sum(axis=1)) / 2
    payoffs = np.stack((payoffs0, -payoffs0), axis=1)
    mean, stderr = payoffs.mean(axis=0), payoffs.std(axis=0, ddof=1) / np.sqrt(episodes)
    np.testing.assert_allclose(first["mean_utility"], mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(first["standard_error"], stderr, rtol=0, atol=1e-12)
    np.testing.assert_allclose(first["confidence_interval_95"],
                               np.stack((mean - 1.96 * stderr, mean + 1.96 * stderr), axis=-1), atol=1e-12)
    np.testing.assert_allclose(first["win_rate"], (payoffs > 0).mean(axis=0))
    assert first["tie_rate"] == (payoffs0 == 0).mean()


def test_policy_checkpoint_roundtrip_and_mismatched_architecture(tmp_path):
    env = Goofspiel(num_cards=3)
    trainer = Trainer(graph(*networks(), num_actions=3), env, batch_size=2)
    frozen = trainer.policy(1)
    path = tmp_path / "policies" / "player1.npz"
    frozen.save(path)
    restored = Policy.load(graph(*networks(), num_actions=3), path)
    assert_same_tree(restored.params, frozen.params)
    state = env.init(jax.random.PRNGKey(0))
    obs, mask = env.features(state)["information_set"], env.legal_action_mask(state)
    np.testing.assert_array_equal(np.asarray(restored(obs, mask)), np.asarray(frozen(obs, mask)))
    with pytest.raises(ValueError, match="architecture|shape"):
        Policy.load(graph(*networks(hidden_size=9), num_actions=3), path)


@pytest.mark.parametrize("kind", ["negative", "unnormalized", "nonfinite", "illegal"])
def test_evaluation_rejects_invalid_probabilities(kind):
    def bad_policy(obs, mask):
        base = uniform_policy(obs, mask)
        if kind == "negative":
            return -base
        if kind == "unnormalized":
            return base * 0.5
        if kind == "nonfinite":
            return base * jnp.nan
        return jnp.ones_like(base) / base.shape[-1]

    with pytest.raises(ValueError, match="invalid|illegal"):
        average_utility(Goofspiel(num_cards=3), bad_policy, lowest_bid, episodes=2, batch_size=2)


class IncompleteGoofspiel(Goofspiel):
    @property
    def max_steps(self):
        return self.num_cards - 1


@pytest.mark.parametrize("invalid_policy", [False, True])
def test_training_rejects_invalid_episodes_without_committing_state(invalid_policy):
    algo = graph(*networks(hidden_size=4), num_actions=3)
    if invalid_policy:
        algo.strategy = algo.strategy * 0.5
        env = Goofspiel(num_cards=3)
    else:
        env = IncompleteGoofspiel(num_cards=3)
    trainer = Trainer(algo, env, batch_size=2)
    initial_states, initial_key = trainer.states, trainer.key
    with pytest.raises(ValueError, match="invalid|illegal|finish"):
        trainer.collect()
    assert trainer.iteration == 0
    assert_same_tree(trainer.states, initial_states)
    np.testing.assert_array_equal(np.asarray(trainer.key), np.asarray(initial_key))


def test_evaluation_rejects_incomplete_episodes():
    with pytest.raises(ValueError, match="finish"):
        average_utility(IncompleteGoofspiel(num_cards=3), lowest_bid, highest_bid,
                        episodes=3, batch_size=2)


def test_nonfinite_parameter_update_is_rejected_without_committing_state():
    class ScalarModel:
        def init(self, key, example):
            del key, example
            return jnp.array(0.0)

        def apply(self, params, observations):
            return jnp.broadcast_to(params, observations.shape[:-1])

    class SingularGradient(DRLGraph):
        def __init__(self):
            super().__init__()
            resource = model(ScalarModel())
            self.value = resource(self.env.information_set)
            self.strategy = masked_softmax(self.legal_action_mask.astype("float32"),
                                           self.legal_action_mask)
            # sqrt(0) has a finite loss but an infinite derivative. Checking only
            # reported losses cannot establish a finite, usable model update.
            self.minimize(self.value ** 0.5, models=[resource])

    trainer = Trainer(SingularGradient(), Goofspiel(num_cards=2),
                      batch_size=2)
    data = trainer.collect()
    size = data[0]["_valid"].size
    batch = jax.tree.map(lambda x: x.reshape((size,) + x.shape[2:]), data)
    initial_states, initial_key = trainer.states, trainer.key
    with pytest.raises(FloatingPointError, match="finite|Finite"):
        trainer.optimize(batch, upd_color=[0])
    assert trainer.iteration == 1
    assert_same_tree(trainer.states, initial_states)
    np.testing.assert_array_equal(np.asarray(trainer.key), np.asarray(initial_key))


def test_trainer_hides_padding_in_loss_gradients_and_skips_empty_minibatches():
    class Regression(DRLGraph):
        def __init__(self):
            super().__init__()
            resource = model(
                init=lambda key, example: jnp.array(1.0),
                apply=lambda weight, inputs: weight * inputs.sum(),
            )
            self.value = resource(self.env.information_set)
            self.strategy = self.legal_action_mask.astype("float32") / self.action_set_size
            # Valid inputs are positive. An unfiltered zero padding record
            # would have an infinite sqrt derivative despite a masked loss.
            self.minimize((self.value ** 0.5 - self.reward) ** 2, models=[resource])

    trainer = Trainer(Regression(), Goofspiel(num_cards=2), batch_size=2)
    states = trainer.states[0]
    width = trainer.graph.input_specs["information_set"].shape[0]
    examples = jnp.pad(jnp.array([[1.0, 2.0], [2.0, 4.0]]), ((0, 0), (0, width - 2)))
    records = {"information_set": examples,
               "reward": jnp.array([0.2, 0.4]), "_valid": jnp.ones(2, bool)}
    padded = jax.tree.map(lambda value: jnp.pad(
        value, ((0, 2),) + ((0, 0),) * (value.ndim - 1)), records)
    update = jax.jit(trainer._update_minibatch)
    expected, expected_metrics = update(states, records)
    actual, actual_metrics = update(states, padded)
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        assert np.isfinite(np.asarray(left)).all()
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)
    for left, right in zip(actual_metrics, expected_metrics):
        np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)
    empty = jax.tree.map(jnp.zeros_like, padded)
    skipped, (losses, count) = update(states, empty)
    assert_same_tree(skipped, states)
    np.testing.assert_array_equal(losses, 0)
    assert float(count) == 0


def test_collect_does_not_execute_the_critic_or_prepare_graph_colors():
    class SamplingOnly(DRLGraph):
        def __init__(self):
            super().__init__()

            def critic_apply(params, inputs):
                raise AssertionError("The collection phase must not call the critic")

            critic = model(init=lambda key, inputs: jnp.array(1.0), apply=critic_apply)
            with leg.backward(color=0):
                self.value = critic(self.env.information_set)
                self.loss = self.value ** 2
                self.minimize(self.loss, models=[critic])
            with leg.backward(color=1):
                self.strategy = self.legal_action_mask.astype("float32") / self.action_set_size

    trainer = Trainer(SamplingOnly(), Goofspiel(num_cards=2), batch_size=2)
    states = trainer.states
    data = trainer.collect()
    assert trainer.iteration == 1
    assert_same_tree(states, trainer.states)
    assert all("sampling_value" not in records for records in data)
    assert all(not any(name.startswith("_drl_node_") for name in records) for records in data)
    with pytest.raises(AssertionError, match="critic"):
        trainer.update(data, upd_color=[0])


def test_optimize_selects_objectives_metrics_and_model_updates_by_color():
    class TwoObjectives(DRLGraph):
        def __init__(self):
            super().__init__()
            self.strategy = self.legal_action_mask.astype("float32") / self.action_set_size
            for color in (0, 1):
                with leg.backward(color=color):
                    resource = model(init=lambda key, x: jnp.array(1.0),
                                     apply=lambda weight, x: weight * x.sum())
                    prediction = resource(self.env.information_set)
                    loss = (prediction - self.reward) ** 2
                    self.minimize(loss, models=[resource])
                    self.metrics[f"color{color}"] = loss

    trainer = Trainer(TwoObjectives(), Goofspiel(num_cards=2), batch_size=2)
    width = trainer.graph.input_specs["information_set"].shape[0]
    batch = tuple({"information_set": jnp.ones((2, width)), "reward": jnp.zeros(2),
                   "_valid": jnp.ones(2, bool)} for _ in range(2))
    before = trainer.states
    report = trainer.optimize(batch, upd_color=[1])
    assert "color1" in report and "color0" not in report
    for old, new in zip(before, trainer.states):
        assert_same_tree(old[0], new[0])
        assert int(new[1].step) == 1
        assert not np.array_equal(old[1].params, new[1].params)
    with pytest.raises(ValueError, match="no training objective"):
        trainer.optimize(batch, upd_color=[7])


def test_tabular_imports_do_not_require_optional_drl_dependencies():
    script = """
import importlib.abc
import sys
class NoDeepLearning(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'jax', 'jaxlib', 'optax'}:
            raise ModuleNotFoundError('Optional dependency blocked: ' + fullname)
sys.meta_path.insert(0, NoDeepLearning())
import LiteEFG
from LiteEFG.baselines import CFR
assert callable(CFR.graph)
assert 'jax' not in sys.modules
assert 'optax' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script],
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
