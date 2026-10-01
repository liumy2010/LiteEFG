"""Check model identity, routed gradients, optimizer ownership, and frozen exports."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
optax = pytest.importorskip("optax")
nn = pytest.importorskip("flax.linen")

import LiteEFG as leg
from LiteEFG.baselines.drl.PPO import graph as PPO
from LiteEFG.drl.graph import operation_node
from LiteEFG.drl.runtime import Policy


class FeatureEnvironment:
    num_players = 2
    num_actions = 2
    max_steps = 1

    def init(self, key):
        return jnp.array(0, jnp.int32)

    def features(self, state):
        return {"information_set": jnp.ones((2, 2)), "target": jnp.zeros(2),
                "full_info": jnp.zeros((2, 3))}

    def legal_action_mask(self, state):
        return jnp.ones((2, 2), bool)

    def active_players(self, state):
        return jnp.ones(2, bool)

    def step(self, state, actions, key):
        reward = (actions[0] - actions[1]).astype(jnp.float32)
        return state + 1, jnp.stack((reward, -reward)), jnp.array(True)


class Regression(leg.DRLGraph):
    def __init__(self, local, common):
        super().__init__()
        self.local = leg.ModelList(local) if local else None
        self.common = common
        with leg.backward(color=0):
            if self.local is not None:
                prediction = self.local[self.player](self.env.information_set)
                self.minimize((prediction - self.env.target) ** 2, models=local)
            self.value = common(self.env.information_set)
            self.minimize((self.value - self.env.target) ** 2, models=[common])
            self.strategy = self.legal_action_mask / self.action_set_size


def make_regression(optimizer, *, local=True):
    initialized = []

    def resource(name):
        def initialize(key, inputs):
            initialized.append(name)
            return jnp.zeros(inputs.shape[-1])
        return leg.model(init=initialize, apply=lambda weights, inputs: jnp.dot(weights, inputs),
                         optimizer=optimizer, name=name)

    local_models = [resource(f"local{player}") for player in range(2)] if local else []
    return Regression(local_models, resource("common")), initialized


def records(inputs, targets, valid, *, player=0):
    return {"information_set": jnp.array(inputs, dtype=jnp.float32),
            "target": jnp.array(targets, dtype=jnp.float32), "_valid": jnp.array(valid),
            "player": jnp.full(len(valid), player, dtype=jnp.int32)}


@pytest.mark.parametrize("optimizer", [optax.sgd(0.1),
    optax.chain(optax.clip_by_global_norm(0.6), optax.adam(0.1))])
def test_common_resource_matches_one_pooled_update_before_clipping_and_adam(optimizer):
    algo, initialized = make_regression(optimizer)
    trainer = algo.bind(FeatureEnvironment(), batch_size=1).trainer
    assert initialized == ["local0", "local1", "common"]
    assert len(algo.models) == 3
    assert trainer.model_state(algo.common) is algo.common.state
    expected = algo.common.state
    batches = [
        (records([[1, 0], [2, 0], [99, 99]], [2, 1, 99], [True, True, False]),
         records([[0, 1], [99, 99], [99, 99]], [3, 99, 99], [True, False, False], player=1)),
        (records([[1, 1], [2, 1], [99, 99]], [-1, 2, 99], [True, True, False]),
         records([[2, -1], [99, 99], [99, 99]], [-4, 99, 99], [True, False, False], player=1)),
    ]
    for step, batch in enumerate(batches, 1):
        # Directly concatenate valid records to independently check the runtime's
        # per-seat reductions, nonlinear clipping, and Adam step count.
        inputs = jnp.concatenate([record["information_set"][record["_valid"]] for record in batch])
        targets = jnp.concatenate([record["target"][record["_valid"]] for record in batch])
        gradient = 2 * inputs.T @ (inputs @ expected.params - targets) / len(targets)
        updates, expected_optimizer = optimizer.update(gradient, expected.opt_state, expected.params)
        expected = expected._replace(params=optax.apply_updates(expected.params, updates),
                                     opt_state=expected_optimizer)
        old_local = [resource.state for resource in algo.local]
        report = trainer.optimize(batch, upd_color=[0])
        assert report["valid_samples"] == [2, 1]
        actual = algo.common.state
        assert int(actual.step) == step
        np.testing.assert_allclose(actual.params, expected.params, rtol=1e-6, atol=1e-7)
        for left, right in zip(jax.tree_util.tree_leaves(actual.opt_state),
                               jax.tree_util.tree_leaves(expected.opt_state)):
            np.testing.assert_allclose(left, right, rtol=1e-6, atol=1e-7)
        for player, resource in enumerate(algo.local):
            valid = batch[player]["_valid"]
            inputs = batch[player]["information_set"][valid]
            targets = batch[player]["target"][valid]
            gradient = 2 * inputs.T @ (inputs @ old_local[player].params - targets) / len(targets)
            updates, _ = optimizer.update(gradient, old_local[player].opt_state, old_local[player].params)
            np.testing.assert_allclose(resource.state.params,
                                       optax.apply_updates(old_local[player].params, updates),
                                       rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("active_player", [0, 1])
def test_single_resource_ignores_an_entire_inactive_nan_seat(active_player):
    algo, _ = make_regression(optax.adam(0.1), local=False)
    algo.bind(FeatureEnvironment(), batch_size=1)
    valid = records([[1, 2], [3, 4]], [1, 2], [True, True], player=active_player)
    invalid = records([[np.nan, np.nan]] * 2, [np.nan] * 2, [False, False], player=1-active_player)
    batch = (valid, invalid) if active_player == 0 else (invalid, valid)
    report = algo.trainer.optimize(batch, upd_color=[0])
    assert report["valid_samples"][active_player] == 2
    assert report["loss"][1 - active_player] == 0
    assert int(algo.common.state.step) == 1
    assert np.isfinite(algo.common.state.params).all()
    before = algo.common.state
    report = algo.trainer.optimize((invalid, invalid), upd_color=[0])
    assert report["valid_samples"] == [0, 0]
    assert report["loss"] == [0, 0]
    assert algo.common.state is before


def test_duplicated_list_entries_and_direct_calls_count_participating_records_once():
    resource = leg.model(init=lambda key, x: jnp.zeros(2),
                         apply=lambda weight, x: jnp.dot(weight, x), optimizer=optax.sgd(0.1))

    class Repeated(leg.DRLGraph):
        def __init__(self):
            super().__init__()
            routes = leg.ModelList([resource, resource])
            self.value = routes[self.player](self.env.information_set) + resource(self.env.information_set)
            self.minimize((self.value - self.env.target) ** 2, models=[resource])
            self.strategy = self.legal_action_mask / self.action_set_size

    algo = Repeated().bind(FeatureEnvironment(), batch_size=1)
    assert algo.models == [resource]
    batch = (records([[1, 0]], [2], [True]), records([[0, 1]], [4], [True], player=1))
    algo.trainer.optimize(batch, upd_color=[0])
    # d mean((2*w.x - y)^2) / dw at w=0 is [-4, -8].
    np.testing.assert_allclose(resource.state.params, [0.4, 0.8], rtol=1e-6)
    assert int(resource.state.step) == 1


def test_unselected_resources_do_not_advance_adam_moments_or_weight_decay():
    algo, _ = make_regression(optax.adamw(0.1, weight_decay=0.2))
    algo.bind(FeatureEnvironment(), batch_size=1)
    batch = (records([[1, 2]], [2], [True]), records([[3, 4]], [1], [True], player=1))
    algo.trainer.optimize(batch, upd_color=[0])
    skipped = algo.local[1].state
    batch = (batch[0], records([[np.nan, np.nan]], [np.nan], [False], player=1))
    algo.trainer.optimize(batch, upd_color=[0])
    assert algo.local[1].state is skipped
    assert int(algo.local[0].state.step) == int(algo.common.state.step) == 2


def test_unselected_branches_cannot_poison_another_seats_gradient():
    resources = [leg.model(init=lambda key, x: jnp.array(1.0),
                           apply=lambda weight, x, sign=sign: weight * jnp.sqrt(sign * x[0]),
                           optimizer=optax.sgd(0.1)) for sign in (1, -1)]

    class Routed(leg.DRLGraph):
        def __init__(self):
            super().__init__()
            self.value = leg.ModelList(resources)[self.player](self.env.information_set)
            self.minimize(self.value ** 2, models=resources)
            self.strategy = self.legal_action_mask / self.action_set_size

    algo = Routed().bind(FeatureEnvironment(), batch_size=1)
    # Each chosen branch is finite; running the other seat's branch produces
    # sqrt(-1). Its unused gradient must never enter that resource's update.
    batch = (records([[1, 0]], [0], [True]), records([[-1, 0]], [0], [True], player=1))
    algo.trainer.optimize(batch, upd_color=[0])
    for resource in resources:
        np.testing.assert_allclose(resource.state.params, 0.8, rtol=1e-6)
        assert int(resource.state.step) == 1


def test_model_lists_must_match_the_environment_player_count():
    actor = leg.model(nn.Dense(2))
    critic = leg.model(nn.Dense(1))
    algo = PPO([actor], critic, num_actions=2)
    with pytest.raises(ValueError, match="ModelList.*player"):
        algo.bind(FeatureEnvironment())


def test_frozen_policy_roundtrip_requires_only_environment_observation(tmp_path):
    class ColoredEnvironment(FeatureEnvironment):
        def features(self, state):
            return {**super().features(state),
                    "information_set": jnp.array([[0.3, 0.7, 0.0], [0.3, 0.7, 1.0]])}

    def graph():
        actors = [leg.model(nn.Dense(2, use_bias=False)) for _ in range(2)]
        return PPO(actors, leg.model(nn.Dense(1)), num_actions=2)

    env = ColoredEnvironment()
    algo = graph().bind(env, batch_size=1)
    trainer = algo.trainer
    assert algo.actor[0].state is not algo.actor[1].state
    for player in (0, 1):
        policy = trainer.policy(player)
        observation = env.features(env.init(jax.random.PRNGKey(0)))["information_set"][player]
        features = {"information_set": observation}
        legal = jnp.array([True, True])
        expected = policy(features, legal)
        actor = algo.actor[player]
        logits = actor.apply(policy.params[algo.model_index(actor)], observation)
        np.testing.assert_allclose(expected, jax.nn.softmax(logits), rtol=1e-6)
        path = tmp_path / f"player-{player}.npz"
        policy.save(path)
        restored = Policy.load(graph(), path)
        assert restored.player == restored.model_player == player
        np.testing.assert_allclose(restored(features, legal), expected, rtol=1e-6)
        np.testing.assert_allclose(restored(observation, legal, player=1-player), expected, rtol=1e-6)


def test_cross_graph_updates_are_live_while_policy_snapshots_stay_frozen():
    calls = []

    def init(key, inputs):
        calls.append(1)
        return jnp.zeros(2)

    resource = leg.model(init=init, apply=lambda weight, x: jnp.dot(weight, x), optimizer=optax.sgd(0.1))

    class ResourcePolicy(leg.DRLGraph):
        def __init__(self):
            super().__init__()
            self.value = resource(self.env.information_set)
            logits = operation_node(lambda value: jnp.stack((value, -value)), self.value)
            self.strategy = leg.masked_softmax(logits, self.legal_action_mask)
            self.minimize((self.value - self.env.target) ** 2, models=[resource])

    first = ResourcePolicy().bind(FeatureEnvironment(), batch_size=2)
    second = ResourcePolicy().bind(FeatureEnvironment(), batch_size=2)
    assert calls == [1]
    assert first.trainer.model_state(resource) is second.trainer.model_state(resource)
    frozen = second.trainer.policy(1)
    observation, legal = jnp.ones(2), jnp.ones(2, bool)
    frozen_prediction = frozen(observation, legal)
    before = second.trainer.collect()
    prepared = second.trainer.update(before, upd_color=[0])
    key = second._node_key(second.value.index)
    np.testing.assert_array_equal(prepared[0][key], 0)
    batch = (records([[1, 0]], [2], [True]), records([[0, 1]], [4], [True], player=1))
    first.trainer.optimize(batch, upd_color=[0])
    assert second.trainer.model_state(resource) is resource.state
    refreshed = second.trainer.update(before, upd_color=[0])
    np.testing.assert_allclose(refreshed[0][key], 0.6)
    after = second.trainer.collect()
    assert not np.allclose(after[0]["log_sampling_prob"], before[0]["log_sampling_prob"])
    np.testing.assert_array_equal(frozen(observation, legal), frozen_prediction)
    second.trainer.optimize(batch, upd_color=[0])
    np.testing.assert_allclose(resource.state.params, [0.38, 0.76], rtol=1e-6)
    assert int(resource.state.step) == 2


def test_failed_update_keeps_every_caller_owned_resource_unchanged():
    algo, _ = make_regression(optax.adam(0.1))
    algo.bind(FeatureEnvironment(), batch_size=1)
    before = tuple(resource.state for resource in algo.models)
    valid = records([[1, 2]], [1], [True])
    invalid = records([[1, 2]], [np.inf], [True], player=1)
    with pytest.raises(FloatingPointError):
        algo.trainer.optimize((valid, invalid), upd_color=[0])
    assert all(resource.state is state for resource, state in zip(algo.models, before))


def test_frozen_policy_keeps_source_head_but_receives_actual_player_seat(tmp_path):
    class HeadPolicy(leg.DRLGraph):
        def __init__(self):
            heads = [leg.model(init=lambda key, x, seat=seat: jnp.array([2.0, 0.0]) if seat == 0
                               else jnp.array([0.0, 2.0]), apply=lambda weights, x: weights)
                     for seat in range(2)]
            super().__init__()
            self.heads = leg.ModelList(heads)
            logits = self.heads[self.player](self.env.information_set)
            logits = logits + self.player * jnp.array([1.0, -1.0])
            self.strategy = leg.masked_softmax(logits, self.legal_action_mask)
            self.value = logits.sum()
            self.minimize(self.value ** 2, models=heads)

    algo = HeadPolicy().bind(FeatureEnvironment(), batch_size=1)
    snapshot = algo.trainer.policy(1)
    observation, legal = jnp.ones(2), jnp.ones(2, bool)
    np.testing.assert_allclose(snapshot(observation, legal, player=0), jax.nn.softmax(jnp.array([0., 2.])))
    np.testing.assert_allclose(snapshot(observation, legal, player=1), [0.5, 0.5])
    path = tmp_path / "head1.npz"
    snapshot.save(path)
    restored = Policy.load(HeadPolicy(), path)
    np.testing.assert_allclose(restored(observation, legal, player=0), snapshot(observation, legal, player=0))
    with pytest.raises(ValueError, match="ModelList"):
        Policy(algo, snapshot.params, 2, 2, model_player=2)
    with pytest.raises(ValueError, match="integer"):
        Policy(algo, snapshot.params, 2, 2, model_player=True)
    routed = Policy(algo, snapshot.params, 2, 2)
    with pytest.raises(ValueError, match="ModelList"):
        routed(observation, legal, player=2)
