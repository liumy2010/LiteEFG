"""Environment-defined inputs retain local shapes through training and matches."""

import json
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
optax = pytest.importorskip("optax")
nn = pytest.importorskip("flax.linen")

from LiteEFG.drl.environment import Environment
from LiteEFG.drl.graph import DRLGraph, masked_softmax, model, operation_node
from LiteEFG.drl.runtime import (
    Policy, Trainer, average_utility, example_inputs, rollout, uniform_policy,
)


class FeatureGame(Environment):
    """Three decisions with different matrix and scalar features at each seat."""

    num_players = 2
    num_actions = 2
    max_steps = 3

    def init(self, key):
        del key
        return {"turn": jnp.array(0, jnp.int32), "total": jnp.array(0.0)}

    def features(self, state):
        seat = jnp.arange(2, dtype=jnp.int32)
        offset = 10 * state["turn"] + 100 * seat
        return {
            "information_set": jnp.full((2, 1), state["turn"], jnp.float32),
            "full_info_feature": offset[:, None, None].astype(jnp.float32)
                                 + jnp.arange(4, dtype=jnp.float32).reshape(2, 2),
            "seat_turn": state["turn"] + 3 * seat,
            "feature_flag": (state["turn"] + seat) % 2 == 0,
        }

    def legal_action_mask(self, state):
        del state
        return jnp.ones((2, 2), bool)

    def active_players(self, state):
        return jnp.full((2,), state["turn"] < self.max_steps)

    def step(self, state, actions, key):
        del key
        desired = (state["turn"] + 3 * jnp.arange(2)) % 2
        payoff = jnp.where(actions[0] == desired[0], 1.0, -1.0)
        payoff += jnp.where(actions[1] == desired[1], 2.0, -2.0)
        after = {"turn": state["turn"] + 1, "total": state["total"] + payoff}
        done = after["turn"] == self.max_steps
        reward = jnp.where(done, after["total"], 0.0)
        return after, jnp.stack((reward, -reward)), done


class FeatureGraph(DRLGraph):
    def __init__(self, *, feature_actor=True):
        super().__init__()
        if feature_actor:
            self.strategy = operation_node(
                lambda turn: jax.nn.one_hot(turn % 2, 2), self.env.seat_turn)
        else:
            self.strategy = self.legal_action_mask.astype(jnp.float32) / self.action_set_size
        self.critic = model(nn.Dense(1), optimizer=optax.sgd(0.005))
        self.value = self.critic(self.env.full_info_feature.reshape(4) / 100).squeeze()
        self.loss = (self.value - self.reward) ** 2
        self.minimize(self.loss, models=[self.critic])
        self.metrics["flag"] = self.env.feature_flag.astype(jnp.float32)


def train(trainer, iterations=1):
    for _ in range(iterations):
        data = trainer.collect()
        size = data[0]["_valid"].size
        data = jax.tree.map(lambda x: x.reshape((size,) + x.shape[2:]), data)
        for start in range(0, size, 5):
            batch = jax.tree.map(lambda x: x[start:start + 5], data)
            metrics = trainer.optimize(batch, upd_color=[0])
    return metrics


def devices():
    return jax.devices("cpu") + [device for device in jax.devices() if device.platform != "cpu"]


def input_batch(env, count=3):
    states = jax.vmap(env.init)(jax.random.split(jax.random.key(0), count))
    return (
        jax.vmap(env.features)(states)["information_set"][:, 0],
        jax.vmap(env.legal_action_mask)(states)[:, 0],
        {name: value[:, 0] for name, value in jax.vmap(env.features)(states).items()},
    )


@pytest.mark.parametrize("device", devices(), ids=lambda device: device.platform)
def test_rollout_preserves_pre_action_matrix_scalar_and_seat_features(device):
    with jax.default_device(device):
        env = FeatureGame()

        def policy(observation, mask, features):
            del mask
            # The feature-aware callback receives the same standard vector
            # both as its first argument and in the complete feature mapping.
            assert observation.shape == features["information_set"].shape
            probabilities = jax.nn.one_hot(features["seat_turn"] % 2, 2)
            value = features["full_info_feature"].sum(axis=(-2, -1))
            value += features["seat_turn"]
            return probabilities, value

        keys = jax.random.split(jax.random.key(0), 4)
        records, complete = jax.jit(lambda keys: rollout(
            env, (policy, policy), keys, feature_policies=True))(keys)
        assert np.asarray(complete).all()
        assert records["full_info_feature"].shape == (3, 4, 2, 2, 2)
        assert records["information_set"].shape == (3, 4, 2, 1)
        assert records["seat_turn"].shape == (3, 4, 2)
        assert records["feature_flag"].dtype == jnp.bool_
        assert not {"old_log_prob", "old_value", "advantage", "return_target"} & records.keys()
        np.testing.assert_array_equal(records["log_sampling_prob"], np.zeros((3, 4, 2)))
        assert all(leaf.device.platform == device.platform for leaf in jax.tree.leaves(records))
        for turn in range(3):
            for seat in range(2):
                expected = (10 * turn + 100 * seat + np.arange(4)).reshape(2, 2)
                np.testing.assert_array_equal(records["full_info_feature"][turn, :, seat],
                                              np.broadcast_to(expected, (4, 2, 2)))
                np.testing.assert_array_equal(records["seat_turn"][turn, :, seat], turn + 3 * seat)
                np.testing.assert_array_equal(records["action"][turn, :, seat], (turn + 3 * seat) % 2)
                np.testing.assert_array_equal(records["sampling_value"][turn, :, seat],
                                              expected.sum() + turn + 3 * seat)
        np.testing.assert_array_equal(records["reward"].sum(0), np.tile([9, -9], (4, 1)))


@pytest.mark.parametrize("device", devices(), ids=lambda device: device.platform)
def test_custom_features_initialize_models_and_survive_training_minibatches(device):
    with jax.default_device(device):
        env, graph = FeatureGame(), FeatureGraph()
        assert not hasattr(env, "observe") and not hasattr(env, "observation_size")
        trainer = Trainer(graph, env, batch_size=4, seed=0)
        assert graph.input_specs["information_set"] == jax.ShapeDtypeStruct((1,), jnp.float32)
        assert graph.input_specs["full_info_feature"].shape == (2, 2)
        assert graph.input_specs["seat_turn"].shape == ()
        assert graph.input_specs["seat_turn"].dtype == jnp.int32
        assert graph.input_specs["feature_flag"].dtype == jnp.bool_
        before = [jax.tree.leaves(player[0].params) for player in trainer.states]
        report = train(trainer, 2)
        assert report["iteration"] == 2
        np.testing.assert_array_equal(report["mean_utility"], [9, -9])
        assert np.isfinite(report["loss"]).all()
        for player, old_leaves in zip(trainer.states, before):
            assert any(not np.array_equal(old, new)
                       for old, new in zip(old_leaves, jax.tree.leaves(player[0].params)))
            assert int(player[0].step) > 0
            assert all(leaf.device.platform == device.platform for leaf in jax.tree.leaves(player))


def test_matrix_input_gradient_matches_direct_external_model():
    with jax.default_device(jax.devices("cpu")[0]):
        env, graph = FeatureGame(), FeatureGraph()
        trainer = Trainer(graph, env, batch_size=2)
        parameters = tuple(state.params for state in trainer.states[0])
        matrices = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 2, 2)
        targets = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)

        def graph_loss(params):
            losses, = graph.evaluate(params, {
                "full_info_feature": matrices, "reward": targets,
            }, (graph.loss,))
            return losses.sum()

        def direct_loss(params):
            predictions = graph.critic.apply(params[0], matrices.reshape(2, 3, 4) / 100)[..., 0]
            return ((predictions - targets) ** 2).sum()

        actual, gradient = jax.jit(jax.value_and_grad(graph_loss))(parameters)
        expected, expected_gradient = jax.value_and_grad(direct_loss)(parameters)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
        for leaf, expected_leaf in zip(jax.tree.leaves(gradient), jax.tree.leaves(expected_gradient)):
            np.testing.assert_allclose(leaf, expected_leaf, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("information_dtype", [jnp.float32, jnp.int32])
def test_custom_policy_checkpoint_restores_feature_shapes_and_dtypes(tmp_path, information_dtype):
    class TypedFeatureGame(FeatureGame):
        def features(self, state):
            features = super().features(state)
            features["information_set"] = features["information_set"].astype(information_dtype)
            return features

    with jax.default_device(jax.devices("cpu")[0]):
        env, graph = TypedFeatureGame(), FeatureGraph()
        trainer = Trainer(graph, env, batch_size=2)
        train(trainer)
        policy = trainer.policy(0)
        observations, mask, features = input_batch(env)
        expected = jax.jit(policy)(features, mask)
        extra = {name: value for name, value in features.items() if name != "information_set"}
        np.testing.assert_array_equal(policy(observations, mask, features=extra), expected)
        path = tmp_path / "custom-features.npz"
        policy.save(path)
        with np.load(path, allow_pickle=False) as archive:
            assert json.loads(str(archive["metadata"]))["format_version"] == 4
        restored = Policy.load(FeatureGraph(), path)
        actual = jax.jit(restored)(features, mask)
        np.testing.assert_array_equal(actual, expected)
        for old, new in zip(jax.tree.leaves(policy.params), jax.tree.leaves(restored.params)):
            np.testing.assert_array_equal(old, new)
        assert restored.graph.input_specs["full_info_feature"].shape == (2, 2)
        assert restored.graph.input_specs["seat_turn"].dtype == jnp.int32
        assert restored.graph.input_specs["feature_flag"].dtype == jnp.bool_
        assert restored.graph.input_specs["information_set"] == jax.ShapeDtypeStruct((1,), information_dtype)
        with pytest.raises((ValueError, KeyError), match="seat_turn"):
            restored(observations, mask)


def test_privileged_critic_feature_is_not_required_for_actor_inference(tmp_path):
    with jax.default_device(jax.devices("cpu")[0]):
        env = FeatureGame()
        trainer = Trainer(FeatureGraph(feature_actor=False), env,
                          batch_size=2)
        policy = trainer.policy(0)
        observations, mask, _ = input_batch(env)
        expected = np.full((3, 2), 0.5)
        np.testing.assert_array_equal(policy(observations, mask), expected)
        np.testing.assert_array_equal(policy({"information_set": observations}, mask), expected)
        path = tmp_path / "critic-features.npz"
        policy.save(path)
        restored = Policy.load(FeatureGraph(feature_actor=False), path)
        np.testing.assert_array_equal(restored(observations, mask), expected)


def test_average_utility_passes_playing_seat_features_and_retains_two_argument_callables():
    with jax.default_device(jax.devices("cpu")[0]):
        env = FeatureGame()
        trainer = Trainer(FeatureGraph(), env, batch_size=2)
        # Swapping checkpoint identities must not swap the live environment's
        # seat-dependent features. Every prescribed action is correct: 3*3=9.
        result = average_utility(env, trainer.policy(1), trainer.policy(0),
                                 episodes=5, batch_size=3, seed=0)
        np.testing.assert_array_equal(result["mean_utility"], [9, -9])
        np.testing.assert_array_equal(result["standard_error"], [0, 0])

        def fixed_zero(observation, mask):
            del mask
            return jnp.broadcast_to(jnp.array([1.0, 0.0]), (observation.shape[0], 2))

        # Correct P0 always contributes +1; fixed P1 contributes -2,+2,-2.
        mixed = average_utility(env, trainer.policy(1), fixed_zero, episodes=5, batch_size=3)
        np.testing.assert_array_equal(mixed["mean_utility"], [1, -1])
        pure = average_utility(env, fixed_zero, fixed_zero, episodes=5, batch_size=3)
        np.testing.assert_array_equal(pure["mean_utility"], [-1, 1])


def test_missing_graph_feature_is_reported_during_trainer_initialization():
    class MissingFeatureGame(FeatureGame):
        def features(self, state):
            return {name: value for name, value in super().features(state).items()
                    if name != "full_info_feature"}

    with pytest.raises((ValueError, KeyError), match="full_info_feature"):
        Trainer(FeatureGraph(), MissingFeatureGame(), batch_size=2)


def test_class_declared_action_count_is_checked_without_declaring_an_input():
    class WrongActionCount(FeatureGraph):
        num_actions = 3

    with pytest.raises(ValueError, match="num_actions must match"):
        Trainer(WrongActionCount(), FeatureGame(), batch_size=2)


@pytest.mark.parametrize("invalid_feature", [
    "reserved", "player_axis", "not_mapping", "missing_information_set",
    "information_set_scalar", "information_set_matrix", "information_set_player_axis",
    "information_set_empty", "removed_information_state",
])
def test_invalid_environment_feature_contract_is_rejected(invalid_feature):
    class InvalidFeatureGame(FeatureGame):
        def features(self, state):
            result = super().features(state)
            if invalid_feature == "reserved":
                result["reward"] = jnp.zeros(2)
            elif invalid_feature == "player_axis":
                result["full_info_feature"] = jnp.zeros((3, 2, 2))
            elif invalid_feature == "not_mapping":
                return jnp.zeros((2, 3))
            elif invalid_feature == "missing_information_set":
                result.pop("information_set")
            elif invalid_feature == "information_set_scalar":
                result["information_set"] = jnp.zeros(2)
            elif invalid_feature == "information_set_matrix":
                result["information_set"] = jnp.zeros((2, 3, 4))
            elif invalid_feature == "information_set_player_axis":
                result["information_set"] = jnp.zeros((3, 2))
            elif invalid_feature == "information_set_empty":
                result["information_set"] = jnp.zeros((2, 0))
            else:
                result["information_state"] = result["information_set"]
            return result

    with pytest.raises((TypeError, ValueError)):
        Trainer(FeatureGraph(), InvalidFeatureGame(), batch_size=2)


def test_legacy_rollout_callback_ignores_optional_features():
    env = FeatureGame()
    keys = jax.random.split(jax.random.key(0), 2)

    def legacy_policy(observation, mask):
        return uniform_policy(observation, mask), jnp.zeros(observation.shape[0])

    records, complete = jax.jit(lambda keys: rollout(
        env, (legacy_policy, legacy_policy), keys))(keys)
    assert np.asarray(complete).all()
    assert records["action"].shape == (3, 2, 2)


@pytest.mark.parametrize("version", [1, 2])
def test_legacy_checkpoint_without_standard_feature_spec_remains_loadable(tmp_path, version):
    class OrdinaryGraph(DRLGraph):
        def __init__(self):
            super().__init__()
            resource = model(nn.Dense(2))
            logits = resource(self.env.information_set)
            self.strategy = masked_softmax(logits, self.legal_action_mask)
            self.value = logits.sum()
            self.minimize(self.value ** 2, models=[resource])

    with jax.default_device(jax.devices("cpu")[0]):
        env = FeatureGame()
        trainer = Trainer(OrdinaryGraph(), env, batch_size=2)
        policy = trainer.policy(0)
        current_path, legacy_path = tmp_path / "current.npz", tmp_path / f"legacy-v{version}.npz"
        policy.save(current_path)
        with np.load(current_path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            metadata["format_version"] = version
            metadata["observation_size"] = policy.observation_size
            if version == 1:
                metadata.pop("input_specs", None)
            else:
                metadata["input_specs"].pop("information_set", None)
            arrays = {name: np.array(archive[name]) for name in archive.files if name != "metadata"}
        np.savez_compressed(legacy_path, metadata=np.array(json.dumps(metadata)), **arrays)
        restored = Policy.load(OrdinaryGraph(), legacy_path)
        assert restored.graph.input_specs["information_set"] == jax.ShapeDtypeStruct((1,), jnp.float32)
        observations, mask, _ = input_batch(env)
        np.testing.assert_array_equal(restored(observations, mask), policy(observations, mask))
        for old, new in zip(jax.tree.leaves(policy.params), jax.tree.leaves(restored.params)):
            np.testing.assert_array_equal(old, new)


def test_environment_features_can_share_names_with_graph_attributes_and_models():
    class SharedNamesGraph(FeatureGraph):
        def __init__(self):
            super().__init__()
            self.environment_setting = 42
            self.metrics["environment_setting"] = self.env.environment_setting
            self.metrics["critic_feature"] = self.env.critic
            self.metrics["evaluate_feature"] = self.env.evaluate
            self.metrics["models_feature"] = self.env.models

    class SharedNamesGame(FeatureGame):
        def features(self, state):
            return dict(super().features(state), environment_setting=jnp.full(2, 7.0),
                        critic=jnp.full(2, 11.0), evaluate=jnp.full(2, 13.0),
                        models=jnp.full(2, 17.0))

    with jax.default_device(jax.devices("cpu")[0]):
        graph = SharedNamesGraph()
        trainer = Trainer(graph, SharedNamesGame(), batch_size=2)
        metrics = train(trainer)
        assert graph.environment_setting == 42
        assert graph.critic in graph.models
        assert callable(graph.evaluate)
        for name, expected in (("environment_setting", 7.0), ("critic_feature", 11.0),
                               ("evaluate_feature", 13.0), ("models_feature", 17.0)):
            np.testing.assert_allclose(metrics[name], [expected, expected])


@pytest.mark.parametrize("operation", ["train", "evaluate", "rollout"])
def test_environment_requires_features_and_information_set_before_execution(operation):
    class ObserveOnlyGame(FeatureGame):
        features = None

        def observe(self, state):
            return jnp.full((2, 1), state["turn"], jnp.float32)

    env = ObserveOnlyGame()
    with pytest.raises((TypeError, ValueError), match="features"):
        if operation == "train":
            Trainer(FeatureGraph(), env, batch_size=2)
        elif operation == "evaluate":
            average_utility(env, uniform_policy, uniform_policy, episodes=2, batch_size=2)
        else:
            policy = lambda obs, mask: (uniform_policy(obs, mask), jnp.zeros(obs.shape[0]))
            rollout(env, (policy, policy), jax.random.split(jax.random.key(0), 2))


def test_policy_mapping_requires_standard_feature_and_one_unambiguous_source():
    with jax.default_device(jax.devices("cpu")[0]):
        trainer = Trainer(FeatureGraph(), FeatureGame(), batch_size=2)
        policy = trainer.policy(0)
        observations, mask, features = input_batch(FeatureGame())
        without_information_set = {name: value for name, value in features.items()
                                   if name != "information_set"}
        with pytest.raises((ValueError, KeyError), match="information_set"):
            policy(without_information_set, mask)
        with pytest.raises((TypeError, ValueError), match="information_set|features"):
            policy(observations, mask, features=features)
        with pytest.raises((TypeError, ValueError), match="features"):
            policy(features, mask, features=without_information_set)


def test_example_inputs_supports_standard_feature_override_and_checks_vector_width():
    values = jnp.arange(3, dtype=jnp.int32)
    inputs = example_inputs(3, 2, features={"information_set": values})
    np.testing.assert_array_equal(inputs["information_set"], values)
    assert inputs["information_set"].dtype == jnp.int32
    assert "information_state" not in inputs
    for invalid in (jnp.zeros(2), jnp.zeros((1, 3)), jnp.zeros(())):
        with pytest.raises(ValueError, match="information_set"):
            example_inputs(3, 2, features={"information_set": invalid})
