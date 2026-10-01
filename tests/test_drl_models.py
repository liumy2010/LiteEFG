"""External Flax MLPs integrate through module and callable wrappers."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
optax = pytest.importorskip("optax")
nn = pytest.importorskip("flax.linen")

from LiteEFG.drl.env import Goofspiel
from LiteEFG.drl.graph import DRLGraph, masked_softmax, model
from LiteEFG.drl.runtime import Policy, example_inputs


def external_mlp(output_size):
    return nn.Sequential([nn.Dense(8), nn.tanh, nn.Dense(output_size)])


@pytest.fixture(autouse=True)
def use_cpu_for_external_mlp_checks():
    # These tests validate the wrapper contract, not GPU throughput.
    with jax.default_device(jax.devices("cpu")[0]):
        yield


def _wrapped_model(network, mode):
    optimizer = optax.sgd(1e-3)
    if mode == "module":
        return model(network, optimizer=optimizer)
    return model(
        init=lambda key, inputs: network.init(key, inputs)["params"],
        apply=lambda parameters, inputs: network.apply({"params": parameters}, inputs),
        optimizer=optimizer,
    )


class RegressionPolicy(DRLGraph):
    def __init__(self, network, mode):
        super().__init__()
        self.network = _wrapped_model(network, mode)
        self.prediction = self.network(self.env.information_set)
        self.strategy = masked_softmax(self.prediction, self.legal_action_mask)
        self.value = self.prediction[0]
        self.loss = ((self.prediction - self.reward) ** 2).mean()
        self.minimize(self.loss, models=[self.network])


@pytest.mark.parametrize("mode", ["module", "callables"])
def test_external_mlp_initialization_gradients_update_and_policy_checkpoint(mode, tmp_path):
    env = Goofspiel()
    graph = RegressionPolicy(external_mlp(env.num_actions), mode)
    observation_size = env.features(env.init(jax.random.key(0)))["information_set"].shape[-1]
    states = graph.init(jax.random.key(0),
                        example_inputs(observation_size, env.num_actions))
    parameters = tuple(state.params for state in states)
    environment_states = jax.vmap(env.init)(jax.random.split(jax.random.key(0), 4))
    observations = jax.vmap(env.features)(environment_states)["information_set"][:, 0]
    legal = jax.vmap(env.legal_action_mask)(environment_states)[:, 0]
    inputs = {
        "information_set": observations,
        "legal_action_mask": legal,
        "reward": jnp.array([-0.5, 0.25, 0.75, -0.25]),
    }

    def objective(params):
        loss, = graph.evaluate(params, inputs, (graph.loss,))
        return loss.mean()

    before, gradients = jax.jit(jax.value_and_grad(objective))(parameters)
    leaves = jax.tree.leaves(gradients)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves)
    assert sum(float(jnp.sum(leaf ** 2)) for leaf in leaves) > 0
    updates, _ = graph.network.optimizer.update(
        gradients[0], states[0].opt_state, parameters[0])
    updated = (optax.apply_updates(parameters[0], updates),)
    after = jax.jit(objective)(updated)
    assert np.isfinite(float(after))
    assert float(after) < float(before)
    assert all(leaf.device.platform == "cpu" for leaf in jax.tree.leaves(updated))

    policy = Policy(graph, updated, observation_size, env.num_actions)
    probabilities = jax.jit(policy)(observations, legal)
    assert probabilities.shape == (4, env.num_actions)
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0, atol=1e-6)
    assert np.isfinite(np.asarray(probabilities)).all()

    checkpoint = tmp_path / "external-policy.npz"
    policy.save(checkpoint)
    restored_graph = RegressionPolicy(external_mlp(env.num_actions), mode)
    restored = Policy.load(restored_graph, checkpoint)
    np.testing.assert_allclose(jax.jit(restored)(observations, legal), probabilities,
                               rtol=1e-6, atol=1e-6)
    for actual, expected in zip(jax.tree.leaves(restored.params), jax.tree.leaves(updated)):
        np.testing.assert_array_equal(actual, expected)


def test_same_resource_trains_across_graphs_without_reinitialization_or_stale_state():
    initialized = []

    def initialize(key, inputs):
        del key
        initialized.append(inputs.shape)
        return jnp.zeros_like(inputs)

    optimizer = optax.adam(0.1)
    resource = model(init=initialize, apply=jnp.dot, optimizer=optimizer)

    class ResourceRegression(DRLGraph):
        def __init__(self, *, prepend=False):
            super().__init__()
            if prepend:
                self.untrained = model(nn.Dense(1))
                self.untrained(self.env.information_set)
            self.value = resource(self.env.information_set)
            self.strategy = self.legal_action_mask / self.action_set_size
            self.minimize((self.value - self.reward) ** 2, models=[resource])

    env = Goofspiel(num_cards=2)
    width = env.features(env.init(jax.random.key(0)))["information_set"].shape[-1]
    first = ResourceRegression().bind(env, batch_size=1, seed=0)
    first_state = resource.state
    second = ResourceRegression(prepend=True).bind(env, batch_size=1, seed=999)
    assert initialized == [(width,)]
    assert resource.state is first_state
    assert first.model_index(resource) != second.model_index(resource)
    assert first.trainer.model_state(resource) is second.trainer.model_state(resource) is first_state
    unused_state = second.untrained.state
    # Policy snapshots are explicit arrays and must remain frozen while another
    # graph updates the resource from which those arrays originated.
    frozen = first.trainer.policy(0)
    frozen_params = tuple(jax.tree.map(np.asarray, params) for params in frozen.params)

    expected = first_state
    inputs = jnp.arange(1, width + 1, dtype=jnp.float32)[None] / width
    for step, algorithm in enumerate((first, second, first), 1):
        targets = jnp.array([2.0 + step], jnp.float32)
        batch = {"information_set": inputs, "reward": targets,
                 "_valid": jnp.ones(1, jnp.bool_)}
        gradient = 2 * inputs[0] * (jnp.dot(expected.params, inputs[0]) - targets[0])
        updates, opt_state = optimizer.update(gradient, expected.opt_state, expected.params)
        expected = expected._replace(params=optax.apply_updates(expected.params, updates),
                                     opt_state=opt_state, step=expected.step + 1)
        algorithm.trainer.optimize((batch, batch), upd_color=[0])
        assert first.trainer.model_state(resource) is second.trainer.model_state(resource) is resource.state
        assert int(resource.state.step) == step
        for actual, reference in zip(jax.tree.leaves(resource.state), jax.tree.leaves(expected)):
            np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=1e-7)
    assert second.untrained.state is unused_state
    assert initialized == [(width,)]
    for actual, reference in zip(jax.tree.leaves(frozen.params), jax.tree.leaves(frozen_params)):
        np.testing.assert_array_equal(actual, reference)
