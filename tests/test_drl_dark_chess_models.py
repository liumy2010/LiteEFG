"""MLP inputs, batched predictions, and shared DarkChess PPO integration."""

from pathlib import Path
import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("flax.linen")
serialization = pytest.importorskip("flax.serialization")

from LiteEFG.drl.env import DarkChess
from LiteEFG.drl.models import MLP

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from dark_chess_ppo import build_graph, save_models


@pytest.fixture(autouse=True)
def cpu_models():
    # These checks establish numerical contracts without depending on a GPU.
    with jax.default_device(jax.devices("cpu")[0]):
        yield


@pytest.mark.parametrize("history_length", [1, 2, 8])
def test_mlp_actor_and_critic_preserve_batch_dimensions(history_length):
    env = DarkChess(history_length=history_length)
    features = env.features(env.init(jax.random.key(0)))
    for feature, output_size in (("information_set", env.num_actions), ("full_info", 1)):
        model = MLP(output_size)
        inputs = features[feature][0]
        variables = model.init(jax.random.key(7), inputs)
        expected = model.apply(variables, inputs)
        result = jax.jit(model.apply)(variables, jnp.broadcast_to(inputs, (2, 3, inputs.size)))
        assert result.shape == (2, 3, output_size)
        assert np.isfinite(result).all()
        np.testing.assert_allclose(result, jnp.broadcast_to(expected, result.shape),
                                   rtol=1e-5, atol=1e-6)


def test_mlp_critic_uses_true_board_both_windows_and_metadata():
    env = DarkChess(history_length=2)
    model = MLP(1, hidden_size=32)
    inputs = env.features(env.init(jax.random.key(0)))["full_info"][0]
    variables = model.init(jax.random.key(3), inputs)
    evaluate = jax.jit(lambda x: model.apply(variables, x)[0])
    gradient = jax.jit(jax.grad(evaluate))(inputs)
    assert float(jnp.linalg.norm(gradient[:896])) > 1e-6
    assert float(jnp.linalg.norm(gradient[896:2 * 896])) > 1e-6
    assert float(jnp.linalg.norm(gradient[3 * 896:4 * 896])) > 1e-6
    assert float(jnp.linalg.norm(gradient[-73:])) > 1e-6


@pytest.mark.parametrize("kwargs", [{"output_size": 0}, {"output_size": True},
                                    {"hidden_size": 0}, {"hidden_size": 1.5}])
def test_invalid_model_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        MLP(**{"output_size": 1, **kwargs}).init(jax.random.key(0), jnp.zeros(15321))


def test_mlp_trains_one_shared_critic_and_exports_actor_only(tmp_path):
    env = DarkChess(max_plies=3, history_length=2)
    graph = build_graph(env)
    graph.bind(env, batch_size=1, seed=3)
    trainer = graph.trainer
    critic_index = graph.model_index(graph.critic)
    assert trainer.states[0][critic_index] is trainer.states[1][critic_index] is graph.critic.state
    assert graph.actor[0] is not graph.actor[1]
    assert graph.actor[0].state is not graph.actor[1].state
    before = jax.tree.map(np.asarray, graph.critic.state.params)
    report = graph.train(iterations=1, epochs=1, minibatch_size=3)
    assert report["valid_samples"] == [2, 1]
    assert np.isfinite(report["loss"]).all()
    critic = graph.critic.state
    assert trainer.states[1][critic_index] is critic
    assert int(critic.step) == 1
    assert any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(critic.params)))
    actors, critic_path = save_models(graph, tmp_path)
    assert critic_path.is_file()
    state = env.init(jax.random.key(0))
    for player, path in enumerate(actors):
        if player == 1:
            state, _, _ = env.step(state, jnp.array([env.action_from_uci("e2e4"), 0]),
                                    jax.random.key(0))
        template = graph.actor[player].state.params
        restored = serialization.from_bytes(template, path.read_bytes())
        assert set(restored["params"]) == set(template["params"])
        local = {"information_set": env.features(state)["information_set"][player]}
        # Actor inference has no full_info key and needs no critic variables.
        legal = env.legal_action_mask(state)[player]
        probabilities = trainer.policy(player)(local, legal, player=player)
        actor = MLP(env.num_actions)
        logits = actor.apply(restored, local["information_set"])
        standalone = jax.nn.softmax(jnp.where(legal, logits, -jnp.inf))
        np.testing.assert_allclose(probabilities, standalone, rtol=1e-5, atol=1e-6)
