"""Execute the neural graph guide's actual training and policy examples."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("optax")
nn = pytest.importorskip("flax.linen")

import LiteEFG as leg
from docs_examples import extract_fenced_blocks


QUICK_START = "docs/guide/deep-learning.md"
OBJECTIVES = "docs/guide/deep-learning/objectives.md"
POLICIES = "docs/guide/deep-learning/policies.md"
ENVIRONMENTS = "docs/guide/deep-learning/environments.md"


def run_example(page, namespace):
    block, = extract_fenced_blocks(page)
    exec(compile(block, f"{page}:python[1]", "exec"), namespace)


def test_documented_ppo_training_two_policy_match_and_reload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    namespace = {}
    run_example(QUICK_START, namespace)

    trainer, env = namespace["trainer"], namespace["env"]
    assert trainer.iteration == namespace["metrics"]["iteration"] == 2
    assert namespace["metrics"]["trajectories"] == 32
    assert env.num_cards == 7
    assert env.imp_info is False
    assert env.points_order == "random"
    assert isinstance(namespace["policy0"], leg.Policy)
    assert isinstance(namespace["policy1"], leg.Policy)

    match = namespace["match"]
    assert match["episodes"] == 64
    means = np.asarray(match["mean_utility"])
    errors = np.asarray(match["standard_error"])
    intervals = np.asarray(match["confidence_interval_95"])
    assert means.shape == errors.shape == (2,)
    assert intervals.shape == (2, 2)
    assert np.isfinite(means).all() and np.isfinite(intervals).all()
    assert (errors >= 0).all()
    assert means.sum() == pytest.approx(0, abs=1e-7)
    assert (np.abs(means) <= 14).all()
    np.testing.assert_allclose(intervals[:, 0], means - 1.96 * errors)
    np.testing.assert_allclose(intervals[:, 1], means + 1.96 * errors)
    assert sum(match["win_rate"]) + match["tie_rate"] == pytest.approx(1.0)

    run_example(POLICIES, namespace)
    assert namespace["loaded_match"] == match
    for player in range(2):
        assert (tmp_path / f"goofspiel-player{player}.npz").is_file()
        original = namespace[f"policy{player}"]
        loaded = namespace[f"loaded{player}"]
        for expected, actual in zip(jax.tree_util.tree_leaves(original.params),
                                    jax.tree_util.tree_leaves(loaded.params)):
            np.testing.assert_array_equal(expected, actual)


def test_documented_custom_objective_is_trainable():
    namespace = {}
    run_example(OBJECTIVES, namespace)
    graph = namespace["algorithm"]
    assert isinstance(graph, leg.DRLGraph)
    graph.bind(leg.Goofspiel(num_cards=7), batch_size=4, seed=0)
    trainer = graph.trainer
    before = tuple(jax.tree_util.tree_leaves(graph.actor[0].state.params))
    data = trainer.update(trainer.collect(), upd_color=[0])
    for batch in leg.dataloader(data, batch_size=14):
        metrics = trainer.optimize(batch, upd_color=[1])
    after = jax.tree_util.tree_leaves(graph.actor[0].state.params)
    assert np.isfinite(metrics["loss"]).all()
    assert any(not np.array_equal(a, b) for a, b in zip(before, after))


def test_documented_environment_feature_critic_and_policy_reload(tmp_path):
    namespace = {}
    run_example(ENVIRONMENTS, namespace)
    trainer = namespace["feature_trainer"]
    env = namespace["feature_env"]
    assert namespace["feature_metrics"]["iteration"] == 1
    assert np.isfinite(namespace["feature_metrics"]["loss"]).all()
    assert namespace["feature_match"]["episodes"] == 32
    assert env.imp_info is True

    state = env.init(jax.random.PRNGKey(0))
    masks = env.legal_action_mask(state)
    features = env.features(state)
    assert features["full_info_feature"].shape == (2, 10)
    # The external critic consumes the environment's ten-value feature, while
    # action inference needs only the information_set feature and legal actions.
    critic = trainer.graph.critic[0].state
    assert critic.params["params"]["kernel"].shape == (10, 1)
    policy = trainer.policy(0)
    policy_features = {"information_set": features["information_set"]}
    probabilities = policy(policy_features, masks)
    np.testing.assert_allclose(np.asarray(probabilities).sum(-1), 1.0, atol=1e-6)

    path = tmp_path / "feature-policy.npz"
    policy.save(path)
    graph = namespace["FeatureActorCritic"](
        [leg.model(nn.Dense(4)) for _ in range(2)],
        [leg.model(nn.Dense(1)) for _ in range(2)],
    )
    loaded = leg.Policy.load(graph, path)
    np.testing.assert_array_equal(loaded(policy_features, masks), probabilities)
