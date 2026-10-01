"""Check the PPO graph against the scalar objective and its gradient contract."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
optax = pytest.importorskip("optax")
nn = pytest.importorskip("flax.linen")

from LiteEFG.baselines.drl.PPO import graph
import LiteEFG as leg


def networks(num_actions=3, hidden_size=8):
    return (
        leg.model(nn.Sequential([nn.Dense(hidden_size), nn.tanh, nn.Dense(num_actions)])),
        leg.model(nn.Sequential([nn.Dense(hidden_size), nn.tanh, nn.Dense(1)])),
    )


def inputs():
    return {
        "information_set": jnp.asarray(
            [[1.0, 0.0, -0.5], [0.0, 1.0, 0.5], [0.3, -0.2, 1.0],
             [-0.5, 0.2, 0.0], [1.0, 0.3, 0.2], [0.2, -0.5, 0.1]]).reshape(3, 2, 3),
        "full_info": jnp.arange(30, dtype=jnp.float32).reshape(3, 2, 5) / 10,
        "legal_action_mask": jnp.asarray(
            [[True, True, False], [False, True, True], [True, False, True],
             [True, True, True], [False, False, True], [True, False, True]]).reshape(3, 2, 3),
        "action": jnp.asarray([0, 1, 2, 1, 2, 0], dtype=jnp.int32).reshape(3, 2),
        "log_sampling_prob": jnp.zeros((3, 2)),
        "reward": jnp.asarray([[1.0, 1.0], [-1.0, -1.0], [0.0, 0.3]]),
    }


def initialized(algo, batch):
    example = {name: value[0, 0] for name, value in batch.items()}
    states = algo.init(jax.random.PRNGKey(0), example)
    return tuple(state.params for state in states)


def rollout_probabilities(algo, params, batch):
    probabilities, = algo.evaluate(params, batch, (algo.strategy,))
    return probabilities


def prepare(algo, params, batch):
    valid = jnp.ones(batch["action"].shape, dtype=bool)
    return algo.update(params, batch, upd_color=[0], valid=valid, alive=valid)


def numpy_gae(batch, values, gamma, gae_lambda):
    rewards = np.asarray(batch["reward"])
    values = np.asarray(values)
    advantages = np.zeros_like(rewards)
    next_value = np.zeros(rewards.shape[1])
    next_advantage = np.zeros(rewards.shape[1])
    for step in reversed(range(rewards.shape[0])):
        delta = rewards[step] + gamma * next_value - values[step]
        advantages[step] = delta + gamma * gae_lambda * next_advantage
        next_value, next_advantage = values[step], advantages[step]
    return advantages, advantages + values


def test_ppo_objective_matches_numpy_for_both_clipping_directions():
    algo = graph(*networks(), num_actions=3, clip_epsilon=0.2,
                 value_coef=0.7, entropy_coef=0.03, gamma=0.9,
                 gae_lambda=0.6, normalize_advantage=False)
    batch = inputs()
    params = initialized(algo, batch)
    probabilities = np.asarray(rollout_probabilities(algo, params, batch))
    selected = np.take_along_axis(
        probabilities, np.asarray(batch["action"])[..., None], axis=-1).squeeze(-1)
    ratios = np.asarray([1.5, 0.5, 0.5, 1.5, 1.0, 1.1]).reshape(3, 2)
    batch["log_sampling_prob"] = jnp.asarray(np.log(selected) - np.log(ratios))
    outputs = (algo.loss, algo.value, algo.policy_loss, algo.value_loss,
               algo.policy_entropy, algo.metrics["approx_kl"],
               algo.metrics["clip_fraction"])
    prepared = prepare(algo, params, batch)
    actual = tuple(np.asarray(value) for value in algo.evaluate(
        params, prepared, outputs, upd_color=[1]))
    advantage, target = numpy_gae(batch, actual[1], gamma=0.9, gae_lambda=0.6)
    policy_loss = -np.minimum(ratios * advantage,
                              np.clip(ratios, 0.8, 1.2) * advantage)
    value_loss = (actual[1] - target) ** 2
    safe_probabilities = np.maximum(probabilities, np.finfo(np.float32).tiny)
    policy_entropy = -(probabilities * np.log(safe_probabilities)).sum(axis=-1)
    expected_loss = policy_loss + 0.7 * value_loss - 0.03 * policy_entropy

    np.testing.assert_allclose(actual[0], expected_loss, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual[2], policy_loss, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual[3], value_loss, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual[4], policy_entropy, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual[5], ratios - 1 - np.log(ratios), atol=2e-6)
    np.testing.assert_array_equal(actual[6], [[1, 1], [1, 1], [0, 0]])


@pytest.mark.parametrize("normalize", [False, True])
def test_ppo_constructs_advantages_and_targets_from_trajectory(normalize):
    algo = graph(*networks(), num_actions=3, gamma=0.9, gae_lambda=0.6,
                 normalize_advantage=normalize)
    batch = inputs()
    params = initialized(algo, batch)
    probabilities = rollout_probabilities(algo, params, batch)
    batch["log_sampling_prob"] = jnp.log(jnp.take_along_axis(
        probabilities, batch["action"][..., None], axis=-1).squeeze(-1))
    prepared = prepare(algo, params, batch)
    policy_loss, value, value_loss = algo.evaluate(
        params, prepared, (algo.policy_loss, algo.value, algo.value_loss), upd_color=[1])
    advantage, target = numpy_gae(batch, value, gamma=0.9, gae_lambda=0.6)
    if normalize:
        advantage = (advantage - advantage.mean()) / np.sqrt(advantage.var() + 1e-8)
    np.testing.assert_allclose(policy_loss, -advantage, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(value_loss, (np.asarray(value) - target) ** 2,
                               rtol=2e-6, atol=2e-6)
    for name in ("advantage", "return_target", "valid"):
        assert name not in algo.input_specs
    assert not hasattr(algo, "valid")


def test_policy_masks_illegal_actions_and_handles_single_legal_action():
    algo = graph(*networks(), num_actions=3)
    batch = inputs()
    params = initialized(algo, batch)
    prepared = prepare(algo, params, batch)
    strategy, entropy, loss = algo.evaluate(
        params, prepared, (algo.current_strategy(), algo.policy_entropy, algo.loss),
        upd_color=[1])
    strategy = np.asarray(strategy)
    np.testing.assert_array_equal(strategy[~np.asarray(batch["legal_action_mask"])], 0)
    np.testing.assert_allclose(strategy.sum(axis=-1), 1.0, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(strategy[2, 0], [0.0, 0.0, 1.0])
    assert float(entropy[2, 0]) == pytest.approx(0.0, abs=1e-7)
    assert np.isfinite(np.asarray(entropy)).all()
    assert np.isfinite(np.asarray(loss)).all()


def test_sampling_data_and_preparation_parameters_are_frozen_for_autodiff():
    algo = graph(*networks(), num_actions=3)
    batch = inputs()
    params = initialized(algo, batch)
    probabilities = rollout_probabilities(algo, params, batch)
    batch["log_sampling_prob"] = jnp.log(jnp.take_along_axis(
        probabilities, batch["action"][..., None], axis=-1).squeeze(-1))

    def loss(log_sampling_prob, reward, preparation_params, live_params):
        local = dict(batch, log_sampling_prob=log_sampling_prob, reward=reward)
        prepared = prepare(algo, preparation_params, local)
        values, = algo.evaluate(live_params, prepared, (algo.loss,), upd_color=[1])
        return values.mean()

    gradients = jax.grad(loss, argnums=(0, 1, 2, 3))(
        batch["log_sampling_prob"], batch["reward"], params, params)
    for gradient in jax.tree_util.tree_leaves(gradients[:3]):
        np.testing.assert_array_equal(np.asarray(gradient), 0.0)
    for model_gradients in gradients[3]:
        assert sum(float(jnp.sum(leaf ** 2)) for leaf in
                   jax.tree_util.tree_leaves(model_gradients)) > 0


@pytest.mark.parametrize("normalize", [False, True])
def test_prepared_ppo_targets_stay_fixed_when_critic_parameters_change(normalize):
    algo = graph(*networks(), num_actions=3, normalize_advantage=normalize)
    batch = inputs()
    params = initialized(algo, batch)
    prepared = prepare(algo, params, batch)
    outputs = (algo.advantage, algo.return_target, algo.value, algo.value_loss)
    advantage, target, value, _ = algo.evaluate(
        params, prepared, outputs, upd_color=[1])
    changed_params = tuple(
        jax.tree_util.tree_map(lambda value: value + 0.5, model_params)
        if index == algo.model_index(algo.critic) else model_params
        for index, model_params in enumerate(params))
    changed_advantage, changed_target, changed_value, changed_loss = algo.evaluate(
        changed_params, prepared, outputs, upd_color=[1])
    np.testing.assert_array_equal(changed_advantage, advantage)
    np.testing.assert_array_equal(changed_target, target)
    assert not np.allclose(changed_value, value)
    np.testing.assert_allclose(changed_loss, (changed_value - target) ** 2,
                               rtol=2e-6, atol=2e-6)

    refreshed = prepare(algo, changed_params, prepared)
    refreshed_target, = algo.evaluate(
        changed_params, refreshed, (algo.return_target,), upd_color=[1])
    assert not np.allclose(refreshed_target, target)


def test_prepared_ppo_targets_and_normalization_survive_minibatch_shuffle():
    algo = graph(*networks(), num_actions=3)
    batch = inputs()
    params = initialized(algo, batch)
    prepared = prepare(algo, params, batch)
    expected, = algo.evaluate(params, prepared, (algo.loss,), upd_color=[1])
    flattened = {name: value.reshape((6,) + value.shape[2:])
                 for name, value in prepared.items()}
    order = np.asarray([4, 0, 5, 3, 2, 1])
    for start in range(0, 6, 2):
        selected = order[start:start + 2]
        minibatch = {name: value[selected] for name, value in flattened.items()}
        actual, = algo.evaluate(params, minibatch, (algo.loss,), upd_color=[1])
        np.testing.assert_allclose(actual, np.asarray(expected).reshape(-1)[selected],
                                   rtol=2e-6, atol=2e-6)


def test_jitted_parameter_update_has_finite_nonzero_gradients():
    algo = graph(*networks(), num_actions=3)
    batch = inputs()
    params = initialized(algo, batch)
    probabilities = rollout_probabilities(algo, params, batch)
    batch["log_sampling_prob"] = jnp.log(jnp.take_along_axis(
        probabilities, batch["action"][..., None], axis=-1).squeeze(-1))
    prepared = prepare(algo, params, batch)

    def loss(parameters):
        values, = algo.evaluate(parameters, prepared, (algo.loss,), upd_color=[1])
        return values.mean()

    old_loss, gradients = jax.jit(jax.value_and_grad(loss))(params)
    for model_gradients in gradients:
        leaves = jax.tree_util.tree_leaves(model_gradients)
        assert all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves)
        assert sum(float(jnp.sum(leaf ** 2)) for leaf in leaves) > 0
    optimizer = optax.adam(1e-4)
    updates, _ = optimizer.update(gradients, optimizer.init(params), params)
    updated = optax.apply_updates(params, updates)
    new_loss = jax.jit(loss)(updated)
    assert np.isfinite(float(new_loss))
    assert float(new_loss) < float(old_loss)


@pytest.mark.parametrize("kwargs", [
    {"num_actions": 0},
    {"clip_epsilon": 1}, {"value_coef": -1}, {"entropy_coef": -1},
    {"gamma": -0.1}, {"gamma": float("nan")}, {"gae_lambda": 1.1},
    {"normalize_advantage": "yes"},
    {"critic_feature": 3}, {"critic_feature": ""},
    {"critic_feature": "_graph"}, {"critic_feature": "reward"},
    {"critic_feature": "not a feature"}, {"critic_feature": "class"},
])
def test_invalid_ppo_configuration(kwargs):
    with pytest.raises(ValueError):
        graph(*networks(num_actions=7), **kwargs)


def test_ppo_preserves_caller_supplied_model_resources_and_optimizer():
    actor, critic = networks()
    algo = graph(actor, critic, num_actions=3)
    assert algo.actor is actor
    assert algo.critic is critic
    assert algo.critic_feature == "full_info"
    assert algo.actor.optimizer is actor.optimizer
    assert algo.critic.optimizer is critic.optimizer


@pytest.mark.parametrize("removed", ["shared_critic", "learning_rate", "max_grad_norm"])
def test_ppo_rejects_graph_owned_model_configuration(removed):
    with pytest.raises(TypeError):
        graph(*networks(), num_actions=3, **{removed: 0.5})


@pytest.mark.parametrize("argument", ["actor", "critic"])
def test_ppo_requires_external_handles_instead_of_bare_modules(argument):
    actor, critic = networks()
    if argument == "actor":
        actor = nn.Dense(3)
    else:
        critic = nn.Dense(1)
    with pytest.raises(TypeError, match="model|Model|resource"):
        graph(actor, critic, num_actions=3)


@pytest.mark.parametrize("container", [list, tuple, leg.ModelList])
def test_ppo_routes_external_player_models_without_copying_resources(container):
    first = leg.model(nn.Dense(3))
    second = leg.model(nn.Sequential([nn.Dense(5), nn.tanh, nn.Dense(3)]))
    critic = leg.model(nn.Dense(1))
    algo = graph(container([first, second]), critic, num_actions=3)
    assert isinstance(algo.actor, leg.ModelList)
    assert algo.actor[0] is first and algo.actor[1] is second
    assert algo.critic is critic
    batch = dict(inputs(), player=jnp.array([[0, 1], [1, 0], [0, 1]], jnp.int32))
    params = initialized(algo, batch)
    strategy, = jax.jit(lambda values: algo.evaluate(values, batch, (algo.strategy,)))(params)
    expected = []
    for observations, masks, players in zip(np.asarray(batch["information_set"]),
                                             np.asarray(batch["legal_action_mask"]),
                                             np.asarray(batch["player"])):
        row = []
        for observation, mask, player in zip(observations, masks, players):
            resource = (first, second)[player]
            logits = resource.apply(params[algo.model_index(resource)], jnp.asarray(observation))
            row.append(jax.nn.softmax(jnp.where(mask, logits, -jnp.inf)))
        expected.append(jnp.stack(row))
    np.testing.assert_allclose(strategy, jnp.stack(expected), rtol=1e-6, atol=1e-6)
    assert len(algo.models) == 3


def test_ppo_single_handle_is_shared_and_repeated_list_entries_do_not_clone():
    actor, critic = networks()
    algo = graph([actor, actor], critic, num_actions=3).bind(
        leg.Goofspiel(num_cards=3), batch_size=1, seed=17,
    )
    assert algo.actor[0] is algo.actor[1] is actor
    assert len(algo.models) == 2
    for resource in (actor, critic):
        index = algo.model_index(resource)
        assert algo.trainer.states[0][index] is algo.trainer.states[1][index] is resource.state
        assert algo.trainer.model_state(resource) is resource.state


def test_privileged_critic_features_do_not_change_actor_or_policy_inputs():
    actor, _ = networks()
    critic = leg.model(nn.Dense(1, use_bias=False, kernel_init=nn.initializers.ones))
    algo = graph(actor, critic, num_actions=3)
    batch = inputs()
    full_info = jnp.arange(30, dtype=jnp.float32).reshape(3, 2, 5) / 10
    batch["full_info"] = full_info
    params = initialized(algo, batch)
    strategy, value = algo.evaluate(params, batch, (algo.strategy, algo.value))
    changed_strategy, changed_value = algo.evaluate(
        params, dict(batch, full_info=full_info + 1), (algo.strategy, algo.value))
    np.testing.assert_array_equal(changed_strategy, strategy)
    np.testing.assert_allclose(value, full_info.sum(-1), atol=1e-6)
    np.testing.assert_allclose(changed_value, value + 5, atol=1e-6)

    policy = leg.Policy(algo, params, observation_size=3, num_actions=3)
    actor_only = policy(batch["information_set"], batch["legal_action_mask"])
    np.testing.assert_array_equal(actor_only, strategy)
    assert algo.input_specs["full_info"].shape == (5,)


def test_ppo_trains_critic_on_custom_environment_feature():
    class FeatureGoofspiel(leg.Goofspiel):
        def features(self, state):
            features = super().features(state)
            full_info = jnp.concatenate((
                state.scores, state.hands.astype(jnp.float32).reshape(-1)))
            features["full_info"] = jnp.broadcast_to(
                full_info, (self.num_players, full_info.size))
            return features

    env = FeatureGoofspiel(num_cards=3)
    algo = graph(*networks(), num_actions=3, critic_feature="full_info")
    algo.bind(env, batch_size=16, seed=0)
    trainer = algo.trainer
    before = jax.tree_util.tree_leaves(algo.critic.state.params)
    for _ in range(2):
        trajectories = trainer.collect()
        prepared = trainer.update(trajectories, upd_color=[0])
        size = prepared[0]["_valid"].size
        prepared = jax.tree.map(lambda x: x.reshape((size,) + x.shape[2:]), prepared)
        for start in range(0, size, 16):
            minibatch = jax.tree.map(lambda x: x[start:start + 16], prepared)
            metrics = trainer.optimize(minibatch, upd_color=[1])
    after = jax.tree_util.tree_leaves(algo.critic.state.params)
    assert any(not np.array_equal(old, new) for old, new in zip(before, after))
    assert all(np.isfinite(np.asarray(value)).all() for value in after)
    assert all(np.isfinite(np.asarray(value)).all() for value in metrics.values())
    assert algo.input_specs["full_info"].shape == (8,)


def test_ppo_train_matches_explicit_schedule_and_weighted_report():
    def make_trainer():
        algo = graph(*networks(), num_actions=3)
        algo.metrics["target"] = algo.return_target
        return algo.bind(leg.Goofspiel(num_cards=3), batch_size=3, seed=7).trainer

    helper = make_trainer()
    manual = make_trainer()
    assert helper.graph is not manual.graph
    original_manual_states, original_manual_key = manual.states, manual.key
    actual = helper.graph.train(iterations=2, epochs=2, minibatch_size=2,
                                compiled_updates=False)
    for actual_state, expected_state in zip(manual.states[0], original_manual_states[0]):
        assert actual_state is expected_state
    np.testing.assert_array_equal(manual.key, original_manual_key)
    assert manual.iteration == 0
    reports, utilities = [], []
    for _ in range(2):
        data = manual.collect()
        data = manual.update(data, upd_color=[0])
        manual.key, key = jax.random.split(manual.key)
        dataloader = leg.dataloader(data, batch_size=2, shuffle=True, generator=key)
        for _ in range(2):
            for batch in dataloader:
                reports.append(manual.optimize(batch, upd_color=[1]))
        utilities.append(reports[-1]["mean_utility"])

    for actual_state, expected_state in zip(jax.tree_util.tree_leaves(helper.states),
                                            jax.tree_util.tree_leaves(manual.states)):
        np.testing.assert_array_equal(actual_state, expected_state)
    np.testing.assert_array_equal(helper.key, manual.key)
    assert helper.iteration == manual.iteration == actual["iteration"] == 2
    assert actual["trajectories"] == 6
    assert actual.keys() == reports[-1].keys()
    assert "target" not in actual

    weights = np.asarray([report["valid_samples"] for report in reports])
    assert len(np.unique(weights)) > 1
    np.testing.assert_array_equal(actual["valid_samples"], weights.sum(axis=0))
    metric_names = (name for name in helper.graph.metrics if name in reports[-1])
    for name in ("loss", *metric_names):
        values = np.asarray([report[name] for report in reports])
        expected = (values * weights).sum(axis=0) / weights.sum(axis=0)
        np.testing.assert_allclose(actual[name], expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual["mean_utility"], np.mean(utilities, axis=0),
                               rtol=0, atol=0)


def test_ppo_epochs_keep_every_sample_cache_and_mask_aligned(monkeypatch):
    algo = graph(*networks(), num_actions=3)
    algo.bind(leg.Goofspiel(num_cards=3), batch_size=2, seed=7)
    trainer = algo.trainer
    update, optimize = trainer.update, trainer.optimize
    prepared, batches = [], []

    def record_update(data, *, upd_color):
        result = update(data, upd_color=upd_color)
        slots = jnp.arange(6).reshape(3, 2)
        # Include an excluded record to check that every field, including the
        # validity mask, follows the same sample permutation and slice.
        result = tuple(dict(records, slot=slots, _valid=slots != player)
                       for player, records in enumerate(result))
        prepared.append(jax.device_get(result))
        return result

    def record_optimize(batch, *, upd_color, **kwargs):
        batches.append(jax.device_get(batch))
        return optimize(batch, upd_color=upd_color, **kwargs)

    monkeypatch.setattr(trainer, "update", record_update)
    monkeypatch.setattr(trainer, "optimize", record_optimize)
    report = algo.train(epochs=2, minibatch_size=1, compiled_updates=False)
    assert len(prepared) == 1
    assert len(batches) == 4
    np.testing.assert_array_equal(report["valid_samples"], [10, 10])
    for epoch in range(2):
        epoch_batches = batches[epoch * 2:epoch * 2 + 2]
        assert [batch[0]["_valid"].size for batch in epoch_batches] == [3, 3]
        for batch in epoch_batches:
            # Players share an episode group; its full time axis stays together.
            np.testing.assert_array_equal(batch[0]["slot"], batch[1]["slot"])
            episode = int(batch[0]["slot"][0])
            np.testing.assert_array_equal(batch[0]["slot"], np.arange(3) * 2 + episode)
        for player in range(2):
            identifiers = np.concatenate([batch[player]["slot"] for batch in epoch_batches])
            np.testing.assert_array_equal(np.sort(identifiers), np.arange(6))
            for name, values in prepared[0][player].items():
                expected = values.reshape((6,) + values.shape[2:])[identifiers]
                actual = np.concatenate([batch[player][name] for batch in epoch_batches])
                np.testing.assert_array_equal(actual, expected)
    assert any(name.startswith("_drl_node_") for name in prepared[0][0])


def test_ppo_train_rejects_invalid_schedule_before_collection(monkeypatch):
    algo = graph(*networks(), num_actions=3)
    algo.bind(leg.Goofspiel(num_cards=3), batch_size=3, seed=7)
    trainer = algo.trainer
    original_states, original_key = trainer.states, trainer.key

    def unexpected_collection():
        pytest.fail("Invalid schedules must be rejected before collecting trajectories")

    monkeypatch.setattr(trainer, "collect", unexpected_collection)
    for name in ("iterations", "epochs", "minibatch_size"):
        for value in (0, -1, True, 1.5, None):
            with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
                algo.train(**{name: value})
    for value in (0, 1, None, "true"):
        with pytest.raises(ValueError, match="compiled_updates must be a boolean"):
            algo.train(compiled_updates=value)
    for actual, expected in zip(trainer.states[0], original_states[0]):
        assert actual is expected
    np.testing.assert_array_equal(trainer.key, original_key)
    assert trainer.iteration == 0


def test_ppo_train_requires_a_bound_trainer():
    algo = graph(*networks(), num_actions=3)
    with pytest.raises(RuntimeError, match="bind"):
        algo.train()
