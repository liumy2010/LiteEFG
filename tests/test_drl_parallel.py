"""Stage selection and synchronized optimization on two isolated CPU devices."""

import os
from pathlib import Path
import re
import subprocess
import sys
from unittest.mock import patch

# Set affinity before importing JAX so virtual CPU tests fit shared CI hosts.
if __name__ == "__main__" and hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
optax = pytest.importorskip("optax")
nn = pytest.importorskip("flax.linen")

import LiteEFG as leg
from LiteEFG.baselines.drl.PPO import graph as PPO
from LiteEFG.drl.env import Goofspiel
from LiteEFG.drl.runtime import Policy


class _Environment:
    num_players = 2
    num_actions = 2
    max_steps = 1

    def init(self, key):
        return jnp.array(0, jnp.int32)

    def features(self, state):
        return {"information_set": jnp.ones((2, 3)), "target": jnp.zeros(2)}

    def legal_action_mask(self, state):
        return jnp.ones((2, 2), bool)

    def active_players(self, state):
        return jnp.ones(2, bool)

    def step(self, state, actions, key):
        return state + 1, jnp.zeros(2), jnp.array(True)


class _Regression(leg.DRLGraph):
    def __init__(self, optimizer=None):
        super().__init__()
        optimizer = optimizer or optax.chain(
            optax.clip_by_global_norm(0.6), optax.adamw(0.02, weight_decay=0.03))

        def model():
            return leg.model(init=lambda key, inputs: jnp.array([0.1, -0.2, 0.3]),
                             apply=lambda weights, inputs: jnp.dot(inputs, weights),
                             optimizer=optimizer)

        self.local = leg.ModelList([model(), model()])
        self.common = model()
        with leg.backward(color=0):
            self.local_value = self.local[self.player](self.env.information_set)
            self.value = self.common(self.env.information_set)
            self.minimize((self.local_value - self.env.target) ** 2,
                          models=list(self.local))
            self.minimize((self.value - self.env.target) ** 2, models=[self.common])
            self.strategy = self.legal_action_mask / self.action_set_size
            self.metrics = {"prediction": self.value}


def _ppo(sharing="shared"):
    actors = [leg.model(nn.Dense(3), optimizer=optax.adam(0.003)) for _ in range(2)]
    critic = leg.model(nn.Dense(1), optimizer=optax.adam(0.003))
    algo = PPO(actors[0] if sharing == "shared" else actors, critic, num_actions=3)
    with leg.backward(color=1):
        algo.metrics["frozen_advantage"] = algo.advantage + 0.0
        algo.metrics["frozen_target"] = algo.return_target + 0.0
    return algo


def _same_tree(actual, expected):
    left, left_structure = jax.tree_util.tree_flatten(actual)
    right, right_structure = jax.tree_util.tree_flatten(expected)
    assert left_structure == right_structure
    for value, reference in zip(left, right):
        if np.issubdtype(np.asarray(value).dtype, np.inexact):
            np.testing.assert_allclose(value, reference, rtol=7e-5, atol=5e-6)
        else:
            np.testing.assert_array_equal(value, reference)


def _replicated(tree, devices):
    for leaf in jax.tree_util.tree_leaves(tree):
        assert isinstance(leaf, jax.Array)
        assert leaf.devices() == set(devices)
        assert leaf.sharding.is_fully_replicated
        assert len(leaf.addressable_shards) == len(devices)
        for shard in leaf.addressable_shards:
            # Replication must not add a leading device axis to public states.
            assert shard.data.shape == leaf.shape
            np.testing.assert_array_equal(shard.data, leaf)


def _with_collective_check(trainer, stage, action, *, gradient_size=None):
    """Inspect the executable actually invoked by a previously warmed API call."""
    keys = [key for key in trainer._compiled if isinstance(key, tuple) and key[0] == stage]
    assert len(keys) == 1
    compiled_function = trainer._compiled[keys[0]]
    arguments = []

    def capture(*args):
        arguments.append(args)
        return compiled_function(*args)

    with patch.dict(trainer._compiled, {keys[0]: capture}):
        result = action()
    assert len(arguments) == 1
    executable = compiled_function.lower(*arguments[0]).compile()
    assert len(executable.runtime_executable().local_devices()) == 2
    # Numeric equivalence alone could pass if every device computed all rows.
    # An actual partitioned reduction must communicate the sample gradients.
    hlo = executable.as_text().lower()
    assert "all-reduce" in hlo
    if gradient_size is not None:
        # The regression has three weights, two scalar metrics, and integer
        # sample counts. A float vector reduction therefore checks gradients,
        # rather than accepting communication of counts or scalar reports.
        reductions = [line.split(" all-reduce", 1)[0] for line in hlo.splitlines()
                      if re.search(r"\ball-reduce(?:-start)?\(", line)]
        assert any(re.search(rf"\bf32\[{gradient_size}\](?:\{{[^}}]*\}})?", result)
                   for result in reductions)
    return result


def _records(masks, *, poison=True):
    slots = jnp.arange(len(masks[0]), dtype=jnp.float32)
    result = []
    for player, valid in enumerate(masks):
        valid = jnp.asarray(valid)
        inputs = jnp.stack((1 + slots / 5, jnp.sin(slots + player),
                            jnp.cos(slots / 3 + player)), axis=-1)
        targets = slots / 7 + 0.3 * player
        if poison:
            inputs = jnp.where(valid[:, None], inputs, jnp.nan)
            targets = jnp.where(valid, targets, jnp.inf)
        result.append({"information_set": inputs, "target": targets, "_valid": valid})
    return tuple(result)


def _check_modes():
    devices = tuple(jax.local_devices())
    assert len(devices) == 2
    env = _Environment()
    default = _Regression().bind(env, batch_size=3).trainer
    assert default.devices == (devices[0],)
    assert set(default.parallel) == {"sampling", "optimizing"}
    assert default.sampling_devices == default.optimizing_devices == (devices[0],)

    reference = None
    for stages in ((), ("sampling",), ("optimizing",), ("sampling", "optimizing")):
        trainer = _Regression().bind(env, batch_size=4, devices=devices,
                                     parallel=stages).trainer
        assert trainer.devices == devices
        assert set(trainer.parallel) == set(stages)
        assert trainer.sampling_devices == (devices if "sampling" in stages else devices[:1])
        assert trainer.optimizing_devices == (devices if "optimizing" in stages else devices[:1])
        data = trainer.collect()
        prepared = trainer.update(data, upd_color=[0])
        assert all(value.shape[:2] == (1, 4) for row in prepared for value in row.values())
        batch = tuple({name: value.reshape((-1,) + value.shape[2:])
                       for name, value in row.items()} for row in prepared)
        assert trainer.optimize(batch, upd_color=[0])["valid_samples"] == [4, 4]
        for leaf in jax.tree_util.tree_leaves(trainer.states):
            assert leaf.devices() == set(trainer.optimizing_devices)

        # Exercise the public schedule and loader directly on collection's own
        # array placements, including a three-trajectory group and short tail.
        # Manually constructing minibatches could hide incompatible shardings.
        algo = _ppo().bind(Goofspiel(num_cards=3), batch_size=4, seed=5,
                           devices=devices, parallel=stages)
        report = algo.train(epochs=1, minibatch_size=3, compiled_updates=False)
        assert report["valid_samples"] == [12, 12]
        assert all(int(resource.state.step) == 2 for resource in algo.models)
        if reference is None:
            reference = report, algo.trainer.states
        else:
            _same_tree(report, reference[0])
            _same_tree(algo.trainer.states, reference[1])

    both = _Regression().bind(env, batch_size=4, devices=devices).trainer
    assert both.sampling_devices == both.optimizing_devices == devices
    legacy = _Regression().bind(env, batch_size=4, sampling_devices=devices[::-1]).trainer
    assert legacy.parallel == ("sampling",)
    assert legacy.sampling_devices == devices[::-1]
    assert legacy.optimizing_devices == devices[:1]
    # Optimization pads rows; only multi-device sampling requires equal chunks.
    optimizing = _Regression().bind(env, batch_size=3, devices=devices,
                                    parallel=("optimizing",)).trainer
    assert optimizing.batch_size == 3

    invalid = [
        {"devices": []}, {"devices": [devices[0], devices[0]]},
        {"devices": [object()]}, {"devices": devices, "batch_size": 3},
        {"parallel": ("gradient",)}, {"parallel": ("sampling", "sampling")},
        {"parallel": True},
        {"devices": devices, "sampling_devices": devices},
        {"parallel": (), "sampling_devices": devices},
    ]
    for options in invalid:
        algo = _Regression()
        with pytest.raises((TypeError, ValueError)):
            algo.bind(env, **dict({"batch_size": 4}, **options))
        assert algo._trainer is None
        algo.bind(env, batch_size=4, devices=devices)
    algo = _Regression()
    with patch.object(jax, "local_devices", return_value=list(devices[:1])):
        with pytest.raises(ValueError):
            algo.bind(env, batch_size=4, devices=devices[1:])
    assert algo._trainer is None


def _check_weighted_gradients():
    devices = tuple(jax.local_devices())
    for compact in (False, True):
        trainer = _Regression().bind(_Environment(), batch_size=3, devices=devices,
                                     parallel=("optimizing",),
                                     compact_batches=compact).trainer
        expected = [resource.state for resource in trainer.graph.models]
        width = 7
        slots = np.arange(width)
        # The final layout includes unequal valid counts, an empty device shard,
        # an entirely inactive seat, and finally an empty update after Adam steps.
        cases = ((slots != 1, slots == 5), (slots < 2, slots == 0),
                 (slots == 0, slots < 0), (slots < 0, slots < 0))
        for index, masks in enumerate(cases):
            batch = _records(masks)
            before = tuple(resource.state for resource in trainer.graph.models)
            for model_index, resource in enumerate(trainer.graph.models):
                players = ([0, 1] if resource is trainer.graph.common else
                           [player for player in range(2) if resource is trainer.graph.local[player]])
                inputs = np.concatenate([np.asarray(batch[player]["information_set"])[masks[player]]
                                         for player in players])
                targets = np.concatenate([np.asarray(batch[player]["target"])[masks[player]]
                                          for player in players])
                if len(targets):
                    state = expected[model_index]
                    gradient = 2 * inputs.T @ (inputs @ np.asarray(state.params) - targets) / len(targets)
                    updates, optimizer_state = resource.optimizer.update(
                        jnp.asarray(gradient), state.opt_state, state.params)
                    expected[model_index] = state._replace(
                        params=optax.apply_updates(state.params, updates),
                        opt_state=optimizer_state, step=state.step + 1)
            action = lambda: trainer.optimize(batch, upd_color=[0])
            # With compaction disabled all cases have the same cached capacity.
            report = (_with_collective_check(trainer, "optimize", action, gradient_size=3)
                      if not compact and index == 1 else action())
            assert report["valid_samples"] == [int(mask.sum()) for mask in masks]
            _same_tree(tuple(resource.state for resource in trainer.graph.models), tuple(expected))
            if index == 2:
                assert trainer.graph.local[1].state is before[trainer.graph.model_index(trainer.graph.local[1])]
            if not any(mask.any() for mask in masks):
                assert all(resource.state is state for resource, state in zip(trainer.graph.models, before))
                assert report["loss"] == [0.0, 0.0]
            _replicated(trainer.states, devices)
            assert all(resource.state.params.shape == (3,) for resource in trainer.graph.models)


def _manual_epochs(trainer, data, options):
    loader = leg.dataloader(data, batch_size=options["minibatch_size"], shuffle=True,
                            generator=options["generator"])
    reports = [trainer.optimize(batch, upd_color=[1])
               for _ in range(options["epochs"]) for batch in loader]
    weights = np.asarray([report["valid_samples"] for report in reports])
    result = dict(reports[-1], valid_samples=weights.sum(0).tolist())
    for name in ("loss", *trainer.graph.metrics):
        values = np.asarray([report[name] for report in reports])
        result[name] = ((values * weights).sum(0) / np.maximum(weights.sum(0), 1)).tolist()
    return result


def _check_compiled_epochs():
    devices = tuple(jax.local_devices())
    for compact, sharing in ((False, "shared"), (True, "partial")):
        reference = _ppo(sharing).bind(Goofspiel(num_cards=3), batch_size=6, seed=17,
                                       compact_batches=False).trainer
        manual = _ppo(sharing).bind(Goofspiel(num_cards=3), batch_size=6, seed=17,
                                    devices=devices, compact_batches=compact).trainer
        compiled = _ppo(sharing).bind(Goofspiel(num_cards=3), batch_size=6, seed=17,
                                      devices=devices, compact_batches=compact).trainer
        data = []
        for trainer in (reference, manual, compiled):
            prepared = trainer.update(trainer.collect(), upd_color=[0])
            assert all(row["_valid"].shape == (3, 6) for row in prepared)
            # Keep five complete trajectories to force a one-trajectory tail.
            prepared = tuple({name: value[:, :5] for name, value in row.items()}
                             for row in prepared)
            slots = jnp.arange(15).reshape(3, 5)
            masks = (slots % 3 != 0, slots == 4)
            data.append(tuple(dict(row, _valid=mask) for row, mask in zip(prepared, masks)))
        _same_tree(data[0], data[1])
        _same_tree(data[0], data[2])
        frozen = compiled.policy(0)
        observation = data[2][0]["information_set"][0, 0]
        legal = data[2][0]["legal_action_mask"][0, 0]
        original_probability = frozen(observation, legal)
        frozen_data = jax.tree_util.tree_map(lambda value: np.asarray(value).copy(), data[2])
        options = dict(epochs=2, minibatch_size=2, generator=jax.random.PRNGKey(31))
        for repeat in range(2):
            expected = _manual_epochs(reference, data[0], options)
            actual_manual = _manual_epochs(manual, data[1], options)
            action = lambda: compiled.optimize_epochs(data[2], upd_color=[1], **options)
            actual_compiled = (_with_collective_check(compiled, "optimize_epochs", action)
                               if repeat else action())
            _same_tree(actual_manual, expected)
            _same_tree(actual_compiled, expected)
            _same_tree(manual.states, reference.states)
            _same_tree(compiled.states, reference.states)
            assert actual_compiled["valid_samples"] == [20, 2]
            assert int(compiled.graph.critic.state.step) == 6 * (repeat + 1)
            _replicated(compiled.states, devices)
            _replicated(manual.states, devices)
        _same_tree(data[2], frozen_data)
        np.testing.assert_array_equal(frozen(observation, legal), original_probability)
        assert not np.allclose(compiled.policy(0)(observation, legal), original_probability)
        # Policy serialization must remain independent of replica placement.
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.npz"
            frozen.save(path)
            restored = Policy.load(_ppo(sharing), path)
            np.testing.assert_allclose(restored(observation, legal), original_probability,
                                       rtol=2e-5, atol=2e-6)
        refreshed = compiled.collect()
        for player, row in enumerate(refreshed):
            probability = compiled.policy(player)(row["information_set"], row["legal_action_mask"])
            selected = jnp.take_along_axis(probability, row["action"][..., None], axis=-1)[..., 0]
            np.testing.assert_allclose(row["log_sampling_prob"], jnp.log(selected),
                                       rtol=2e-5, atol=2e-6)


def _check_failed_updates():
    devices = tuple(jax.local_devices())
    trainer = _Regression().bind(_Environment(), batch_size=4, devices=devices).trainer
    batch = _records((np.ones(7, bool), np.ones(7, bool)))
    bad = (batch[0], dict(batch[1], target=batch[1]["target"].at[-1].set(jnp.inf)))
    before = tuple(resource.state for resource in trainer.graph.models)
    for _ in range(2):
        with pytest.raises(FloatingPointError):
            trainer.optimize(bad, upd_color=[0])
        assert all(resource.state is state for resource, state in zip(trainer.graph.models, before))

    # A concurrent caller-owned model replacement must also reject the complete
    # candidate transaction, even when that candidate is numerically finite.
    key = next(key for key in trainer._compiled if isinstance(key, tuple) and key[0] == "optimize")
    execute = trainer._compiled[key]
    replacement = before[0]._replace(step=before[0].step + 10)

    def replace_during_execution(*args):
        result = execute(*args)
        trainer.graph.models[0].state = replacement
        return result

    with patch.dict(trainer._compiled, {key: replace_during_execution}):
        with pytest.raises(RuntimeError, match="stale"):
            trainer.optimize(batch, upd_color=[0])
    assert trainer.graph.models[0].state is replacement
    assert all(resource.state is state
               for resource, state in zip(trainer.graph.models[1:], before[1:]))

    def unstable_update(gradient, count, params=None):
        factor = jnp.where(count == 0, -0.001, jnp.nan)
        return jax.tree_util.tree_map(lambda value: value * factor, gradient), count + 1

    optimizer = optax.GradientTransformation(lambda params: jnp.array(0, jnp.int32), unstable_update)
    trainers = [_Regression(optimizer).bind(_Environment(), batch_size=3,
                                            devices=selected,
                                            parallel=("optimizing",)).trainer
                for selected in (devices[:1], devices)]
    trajectories = tuple({name: value[None] for name, value in row.items()} for row in batch)
    options = dict(epochs=2, minibatch_size=3, generator=jax.random.PRNGKey(13), upd_color=[0])
    for current in trainers:
        with pytest.raises(FloatingPointError):
            current.optimize_epochs(trajectories, **options)
        assert all(int(resource.state.step) == 1 for resource in current.graph.models)
        assert all(np.isfinite(np.asarray(value)).all()
                   for value in jax.tree_util.tree_leaves(current.states))
    _same_tree(trainers[0].states, trainers[1].states)
    _replicated(trainers[1].states, devices)


def _check_shared_resources():
    devices = tuple(jax.local_devices())
    initialized = []

    def initialize(key, inputs):
        initialized.append(True)
        return jnp.array([0.1, -0.2, 0.3])

    shared = leg.model(init=initialize, apply=lambda weights, inputs: jnp.dot(inputs, weights),
                       optimizer=optax.sgd(0.1))

    class SharedGraph(leg.DRLGraph):
        def __init__(self):
            super().__init__()
            with leg.backward(color=0):
                self.value = shared(self.env.information_set)
                self.minimize((self.value - self.env.target) ** 2, models=[shared])
                self.strategy = self.legal_action_mask / self.action_set_size

    dual = SharedGraph().bind(_Environment(), batch_size=4, devices=devices).trainer
    single = SharedGraph().bind(_Environment(), batch_size=4, devices=devices[:1]).trainer
    assert initialized == [True]
    assert dual.model_state(shared) is single.model_state(shared) is shared.state
    expected = np.asarray(shared.state.params).copy()
    expected_step = 0
    slots = np.arange(8)
    batch = _records((slots != 3, slots % 3 == 0), poison=False)

    def apply_oracle(records):
        nonlocal expected, expected_step
        inputs = np.concatenate([np.asarray(row["information_set"])[np.asarray(row["_valid"])]
                                 for row in records])
        targets = np.concatenate([np.asarray(row["target"])[np.asarray(row["_valid"])]
                                  for row in records])
        gradient = 2 * inputs.T @ (inputs @ expected - targets) / len(targets)
        expected -= 0.1 * gradient
        expected_step += 1

    def assert_live(trainer):
        assert dual.model_state(shared) is single.model_state(shared) is shared.state
        assert dual.states[0][0] is single.states[1][0] is shared.state
        assert int(shared.state.step) == expected_step
        np.testing.assert_allclose(shared.state.params, expected, rtol=2e-5, atol=2e-6)
        if trainer is dual:
            _replicated(shared.state, devices)
        else:
            assert all(leaf.devices() == {devices[0]}
                       for leaf in jax.tree_util.tree_leaves(shared.state))

    apply_oracle(batch)
    dual.optimize(batch, upd_color=[0])
    assert_live(dual)

    # A single-device graph must see another graph's latest replicated model.
    # Its update also receives globally shaped, episode-sharded input arrays.
    collected = dual.collect()
    assert all(leaf.devices() == set(devices) for leaf in jax.tree_util.tree_leaves(collected))
    before = shared.state
    prepared = single.update(collected, upd_color=[0])
    assert shared.state is before
    prediction_key = single.graph._node_key(single.graph.value.index)
    for row in prepared:
        np.testing.assert_allclose(row[prediction_key], expected.sum(), rtol=2e-5, atol=2e-6)
    assert all(leaf.devices() == {devices[0]} for leaf in jax.tree_util.tree_leaves(prepared))

    flattened = tuple({name: value.reshape((-1,) + value.shape[2:])
                       for name, value in row.items()} for row in collected)
    apply_oracle(flattened)
    single.optimize(flattened, upd_color=[0])
    assert_live(single)
    apply_oracle(batch)
    dual.optimize(batch, upd_color=[0])
    assert_live(dual)

    # Compiled epochs must support the same transitions in state placement.
    # Every collected observation is identical, so each shuffled two-episode
    # group has the same independent pooled gradient as the complete batch.
    for trainer in (single, dual):
        apply_oracle(flattened)
        apply_oracle(flattened)
        report = trainer.optimize_epochs(collected, epochs=1, minibatch_size=2,
                                           generator=jax.random.PRNGKey(23), upd_color=[0])
        assert report["valid_samples"] == [4, 4]
        assert_live(trainer)
    assert expected_step == 7
    assert initialized == [True]


@pytest.mark.parametrize("check", ["modes", "weighted_gradients", "compiled_epochs",
                                   "failed_updates", "shared_resources"])
def test_parallel_stages_in_isolated_runtime(check):
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, JAX_PLATFORMS="cpu",
                       XLA_FLAGS="--xla_force_host_platform_device_count=2",
                       XLA_PYTHON_CLIENT_PREALLOCATE="false",
                       PYTHONPATH=os.pathsep.join((str(root), os.environ.get("PYTHONPATH", ""))))
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), check],
                            cwd=root, env=environment, capture_output=True,
                            text=True, timeout=240)
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    {"modes": _check_modes, "weighted_gradients": _check_weighted_gradients,
     "compiled_epochs": _check_compiled_epochs,
     "failed_updates": _check_failed_updates,
     "shared_resources": _check_shared_resources}[sys.argv[1]]()
