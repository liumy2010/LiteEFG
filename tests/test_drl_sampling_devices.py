"""Shared-policy sampling checks with two isolated virtual CPU devices."""

import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

# Keep virtual-device subprocesses small on hosts with many logical CPUs.
if __name__ == "__main__" and hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
optax = pytest.importorskip("optax")
nn = pytest.importorskip("flax.linen")

from LiteEFG.baselines.drl.PPO import graph
from LiteEFG.drl.env import Goofspiel
from LiteEFG.drl.runtime import Trainer
import LiteEFG as leg


def _algorithm():
    actors = [leg.model(nn.Dense(3), optimizer=optax.adam(0.03)) for _ in range(2)]
    critics = [leg.model(nn.Dense(1), optimizer=optax.adam(0.03)) for _ in range(2)]
    return graph(actors, critics, num_actions=3)


def _assert_same_tree(left, right):
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    assert left_tree == right_tree
    for actual, expected in zip(left_leaves, right_leaves):
        if np.issubdtype(np.asarray(actual).dtype, np.inexact):
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
        else:
            np.testing.assert_array_equal(actual, expected)


def _assert_on_device(tree, device):
    for value in jax.tree_util.tree_leaves(tree):
        if isinstance(value, jax.Array):
            assert value.devices() == {device}


def _check_shared_sampling():
    devices = jax.local_devices()
    assert len(devices) == 2
    env = Goofspiel(num_cards=3)
    single = Trainer(_algorithm(), env, batch_size=16, seed=19)
    algo = _algorithm()
    # Sampling order must not change the learner's device or episode order.
    assert algo.bind(env, batch_size=16, seed=19,
                     sampling_devices=devices[::-1]) is algo
    dual = algo.trainer
    assert tuple(single.sampling_devices) == (devices[0],)
    assert tuple(dual.sampling_devices) == tuple(devices[::-1])
    _assert_same_tree(single.states, dual.states)
    before = dual.states

    single_data, dual_data = single.collect(), dual.collect()
    _assert_same_tree(single_data, dual_data)
    _assert_same_tree(single.key, dual.key)
    np.testing.assert_allclose(single._mean_utility, dual._mean_utility)
    assert single.iteration == dual.iteration == 1
    assert dual.states is before
    for records in dual_data:
        assert records["_valid"].shape == (env.max_steps, 16)
        assert bool(records["_valid"].all())
        assert all(value.shape[:2] == (env.max_steps, 16)
                   for value in records.values())
        # Reusing the first shard's keys would duplicate these complete paths.
        assert not np.array_equal(records["full_info"][:, :8],
                                  records["full_info"][:, 8:])
    _assert_on_device((dual_data, dual.states, dual.key), devices[0])

    frozen = tuple(dual.policy(player) for player in range(env.num_players))
    reports = []
    for trainer, data in ((single, single_data), (dual, dual_data)):
        prepared = trainer.update(data, upd_color=[0])
        _assert_on_device(prepared, devices[0])
        flattened = tuple({name: value.reshape((-1,) + value.shape[2:])
                           for name, value in records.items()}
                          for records in prepared)
        reports.append(trainer.optimize(flattened, upd_color=[1]))
    _assert_same_tree(reports[0], reports[1])
    _assert_same_tree(single.states, dual.states)
    assert reports[1]["trajectories"] == 16
    assert reports[1]["valid_samples"] == [48, 48]
    assert len(dual.states) == env.num_players
    assert len(dual.graph.models) == 4
    for old_model, resource in zip(before[0], dual.graph.models):
        new_model = resource.state
        assert int(new_model.step) == 1
        assert any(not np.array_equal(old, new) for old, new in zip(
            jax.tree_util.tree_leaves(old_model.params),
            jax.tree_util.tree_leaves(new_model.params)))
    _assert_on_device((dual.states, dual.key), devices[0])

    # A compiled sampler must receive the latest weights on every device.
    single_data, dual_data = single.collect(), dual.collect()
    _assert_same_tree(single_data, dual_data)
    _assert_same_tree(single.key, dual.key)
    np.testing.assert_allclose(single._mean_utility, dual._mean_utility)
    assert single.iteration == dual.iteration == 2
    for player, records in enumerate(dual_data):
        probabilities = dual.policy(player)(records["information_set"],
                                             records["legal_action_mask"])
        selected = jnp.take_along_axis(probabilities,
                                      records["action"][..., None], axis=-1)[..., 0]
        np.testing.assert_allclose(records["log_sampling_prob"], jnp.log(selected),
                                   rtol=2e-5, atol=2e-6)
        old_probabilities = frozen[player](records["information_set"],
                                           records["legal_action_mask"])
        assert not np.allclose(probabilities, old_probabilities)
    _assert_on_device(dual_data, devices[0])


def _check_invalid_devices():
    devices = jax.local_devices()
    assert len(devices) == 2
    env = Goofspiel(num_cards=3)
    for selected, batch_size in (([], 4), ([devices[0], devices[0]], 4),
                                  (devices, 3), ([object()], 4)):
        algo = _algorithm()
        with pytest.raises((TypeError, ValueError)):
            algo.bind(env, batch_size=batch_size, sampling_devices=selected)
        assert algo._trainer is None
        # Validation failure must leave this graph available for a valid bind.
        algo.bind(env, batch_size=4, sampling_devices=devices)
        assert algo.trainer.iteration == 0
    algo = _algorithm()
    with patch.object(jax, "local_devices", return_value=devices[:1]):
        with pytest.raises(ValueError):
            algo.bind(env, batch_size=4, sampling_devices=devices[1:])
    assert algo._trainer is None


def _check_failed_collection():
    class TruncatedGoofspiel(Goofspiel):
        @property
        def max_steps(self):
            return 1

    for invalid_policy in (False, True):
        algo = _algorithm()
        if invalid_policy:
            algo.strategy = algo.strategy * 0.0
        env = Goofspiel(num_cards=3) if invalid_policy else TruncatedGoofspiel(num_cards=3)
        trainer = Trainer(algo, env, batch_size=4, seed=11,
                          sampling_devices=jax.local_devices())
        original_key = np.asarray(trainer.key).copy()
        original_utility = trainer._mean_utility.copy()
        original_states = trainer.states
        match = "invalid|illegal" if invalid_policy else "max_steps|truncated"
        # Retry exercises the cached sampler as well as its first invocation.
        for _ in range(2):
            with pytest.raises(ValueError, match=match):
                trainer.collect()
            np.testing.assert_array_equal(trainer.key, original_key)
            np.testing.assert_array_equal(trainer._mean_utility, original_utility)
            assert trainer.iteration == 0
            assert trainer.states is original_states


@pytest.mark.parametrize("check", ["shared_sampling", "invalid_devices", "failed_collection"])
def test_sampling_devices_in_isolated_runtime(check):
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, JAX_PLATFORMS="cpu",
                       XLA_FLAGS="--xla_force_host_platform_device_count=2",
                       XLA_PYTHON_CLIENT_PREALLOCATE="false",
                       PYTHONPATH=os.pathsep.join((str(root), os.environ.get("PYTHONPATH", ""))))
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), check],
                            cwd=root, env=environment,
                            capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    {"shared_sampling": _check_shared_sampling,
     "invalid_devices": _check_invalid_devices,
     "failed_collection": _check_failed_collection}[sys.argv[1]]()
