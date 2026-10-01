"""Whole-trajectory microbatches accumulate one gradient per optimizer update."""

import os
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__" and hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
optax = pytest.importorskip("optax")

import LiteEFG as leg
from LiteEFG.drl.env import Goofspiel
from test_drl_compact_batches import (
    _Regression, _RegressionEnvironment, _assert_close_tree, _oracle_epoch_batches,
    _records, _trainer,
)
from test_drl_compiled_updates import _algorithm
from test_drl_parallel import _replicated, _with_collective_check


def _oracle_step(trainer, batch):
    """Differentiate the regression in NumPy; run each supplied optimizer once."""
    graph = trainer.graph
    states = tuple(resource.state for resource in graph.models)
    numerators = [np.zeros_like(np.asarray(state.params)) for state in states]
    participation = np.zeros(len(states), dtype=int)
    metrics = []
    for player, row in enumerate(batch):
        valid = np.asarray(row["_valid"])
        features = np.asarray(row["information_set"])[valid]
        target = np.asarray(row["target"])[valid]
        indices = [graph.model_index(graph.local[player])]
        if graph.common is not None:
            indices.append(graph.model_index(graph.common))
        prediction = sum(features @ np.asarray(states[index].params) for index in indices)
        error = prediction - target
        metrics.append([np.mean(error ** 2), np.mean(prediction),
                        np.asarray(row["sample_id"])[valid].mean()] if valid.any()
                       else [0.0, 0.0, 0.0])
        for index in indices:
            numerators[index] += 2 * features.T @ error
            participation[index] += int(valid.sum())
    expected = []
    for resource, state, numerator, count in zip(graph.models, states, numerators, participation):
        if count:
            updates, optimizer_state = resource.optimizer.update(
                jnp.asarray(numerator / count), state.opt_state, state.params)
            state = state._replace(params=optax.apply_updates(state.params, updates),
                                   opt_state=optimizer_state, step=state.step + 1)
        expected.append(state)
    return tuple(expected), np.asarray(metrics), participation


def _hard_records(shape, masks):
    records = _records(shape, masks, poison=True)
    slots = jnp.arange(np.prod(shape)).reshape(shape)
    return tuple(dict(row, target=row["target"] + jnp.where(slots % 3 == player, -7.0, 3.0))
                 for player, row in enumerate(records))


@pytest.mark.parametrize("sharing", ["independent", "shared", "partial"])
@pytest.mark.parametrize("compact_batches", [False, True])
@pytest.mark.parametrize("microbatch_size", [3, 64])
def test_microbatch_matches_analytic_pooled_gradient_and_single_adam_step(
        sharing, compact_batches, microbatch_size):
    trainer = _trainer(sharing, compact_batches=compact_batches)
    slots = np.arange(17)
    masks = ((slots % 3 != 0, slots == 11),
             (slots % 4 == 0, slots < 0),
             (slots < 0, slots < 0))
    for current_masks in masks:
        batch = _hard_records((17,), current_masks)
        before = tuple(resource.state for resource in trainer.graph.models)
        expected, metrics, participation = _oracle_step(trainer, batch)
        report = trainer.optimize(batch, upd_color=[1], microbatch_size=microbatch_size)
        _assert_close_tree(tuple(resource.state for resource in trainer.graph.models), expected)
        np.testing.assert_array_equal(report["valid_samples"], [mask.sum() for mask in current_masks])
        np.testing.assert_allclose(np.asarray([report[name] for name in
                                             ("loss", "prediction", "sample_id")]).T,
                                   metrics, rtol=5e-5, atol=3e-6)
        for resource, previous, count in zip(trainer.graph.models, before, participation):
            assert int(resource.state.step) == int(previous.step) + bool(count)
            if not count:
                assert resource.state is previous


@pytest.mark.parametrize("compiled", [False, True])
def test_one_step_trajectory_microbatches_bound_model_rows_and_keep_optimizer_steps(monkeypatch, compiled):
    trainer = _trainer("partial", compact_batches=False)
    batch = _hard_records((17,), (np.ones(17, bool), np.arange(17) == 16))
    widths = []
    differentiate = trainer._minibatch_gradients

    def record_width(states, records, upd_color=None):
        widths.append(records["_valid"].shape[0])
        return differentiate(states, records, upd_color)

    monkeypatch.setattr(trainer, "_minibatch_gradients", record_width)
    if compiled:
        trajectories = jax.tree_util.tree_map(lambda value: value[None], batch)
        report = trainer.optimize_epochs(
            trajectories, epochs=1, minibatch_size=17, generator=jax.random.PRNGKey(0),
            upd_color=[1], microbatch_size=3)
    else:
        report = trainer.optimize(batch, upd_color=[1], microbatch_size=3)
    assert widths and max(widths) <= 3
    assert report["valid_samples"] == [17, 1]
    assert all(int(resource.state.step) == 1 for resource in trainer.graph.models)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("compact_batches", [False, True])
def test_microbatches_keep_complete_episode_groups_before_compaction(
        monkeypatch, compiled, compact_batches):
    horizon, episodes = 5, 7
    times, episode_ids = np.indices((horizon, episodes))
    masks = (times < np.array([5, 3, 1, 4, 5, 2, 5]),
             times < np.array([1, 5, 2, 3, 1, 4, 2]))
    data = _hard_records((horizon, episodes), masks)
    trainer = _trainer("partial", compact_batches=compact_batches)
    observed = []
    differentiate = trainer._minibatch_gradients

    def record_values(sample_ids, valid, players):
        valid = np.asarray(valid)
        observed.append((int(np.asarray(players)[0]), np.asarray(sample_ids)[valid].astype(int)))

    def record_chunk(states, batch, upd_color=None):
        jax.debug.callback(record_values, batch["sample_id"], batch["_valid"], batch["player"],
                           ordered=True)
        return differentiate(states, batch, upd_color)

    monkeypatch.setattr(trainer, "_minibatch_gradients", record_chunk)
    key = jax.random.PRNGKey(17)
    minibatch_size = 3 if compiled else episodes
    if compiled:
        report = trainer.optimize_epochs(data, epochs=1, minibatch_size=minibatch_size,
                                          generator=key, upd_color=[1], microbatch_size=2)
    else:
        batch = next(iter(leg.dataloader(data, batch_size=episodes, shuffle=True, generator=key)))
        assert batch.trajectory_steps == horizon
        report = trainer.optimize(batch, upd_color=[1], microbatch_size=2)
    jax.effects_barrier()

    _, shuffle_key = jax.random.split(key)
    order = np.asarray(jax.random.permutation(shuffle_key, episodes))
    expected_groups = [order[start + offset:min(start + offset + 2, start + minibatch_size, episodes)]
                       for start in range(0, episodes, minibatch_size)
                       for offset in range(0, min(minibatch_size, episodes - start), 2)]
    for player in range(2):
        chunks = [ids - 100 * player for seat, ids in observed if seat == player]
        assert len(chunks) == len(expected_groups)
        for actual, group in zip(chunks, expected_groups):
            expected = np.concatenate([
                np.arange(horizon)[masks[player][:, episode]] * episodes + episode
                for episode in group])
            np.testing.assert_array_equal(actual, expected)
            assert len(np.unique(actual % episodes)) <= 2
            for episode in group:
                np.testing.assert_array_equal(actual[actual % episodes == episode] // episodes,
                                              times[:, episode][masks[player][:, episode]])
    assert report["valid_samples"] == [int(mask.sum()) for mask in masks]
    expected_steps = (episodes + minibatch_size - 1) // minibatch_size
    assert all(int(resource.state.step) == expected_steps for resource in trainer.graph.models)


@pytest.mark.parametrize("compact_batches", [False, True])
@pytest.mark.parametrize("microbatch_size", [1, 4, 64])
def test_compiled_microbatches_preserve_complete_trajectory_groups_and_remainders(
        compact_batches, microbatch_size):
    times, episodes = np.indices((5, 7))
    data = _hard_records((5, 7), (times < np.array([1, 2, 3, 4, 5, 1, 4]),
                                (episodes == 3) & (times < 2)))
    reference = _trainer("partial", compact_batches=False)
    actual = _trainer("partial", compact_batches=compact_batches)
    options = dict(epochs=2, minibatch_size=3, generator=jax.random.PRNGKey(31))
    reports = [reference.optimize(batch, upd_color=[1])
               for batch in _oracle_epoch_batches(data, **options)]
    assert len(reports) == 6
    report = actual.optimize_epochs(data, upd_color=[1], microbatch_size=microbatch_size,
                                      **options)
    _assert_close_tree(actual.states, reference.states)
    counts = np.asarray([value["valid_samples"] for value in reports])
    np.testing.assert_array_equal(report["valid_samples"], counts.sum(0))
    for name in ("loss", *actual.graph.metrics):
        expected = (np.asarray([value[name] for value in reports]) * counts).sum(0)
        expected /= np.maximum(counts.sum(0), 1)
        np.testing.assert_allclose(report[name], expected, rtol=5e-5, atol=3e-6)
    assert int(actual.graph.common.state.step) == 6


@pytest.mark.parametrize("sharing", ["independent", "partial"])
def test_ppo_manual_and_compiled_microbatches_match_full_updates_and_frozen_targets(sharing):
    trainers = []
    for _ in range(3):
        algorithm = _algorithm(sharing)
        with leg.backward(color=1):
            algorithm.metrics["frozen_advantage"] = algorithm.advantage + 0.0
            algorithm.metrics["frozen_target"] = algorithm.return_target + 0.0
        trainers.append(algorithm.bind(Goofspiel(num_cards=3), batch_size=5, seed=11).trainer)
    reports = []
    for trainer, compiled, microbatch in zip(trainers, (False, False, True), (-1, 1, 1)):
        reports.append(trainer.graph.train(iterations=1, epochs=2, minibatch_size=2,
                                           compiled_updates=compiled,
                                           microbatch_size=microbatch))
    for trainer, report in zip(trainers[1:], reports[1:]):
        _assert_close_tree(report, reports[0])
        _assert_close_tree(trainer.states, trainers[0].states)
        np.testing.assert_array_equal(trainer.key, trainers[0].key)
    assert reports[2]["valid_samples"] == [30, 30]
    assert all(int(resource.state.step) == 6 for resource in trainers[2].graph.models)


def test_microbatch_optimization_preserves_cached_ppo_targets(monkeypatch):
    algorithm = _algorithm("partial")
    with leg.backward(color=1):
        algorithm.metrics["frozen_advantage"] = algorithm.advantage + 0.0
        algorithm.metrics["frozen_target"] = algorithm.return_target + 0.0
    trainer = algorithm.bind(Goofspiel(num_cards=3), batch_size=5, seed=11).trainer
    data = trainer.update(trainer.collect(), upd_color=[0])
    batch = next(iter(leg.dataloader(data, batch_size=5, shuffle=True,
                                   generator=jax.random.PRNGKey(7))))
    batch = jax.tree.map(lambda value: value, batch)
    assert batch.trajectory_steps == 3
    before = jax.tree.map(lambda value: np.asarray(value).copy(), batch)
    parameters = tuple(resource.state.params for resource in algorithm.models)
    expected = np.asarray([
        [np.asarray(value).mean() for value in algorithm.evaluate(
            parameters, row, (algorithm.advantage, algorithm.return_target), upd_color=[1])]
        for row in batch])

    def unexpected_preparation(*args, **kwargs):
        pytest.fail("Optimizer microbatches must reuse prepared trajectory targets")

    monkeypatch.setattr(algorithm, "update", unexpected_preparation)
    for _ in range(2):
        report = trainer.optimize(batch, upd_color=[1], microbatch_size=2)
        np.testing.assert_allclose(report["frozen_advantage"], expected[:, 0],
                                   rtol=5e-5, atol=3e-6)
        np.testing.assert_allclose(report["frozen_target"], expected[:, 1],
                                   rtol=5e-5, atol=3e-6)
    for original, current in zip(jax.tree.leaves(before), jax.tree.leaves(batch)):
        np.testing.assert_array_equal(current, original)


def test_incomplete_trajectory_metadata_is_rejected_before_optimization():
    trainer = _trainer()
    data = _records((5, 7), (np.ones((5, 7), bool), np.ones((5, 7), bool)))
    batch = next(iter(leg.dataloader(data, batch_size=3)))
    broken = jax.tree.map(lambda value: value[:-1], batch)
    assert broken.trajectory_steps == 5
    before = trainer.states
    with pytest.raises(ValueError):
        trainer.optimize(broken, upd_color=[1], microbatch_size=2)
    assert trainer.states is before


@pytest.mark.parametrize("microbatch_size", [None, 0, -2, True, False, 1.5, "3"])
@pytest.mark.parametrize("compiled", [False, True])
def test_invalid_microbatch_size_does_not_update_or_compile(microbatch_size, compiled):
    trainer = _trainer()
    batch = _records((7,), (np.ones(7, bool), np.ones(7, bool)))
    before = trainer.states
    with pytest.raises(ValueError, match="microbatch_size"):
        if compiled:
            data = jax.tree_util.tree_map(lambda value: value[None], batch)
            trainer.optimize_epochs(data, epochs=1, minibatch_size=3,
                                      generator=jax.random.PRNGKey(0), upd_color=[1],
                                      microbatch_size=microbatch_size)
        else:
            trainer.optimize(batch, upd_color=[1], microbatch_size=microbatch_size)
    assert trainer.states is before
    assert not trainer._compiled


def test_ppo_rejects_invalid_microbatch_before_collecting(monkeypatch):
    algorithm = _algorithm().bind(Goofspiel(num_cards=3), batch_size=1, seed=11)
    before = algorithm.trainer.states

    def unexpected_collection():
        pytest.fail("Invalid microbatch sizes must be rejected before collecting trajectories")

    monkeypatch.setattr(algorithm.trainer, "collect", unexpected_collection)
    for compiled in (False, True):
        for microbatch_size in (None, 0, -2, True, False, 1.5, "3"):
            with pytest.raises(ValueError, match="microbatch_size"):
                algorithm.train(microbatch_size=microbatch_size, compiled_updates=compiled)
    assert algorithm.trainer.states is before
    assert algorithm.trainer.iteration == 0


def test_default_and_minus_one_reuse_original_full_minibatch_cache():
    trainer = _trainer()
    batch = _records((7,), (np.ones(7, bool), np.ones(7, bool)))
    trainer.optimize(batch, upd_color=[1])
    keys = set(trainer._compiled)
    trainer.optimize(batch, upd_color=[1], microbatch_size=-1)
    assert set(trainer._compiled) == keys
    assert {key for key in keys if isinstance(key, tuple)} == {
        ("optimize", trainer.graph._normalize_colors([1]), 7)}
    data = jax.tree_util.tree_map(lambda value: value[None], batch)
    options = dict(epochs=1, minibatch_size=3, generator=jax.random.PRNGKey(0), upd_color=[1])
    trainer.optimize_epochs(data, **options)
    keys = set(trainer._compiled)
    trainer.optimize_epochs(data, microbatch_size=-1, **options)
    assert set(trainer._compiled) == keys
    assert ("optimize_epochs", trainer.graph._normalize_colors([1]), 1, 3, 3) in keys


@pytest.mark.parametrize("invalid_kind", ["loss", "optimizer"])
@pytest.mark.parametrize("microbatch_unit", ["trajectories", "points"])
def test_failed_microbatch_update_never_commits_partial_model_changes(invalid_kind, microbatch_unit):
    graph = _Regression("partial")
    if invalid_kind == "optimizer":
        graph.common.optimizer = optax.GradientTransformation(
            lambda params: (),
            lambda gradients, state, params=None: (jax.tree_util.tree_map(
                lambda value: value * jnp.nan, gradients), state))
    trainer = graph.bind(_RegressionEnvironment(), batch_size=7,
                         compact_batches=False).trainer
    batch = _records((17,), (np.ones(17, bool), np.ones(17, bool)))
    if invalid_kind == "loss":
        batch = (dict(batch[0], target=batch[0]["target"].at[-1].set(jnp.nan)), batch[1])
    before = tuple(resource.state for resource in graph.models)
    with pytest.raises(FloatingPointError, match="finite|Finite"):
        trainer.optimize(batch, upd_color=[1], microbatch_size=3, microbatch_unit=microbatch_unit)
    assert all(resource.state is state for resource, state in zip(graph.models, before))


@pytest.mark.parametrize("microbatch_unit", ["trajectories", "points"])
def test_compiled_microbatch_failure_keeps_preceding_successful_updates(microbatch_unit):
    def make_trainer():
        graph = _Regression("partial")

        def update(gradients, count, params=None):
            factor = jnp.where(count == 0, -0.001, jnp.nan)
            return jax.tree_util.tree_map(lambda value: factor * value, gradients), count + 1

        for resource in graph.models:
            resource.optimizer = optax.GradientTransformation(
                lambda params: jnp.array(0, jnp.int32), update)
        return graph.bind(_RegressionEnvironment(), batch_size=7).trainer

    data = _records((3, 7), (np.ones((3, 7), bool), np.ones((3, 7), bool)))
    options = dict(epochs=2, minibatch_size=3, generator=jax.random.PRNGKey(19))
    manual, compiled = make_trainer(), make_trainer()
    with pytest.raises(FloatingPointError):
        loader = leg.dataloader(data, batch_size=options["minibatch_size"], shuffle=True,
                                generator=options["generator"])
        for _ in range(options["epochs"]):
            for batch in loader:
                manual.optimize(batch, upd_color=[1], microbatch_size=2,
                                microbatch_unit=microbatch_unit)
    with pytest.raises(FloatingPointError):
        compiled.optimize_epochs(data, upd_color=[1], microbatch_size=2,
                                  microbatch_unit=microbatch_unit, **options)
    _assert_close_tree(compiled.states, manual.states)
    assert all(int(resource.state.step) == 1 for resource in compiled.graph.models)
    assert all(np.isfinite(np.asarray(leaf)).all()
               for leaf in jax.tree_util.tree_leaves(compiled.states))


def _check_two_devices():
    devices = tuple(jax.local_devices())
    assert len(devices) == 2
    slots = np.arange(17)
    batch = _hard_records((17,), (slots % 3 != 0, slots == 11))
    for compact_batches in (False, True):
        trainer = _trainer("partial", devices=devices, parallel=("optimizing",),
                            compact_batches=compact_batches)
        for repeat in range(2):
            expected, _, _ = _oracle_step(trainer, batch)
            action = lambda: trainer.optimize(batch, upd_color=[1], microbatch_size=3)
            report = (_with_collective_check(trainer, "optimize", action, gradient_size=3)
                      if repeat else action())
            _assert_close_tree(tuple(resource.state for resource in trainer.graph.models), expected)
            assert report["valid_samples"] == [11, 1]
            _replicated(trainer.states, devices)

        actual = _trainer("partial", devices=devices, parallel=("optimizing",),
                           compact_batches=compact_batches)
        reference = _trainer("partial", compact_batches=False)
        masks = np.arange(35).reshape(5, 7)
        data = _hard_records((5, 7), (masks % 3 != 0, masks == 11))
        options = dict(epochs=2, minibatch_size=4, generator=jax.random.PRNGKey(31))
        for repeat in range(2):
            for chunk in _oracle_epoch_batches(data, **options):
                reference.optimize(chunk, upd_color=[1])
            action = lambda: actual.optimize_epochs(data, upd_color=[1], microbatch_size=3,
                                                      **options)
            if repeat:
                _with_collective_check(actual, "optimize_epochs", action)
            else:
                action()
            _assert_close_tree(actual.states, reference.states)
            _replicated(actual.states, devices)


def test_odd_microbatches_use_two_devices_and_gradient_collectives():
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ, JAX_PLATFORMS="cpu",
                       XLA_FLAGS="--xla_force_host_platform_device_count=2",
                       XLA_PYTHON_CLIENT_PREALLOCATE="false",
                       PYTHONPATH=os.pathsep.join((str(root), os.environ.get("PYTHONPATH", ""))))
    result = subprocess.run([sys.executable, str(Path(__file__).resolve())],
                            cwd=root, env=environment, capture_output=True,
                            text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _check_two_devices()
