"""Point microbatches preserve optimizer updates and bound feature evaluation."""

import os
from pathlib import Path
import re
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

if __name__ == "__main__" and hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import LiteEFG as leg
from LiteEFG.drl.env import Goofspiel
from test_drl_compact_batches import _assert_close_tree, _oracle_epoch_batches, _trainer
from test_drl_compiled_updates import _algorithm
from test_drl_microbatch import _hard_records, _oracle_step
from test_drl_parallel import _replicated


def _episode_data():
    times, _ = np.indices((5, 7))
    masks = (times < np.array([5, 2, 1, 4, 3, 1, 5]),
             times < np.array([0, 1, 0, 2, 0, 1, 0]))
    return _hard_records((5, 7), masks)


def _assert_weighted_report(report, reports):
    counts = np.asarray([value["valid_samples"] for value in reports])
    np.testing.assert_array_equal(report["valid_samples"], counts.sum(0))
    for name in ("loss", "prediction", "sample_id"):
        expected = (np.asarray([value[name] for value in reports]) * counts).sum(0)
        expected /= np.maximum(counts.sum(0), 1)
        np.testing.assert_allclose(report[name], expected, rtol=5e-5, atol=3e-6)


@pytest.mark.parametrize("sharing", ["independent", "shared", "partial"])
@pytest.mark.parametrize("point_size", [3, 64])
@pytest.mark.parametrize("compact_batches", [False, True])
def test_point_chunks_match_independent_weighted_gradient_and_adam_oracle(
        sharing, point_size, compact_batches):
    trainer = _trainer(sharing, compact_batches=compact_batches)
    slots = np.arange(35).reshape(5, 7)
    for masks in ((slots % 3 != 0, slots == 31),
                  (slots % 7 == 0, slots < 0),
                  (slots < 0, slots < 0)):
        data = _hard_records((5, 7), masks)
        batch = next(iter(leg.dataloader(data, batch_size=7)))
        assert batch.trajectory_steps == 5
        before = tuple(resource.state for resource in trainer.graph.models)
        expected, metrics, participation = _oracle_step(trainer, batch)
        report = trainer.optimize(batch, upd_color=[1], microbatch_size=point_size, microbatch_unit="points")
        _assert_close_tree(tuple(resource.state for resource in trainer.graph.models), expected)
        np.testing.assert_allclose(np.asarray([report[name] for name in
                                             ("loss", "prediction", "sample_id")]).T,
                                   metrics, rtol=5e-5, atol=3e-6)
        np.testing.assert_array_equal(report["valid_samples"], [mask.sum() for mask in masks])
        for resource, previous, count in zip(trainer.graph.models, before, participation):
            assert int(resource.state.step) == int(previous.step) + bool(count)
            if not count:
                assert resource.state is previous


@pytest.mark.parametrize("sharing", ["independent", "shared", "partial"])
@pytest.mark.parametrize("point_size", [1, 3, 128])
def test_point_chunks_preserve_original_trajectory_updates_and_final_partial_group(
        sharing, point_size):
    data = _episode_data()
    original = jax.tree.map(lambda value: np.asarray(value).copy(), data)
    reference = _trainer(sharing, compact_batches=False)
    actual = _trainer(sharing)
    options = dict(epochs=2, minibatch_size=3, generator=jax.random.PRNGKey(31))
    reports = [reference.optimize(batch, upd_color=[1])
               for batch in _oracle_epoch_batches(data, **options)]
    assert len(reports) == 6
    report = actual.optimize_epochs(data, upd_color=[1], microbatch_size=point_size, microbatch_unit="points", **options)
    _assert_close_tree(actual.states, reference.states)
    _assert_weighted_report(report, reports)
    for before, after in zip(jax.tree.leaves(original), jax.tree.leaves(data)):
        np.testing.assert_array_equal(before, after)


def test_empty_compiled_groups_preserve_nonzero_optimizer_moments():
    trainer = _trainer()
    data = _episode_data()
    options = dict(epochs=2, minibatch_size=3, generator=jax.random.PRNGKey(31), upd_color=[1],
                   microbatch_size=3, microbatch_unit="points")
    trainer.optimize_epochs(data, **options)
    before = tuple(resource.state for resource in trainer.graph.models)
    empty = _hard_records((5, 7), (np.zeros((5, 7), bool), np.zeros((5, 7), bool)))
    report = trainer.optimize_epochs(empty, **options)
    assert report["valid_samples"] == [0, 0]
    assert report["loss"] == [0.0, 0.0]
    assert all(resource.state is state for resource, state in zip(trainer.graph.models, before))


@pytest.mark.parametrize("compiled", [False, True])
def test_raw_point_executable_is_reused_when_only_valid_density_changes(compiled):
    trainer = _trainer(compact_batches=False)
    slots = np.arange(35).reshape(5, 7)
    execute = None
    for masks in ((slots >= 0, slots >= 0),
                  (slots % 9 == 0, slots == 31),
                  (slots < 0, slots < 0),
                  (slots % 2 == 0, slots % 3 == 0)):
        data = _hard_records((5, 7), masks)
        if compiled:
            trainer.optimize_epochs(data, epochs=1, minibatch_size=3,
                                      generator=jax.random.PRNGKey(7), upd_color=[1],
                                      microbatch_size=3, microbatch_unit="points")
            stage = "optimize_epochs"
        else:
            batch = next(iter(leg.dataloader(data, batch_size=7)))
            trainer.optimize(batch, upd_color=[1], microbatch_size=3, microbatch_unit="points")
            stage = "optimize"
        keys = [key for key in trainer._compiled if isinstance(key, tuple) and key[0] == stage]
        assert len(keys) == 1
        current = trainer._compiled[keys[0]]
        assert execute is None or current is execute
        execute = current
        assert execute._cache_size() == 1


def _equations(value):
    if hasattr(value, "jaxpr"):
        yield from _equations(value.jaxpr)
    elif hasattr(value, "eqns"):
        for equation in value.eqns:
            yield equation
            for parameter in equation.params.values():
                yield from _equations(parameter)
    elif isinstance(value, dict):
        for member in value.values():
            yield from _equations(member)
    elif isinstance(value, (tuple, list)):
        for member in value:
            yield from _equations(member)


def _compaction_equation(equation):
    name = equation.primitive.name + " " + str(equation.params.get("name", ""))
    return "cumsum" in name or "nonzero" in name


def _assert_bounded_feature_gathers_and_one_compaction(execute, arguments, point_size):
    program = jax.make_jaxpr(execute)(*arguments)
    equations = list(_equations(program))
    feature_gathers = [output.aval.shape for equation in equations
                       if equation.primitive.name == "gather"
                       for output in equation.outvars
                       if hasattr(output.aval, "shape") and len(output.aval.shape) == 2
                       and output.aval.shape[-1] == 3]
    assert feature_gathers, "The inspection must find actual information_set feature gathers"
    assert all(shape[0] <= point_size for shape in feature_gathers), feature_gathers
    assert any(_compaction_equation(equation) for equation in equations)
    point_loops = [equation for equation in equations if equation.primitive.name == "while"]
    assert point_loops, "The point chunk loop must be inspected"
    for loop in point_loops:
        assert not any(_compaction_equation(equation)
                       for equation in _equations(loop.params["body_jaxpr"])), (
                           "Valid-index compaction must be outside the gradient chunk loop")


def test_raw_compaction_precedes_point_chunks_without_a_dense_feature_gather(monkeypatch):
    data = _episode_data()
    trainer = _trainer("partial", compact_batches=False)
    point_size = 3
    options = dict(epochs=1, minibatch_size=3, generator=jax.random.PRNGKey(17), upd_color=[1],
                   microbatch_size=point_size, microbatch_unit="points")
    common_index = trainer.graph.model_index(trainer.graph.common)
    observations = []
    differentiate = trainer._minibatch_gradients

    def record_values(step, player, sample_ids, valid):
        observations.append((int(np.asarray(step)), int(np.asarray(player)[0]),
                             np.asarray(sample_ids)[np.asarray(valid)].astype(int)))

    def record_chunk(states, batch, upd_color=None):
        assert batch["_valid"].shape[0] <= point_size
        jax.debug.callback(record_values, states[common_index].step, batch["player"],
                           batch["sample_id"], batch["_valid"], ordered=True)
        return differentiate(states, batch, upd_color)

    monkeypatch.setattr(trainer, "_minibatch_gradients", record_chunk)
    trainer.optimize_epochs(data, **options)
    keys = [key for key in trainer._compiled if isinstance(key, tuple)
            and key[0] == "optimize_epochs"]
    assert len(keys) == 1
    execute = trainer._compiled[keys[0]]
    arguments = []

    def capture(*args):
        arguments.append(args)
        return execute(*args)

    with patch.dict(trainer._compiled, {keys[0]: capture}):
        trainer.optimize_epochs(data, **options)
    jax.effects_barrier()
    batches = list(_oracle_epoch_batches(
        data, epochs=1, minibatch_size=3, generator=options["generator"])) * 2
    for step, batch in enumerate(batches):
        expected_ids = [np.asarray(row["sample_id"])[np.asarray(row["_valid"])].astype(int)
                        for row in batch]
        chunks = max(1, (max(map(len, expected_ids)) + point_size - 1) // point_size)
        for player, expected in enumerate(expected_ids):
            actual = [ids for index, seat, ids in observations if index == step and seat == player]
            assert len(actual) == chunks
            for index, ids in enumerate(actual):
                np.testing.assert_array_equal(ids, expected[index * point_size:(index + 1) * point_size])
    assert len(arguments) == 1
    _assert_bounded_feature_gathers_and_one_compaction(execute, arguments[0], point_size)
    assert int(trainer.graph.common.state.step) == 6



@pytest.mark.parametrize("compiled", [False, True])
def test_raw_switching_microbatch_units_uses_distinct_cached_model_shapes(monkeypatch, compiled):
    trainer = _trainer("partial", compact_batches=False)
    data = _hard_records((5, 2), (np.ones((5, 2), bool), np.ones((5, 2), bool)))
    batch = next(iter(leg.dataloader(data, batch_size=2)))
    widths = []
    differentiate = trainer._minibatch_gradients

    def record_chunk(states, records, upd_color=None):
        jax.debug.callback(lambda value: widths.append(int(np.asarray(value))),
                           jnp.asarray(records["_valid"].size), ordered=True)
        return differentiate(states, records, upd_color)

    monkeypatch.setattr(trainer, "_minibatch_gradients", record_chunk)
    point_executable = None
    for unit in ("points", "trajectories", "points"):
        start = len(widths)
        options = dict(upd_color=[1], microbatch_size=2, microbatch_unit=unit)
        if compiled:
            trainer.optimize_epochs(data, epochs=1, minibatch_size=2,
                                      generator=jax.random.PRNGKey(7), **options)
            stage = "optimize_epochs"
        else:
            trainer.optimize(batch, **options)
            stage = "optimize"
        jax.effects_barrier()
        actual_widths = widths[start:]
        assert actual_widths
        assert max(actual_widths) <= 2 if unit == "points" else max(actual_widths) > 2
        executables = [value for key, value in trainer._compiled.items()
                       if isinstance(key, tuple) and key[0] == stage]
        if point_executable is None:
            assert len(executables) == 1
            point_executable = executables[0]
        else:
            assert len(executables) == 2
            assert point_executable in executables
        assert point_executable._cache_size() == 1


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("compact_batches", [False, True])
def test_full_minibatch_ignores_valid_unit_and_reuses_default_cache(compiled, compact_batches):
    data = _episode_data()
    batch = next(iter(leg.dataloader(data, batch_size=7)))
    actual = _trainer("partial", compact_batches=compact_batches)
    reference = _trainer("partial", compact_batches=compact_batches)
    execute = None
    for unit_options in ({}, {"microbatch_size": -1, "microbatch_unit": "trajectories"},
                         {"microbatch_size": -1, "microbatch_unit": "points"}):
        if compiled:
            options = dict(epochs=1, minibatch_size=3, generator=jax.random.PRNGKey(7),
                           upd_color=[1])
            expected = reference.optimize_epochs(data, **options)
            report = actual.optimize_epochs(data, **unit_options, **options)
            stage = "optimize_epochs"
        else:
            expected = reference.optimize(batch, upd_color=[1])
            report = actual.optimize(batch, upd_color=[1], **unit_options)
            stage = "optimize"
        _assert_close_tree(actual.states, reference.states)
        _assert_close_tree(report, expected)
        keys = [key for key in actual._compiled if isinstance(key, tuple) and key[0] == stage]
        assert len(keys) == 1
        current = actual._compiled[keys[0]]
        assert execute is None or current is execute
        execute = current


@pytest.mark.parametrize("compiled", [False, True])
def test_default_unit_matches_explicit_trajectories_without_new_cache(compiled):
    trainer = _trainer("partial")
    data = _episode_data()
    batch = next(iter(leg.dataloader(data, batch_size=7)))
    keys = None
    for extra in ({}, {"microbatch_unit": "trajectories"}):
        if compiled:
            trainer.optimize_epochs(data, epochs=1, minibatch_size=3,
                                      generator=jax.random.PRNGKey(7), upd_color=[1],
                                      microbatch_size=2, **extra)
        else:
            trainer.optimize(batch, upd_color=[1], microbatch_size=2, **extra)
        if keys is None:
            keys = dict(trainer._compiled)
        else:
            assert trainer._compiled == keys


@pytest.mark.parametrize("unit", [None, "", "point", "trajectory", "Points", True, 1, [], {}])
@pytest.mark.parametrize("method", ["optimize", "epochs", "ppo"])
def test_invalid_unit_is_rejected_before_collection_or_updates(monkeypatch, unit, method):
    if method == "ppo":
        graph = _algorithm().bind(Goofspiel(num_cards=3), batch_size=1, seed=11)
        trainer = graph.trainer

        def unexpected_collection():
            pytest.fail("Invalid microbatch units must be rejected before collecting trajectories")

        monkeypatch.setattr(trainer, "collect", unexpected_collection)
    else:
        trainer = _trainer()
    before = trainer.states
    for size in (-1, 2):
        with pytest.raises(ValueError, match="microbatch_unit"):
            if method == "ppo":
                graph.train(microbatch_size=size, microbatch_unit=unit)
            elif method == "epochs":
                trainer.optimize_epochs(_episode_data(), epochs=1, minibatch_size=3,
                                          generator=jax.random.PRNGKey(0), upd_color=[1],
                                          microbatch_size=size, microbatch_unit=unit)
            else:
                batch = next(iter(leg.dataloader(_episode_data(), batch_size=7)))
                trainer.optimize(batch, upd_color=[1], microbatch_size=size, microbatch_unit=unit)
    assert trainer.states is before
    assert trainer.iteration == 0
    assert not trainer._compiled


@pytest.mark.parametrize("sharing", ["independent", "partial"])
def test_ppo_forwards_points_to_both_schedules_and_matches_full_updates(monkeypatch, sharing):
    trainers = [_algorithm(sharing).bind(Goofspiel(num_cards=3), batch_size=5, seed=11).trainer
                for _ in range(3)]
    expected = trainers[0].graph.train(iterations=1, epochs=2, minibatch_size=2,
                                        compiled_updates=False)
    for trainer, compiled in zip(trainers[1:], (False, True)):
        name = "optimize_epochs" if compiled else "optimize"
        method = getattr(trainer, name)
        calls = []

        def record_call(*args, **kwargs):
            calls.append(kwargs)
            return method(*args, **kwargs)

        monkeypatch.setattr(trainer, name, record_call)
        actual = trainer.graph.train(iterations=1, epochs=2, minibatch_size=2,
                                       microbatch_size=2, microbatch_unit="points",
                                       compiled_updates=compiled)
        assert calls and all(call["microbatch_unit"] == "points" for call in calls)
        assert all(call["microbatch_size"] == 2 for call in calls)
        _assert_close_tree(actual, expected)
        _assert_close_tree(trainer.states, trainers[0].states)
        np.testing.assert_array_equal(trainer.key, trainers[0].key)


def _with_device_collective_check(trainer, stage, action, device_count):
    keys = [key for key in trainer._compiled if isinstance(key, tuple) and key[0] == stage]
    assert len(keys) == 1
    execute = trainer._compiled[keys[0]]
    arguments = []

    def capture(*args):
        arguments.append(args)
        return execute(*args)

    with patch.dict(trainer._compiled, {keys[0]: capture}):
        result = action()
    assert len(arguments) == 1
    executable = execute.lower(*arguments[0]).compile()
    assert len(executable.runtime_executable().local_devices()) == device_count
    hlo = executable.as_text().lower()
    reductions = [line.split(" all-reduce", 1)[0] for line in hlo.splitlines()
                  if re.search(r"\ball-reduce(?:-start)?\(", line)]
    assert any(re.search(r"\bf32\[3\](?:\{[^}]*\})?", value) for value in reductions), (
        "The distributed executable must reduce three-weight model gradients")
    return result


def _check_devices(device_count):
    devices = tuple(jax.local_devices())
    assert len(devices) == device_count
    trainer = _trainer("partial", devices=devices, parallel=("optimizing",))
    data = _episode_data()
    batch = next(iter(leg.dataloader(data, batch_size=7)))
    for repeat in range(2):
        expected, _, _ = _oracle_step(trainer, batch)
        action = lambda: trainer.optimize(batch, upd_color=[1], microbatch_size=3, microbatch_unit="points")
        if repeat:
            _with_device_collective_check(trainer, "optimize", action, device_count)
        else:
            action()
        _assert_close_tree(tuple(resource.state for resource in trainer.graph.models), expected)
        _replicated(trainer.states, devices)

    actual = _trainer("partial", devices=devices, parallel=("optimizing",))
    reference = _trainer("partial", compact_batches=False)
    options = dict(epochs=2, minibatch_size=3, generator=jax.random.PRNGKey(31))
    for repeat in range(2):
        reports = [reference.optimize(chunk, upd_color=[1])
                   for chunk in _oracle_epoch_batches(data, **options)]
        action = lambda: actual.optimize_epochs(data, upd_color=[1], microbatch_size=3, microbatch_unit="points", **options)
        report = (_with_device_collective_check(actual, "optimize_epochs", action, device_count)
                  if repeat else action())
        _assert_close_tree(actual.states, reference.states)
        _assert_weighted_report(report, reports)
        _replicated(actual.states, devices)


@pytest.mark.parametrize("device_count", [2, 4])
def test_odd_point_sizes_use_all_devices_and_reduce_gradients(device_count):
    environment = dict(os.environ, JAX_PLATFORMS="cpu", OPENBLAS_NUM_THREADS="1",
                       OMP_NUM_THREADS="1",
                       XLA_FLAGS=f"--xla_force_host_platform_device_count={device_count}",
                       XLA_PYTHON_CLIENT_PREALLOCATE="false")
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(device_count)],
                            cwd=ROOT, env=environment, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _check_devices(int(sys.argv[1]))
