"""Equivalence and failed-update checks for PPO's compiled optimization schedule."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
optax = pytest.importorskip("optax")
nn = pytest.importorskip("flax.linen")

from LiteEFG.baselines.drl.PPO import graph
from LiteEFG.drl.env import Goofspiel
import LiteEFG as leg


def _algorithm(sharing="independent"):
    actors = [leg.model(nn.Dense(3)) for _ in range(2)]
    critics = [leg.model(nn.Dense(1)) for _ in range(2)]
    actor = actors[0] if sharing == "shared" else actors
    critic = critics if sharing == "independent" else critics[0]
    algo = graph(actor, critic, num_actions=3)
    # Prepared values with another color must not become optimization metrics.
    algo.metrics["target"] = algo.return_target
    return algo


def _assert_close_tree(actual, expected):
    actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_tree = jax.tree_util.tree_flatten(expected)
    assert actual_tree == expected_tree
    for left, right in zip(actual_leaves, expected_leaves):
        if np.issubdtype(np.asarray(left).dtype, np.inexact):
            np.testing.assert_allclose(left, right, rtol=5e-5, atol=5e-6)
        else:
            np.testing.assert_array_equal(left, right)


def _bind_with_masks(monkeypatch, mode, sharing="independent"):
    trainer = _algorithm(sharing).bind(Goofspiel(num_cards=3), batch_size=3, seed=7).trainer
    collect, update = trainer.collect, trainer.update
    actions = []

    def record_collect():
        result = collect()
        actions.append(tuple(np.asarray(player["action"]) for player in result))
        return result

    def masked_update(data, *, upd_color):
        result = update(data, upd_color=upd_color)
        slots = jnp.arange(9).reshape(3, 3)
        masks = ((slots != 0, slots % 2 == 0) if mode == "mixed"
                 else (slots == 0, jnp.zeros_like(slots, dtype=bool)))
        return tuple(dict(records, _valid=mask)
                     for records, mask in zip(result, masks))

    monkeypatch.setattr(trainer, "collect", record_collect)
    monkeypatch.setattr(trainer, "update", masked_update)
    return trainer, actions


@pytest.mark.parametrize("minibatch_size,mask_mode,expected_steps", [
    (2, "mixed", (8, 8)),
    (2, "sparse", (4, 0)),
    (5, "mixed", (4, 4)),
])
def test_compiled_epochs_match_manual_schedule_and_weighted_metrics(
        monkeypatch, minibatch_size, mask_mode, expected_steps):
    manual, manual_actions = _bind_with_masks(monkeypatch, mask_mode)
    compiled, compiled_actions = _bind_with_masks(monkeypatch, mask_mode)
    initial_states = compiled.states
    minibatch_reports = []
    optimize = manual.optimize

    def record_optimize(batch, *, upd_color, **kwargs):
        report = optimize(batch, upd_color=upd_color, **kwargs)
        minibatch_reports.append(dict(report))
        return report

    monkeypatch.setattr(manual, "optimize", record_optimize)
    expected = manual.graph.train(iterations=2, epochs=2, minibatch_size=minibatch_size,
                                  compiled_updates=False)
    actual = compiled.graph.train(iterations=2, epochs=2,
                                  minibatch_size=minibatch_size)
    assert minibatch_reports
    _assert_close_tree(actual, expected)
    _assert_close_tree(compiled.states, manual.states)
    np.testing.assert_array_equal(compiled.key, manual.key)
    np.testing.assert_array_equal(compiled_actions, manual_actions)
    assert compiled.iteration == manual.iteration == actual["iteration"] == 2
    assert actual["trajectories"] == 6
    assert "target" not in actual
    assert actual["valid_samples"] == ([32, 20] if mask_mode == "mixed" else [4, 0])

    weights = np.asarray([report["valid_samples"] for report in minibatch_reports])
    np.testing.assert_array_equal(actual["valid_samples"], weights.sum(axis=0))
    for name in ("loss", *(metric for metric in compiled.graph.metrics if metric in actual)):
        values = np.asarray([report[name] for report in minibatch_reports])
        weighted_mean = (values * weights).sum(axis=0) / np.maximum(weights.sum(axis=0), 1)
        np.testing.assert_allclose(actual[name], weighted_mean, rtol=5e-5, atol=5e-6)

    for player, steps in enumerate(expected_steps):
        for resource in (compiled.graph.actor[player], compiled.graph.critic[player]):
            assert int(resource.state.step) == steps
            if steps == 0:
                initial = initial_states[0][compiled.graph.model_index(resource)]
                assert resource.state is initial


def test_compiled_failure_keeps_only_updates_accepted_before_nonfinite_state():
    def make_trainer():
        algo = _algorithm()

        def update(gradients, count, params=None):
            del params
            factor = jnp.where(count == 0, -0.001, jnp.nan)
            return jax.tree.map(lambda gradient: factor * gradient, gradients), count + 1

        # The first minibatch succeeds; every later candidate is invalid. This
        # detects both rollback of valid work and accidental commit after failure.
        for actor in algo.actor:
            actor.optimizer = optax.GradientTransformation(
                lambda params: jnp.array(0, dtype=jnp.int32), update)
        return algo.bind(Goofspiel(num_cards=3), batch_size=3, seed=7).trainer

    manual, compiled = make_trainer(), make_trainer()
    for trainer, use_compiled in ((manual, False), (compiled, True)):
        with pytest.raises(FloatingPointError, match="finite|Finite"):
            trainer.graph.train(iterations=1, epochs=2, minibatch_size=2,
                                compiled_updates=use_compiled)
        assert trainer.iteration == 1
        assert all(int(state.step) == 1 for player in trainer.states for state in player)
        assert all(np.isfinite(np.asarray(leaf)).all()
                   for leaf in jax.tree_util.tree_leaves(trainer.states))
    _assert_close_tree(compiled.states, manual.states)
    np.testing.assert_array_equal(compiled.key, manual.key)


@pytest.mark.parametrize("sharing", ["shared", "partial"])
@pytest.mark.parametrize("mask_mode", ["mixed", "sparse"])
def test_compiled_resource_sharing_matches_manual_updates(monkeypatch, sharing, mask_mode):
    manual, _ = _bind_with_masks(monkeypatch, mask_mode, sharing)
    compiled, _ = _bind_with_masks(monkeypatch, mask_mode, sharing)
    before = tuple(resource.state for resource in compiled.graph.models)
    expected = manual.graph.train(iterations=1, epochs=2, minibatch_size=16,
                                  compiled_updates=False)
    actual = compiled.graph.train(iterations=1, epochs=2, minibatch_size=16)
    _assert_close_tree(actual, expected)
    _assert_close_tree(compiled.states, manual.states)
    assert len(compiled.graph.models) == (2 if sharing == "shared" else 3)
    for resource, initial in zip(compiled.graph.models, before):
        inactive = sharing == "partial" and mask_mode == "sparse" and resource is compiled.graph.actor[1]
        assert int(resource.state.step) == (0 if inactive else 2)
        if inactive:
            assert resource.state is initial
        index = compiled.graph.model_index(resource)
        assert compiled.states[0][index] is compiled.states[1][index] is resource.state


def test_default_schedule_uses_compiled_updates(monkeypatch):
    algo = _algorithm("shared").bind(Goofspiel(num_cards=3), batch_size=1, seed=7)
    calls = []
    optimize_epochs = algo.trainer.optimize_epochs

    def record_epochs(*args, **kwargs):
        calls.append(kwargs)
        return optimize_epochs(*args, **kwargs)

    def unexpected_optimize(*args, **kwargs):
        pytest.fail("The default PPO schedule must use compiled optimizer epochs")

    monkeypatch.setattr(algo.trainer, "optimize_epochs", record_epochs)
    monkeypatch.setattr(algo.trainer, "optimize", unexpected_optimize)
    algo.train(epochs=1)
    assert len(calls) == 1
    assert calls[0]["epochs"] == 1
    assert calls[0]["minibatch_size"] == 128


def test_compiled_epochs_do_not_overwrite_a_resource_changed_during_execution(monkeypatch):
    algo = _algorithm("shared").bind(Goofspiel(num_cards=3), batch_size=1, seed=7)
    trainer = algo.trainer
    data = trainer.update(trainer.collect(), upd_color=[0])
    options = dict(epochs=1, minibatch_size=3, generator=jax.random.PRNGKey(9), upd_color=[1])
    trainer.optimize_epochs(data, **options)
    before = tuple(resource.state for resource in algo.models)
    replacement = before[0]._replace(step=before[0].step + 10)
    cache_key = ("optimize_epochs", algo._normalize_colors([1]), 1, 3, 3)
    execute = trainer._compiled[cache_key]

    def concurrent_update(*args):
        result = execute(*args)
        algo.models[0].state = replacement
        return result

    monkeypatch.setitem(trainer._compiled, cache_key, concurrent_update)
    with pytest.raises(RuntimeError, match="stale"):
        trainer.optimize_epochs(data, **options)
    assert algo.models[0].state is replacement
    assert all(resource.state is state for resource, state in zip(algo.models[1:], before[1:]))
