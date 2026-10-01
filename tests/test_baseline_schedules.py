"""Check configurable update schedules against native graph state on Kuhn poker."""

import os

import numpy as np
import pyspiel
import pytest

import LiteEFG as leg
from LiteEFG.baselines import CMD, Reg_CFR, Reg_DOMD


@pytest.fixture
def kuhn_env(tmp_path, monkeypatch):
    expanduser = os.path.expanduser
    monkeypatch.setattr(os.path, "expanduser", lambda path:
                        str(tmp_path) if path == "~" else expanduser(path))
    return leg.OpenSpielEnv(pyspiel.load_game("kuhn_poker(players=2)"),
                            regenerate=True)


def node_values(env, node):
    return np.asarray([
        value
        for player in (1, 2)
        for _, values in env.get_value(player, node)
        for value in values
    ])


def trace_updates(env, monkeypatch):
    update = env.update
    calls = []

    def traced_update(strategy, **kwargs):
        calls.append((strategy, tuple(kwargs["upd_color"])))
        return update(strategy, **kwargs)

    monkeypatch.setattr(env, "update", traced_update)
    return calls


@pytest.mark.parametrize("regularizer", ["Entropy", "Euclidean"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("inner_epoch", [1, 10])
def test_cmd_publishes_strategy_only_after_complete_inner_epoch(
        kuhn_env, monkeypatch, regularizer, weighted, inner_epoch):
    graph = CMD.graph(regularizer=regularizer, weighted=weighted,
                      inner_epoch=inner_epoch)
    kuhn_env.set_graph(graph)
    calls = trace_updates(kuhn_env, monkeypatch)
    published = node_values(kuhn_env, graph.current_strategy())
    boundaries = set(range(inner_epoch, 51, inner_epoch))

    for iteration in range(1, 51):
        graph.update_graph(kuhn_env)
        inner = node_values(kuhn_env, graph.u)
        actual = node_values(kuhn_env, graph.current_strategy())
        assert graph.current_strategy() is graph.bar_u
        assert len(calls) == iteration
        assert calls[-1][0] is graph.u
        assert calls[-1][1][0] == 0
        assert sum(1 in colors for _, colors in calls) == iteration // inner_epoch

        if iteration in boundaries:
            published = inner
        np.testing.assert_array_equal(actual, published)
        if iteration == 1 and inner_epoch > 1:
            # A held outer strategy must not hide a missing inner update.
            assert np.max(np.abs(inner - published)) > 1e-6


@pytest.mark.parametrize("module,warmup", [(Reg_CFR, 1), (Reg_DOMD, 0)],
                         ids=["Reg_CFR", "Reg_DOMD"])
@pytest.mark.parametrize("regularizer", ["Entropy", "Euclidean"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("out_reg", [False, True])
@pytest.mark.parametrize("shrink_iter", [1, 10, 100000])
def test_regularization_halves_at_scheduled_boundaries_after_warmup(
        kuhn_env, monkeypatch, module, warmup, regularizer, weighted,
        out_reg, shrink_iter):
    initial_tau = 0.1
    graph = module.graph(tau=initial_tau, regularizer=regularizer,
                         weighted=weighted, out_reg=out_reg,
                         shrink_iter=shrink_iter)
    kuhn_env.set_graph(graph)
    calls = trace_updates(kuhn_env, monkeypatch)

    for iteration in range(1, 51):
        graph.update_graph(kuhn_env)
        # Count completed schedule intervals, excluding the initialization
        # interval where Reg_CFR uses its initial learning rate unchanged.
        halvings = iteration // shrink_iter - warmup // shrink_iter
        expected_tau = initial_tau * 2.0 ** -halvings
        np.testing.assert_allclose(node_values(kuhn_env, graph.tau),
                                   expected_tau, rtol=1e-12, atol=0)
        np.testing.assert_allclose(node_values(kuhn_env, graph.coef),
                                   expected_tau, rtol=1e-12, atol=0)
        assert len(calls) == iteration
        assert calls[-1][0] is graph.u
        assert calls[-1][1][0] == 0
        assert sum(1 in colors for _, colors in calls) == halvings
        expected_adaptations = iteration - warmup if warmup else 0
        assert sum(2 in colors for _, colors in calls) == expected_adaptations
        if iteration == 1 and warmup:
            np.testing.assert_array_equal(node_values(kuhn_env, graph.eta), 1.0)
