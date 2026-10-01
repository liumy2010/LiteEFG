"""Compare DCFR policy selection and finite boundary exponents with OpenSpiel."""

import os
import runpy
import sys

import numpy as np
import pyspiel
import pytest
from open_spiel.python.algorithms import discounted_cfr

import LiteEFG as leg
from LiteEFG.baselines import CFR, DCFR, utils


@pytest.fixture
def kuhn_env(tmp_path, monkeypatch):
    expanduser = os.path.expanduser
    monkeypatch.setattr(os.path, "expanduser", lambda path:
                        str(tmp_path) if path == "~" else expanduser(path))
    return leg.OpenSpielEnv(pyspiel.load_game("kuhn_poker(players=2)"), regenerate=True)


@pytest.mark.parametrize("parameters", [
    {}, {"alpha": 10}, {"beta": 10},
])
def test_policies_match_openspiel_dcfr(kuhn_env, parameters):
    graph = DCFR.graph(**parameters)
    kuhn_env.set_graph(graph)
    reference = discounted_cfr.DCFRSolver(kuhn_env.game, **parameters)

    for iteration in range(1, 51):
        graph.update_graph(kuhn_env)
        reference.evaluate_and_update_policy()
        if iteration not in (1, 2, 10, 25, 50):
            continue

        for selector, expected in (
                (None, reference.current_policy()),
                ("last-iterate", reference.current_policy()),
                ("average-iterate", reference.average_policy())):
            node = (graph.current_strategy() if selector is None
                    else graph.current_strategy(selector))
            for player in (1, 2):
                for infoset, actual in kuhn_env.get_value(player, node):
                    index = expected.state_lookup[infoset]
                    probabilities = expected.action_probability_array[index][
                        expected.legal_actions_mask[index] > 0]
                    np.testing.assert_allclose(
                        actual, probabilities,
                        rtol=1e-8, atol=1e-10,
                        err_msg=f"{parameters}, iteration {iteration}, {selector}, {infoset}")


@pytest.mark.parametrize("exponent", [-11, -10, 10, 11])
def test_boundary_coefficients_and_infinity_approximation(kuhn_env, exponent):
    graph = DCFR.graph(alpha=exponent, beta=exponent)
    kuhn_env.set_graph(graph)
    for timestep in (1, 2, 3):
        graph.update_graph(kuhn_env)
        if abs(exponent) <= 10:
            power = timestep ** exponent
            expected = power / (power + 1)
        else:
            expected = float(exponent > 0)
        for player in (1, 2):
            for node in (graph.pos_coeff, graph.neg_coeff):
                for _, coefficient in kuhn_env.get_value(player, node):
                    assert coefficient == pytest.approx([expected], rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("module, average", [(DCFR, True), (CFR, False)])
def test_command_line_evaluates_intended_policy(kuhn_env, monkeypatch, module, average):
    monkeypatch.setattr(leg, "OpenSpielEnv", lambda *args, **kwargs: kuhn_env)
    monkeypatch.setitem(sys.modules, "utils", utils)
    monkeypatch.setattr(sys, "argv", [module.__file__, "--game", "kuhn_poker",
                                      "--iter", "2", "--print_freq", "1"])
    graphs = []
    evaluated_nodes = []
    original_set_graph = kuhn_env.set_graph
    original_exploitability = kuhn_env.exploitability

    def set_graph(graph):
        graphs.append(graph)
        return original_set_graph(graph)

    def exploitability(node, convergence_type):
        evaluated_nodes.append(node)
        return original_exploitability(node, convergence_type)

    monkeypatch.setattr(kuhn_env, "set_graph", set_graph)
    monkeypatch.setattr(kuhn_env, "exploitability", exploitability)
    runpy.run_path(module.__file__, run_name="__main__")

    assert len(graphs) == 1
    expected = graphs[0].avg_strategy if average else graphs[0].strategy
    assert len(evaluated_nodes) == 2
    assert all(node is expected for node in evaluated_nodes)
