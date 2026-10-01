"""Check Reg-DOMD against the full sequence-form objective, not saved outputs.

Use the dilated regularizer in (3.3) and the update in (4.2) of
https://arxiv.org/pdf/2206.09495. A child reached through only one root action
makes omitted value constants observable in the root strategy.
"""

import numpy as np
import pytest

import LiteEFG as leg
from LiteEFG.baselines import Reg_DOMD


def two_level_env(tmp_path, payoffs):
    stop, left, right = payoffs
    path = tmp_path / "two_level.game"
    path.write_text("\n".join([
        "# Opt {", "#     openspiel,", "#     players: 2,", "# }",
        "node dummy player 2 actions start",
        "node start player 1 actions stop branch",
        "node branch player 1 actions left right",
        f"node stop leaf payoffs 1={stop} 2={-stop}",
        f"node left leaf payoffs 1={left} 2={-left}",
        f"node right leaf payoffs 1={right} 2={-right}",
        "infoset dummy nodes dummy",
        "infoset root nodes start",
        "infoset child nodes branch",
    ]) + "\n")
    return leg.FileEnv(str(path))


def sequence_strategy(env, node):
    values = dict(env.get_value(1, node))
    for probabilities in values.values():
        # The test cases have interior optima, so equality-constrained
        # first-order optimality below is sufficient for this convex problem.
        assert np.all(np.asarray(probabilities) > 0)
        assert sum(probabilities) == pytest.approx(1.0, abs=1e-14)
    root, child = np.asarray(values["root"]), np.asarray(values["child"])
    return np.concatenate((root, root[1] * child))


def dilated_gradient(sequence, alpha, regularizer):
    """Differentiate sum_h alpha_h * s_h * psi(x_h / s_h), s_h=sum(x_h).

    The sibling sums equal parent sequence probabilities on the feasible set.
    This computes the global gradient without local updates or value recursion.
    """
    blocks = sequence.reshape(2, 2)
    probabilities = blocks / blocks.sum(axis=1, keepdims=True)
    if regularizer == "Euclidean":
        gradient = probabilities - 0.5 * (probabilities**2).sum(axis=1, keepdims=True)
    else:
        gradient = np.log(probabilities)
    return (np.asarray(alpha)[:, None] * gradient).ravel()


def assert_global_optimum(candidate, reference, reward, alpha, regularizer, eta, tau):
    # Minimize -<reward, x> + tau*<grad Psi(reference), x>
    #          + D_Psi(x, reference)/eta over sequence-form strategies.
    reference_gradient = dilated_gradient(reference, alpha, regularizer)
    gradient = ((dilated_gradient(candidate, alpha, regularizer) - reference_gradient)
                / eta + tau * reference_gradient - reward)
    # Feasibility is x0+x1=1, x2+x3=x1. These directions span its tangent space.
    tangent = np.array([[1.0, -1.0, 0.0, -1.0], [0.0, 0.0, 1.0, -1.0]])
    np.testing.assert_allclose(tangent @ gradient, 0.0, rtol=0, atol=3e-11)


@pytest.mark.parametrize("regularizer", ["Euclidean", "Entropy"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("tau", [0.0, 0.1])
@pytest.mark.parametrize("initial,payoffs", [
    pytest.param({"root": [0.5, 0.5], "child": [0.5, 0.5]},
                 (0.0, 0.0, 0.0), id="uniform_zero_payoff"),
    pytest.param({"root": [0.45, 0.55], "child": [0.3, 0.7]},
                 (0.2, 0.7, -0.3), id="nonuniform_payoff"),
])
def test_both_updates_solve_sequence_form_objective(
        tmp_path, regularizer, weighted, tau, initial, payoffs):
    env = two_level_env(tmp_path, payoffs)
    eta = 0.1
    graph = Reg_DOMD.graph(eta=eta, tau=tau, regularizer=regularizer,
                          weighted=weighted, out_reg=False)
    env.set_graph(graph)
    for node in (graph.bar_u, graph.u):
        env.set_value(1, node, [initial[name] for name, _ in env.get_value(1, node)])
    if weighted:
        weights = dict(env.get_value(1, graph.alpha))
        alpha = [weights[name][0] for name in ("root", "child")]
    else:
        alpha = [1.0, 1.0]
    reward = np.array([payoffs[0], 0.0, payoffs[1], payoffs[2]])

    for _ in range(3):
        reference = sequence_strategy(env, graph.bar_u)
        graph.update_graph(env)
        bar = sequence_strategy(env, graph.bar_u)
        strategy = sequence_strategy(env, graph.u)
        # Both steps use the same game gradient here: player 2 has one action.
        assert_global_optimum(bar, reference, reward, alpha, regularizer, eta, tau)
        assert_global_optimum(strategy, bar, reward, alpha, regularizer, eta, tau)
