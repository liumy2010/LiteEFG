"""Numerical and overload regressions exercised through the native graph API."""

import math
import operator
from pathlib import Path
import subprocess
import sys

import pytest

import LiteEFG as leg


@pytest.fixture
def operation_env(tmp_path):
    path = tmp_path / "operations.game"
    path.write_text("\n".join([
        "# Opt {", "#     openspiel,", "#     players: 2,", "# }",
        "node start player 1 actions left right",
        "node left player 2 actions ll lr",
        "node right player 2 actions rl rr",
        "node ll leaf payoffs 1=0 2=0",
        "node lr leaf payoffs 1=0 2=0",
        "node rl leaf payoffs 1=0 2=0",
        "node rr leaf payoffs 1=0 2=0",
        "infoset p1 nodes start",
        "infoset p2 nodes left right",
    ]) + "\n")
    return leg.FileEnv(str(path))


def static_operation(values, operation):
    graph = leg.Graph()
    with leg.backward(is_static=True):
        source = (leg.cat([leg.const(1, value) for value in values])
                  if values else leg.const(0, 0.0))
        result = operation(source)
    return graph, result


@pytest.mark.parametrize("operation,reference", [
    (leg.argmax, max), (leg.argmin, min),
])
@pytest.mark.parametrize("values", [
    [0.9, 0.1], [-0.9, -0.1],
    [-1e30, -1e29], [1e30, 1e29],
    [-0.2, -0.2], [0.25],
    [-math.inf, math.inf],
])
def test_extrema_preserve_floating_point_order_and_first_tie(
        operation_env, operation, reference, values):
    graph, result = static_operation(values, operation)
    operation_env.set_graph(graph)
    expected = [0.0] * len(values)
    expected[values.index(reference(values))] = 1.0
    for player in (1, 2):
        assert operation_env.get_value(player, result)[0][1] == expected


@pytest.mark.parametrize("operation", [leg.argmax, leg.argmin])
def test_extrema_reject_empty_inputs(operation_env, operation):
    graph, _ = static_operation([], operation)
    with pytest.raises(ValueError, match="requires a non-empty input"):
        operation_env.set_graph(graph)


@pytest.mark.parametrize("size,value", [(0, 0.25), (1, -0.5), (4, 0.25)])
def test_const_accepts_integer_size_and_graph_value(operation_env, size, value):
    graph = leg.Graph()
    with leg.backward(is_static=True):
        # The value is a graph expression, so the overload must preserve its
        # dependency and use its result when initializing the constant vector.
        scalar = leg.const(1, value / 2) * 2
        result = leg.const(size=size, val=scalar)
    operation_env.set_graph(graph)
    for player in (1, 2):
        assert operation_env.get_value(player, result)[0][1] == [value] * size


@pytest.mark.parametrize("operation", [
    operator.add, operator.sub, operator.mul, operator.truediv,
], ids=["add", "subtract", "multiply", "divide"])
@pytest.mark.parametrize("scalar", [-2.5, 3.25])
@pytest.mark.parametrize("size", [4, 17])
def test_scalar_left_broadcast_preserves_value_when_growing(
        operation_env, operation, scalar, size):
    values = [-4.0, 0.5, 2.0, 8.0] * (size // 4 + 1)
    values = values[:size]
    graph, result = static_operation(
        values, lambda source: operation(leg.const(1, scalar), source))
    # A fresh operation starts with a scalar temporary. Broadcasting to more
    # than two entries must grow its storage while preserving the scalar.
    operation_env.set_graph(graph)
    expected = [operation(scalar, value) for value in values]
    for player in (1, 2):
        assert operation_env.get_value(player, result)[0][1] == pytest.approx(
            expected, rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("shifted", [False, True])
@pytest.mark.parametrize("values", [
    [1.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5],
    [0.2, 0.3, 0.5], [1e-12, 1 - 1e-12], [-1e-12, 1.0],
])
def test_entropy_extends_to_zero_probability(operation_env, values, shifted):
    graph, result = static_operation(
        values, lambda source: leg.negative_entropy(source, shifted=shifted))
    operation_env.set_graph(graph)
    expected = sum(value * math.log(value) for value in values if value > 0)
    if shifted:
        expected += math.log(len(values))
    for player in (1, 2):
        actual = operation_env.get_value(player, result)[0][1][0]
        assert math.isfinite(actual)
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_entropy_still_rejects_negative_probability(operation_env):
    graph, _ = static_operation([-0.1, 1.1], leg.negative_entropy)
    with pytest.raises(ValueError, match="non-negative"):
        operation_env.set_graph(graph)


def test_entropy_does_not_hide_nan_input(operation_env):
    graph, result = static_operation([math.nan, 1.0], leg.negative_entropy)
    operation_env.set_graph(graph)
    for player in (1, 2):
        assert math.isnan(operation_env.get_value(player, result)[0][1][0])


@pytest.mark.parametrize("scores,mu,gamma", [
    ([0.8, 0.2], [0.8, 0.2], 0.5),
    ([-0.4, 1.3], [0.8, 0.2], 0.5),
    ([0.8, 0.1, 0.1], [0.2, 0.3, 0.5], 0.9),
    ([2.0, -1.0, 0.4], [0.0, 0.3, 0.7], 0.7),
    ([0.2, 0.3, 0.5], [0.6, 0.2, 0.2], 0.0),
    ([0.8, 0.2], [0.5, 0.5], 0.5),
])
def test_l2_projection_satisfies_nonuniform_lower_bound_optimality(
        operation_env, scores, mu, gamma):
    graph = leg.Graph()
    with leg.backward(is_static=True):
        source = leg.cat([leg.const(1, value) for value in scores])
        reference = leg.cat([leg.const(1, value) for value in mu])
        result = source.project("L2", gamma, reference)
        repeated = result.project("L2", gamma, reference)
    operation_env.set_graph(graph)

    # Independently solve sum(max(score_i - threshold, lower_i)) = 1.
    # This is the optimality condition in the original, untranslated variables.
    bounds = [gamma * value for value in mu]
    left, right = min(scores) - 1, max(scores) + 1
    for _ in range(100):
        threshold = (left + right) / 2
        mass = sum(max(value - threshold, bound)
                   for value, bound in zip(scores, bounds))
        if mass > 1:
            left = threshold
        else:
            right = threshold
    expected = [max(value - (left + right) / 2, bound)
                for value, bound in zip(scores, bounds)]
    for player in (1, 2):
        actual = operation_env.get_value(player, result)[0][1]
        assert actual == pytest.approx(expected, abs=1e-12)
        assert sum(actual) == pytest.approx(1.0, abs=1e-12)
        assert all(value >= bound - 1e-12
                   for value, bound in zip(actual, bounds))
        assert operation_env.get_value(player, repeated)[0][1] == pytest.approx(
            actual, abs=1e-12)


@pytest.mark.parametrize("context", [leg.backward, leg.forward])
def test_set_graph_initializes_all_colors_after_selective_update(
        operation_env, context):
    graph = leg.Graph()
    with leg.backward(is_static=True):
        strategy = leg.const(2, 0.5)
    with context(is_static=True, color=7):
        state = leg.const(1, 1.0) + 1.0
    operation_env.set_graph(graph)
    operation_env.update(strategy, upd_color=[0])
    for player in (1, 2):
        operation_env.set_value(player, state, [9.0])
    operation_env.set_graph(graph)
    for player in (1, 2):
        assert operation_env.get_value(player, state)[0][1] == [2.0]


@pytest.mark.parametrize("call,expected_error", [
    ("env.update([])", "strategies size"),
    ("env.update([strategy])", "strategies size"),
    ("env.update([strategy] * 3)", "strategies size"),
    ("env.update(strategy, upd_player=-2)", "upd_player"),
    ("env.update(strategy, upd_player=0)", "upd_player"),
    ("env.update(strategy, upd_player=3)", "upd_player"),
])
def test_update_rejects_invalid_players_and_strategy_lists(call, expected_error):
    root = Path(__file__).resolve().parents[1]
    script = f"""
import LiteEFG as leg
env = leg.FileEnv({str(root / 'LiteEFG/game_instances/kuhn.game')!r})
graph = leg.Graph()
strategy = leg.const(2, 0.5)
env.set_graph(graph)
try:
    {call}
except ValueError as exc:
    assert {expected_error!r} in str(exc), str(exc)
else:
    raise AssertionError('invalid update was accepted')
"""
    # A bad native vector index used to terminate the interpreter. Keep this
    # boundary regression isolated so a recurrence cannot abort the full suite.
    result = subprocess.run([sys.executable, "-c", script], cwd=root,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("base,power", [(2.0, 3.0), (0.5, -2.0), (3, 0.5)])
def test_scalar_base_power_uses_graph_exponent(operation_env, base, power):
    graph, result = static_operation([power], lambda source: base ** source)
    operation_env.set_graph(graph)
    for player in (1, 2):
        assert operation_env.get_value(player, result)[0][1] == pytest.approx(
            [base ** power])
