"""Public operators retain native behavior and lazily accept transient nodes."""

import ast
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pytest

import LiteEFG as leg


ROOT = Path(__file__).resolve().parents[1]


def operations(source):
    return (
        leg.copy(source), leg.exp(source), leg.log(source),
        leg.minimum(source, 0.8), leg.maximum(source, 0.8),
        leg.dot(source, source), leg.sum(source), leg.mean(source),
        leg.min(source), leg.max(source),
        leg.normalize(source, p_norm=2.0, ignore_negative=False),
    )


def expected_operations(values):
    return (
        values, np.exp(values), np.log(values),
        np.minimum(values, 0.8), np.maximum(values, 0.8),
        (values * values).sum(axis=-1), values.sum(axis=-1), values.mean(axis=-1),
        values.min(axis=-1), values.max(axis=-1),
        values / np.sqrt((values * values).sum(axis=-1, keepdims=True)),
    )


def test_public_operators_keep_native_graph_results():
    native_graph = leg.Graph()
    with leg.backward(is_static=True):
        source = leg.cat([leg.const(1, 0.5), leg.const(1, 2.0)])
        outputs = operations(source)
    assert all(isinstance(node, leg.GraphNode) for node in outputs)
    env = leg.FileEnv(str(ROOT / "LiteEFG/game_instances/kuhn.game"))
    env.set_graph(native_graph)
    for node, expected in zip(outputs, expected_operations(np.array([0.5, 2.0]))):
        for player in (1, 2):
            np.testing.assert_allclose(env.get_value(player, node)[0][1],
                                       np.atleast_1d(expected), rtol=1e-12, atol=1e-12)


def test_public_operators_lower_batched_drl_nodes():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("optax")
    program = leg.DRLGraph()
    outputs = operations(program.env.information_set)
    values = np.array([[0.5, 2.0], [1.0, 3.0]], dtype=np.float32)
    actual = jax.jit(lambda x: program.evaluate(
        (), {"information_set": x}, outputs))(jnp.asarray(values))
    for result, expected in zip(actual, expected_operations(values)):
        np.testing.assert_allclose(result, expected, rtol=2e-6, atol=1e-7)


def test_drl_dispatch_accepts_second_argument_nodes_keywords_and_where():
    pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("optax")
    program = leg.DRLGraph()
    source = program.env.information_set
    outputs = (
        leg.minimum(0.8, source), leg.maximum(arg0=0.8, arg1=source),
        leg.exp(arg0=source), leg.sum(arg0=source),
        leg.normalize(arg0=source, p_norm=1.0),
        leg.where(source > 1, source, 0),
    )
    values = jnp.array([0.5, 2.0])
    actual = program.evaluate((), {"information_set": values}, outputs)
    expected = ([0.5, 0.8], [0.8, 2], np.exp(np.array([0.5, 2])),
                2.5, [0.2, 0.8], [0, 2])
    for result, reference in zip(actual, expected):
        np.testing.assert_allclose(result, reference, rtol=2e-6, atol=1e-7)


def test_public_and_method_aggregate_lower_to_the_same_drl_trajectory_operation():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("optax")
    program = leg.DRLGraph()
    public = leg.aggregate(program.reward, "sum", "descendants", "self", -3.0,
                           discount=0.9, decay=0.8, include_self=True)
    method = program.reward.aggregate("sum", "descendants", "self", -3.0,
                                      discount=0.9, decay=0.8, include_self=True)
    program.metrics = {"public": public, "method": method}
    program.init(jax.random.key(0), {})
    rewards = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    valid = jnp.ones((3, 2), bool)

    def execute(values):
        prepared = program.update((), {"reward": values}, upd_color=0, valid=valid, alive=valid)
        return program.evaluate((), prepared, (public, method))

    result = jax.jit(execute)(rewards)
    expected = np.array([[1 + 0.72 * 3 + 0.72 ** 2 * 5,
                          2 + 0.72 * 4 + 0.72 ** 2 * 6],
                         [3 + 0.72 * 5, 4 + 0.72 * 6], [5, 6]])
    for actual in result:
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-7)


def test_public_and_method_aggregate_default_to_sum_and_accept_explicit_mean():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("optax")
    program = leg.DRLGraph()
    total = leg.aggregate(program.reward, object="batch")
    method_total = program.reward.aggregate(object="batch")
    mean = leg.aggregate(program.reward, "mean", object="batch")
    variance = ((program.reward - mean) ** 2).aggregate("mean", object="batch")
    program.metrics = {"total": total, "method_total": method_total,
                       "mean": mean, "variance": variance}
    program.init(jax.random.key(0), {})
    rewards = jnp.array([[2.0, jnp.nan], [4.0, jnp.nan], [6.0, jnp.nan]])
    valid = jnp.array([[True, False]] * 3)

    def execute(values):
        prepared = program.update((), {"reward": values}, upd_color=0, valid=valid, alive=valid)
        return program.evaluate((), prepared, (total, method_total, mean, variance))

    actual_total, actual_method_total, actual_mean, actual_variance = jax.jit(execute)(rewards)
    np.testing.assert_allclose(actual_total, 12.0)
    np.testing.assert_allclose(actual_method_total, 12.0)
    np.testing.assert_allclose(actual_mean, 4.0)
    np.testing.assert_allclose(actual_variance, 8.0 / 3.0)


@pytest.mark.parametrize("aggregator", [None, "mean"])
def test_public_aggregate_preserves_native_parameters_and_results(aggregator):
    import LiteEFG._LiteEFG as native

    graph = leg.Graph()
    with leg.backward(is_static=True):
        source = leg.const(2, 2.5)
        actual = (leg.aggregate(source, padding=-7.0) if aggregator is None else
                  leg.aggregate(source, aggregator, "children", "self", -7.0))
        expected = native.aggregate(source, "sum" if aggregator is None else aggregator,
                                    "children", "self", -7.0)
    assert isinstance(actual, leg.GraphNode)
    env = leg.FileEnv(str(ROOT / "LiteEFG/game_instances/kuhn.game"))
    env.set_graph(graph)
    for player in (1, 2):
        actual_values, expected_values = dict(env.get_value(player, actual)), dict(env.get_value(player, expected))
        assert actual_values.keys() == expected_values.keys()
        for infoset, values in actual_values.items():
            np.testing.assert_array_equal(values, expected_values[infoset])


@pytest.mark.parametrize("native_source", [False, True])
def test_aggregate_rejects_mixed_native_and_drl_nodes(native_source):
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    native_graph = leg.Graph()
    with leg.backward(is_static=True):
        native = leg.const(1, 1.0)
    program = leg.DRLGraph()
    source, padding = ((native, program.reward) if native_source else (program.reward, native))
    with pytest.raises(TypeError, match="Cannot mix native Graph and DRLGraph"):
        leg.aggregate(source, padding=padding)


def test_native_calls_do_not_import_optional_jax_dependencies():
    script = """
import importlib.abc
import sys
class BlockJax(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'jax', 'jaxlib', 'optax'}:
            raise ModuleNotFoundError('Optional dependency blocked: ' + fullname)
sys.meta_path.insert(0, BlockJax())
import LiteEFG as leg
import LiteEFG._LiteEFG as native
graph = leg.Graph()
with leg.backward(is_static=True):
    source = leg.const(2, 0.5)
    for name in ('copy', 'exp', 'log', 'sum', 'mean', 'min', 'max'):
        assert isinstance(getattr(leg, name)(source), leg.GraphNode)
    for name in ('minimum', 'maximum', 'dot'):
        assert isinstance(getattr(leg, name)(source, source), leg.GraphNode)
    assert isinstance(leg.normalize(source, p_norm=1.0, ignore_negative=True), leg.GraphNode)
    assert isinstance(leg.aggregate(source, 'sum', padding=-1.0), leg.GraphNode)
for call in (lambda: leg.exp(0), lambda: native.exp(0)):
    try:
        call()
    except TypeError as error:
        text = str(error)
        if 'first_error' in globals():
            assert text == first_error
        first_error = text
    else:
        raise AssertionError('Invalid native argument was accepted')
assert not any(name in sys.modules for name in ('jax', 'jaxlib', 'optax'))
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_stub_exports_all_lazy_deep_learning_names():
    module = ast.parse((ROOT / "LiteEFG/__init__.pyi").read_text(encoding="utf-8"))
    all_statement = next(node for node in module.body if isinstance(node, ast.Assign)
                         and any(isinstance(target, ast.Name) and target.id == "__all__"
                                 for target in node.targets))
    exports = set(ast.literal_eval(all_statement.value))
    assert {
        "DRLGraph", "Model", "ModelList", "model", "minimize", "clip", "stop_gradient",
        "masked_softmax", "entropy", "gather", "mse", "where", "Goofspiel",
        "Trainer", "Policy", "average_utility", "uniform_policy",
    } <= exports
