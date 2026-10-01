"""Graph construction scopes select a backend without changing node ownership."""

from contextlib import nullcontext
import operator
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import LiteEFG as leg


ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("native", "drl")


def graph_base(backend):
    if backend == "native":
        return leg.Graph
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    return leg.DRLGraph


def static_operations(backend):
    return leg.backward(is_static=True) if backend == "native" else nullcontext()


def assert_value(graph, node, expected):
    if getattr(node, "_liteefg_drl_node", False):
        assert node.graph is graph
        actual, = graph.evaluate((), {}, (node,))
        np.testing.assert_allclose(actual, expected)
    else:
        assert isinstance(node, leg.GraphNode)
        environment = leg.FileEnv(str(ROOT / "LiteEFG/game_instances/kuhn.game"))
        environment.set_graph(graph)
        for player in (1, 2):
            rows = environment.get_value(player, node)
            assert rows
            for _, value in rows:
                np.testing.assert_allclose(value, np.atleast_1d(expected))


@pytest.mark.parametrize("backend", BACKENDS)
def test_subclass_super_init_selects_const_and_dot_backend(backend):
    class Program(graph_base(backend)):
        def __init__(self):
            super().__init__()
            with static_operations(backend):
                self.source = leg.const(size=3, val=2.0)
                self.result = leg.dot(self.source, self.source)

    graph = Program()
    assert_value(graph, graph.source, [2.0, 2.0, 2.0])
    assert_value(graph, graph.result, 12.0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("size", [0, 1, 3])
def test_direct_base_construction_keeps_procedural_builder_active(backend, size):
    graph = graph_base(backend)()
    with static_operations(backend):
        node = leg.const(size, 1.25)
    assert_value(graph, node, np.full(size, 1.25))


@pytest.mark.parametrize("backend", BACKENDS)
def test_status_block_preserves_explicit_base_graph_activation(backend):
    base = graph_base(backend)
    original = leg.Graph()
    with leg.backward(is_static=True):
        original_node = leg.const(3, 2.0)
        active = base()
    # forward/backward scope restores traversal status, not graph selection.
    node = leg.const(3, 7.0)
    assert_value(active, node, [7.0] * 3)
    assert_value(original, original_node, [2.0] * 3)


@pytest.mark.parametrize("outer_backend", BACKENDS)
@pytest.mark.parametrize("inner_backend", BACKENDS)
def test_nested_graph_constructors_restore_outer_builder(outer_backend, inner_backend):
    class Child(graph_base(inner_backend)):
        def __init__(self):
            super().__init__()
            with static_operations(inner_backend):
                self.result = leg.const(3, 2.0)
                for _ in range(20):
                    self.result = self.result + 1.0

    class Parent(graph_base(outer_backend)):
        def __init__(self):
            super().__init__()
            with static_operations(outer_backend):
                self.before = leg.const(3, 1.0)
                self.child = Child()
                self.after = leg.const(3, 3.0)
                self.result = self.before + self.after

    graph = Parent()
    assert_value(graph, graph.result, [4.0] * 3)
    assert_value(graph.child, graph.child.result, [22.0] * 3)


@pytest.mark.parametrize("outer_backend", BACKENDS)
@pytest.mark.parametrize("inner_backend", BACKENDS)
def test_failed_nested_constructor_restores_outer_builder(outer_backend, inner_backend):
    class Broken(graph_base(inner_backend)):
        def __init__(self):
            super().__init__()
            with static_operations(inner_backend):
                self.partial = leg.const(3, 100.0)
            raise RuntimeError("intentional constructor failure")

    class Parent(graph_base(outer_backend)):
        def __init__(self):
            super().__init__()
            with static_operations(outer_backend):
                self.before = leg.const(3, 2.0)
                with pytest.raises(RuntimeError, match="intentional constructor failure"):
                    Broken()
                self.result = self.before + leg.const(3, 5.0)

    graph = Parent()
    assert_value(graph, graph.result, [7.0] * 3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_inherited_initializers_keep_scope_until_complete_subclass_returns(backend):
    base = graph_base(backend)
    anchor = base()

    class Parent(base):
        def __init__(self):
            super().__init__()
            with static_operations(backend):
                self.first = leg.const(3, 2.0)

    class Child(Parent):
        def __init__(self):
            super().__init__()
            with static_operations(backend):
                self.result = self.first + leg.const(3, 5.0)

    graph = Child()
    with static_operations(backend):
        anchor_result = leg.const(3, 11.0)
    assert_value(graph, graph.result, [7.0] * 3)
    assert_value(anchor, anchor_result, [11.0] * 3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_subclass_inheriting_base_initializer_restores_active_builder(backend):
    base = graph_base(backend)
    anchor = base()

    class Empty(base):
        pass

    child = Empty()
    with static_operations(backend):
        result = leg.const(3, 4.0)
    assert child is not anchor
    assert_value(anchor, result, [4.0] * 3)


@pytest.mark.parametrize("keywords", [False, True], ids=["positional", "keywords"])
def test_shared_operators_with_only_numeric_operands_follow_drl_scope(keywords):
    class Program(graph_base("drl")):
        def __init__(self):
            super().__init__()
            if keywords:
                self.one = leg.exp(arg0=0.0)
                self.product = leg.dot(arg0=[1.0, 2.0], arg1=[3.0, 4.0])
                self.total = leg.sum(arg0=[2.0, 3.0])
                self.unit = leg.normalize(arg0=[3.0, 4.0], p_norm=2.0)
            else:
                self.one = leg.exp(0.0)
                self.product = leg.dot([1.0, 2.0], [3.0, 4.0])
                self.total = leg.sum([2.0, 3.0])
                self.unit = leg.normalize([3.0, 4.0], p_norm=2.0)

    graph = Program()
    assert_value(graph, graph.one, 1.0)
    assert_value(graph, graph.product, 11.0)
    assert_value(graph, graph.total, 5.0)
    assert_value(graph, graph.unit, [0.6, 0.8])


@pytest.mark.parametrize("old_backend", BACKENDS)
@pytest.mark.parametrize("new_backend", BACKENDS)
def test_existing_node_keeps_owner_after_other_graph_is_activated(old_backend, new_backend):
    first = graph_base(old_backend)()
    with static_operations(old_backend):
        old_node = leg.const(3, 2.0)
    second = graph_base(new_backend)()
    with static_operations(old_backend):
        result = leg.dot(old_node, old_node) + 1.0
    # An operation on an old node must not steal the currently active builder.
    with static_operations(new_backend):
        second_node = leg.const(3, 9.0)
    assert_value(first, result, 13.0)
    assert_value(second, second_node, [9.0] * 3)


@pytest.mark.parametrize("left_backend,right_backend", [
    ("native", "native"), ("drl", "drl"), ("native", "drl"), ("drl", "native"),
])
@pytest.mark.parametrize("operation", [operator.add, leg.dot, lambda x, y: x.inplace(y)],
                         ids=["addition", "dot", "inplace"])
def test_cross_graph_and_cross_backend_nodes_are_rejected(left_backend, right_backend, operation):
    first = graph_base(left_backend)()
    with static_operations(left_backend):
        left = leg.const(3, 2.0)
    second = graph_base(right_backend)()
    with static_operations(right_backend):
        right = leg.const(3, 5.0)
    with pytest.raises((ValueError, TypeError, RuntimeError), match="(?i)graph|backend"):
        operation(left, right)
    # A rejected operation must not corrupt either graph or mutate its operand.
    assert_value(first, left, [2.0] * 3)
    assert_value(second, right, [5.0] * 3)


@pytest.mark.parametrize("local_scalar", [True, False], ids=["scalar", "length_one"])
def test_drl_const_accepts_scalar_node_value_and_broadcasts_per_sample(local_scalar):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    base = graph_base("drl")

    class Program(base):
        def __init__(self):
            super().__init__()
            value = self.reward if local_scalar else self.reward.reshape(1)
            self.result = leg.const(3, value)

    graph = Program()
    result, = jax.jit(lambda reward: graph.evaluate(
        (), {"reward": reward}, (graph.result,)))(jnp.array([2.0, 5.0]))
    np.testing.assert_array_equal(result, [[2.0] * 3, [5.0] * 3])


def test_drl_const_rejects_dynamic_length_with_clear_error():
    graph = graph_base("drl")()
    with pytest.raises((ValueError, TypeError), match="(?i)static|integer|fixed"):
        leg.const(graph.action_set_size, 1.0)


def test_drl_const_rejects_negative_length():
    graph_base("drl")()
    with pytest.raises((ValueError, TypeError), match="(?i)non.?negative|negative|size|length"):
        leg.const(-1, 1.0)


def test_native_only_import_and_empty_construction_context_are_safe():
    script = r'''
import importlib.abc
import sys

class BlockJax(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'jax', 'jaxlib', 'optax'}:
            raise ModuleNotFoundError('Optional dependency blocked: ' + fullname)

sys.meta_path.insert(0, BlockJax())
import LiteEFG as leg

def no_active_graph():
    try:
        leg.const(3, 1.0)
    except (RuntimeError, ValueError) as error:
        assert 'graph' in str(error).lower(), str(error)
    else:
        raise AssertionError('A pure constant needs an active graph')

no_active_graph()

class Working(leg.Graph):
    def __init__(self):
        super().__init__()
        with leg.backward(is_static=True):
            self.result = leg.dot(leg.const(3, 2.0), leg.const(3, 2.0))

class Broken(leg.Graph):
    def __init__(self):
        super().__init__()
        with leg.backward(is_static=True):
            self.result = leg.const(3, 1.0)
        raise RuntimeError('intentional failure')

working = Working()
assert isinstance(working.result, leg.GraphNode)
no_active_graph()
try:
    Broken()
except RuntimeError as error:
    assert str(error) == 'intentional failure'
else:
    raise AssertionError('Expected constructor failure')
no_active_graph()
assert not any(name in sys.modules for name in ('jax', 'jaxlib', 'optax'))
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
