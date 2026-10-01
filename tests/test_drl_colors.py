"""Explicit color boundaries freeze ordinary values and trajectory reductions."""

import numpy as np
import pytest
from pathlib import Path

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("optax")

import LiteEFG as leg
from LiteEFG.drl.graph import DRLGraph, model, operation_node


def shared_model_graph():
    graph = DRLGraph()
    resource = model(init=lambda key, x: jnp.array(2.0), apply=lambda weight, x: weight * x)
    with leg.backward(color=0):
        old_value = resource(graph.env.information_set)
        target = old_value * 3
    with leg.backward(color=1):
        current_value = resource(graph.env.information_set)
        loss = ((current_value - target) ** 2).sum()
    graph.minimize(loss, models=[resource])
    graph.init(jax.random.key(0), {"information_set": jnp.zeros(2)})
    return graph, target, current_value, loss


def test_ordinary_expressions_freeze_at_color_boundaries_but_selected_models_differentiate():
    graph, target, current, loss = shared_model_graph()
    data = {"information_set": jnp.arange(12, dtype=jnp.float32).reshape(3, 2, 2)}

    def objective(snapshot_parameter, live_parameter):
        prepared = graph.update((snapshot_parameter,), data, upd_color=[0])
        losses, = graph.evaluate((live_parameter,), prepared, (loss,), upd_color=[1])
        return losses.sum()

    gradients = jax.jit(jax.grad(objective, argnums=(0, 1)))(jnp.array(2.0), jnp.array(5.0))
    np.testing.assert_array_equal(gradients[0], 0)
    np.testing.assert_allclose(gradients[1], -2 * np.square(data["information_set"]).sum())
    prepared = graph.update((jnp.array(2.0),), data, upd_color=0)
    selected = jnp.array([4, 1])
    minibatch = {name: value.reshape((-1,) + value.shape[2:])[selected]
                 for name, value in prepared.items()}
    frozen, live = graph.evaluate((jnp.array(5.0),), minibatch, (target, current), upd_color=1)
    np.testing.assert_allclose(frozen, 6 * data["information_set"].reshape(6, 2)[selected])
    np.testing.assert_allclose(live, 5 * data["information_set"].reshape(6, 2)[selected])


def test_updating_a_color_refreshes_its_ordinary_and_aggregate_caches():
    graph = DRLGraph()
    resource = model(init=lambda key, x: jnp.array(2.0), apply=lambda weight, x: weight * x)
    with leg.backward(color=0):
        values = resource(graph.env.information_set)
        centered = values - values.aggregate("mean", object="batch")
        target = centered * 2
    with leg.backward(color=1):
        result = target + 1
    data = {"information_set": jnp.arange(12, dtype=jnp.float32).reshape(3, 2, 2)}
    graph.init(jax.random.key(0), {"information_set": data["information_set"][0, 0]})
    valid = jnp.ones((3, 2), bool)
    execute = jax.jit(lambda parameter, records: graph.update(
        (parameter,), records, upd_color=0, valid=valid, alive=valid))
    prepared = execute(jnp.array(2.0), data)
    refreshed = execute(jnp.array(3.0), prepared)
    actual, = graph.evaluate((), refreshed, (result,), upd_color=[1])
    expected = 6 * (data["information_set"] - data["information_set"].mean((0, 1))) + 1
    np.testing.assert_allclose(actual, expected)
    assert not any(name.startswith("_drl_aggregate_") for name in refreshed)


def test_inference_uses_cache_but_selected_color_recomputes_its_loss():
    graph, target, current, loss = shared_model_graph()
    data = {"information_set": jnp.array([[1.0, 2.0]])}
    prepared = graph.update((jnp.array(2.0),), data, upd_color=[0])
    previous_loss = graph.update((jnp.array(5.0),), prepared, upd_color=[1])
    cached, = graph.evaluate((), previous_loss, (loss,))
    refreshed, = graph.evaluate((jnp.array(4.0),), previous_loss, (loss,), upd_color=[1])
    np.testing.assert_array_equal(cached, [5])
    np.testing.assert_array_equal(refreshed, [20])


def test_missing_preceding_color_errors_without_running_its_operations():
    graph = DRLGraph()
    calls = []
    with leg.forward(color=4):
        earlier = operation_node(lambda x: calls.append(True) or x * 2, graph.reward)
    with leg.backward(color=9):
        later = earlier + 1
    data = {"reward": jnp.array([2.0, 3.0])}
    for execute in (
        lambda: graph.evaluate((), data, (later,), upd_color=9),
        lambda: graph.update((), data, upd_color=9),
    ):
        with pytest.raises(ValueError, match="cached color 4"):
            execute()
    assert calls == []
    prepared = graph.update((), data, upd_color=4)
    actual, = graph.evaluate((), prepared, (later,), upd_color=9)
    np.testing.assert_array_equal(actual, [5, 7])
    assert len(calls) == 1


def test_aggregate_only_runs_on_explicit_full_trajectory_update():
    graph = DRLGraph()
    with leg.backward(color=3):
        total = graph.reward.aggregate(object="descendants", include_self=True)
    with leg.backward(color=8):
        output = total * 2
    rewards = jnp.array([[1.0], [2.0], [4.0]])
    valid = jnp.ones((3, 1), bool)
    data = {"reward": rewards}
    with pytest.raises(ValueError, match="valid/alive masks"):
        graph.update((), data, upd_color=3)
    with pytest.raises(ValueError, match="requires prepared"):
        graph.evaluate((), data, (total,), upd_color=3)
    prepared = graph.update((), data, upd_color=3, valid=valid, alive=valid)
    with pytest.raises(ValueError, match="requires prepared"):
        graph.evaluate((), prepared, (total,), upd_color=3)
    only_cache = {name: value.reshape(-1)[jnp.array([2, 0])]
                  for name, value in prepared.items() if name.startswith("_drl_node_")}
    actual, = graph.evaluate((), only_cache, (output,), upd_color=8)
    np.testing.assert_array_equal(actual, [8, 14])


def test_scope_colors_restore_after_nesting_and_exceptions():
    graph = DRLGraph()
    with leg.backward(color=2):
        first = graph.reward + 1
        with leg.forward(color=7):
            nested = graph.reward + 2
        restored = graph.reward + 3
        with pytest.raises(RuntimeError):
            with leg.backward(color=8):
                raise RuntimeError("leave inner scope")
        after_error = graph.reward + 4
    default = graph.reward + 5
    assert [graph._colors[node.index] for node in
            (first, nested, restored, after_error, default)] == [2, 7, 2, 2, 0]
    assert graph._colors[graph.reward.index] is None
    assert graph._normalize_colors([-1]) == (0, 2, 7)
    assert graph._normalize_colors([7, 2, 7]) == (2, 7)
    with pytest.raises(ValueError, match="is_static"):
        with leg.backward(is_static=True):
            graph.reward + 1


def test_neutral_values_broadcast_across_selected_color_batch():
    graph = DRLGraph()
    with leg.backward(color=2):
        constant_expression = graph.as_node(3.0).copy()
        reward_expression = graph.reward + constant_expression
    data = {"reward": jnp.arange(6, dtype=jnp.float32).reshape(3, 2)}
    prepared = graph.update((), data, upd_color=2)
    constant, reward = graph.evaluate((), prepared, (constant_expression, reward_expression))
    np.testing.assert_array_equal(constant, np.full((3, 2), 3.0))
    np.testing.assert_array_equal(reward, data["reward"] + 3)


def test_native_scope_dispatch_preserves_bound_scope_types():
    from LiteEFG import _LiteEFG as native

    graph = leg.Graph()
    assert isinstance(leg.forward(is_static=True, color=7), native.forward)
    assert isinstance(leg.backward(is_static=True, color=7), native.backward)
    with leg.backward(is_static=True, color=7):
        assert isinstance(leg.const(1, 2.0), native.GraphNode)


def test_scope_colors_follow_old_drl_nodes_after_another_graph_activates():
    graph = DRLGraph()
    active_native = leg.Graph()
    with leg.forward(color=6):
        result = graph.reward + 1
    assert graph._colors[result.index] == 6
    with leg.backward(is_static=True):
        assert isinstance(leg.const(1, 2.0), leg.GraphNode)
    assert active_native is not None


def test_native_dynamic_colors_work_while_a_drl_graph_is_active():
    graph = leg.Graph()
    with leg.backward(is_static=True):
        strategy = leg.const(graph.action_set_size, 1.0).normalize(1.0)
        counter = leg.const(1, 2.0)
    active_drl = DRLGraph()
    with leg.forward(color=4):
        counter.inplace(counter + 1)
        drl_result = active_drl.reward + 1
    assert active_drl._colors[drl_result.index] == 4
    path = Path(__file__).resolve().parents[1] / "LiteEFG/game_instances/kuhn.game"
    environment = leg.FileEnv(str(path))
    environment.set_graph(graph)
    for color, expected in ((None, 2.0), (4, 3.0)):
        if color is not None:
            environment.update(strategy, upd_color=[color])
        for player in (1, 2):
            for _, value in environment.get_value(player, counter):
                np.testing.assert_array_equal(value, [expected])


@pytest.mark.parametrize("fail", [False, True])
def test_child_constructor_resets_and_restores_lexical_color(fail):
    expressions = []

    class Child(DRLGraph):
        def __init__(self):
            super().__init__()
            expressions.append(self.reward + 1)
            with leg.forward(color=8):
                expressions.append(self.reward + 2)
            if fail:
                raise RuntimeError("child constructor failed")

    parent = DRLGraph()
    with leg.backward(color=3):
        if fail:
            with pytest.raises(RuntimeError, match="child constructor failed"):
                Child()
        else:
            Child()
        parent_result = parent.reward + 1
    assert [node.graph._colors[node.index] for node in expressions] == [0, 8]
    assert parent._colors[parent_result.index] == 3
    default = parent.reward + 1
    assert parent._colors[default.index] == 0
