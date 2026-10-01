"""Independent sampled-path references for transient DRL aggregation."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("optax")

import LiteEFG as leg
from LiteEFG.drl.graph import DRLGraph, aggregate, model


def records(values):
    shape = values.shape[:2]
    return {
        "information_set": jnp.zeros(shape + (2,), jnp.float32),
        "legal_action_mask": jnp.ones(shape + (3,), bool),
        "signal": jnp.asarray(values, jnp.float32),
    }


def prepared_graph(values, valid, alive, **options):
    graph = DRLGraph()
    result = aggregate(graph.env.signal, **options)
    graph.metrics = {"result": result}
    inputs = records(values)
    graph.init(jax.random.PRNGKey(0), {key: value[0, 0] for key, value in inputs.items()})
    prepared = jax.jit(lambda data, active, live: graph.update(
        (), data, upd_color=0, valid=active, alive=live))(inputs, jnp.asarray(valid), jnp.asarray(alive))
    actual, = graph.evaluate((), prepared, (result,))
    return np.asarray(actual), graph, inputs, prepared, result


def reference(values, valid, alive, *, object, discount=1, decay=1,
              include_self=False, padding=-7, aggregator="sum"):
    result = np.full(values.shape, padding, dtype=float)
    if object == "batch":
        selected = values[valid & alive]
        reduced = getattr(np, aggregator)(selected, axis=0) if len(selected) else padding
        return np.broadcast_to(reduced, values.shape)
    for episode in range(valid.shape[1]):
        decisions = np.flatnonzero(valid[:, episode] & alive[:, episode])
        for position, step in enumerate(decisions):
            if object in {"children", "parent"}:
                other = position + (1 if object == "children" else -1)
                if 0 <= other < len(decisions):
                    source = decisions[other]
                    result[step, episode] = discount ** abs(source - step) * values[source, episode]
            elif object == "descendants":
                sources = decisions[position if include_self else position + 1:]
                if len(sources):
                    result[step, episode] = sum(
                        discount ** int(source - step)
                        * decay ** int(np.where(decisions == source)[0][0] - position)
                        * values[source, episode] for source in sources)
            else:
                end = decisions[position + 1] if position + 1 < len(decisions) else int(alive[:, episode].sum())
                result[step, episode] = sum(discount ** (source - step) * values[source, episode]
                                            for source in range(step, end))
    return result


@pytest.fixture
def paths():
    # Episode 0 ends at t=5; episode 1 ends at t=3. Some live steps
    # have no own decision, including the terminal step of episode 0.
    values = np.arange(1, 1 + 6 * 2 * 2 * 3, dtype=float).reshape(6, 2, 2, 3)
    alive = np.array([[1, 1], [1, 1], [1, 1], [1, 1], [1, 0], [1, 0]], bool)
    valid = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 0]], bool)
    values[~alive] = np.nan
    return values, valid, alive


@pytest.mark.parametrize("object", ["children", "parent", "descendants", "segment"])
@pytest.mark.parametrize("discount", [0.0, 0.5, 1.0])
def test_trajectory_aggregates_preserve_local_tensor_axes(paths, object, discount):
    values, valid, alive = paths
    options = dict(object=object, discount=discount, padding=-7)
    actual, *_ = prepared_graph(values, valid, alive, **options)
    np.testing.assert_allclose(actual, reference(values, valid, alive, **options))


@pytest.mark.parametrize("decay", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("include_self", [False, True])
def test_descendant_decay_counts_decisions_and_discount_counts_environment_steps(paths, decay, include_self):
    values, valid, alive = paths
    options = dict(object="descendants", discount=0.5, decay=decay,
                   include_self=include_self, padding=-7)
    actual, *_ = prepared_graph(values, valid, alive, **options)
    np.testing.assert_allclose(actual, reference(values, valid, alive, **options), atol=1e-6)


@pytest.mark.parametrize("aggregator", [None, "sum", "mean", "min", "max"])
@pytest.mark.parametrize("empty", [False, True])
def test_batch_reduction_ignores_invalid_samples_and_broadcasts_local_tensor(paths, aggregator, empty):
    values, valid, alive = paths
    if empty:
        valid = np.zeros_like(valid)
    values[~valid] = np.nan
    options = dict(object="batch", padding=-7)
    if aggregator is not None:
        options["aggregator"] = aggregator
    actual, *_ = prepared_graph(values, valid, alive, **options)
    options["aggregator"] = "sum" if aggregator is None else aggregator
    np.testing.assert_allclose(actual, reference(values, valid, alive, **options))


@pytest.mark.parametrize("aggregator", ["sum", "mean", "min", "max"])
@pytest.mark.parametrize("object", ["children", "parent"])
def test_single_neighbor_reductions_return_neighbor_tensor(paths, aggregator, object):
    values, valid, alive = paths
    options = dict(object=object, aggregator=aggregator, padding=-7)
    actual, *_ = prepared_graph(values, valid, alive, **options)
    np.testing.assert_allclose(actual, reference(values, valid, alive, **options))


@pytest.mark.parametrize("lam, expected_first", [(1.0, 0.5), (0.0, 0.375)])
def test_gae_is_composed_from_generic_aggregates_across_inactive_terminal_rewards(lam, expected_first):
    graph = DRLGraph()
    discounted_reward = graph.reward.aggregate("sum", object="segment", discount=0.5)
    following_value = graph.sampling_value.aggregate(object="children", discount=0.5)
    delta = discounted_reward + following_value - graph.sampling_value
    advantage = delta.aggregate("sum", object="descendants", discount=0.5, decay=lam, include_self=True)
    target = advantage + graph.sampling_value
    graph.metrics = {"advantage": advantage, "target": target}
    valid = jnp.array([[True], [False], [True], [False]])
    data = records(np.zeros((4, 1)))
    data.pop("signal")
    data.update(sampling_value=jnp.array([[0.5], [91], [1.5], [93]]),
                reward=jnp.array([[0.0], [1], [0], [4]]))
    graph.init(jax.random.PRNGKey(0), {key: value[0, 0] for key, value in data.items()})
    frozen = graph.update((), data, upd_color=0, valid=valid, alive=jnp.ones_like(valid))
    actual, actual_target = graph.evaluate((), frozen, (advantage, target))
    np.testing.assert_allclose(np.asarray(actual)[valid], [expected_first, 0.5])
    np.testing.assert_allclose(np.asarray(actual_target)[valid], [expected_first + 0.5, 2])


def test_nested_aggregates_are_frozen_and_minibatches_need_only_cached_dependencies():
    graph = DRLGraph()
    resource = model(init=lambda key, x: jnp.array(2.0), apply=lambda weight, x: weight * x)
    with leg.backward(color=0):
        snapshot = resource(graph.env.information_set)
        following = snapshot.aggregate(object="children")
        centered = following - following.aggregate("mean", object="batch")
    graph.metrics = {"centered": centered}
    with leg.backward(color=1):
        value = resource(graph.env.information_set)
        graph.minimize(((value - centered) ** 2).sum(), models=[resource])
    data = {"information_set": jnp.arange(12, dtype=jnp.float32).reshape(3, 2, 2),
            "legal_action_mask": jnp.ones((3, 2, 3), bool)}
    states = graph.init(jax.random.PRNGKey(0), {key: array[0, 0] for key, array in data.items()})
    valid = jnp.ones((3, 2), bool)
    prepared = graph.update((states[0].params,), data, upd_color=0, valid=valid, alive=valid)
    before, = graph.evaluate((jnp.array(2.0),), prepared, (centered,))
    after, = graph.evaluate((jnp.array(200.0),), prepared, (centered,))
    np.testing.assert_array_equal(before, after)
    private = {key: array.reshape((-1,) + array.shape[2:])[jnp.array([1, 4])]
               for key, array in prepared.items() if key.startswith("_drl_node_")}
    minibatch, = graph.evaluate((), private, (centered,))
    np.testing.assert_allclose(minibatch, np.asarray(before).reshape(6, 2)[[1, 4]])
    cache_key = graph._node_key(centered.index)
    gradient = jax.grad(lambda parameter: graph.update(
        (parameter,), data, upd_color=0, valid=valid, alive=valid)[cache_key].sum())(jnp.array(2.0))
    assert float(gradient) == 0
    current_gradient = jax.grad(lambda parameter: graph.evaluate(
        (parameter,), prepared, (graph.objectives[0][0],), upd_color=1)[0].sum())(jnp.array(2.0))
    assert abs(float(current_gradient)) > 0


@pytest.mark.parametrize("options", [
    {"object": "unknown"}, {"aggregator": "median"}, {"player": "opponents"},
    {"object": "segment", "aggregator": "mean"},
    {"object": "descendants", "aggregator": "max"},
    {"discount": -0.1}, {"decay": 1.1}, {"discount": float("nan")},
    {"discount": True}, {"padding": float("inf")}, {"include_self": 1},
    {"include_self": True}, {"decay": 0.5},
    {"object": "segment", "aggregator": "sum", "decay": 0.5},
    {"object": "batch", "discount": 0.5},
])
def test_invalid_aggregate_options_are_rejected(options):
    with pytest.raises(ValueError):
        aggregate(DRLGraph().reward, **options)


def test_aggregate_requires_graph_node_and_static_coefficients():
    with pytest.raises(TypeError):
        aggregate(jnp.ones(3))
    graph = DRLGraph()
    with pytest.raises(ValueError, match="static scalar"):
        aggregate(graph.reward, discount=graph.sampling_value)


def test_local_evaluation_requires_preparation_but_local_inference_still_works():
    graph = DRLGraph()
    result = graph.reward.aggregate()
    graph.metrics = {"result": result}
    graph.init(jax.random.PRNGKey(0), {"reward": jnp.array(1.0)})
    with pytest.raises(ValueError, match="requires prepared"):
        graph.evaluate((), {"reward": jnp.array(1.0)}, (result,))
    value, = graph.evaluate((), {"reward": jnp.array(1.0)}, (graph.reward,))
    assert float(value) == 1


@pytest.mark.parametrize("role", ["strategy", "value", "model"])
def test_sampling_and_model_initialization_cannot_depend_on_aggregates(role):
    graph = DRLGraph()
    result = graph.env.information_set.aggregate()
    if role == "model":
        resource = model(init=lambda key, x: jnp.array(1.0), apply=lambda p, x: p * x)
        resource(result)
    else:
        setattr(graph, role, result)
    with pytest.raises(ValueError, match="cannot depend"):
        graph.init(jax.random.PRNGKey(0), {"information_set": jnp.zeros(2)})


def test_removed_algorithm_and_internal_inputs_do_not_reappear_as_custom_features():
    graph = DRLGraph()
    for name in ("old_log_prob", "old_value", "advantage", "return_target", "valid"):
        with pytest.raises(AttributeError):
            getattr(graph, name)
    assert not graph._custom_inputs
    assert graph.log_sampling_prob.graph is graph
    assert graph.sampling_value.graph is graph
    graph.advantage = graph.reward.aggregate("sum", object="descendants", include_self=True)
    assert graph.advantage.graph is graph


def test_update_with_no_selected_colors_preserves_inputs():
    graph = DRLGraph()
    data = {"reward": jnp.array(1.0)}
    result = graph.update((), data, upd_color=[])
    assert result.keys() == data.keys()
    assert result["reward"] is data["reward"]


@pytest.mark.parametrize("valid, alive", [
    (jnp.ones(2, bool), jnp.ones(2, bool)),
    (jnp.ones((2, 1)), jnp.ones((2, 1), bool)),
    (jnp.ones((2, 1), bool), jnp.ones((3, 1), bool)),
    (jnp.ones((0, 1), bool), jnp.ones((0, 1), bool)),
])
def test_preparation_rejects_malformed_masks(valid, alive):
    graph = DRLGraph()
    graph.metrics = {"result": graph.reward.aggregate()}
    with pytest.raises(ValueError, match="masks"):
        graph.update((), {}, upd_color=0, valid=valid, alive=alive)
