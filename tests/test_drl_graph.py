"""Regression checks for infoset-local graph semantics and model resources."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("optax")
nn = pytest.importorskip("flax.linen")

from LiteEFG.drl.graph import DRLGraph, ModelList, Node, model, where


def test_training_schedule_hook_is_optional():
    graph = DRLGraph()
    output = graph.reward + 1
    graph.init(jax.random.key(0), {"reward": jnp.array(0.0)})
    actual, = graph.evaluate((), {"reward": jnp.array([2.0, 3.0])}, (output,))
    np.testing.assert_array_equal(actual, [3.0, 4.0])
    with pytest.raises(NotImplementedError, match="training schedule"):
        graph.train()


@pytest.mark.parametrize("batch_shape", [(), (2,), (5,), (2, 3)])
def test_vector_scalar_formulas_keep_infoset_local_broadcasting(batch_shape):
    # A batch size equal to the feature width can silently produce incorrect
    # results if ordinary [B,D] * [B] broadcasting is used instead of vmap.
    width = 2
    count = int(np.prod(batch_shape)) if batch_shape else 1
    observations = np.arange(1, count * width + 1, dtype=np.float32)
    observations = observations.reshape(batch_shape + (width,))
    multipliers = np.arange(1, count + 1, dtype=np.float32).reshape(batch_shape) * 10
    divisors = np.arange(1, count + 1, dtype=np.float32).reshape(batch_shape)
    graph = DRLGraph()
    formula = ((graph.env.information_set * graph.reward)
               / graph.sampling_value + graph.player + jnp.array([1.0, -1.0]))
    # axis=0 describes the local feature axis, never an outer sample axis.
    feature_sum = formula.sum(axis=0)
    inputs = {
        "information_set": jnp.asarray(observations),
        "reward": jnp.asarray(multipliers),
        "sampling_value": jnp.asarray(divisors),
        "player": jnp.array(3.0),
    }
    actual, reduced = jax.jit(lambda data: graph.evaluate(
        (), data, (formula, feature_sum)))(inputs)
    expected = (observations * multipliers[..., None] / divisors[..., None]
                + 3.0 + np.array([1.0, -1.0], dtype=np.float32))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(reduced, expected.sum(axis=-1))


def test_symbolic_equality_and_inequality_select_per_sample():
    graph = DRLGraph()
    is_first = graph.player == 0
    is_other = graph.player != 0
    assert isinstance(is_first, Node)
    assert isinstance(is_other, Node)
    selected = where(is_first, graph.env.information_set, -graph.env.information_set)
    both = is_first & ~is_other
    observations = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    actual = jax.jit(lambda data: graph.evaluate(
        (), data, (is_first, is_other, selected, both)))({
            "player": jnp.array([0, 1, 0]),
            "information_set": observations,
        })
    np.testing.assert_array_equal(actual[0], [True, False, True])
    np.testing.assert_array_equal(actual[1], [False, True, False])
    np.testing.assert_array_equal(actual[2], [[1, 2], [-3, -4], [5, 6]])
    np.testing.assert_array_equal(actual[3], [True, False, True])
    with pytest.raises(TypeError, match="symbolic"):
        bool(is_first)


def test_constant_and_scalar_seat_outputs_receive_observation_batch_axes():
    graph = DRLGraph()
    constant = graph.as_node(jnp.array([2.0, 3.0]))
    seat = graph.player + 1
    actual = jax.jit(lambda inputs: graph.evaluate((), inputs, (constant, seat)))({
        "information_set": jnp.zeros((2, 3, 7)), "player": jnp.array(1),
    })
    np.testing.assert_array_equal(actual[0], np.broadcast_to([2, 3], (2, 3, 2)))
    np.testing.assert_array_equal(actual[1], np.full((2, 3), 2))
    constant_only, = graph.evaluate((), {"information_set": jnp.zeros((5, 7))}, (constant,))
    np.testing.assert_array_equal(constant_only, np.broadcast_to([2, 3], (5, 2)))


@pytest.mark.parametrize("mutation", ["inplace", "minimize", "model_minimize"])
def test_initialization_seals_existing_nodes_and_objectives(mutation):
    graph = DRLGraph()
    resource = model(nn.Dense(features=1))
    prediction = resource(graph.env.information_set).squeeze()
    original = prediction ** 2
    replacement = original + 1
    graph.minimize(original, models=[resource])
    graph.init(jax.random.key(0), {"information_set": jnp.zeros(2)})
    original_index = original.index
    original_objectives = len(graph.objectives)
    with pytest.raises(RuntimeError):
        if mutation == "inplace":
            original.inplace(replacement)
        elif mutation == "minimize":
            graph.minimize(replacement, models=[resource])
        else:
            resource.minimize(replacement)
    assert original.index == original_index
    assert len(graph.objectives) == original_objectives


def test_jitted_local_inplace_values_do_not_persist_between_executions():
    graph = DRLGraph()
    temporary = graph.env.information_set.copy()
    before_write = temporary.sum()
    temporary.inplace(temporary + graph.reward)
    after_first_write = temporary.sum()
    temporary.inplace(temporary * 2)
    after_second_write = temporary.sum()
    graph.init(jax.random.key(0), {})
    expression_count = len(graph._expressions)
    execute = jax.jit(lambda observations, reward: graph.evaluate(
        (), {"information_set": observations, "reward": reward},
        (before_write, after_first_write, after_second_write),
    ))
    first = (jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), jnp.array([2.0, 3.0]))
    second = (jnp.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]), jnp.array([-1.0, -2.0]))
    for observations, reward in (first, second, first):
        original, updated, doubled = execute(observations, reward)
        expected = np.asarray(observations) + np.asarray(reward)[:, None]
        np.testing.assert_array_equal(original, np.asarray(observations).sum(axis=-1))
        np.testing.assert_array_equal(updated, expected.sum(axis=-1))
        np.testing.assert_array_equal(doubled, 2 * expected.sum(axis=-1))
    assert len(graph._expressions) == expression_count


def test_rejected_model_registration_does_not_change_sealed_graph():
    graph = DRLGraph()
    graph.init(jax.random.key(0), {})
    resource = model(nn.Sequential([nn.Dense(features=4), nn.tanh, nn.Dense(features=1)]))
    with pytest.raises(RuntimeError, match="initialization"):
        resource(graph.env.information_set)
    assert graph.models == []
    assert resource.state is None
    assert not hasattr(resource, "graph")
    assert graph.init(jax.random.key(0), {}) == ()


def test_multiple_model_calls_share_one_parameter_and_optimizer_resource():
    graph = DRLGraph()
    resource = model(nn.Dense(features=2), name="shared")
    first = resource(graph.env.information_set)
    second = resource(graph.env.information_set + 1)
    joint_loss = (first + second).sum()
    graph.minimize(joint_loss, models=[resource])
    inputs = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    states = graph.init(jax.random.key(0), {"information_set": inputs[0]})
    assert len(graph.models) == 1
    assert graph.models[0] is resource
    assert len(states) == 1
    parameters = tuple(state.params for state in states)
    first_value, second_value = jax.jit(lambda params: graph.evaluate(
        params, {"information_set": inputs}, (first, second)))(parameters)
    np.testing.assert_allclose(first_value, resource.module.apply(parameters[0], inputs),
                               rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(second_value, resource.module.apply(parameters[0], inputs + 1),
                               rtol=1e-6, atol=1e-6)

    def graph_loss(params):
        loss, = graph.evaluate(params, {"information_set": inputs}, (joint_loss,))
        return loss.sum()

    def explicit_shared_loss(params):
        return (resource.module.apply(params[0], inputs).sum()
                + resource.module.apply(params[0], inputs + 1).sum())

    actual_gradient = jax.jit(jax.grad(graph_loss))(parameters)
    expected_gradient = jax.grad(explicit_shared_loss)(parameters)
    for actual, expected in zip(jax.tree.leaves(actual_gradient),
                                jax.tree.leaves(expected_gradient)):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_model_list_routes_different_architectures_and_accumulates_repeated_identity_gradients():
    first = model(nn.Dense(2))
    second = model(nn.Sequential([nn.Dense(5), nn.tanh, nn.Dense(2)]))
    choices = ModelList([first, second, first])
    assert choices[0] is choices[2] is first
    assert choices[1] is second
    graph = DRLGraph()
    prediction = choices[graph.player](graph.env.information_set)
    graph.minimize(prediction.sum(), models=[first, second])
    states = graph.init(jax.random.key(31), {
        "information_set": jnp.ones(3), "player": jnp.array(0, jnp.int32),
    })
    assert len(graph.models) == len(states) == 2
    assert graph.models[graph.model_index(first)] is first
    assert graph.models[graph.model_index(second)] is second
    assert first.state is states[graph.model_index(first)]
    assert second.state is states[graph.model_index(second)]
    params = tuple(state.params for state in states)
    observations = jnp.arange(12, dtype=jnp.float32).reshape(4, 3) / 10
    seats = jnp.array([0, 1, 2, 1], jnp.int32)
    batch = {"information_set": observations, "player": seats}

    def routed(parameters):
        return graph.evaluate(parameters, batch, (prediction,))[0]

    def explicit(parameters):
        return jnp.stack([
            choices[seat].apply(parameters[graph.model_index(choices[seat])], observations[index])
            for index, seat in enumerate((0, 1, 2, 1))
        ])

    np.testing.assert_allclose(jax.jit(routed)(params), explicit(params), rtol=1e-6, atol=1e-6)
    actual = jax.jit(jax.grad(lambda parameters: routed(parameters).sum()))(params)
    expected = jax.grad(lambda parameters: explicit(parameters).sum())(params)
    for observed, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(observed, reference, rtol=1e-6, atol=1e-6)

    # A frozen policy may play another seat while retaining its source head.
    # Participation must use exactly the same override as the forward route.
    frozen_batch = dict(batch, _model_player=jnp.array(0, jnp.int32))
    usage = graph.model_usage(params, frozen_batch)
    np.testing.assert_array_equal(usage[graph.model_index(first)], True)
    np.testing.assert_array_equal(usage[graph.model_index(second)], False)
    predicted, = graph.evaluate(params, frozen_batch, (prediction,))
    np.testing.assert_allclose(predicted, first.apply(first.state.params, observations),
                               rtol=1e-6, atol=1e-6)


def test_model_list_checks_every_branch_output_before_committing_initialization():
    first, second = model(nn.Dense(2)), model(nn.Dense(3))
    graph = DRLGraph()
    ModelList([first, second])[graph.player](graph.env.information_set)
    with pytest.raises((TypeError, ValueError), match="shape|output|branch"):
        graph.init(jax.random.key(0), {
            "information_set": jnp.zeros(4), "player": jnp.array(0, jnp.int32),
        })
    assert first.state is second.state is None
    assert not graph._sealed


def test_every_call_to_a_model_must_match_its_local_input_shape():
    resource = model(nn.Dense(1))
    graph = DRLGraph()
    resource(graph.env.information_set)
    resource(graph.env.information_set[:2])
    with pytest.raises(ValueError, match="shape|input"):
        graph.init(jax.random.key(0), {"information_set": jnp.zeros(3)})
    assert resource.state is None


@pytest.mark.parametrize("bad_input", [jnp.zeros(2), jnp.zeros(3, jnp.int32)])
def test_cross_graph_input_mismatch_does_not_publish_any_new_model_state(bad_input):
    shared = model(nn.Dense(1))
    first = DRLGraph()
    shared(first.env.information_set)
    first.init(jax.random.key(0), {"information_set": jnp.zeros(3)})
    original_state = shared.state

    pending = model(nn.Dense(1))
    second = DRLGraph()
    pending(second.env.information_set)
    shared(second.env.information_set)
    with pytest.raises(ValueError, match="shape|dtype|input"):
        second.init(jax.random.key(1), {"information_set": bad_input})
    assert shared.state is original_state
    assert pending.state is None
    assert not second._sealed
    # A failed bind must also leave local specifications available for repair.
    states = second.init(jax.random.key(2), {"information_set": jnp.zeros(3)})
    assert states[second.model_index(shared)] is original_state
    assert states[second.model_index(pending)] is pending.state


def test_model_minimize_uses_the_losses_graph_after_resource_reuse():
    shared = model(nn.Dense(1))
    first, second = DRLGraph(), DRLGraph()
    first_loss = shared(first.env.information_set).sum()
    second_loss = shared(second.env.information_set).sum()
    shared.minimize(first_loss)
    shared.minimize(second_loss)
    assert first.objectives[0][1] == (first.model_index(shared),)
    assert second.objectives[0][1] == (second.model_index(shared),)
    for removed in ("graph", "index", "input_index", "shared"):
        assert not hasattr(shared, removed)


@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
def test_custom_matrix_and_scalar_inputs_keep_local_axis_semantics(batch_shape):
    graph = DRLGraph()
    matrix = graph.env.full_info_feature
    assert graph.env.full_info_feature is matrix
    formula = matrix * graph.env.temperature + graph.env.information_set.sum()
    column_sums = formula.sum(axis=0)
    graph.init(jax.random.key(0), {
        "full_info_feature": jnp.zeros((2, 3), jnp.float32),
        "temperature": jnp.array(0.0, jnp.float32),
        "information_set": jnp.zeros(4),
    })
    assert graph.input_specs["full_info_feature"] == jax.ShapeDtypeStruct((2, 3), jnp.float32)
    assert graph.input_specs["temperature"] == jax.ShapeDtypeStruct((), jnp.float32)
    count = int(np.prod(batch_shape)) if batch_shape else 1
    matrices = np.arange(count * 6, dtype=np.float32).reshape(batch_shape + (2, 3))
    temperatures = np.arange(1, count + 1, dtype=np.float32).reshape(batch_shape)
    observations = np.arange(count * 4, dtype=np.float32).reshape(batch_shape + (4,))
    actual, reduced = jax.jit(lambda inputs: graph.evaluate((), inputs, (formula, column_sums)))({
        "full_info_feature": matrices,
        "temperature": temperatures,
        "information_set": observations,
    })
    expected = matrices * temperatures[..., None, None] + observations.sum(-1)[..., None, None]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(reduced, expected.sum(axis=-2))


def test_local_custom_inputs_broadcast_to_observation_sample_dimensions():
    graph = DRLGraph()
    formula = graph.env.full_info_feature * graph.env.temperature
    matrix = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
    examples = {"full_info_feature": matrix, "temperature": jnp.array(2.0)}
    graph.init(jax.random.key(0), examples)
    actual, = jax.jit(lambda inputs: graph.evaluate((), inputs, (formula,)))({
        **examples, "information_set": jnp.zeros((4, 5, 7)),
    })
    np.testing.assert_array_equal(actual, np.broadcast_to(np.asarray(matrix) * 2, (4, 5, 2, 3)))


def test_custom_input_specs_bind_before_models_and_reinitialization_checks_them():
    graph = DRLGraph()
    received_shapes = []

    def initialize(key, example):
        received_shapes.append(example.shape)
        return jnp.ones_like(example)

    resource = model(init=initialize, apply=lambda params, values: params * values)
    prediction = resource(graph.env.full_info_feature.reshape(6))
    examples = {"full_info_feature": jnp.ones((2, 3), jnp.float32)}
    for _ in range(2):
        states = graph.init(jax.random.key(0), examples)
        result, = graph.evaluate(tuple(state.params for state in states), examples, (prediction,))
        np.testing.assert_array_equal(result, np.ones(6))
    assert received_shapes == [(6,)]
    for bad in (jnp.zeros((3, 2), jnp.float32), jnp.zeros((2, 3), jnp.int32)):
        with pytest.raises(ValueError, match="full_info_feature.*retain shape.*dtype"):
            graph.init(jax.random.key(0), {"full_info_feature": bad})
    assert received_shapes == [(6,)]


def test_missing_custom_names_fail_before_model_initialization():
    graph = DRLGraph()
    called = []
    resource = model(init=lambda key, value: called.append(value) or value,
                     apply=lambda params, value: params + value)
    resource(graph.env.information_set)
    # Even an unused declaration must match an environment feature. This turns
    # a misspelled attribute into an explicit binding error at initialization.
    graph.env.full_info_featre
    with pytest.raises(ValueError, match="Missing example graph inputs: full_info_featre"):
        graph.init(jax.random.key(0), {"information_set": jnp.zeros(3),
                                     "full_info_feature": jnp.zeros((2, 3))})
    assert called == []
    assert graph.input_specs == {}


def test_custom_evaluation_rejects_unbound_missing_shape_and_dtype_errors():
    graph = DRLGraph()
    feature = graph.env.full_info_feature
    with pytest.raises(ValueError, match="full_info_feature.*call graph.init"):
        graph.evaluate((), {"full_info_feature": jnp.zeros((2, 3))}, (feature,))
    graph.init(jax.random.key(0), {"full_info_feature": jnp.zeros((2, 3))})
    with pytest.raises(ValueError, match="Missing graph inputs: full_info_feature"):
        graph.evaluate((), {}, (feature,))
    for bad in (jnp.zeros(3), jnp.zeros((2, 4)), jnp.zeros((4, 3, 2))):
        with pytest.raises(ValueError, match="full_info_feature.*local shape"):
            graph.evaluate((), {"full_info_feature": bad}, (feature,))
    with pytest.raises(ValueError, match="full_info_feature.*dtype"):
        graph.evaluate((), {"full_info_feature": jnp.zeros((2, 3), jnp.int32)}, (feature,))
    with pytest.raises(ValueError, match="inconsistent sample dimensions"):
        graph.evaluate((), {"full_info_feature": jnp.zeros((4, 2, 3)),
                           "information_set": jnp.zeros((5, 7))}, (feature,))


def test_reserved_private_and_sealed_attributes_never_declare_environment_inputs():
    graph = DRLGraph()
    for name in ("_unknown", "strategy", "value"):
        with pytest.raises(AttributeError):
            getattr(graph, name)
    with pytest.raises(AttributeError):
        graph.current_strategy()
    with pytest.raises(AttributeError):
        graph.current_value()
    assert graph._custom_inputs == {}
    assert callable(graph.evaluate)
    feature = graph.env.full_info_feature
    graph.init(jax.random.key(0), {"full_info_feature": jnp.array(1.0)})
    assert graph.env.full_info_feature is feature
    with pytest.raises(AttributeError):
        graph.env.other_feature
    assert set(graph.input_specs) == {"full_info_feature"}


@pytest.mark.parametrize("name", ["information_set", "full_info_feature", "misspelled_model"])
def test_top_level_feature_access_fails_without_implicitly_creating_inputs(name):
    graph = DRLGraph()
    before = len(graph._expressions)
    with pytest.raises(AttributeError, match=name):
        getattr(graph, name)
    assert name not in graph.__dict__
    assert name not in graph._custom_inputs
    assert len(graph._expressions) == before
    feature = getattr(graph.env, name)
    assert getattr(graph.env, name) is feature
    assert name not in graph.__dict__


@pytest.mark.parametrize("name", ["reward", "action", "legal_action_mask", "information_state", "_private"])
def test_environment_namespace_rejects_reserved_inputs_and_private_attributes(name):
    graph = DRLGraph()
    before = len(graph._expressions)
    with pytest.raises(AttributeError, match=name):
        getattr(graph.env, name)
    assert name not in graph._custom_inputs
    assert len(graph._expressions) == before


def test_environment_namespace_cannot_be_replaced_or_mutated():
    graph = DRLGraph()
    namespace = graph.env
    feature = namespace.full_info_feature
    for operation in (
        lambda: setattr(graph, "env", {}),
        lambda: delattr(graph, "env"),
        lambda: setattr(namespace, "full_info_feature", graph.reward),
        lambda: delattr(namespace, "full_info_feature"),
    ):
        with pytest.raises(AttributeError):
            operation()
    assert graph.env is namespace
    assert graph.env.full_info_feature is feature


@pytest.mark.parametrize("name", [
    "old_log_prob", "old_value", "advantage", "return_target", "valid", "information_state",
])
def test_removed_training_fields_cannot_be_implicitly_declared_as_custom_inputs(name):
    graph = DRLGraph()
    expression_count = len(graph._expressions)
    with pytest.raises(AttributeError, match=name):
        getattr(graph, name)
    assert name not in graph.__dict__
    assert name not in graph._custom_inputs
    assert len(graph._expressions) == expression_count


def test_algorithm_can_explicitly_name_its_derived_advantage_and_target():
    graph = DRLGraph()
    graph.advantage = graph.reward - graph.sampling_value
    graph.return_target = graph.advantage + graph.sampling_value
    graph.init(jax.random.key(0), {})
    actual = graph.evaluate((), {
        "reward": jnp.array([2.0, 4.0]),
        "sampling_value": jnp.array([0.5, 1.0]),
    }, (graph.advantage, graph.return_target))
    np.testing.assert_array_equal(actual[0], [1.5, 3.0])
    np.testing.assert_array_equal(actual[1], [2.0, 4.0])
    assert graph._custom_inputs == {}


def test_initialization_requires_a_single_local_example_record():
    graph = DRLGraph()
    with pytest.raises(ValueError, match="information_set.*without sample dimensions"):
        graph.init(jax.random.key(0), {"information_set": jnp.zeros((2, 3))})


def test_standard_information_set_binds_shape_and_dtype_like_other_features():
    graph = DRLGraph()
    result = graph.env.information_set.sum()
    graph.init(jax.random.key(0), {"information_set": jnp.zeros(3, jnp.int32)})
    assert graph.input_specs["information_set"] == jax.ShapeDtypeStruct((3,), jnp.int32)
    actual, = graph.evaluate((), {"information_set": jnp.arange(12, dtype=jnp.int32).reshape(4, 3)},
                            (result,))
    np.testing.assert_array_equal(actual, [3, 12, 21, 30])
    with pytest.raises(ValueError, match="information_set.*local shape"):
        graph.evaluate((), {"information_set": jnp.zeros((4, 2), jnp.int32)}, (result,))
    with pytest.raises(ValueError, match="information_set.*dtype"):
        graph.evaluate((), {"information_set": jnp.zeros((4, 3), jnp.float32)}, (result,))
