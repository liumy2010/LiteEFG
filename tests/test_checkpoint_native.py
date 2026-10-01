"""Execute restored operations, including parameters absent from baselines."""

import pickle

import pytest

import LiteEFG as leg
from LiteEFG import _LiteEFG as native


def _operations(source, other):
    return {
        "copy": lambda: source.copy(),
        "add": lambda: source + other,
        "subtract": lambda: source - other,
        "multiply": lambda: source * other,
        "divide": lambda: source / other,
        "exp": lambda: source.exp(),
        "log": lambda: source.log(),
        "sum": lambda: source.sum(),
        "mean": lambda: source.mean(),
        "max": lambda: source.max(),
        "min": lambda: source.min(),
        "dot": lambda: source.dot(other),
        "argmax": lambda: source.argmax(),
        "argmin": lambda: source.argmin(),
        "maximum": lambda: leg.maximum(source, other),
        "minimum": lambda: leg.minimum(source, other),
        "euclidean": lambda: source.euclidean(),
        "entropy": lambda: source.negative_entropy(),
        "shifted_entropy": lambda: source.negative_entropy(shifted=True),
        "normalize": lambda: source.normalize(3.25),
        "normalize_ignore_negative": lambda: (source - 0.25).normalize(1, True),
        "greater_than": lambda: source > 0.25,
        "greater_equal": lambda: source >= 0.25,
        "less_than": lambda: source < 0.25,
        "less_equal": lambda: source <= 0.25,
        "equal": lambda: source == 0.3,
        "pow": lambda: source ** 2.5,
        "concat": lambda: leg.cat([source, other]),
        "dynamic_const": lambda: leg.const(3, source.sum()),
        "aggregate_sum": lambda: leg.aggregate(source, "sum", padding=0.75),
        "aggregate_mean": lambda: leg.aggregate(source, "mean", padding=0.75),
        "aggregate_max": lambda: leg.aggregate(source, "max", padding=-0.75),
        "aggregate_min": lambda: leg.aggregate(source, "min", padding=0.5),
        "aggregate_parent": lambda: leg.aggregate(
            source.sum(), "sum", object="parent", player="opponents", padding=0.125),
        "project_l2": lambda: source.project("L2", 0.2, other),
        "project_kl": lambda: source.project("KL", 0.3, other),
        "uniform": lambda: native._uniform(leg.const(1, 3), lower=-1.5, upper=2.25),
        "normal": lambda: native._normal(leg.const(1, 3), mean=0.25, stddev=1.75),
        "exponential": lambda: native._exponential(leg.const(1, 3), lambda_=2.75),
    }


# Names are independent of a live graph so test collection does not mutate the
# native process-wide graph builder or its RNG.
OPERATION_NAMES = tuple(_operations(None, None))


@pytest.fixture
def checkpoint_game(tmp_path):
    path = tmp_path / "checkpoint-operations.game"
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
    return path


@pytest.mark.parametrize("operation_name", OPERATION_NAMES)
def test_pickle_preserves_operation_parameters_and_execution(checkpoint_game, operation_name):
    graph = leg.Graph()
    with leg.backward(is_static=True):
        source = leg.cat([leg.const(1, value) for value in (0.2, 0.3, 0.5)])
        other = leg.cat([leg.const(1, value) for value in (0.4, 0.1, 0.5)])
        graph.result = _operations(source, other)[operation_name]()

    random_state = native._checkpoint_get_random_state()
    try:
        # Pickle an uninitialized graph, then initialize the two copies in
        # separate environments. No graph constructor or operation is rerun
        # while loading, and random operations retain nondefault parameters.
        restored = pickle.loads(pickle.dumps(graph))
        assert native._checkpoint_get_random_state() == random_state
        original_env = leg.FileEnv(str(checkpoint_game))
        original_env.set_graph(graph)
        expected = [original_env.get_value(player, graph.result) for player in (1, 2)]

        native._checkpoint_set_random_state(random_state)
        restored_env = leg.FileEnv(str(checkpoint_game))
        restored_env.set_graph(restored)
        actual = [restored_env.get_value(player, restored.result) for player in (1, 2)]
        assert actual == expected
    finally:
        native._checkpoint_set_random_state(random_state)
