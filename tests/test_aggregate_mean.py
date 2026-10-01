"""Check infoset means against hand-computed values, including empty inputs."""

import pytest

import LiteEFG as leg


@pytest.fixture
def aggregation_env(tmp_path):
    # Each continuing root action reaches two infosets of unequal vector sizes.
    records = [
        "# Opt {", "#     openspiel,", "#     players: 2,", "# }",
        "node root player 1 actions self_split opponent_split end0",
        "node self_split chance actions self_two=0.25 self_three=0.75",
        "node opponent_split chance actions opponent_two=0.25 opponent_three=0.75",
        "node self_two player 1 actions end1 end2",
        "node self_three player 1 actions end3 end4 end5",
        "node opponent_two player 2 actions end6 end7",
        "node opponent_three player 2 actions end8 end9 end10",
    ]
    records += [f"node end{i} leaf payoffs 1=0 2=0" for i in range(11)]
    records += [f"infoset {name} nodes {name}" for name in (
        "root", "self_two", "self_three", "opponent_two", "opponent_three")]
    path = tmp_path / "aggregate-mean.game"
    path.write_text("\n".join(records) + "\n")
    return leg.FileEnv(str(path))


@pytest.mark.parametrize("player,is_static", [
    ("self", True), ("self", False), ("opponents", False),
])
def test_child_mean_reduces_concatenated_elements(aggregation_env, player, is_static):
    graph = leg.Graph()
    with leg.backward(is_static=True):
        strategy = leg.const(graph.action_set_size, 1 / graph.action_set_size)
        source = leg.const(graph.action_set_size, graph.action_set_size)
    with leg.backward(is_static=is_static):
        result = leg.aggregate(source, "mean", player=player, padding=-7.0)
    aggregation_env.set_graph(graph)
    if not is_static:
        aggregation_env.update(strategy)

    # [2, 2] concatenated with [3, 3, 3] has mean 13/5, not the mean
    # of the two infoset means (5/2), nor a chance-weighted mean (11/4).
    root_expected = [13 / 5, -7.0, -7.0] if player == "self" else [-7.0, 13 / 5, -7.0]
    expected = {
        1: {"root": root_expected, "self_two": [-7.0] * 2, "self_three": [-7.0] * 3},
        2: {"opponent_two": [-7.0] * 2, "opponent_three": [-7.0] * 3},
    }
    for owner in (1, 2):
        actual = dict(aggregation_env.get_value(owner, result))
        assert actual.keys() == expected[owner].keys()
        for name, values in expected[owner].items():
            assert actual[name] == pytest.approx(values)


@pytest.mark.parametrize("player", ["self", "opponents"])
def test_child_mean_uses_padding_for_empty_source_vectors(aggregation_env, player):
    graph = leg.Graph()
    with leg.backward(is_static=True):
        strategy = leg.const(graph.action_set_size, 1 / graph.action_set_size)
        source = leg.const(0, 0.0)
    with leg.backward():
        result = leg.aggregate(source, "mean", player=player, padding=4.5)
    aggregation_env.set_graph(graph)
    aggregation_env.update(strategy)

    assert dict(aggregation_env.get_value(1, result)) == {
        "root": [4.5] * 3, "self_two": [4.5] * 2, "self_three": [4.5] * 3,
    }
    assert dict(aggregation_env.get_value(2, result)) == {
        "opponent_two": [4.5] * 2, "opponent_three": [4.5] * 3,
    }


@pytest.mark.parametrize("scalar", [True, False], ids=["scalar", "action-vector"])
def test_parent_mean_reads_scalar_or_selected_action(aggregation_env, scalar):
    graph = leg.Graph()
    with leg.forward(is_static=True):
        source = (leg.const(1, 7.0) if scalar else
                  leg.cat([leg.const(1, 10.0), leg.const(graph.action_set_size - 1, 20.0)]))
        result = leg.aggregate(source, "mean", object="parent", padding=-3.0)
    aggregation_env.set_graph(graph)

    # Both P1 descendants follow the first action: select 10, not the mean
    # of the root's full vector [10, 20, 20]. P2 has no same-player parent.
    expected = 7.0 if scalar else 10.0
    assert dict(aggregation_env.get_value(1, result)) == {
        "root": [-3.0], "self_two": [expected], "self_three": [expected],
    }
    assert dict(aggregation_env.get_value(2, result)) == {
        "opponent_two": [-3.0], "opponent_three": [-3.0],
    }
