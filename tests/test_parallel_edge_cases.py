"""Preserve serial traversal semantics when infoset histories are irregular."""

import pytest

import LiteEFG as leg
from LiteEFG import _LiteEFG as native


@pytest.fixture(autouse=True)
def restore_runtime_state():
    threads = leg.get_threads()
    random_state = native._checkpoint_get_random_state()
    try:
        yield
    finally:
        leg.set_threads(threads)
        native._checkpoint_set_random_state(random_state)


def write_game(tmp_path, name, records):
    path = tmp_path / name
    header = ["# Opt {", "#     openspiel,", "#     players: 2,", "# }"]
    path.write_text("\n".join(header + records) + "\n")
    return path


@pytest.fixture
def mixed_depth_game(tmp_path):
    # These three children share one P1 parent/action but occur at different
    # history depths. FileEnv visits them in the order a, c, b; deep_b belongs
    # to the next P1 layer. Root and b have one action, c/deep_b two, a three.
    records = [
        "node root player 1 actions split",
        "node split chance actions a=0.25 wait_b=0.5 wait_c=0.25",
        "node a player 1 actions end0 end1 end2",
        "node wait_b chance actions wait_b2=1",
        "node wait_c chance actions c=1",
        "node wait_b2 chance actions b=1",
        "node c player 1 actions end3 end4",
        "node b player 1 actions deep_b",
        "node deep_b player 1 actions end5 end6",
    ]
    records += [f"node end{i} leaf payoffs 1={i} 2={-i}" for i in range(7)]
    records += [f"infoset {name} nodes {name}" for name in
                ("root", "a", "b", "c", "deep_b")]
    return write_game(tmp_path, "mixed-depth.game", records)


@pytest.mark.parametrize("threads", [1, 2, 4])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_child_gather_preserves_historical_floating_point_order(
        mixed_depth_game, threads, reduction):
    leg.set_threads(threads)
    graph = leg.Graph()
    with leg.backward(is_static=True):
        strategy = leg.const(graph.action_set_size, 1 / graph.action_set_size)
        source = ((graph.action_set_size == 3) * 1e16
                  + (graph.action_set_size == 2) * -1e16
                  + (graph.action_set_size == 1))
        static_backward = leg.aggregate(source, reduction)
    with leg.forward(is_static=True):
        static_forward = leg.aggregate(source, reduction)
    with leg.backward():
        backward = leg.aggregate(source, reduction)
    with leg.forward():
        forward = leg.aggregate(source, reduction)
    env = leg.FileEnv(str(mixed_depth_game))
    env.set_graph(graph)

    # Static gathers reverse the infoset order in both directions. Dynamic
    # backward also reverses it, but dynamic forward follows traversal order.
    # (1 - 1e16) + 1e16 = 0; (1e16 - 1e16) + 1 = 1 exactly.
    assert dict(env.get_value(1, static_backward))["root"] == [0.0]
    assert dict(env.get_value(1, static_forward))["root"] == [0.0]
    for _ in range(3):
        env.update(strategy)
        assert dict(env.get_value(1, backward))["root"] == [0.0]
        expected = 1.0 if reduction == "sum" else 1.0 / 3.0
        assert dict(env.get_value(1, forward))["root"] == [expected]


def test_opponent_dependency_cycle_keeps_legacy_update_order(tmp_path):
    # Each player can precede the other in a different hidden chance branch.
    # Both infosets have own-player depth one, yet opponent aggregation forms
    # a cycle: they cannot be updated as independent same-depth infosets.
    path = write_game(tmp_path, "opponent-cycle.game", [
        "node start chance actions a=0.5 b=0.5",
        "node a player 1 actions a2",
        "node b player 2 actions b2",
        "node a2 player 2 actions end0",
        "node b2 player 1 actions end1",
        "node end0 leaf payoffs 1=0 2=0",
        "node end1 leaf payoffs 1=0 2=0",
        "infoset p1 nodes a b2",
        "infoset p2 nodes b a2",
    ])

    def run(threads):
        leg.set_threads(threads)
        graph = leg.Graph()
        with leg.backward(is_static=True):
            strategy = leg.const(1, 1.0)
            state = leg.const(1, 1.0)
        with leg.backward():
            children = leg.aggregate(state, "sum", player="opponents")
            state.inplace(state + children.sum() + 1)
        env = leg.FileEnv(str(path))
        env.set_graph(graph)
        snapshots = []
        for _ in range(5):
            env.update(strategy)
            snapshots.append([env.get_value(player, state) for player in (1, 2)])
        return snapshots

    expected = run(1)
    assert expected[0] == [[("p1", [4.0])], [("p2", [2.0])]]
    assert run(2) == expected
    assert run(4) == expected


def test_opponent_multiple_parent_sequences_keep_last_parent_selection(tmp_path):
    path = write_game(tmp_path, "opponent-multiple-parents.game", [
        "node start chance actions a=0.5 b=0.5",
        "node a player 1 actions c0",
        "node b player 1 actions c1",
        "node c0 player 2 actions d0 end0",
        "node c1 player 2 actions d1 end1",
        "node d0 player 2 actions end2",
        "node d1 player 2 actions end3",
        "node end0 leaf payoffs 1=0 2=0",
        "node end1 leaf payoffs 1=0 2=0",
        "node end2 leaf payoffs 1=0 2=0",
        "node end3 leaf payoffs 1=0 2=0",
        "infoset a nodes a",
        "infoset b nodes b",
        "infoset c nodes c0 c1",
        "infoset d nodes d0 d1",
    ])

    def run(threads):
        leg.set_threads(threads)
        graph = leg.Graph()
        with leg.backward(is_static=True):
            strategy = leg.const(graph.action_set_size, 1 / graph.action_set_size)
            source = leg.const(1, 0.0)
        with leg.backward():
            children = leg.aggregate(source, "sum", player="opponents")
        with leg.forward():
            parent = leg.aggregate(source, "sum", object="parent",
                                   player="opponents", padding=-1.0)
        env = leg.FileEnv(str(path))
        env.set_graph(graph)
        env.set_value(1, source, [[3.0], [7.0]])
        env.set_value(2, source, [[2.0], [5.0]])
        env.update(strategy)
        return env.get_value(1, children), env.get_value(2, parent)

    expected = run(1)
    assert dict(expected[0]) == {"a": [7.0], "b": [7.0]}
    assert dict(expected[1]) == {"c": [-1.0], "d": [7.0]}
    assert run(2) == expected
    assert run(4) == expected


def test_multiple_self_parents_preserve_accepted_game_file_behavior(tmp_path):
    # The file reader historically accepts imperfect-recall infosets. Their
    # parent_sequences contains both a and b, although parent stores only b.
    # Such an infoset cannot use the ordinary tree-of-infosets gather plan.
    path = write_game(tmp_path, "multiple-self-parents.game", [
        "node start chance actions a=0.5 b=0.5",
        "node a player 1 actions c0",
        "node b player 1 actions c1",
        "node c0 player 1 actions end0",
        "node c1 player 1 actions end1",
        "node end0 leaf payoffs 1=0 2=0",
        "node end1 leaf payoffs 1=0 2=0",
        "infoset a nodes a",
        "infoset b nodes b",
        "infoset c nodes c0 c1",
    ])

    def run(threads):
        leg.set_threads(threads)
        graph = leg.Graph()
        with leg.backward(is_static=True):
            strategy = leg.const(1, 1.0)
            source = leg.const(1, 2.0)
        with leg.backward():
            children = leg.aggregate(source, "sum")
        env = leg.FileEnv(str(path))
        env.set_graph(graph)
        env.update(strategy)
        return env.get_value(1, children)

    expected = run(1)
    assert dict(expected) == {"a": [2.0], "b": [2.0], "c": [0.0]}
    assert run(2) == expected
    assert run(4) == expected


@pytest.mark.parametrize("distribution", ["uniform", "normal", "exponential"])
@pytest.mark.parametrize("width_mode", ["fixed", "inplace", "derived"])
def test_random_draws_preserve_values_widths_and_complete_rng_state(
        mixed_depth_game, distribution, width_mode):
    initial_random_state = native._checkpoint_get_random_state()

    def run(threads):
        leg.set_threads(threads)
        native._checkpoint_set_random_state(initial_random_state)
        graph = leg.Graph()
        with leg.backward(is_static=True):
            strategy = leg.const(graph.action_set_size, 1 / graph.action_set_size)
            width = leg.const(1, 1.0)
        with leg.backward():
            # Aggregators allocate one helper per action, so the position of
            # the random operation differs across these infosets (1/2/3 actions).
            leg.aggregate(graph.action_set_size, "sum")
            if width_mode == "inplace":
                # Its idx aliases a prior static node, so checking only the
                # defining node's status misses this same-phase mutation.
                width.inplace(width + 1)
            elif width_mode == "derived":
                width = graph.action_set_size + 1
            else:
                width = graph.action_set_size
            random_node = getattr(native, "_" + distribution)(width)
            observed = random_node.copy()
        env = leg.FileEnv(str(mixed_depth_game))
        env.set_graph(graph)
        snapshots = []
        for step in range(4):
            env.update(strategy)
            values = env.get_value(1, observed)
            if width_mode == "inplace":
                assert all(len(value) == step + 2 for _, value in values)
            snapshots.append(values)
        return snapshots, native._checkpoint_get_random_state()

    expected = run(1)
    assert run(2) == expected
    assert run(4) == expected
