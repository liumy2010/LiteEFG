"""Check balanced weights against hand calculations and Definition 2 of Bai et al."""

from functools import lru_cache
import math
import os

import numpy as np
import pyspiel
import pytest

import LiteEFG as leg
from LiteEFG.baselines import Balanced_OMD


@pytest.fixture
def branching_game(tmp_path):
    # Two P1 roots, several infosets after one action, unequal action counts,
    # an early terminal action, and different maximum depths for the players.
    records = [
        "# Opt {", "#     openspiel,", "#     players: 2,", "# }",
        "node start chance actions root0=0.5 root1=0.5",
        "node root0 player 1 actions split w end0",
        "node split chance actions u=0.25 v=0.75",
        "node root1 player 1 actions z end1",
        "node u player 1 actions p2a end2",
        "node v player 1 actions end3 end4 end5",
        "node w player 1 actions end6 end7",
        "node z player 1 actions end8 end9",
        "node p2a player 2 actions p2b end10",
        "node p2b player 2 actions deep end11",
        "node deep player 1 actions end12 end13",
    ]
    records += ["node end{} leaf payoffs 1=0 2=0".format(i) for i in range(14)]
    records += ["infoset {} nodes {}".format(name, name)
                for name in ("root0", "root1", "u", "v", "w", "z", "deep", "p2a", "p2b")]
    path = tmp_path / "branching.game"
    path.write_text("\n".join(records) + "\n")
    return path


@pytest.mark.parametrize("traversal", ["Enumerate", "Outcome"])
def test_branching_weights_and_one_time_initialization(branching_game, traversal):
    env = leg.FileEnv(str(branching_game), traverse_type=traversal)
    graph = Balanced_OMD.graph()
    env.set_graph(graph)

    class TracedEnvironment:
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            return getattr(env, name)

        def update(self, *args, **kwargs):
            self.calls.append(kwargs)
            return env.update(*args, **kwargs)

    traced = TracedEnvironment()
    expected = {1: {"root0": 1.0, "root1": 1.0, "u": 2 / 3, "v": 2 / 3,
                    "w": 1 / 3, "z": 1.0, "deep": 1.0},
                2: {"p2a": 1.0, "p2b": 1.0}}
    for _ in range(3):
        graph.update_graph(traced)
        for player in (1, 2):
            weights = dict(env.get_value(player, graph.visit_prob))
            assert set(weights) == set(expected[player])
            for name, value in expected[player].items():
                assert weights[name] == pytest.approx([value])
            # Zero-payoff play leaves the initial uniform strategy unchanged.
            for _, probabilities in env.get_value(player, graph.strategy):
                np.testing.assert_allclose(probabilities, 1 / len(probabilities))
            for node in (graph.transition_prob, graph.parent_subtree_size):
                assert all(math.isfinite(value) for _, row in env.get_value(player, node)
                           for value in row)

    initialization = [call for call in traced.calls if call["upd_color"] == [1]]
    assert [call["upd_player"] for call in initialization] == [1, 1, 1, 2, 2]
    assert all(call["traverse_type"] == "Enumerate" for call in initialization)
    assert traced.calls[-2:] == [{"upd_color": [0]}, {"upd_color": [0]}]
    if traversal == "Enumerate":
        action_weights = dict(env.get_value(1, graph.visit_action_prob))
        assert action_weights["u"] == pytest.approx([1 / 3])
        assert action_weights["v"] == pytest.approx([2 / 9])
        assert action_weights["w"] == pytest.approx([1 / 6])


def balanced_reach_from_definition(game):
    """Multiply the ancestor action probabilities in Definition 2 directly.

    This reference uses OpenSpiel's infoset tree and no transition-probability
    recursion, reciprocal conversion, or LiteEFG aggregate operations.
    """
    paths = [{} for _ in range(game.num_players())]
    children = [{} for _ in paths]

    def visit(state, histories):
        if state.is_terminal():
            return
        if state.is_chance_node():
            for action, _ in state.chance_outcomes():
                visit(state.child(action), histories)
            return
        player = state.current_player()
        infoset = state.information_state_string()
        if infoset not in paths[player]:
            paths[player][infoset] = histories[player]
            if histories[player]:
                children[player].setdefault(histories[player][-1], set()).add(infoset)
        for action in state.legal_actions():
            following = list(histories)
            following[player] = histories[player] + ((infoset, action),)
            visit(state.child(action), following)

    visit(game.new_initial_state(), [() for _ in paths])
    weights = []
    for player, player_paths in enumerate(paths):
        descendants = {infoset: [] for infoset in player_paths}
        for (parent, _), child_infosets in children[player].items():
            descendants[parent].extend(child_infosets)

        @lru_cache(None)
        def count(infoset, target_depth):
            return (int(len(player_paths[infoset]) + 1 == target_depth)
                    + sum(count(child, target_depth) for child in descendants[infoset]))

        result = {}
        for infoset, path in player_paths.items():
            target_depth = len(path) + 1
            reach = 1.0
            for ancestor, action in path:
                numerator = sum(count(child, target_depth)
                                for child in children[player][ancestor, action])
                reach *= numerator / count(ancestor, target_depth)
            result[infoset] = reach
        weights.append(result)
    return weights


@pytest.mark.parametrize("game_spec", [
    "kuhn_poker(players=2)",
    "leduc_poker(players=2,suit_isomorphism=True)",
    "goofspiel(num_cards=3,players=2,imp_info=True,points_order=descending)",
])
def test_weights_match_balanced_policy_products(game_spec, tmp_path, monkeypatch):
    expanduser = os.path.expanduser
    monkeypatch.setattr(os.path, "expanduser", lambda path:
                        str(tmp_path) if path == "~" else expanduser(path))
    env = leg.OpenSpielEnv(pyspiel.load_game(game_spec), traverse_type="Outcome", regenerate=True)
    expected = balanced_reach_from_definition(env.game)
    graph = Balanced_OMD.graph()
    env.set_graph(graph)
    graph.update_graph(env)

    for player, expected_weights in enumerate(expected, 1):
        actual = dict(env.get_value(player, graph.visit_prob))
        assert set(actual) == set(expected_weights)
        for infoset, weight in expected_weights.items():
            assert actual[infoset] == pytest.approx([weight], rel=1e-12, abs=1e-14)
