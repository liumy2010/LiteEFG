"""Run the graph and environment guides' actual fenced Python examples."""

import contextlib
import io
import os
from pathlib import Path

import numpy as np
import pytest
from open_spiel.python.algorithms import expected_game_score

import LiteEFG as leg
from LiteEFG.baselines import CFR

from docs_examples import extract_fenced_blocks


ROOT = Path(__file__).resolve().parents[1]
GRAPH_DOC = "docs/guide/computation-graph.md"
OPEN_SPIEL_DOC = "docs/guide/environments/open-spiel.md"
FILE_ENV_DOC = "docs/guide/environments/file-env.md"
# The API sections following these examples are exercised by test_docs_api.py.
GRAPH_BLOCKS = extract_fenced_blocks(GRAPH_DOC)[:7]
OPEN_SPIEL_BLOCKS = extract_fenced_blocks(OPEN_SPIEL_DOC)[:5]
FILE_ENV_BLOCKS = extract_fenced_blocks(FILE_ENV_DOC)[:1]


class SnippetGraph(leg.Graph):
    """Python subclass providing instance attributes used by guide fragments."""


def execute(blocks, index, namespace, document):
    exec(compile(blocks[index], f"{document}: Python block {index + 1}", "exec"),
         namespace)


def assert_probabilities(values):
    array = np.asarray(values, dtype=float)
    assert np.isfinite(array).all()
    assert (array >= 0).all()
    np.testing.assert_allclose(array.sum(axis=-1), 1.0, atol=1e-12)


@pytest.fixture
def tree_env(tmp_path):
    """A full tree with two same-player children below the root's left action."""
    path = tmp_path / "guide.game"
    path.write_text("\n".join([
        "# Opt {", "#     num_players: 2,", "# }",
        "node / player 1 actions l r",
        "node /P1:l player 2 actions a b",
        "node /P1:r leaf payoffs 1=0 2=0",
        "node /P1:l/P2:a player 1 actions x y",
        "node /P1:l/P2:b player 1 actions x y",
        "node /P1:l/P2:a/P1:x leaf payoffs 1=1 2=-1",
        "node /P1:l/P2:a/P1:y leaf payoffs 1=-1 2=1",
        "node /P1:l/P2:b/P1:x leaf payoffs 1=2 2=-2",
        "node /P1:l/P2:b/P1:y leaf payoffs 1=-2 2=2",
        "infoset root nodes /",
        "infoset middle nodes /P1:l",
        "infoset left nodes /P1:l/P2:a",
        "infoset right nodes /P1:l/P2:b",
    ]) + "\n")
    return leg.FileEnv(str(path))


def test_guide_snippet_inventory():
    # A newly added example must receive an execution context in this suite.
    assert len(GRAPH_BLOCKS) == 7
    assert len(OPEN_SPIEL_BLOCKS) == 5
    assert len(FILE_ENV_BLOCKS) == 1
    assert len(extract_fenced_blocks(OPEN_SPIEL_DOC, "text")) == 1
    assert len(extract_fenced_blocks(FILE_ENV_DOC, "text")) == 3


def test_graph_guide_cfr_excerpt():
    namespace = {}
    execute(GRAPH_BLOCKS, 0, namespace, GRAPH_DOC)
    graph = namespace["graph"]()
    env = leg.FileEnv(str(ROOT / "LiteEFG/game_instances/kuhn.game"))
    env.set_graph(graph)
    initial_gain = sum(env.exploitability(graph.strategy))
    # The guide explicitly identifies this as the constructor excerpt and
    # describes the omitted update/current_strategy methods immediately below.
    for _ in range(50):
        env.update(graph.strategy)
        env.update_strategy(graph.strategy)
    for player in (1, 2):
        for _, probabilities in env.get_strategy(player, graph.strategy,
                                                  "avg-iterate"):
            assert_probabilities(probabilities)
    utility = env.utility(graph.strategy, "avg-iterate")
    gains = env.exploitability(graph.strategy, "avg-iterate")
    assert np.isfinite(utility).all()
    assert np.isfinite(gains).all()
    assert sum(utility) == pytest.approx(0.0, abs=1e-12)
    assert 0 <= sum(gains) < initial_gain


def test_graph_guide_inplace_fragment(tree_env):
    graph = SnippetGraph()
    with leg.backward(is_static=True):
        graph.strategy = leg.const(graph.action_set_size, 1 / graph.action_set_size)
        graph.regret_buffer = leg.const(graph.action_set_size, 0.0)
        advantage = leg.const(graph.action_set_size, 0.75)
    with leg.backward():
        execute(GRAPH_BLOCKS, 1, {"self": graph, "advantage": advantage}, GRAPH_DOC)
    tree_env.set_graph(graph)
    for _ in range(3):
        tree_env.update(graph.strategy)
    for player in (1, 2):
        for _, values in tree_env.get_value(player, graph.regret_buffer):
            assert values == pytest.approx([2.25, 2.25])


def test_graph_guide_fibonacci_fragment(tree_env):
    graph = leg.Graph()
    with leg.backward(is_static=True):
        strategy = leg.const(graph.action_set_size, 1 / graph.action_set_size)
        a, b = leg.const(1, 0.0), leg.const(1, 1.0)
    with leg.backward():
        execute(GRAPH_BLOCKS, 2, {"a": a, "b": b}, GRAPH_DOC)
    tree_env.set_graph(graph)
    for expected_a, expected_b in [(1, 1), (1, 2), (2, 3), (3, 5), (5, 8), (8, 13)]:
        tree_env.update(strategy)
        for player in (1, 2):
            for _, values in tree_env.get_value(player, a):
                assert values == [expected_a]
            for _, values in tree_env.get_value(player, b):
                assert values == [expected_b]


def test_graph_guide_child_aggregation_fragment(tree_env):
    graph = leg.Graph()
    with leg.backward(is_static=True):
        strategy = leg.const(graph.action_set_size, 1 / graph.action_set_size)
        expectation = leg.const(1, 1.0)
    namespace = {"leg": leg, "expectation": expectation}
    with leg.backward():
        execute(GRAPH_BLOCKS, 3, namespace, GRAPH_DOC)
    tree_env.set_graph(graph)
    tree_env.update(strategy)
    assert dict(tree_env.get_value(1, namespace["continuation"])) == {
        "root": [2.0, 0.0], "left": [0.0, 0.0], "right": [0.0, 0.0],
    }
    assert dict(tree_env.get_value(2, namespace["continuation"])) == {
        "middle": [0.0, 0.0],
    }


def test_graph_guide_parent_depth_fragment(tree_env):
    class DepthGraph(leg.Graph):
        def __init__(self):
            super().__init__()
            execute(GRAPH_BLOCKS, 4, {"leg": leg, "self": self}, GRAPH_DOC)

    graph = DepthGraph()
    tree_env.set_graph(graph)
    assert dict(tree_env.get_value(1, graph.depth)) == {
        "root": [1.0], "left": [2.0], "right": [2.0],
    }
    assert dict(tree_env.get_value(2, graph.depth)) == {"middle": [1.0]}


def test_graph_guide_color_copy_and_update_schedule(tree_env):
    namespace = {"leg": leg}
    execute(GRAPH_BLOCKS, 6, namespace, GRAPH_DOC)

    class ColorGraph(leg.Graph):
        update_graph = namespace["update_graph"]

        def __init__(self):
            super().__init__()
            self.timestep = 0
            self.inner_epoch = 3
            with leg.backward(is_static=True):
                self.u = leg.const(self.action_set_size, 1 / self.action_set_size)
                self.bar_u = self.u.copy()
                increment = leg.cat([leg.const(1, 1.0), leg.const(1, 0.0)])
            with leg.backward(color=0):
                self.u.inplace(leg.normalize(self.u + increment, p_norm=1.0))
            execute(GRAPH_BLOCKS, 5, {"leg": leg, "self": self}, GRAPH_DOC)

    graph = ColorGraph()
    tree_env.set_graph(graph)
    for iteration in range(1, 7):
        graph.update_graph(tree_env)
        refreshed_iteration = iteration - iteration % graph.inner_epoch
        expected_u = [1 - 2 ** (-iteration - 1), 2 ** (-iteration - 1)]
        expected_bar = [1 - 2 ** (-refreshed_iteration - 1),
                        2 ** (-refreshed_iteration - 1)]
        for player in (1, 2):
            for _, values in tree_env.get_value(player, graph.u):
                assert values == pytest.approx(expected_u)
            for _, values in tree_env.get_value(player, graph.bar_u):
                assert values == pytest.approx(expected_bar)


def test_environment_guide_construction_export_and_interaction(tmp_path, monkeypatch):
    expanduser = os.path.expanduser
    monkeypatch.setattr(os.path, "expanduser", lambda path:
                        str(tmp_path) if path == "~" else expanduser(path))
    monkeypatch.chdir(tmp_path)
    namespace = {}
    execute(OPEN_SPIEL_BLOCKS, 0, namespace, OPEN_SPIEL_DOC)
    initial_env = namespace["env"]
    cache_files = list((tmp_path / "game_instances").glob("*.openspiel"))
    assert len(cache_files) == 1
    original_cache = cache_files[0].read_text()
    cache_files[0].write_text("invalid cache; regeneration must replace this\n")
    execute(OPEN_SPIEL_BLOCKS, 1, namespace, OPEN_SPIEL_DOC)
    assert cache_files[0].read_text() == original_cache
    regenerated_env = namespace["env"]
    execute(OPEN_SPIEL_BLOCKS, 2, namespace, OPEN_SPIEL_DOC)
    external_env = namespace["env"]
    # The bundled-file snippet explicitly requires the repository root.
    monkeypatch.chdir(ROOT)
    execute(FILE_ENV_BLOCKS, 0, namespace, FILE_ENV_DOC)
    bundled_env = namespace["env"]
    for env in (initial_env, regenerated_env, external_env, bundled_env):
        graph = CFR.graph()
        env.set_graph(graph)
        for _ in range(10):
            graph.update_graph(env)
            env.update_strategy(graph.current_strategy())
        assert np.isfinite(env.utility(graph.current_strategy())).all()
        assert np.isfinite(env.exploitability(graph.current_strategy())).all()
    # The export section explicitly requires a trained OpenSpiel environment.
    graph = CFR.graph()
    initial_env.set_graph(graph)
    for _ in range(50):
        graph.update_graph(initial_env)
        initial_env.update_strategy(graph.current_strategy())
    namespace.update(env=initial_env, graph=graph)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        execute(OPEN_SPIEL_BLOCKS, 3, namespace, OPEN_SPIEL_DOC)
    policy, tables = namespace["policy"], namespace["tables"]
    assert len(tables) == 2
    assert [len(table) for table in tables] == [6, 6]
    for table in tables:
        assert_probabilities(table.iloc[:, 1:].to_numpy(dtype=float))
    assert_probabilities(policy.action_probability_array)
    expected = expected_game_score.policy_value(
        initial_env.game.new_initial_state(), [policy, policy])
    np.testing.assert_allclose(
        initial_env.utility(graph.current_strategy(), "avg-iterate"),
        expected, atol=1e-12)
    prompts = []

    def choose_action(prompt):
        prompts.append(prompt)
        return "0"  # Pass/fold is legal at every Kuhn decision.

    monkeypatch.setattr("builtins.input", choose_action)
    with contextlib.redirect_stdout(output):
        execute(OPEN_SPIEL_BLOCKS, 4, namespace, OPEN_SPIEL_DOC)
    assert len(prompts) >= 10
    assert output.getvalue().count("Average Payoff of Each Player:") == 10


def test_environment_guide_game_format_excerpts(tmp_path):
    text_blocks = extract_fenced_blocks(FILE_ENV_DOC, "text")
    source = (ROOT / "LiteEFG/game_instances/kuhn.game").read_text()
    # The guide labels these as excerpts; its linked file supplies omitted nodes.
    # Verify each excerpt verbatim and then load the complete described game.
    for block in text_blocks:
        assert block.strip() in source
    path = tmp_path / "documented-kuhn.game"
    path.write_text(source)
    env = leg.FileEnv(str(path))
    graph = CFR.graph()
    env.set_graph(graph)
    graph.update_graph(env)
    for player in (1, 2):
        rows = env.get_strategy(player, graph.current_strategy())
        assert len(rows) == 6
        for _, probabilities in rows:
            assert_probabilities(probabilities)
