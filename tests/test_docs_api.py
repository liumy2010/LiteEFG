"""Execute the API reference's actual Markdown examples with valid inputs.

API signatures are fragments: the fixtures supply the graph, game, and named
arguments described by their surrounding prose. Expression results are retained
so the native operations are evaluated and checked, not merely constructed.
"""

import ast
from contextlib import redirect_stdout
import io
import math
import os

import numpy as np
import pyspiel
import pytest
from open_spiel.python.algorithms import expected_game_score, exploitability

import LiteEFG as leg
from docs_examples import extract_fenced_blocks


GRAPH_DOC = "docs/guide/computation-graph.md"
ENVIRONMENT_DOC = "docs/guide/environments.md"
FILE_ENV_DOC = "docs/guide/environments/file-env.md"
OPEN_SPIEL_DOC = "docs/guide/environments/open-spiel.md"

# Guide examples precede the API reference on each page and are exercised by
# test_docs_graph_environments.py. Keep the existing API case order below.
GRAPH_BLOCKS = extract_fenced_blocks(GRAPH_DOC)[7:]
SHARED_ENVIRONMENT_BLOCKS = extract_fenced_blocks(ENVIRONMENT_DOC)
FILE_ENV_BLOCKS = extract_fenced_blocks(FILE_ENV_DOC)[1:]
OPEN_SPIEL_BLOCKS = extract_fenced_blocks(OPEN_SPIEL_DOC)[5:]
ENVIRONMENT_EXAMPLES = [
    (FILE_ENV_DOC, FILE_ENV_BLOCKS[0]),
    (OPEN_SPIEL_DOC, OPEN_SPIEL_BLOCKS[0]),
    *((ENVIRONMENT_DOC, block) for block in SHARED_ENVIRONMENT_BLOCKS),
    (FILE_ENV_DOC, FILE_ENV_BLOCKS[1]),
    *((OPEN_SPIEL_DOC, block) for block in OPEN_SPIEL_BLOCKS[1:]),
]


def execute(source, namespace, filename):
    """Preserve top-level expression results without changing their calls."""
    tree = ast.parse(source, filename=filename)
    for index, statement in enumerate(tree.body):
        if isinstance(statement, ast.Expr):
            tree.body[index] = ast.copy_location(
                ast.Expr(value=ast.Call(
                    func=ast.Attribute(value=ast.Name(id="_results", ctx=ast.Load()),
                                       attr="append", ctx=ast.Load()),
                    args=[statement.value], keywords=[])), statement)
    ast.fix_missing_locations(tree)
    namespace["_results"] = []
    exec(compile(tree, filename, "exec"), namespace)
    return namespace["_results"]


@pytest.fixture
def game_file(tmp_path):
    path = tmp_path / "api.game"
    path.write_text("\n".join([
        "# Opt {", "#     openspiel,", "#     players: 2,", "# }",
        "node start player 1 actions left right",
        "node left player 2 actions ll lr",
        "node right player 2 actions rl rr",
        "node ll player 1 actions lla llb",
        "node lla leaf payoffs 1=0 2=0",
        "node llb leaf payoffs 1=0 2=0",
        "node lr leaf payoffs 1=0 2=0",
        "node rl leaf payoffs 1=0 2=0",
        "node rr leaf payoffs 1=0 2=0",
        "infoset p1start nodes start",
        "infoset p1continue nodes ll",
        "infoset p2 nodes left right",
    ]) + "\n")
    return str(path)


def vectors(env, node):
    return [values for player in (1, 2)
            for _, values in env.get_value(player, node)]


def assert_vectors(env, node, expected):
    for values in vectors(env, node):
        assert values == pytest.approx(expected)


def test_api_example_inventory():
    # New snippets must be assigned a valid execution context below.
    assert len(GRAPH_BLOCKS) == 12
    assert len(SHARED_ENVIRONMENT_BLOCKS) == 9
    assert len(FILE_ENV_BLOCKS) == 2
    assert len(OPEN_SPIEL_BLOCKS) == 5
    assert len(ENVIRONMENT_EXAMPLES) == 16


@pytest.mark.parametrize("block", range(12))
def test_graph_api_examples(block, game_file):
    namespace = {"leg": leg}
    if block == 0:
        result, = execute(GRAPH_BLOCKS[block], namespace, GRAPH_DOC)
        assert isinstance(result, leg.Graph)
        return

    graph = leg.Graph()
    with leg.backward(is_static=True):
        x = leg.cat([leg.const(1, 0.25), leg.const(1, 0.75)])
        y = leg.const(1, 0.5)
        target = leg.const(2, 0.0)
        expression = x + 1.0
        strategy = leg.const(graph.action_set_size, 1.0 / graph.action_set_size)
        regrets = leg.cat([leg.const(1, -1.0), leg.const(1, 3.0)])
        mu = leg.cat([leg.const(1, 0.3), leg.const(1, 0.7)])

    namespace.update(self=graph, x=x, y=y, target=target,
                     expression=expression, strategy=strategy, regrets=regrets,
                     size=2, val=0.25, nodes=[x, y], p_norm=1.0,
                     aggregator="sum", distance="L2", gamma=0.2, mu=mu,
                     seed=0, node=graph.action_set_size)

    if block in (1, 4):
        results = execute(GRAPH_BLOCKS[block], namespace, GRAPH_DOC)
        if block == 1:
            dynamic_nodes = []
            for context in results:
                with context:
                    dynamic_nodes.append(x + 1.0)
    else:
        with leg.backward(is_static=block != 11):
            results = execute(GRAPH_BLOCKS[block], namespace, GRAPH_DOC)

    env = leg.FileEnv(game_file)
    env.set_graph(graph)
    env.update(strategy)
    if block == 1:
        for node in dynamic_nodes:
            assert_vectors(env, node, [1.25, 1.75])
    elif block == 2:
        assert results[0] is None
        assert_vectors(env, target, [1.25, 1.75])
        for node in results[1:]:
            assert_vectors(env, node, [0.25, 0.75])
    elif block == 3:
        assert_vectors(env, results[0], [0.25, 0.25])
    elif block == 4:
        assert_vectors(env, namespace["count"], [0.0])
        assert_vectors(env, namespace["strategy"], [0.5, 0.5])
    elif block == 5:
        for node, expected in zip(results, ([0.5, 0.75], [0.25, 0.5],
                                           [0.25, 0.75, 0.5]), strict=True):
            assert_vectors(env, node, expected)
    elif block == 6:
        for node in results:
            assert_vectors(env, node, [0.25, 0.75])
    elif block == 7:
        assert_vectors(env, strategy, [0.0, 1.0])
    elif block == 8:
        rows = dict(env.get_value(1, results[0]))
        assert rows["p1start"] == pytest.approx([1.0, 0.0])
        assert rows["p1continue"] == pytest.approx([0.0, 0.0])
    elif block == 9:
        # x already sums to one and exceeds both the uniform lower bound and
        # gamma * mu = [0.06, 0.14], so each Euclidean projection must fix x.
        assert len(results) == 4
        for node in results:
            assert_vectors(env, node, [0.25, 0.75])
    elif block == 10:
        assert results[0] is None
        for index, node in enumerate(results[1:]):
            for values in vectors(env, node):
                assert len(values) == 2 and all(map(math.isfinite, values))
                if index == 0:
                    assert all(0 <= value < 1 for value in values)
                elif index == 2:
                    assert all(value >= 0 for value in values)
    elif block == 11:
        before = vectors(env, namespace["noise"])
        env.update(strategy)
        after = vectors(env, namespace["noise"])
        assert before != after
        assert all(len(row) == 2 and all(value >= 0 for value in row)
                   for row in after)


class ExampleGraph(leg.Graph):
    """Supply persistent state and a visible dynamic update to API fragments."""

    def __init__(self):
        super().__init__()
        with leg.backward(is_static=True):
            self.strategy = leg.const(self.action_set_size, 1.0 / self.action_set_size)
            self.count = leg.const(1, 0.0)
        with leg.backward():
            self.count.inplace(self.count + 1.0)

    def current_strategy(self):
        return self.strategy


@pytest.mark.parametrize("block", range(16))
def test_environment_api_examples(block, game_file, tmp_path, monkeypatch):
    expanduser = os.path.expanduser
    monkeypatch.setattr(os.path, "expanduser", lambda path:
                        str(tmp_path) if path == "~" else expanduser(path))
    monkeypatch.chdir(tmp_path)
    game = pyspiel.load_game("kuhn_poker")
    env = (leg.FileEnv(game_file) if block == 11
           else leg.OpenSpielEnv(game))
    graph = ExampleGraph()
    if block != 2:
        env.set_graph(graph)
        env.update(graph.strategy)
        env.update_strategy(graph.strategy)

    namespace = dict(leg=leg, file_name=game_file, game=game, env=env, graph=graph,
                     strategy=graph.strategy, strategy_node=graph.strategy,
                     strategies=[graph.strategy, graph.strategy], player=1,
                     node=graph.strategy)
    if block in (9, 10):
        namespace["values"] = [row for _, row in env.get_value(1, graph.strategy)]
    if block == 15:
        namespace["policy"], _ = env.get_strategy(graph.strategy)
        # Action 0 is legal at every Kuhn decision. Exercise the original
        # 1000-game interaction while replacing only terminal input/output.
        monkeypatch.setattr("builtins.input", lambda prompt: "0")

    output = io.StringIO()
    with redirect_stdout(output):
        document, source = ENVIRONMENT_EXAMPLES[block]
        results = execute(source, namespace, document)

    if block in (0, 1):
        assert isinstance(results[0], leg.FileEnv)
        results[0].set_graph(graph)
        assert_vectors(results[0], graph.strategy, [0.5, 0.5])
    elif block == 2:
        assert_vectors(env, graph.count, [0.0])
    elif block == 3:
        assert_vectors(env, graph.count, [3.0])
    elif block == 4:
        assert env.utility(graph.strategy, "avg-iterate") == pytest.approx(
            env.utility(graph.strategy))
    elif block in (5, 6, 7):
        policy, _ = env.get_strategy(graph.strategy)
        if block == 5:
            expected = expected_game_score.policy_value(
                game.new_initial_state(), [policy, policy])
            assert results[0] == pytest.approx(expected)
        else:
            actual = results[0] if block == 6 else namespace["exploitabilities"]
            assert len(actual) == 2 and all(value >= 0 for value in actual)
            assert sum(actual) == pytest.approx(exploitability.nash_conv(game, policy))
    elif block in (8, 11, 14):
        rows = namespace["rows"] if block == 14 else results[0]
        assert rows and all(isinstance(name, str) for name, _ in rows)
        for _, row in rows:
            assert row == pytest.approx([0.5, 0.5])
    elif block in (9, 10):
        assert_vectors(env, graph.strategy, [0.5, 0.5])
    elif block in (12, 13):
        policy, tables = ((namespace["policy"], namespace["tables"])
                          if block == 13 else results[0])
        assert len(tables) == 2 and all("Infoset" in table for table in tables)
        np.testing.assert_allclose(policy.action_probability_array.sum(axis=1), 1.0)
        assert expected_game_score.policy_value(
            game.new_initial_state(), [policy, policy]) == pytest.approx(
                env.utility(graph.strategy))
    elif block == 15:
        assert output.getvalue().count("Average Payoff of Each Player:") == 1000
