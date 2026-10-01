"""Independent exact-payoff, response, and graph checks for implicit C++ games."""

from itertools import product
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import LiteEFG as leg
from LiteEFG.baselines import CFR, OS_MCCFR

from test_traversal_override import chance_game


SOURCE = Path(__file__).parent / "cpp_games" / "test_games.cpp"


@pytest.fixture(scope="module", autouse=True)
def isolated_cpp_cache(tmp_path_factory):
    cache = tmp_path_factory.mktemp("implicit-game-cache")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("XDG_CACHE_HOME", str(cache))
        yield cache
    assert not list(cache.rglob("*.game"))


def make_env(mode="hidden", traverse="Enumerate"):
    return leg.CppEnv(SOURCE, parameters={"mode": mode}, traverse_type=traverse)


def uniform_graph():
    graph = leg.Graph()
    with leg.backward(is_static=True):
        strategy = leg.const(graph.action_set_size, 1.0 / graph.action_set_size)
    return graph, strategy


def named(entries):
    return {key.decode("ascii") if isinstance(key, bytes) else key: values for key, values in entries}


def test_exact_response_cannot_see_hidden_chance():
    env = make_env()
    graph, strategy = uniform_graph()
    env.set_graph(graph)
    assert env.utility(strategy) == pytest.approx([0.0, 0.0])
    # The hidden bit is 1 with probability .75. The best legal guess earns .5;
    # selecting an action separately at each hidden state would incorrectly earn 1.
    assert env.exploitability(strategy) == pytest.approx([0.5, 0.0])
    assert named(env.get_strategy(1, strategy)) == {"guess": [0.5, 0.5]}


def test_negative_payoffs_without_player_decisions():
    env = make_env("no_decision")
    graph, strategy = uniform_graph()
    env.set_graph(graph)
    metrics = env.evaluate(strategy)
    assert metrics["utility"] == pytest.approx([-3.5, -4.0])
    assert metrics["deviation_gain"] == pytest.approx([0.0, 0.0])
    assert metrics["nash_conv"] == 0.0
    assert env.stats()["information_sets"] == 0


def test_inspection_uses_action_ids_and_requests_returns_only_at_terminals():
    env = make_env()
    initial = env.inspect([])
    assert initial["current_player"] == 0
    assert initial["chance_outcomes"] == [(2, 0.25), (5, 0.75)]
    decision = env.inspect([2])
    assert decision["current_player"] == 1
    assert decision["legal_actions"] == [10, 20]
    assert decision["infoset"] == b"guess"
    terminal = env.inspect([2, 10])
    assert terminal["current_player"] == -1
    assert terminal["returns"] == [1.0, -1.0]


def normal_form_payoff(a, b, c):
    return np.asarray([
        2*a - b + 3*c - 4*a*c + a*b,
        -a + 2*b - c + a*b - 3*b*c,
        a - 2*b + c + 2*a*c + b*c,
    ], dtype=float)


def normal_form_metrics(strategies):
    """Enumerate action profiles, then each pure unilateral deviation."""
    utility = np.zeros(3)
    action_values = [np.zeros(len(strategy)) for strategy in strategies]
    for actions in product(*(range(len(strategy)) for strategy in strategies)):
        payoff = normal_form_payoff(*actions)
        probabilities = [strategies[i][actions[i]] for i in range(3)]
        utility += np.prod(probabilities) * payoff
        for player in range(3):
            opponent_weight = np.prod([probabilities[i] for i in range(3) if i != player])
            action_values[player][actions[player]] += opponent_weight * payoff[player]
    return utility, np.asarray([max(values) for values in action_values]) - utility


@pytest.mark.parametrize("traverse", ["Enumerate", "External"])
def test_three_player_general_sum_exact_metrics_against_normal_form(traverse):
    leg.set_seed(0)
    env = make_env("normal_form3", traverse)
    graph = CFR.graph()
    env.set_graph(graph)
    strategy = graph.current_strategy()
    for iteration in range(26):
        if iteration:
            graph.update_graph(env)
            env.update_strategy(strategy)
        if iteration not in (0, 1, 4, 25):
            continue
        selectors = ["default"] if not iteration else [
            "default", "last-iterate", "avg-iterate", "linear-avg-iterate"]
        for selector in selectors:
            # Exact utility also discovers infosets not sampled during training.
            actual_utility = env.utility(strategy, selector)
            policies = [named(env.get_strategy(player, strategy, selector))[f"p{player}"]
                        for player in (1, 2, 3)]
            expected_utility, expected_gain = normal_form_metrics(policies)
            assert actual_utility == pytest.approx(expected_utility, abs=1e-11)
            assert env.exploitability(strategy, selector) == pytest.approx(expected_gain, abs=1e-11)


def test_exact_metrics_accept_individual_player_strategies():
    env = make_env("normal_form3")
    graph = leg.Graph()
    policies = [[0.8, 0.2], [0.1, 0.3, 0.6], [0.4, 0.6]]
    with leg.backward(is_static=True):
        strategies = [leg.cat([leg.const(1, probability) for probability in policy])
                      for policy in policies]
    env.set_graph(graph)
    expected_utility, expected_gain = normal_form_metrics(policies)
    assert env.utility(strategies) == pytest.approx(expected_utility)
    assert env.exploitability(strategies) == pytest.approx(expected_gain)


def test_enumerate_update_matches_independent_counterfactual_values():
    env = make_env("matrix", "External")
    graph = leg.Graph()
    with leg.backward(is_static=True):
        strategy = leg.const(graph.action_set_size, 1.0 / graph.action_set_size)
        observed = leg.const(graph.action_set_size, 0.0)
    with leg.backward():
        observed.inplace(graph.utility.copy())
    env.set_graph(graph)
    env.update(strategy, traverse_type="Enumerate")
    assert named(env.get_value(1, observed))["p1"] == pytest.approx([4.0, 3.0])
    assert named(env.get_value(2, observed))["p2"] == pytest.approx([-0.5, -6.5])
    assert env.utility(strategy) == pytest.approx([3.5, -3.5])
    assert env.exploitability(strategy) == pytest.approx([0.5, 3.0])


@pytest.mark.parametrize("traverse", ["Enumerate", "External", "Outcome"])
def test_graph_updates_and_exact_metrics_match_tiny_explicit_fixture(chance_game, traverse):
    implicit = make_env("matrix", traverse)
    explicit = leg.FileEnv(str(chance_game), traverse_type=traverse)
    graph = OS_MCCFR.graph(balanced=False) if traverse == "Outcome" else CFR.graph()
    implicit.set_graph(graph)
    explicit.set_graph(graph)
    strategy = graph.current_strategy()
    leg.set_seed(0)
    for iteration in range(1, 26):
        random_state = leg._LiteEFG._checkpoint_get_random_state()
        for env in (implicit, explicit):
            leg._LiteEFG._checkpoint_set_random_state(random_state)
            graph.update_graph(env)
            env.update_strategy(strategy)
        for player in (1, 2):
            for node in (strategy, graph.regret_buffer):
                actual = named(implicit.get_value(player, node))
                expected = named(explicit.get_value(player, node))
                assert actual.keys() == expected.keys()
                for key in actual:
                    assert actual[key] == pytest.approx(expected[key], abs=1e-11)
        if iteration in (1, 4, 25):
            for selector in ("default", "avg-iterate", "linear-avg-iterate"):
                assert implicit.utility(strategy, selector) == pytest.approx(
                    explicit.utility(strategy, selector), abs=1e-11)
                assert implicit.exploitability(strategy, selector) == pytest.approx(
                    explicit.exploitability(strategy, selector), abs=1e-11)


def test_cfr_descendant_aggregation_and_realization_average():
    env = make_env("sequential")
    graph = CFR.graph()
    strategy = graph.current_strategy()
    env.set_graph(graph)
    assert env.utility(strategy) == pytest.approx([3.5, -3.5])
    assert env.exploitability(strategy) == pytest.approx([4.0, 0.0])

    graph.update_graph(env)
    env.update_strategy(strategy)
    policies = named(env.get_strategy(1, strategy))
    assert policies["root"] == pytest.approx([1.0, 0.0])
    assert policies["left"] == pytest.approx([0.0, 1.0])
    assert policies["right"] == pytest.approx([0.0, 1.0])
    assert env.utility(strategy) == pytest.approx([5.5, -5.5])

    graph.update_graph(env)
    env.update_strategy(strategy)
    assert named(env.get_strategy(1, strategy))["root"] == pytest.approx([0.25, 0.75])
    assert env.utility(strategy) == pytest.approx([7.0, -7.0])
    assert env.utility(strategy, "avg-iterate") == pytest.approx([6.25, -6.25])
    assert env.exploitability(strategy, "avg-iterate") == pytest.approx([1.25, 0.0])
    assert env.utility(strategy, "linear-avg-iterate") == pytest.approx([6.5, -6.5])


def test_average_strategies_weight_descendants_by_own_reach():
    env = make_env("sequential")
    graph = leg.Graph()
    with leg.backward(is_static=True):
        strategy = leg.cat([leg.const(1, 0.8), leg.const(1, 0.2)])
    with leg.backward():
        strategy.inplace(1.0 - strategy)
    env.set_graph(graph)
    for _ in range(2):
        env.update(strategy)
        env.update_strategy(strategy)

    average = named(env.get_strategy(1, strategy, "avg-iterate"))
    assert average["root"] == pytest.approx([0.5, 0.5])
    assert average["left"] == pytest.approx([0.68, 0.32])
    assert average["right"] == pytest.approx([0.32, 0.68])
    assert env.utility(strategy, "avg-iterate") == pytest.approx([4.04, -4.04])
    linear = named(env.get_strategy(1, strategy, "linear-avg-iterate"))
    assert linear["root"] == pytest.approx([0.6, 0.4])
    assert linear["left"] == pytest.approx([11 / 15, 4 / 15])
    assert linear["right"] == pytest.approx([0.4, 0.6])
    assert env.utility(strategy, "linear-avg-iterate") == pytest.approx([3.54, -3.54])


def test_late_discovered_infoset_preserves_all_prior_average_reach():
    env = make_env("sequential", "Outcome")
    graph = leg.Graph()
    with leg.backward(is_static=True):
        strategy = leg.cat([leg.const(1, 0.8), leg.const(1, 0.2)])
    with leg.backward():
        strategy.inplace(1.0 - strategy)
    env.set_graph(graph)
    for _ in range(2):
        # Linux/libstdc++ seed 0 selects the first root action at both .8 and .2.
        leg.set_seed(0)
        env.update(strategy, upd_player=1)
        env.update_strategy(strategy)
    assert set(named(env.get_strategy(1, strategy))) == {"root", "left"}
    assert env.utility(strategy, "avg-iterate") == pytest.approx([1.88, -1.88])
    assert named(env.get_strategy(1, strategy, "avg-iterate"))["right"] == pytest.approx([0.8, 0.2])
    assert env.utility(strategy, "linear-avg-iterate") == pytest.approx([2.1, -2.1])


@pytest.mark.parametrize("direction", [leg.backward, leg.forward])
def test_static_parent_values_do_not_depend_on_discovery_time(direction):
    graph = leg.Graph()
    with direction(is_static=True):
        strategy = leg.cat([leg.const(1, 0.8), leg.const(1, 0.2)])
        counter = leg.const(1, 1.0)
        inherited = leg.aggregate(counter, "sum", object="parent", padding=0.0)
    with leg.backward():
        counter.inplace(counter + 1.0)

    for eager in (True, False):
        env = make_env("sequential", "Outcome")
        env.set_graph(graph)
        if eager:
            env.utility(strategy)
        for _ in range(2):
            leg.set_seed(0)
            env.update(strategy, upd_player=1)
        if not eager:
            assert set(named(env.get_value(1, counter))) == {"root", "left"}
        env.utility(strategy)
        assert named(env.get_value(1, counter))["root"] == [3.0]
        assert named(env.get_value(1, inherited)) == {
            "root": [0.0], "left": [1.0], "right": [1.0],
        }


def test_sampled_cfr_learns_biased_hidden_bit():
    leg.set_seed(0)
    env = make_env(traverse="External")
    graph = CFR.graph()
    env.set_graph(graph)
    strategy = graph.current_strategy()
    for _ in range(2000):
        graph.update_graph(env)
        env.update_strategy(strategy)
    assert env.utility(strategy, "avg-iterate")[0] > 0.45
    assert env.exploitability(strategy, "avg-iterate")[0] < 0.05


@pytest.mark.parametrize("mode,reason", [
    ("invalid_actions", "action"), ("invalid_recall", "recall|parent|sequence"),
])
def test_rejects_inconsistent_infosets(mode, reason):
    env = make_env(mode)
    graph, strategy = uniform_graph()
    env.set_graph(graph)
    with pytest.raises((ValueError, RuntimeError), match=reason):
        env.exploitability(strategy)


def test_streamed_exact_evaluation_of_many_nodes_with_one_infoset():
    env = make_env("deep_chance", "External")
    graph, strategy = uniform_graph()
    env.set_graph(graph)
    assert env.stats()["stored_nodes"] == 0
    assert env.stats()["information_sets"] == 0
    assert env.exploitability(strategy) == pytest.approx([0.0, 0.0], abs=1e-12)
    assert len(env.get_strategy(1, strategy)) == 1
    assert env.get_strategy(2, strategy) == []
    stats = env.stats()
    assert stats["stored_nodes"] == 0
    assert stats["information_sets"] == 1
    assert stats["evaluation_nodes"] >= 262143
    assert stats["peak_depth"] <= 18


def test_dark_hex_runner_evaluates_at_interval_and_final_iteration(tmp_path):
    report_path = tmp_path / "benchmark.json"
    result = subprocess.run([
        sys.executable, str(SOURCE.parents[2] / "examples/dark_hex.py"),
        "--board-size", "2", "--iterations", "3", "--eval-every", "2",
        "--output", str(report_path),
    ], capture_output=True, text=True, timeout=90, check=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert json.loads(result.stdout) == report
    assert [checkpoint["iteration"] for checkpoint in report["checkpoints"]] == [2, 3]
    assert report["strategy"] == "default"
    assert len(report["engine_binary_sha256"]) == 64
    assert len(report["baseline_source_sha256"]) == 64
    for checkpoint in report["checkpoints"]:
        assert checkpoint["evaluation_seconds"] > 0
        assert checkpoint["exploitability"] >= -1e-12
        assert checkpoint["stats"]["stored_nodes"] == 0
    assert not list(tmp_path.rglob("*.game"))
