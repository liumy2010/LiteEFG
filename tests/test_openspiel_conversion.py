"""Regression coverage for converting OpenSpiel games to tabular environments."""

import os

import numpy as np
import pyspiel
import pytest
from open_spiel.python.algorithms import expected_game_score

import LiteEFG as leg
from LiteEFG.baselines import CFR


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    expanduser = os.path.expanduser
    monkeypatch.setattr(os.path, "expanduser", lambda path:
                        str(tmp_path) if path == "~" else expanduser(path))
    return tmp_path / "game_instances"


class ConversionGame(pyspiel.Game):
    """A tiny tree with distinct information states and non-round numbers."""

    def __init__(self, *, whitespace=False, fail=False):
        game_type = pyspiel.GameType(
            short_name="conversion_test", long_name="Conversion test",
            dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
            chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
            information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
            utility=pyspiel.GameType.Utility.ZERO_SUM,
            reward_model=pyspiel.GameType.RewardModel.TERMINAL,
            max_num_players=2, min_num_players=2,
            provides_information_state_string=True,
            provides_information_state_tensor=False,
            provides_observation_string=False,
            provides_observation_tensor=False)
        game_info = pyspiel.GameInfo(
            num_distinct_actions=2, max_chance_outcomes=2, num_players=2,
            min_utility=-2.0, max_utility=2.0, utility_sum=0.0,
            max_game_length=3)
        super().__init__(game_type, game_info, {})
        self.whitespace = whitespace
        self.fail = fail

    def new_initial_state(self):
        return ConversionState(self)


class ConversionState(pyspiel.State):
    def __init__(self, game):
        super().__init__(game)
        self.actions = []

    def current_player(self):
        return [pyspiel.PlayerId.CHANCE, 0, 1, pyspiel.PlayerId.TERMINAL][len(self.actions)]

    def _legal_actions(self, player):
        return [0, 1]

    def chance_outcomes(self):
        return [(0, 0.123456789123), (1, 0.876543210877)]

    def _apply_action(self, action):
        self.actions.append(action)

    def _action_to_string(self, player, action):
        return str(action)

    def is_terminal(self):
        return len(self.actions) == 3

    def returns(self):
        if self.get_game().fail:
            raise RuntimeError("conversion interrupted")
        payoff = [0.123456789123, 1.876543210877][self.actions[0]]
        return [payoff, -payoff]

    def information_state_string(self, player=None):
        player = self.current_player() if player is None else player
        if player == 0:
            return ["state one", "state_one"][self.actions[0]]
        return "player two"

    def serialize(self):
        prefix = "state =\t" if self.get_game().whitespace else "state/"
        return prefix + "/".join(map(str, self.actions))

    def __str__(self):
        return self.serialize()


def test_conversion_preserves_numeric_precision(isolated_cache):
    env = leg.OpenSpielEnv(ConversionGame(), regenerate=True)
    graph = CFR.graph()
    env.set_graph(graph)
    expected = (0.123456789123 ** 2 + 0.876543210877 * 1.876543210877)
    np.testing.assert_allclose(env.utility(graph.current_strategy()),
                               [expected, -expected], rtol=0, atol=1e-14)


def test_distinct_information_states_export_separately(isolated_cache):
    env = leg.OpenSpielEnv(ConversionGame(), regenerate=True)
    graph = CFR.graph()
    env.set_graph(graph)
    node = graph.current_strategy()
    rows = env.get_value(1, node)
    assert {name for name, _ in rows} == {"state one", "state_one"}
    env.set_value(1, node, [[1.0, 0.0], [0.0, 1.0]])
    policy, tables = env.get_strategy(node)
    for (name, _), expected in zip(rows, ([1.0, 0.0], [0.0, 1.0])):
        np.testing.assert_array_equal(
            policy.action_probability_array[policy.state_lookup[name]], expected)
    assert set(tables[0]["Infoset"]) == {"state one", "state_one"}


def test_state_serialization_is_safe_for_game_file_tokens(isolated_cache):
    env = leg.OpenSpielEnv(ConversionGame(whitespace=True), regenerate=True)
    graph = CFR.graph()
    env.set_graph(graph)
    assert len(env.get_value(1, graph.current_strategy())) == 2


def test_failed_conversion_preserves_existing_cache(isolated_cache):
    leg.OpenSpielEnv(ConversionGame(), regenerate=True)
    cache_file, = isolated_cache.glob("*.openspiel")
    original = cache_file.read_bytes()
    with pytest.raises(RuntimeError, match="conversion interrupted"):
        leg.OpenSpielEnv(ConversionGame(fail=True), regenerate=True)
    assert cache_file.read_bytes() == original
    assert list(isolated_cache.glob("*.tmp")) == []


def test_legacy_cache_is_upgraded_and_then_reused(isolated_cache, monkeypatch):
    leg.OpenSpielEnv(ConversionGame(), regenerate=True)
    cache_file, = isolated_cache.glob("*.openspiel")
    current = cache_file.read_bytes()
    cache_file.write_bytes(b"# legacy partial cache\n")
    leg.OpenSpielEnv(ConversionGame())
    assert cache_file.read_bytes() == current

    def unexpected_conversion(self, file):
        pytest.fail("An up-to-date cache should be reused")

    monkeypatch.setattr(leg.OpenSpielEnv, "_write_game", unexpected_conversion)
    leg.OpenSpielEnv(ConversionGame())


def test_failed_first_conversion_does_not_publish_cache(isolated_cache):
    with pytest.raises(RuntimeError, match="conversion interrupted"):
        leg.OpenSpielEnv(ConversionGame(fail=True))
    assert list(isolated_cache.iterdir()) == []


def test_long_simultaneous_game_cache_names_are_distinct_and_reused(isolated_cache, monkeypatch):
    games = [pyspiel.load_game(
        f"goofspiel(num_cards={cards},players=2,imp_info=True,points_order=descending)")
        for cards in (2, 3)]
    for game in games:
        env = leg.OpenSpielEnv(game)
        graph = CFR.graph()
        env.set_graph(graph)
        policy, _ = env.get_strategy(graph.current_strategy())
        expected = expected_game_score.policy_value(
            env.game.new_initial_state(), [policy] * env.game.num_players())
        np.testing.assert_allclose(env.utility(graph.current_strategy()), expected,
                                   rtol=0, atol=1e-14)
    cache_files = list(isolated_cache.glob("*.openspiel"))
    assert len(cache_files) == 2
    assert all(len(path.name.encode("utf-8")) <= 255 for path in cache_files)

    def unexpected_conversion(self, file):
        pytest.fail("A long game name should resolve to the same cache on reload")

    monkeypatch.setattr(leg.OpenSpielEnv, "_write_game", unexpected_conversion)
    for game in games:
        leg.OpenSpielEnv(game)


def test_tabular_game_without_observation_tensor(tmp_path, monkeypatch):
    game = pyspiel.load_game("coordinated_mp")
    assert not game.get_type().provides_observation_tensor

    expanduser = os.path.expanduser
    monkeypatch.setattr(os.path, "expanduser", lambda path:
                        str(tmp_path) if path == "~" else expanduser(path))
    env = leg.OpenSpielEnv(game, regenerate=True)
    graph = CFR.graph()
    env.set_graph(graph)
    graph.update_graph(env)

    strategy = graph.current_strategy()
    policy, _ = env.get_strategy(strategy)
    expected = expected_game_score.policy_value(
        env.game.new_initial_state(), [policy] * env.game.num_players())
    np.testing.assert_allclose(env.utility(strategy), expected, atol=1e-10)
