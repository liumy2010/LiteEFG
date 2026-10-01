"""Procedural Dark Hex rules and private perfect-recall observations."""

from pathlib import Path
import json
import os
import random
import shlex
import subprocess

import pytest
import pyspiel

import LiteEFG as leg


SOURCE = Path(__file__).resolve().parents[1] / "examples" / "cpp_games" / "dark_hex.cpp"


@pytest.fixture(scope="module")
def dark_hex_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("dark-hex-cpp-cache")


@pytest.fixture(scope="module")
def dark_hex_envs(dark_hex_cache):
    return {
        size: leg.CppEnv(SOURCE, {"board_size": size}, cache_dir=dark_hex_cache)
        for size in (2, 3)
    }


def _compare(env, state, history, observation_to_key, key_to_observation, parents, seen_parents):
    actual = env.inspect(history)
    expected_player = -1 if state.is_terminal() else state.current_player() + 1
    assert actual["current_player"] == expected_player
    assert actual["legal_actions"] == state.legal_actions()
    assert actual["returns"] == state.returns()
    if state.is_terminal():
        return None
    player = state.current_player()
    key = (player, actual["infoset"])
    private_history = (
        player, state.observation_string(player),
        tuple(event.action for event in state.full_history() if event.player == player),
    )
    assert observation_to_key.setdefault(private_history, key) == key
    assert key_to_observation.setdefault(key, private_history) == private_history
    # Every information set has exactly one preceding own decision sequence.
    assert seen_parents.setdefault(key, parents[player]) == parents[player]
    return key


def test_dark_hex_exhaustive_2x2_matches_open_spiel(dark_hex_envs):
    env = dark_hex_envs[2]
    game = pyspiel.load_game("dark_hex(board_size=2)")
    observation_to_key, key_to_observation, seen_parents = {}, {}, {}
    visits = 0

    def visit(state, history, parents):
        nonlocal visits
        visits += 1
        key = _compare(env, state, history, observation_to_key, key_to_observation, parents, seen_parents)
        if state.is_terminal():
            return
        player = state.current_player()
        for action in state.legal_actions():
            next_parents = list(parents)
            next_parents[player] = (key, action)
            visit(state.child(action), history + [action], next_parents)

    visit(game.new_initial_state(), [], [None, None])
    assert visits > 400
    assert env.stats()["stored_nodes"] == 0


def test_dark_hex_3x3_random_histories_match_open_spiel(dark_hex_envs):
    env = dark_hex_envs[3]
    game = pyspiel.load_game("dark_hex(board_size=3)")
    rng = random.Random(0)
    observation_to_key, key_to_observation, seen_parents = {}, {}, {}
    winners = set()
    for _ in range(1000):
        state = game.new_initial_state()
        history, parents = [], [None, None]
        while True:
            key = _compare(env, state, history, observation_to_key, key_to_observation, parents, seen_parents)
            if state.is_terminal():
                winners.add(tuple(state.returns()))
                break
            player = state.current_player()
            action = rng.choice(state.legal_actions())
            parents[player] = (key, action)
            state.apply_action(action)
            history.append(action)
    assert winners == {(1.0, -1.0), (-1.0, 1.0)}
    assert env.stats()["stored_nodes"] == 0


def test_dark_hex_hides_opponent_failures_and_remembers_own_history(dark_hex_envs):
    env = dark_hex_envs[3]
    # Player 2's extra collision at cell 0 cannot be observed by player 1.
    assert env.inspect([0, 1])["infoset"] == env.inspect([0, 0, 1])["infoset"]
    # Equal visible boards do not erase the order of a player's own attempts.
    assert env.inspect([0, 1, 2, 3])["infoset"] != env.inspect([2, 1, 0, 3])["infoset"]
    collision = env.inspect([0, 0])
    assert collision["current_player"] == 2
    assert 0 not in collision["legal_actions"]
    assert len(collision["legal_actions"]) == 8


@pytest.mark.parametrize("parameters", [{"board_size": 1}, {"board_size": 5}, {"gameversion": "adh"}])
def test_dark_hex_rejects_unsupported_parameters(parameters, dark_hex_cache):
    with pytest.raises((ValueError, RuntimeError), match="Dark Hex parameters"):
        leg.CppEnv(SOURCE, parameters, cache_dir=dark_hex_cache)


def test_dark_hex_structural_census_matches_full_2x2_tree(tmp_path):
    source = SOURCE.parent.parent / "dark_hex_size.cpp"
    executable = tmp_path / "dark_hex_size"
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    subprocess.run(compiler + ["-O3", "-std=c++17", str(source), "-o", str(executable)], check=True)
    actual = json.loads(subprocess.check_output([str(executable), "2"], text=True))
    infosets = [set(), set()]

    def visit(state):
        if state.is_terminal():
            return 1
        player = state.current_player()
        infosets[player].add((
            state.observation_string(player),
            tuple(event.action for event in state.full_history() if event.player == player),
        ))
        return 1 + sum(visit(state.child(action)) for action in state.legal_actions())

    nodes = visit(pyspiel.load_game("dark_hex(board_size=2)").new_initial_state())
    assert actual["nodes"] == nodes
    assert actual["infosets_per_player"] == [len(player_infosets) for player_infosets in infosets]
    assert actual["information_sets"] == sum(len(player_infosets) for player_infosets in infosets)
    assert actual["game_file_written"] is False
