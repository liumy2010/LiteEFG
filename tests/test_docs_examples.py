"""Execute the homepage and introductory guide examples from their Markdown."""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pyspiel
import pytest

import LiteEFG as leg
from docs_examples import ROOT, extract_fenced_blocks


@pytest.fixture(autouse=True)
def isolated_example_files(tmp_path, monkeypatch):
    """Keep the documented caches and CSV exports out of the user's files."""
    expanduser = os.path.expanduser
    monkeypatch.setattr(
        os.path, "expanduser",
        lambda path: str(tmp_path) if path == "~" else expanduser(path),
    )
    monkeypatch.chdir(tmp_path)


def run_block(path, blocks, index, namespace):
    exec(compile(blocks[index], f"{path}:python[{index + 1}]", "exec"), namespace)


def assert_valid_strategy(namespace, selector="default"):
    env = namespace["env"]
    strategy = namespace["algorithm"].current_strategy()
    improvements = np.asarray(env.exploitability(strategy, selector))
    utilities = np.asarray(env.utility(strategy, selector))
    assert improvements.shape == utilities.shape == (2,)
    assert np.isfinite(improvements).all()
    assert np.isfinite(utilities).all()
    assert (improvements >= -1e-10).all()
    assert utilities.sum() == pytest.approx(0, abs=1e-10)
    policy, frames = env.get_strategy(strategy, selector)
    assert_valid_export(policy, frames)


def assert_valid_export(policy, frames):
    assert len(frames) == 2
    for probabilities in [policy.action_probability_array] + [
        frame.drop(columns="Infoset").to_numpy(dtype=float) for frame in frames
    ]:
        assert probabilities.shape == (12, 2) or probabilities.shape == (6, 2)
        assert np.isfinite(probabilities).all()
        assert (probabilities >= -1e-10).all()
        assert (probabilities <= 1 + 1e-10).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-10)


def assert_csv_export(path, expected):
    actual = pd.read_csv(path)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


def test_homepage_cfr_loop():
    path = "docs/index.md"
    blocks = extract_fenced_blocks(path)
    assert len(blocks) == 1, "Add execution coverage for new homepage examples"
    namespace = {}
    run_block(path, blocks, 0, namespace)
    assert np.isfinite(namespace["gaps"]).all()
    assert_valid_strategy(namespace, "avg-iterate")


def test_quick_start_training_and_csv_export(tmp_path):
    path = "docs/guide/quick-start.md"
    blocks = extract_fenced_blocks(path)
    assert len(blocks) == 2, "Add execution coverage for new quick-start examples"
    namespace = {}
    for index in range(len(blocks)):
        run_block(path, blocks, index, namespace)
    assert namespace["iteration"] == 1000
    assert_valid_strategy(namespace, "avg-iterate")
    assert_valid_export(namespace["policy"], namespace["tables"])
    for player, table in enumerate(namespace["tables"]):
        assert_csv_export(tmp_path / f"kuhn_player_{player}.csv", table)
    assert namespace["regrets"]
    assert all(np.isfinite(values).all() for _, values in namespace["regrets"])


def test_algorithm_interface_with_documented_environment_context():
    path = "docs/guide/algorithms.md"
    blocks = extract_fenced_blocks(path)
    assert len(blocks) == 1, "Add execution coverage for new algorithm examples"
    namespace = {"env": leg.OpenSpielEnv(pyspiel.load_game("kuhn_poker"))}
    run_block(path, blocks, 0, namespace)
    assert isinstance(namespace["strategy"], leg.GraphNode)
    assert_valid_strategy(namespace)


def test_experiment_recipes_include_interaction_and_best_policy(tmp_path, monkeypatch):
    path = "docs/guide/examples.md"
    blocks = extract_fenced_blocks(path)
    assert len(blocks) == 7, "Add execution coverage for new experiment recipes"

    current = {}
    run_block(path, blocks, 0, current)
    assert current["iteration"] == 1000
    assert np.isfinite(current["improvements"]).all()
    assert_valid_strategy(current)

    average = {}
    run_block(path, blocks, 1, average)
    assert_valid_strategy(average, "avg-iterate")
    assert_valid_export(average["policy"], average["frames"])
    for player, frame in enumerate(average["frames"]):
        assert_csv_export(tmp_path / f"kuhn-player-{player}.csv", frame)

    # Kuhn poker always permits action 0 (pass/fold), so the actual terminal
    # interaction can run to completion without waiting for a human.
    prompts = []

    def choose_action(prompt):
        prompts.append(prompt)
        return "0"

    monkeypatch.setattr("builtins.input", choose_action)
    run_block(path, blocks, 2, average)
    # Player 0 acts again if the opponent bets after the opening pass.
    assert 3 <= len(prompts) <= 6

    run_block(path, blocks, 3, average)
    run_block(path, blocks, 4, average)
    assert_valid_strategy(average, "best-iterate")
    assert_valid_export(average["best_policy"], average["best_frames"])

    saved = {}
    run_block(path, blocks, 5, saved)
    assert saved["step"] == 50
    assert (tmp_path / "cfr.checkpoint").is_file()
    graph, env = saved["graph"], saved["env"]
    graph.update_graph(env)
    env.update_strategy(graph.current_strategy())
    expected = env.utility(graph.current_strategy(), "avg-iterate")

    # Execute the documented resumption block in a fresh interpreter and check
    # that its 51st update agrees with continuing the existing environment.
    process_environment = os.environ.copy()
    process_environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(ROOT), process_environment.get("PYTHONPATH"))))
    script = blocks[6] + '\nimport json\nprint(json.dumps(env.utility(graph.current_strategy(), "avg-iterate")))\n'
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=tmp_path, env=process_environment,
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == pytest.approx(expected)
