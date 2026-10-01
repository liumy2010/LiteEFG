"""Checkpoint identity, complete training state and fresh-process resumption."""

import importlib
import json
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import LiteEFG as leg
from LiteEFG import checkpoint


@pytest.fixture(autouse=True)
def current_baseline_modules():
    # Reference output checks select baseline packages by replacing sys.modules.
    # Resolve classes when each test runs so pickle sees their active module,
    # even after another suite reimports the same workspace sources.
    global CFR, CMD, FTPL
    CFR, CMD, FTPL = (
        importlib.import_module("LiteEFG.baselines." + name)
        for name in ("CFR", "CMD", "FTPL"))


class NoiseGraph(leg.Graph):
    """One normal draw per player update leaves a cached normal variate."""

    def __init__(self):
        super().__init__()
        with leg.backward(is_static=True):
            self.strategy = leg.const(self.action_set_size, 1.0 / self.action_set_size)
            self.one = leg.const(1, 1.0)
            self.noise = leg.const(1, 0.0)
        with leg.backward():
            self.noise.inplace(
                leg.random.normal(self.one)
                + leg.random.uniform(self.one)
                + leg.random.exponential(self.one))


@pytest.fixture
def game_file(tmp_path):
    path = tmp_path / "matrix.game"
    path.write_text("\n".join([
        "# Opt {", "#     openspiel,", "#     players: 2,", "# }",
        "node start player 1 actions left right",
        "node left player 2 actions ll lr",
        "node right player 2 actions rl rr",
        "node ll leaf payoffs 1=3 2=-3",
        "node lr leaf payoffs 1=-2 2=2",
        "node rl leaf payoffs 1=0 2=0",
        "node rr leaf payoffs 1=1 2=-1",
        "infoset p1 nodes start",
        "infoset p2 nodes left right",
    ]) + "\n")
    return path


def advance(env, graph, iterations):
    for _ in range(iterations):
        graph.update_graph(env)
        env.update_strategy(graph.current_strategy(), update_best=True)


def snapshot(env, graph):
    strategy = graph.current_strategy()
    selectors = ("last-iterate", "avg-iterate", "linear-avg-iterate", "best-iterate")
    return {
        "values": {
            name: [env.get_value(player, value) for player in (1, 2)]
            for name, value in vars(graph).items()
            if isinstance(value, leg.GraphNode)
        },
        "strategies": {
            selector: [env.get_strategy(player, strategy, selector) for player in (1, 2)]
            for selector in selectors
        },
        "utilities": {selector: env.utility(strategy, selector) for selector in selectors},
        "deviation_gains": {
            selector: env.exploitability(strategy, selector) for selector in selectors
        },
    }


def rng_snapshot():
    return pickle.dumps((random.getstate(), np.random.get_state(),
                         leg._LiteEFG._checkpoint_get_random_state()))


def test_save_load_preserves_graph_environment_values_histories_and_schedule(
        game_file, tmp_path, monkeypatch):
    graph = CMD.graph(inner_epoch=10)
    env = leg.FileEnv(str(game_file))
    env.set_graph(graph)
    graph.metadata = {"label": "saved", "aliases": [graph.u, graph.u]}
    checkpoint_steps = (7, 14, 20)
    advance(env, graph, checkpoint_steps[0])
    path = tmp_path / "state.ckpt"

    def forbidden_constructor(*args, **kwargs):
        raise AssertionError("Checkpoint loading must not rerun the graph constructor")

    monkeypatch.setattr(CMD.graph, "__init__", forbidden_constructor)
    for index, step in enumerate(checkpoint_steps):
        assert graph.timestep == step
        saved = snapshot(env, graph)
        saved_strategy = graph.current_strategy()
        saved_native_environment = leg._LiteEFG._checkpoint_save_environment(env, graph)
        saved_native_graph = graph.__getstate__()[0]
        graph.save(path)
        if index == 0:
            game_file.unlink()

        # Advance to the next checkpoint before restoring. At the final point,
        # take one extra update so loading step 20 also undoes a real change.
        next_step = checkpoint_steps[index + 1] if index + 1 < len(checkpoint_steps) else step + 1
        advance(env, graph, next_step - step)
        expected_continuation = snapshot(env, graph)
        graph.metadata["label"] = "changed"
        graph.transient = True

        assert graph.load(path) is graph
        # Compare all intermediate values and native bookkeeping before metric
        # queries update sequence-form scratch buffers.
        assert leg._LiteEFG._checkpoint_save_environment(env, graph) == saved_native_environment
        assert graph.__getstate__()[0] == saved_native_graph
        assert graph._checkpoint_environment is env
        assert graph.timestep == step
        assert not hasattr(graph, "transient")
        assert graph.metadata["label"] == "saved"
        assert graph.metadata["aliases"][0] is graph.metadata["aliases"][1]
        assert graph.metadata["aliases"][0] is graph.u
        assert snapshot(env, graph) == saved
        assert env.get_value(1, saved_strategy) == env.get_value(1, graph.current_strategy())

        if index + 1 < len(checkpoint_steps):
            advance(env, graph, next_step - step)
            assert graph.timestep == next_step
            assert snapshot(env, graph) == expected_continuation
    assert graph.timestep == checkpoint_steps[-1]


def test_save_load_restores_python_numpy_and_cached_native_random_streams(game_file, tmp_path):
    random.seed(0)
    np.random.seed(0)
    leg.set_seed(0)
    graph = NoiseGraph()
    env = leg.FileEnv(str(game_file))
    env.set_graph(graph)
    env.update(graph.strategy, upd_player=1)
    path = tmp_path / "rng.ckpt"
    original_rng = rng_snapshot()
    graph.save(path)
    assert rng_snapshot() == original_rng

    def trajectory():
        result = []
        for _ in range(20):
            env.update(graph.strategy, upd_player=1)
            result.append((random.random(), np.random.random(),
                           env.get_value(1, graph.noise)))
        return result

    expected = trajectory()
    random.seed(0)
    np.random.seed(0)
    leg.set_seed(0)
    trajectory()
    assert rng_snapshot() != original_rng
    graph.load(path)
    assert rng_snapshot() == original_rng
    assert trajectory() == expected


def test_resume_in_fresh_process_without_source_game_file(game_file, tmp_path):
    graph = FTPL.graph(noise_type="normal")
    env = leg.FileEnv(str(game_file), traverse_type="External")
    env.set_graph(graph)
    leg.set_seed(0)
    advance(env, graph, 7)
    path = tmp_path / "process.ckpt"
    graph.save(path)
    advance(env, graph, 43)
    expected = json.loads(json.dumps(snapshot(env, graph)))
    game_file.unlink()
    result_path = tmp_path / "resumed.json"
    script = """
import json
import sys
import LiteEFG as leg
from test_checkpoint import advance, snapshot
env, graph = leg.load_checkpoint(sys.argv[1])
advance(env, graph, 43)
with open(sys.argv[2], 'w') as stream:
    json.dump(snapshot(env, graph), stream)
"""
    process_environment = dict(os.environ)
    project = Path(__file__).resolve().parents[1]
    process_environment["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(project), str(project / "tests"), process_environment.get("PYTHONPATH"))))
    completed = subprocess.run(
        [sys.executable, "-c", script, str(path), str(result_path)],
        cwd=tmp_path, env=process_environment, text=True, capture_output=True,
        timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(result_path.read_text()) == expected


@pytest.mark.parametrize("mismatch", ["class", "operations", "game"])
def test_load_rejects_other_graph_or_game_without_changing_state(
        game_file, tmp_path, mismatch):
    source = CMD.graph(eta=0.1)
    source_env = leg.FileEnv(str(game_file))
    source_env.set_graph(source)
    advance(source_env, source, 7)
    path = tmp_path / "identity.ckpt"
    source.save(path)

    if mismatch == "class":
        graph = CFR.graph()
    else:
        graph = CMD.graph(eta=0.2 if mismatch == "operations" else 0.1)
    if mismatch == "game":
        game_file.write_text(game_file.read_text().replace("1=3 2=-3", "1=5 2=-5"))
    env = leg.FileEnv(str(game_file))
    env.set_graph(graph)
    advance(env, graph, 3)
    before = snapshot(env, graph)
    before_rng = rng_snapshot()
    with pytest.raises(ValueError, match="match|different|incompatible"):
        graph.load(path)
    assert snapshot(env, graph) == before
    assert rng_snapshot() == before_rng
    assert graph._checkpoint_environment is env
    advance(env, graph, 1)


def rewrite_payload(path, transform):
    with path.open("rb") as stream:
        magic = stream.read(len(checkpoint._MAGIC))
        payload = pickle.load(stream)
    transform(payload)
    with path.open("wb") as stream:
        stream.write(magic)
        pickle.dump(payload, stream)


@pytest.mark.parametrize("damage", ["magic", "truncated", "version", "runtime", "native_rng"])
def test_invalid_checkpoint_preserves_live_state_and_rng(game_file, tmp_path, damage):
    graph = CMD.graph()
    env = leg.FileEnv(str(game_file))
    env.set_graph(graph)
    advance(env, graph, 7)
    path = tmp_path / "invalid.ckpt"
    graph.save(path)
    if damage == "magic":
        path.write_bytes(b"not a checkpoint")
    elif damage == "truncated":
        path.write_bytes(path.read_bytes()[:len(checkpoint._MAGIC) + 5])
    elif damage == "version":
        rewrite_payload(path, lambda state: state.update(format_version=999))
    elif damage == "runtime":
        rewrite_payload(path, lambda state: state["runtime"].update(platform="other"))
    else:
        rewrite_payload(path, lambda state: state.update(native_random={}))
    before = snapshot(env, graph)
    before_rng = rng_snapshot()
    with pytest.raises((ValueError, RuntimeError, KeyError)):
        graph.load(path)
    assert snapshot(env, graph) == before
    assert rng_snapshot() == before_rng
    assert graph.timestep == 7


def test_failed_save_keeps_previous_checkpoint_and_cleans_temporary_file(
        game_file, tmp_path, monkeypatch):
    graph = CMD.graph()
    env = leg.FileEnv(str(game_file))
    env.set_graph(graph)
    advance(env, graph, 7)
    path = tmp_path / "atomic.ckpt"
    graph.save(path)
    previous = path.read_bytes()
    advance(env, graph, 3)

    def failed_replace(*args):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(checkpoint.os, "replace", failed_replace)
    with pytest.raises(OSError, match="simulated"):
        graph.save(path)
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob(".atomic.ckpt.*.tmp"))
    graph.load(path)
    assert graph.timestep == 7


def test_graph_must_be_associated_with_an_environment(tmp_path):
    graph = CMD.graph()
    with pytest.raises(ValueError, match="set_graph"):
        graph.save(tmp_path / "uninitialized.ckpt")
    with pytest.raises(ValueError, match="set_graph"):
        graph.load(tmp_path / "uninitialized.ckpt")
