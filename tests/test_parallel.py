"""Check depth-parallel execution against serial state and reviewed references."""

import concurrent.futures
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

SPEC = importlib.util.spec_from_file_location(
    "parallel_baseline_pipeline", Path(__file__).with_name("reference_output_check.py"))
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)
import LiteEFG as leg


THREAD_COUNTS = (1, 2, 4)
ARGUMENT_MANIFEST = pipeline.MANIFEST.with_name("manifest-arguments.json")
SUITES = {
    "arguments": ("workspace", ARGUMENT_MANIFEST,
                  ARGUMENT_MANIFEST.with_name("reference-arguments-" + sys.platform + ".json")),
    "public": ("public", pipeline.MANIFEST, pipeline.PUBLIC_BASELINE_REFERENCE),
}
MANIFESTS = {name: pipeline.load_manifest(config[1]) for name, config in SUITES.items()}


def fingerprint(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def comparison_worker(suite, case_id, game_id):
    """Reuse imports, but restore even cached normal draws between executions."""
    assert leg.get_threads() == 1, "A fresh interpreter must default to one thread"
    pristine_random = leg._LiteEFG._checkpoint_get_random_state()
    original_measure = pipeline.measure
    results = {}
    for threads in THREAD_COUNTS:
        leg.set_threads(threads)
        leg._LiteEFG._checkpoint_set_random_state(pristine_random)
        strategies = []

        def measure_with_strategy(env, strategy, kind):
            metrics = original_measure(env, strategy, kind)
            # Include every probability, not just utilities/exploitability: many
            # different strategies can have the same scalar metrics.
            rows = [leg.Environment.get_strategy(env, player, strategy, kind)
                    for player in (1, 2)]
            strategies.append({"selector": kind, "sha256": fingerprint(rows)})
            return metrics

        with patch.object(pipeline, "measure", measure_with_strategy):
            checkpoints = pipeline.run_worker(
                MANIFESTS[suite], case_id, game_id, baseline_source=SUITES[suite][0])
        results[str(threads)] = {
            "checkpoints": checkpoints,
            "strategy_fingerprints": strategies,
            "random_state": leg._LiteEFG._checkpoint_get_random_state(),
        }
    return results


@pytest.fixture(scope="module")
def parallel_snapshots(request, tmp_path_factory):
    suite = request.param
    source, _, reference = SUITES[suite]
    manifest = MANIFESTS[suite]
    pipeline.ensure_local_build()
    assert reference.is_file(), "Missing reviewed reference: " + str(reference)
    original_reference = reference.read_bytes()
    expected = pipeline.read_json(reference)
    pipeline.validate_source(expected, source, "workspace")
    invalid = expected["errors"] + pipeline.validate_snapshot(expected)
    assert not invalid, "\n".join(invalid)
    assert expected["contract"]["manifest"] == manifest
    assert expected["contract"]["runtime"] == pipeline.runtime_contract()
    directory = tmp_path_factory.mktemp("parallel-" + suite)
    worker_environment = dict(os.environ, PYTHONHASHSEED=str(manifest["seed"]),
                              OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                              MKL_NUM_THREADS="1")

    def run(case, game):
        key = case["id"] + "/" + game["id"]
        output = directory / (case["id"] + "-" + game["id"] + ".json")
        command = [sys.executable, str(Path(__file__).resolve()), "_worker", suite,
                   case["id"], game["id"], str(output)]
        try:
            process = subprocess.run(command, cwd=pipeline.ROOT, env=worker_environment,
                                     capture_output=True, text=True, timeout=180)
            if process.returncode:
                return key, {"error": process.stdout[-4000:] + process.stderr[-4000:]}
            return key, pipeline.read_json(output)
        except subprocess.TimeoutExpired:
            return key, {"error": "Parallel comparison worker timed out after 180 seconds"}

    # Each worker already uses up to four native threads. Limit process-level
    # concurrency so this suite also remains usable on small CI runners.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run, case, game) for case in manifest["baselines"]
                   for game in manifest["games"]]
        results = dict(future.result() for future in concurrent.futures.as_completed(futures))
    assert reference.read_bytes() == original_reference, "Ground truth must remain unchanged"
    artifact = pipeline.ARTIFACTS / "parallel" / suite / "actual.json"
    pipeline.write_json(artifact, {
        "provenance": pipeline.provenance(source),
        "reference": str(reference), "threads": THREAD_COUNTS, "results": results,
    })
    return expected, results, artifact


@pytest.mark.parametrize("parallel_snapshots,case,game", [
    pytest.param(suite, case, game, id=suite + "/" + case["id"] + "/" + game["id"])
    for suite, manifest in MANIFESTS.items()
    for case in manifest["baselines"] for game in manifest["games"]
], indirect=["parallel_snapshots"], scope="module")
def test_parallel_baseline_outputs(parallel_snapshots, case, game):
    expected, results, artifact = parallel_snapshots
    key = case["id"] + "/" + game["id"]
    actual = results[key]
    assert "error" not in actual, key + ": " + actual.get("error", "")
    tolerance = expected["contract"]["manifest"]["tolerances"]
    differences = pipeline.compare_values(
        expected["results"][key], actual["1"]["checkpoints"],
        tolerance["absolute"], tolerance["relative"], key + "/reference")
    for threads in THREAD_COUNTS[1:]:
        # Parallel execution must preserve exact floating-point values and the
        # subsequent global RNG stream; reference tolerance is not used here.
        differences += pipeline.compare_values(
            actual["1"], actual[str(threads)], 0.0, 0.0, key + "/threads=" + str(threads))
    assert not differences, "\n".join(differences[:20]) + "\nFull results: " + str(artifact)


@pytest.fixture
def restore_thread_settings():
    count = leg.get_threads()
    random_state = leg._LiteEFG._checkpoint_get_random_state()
    yield
    leg.set_threads(count)
    leg._LiteEFG._checkpoint_set_random_state(random_state)


@pytest.mark.parametrize("value", [0, -1, -20])
def test_thread_count_rejects_nonpositive_values(value, restore_thread_settings):
    leg.set_threads(2)
    with pytest.raises(ValueError):
        leg.set_threads(value)
    assert leg.get_threads() == 2


@pytest.mark.parametrize("value", [1.5, "2", None])
def test_thread_count_requires_an_integer(value, restore_thread_settings):
    leg.set_threads(2)
    with pytest.raises(TypeError):
        leg.set_threads(value)
    assert leg.get_threads() == 2


@pytest.fixture
def branching_game(tmp_path):
    records = ["# Opt {", "#     openspiel,", "#     players: 2,", "# }",
               "node start chance actions p1=0.5 p2=0.5"]
    sizes, depths = {}, {}

    def branch(name, player, depth):
        if depth == 4:
            records.append(f"node {name} leaf payoffs 1=1 2=-1")
            return 0
        children = [name + str(action) for action in range(2 + depth % 2)]
        records.append(f"node {name} player {player} actions " + " ".join(children))
        depths[name] = depth + 1
        sizes[name] = 1 + sum(branch(child, player, depth + 1) for child in children)
        return sizes[name]

    branch("p1", 1, 0)
    branch("p2", 2, 0)
    records += [f"infoset {name} nodes {name}" for name in depths]
    path = tmp_path / "branching.game"
    path.write_text("\n".join(records) + "\n")
    return path, sizes, depths


def dependency_graph():
    graph = leg.Graph()
    with leg.backward(is_static=True):
        graph.strategy = leg.const(graph.action_set_size, 1.0 / graph.action_set_size)
        graph.mass = leg.const(1, 1.0)
        graph.mass.inplace(leg.aggregate(graph.mass, "sum").sum() + 1)
        graph.up = leg.const(1, 0.0)
        graph.down = leg.const(1, 0.0)
        graph.previous_parent = leg.const(1, 0.0)
        graph.previous_children = leg.const(1, 0.0)
        graph.seen = leg.const(1, 0.0)
    with leg.forward(is_static=True):
        graph.depth = leg.const(1, 0.0)
        graph.depth.inplace(leg.aggregate(graph.depth, "sum", object="parent") + 1)
    with leg.backward(color=3):
        graph.up.inplace(leg.aggregate(graph.up, "sum").sum() + graph.up + 1)
        graph.previous_parent.inplace(leg.aggregate(graph.up, "sum", object="parent"))
    with leg.forward(color=7):
        graph.down.inplace(leg.aggregate(graph.down, "sum", object="parent") + graph.down + 1)
        graph.previous_children.inplace(leg.aggregate(graph.down, "sum").sum())
    with leg.backward(color=9):
        graph.seen.inplace(graph.up + graph.down)
    return graph


def values(env, graph):
    return {name: [env.get_value(player, value) for player in (1, 2)]
            for name, value in vars(graph).items() if isinstance(value, leg.GraphNode)}


@pytest.mark.parametrize("threads", [1, 2, 4, 8])
def test_static_depth_dependencies_match_tree(branching_game, threads, restore_thread_settings):
    path, sizes, depths = branching_game
    leg.set_threads(threads)
    graph = dependency_graph()
    env = leg.FileEnv(str(path))
    env.set_graph(graph)
    assert leg.get_threads() == threads
    for player in (1, 2):
        assert dict(env.get_value(player, graph.mass)) == {
            name: [float(size)] for name, size in sizes.items() if name.startswith(f"p{player}")}
        assert dict(env.get_value(player, graph.depth)) == {
            name: [float(depth)] for name, depth in depths.items() if name.startswith(f"p{player}")}


@pytest.mark.parametrize("threads", [2, 4])
def test_worker_exception_propagates_and_pool_recovers(branching_game, threads):
    # Isolate the batch failure so a broken barrier produces a bounded failure
    # rather than hanging the entire test session.
    script = """
import sys
import LiteEFG as leg
leg.set_threads(int(sys.argv[2]))
env = leg.FileEnv(sys.argv[1])
bad = leg.Graph()
with leg.backward(is_static=True):
    strategy = leg.const(bad.action_set_size, 1.0 / bad.action_set_size)
    empty = leg.const(0, 0.0)
with leg.backward():
    invalid = empty.argmax()
env.set_graph(bad)
try:
    env.update(strategy)
except ValueError as error:
    assert 'non-empty' in str(error), str(error)
else:
    raise AssertionError('Worker exception did not reach Python')
for count in (4, 2, 1):
    leg.set_threads(count)
    graph = leg.Graph()
    with leg.backward(is_static=True):
        strategy = leg.const(graph.action_set_size, 1.0 / graph.action_set_size)
        state = leg.const(1, 0.0)
    with leg.backward():
        state.inplace(state + 1)
    env.set_graph(graph)
    env.update(strategy)
    for player in (1, 2):
        assert all(value == [1.0] for _, value in env.get_value(player, state))
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(branching_game[0]), str(threads)],
        cwd=pipeline.ROOT, capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stdout + process.stderr


@pytest.mark.parametrize("traversal", ["Enumerate", "External", "Outcome"])
@pytest.mark.parametrize("player", [-1, 1, 2])
def test_dynamic_dependencies_colors_players_and_count_changes(
        branching_game, traversal, player, restore_thread_settings):
    path, _, _ = branching_game
    snapshots = []
    for counts in ([1] * 8, [4, 2, 1, 8, 2, 4, 1, 4]):
        leg.set_threads(counts[0])
        leg.set_seed(0)
        graph = dependency_graph()
        env = leg.FileEnv(str(path), traverse_type="Enumerate")
        env.set_graph(graph)
        trajectory = [values(env, graph)]
        for iteration, count in enumerate(counts):
            leg.set_threads(count)
            colors = ([3], [7], [3, 7, 9], [9])[iteration % 4]
            before = values(env, graph)
            env.update(graph.strategy, upd_player=player, upd_color=colors,
                       traverse_type=traversal)
            after = values(env, graph)
            for name, color in (("up", 3), ("previous_parent", 3), ("down", 7),
                                ("previous_children", 7), ("seen", 9)):
                if color not in colors:
                    assert after[name] == before[name]
                if player != -1:
                    assert after[name][2 - player] == before[name][2 - player]
            trajectory.append(after)
        snapshots.append((trajectory, leg._LiteEFG._checkpoint_get_random_state()))
    assert snapshots[1] == snapshots[0]


@pytest.mark.parametrize("module_name,kwargs,traversal", [
    ("CMD", {"inner_epoch": 10, "weighted": True}, "Enumerate"),
    ("FTPL", {"noise_type": "normal"}, "Enumerate"),
    ("FTPL", {"noise_type": "normal"}, "External"),
    ("OS_MCCFR", {"balanced": True}, "Outcome"),
])
def test_checkpoint_resume_rebuilds_schedule_for_changed_thread_count(
        branching_game, tmp_path, module_name, kwargs, traversal, restore_thread_settings):
    pipeline.activate_package(baseline_source="workspace")
    module = importlib.import_module("LiteEFG.baselines." + module_name)
    leg.set_seed(0)
    leg.set_threads(4)
    graph = module.graph(**kwargs)
    env = leg.FileEnv(str(branching_game[0]), traverse_type=traversal)
    env.set_graph(graph)

    def advance(environment, algorithm, steps):
        for _ in range(steps):
            algorithm.update_graph(environment)
            environment.update_strategy(algorithm.current_strategy())

    def snapshot(environment, algorithm):
        strategy = algorithm.current_strategy()
        return {
            "values": values(environment, algorithm),
            "strategies": {kind: [leg.Environment.get_strategy(environment, owner, strategy, kind)
                                    for owner in (1, 2)]
                           for kind in ("last-iterate", "avg-iterate", "linear-avg-iterate")},
            "random_state": leg._LiteEFG._checkpoint_get_random_state(),
        }

    advance(env, graph, 7)
    checkpoint = tmp_path / "parallel.ckpt"
    graph.save(checkpoint)
    advance(env, graph, 13)
    expected = snapshot(env, graph)
    for threads in (1, 2, 4):
        leg.set_threads(threads)
        restored_env, restored_graph = leg.load_checkpoint(checkpoint)
        assert leg.get_threads() == threads
        advance(restored_env, restored_graph, 13)
        assert snapshot(restored_env, restored_graph) == expected


if __name__ == "__main__":
    assert len(sys.argv) == 6 and sys.argv[1] == "_worker"
    pipeline.write_json(Path(sys.argv[5]), comparison_worker(*sys.argv[2:5]))
