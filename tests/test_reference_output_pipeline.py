"""Unit tests for reference output comparisons; no native extension or games are loaded."""

from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SPEC = importlib.util.spec_from_file_location(
    "reference_output_check", Path(__file__).with_name("reference_output_check.py"))
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)


@pytest.fixture
def snapshot():
    metric = {
        "utility": [0.25, -0.25],
        "deviation_gain": [0.125, 0.375],
        "nash_conv": 0.5,
        "exploitability": 0.25,
        "information_sets": [6, 6],
    }
    return {
        "contract": {
            "manifest": {
                "schema_version": 1,
                "iterations": 50,
                "checkpoints": [1, 50],
                "strategy_types": ["last-iterate"],
                "seed": 0,
                "tolerances": {"absolute": 1e-10, "relative": 1e-8},
                "games": [{"id": "kuhn", "spec": "kuhn_poker(players=2)"}],
                "baselines": [{"id": "CFR", "module": "CFR", "kwargs": {}}],
            },
            "runtime": {"platform": "darwin", "python": "3.12"},
            "constructor_defaults": {"CFR": {"discount": 1.0}},
        },
        "provenance": {"source_sha256": "before", "extension_sha256": "before",
                       "engine_origin": "pypi", "baseline_source": "public"},
        "results": {"CFR/kuhn": {"1": {"last-iterate": deepcopy(metric)},
                                  "50": {"last-iterate": deepcopy(metric)}}},
        "errors": [],
    }


def test_small_numerical_drift_is_accepted(snapshot):
    actual = deepcopy(snapshot)
    metric = actual["results"]["CFR/kuhn"]["50"]["last-iterate"]
    metric["utility"][0] += 1e-9  # Within the relative tolerance.
    metric["utility"][1] -= 1e-11  # Within the absolute tolerance.
    assert pipeline.compare_snapshots(snapshot, actual) == []
    assert pipeline.compare_values(0.0, 5e-11, 1e-10, 1e-8) == []


def test_output_mismatch_reports_the_metric_path(snapshot):
    actual = deepcopy(snapshot)
    actual["results"]["CFR/kuhn"]["50"]["last-iterate"]["utility"][0] += 0.01
    differences = pipeline.compare_snapshots(snapshot, actual)
    assert len(differences) == 1
    assert differences[0].startswith("results.CFR/kuhn.50.last-iterate.utility[0]:")
    assert "expected=" in differences[0] and "actual=" in differences[0]


@pytest.mark.parametrize("nonfinite", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("side", ["expected", "actual", "both"])
def test_nonfinite_values_are_never_equal(nonfinite, side):
    expected = nonfinite if side in ("expected", "both") else 0.0
    actual = nonfinite if side in ("actual", "both") else 0.0
    assert pipeline.compare_values({"metric": [expected]}, {"metric": [actual]},
                                   1e-10, 1e-8)[0].startswith("output.metric[0]:")


@pytest.mark.parametrize("level", ["case", "checkpoint"])
@pytest.mark.parametrize("operation", ["missing", "extra"])
def test_changed_case_or_checkpoint_coverage_fails(snapshot, level, operation):
    actual = deepcopy(snapshot)
    target = actual["results"] if level == "case" else actual["results"]["CFR/kuhn"]
    original = "CFR/kuhn" if level == "case" else "50"
    if operation == "missing":
        del target[original]
        changed_key = original
    else:
        changed_key = "CFR/extra_game" if level == "case" else "25"
        target[changed_key] = deepcopy(target[original])
    differences = pipeline.compare_snapshots(snapshot, actual)
    assert len(differences) == 1
    assert operation + "=['" + changed_key + "']" in differences[0]


@pytest.mark.parametrize("field,value", [
    ("iterations", 51),
    ("seed", None),
    ("checkpoints", [1, 25, 50]),
    ("games", [{"id": "kuhn", "spec": "kuhn_poker(players=3)"}]),
    ("baselines", [{"id": "CFR", "module": "CFR", "kwargs": {"discount": 0.5}}]),
    ("tolerances", {"absolute": 1.000000001e-10, "relative": 1e-8}),
])
def test_manifest_changes_fail_exactly(snapshot, field, value):
    actual = deepcopy(snapshot)
    actual["contract"]["manifest"][field] = value
    differences = pipeline.compare_snapshots(snapshot, actual)
    assert differences
    assert all(item.startswith("contract.manifest." + field) for item in differences)


def test_constructor_default_changes_fail_even_below_output_tolerance(snapshot):
    actual = deepcopy(snapshot)
    actual["contract"]["constructor_defaults"]["CFR"]["discount"] += 1e-12
    differences = pipeline.compare_snapshots(snapshot, actual)
    assert len(differences) == 1
    assert differences[0].startswith("contract.constructor_defaults.CFR.discount:")


def test_provenance_does_not_change_expected_numerics(snapshot):
    actual = deepcopy(snapshot)
    actual["provenance"] = {"source_sha256": "after", "extension_sha256": "after",
                            "platform_description": "A rebuilt checkout"}
    assert pipeline.compare_snapshots(snapshot, actual) == []


def test_complete_snapshot_is_valid(snapshot):
    assert pipeline.validate_snapshot(snapshot) == []


@pytest.mark.parametrize("metric,value", [
    ("utility", [math.nan, 0.0]),
    ("deviation_gain", [0.0, math.inf]),
    ("utility", [-math.inf, 0.0]),
    ("information_sets", [6]),
])
def test_snapshot_validation_rejects_invalid_player_vectors(snapshot, metric, value):
    snapshot["results"]["CFR/kuhn"]["50"]["last-iterate"][metric] = value
    differences = pipeline.validate_snapshot(snapshot)
    assert differences
    assert all(item.startswith("CFR/kuhn/50/last-iterate/" + metric + ":")
               for item in differences)


@pytest.mark.parametrize("field,value", [
    ("strategy_types", []),
    ("strategy_types", ["last-iterate", "last-iterate"]),
    ("strategy_types", ["unknown-iterate"]),
    ("checkpoints", []),
    ("games", []),
    ("baselines", []),
])
def test_manifest_rejects_empty_or_invalid_selectors(tmp_path, snapshot, field, value):
    manifest = snapshot["contract"]["manifest"]
    manifest[field] = value
    path = tmp_path / "manifest.json"
    pipeline.write_json(path, manifest)
    with pytest.raises(ValueError):
        pipeline.load_manifest(path)


def test_strategy_vector_length_and_integer_counts_are_exact(snapshot):
    actual = deepcopy(snapshot)
    metric = actual["results"]["CFR/kuhn"]["50"]["last-iterate"]
    metric["utility"].append(0.0)
    metric["information_sets"][0] += 1
    differences = pipeline.compare_snapshots(snapshot, actual)
    assert len(differences) == 2
    assert any("utility: length 2 != 3" in item for item in differences)
    assert any("information_sets[0]: expected=6, actual=7" in item for item in differences)


def forbidden_collection(*args, **kwargs):
    pytest.fail("Refusal must happen before expensive collection or environment setup")


@pytest.mark.parametrize("command", ["record", "record-workspace"])
def test_record_refuses_existing_reference_before_collection(tmp_path, monkeypatch, snapshot, command):
    reference = tmp_path / "reference.json"
    reference.write_text("existing reference\n")
    monkeypatch.setattr(pipeline, "load_manifest", lambda *args: snapshot["contract"]["manifest"])
    monkeypatch.setattr(pipeline, "collect_snapshot", forbidden_collection)
    with pytest.raises(SystemExit) as exc:
        pipeline.main([command, "--reference", str(reference),
                       "--artifacts", str(tmp_path / "artifacts")])
    assert exc.value.code == 2
    assert reference.read_text() == "existing reference\n"
    assert not (tmp_path / "artifacts").exists()


def test_check_never_creates_a_missing_reference(tmp_path, monkeypatch):
    reference = tmp_path / "missing" / "reference.json"
    monkeypatch.setattr(pipeline, "collect_snapshot", forbidden_collection)
    monkeypatch.setattr(pipeline, "runtime_contract", forbidden_collection)
    with pytest.raises(ValueError, match="check never creates one"):
        pipeline.check(reference=reference, artifacts=tmp_path / "artifacts")
    assert not reference.parent.exists()
    assert not (tmp_path / "artifacts").exists()


def test_record_requires_an_explicit_pypi_oracle(tmp_path, monkeypatch, snapshot):
    monkeypatch.setattr(pipeline, "load_manifest", lambda *args: snapshot["contract"]["manifest"])
    monkeypatch.setattr(pipeline, "collect_snapshot", forbidden_collection)
    reference = tmp_path / "reference.json"
    with pytest.raises(SystemExit) as exc:
        pipeline.main(["record", "--reference", str(reference)])
    assert exc.value.code == 2
    assert not reference.exists()


@pytest.mark.parametrize("relative", [".", "nested-oracle"])
def test_oracle_cannot_be_the_checkout_or_its_subdirectory(relative):
    with pytest.raises(ValueError, match="outside this checkout"):
        pipeline.activate_package(pipeline.ROOT / relative, "public")


def test_oracle_requires_a_pip_distribution(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline.importlib.metadata, "distributions", lambda **kwargs: [])
    with pytest.raises(ValueError, match="pip-installed LiteEFG distribution"):
        pipeline.activate_package(tmp_path, "public")


@pytest.mark.parametrize("engine,baselines", [
    ("pypi", "workspace"), ("unknown", "public"), (None, "public"),
    ("workspace", None),
])
def test_check_rejects_invalid_reference_provenance(
        tmp_path, monkeypatch, snapshot, engine, baselines):
    snapshot["provenance"].update(engine_origin=engine, baseline_source=baselines)
    reference = tmp_path / "reference.json"
    pipeline.write_json(reference, snapshot)
    monkeypatch.setattr(pipeline, "collect_snapshot", forbidden_collection)
    monkeypatch.setattr(pipeline, "runtime_contract", forbidden_collection)
    with pytest.raises(ValueError, match="Invalid reference provenance"):
        pipeline.check(reference=reference, artifacts=tmp_path / "artifacts")
    assert not (tmp_path / "artifacts").exists()


def test_record_runs_pypi_engine_and_public_baselines_twice(tmp_path, monkeypatch, snapshot):
    manifest = snapshot["contract"]["manifest"]
    oracle = tmp_path / "pypi"
    reference = tmp_path / "reference.json"
    calls = []

    def collect(selected_manifest, jobs, oracle_root, baseline_source):
        calls.append((selected_manifest, jobs, oracle_root, baseline_source))
        return deepcopy(snapshot)

    monkeypatch.setattr(pipeline, "load_manifest", lambda *args: manifest)
    monkeypatch.setattr(pipeline, "collect_snapshot", collect)
    pipeline.main(["record", "--reference", str(reference), "--oracle-root", str(oracle),
                   "--artifacts", str(tmp_path / "artifacts"), "--baseline-source", "public"])
    assert calls == [(manifest, 4, oracle, "public")] * 2
    assert pipeline.read_json(reference) == snapshot


@pytest.mark.parametrize("failure", ["unstable", "worker_error"])
@pytest.mark.parametrize("command", ["record", "record-workspace"])
def test_record_preserves_reference_if_the_runs_fail(
        tmp_path, monkeypatch, snapshot, failure, command):
    if command == "record-workspace":
        snapshot["provenance"].update(engine_origin="workspace", baseline_source="workspace")
    first, second = deepcopy(snapshot), deepcopy(snapshot)
    if failure == "unstable":
        second["results"]["CFR/kuhn"]["50"]["last-iterate"]["exploitability"] += 0.01
    else:
        second["errors"] = ["CFR/kuhn: worker failed"]
    runs = iter([first, second])
    reference = tmp_path / "reference.json"
    reference.write_text("existing reference\n")
    monkeypatch.setattr(pipeline, "load_manifest", lambda *args: snapshot["contract"]["manifest"])
    monkeypatch.setattr(pipeline, "collect_snapshot", lambda *args, **kwargs: next(runs))
    with pytest.raises(AssertionError, match="Refusing unstable/failed reference"):
        pipeline.main([command, "--reference", str(reference), "--overwrite",
                       "--artifacts", str(tmp_path / "artifacts")]
                      + (["--oracle-root", str(tmp_path / "pypi")] if command == "record" else []))
    assert reference.read_text() == "existing reference\n"


@pytest.mark.parametrize("missing", ["case", "checkpoint", "strategy", "metric"])
@pytest.mark.parametrize("command", ["record", "record-workspace"])
def test_record_rejects_two_equally_incomplete_runs(
        tmp_path, monkeypatch, snapshot, missing, command):
    if command == "record-workspace":
        snapshot["provenance"].update(engine_origin="workspace", baseline_source="workspace")
    if missing == "case":
        snapshot["results"].clear()
    elif missing == "checkpoint":
        del snapshot["results"]["CFR/kuhn"]["50"]
    elif missing == "strategy":
        snapshot["results"]["CFR/kuhn"]["50"].clear()
    else:
        del snapshot["results"]["CFR/kuhn"]["50"]["last-iterate"]["exploitability"]
    # Agreement alone is insufficient: both runs can have the same missing data.
    assert pipeline.compare_snapshots(snapshot, deepcopy(snapshot)) == []
    assert pipeline.validate_snapshot(snapshot)
    monkeypatch.setattr(pipeline, "load_manifest", lambda *args: snapshot["contract"]["manifest"])
    monkeypatch.setattr(pipeline, "collect_snapshot", lambda *args, **kwargs: deepcopy(snapshot))
    reference = tmp_path / "reference.json"
    with pytest.raises(AssertionError, match="Refusing unstable/failed reference"):
        pipeline.main([command, "--reference", str(reference),
                       "--artifacts", str(tmp_path / "artifacts")]
                      + (["--oracle-root", str(tmp_path / "pypi")] if command == "record" else []))
    assert not reference.exists()


@pytest.mark.parametrize("nonfinite", [math.nan, math.inf, -math.inf])
def test_nonfinite_json_cannot_replace_an_existing_reference(tmp_path, nonfinite):
    reference = tmp_path / "reference.json"
    reference.write_text('{"previous": 1}\n')
    with pytest.raises(ValueError):
        pipeline.write_json(reference, {"metric": nonfinite})
    assert reference.read_text() == '{"previous": 1}\n'


@pytest.mark.parametrize("baseline_source", ["workspace", "public"])
def test_record_workspace_runs_the_requested_layer_twice(
        tmp_path, monkeypatch, snapshot, baseline_source):
    snapshot["provenance"].update(engine_origin="workspace", baseline_source=baseline_source)
    manifest = snapshot["contract"]["manifest"]
    reference = tmp_path / "reference.json"
    calls = []

    def collect(selected_manifest, jobs, oracle_root, selected_source):
        calls.append((selected_manifest, jobs, oracle_root, selected_source))
        return deepcopy(snapshot)

    monkeypatch.setattr(pipeline, "load_manifest", lambda *args: manifest)
    monkeypatch.setattr(pipeline, "collect_snapshot", collect)
    pipeline.main(["record-workspace", "--reference", str(reference),
                   "--artifacts", str(tmp_path / "artifacts"),
                   "--baseline-source", baseline_source])
    assert calls == [(manifest, 4, None, baseline_source)] * 2
    assert pipeline.read_json(reference) == snapshot
    assert pipeline.read_json(tmp_path / "artifacts/record-first.json") == snapshot
    assert pipeline.read_json(tmp_path / "artifacts/record-second.json") == snapshot


@pytest.mark.parametrize("baseline_source", ["workspace", "public"])
def test_check_rejects_workspace_reference_from_another_layer(
        tmp_path, monkeypatch, snapshot, baseline_source):
    other_source = "public" if baseline_source == "workspace" else "workspace"
    snapshot["provenance"].update(engine_origin="workspace", baseline_source=other_source)
    reference = tmp_path / "reference.json"
    pipeline.write_json(reference, snapshot)
    monkeypatch.setattr(pipeline, "collect_snapshot", forbidden_collection)
    monkeypatch.setattr(pipeline, "runtime_contract", forbidden_collection)
    with pytest.raises(ValueError, match="baseline_source mismatch"):
        pipeline.check(reference=reference, artifacts=tmp_path / "artifacts",
                       baseline_source=baseline_source)
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("engine_origin", ["pypi", "workspace"])
@pytest.mark.parametrize("baseline_source", ["workspace", "public"])
def test_check_accepts_matching_workspace_layer_or_historical_pypi_reference(
        tmp_path, monkeypatch, snapshot, engine_origin, baseline_source):
    snapshot["provenance"].update(engine_origin=engine_origin,
                                  baseline_source="public" if engine_origin == "pypi"
                                  else baseline_source)
    actual = deepcopy(snapshot)
    actual["provenance"].update(engine_origin="workspace", baseline_source=baseline_source,
                                source_sha256="rebuilt", extension_sha256="rebuilt")
    reference = tmp_path / "reference.json"
    pipeline.write_json(reference, snapshot)
    monkeypatch.setattr(pipeline, "runtime_contract", lambda: snapshot["contract"]["runtime"])
    monkeypatch.setattr(pipeline, "load_manifest", lambda *args: snapshot["contract"]["manifest"])
    monkeypatch.setattr(pipeline, "collect_snapshot", lambda *args, **kwargs: deepcopy(actual))
    assert pipeline.check(reference, tmp_path / "artifacts", baseline_source=baseline_source) == 1
    actual["results"]["CFR/kuhn"]["50"]["last-iterate"]["exploitability"] += 0.01
    with pytest.raises(AssertionError, match="results.CFR/kuhn.50.last-iterate.exploitability"):
        pipeline.check(reference, tmp_path / "artifacts", baseline_source=baseline_source)


@pytest.mark.parametrize("command,options", [
    ("check", ["--overwrite"]),
    ("check", ["--oracle-root", "/tmp/oracle"]),
    ("record-workspace", ["--oracle-root", "/tmp/oracle"]),
    ("record", ["--baseline-source", "workspace", "--oracle-root", "/tmp/oracle"]),
    ("check", ["--jobs", "0"]),
    ("_worker", ["--oracle-root", "/tmp/oracle", "--baseline-source", "workspace"]),
    ("check", ["--checkpoint-interval", "0"]),
    ("check", ["--checkpoint-interval", "-10"]),
    ("record", ["--checkpoint-interval", "10"]),
    ("record-workspace", ["--checkpoint-interval", "10"]),
    ("list", ["--checkpoint-interval", "10"]),
    ("_worker", ["--checkpoint-interval", "10", "--oracle-root", "/tmp/oracle"]),
])
def test_incompatible_recording_flags_are_rejected_before_collection(
        tmp_path, monkeypatch, command, options):
    monkeypatch.setattr(pipeline, "collect_snapshot", forbidden_collection)
    monkeypatch.setattr(pipeline, "load_manifest", forbidden_collection)
    with pytest.raises(SystemExit) as exc:
        pipeline.main([command, "--reference", str(tmp_path / "reference.json")] + options)
    assert exc.value.code == 2
    assert not (tmp_path / "reference.json").exists()


@pytest.mark.parametrize("command,engine,baselines", [
    ("record", "workspace", "public"),
    ("record-workspace", "pypi", "public"),
    ("record-workspace", "workspace", "public"),
])
@pytest.mark.parametrize("bad_run", [0, 1])
def test_recording_refuses_mislabeled_sources_in_either_run(
        tmp_path, monkeypatch, snapshot, command, engine, baselines, bad_run):
    if command == "record-workspace":
        snapshot["provenance"].update(engine_origin="workspace", baseline_source="workspace")
    runs = [deepcopy(snapshot), deepcopy(snapshot)]
    runs[bad_run]["provenance"].update(engine_origin=engine, baseline_source=baselines)
    iterator = iter(runs)
    monkeypatch.setattr(pipeline, "collect_snapshot", lambda *args: next(iterator))
    monkeypatch.setattr(pipeline, "load_manifest", lambda *args: snapshot["contract"]["manifest"])
    reference = tmp_path / "reference.json"
    reference.write_text("existing reference\n")
    with pytest.raises(ValueError, match="mismatch"):
        pipeline.main([command, "--reference", str(reference), "--overwrite",
                       "--artifacts", str(tmp_path / "artifacts")]
                      + (["--oracle-root", str(tmp_path / "oracle")] if command == "record" else []))
    assert reference.read_text() == "existing reference\n"


def test_default_reference_paths_are_distinct_for_all_three_origins():
    assert pipeline.default_reference("workspace") == pipeline.REFERENCE
    assert pipeline.default_reference("public") == pipeline.PUBLIC_BASELINE_REFERENCE
    assert pipeline.default_reference("public", "pypi") == pipeline.PYPI_REFERENCE
    assert len({pipeline.REFERENCE, pipeline.PUBLIC_BASELINE_REFERENCE, pipeline.PYPI_REFERENCE}) == 3


@pytest.mark.parametrize("baseline_source", ["workspace", "public"])
def test_check_resolves_the_default_reference_for_its_layer(
        tmp_path, monkeypatch, snapshot, baseline_source):
    snapshot["provenance"].update(engine_origin="workspace", baseline_source=baseline_source)
    reference = tmp_path / (baseline_source + ".json")
    constant = "REFERENCE" if baseline_source == "workspace" else "PUBLIC_BASELINE_REFERENCE"
    monkeypatch.setattr(pipeline, constant, reference)
    pipeline.write_json(reference, snapshot)
    monkeypatch.setattr(pipeline, "runtime_contract", lambda: snapshot["contract"]["runtime"])
    monkeypatch.setattr(pipeline, "load_manifest", lambda *args: snapshot["contract"]["manifest"])
    monkeypatch.setattr(pipeline, "collect_snapshot", lambda *args, **kwargs: deepcopy(snapshot))
    assert pipeline.check(artifacts=tmp_path / "artifacts", baseline_source=baseline_source) == 1
    assert pipeline.read_json(tmp_path / "artifacts/comparison.json")["reference"] == str(reference)


@pytest.mark.parametrize("command,options,constant,engine,baselines", [
    ("record", ["--oracle-root", "/tmp/oracle"], "PYPI_REFERENCE", "pypi", "public"),
    ("record-workspace", [], "REFERENCE", "workspace", "workspace"),
    ("record-workspace", ["--baseline-source", "public"],
     "PUBLIC_BASELINE_REFERENCE", "workspace", "public"),
])
def test_recording_resolves_the_default_reference_for_its_origin(
        tmp_path, monkeypatch, snapshot, command, options, constant, engine, baselines):
    reference = tmp_path / "reference.json"
    snapshot["provenance"].update(engine_origin=engine, baseline_source=baselines)
    monkeypatch.setattr(pipeline, constant, reference)
    monkeypatch.setattr(pipeline, "load_manifest", lambda *args: snapshot["contract"]["manifest"])
    monkeypatch.setattr(pipeline, "collect_snapshot", lambda *args, **kwargs: deepcopy(snapshot))
    pipeline.main([command, "--artifacts", str(tmp_path / "artifacts")] + options)
    assert pipeline.read_json(reference) == snapshot


@pytest.mark.parametrize("interval", [0, -1, 0.5, True, "10"])
def test_checkpoint_interval_rejects_invalid_values_before_loading_native_code(
        monkeypatch, interval):
    monkeypatch.setattr(pipeline, "activate_package", forbidden_collection)
    with pytest.raises(ValueError, match="positive integer"):
        pipeline.run_worker({}, "CFR", "kuhn", checkpoint_interval=interval)
    with pytest.raises(ValueError, match="positive integer"):
        pipeline.collect_snapshot({}, checkpoint_interval=interval)


def test_checkpoint_worker_performs_five_roundtrips_before_measurement(monkeypatch, snapshot):
    """Guard the actual training loop: no sixth segment or missed final reload."""
    events = []

    class Graph:
        def __init__(self):
            self.iteration = 0
            self.reloads = 0

        def current_strategy(self):
            return self

        def update_graph(self, env):
            self.iteration += 1
            events.append(("update", self.iteration))

        def save(self, path):
            events.append(("save", self.iteration))
            pipeline.write_json(path, {"iteration": self.iteration})

        def load(self, path):
            saved = pipeline.read_json(path)
            assert saved["iteration"] == self.iteration
            events.append(("load", self.iteration))
            self.reloads += 1

    class Environment:
        def __init__(self, *args, **kwargs):
            pass

        def set_graph(self, graph):
            self.graph = graph

        def update_strategy(self, strategy):
            assert strategy is self.graph
            events.append(("strategy", strategy.iteration))

    game = SimpleNamespace(num_players=lambda: 2,
                           get_type=lambda: SimpleNamespace(utility="zero-sum"))
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(
        random=SimpleNamespace(seed=lambda seed: None)))
    monkeypatch.setitem(sys.modules, "pyspiel", SimpleNamespace(
        load_game=lambda spec: game, GameType=SimpleNamespace(
            Utility=SimpleNamespace(ZERO_SUM="zero-sum"))))
    monkeypatch.setitem(sys.modules, "LiteEFG", SimpleNamespace(
        set_seed=lambda seed: None, OpenSpielEnv=Environment))
    monkeypatch.setattr(pipeline, "activate_package", lambda *args: None)
    import_module = pipeline.importlib.import_module
    monkeypatch.setattr(pipeline.importlib, "import_module", lambda name:
                        SimpleNamespace(graph=Graph) if name == "LiteEFG.baselines.CFR"
                        else import_module(name))

    def measure(env, strategy, kind):
        assert strategy is env.graph
        events.append(("measure", strategy.iteration))
        return {"reloads": strategy.reloads}

    monkeypatch.setattr(pipeline, "measure", measure)
    manifest = snapshot["contract"]["manifest"]
    manifest["checkpoints"] = [1, 10, 25, 50]
    manifest["baselines"][0]["traversal"] = "Enumerate"
    observed = pipeline.run_worker(manifest, "CFR", "kuhn", checkpoint_interval=10)
    assert observed == {
        str(iteration): {"last-iterate": {"reloads": reloads}}
        for iteration, reloads in [(1, 0), (10, 1), (25, 2), (50, 5)]
    }
    expected_events = []
    for iteration in range(1, 51):
        expected_events.extend([("update", iteration), ("strategy", iteration)])
        if iteration % 10 == 0:
            expected_events.extend([("save", iteration), ("load", iteration)])
        if iteration in manifest["checkpoints"]:
            expected_events.append(("measure", iteration))
    assert events == expected_events


def test_checkpoint_collection_passes_interval_to_workers_without_changing_contract(
        monkeypatch, snapshot):
    manifest = snapshot["contract"]["manifest"]
    monkeypatch.setattr(pipeline, "activate_package", lambda *args: pipeline.ROOT)
    monkeypatch.setattr(pipeline, "ensure_local_build", lambda: None)
    monkeypatch.setattr(pipeline, "baseline_defaults", lambda *args: {})
    monkeypatch.setattr(pipeline, "public_source", lambda: {})
    monkeypatch.setattr(pipeline, "runtime_contract", lambda: {})
    monkeypatch.setattr(pipeline, "provenance", lambda *args: {
        "engine_origin": "workspace", "baseline_source": "workspace"})

    def worker(command, **kwargs):
        assert command[command.index("--checkpoint-interval") + 1] == "10"
        output = command[command.index("--output") + 1]
        pipeline.write_json(output, snapshot["results"]["CFR/kuhn"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(pipeline.subprocess, "run", worker)
    actual = pipeline.collect_snapshot(manifest, jobs=1, checkpoint_interval=10)
    assert actual["errors"] == []
    assert actual["results"] == snapshot["results"]
    assert actual["contract"]["manifest"] == manifest
    assert "checkpoint_interval" not in actual["contract"]
    assert actual["provenance"]["checkpoint_interval"] == 10


def test_checkpoint_cli_passes_interval_to_check(monkeypatch, snapshot):
    calls = []
    monkeypatch.setattr(pipeline, "load_manifest", lambda *args: snapshot["contract"]["manifest"])
    monkeypatch.setattr(pipeline, "check", lambda *args: calls.append(args) or 1)
    assert pipeline.main(["check", "--checkpoint-interval", "10"]) == 0
    assert len(calls) == 1 and calls[0][-1] == 10
