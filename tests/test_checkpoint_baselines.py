"""Resume every reviewed baseline configuration against unchanged ground truth."""

import importlib.util
from pathlib import Path
import sys

import pytest


SPEC = importlib.util.spec_from_file_location(
    "checkpoint_baseline_pipeline", Path(__file__).with_name("reference_output_check.py"))
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)
ARGUMENT_MANIFEST = pipeline.MANIFEST.with_name("manifest-arguments.json")
SUITES = {
    "workspace": ("workspace", pipeline.MANIFEST, pipeline.REFERENCE),
    "public": ("public", pipeline.MANIFEST, pipeline.PUBLIC_BASELINE_REFERENCE),
    "arguments": ("workspace", ARGUMENT_MANIFEST,
                  ARGUMENT_MANIFEST.with_name("reference-arguments-" + sys.platform + ".json")),
}
MANIFESTS = {name: pipeline.load_manifest(config[1]) for name, config in SUITES.items()}


@pytest.fixture(scope="module")
def checkpoint_snapshots(request):
    suite = request.param
    baseline_source, _, reference = SUITES[suite]
    manifest = MANIFESTS[suite]
    artifacts = pipeline.ARTIFACTS / "checkpoints" / suite
    if not reference.is_file():
        pytest.fail("No reviewed reference for this platform: " + str(reference)
                    + ". Checkpoint tests never create references.")
    original_reference = reference.read_bytes()
    expected = pipeline.read_json(reference)
    pipeline.validate_source(expected, baseline_source, "workspace")
    invalid = expected["errors"] + pipeline.validate_snapshot(expected)
    assert not invalid, "Invalid checkpoint reference:\n" + "\n".join(invalid)
    contract = expected["contract"]
    before_collection = pipeline.compare_values(contract["manifest"], manifest, 0.0, 0.0,
                                                "contract.manifest")
    before_collection += pipeline.compare_values(contract["runtime"],
                                                 pipeline.runtime_contract(), 0.0, 0.0,
                                                 "contract.runtime")
    assert not before_collection, "Reference mismatch:\n" + "\n".join(before_collection)

    # The manifest and measured checkpoints remain identical to uninterrupted
    # training. Only the execution path changes: five complete save/load cycles.
    actual = pipeline.collect_snapshot(manifest, jobs=4, baseline_source=baseline_source,
                                       checkpoint_interval=10)
    assert reference.read_bytes() == original_reference, "Ground truth must remain unchanged"
    pipeline.validate_source(actual, baseline_source, "workspace")
    assert actual["provenance"]["checkpoint_interval"] == 10
    pipeline.write_json(artifacts / "actual.json", actual)
    differences = (actual["errors"] + pipeline.validate_snapshot(actual)
                   + pipeline.compare_snapshots(expected, actual))
    pipeline.write_json(artifacts / "comparison.json", {
        "passed": not differences, "reference": str(reference),
        "case_count": len(actual["results"]), "checkpoint_interval": 10,
        "differences": differences,
    })
    contract_diff = pipeline.compare_values(contract, actual["contract"], 0.0, 0.0,
                                            "contract")
    assert not contract_diff, "Checkpoint contract mismatch:\n" + "\n".join(contract_diff)
    keys = {case["id"] + "/" + game["id"] for case in manifest["baselines"]
            for game in manifest["games"]}
    assert not set(actual["results"]) - keys, "Unexpected checkpoint worker results"
    unassigned_errors = [error for error in actual["errors"]
                         if not any(error.startswith(key + ":") for key in keys)]
    assert not unassigned_errors, "\n".join(unassigned_errors)
    return expected, actual, artifacts


# Explicit scope keeps the expensive fixture shared per suite: mixing direct
# case/game arguments with an indirect fixture otherwise selects function scope.
@pytest.mark.parametrize("checkpoint_snapshots,case,game", [
    pytest.param(suite, case, game, id=suite + "/" + case["id"] + "/" + game["id"])
    for suite, manifest in MANIFESTS.items()
    for case in manifest["baselines"] for game in manifest["games"]
], indirect=["checkpoint_snapshots"], scope="module")
def test_baseline_checkpoint_outputs(checkpoint_snapshots, case, game):
    expected, actual, artifacts = checkpoint_snapshots
    key = case["id"] + "/" + game["id"]
    errors = [error for error in actual["errors"] if error.startswith(key + ":")]
    assert not errors, "\n".join(errors)
    assert key in actual["results"], key + ": missing checkpoint worker output"
    # Attribute worker failures to individual baseline/game tests so one failed
    # configuration does not hide the remaining baseline results.
    manifest = actual["contract"]["manifest"]
    single_case = dict(actual, contract=dict(actual["contract"], manifest=dict(
        manifest, baselines=[case], games=[game])), results={key: actual["results"][key]})
    differences = pipeline.validate_snapshot(single_case)
    tolerance = expected["contract"]["manifest"]["tolerances"]
    differences += pipeline.compare_values(expected["results"][key], actual["results"][key],
                                           tolerance["absolute"], tolerance["relative"], key)
    assert not differences, "\n".join(differences[:20]) + "\nFull results: " + str(artifacts)
