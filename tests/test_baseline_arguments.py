"""Regress every reviewed combination of algorithm-changing baseline arguments."""

import argparse
import importlib
import importlib.util
import inspect
import itertools
import json
from pathlib import Path
import runpy
import sys
from typing import Literal, get_args, get_origin, get_type_hints
from unittest.mock import patch

import pytest


SPEC = importlib.util.spec_from_file_location(
    "baseline_argument_pipeline", Path(__file__).with_name("reference_output_check.py"))
pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pipeline)
MANIFEST_PATH = pipeline.ROOT / "tests/baselines/manifest-arguments.json"
REFERENCE = MANIFEST_PATH.with_name("reference-arguments-" + sys.platform + ".json")
ARTIFACTS = pipeline.ARTIFACTS / "arguments"
MANIFEST = pipeline.load_manifest(MANIFEST_PATH)

# This independently reviewed specification must not be generated from the
# manifest: dropping a value or a combination there must fail a coverage test.
# Numerical axes cover branches/schedules, rather than arbitrary learning rates.
REGULARIZERS = ("Entropy", "Euclidean")
FEEDBACK = ("Q", "traj-Q", "counterfactual", "Outcome")
BOOLEAN = (False, True)
REVIEWED_AXES = {
    "Balanced_FTRL": {},
    "Balanced_OMD": {},
    "CFR": {},
    "CFRplus": {},
    "CMD": {"regularizer": REGULARIZERS, "weighted": BOOLEAN, "inner_epoch": (1, 10)},
    "DCFR": {"alpha": (-11, 1.5, 11), "beta": (-11, 0, 11)},
    "DOMD": {"regularizer": REGULARIZERS, "weighted": BOOLEAN},
    "FTPL": {"noise_type": ("exponential", "uniform", "normal")},
    "IXOMD": {},
    "MMD": {"regularizer": REGULARIZERS, "feedback": FEEDBACK, "weighted": BOOLEAN},
    "OS_MCCFR": {"rm_plus": BOOLEAN, "balanced": BOOLEAN},
    "PCFR": {},
    "QFR": {"regularizer": REGULARIZERS, "feedback": FEEDBACK, "weighted": BOOLEAN},
    "Reg_CFR": {"regularizer": REGULARIZERS, "weighted": BOOLEAN,
                "out_reg": BOOLEAN, "shrink_iter": (1, 10, 100000)},
    "Reg_DOMD": {"regularizer": REGULARIZERS, "weighted": BOOLEAN,
                 "out_reg": BOOLEAN, "shrink_iter": (1, 10, 100000)},
}
OUTCOME_BASELINES = {"Balanced_FTRL", "Balanced_OMD", "IXOMD", "OS_MCCFR"}
CLI_RUNTIME_ARGUMENTS = {"help", "game", "traverse_type", "iter", "print_freq"}
CLI_ALIASES = {("FTPL", "noise"): "noise_type"}


def canonical(value):
    """Retain JSON types, including the distinction between booleans and numbers."""
    return json.dumps(value, sort_keys=True, allow_nan=False)


@pytest.fixture(scope="module")
def constructor_defaults():
    directory = pipeline.activate_package(baseline_source="workspace")
    return pipeline.baseline_defaults(MANIFEST, directory)


def test_argument_manifest_preserves_benchmarks():
    defaults = pipeline.load_manifest()
    assert {key: value for key, value in MANIFEST.items() if key != "baselines"} == {
        key: value for key, value in defaults.items() if key != "baselines"}


def test_argument_manifest_covers_all_baseline_modules(constructor_defaults):
    assert set(constructor_defaults) == set(REVIEWED_AXES), (
        "Review the argument axes when adding or removing a baseline")


@pytest.mark.parametrize("module_name", REVIEWED_AXES)
def test_argument_manifest_covers_cartesian_product(module_name, constructor_defaults):
    axes = REVIEWED_AXES[module_name]
    defaults = constructor_defaults[module_name]
    assert set(axes) <= set(defaults), module_name + ": unknown reviewed argument"
    expected = {
        canonical(dict(defaults, **dict(zip(axes, values))))
        for values in itertools.product(*axes.values())
    }
    observed = []
    module = importlib.import_module("LiteEFG.baselines." + module_name)
    for case in MANIFEST["baselines"]:
        if case["module"] != module_name:
            continue
        inspect.signature(module.graph).bind(**case["kwargs"])
        effective = dict(defaults, **case["kwargs"])
        if case["id"] == "CFR_external":
            assert module_name == "CFR" and case["traversal"] == "External"
            assert effective == defaults
            continue
        traversal = ("Outcome" if module_name in OUTCOME_BASELINES
                     or effective.get("feedback") == "Outcome" else "Enumerate")
        assert case["traversal"] == traversal, case["id"] + ": wrong traversal"
        observed.append(canonical(effective))
    assert len(observed) == len(set(observed)), module_name + ": duplicate combinations"
    missing, extra = expected - set(observed), set(observed) - expected
    assert not missing and not extra, (
        module_name + ": missing combinations=" + repr(sorted(missing))
        + ", unexpected combinations=" + repr(sorted(extra)))
    if module_name == "CFR":
        assert any(case["id"] == "CFR_external" for case in MANIFEST["baselines"])


def cli_switches(source):
    """Inspect argparse declarations, stopping before CLI training can begin."""
    actions = []

    class ParserInspected(Exception):
        pass

    def stop_at_parse(parser, *args, **kwargs):
        actions.extend(parser._actions)
        raise ParserInspected

    with patch.object(argparse.ArgumentParser, "parse_args", stop_at_parse):
        with pytest.raises(ParserInspected):
            runpy.run_path(str(source), run_name="__main__")
    return actions


@pytest.mark.parametrize("module_name", REVIEWED_AXES)
def test_cli_algorithm_defaults_match_constructor(module_name, constructor_defaults):
    module = importlib.import_module("LiteEFG.baselines." + module_name)
    defaults = constructor_defaults[module_name]
    cli_defaults = [
        (CLI_ALIASES.get((module_name, action.dest), action.dest), action.default)
        for action in cli_switches(Path(module.__file__))
        if action.dest not in CLI_RUNTIME_ARGUMENTS
    ]
    assert {name for name, _ in cli_defaults} == set(defaults), (
        module_name + ": CLI and constructor expose different algorithm arguments")
    # Both spellings of a shared destination must agree with its constructor.
    for name, value in cli_defaults:
        assert canonical(value) == canonical(defaults[name]), (
            module_name + "." + name + ": CLI default " + repr(value)
            + " differs from constructor default " + repr(defaults[name]))


@pytest.mark.parametrize("module_name", REVIEWED_AXES)
def test_argument_manifest_reviews_declared_switches(module_name, constructor_defaults):
    module = importlib.import_module("LiteEFG.baselines." + module_name)
    parameters = inspect.signature(module.graph).parameters
    annotations = get_type_hints(module.graph.__init__)
    axes = REVIEWED_AXES[module_name]
    declared = {}
    for name, parameter in parameters.items():
        annotation = annotations.get(name)
        if isinstance(parameter.default, bool) or annotation is bool:
            declared[name] = BOOLEAN
        elif get_origin(annotation) is Literal:
            declared[name] = get_args(annotation)
    for action in cli_switches(Path(module.__file__)):
        if action.dest in CLI_RUNTIME_ARGUMENTS:
            continue
        name = CLI_ALIASES.get((module_name, action.dest), action.dest)
        if action.choices is not None:
            values = tuple(action.choices)
        elif isinstance(action, (argparse._StoreTrueAction, argparse._StoreFalseAction,
                                 argparse.BooleanOptionalAction)):
            values = BOOLEAN
        else:
            continue
        assert name in parameters, module_name + ": review CLI switch " + action.dest
        if name in declared:
            assert {canonical(value) for value in declared[name]} == {
                canonical(value) for value in values}, module_name + ": CLI/constructor mismatch"
        declared[name] = values
    for name, values in declared.items():
        assert name in axes, module_name + ": unreviewed algorithm switch " + name
        assert {canonical(value) for value in axes[name]} == {
            canonical(value) for value in values}, module_name + "." + name + ": uncovered values"


@pytest.fixture(scope="module")
def argument_snapshots():
    if not REFERENCE.is_file():
        pytest.fail("No reviewed argument reference for this platform: " + str(REFERENCE)
                    + ". Record it explicitly; tests never create references.")
    expected = pipeline.read_json(REFERENCE)
    pipeline.validate_source(expected, "workspace", "workspace")
    invalid = expected["errors"] + pipeline.validate_snapshot(expected)
    assert not invalid, "Invalid argument reference:\n" + "\n".join(invalid)
    contract = expected["contract"]
    before_collection = pipeline.compare_values(contract["manifest"], MANIFEST, 0.0, 0.0,
                                                "contract.manifest")
    before_collection += pipeline.compare_values(contract["runtime"],
                                                 pipeline.runtime_contract(), 0.0, 0.0,
                                                 "contract.runtime")
    assert not before_collection, "Argument reference mismatch:\n" + "\n".join(before_collection)

    # One collection per test session; each case/game gets a fresh seeded worker.
    actual = pipeline.collect_snapshot(MANIFEST, jobs=4, baseline_source="workspace")
    pipeline.write_json(ARTIFACTS / "actual.json", actual)
    pipeline.validate_source(actual, "workspace", "workspace")
    differences = (actual["errors"] + pipeline.validate_snapshot(actual)
                   + pipeline.compare_snapshots(expected, actual))
    pipeline.write_json(ARTIFACTS / "comparison.json", {
        "passed": not differences, "reference": str(REFERENCE),
        "case_count": len(actual["results"]), "differences": differences,
    })
    contract_diff = pipeline.compare_values(contract, actual["contract"], 0.0, 0.0,
                                            "contract")
    assert not contract_diff, "Argument contract mismatch:\n" + "\n".join(contract_diff)
    keys = {case["id"] + "/" + game["id"] for case in MANIFEST["baselines"]
            for game in MANIFEST["games"]}
    assert not set(actual["results"]) - keys, "Unexpected argument worker results"
    unassigned_errors = [error for error in actual["errors"]
                         if not any(error.startswith(key + ":") for key in keys)]
    assert not unassigned_errors, "\n".join(unassigned_errors)
    return expected, actual


@pytest.mark.parametrize("case,game", [
    pytest.param(case, game, id=case["id"] + "/" + game["id"])
    for case in MANIFEST["baselines"] for game in MANIFEST["games"]
])
def test_baseline_argument_outputs(case, game, argument_snapshots):
    expected, actual = argument_snapshots
    key = case["id"] + "/" + game["id"]
    errors = [error for error in actual["errors"] if error.startswith(key + ":")]
    assert not errors, "\n".join(errors)
    assert key in actual["results"], key + ": missing worker output"
    # Validate each result separately so a failed worker does not hide the other
    # configurations behind a shared fixture failure.
    single_case = dict(actual, contract=dict(actual["contract"], manifest=dict(
        MANIFEST, baselines=[case], games=[game])), results={key: actual["results"][key]})
    differences = pipeline.validate_snapshot(single_case)
    tolerance = expected["contract"]["manifest"]["tolerances"]
    differences += pipeline.compare_values(expected["results"][key], actual["results"][key],
                                           tolerance["absolute"], tolerance["relative"], key)
    assert not differences, "\n".join(differences[:20]) + "\nFull results: " + str(ARTIFACTS)
