"""Check change-based CI coverage without importing JAX or the native engine."""

import json
import subprocess
from types import SimpleNamespace

import pytest

import select_ci_tests as selection


@pytest.mark.parametrize("path", [
    "LiteEFG/src/Computation/Graph.cpp", "LiteEFG/_graph_context.py",
    "LiteEFG/__init__.py", "LiteEFG/baselines/__init__.py", "setup.py",
    "pyproject.toml", "tests/requirements-drl.txt", "tests/conftest.py",
    ".github/workflows/reference-output-checks.yml", "unknown/new-file.py",
    "tests/select_ci_tests.py", "tests/test_ci_selection.py",
    "tests/test_deleted_module.py", "../outside.py",
])
def test_shared_unknown_and_missing_paths_cannot_reduce_coverage(path):
    plan = selection.select_tests([path])
    assert plan["mode"] == "full"
    assert plan["tests"] == ["tests"]
    assert plan["reasons"]


def test_docs_only_runs_all_executable_docs_without_heavy_drl_suites():
    plan = selection.select_tests(["docs/guide/deep-learning.md"])
    expected = {p.relative_to(selection.ROOT).as_posix()
                for p in (selection.ROOT / "tests").glob("test_docs_*.py")}
    assert plan["mode"] == "selected"
    assert set(plan["tests"]) == expected | {selection.SELF_TEST}


def test_native_baseline_keeps_references_checkpoints_and_parallel_coverage():
    plan = selection.select_tests(["LiteEFG/baselines/CFR.py"])
    assert plan["mode"] == "selected"
    assert {"tests/test_reference_outputs.py", "tests/test_baseline_arguments.py",
            "tests/test_checkpoint_baselines.py", "tests/test_parallel.py",
            "tests/test_docs_commands.py"} <= set(plan["tests"])
    assert not any(p.startswith("tests/test_drl_") for p in plan["tests"])


@pytest.mark.parametrize("path", ["LiteEFG/drl/runtime.py", "LiteEFG/drl/graph.py",
                                 "LiteEFG/drl/models/MLP.py", "LiteEFG/drl/new_module.py"])
def test_shared_drl_includes_every_drl_module_and_public_integration(path):
    plan = selection.select_tests([path])
    drl = {p.relative_to(selection.ROOT).as_posix()
           for p in (selection.ROOT / "tests").glob("test_drl_*.py")}
    assert plan["mode"] == "selected"
    assert drl | selection.DRL_SMOKE | {"tests/test_docs_drl.py"} <= set(plan["tests"])


def test_ppo_selects_transitive_microbatch_and_resident_users():
    plan = selection.select_tests(["LiteEFG/baselines/drl/PPO.py"])
    assert plan["mode"] == "selected"
    assert {"tests/test_drl_ppo.py", "tests/test_drl_shared_critic_gae.py",
            "tests/test_drl_compiled_updates.py", "tests/test_drl_microbatch.py",
            "tests/test_drl_point_microbatch.py", "tests/test_drl_resident.py",
            "tests/test_docs_drl.py"} <= set(plan["tests"])


def test_dark_chess_selects_rules_oracle_models_and_api_without_other_games():
    plan = selection.select_tests(["LiteEFG/drl/env/_dark_chess.py"])
    assert plan["mode"] == "selected"
    assert {"tests/test_drl_dark_chess.py", "tests/test_drl_dark_chess_oracle.py",
            "tests/test_drl_dark_chess_models.py", "tests/test_drl_environment.py"} <= set(plan["tests"])
    assert "tests/test_drl_goofspiel.py" not in plan["tests"]
    assert "tests/test_drl_microbatch.py" not in plan["tests"]


def test_goofspiel_includes_training_users_and_doc_examples():
    plan = selection.select_tests(["LiteEFG/drl/env/goofspiel.py"])
    assert plan["mode"] == "selected"
    assert {"tests/test_drl_goofspiel.py", "tests/test_drl_ppo.py",
            "tests/test_drl_runtime.py", "tests/test_drl_resident.py",
            "tests/test_docs_drl.py"} <= set(plan["tests"])


def test_test_module_changes_include_consumers_of_shared_helpers():
    plan = selection.select_tests(["tests/test_drl_compact_batches.py"])
    assert {"tests/test_drl_compact_batches.py", "tests/test_drl_microbatch.py",
            "tests/test_drl_point_microbatch.py", "tests/test_drl_resident.py"} <= set(plan["tests"])


def test_helper_dependency_walk_includes_indirect_and_subprocess_imports():
    sources = {
        "tests/test_first.py": "",
        "tests/helper.py": "from test_first import factory",
        "tests/test_second.py": "import helper",
        "tests/test_third.py": 'script = "from test_second import case"',
        "tests/test_unrelated.py": "",
    }
    assert selection.dependent_tests({"tests/test_first.py"}, sources) == {
        "tests/test_first.py", "tests/test_second.py", "tests/test_third.py"}


def test_multiple_changes_take_the_union_and_unknown_paths_override_it():
    paths = ["docs/index.md", "tests/test_drl_ppo.py"]
    expected = set().union(*(selection.select_tests([p])["tests"] for p in paths))
    assert set(selection.select_tests(paths + paths)["tests"]) == expected
    assert selection.select_tests(paths + ["setup.py"])["tests"] == ["tests"]
    assert selection.select_tests([])["tests"] == ["tests"]


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()

    def git(*args):
        return subprocess.run(["git", "-c", "user.name=CI selection test",
                               "-c", "user.email=ci@example.invalid", *args],
                              cwd=root, check=True, capture_output=True, text=True).stdout.strip()

    git("init", "-b", "main")

    def commit(path, content="text"):
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        git("add", "--all")
        git("commit", "-m", "fixture")
        return git("rev-parse", "HEAD")

    return root, git, commit


def test_push_compares_all_commits_in_the_event(repository):
    root, git, commit = repository
    base = commit("README.md")
    commit("LiteEFG/drl/runtime.py")
    head = commit("docs/index.md")
    assert set(selection.event_changes("push", {"before": base, "after": head}, root)) == {
        "LiteEFG/drl/runtime.py", "docs/index.md"}


def test_pull_request_uses_merge_base_and_excludes_base_branch_only_changes(repository):
    root, git, commit = repository
    commit("README.md")
    git("checkout", "-b", "feature")
    head = commit("docs/index.md")
    git("checkout", "main")
    base = commit("setup.py")
    event = {"pull_request": {"base": {"sha": base}, "head": {"sha": head}}}
    assert selection.event_changes("pull_request", event, root) == ["docs/index.md"]


def test_renames_include_both_source_and_destination(repository):
    root, git, commit = repository
    base = commit("LiteEFG/drl/env/dark_chess.py")
    (root / "docs").mkdir()
    git("mv", "LiteEFG/drl/env/dark_chess.py", "docs/example.py")
    head = commit("README.md")
    changed = selection.event_changes("push", {"before": base, "after": head}, root)
    assert "LiteEFG/drl/env/dark_chess.py" in changed
    assert "docs/example.py" in changed


@pytest.mark.parametrize("event_name,event", [
    ("push", {"before": "0" * 40, "after": "1" * 40}),
    ("push", {"before": "1" * 40, "after": "2" * 40}),
    ("pull_request", {}), ("push", {"before": "--help", "after": "2" * 40}),
    ("schedule", {}),
])
def test_incomplete_or_unavailable_history_falls_back_to_full(tmp_path, event_name, event):
    path = tmp_path / "event.json"
    path.write_text(json.dumps(event), encoding="utf-8")
    assert selection.plan_event(event_name, path, tmp_path)["tests"] == ["tests"]


def test_manual_missing_and_malformed_events_are_full(tmp_path):
    path = tmp_path / "event.json"
    assert selection.plan_event("workflow_dispatch", path)["tests"] == ["tests"]
    assert selection.plan_event("push", path)["tests"] == ["tests"]
    path.write_text("not json", encoding="utf-8")
    assert selection.plan_event("push", path)["tests"] == ["tests"]


def test_plan_artifact_and_summary_record_reasons(tmp_path, monkeypatch):
    output = tmp_path / "artifacts/plan.json"
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    plan = selection.select_tests(["docs/index.md"])
    selection.write_plan(plan, output)
    assert json.loads(output.read_text()) == plan
    assert plan["reasons"][0] in summary.read_text()
    assert b"\r\n" not in output.read_bytes()


def test_execution_passes_only_selected_modules_and_propagates_pytest_failure(tmp_path, monkeypatch):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"tests": [selection.SELF_TEST]}), encoding="utf-8")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(selection.subprocess, "run", run)
    assert selection.run_plan(path) == 1
    command, options = calls[0]
    assert command[-1] == selection.SELF_TEST
    assert "tests" not in command
    assert "--durations=20" in command
    assert options == {"cwd": selection.ROOT}


@pytest.mark.parametrize("tests", [[], ["--ignore=tests"], ["../outside.py"]])
def test_empty_or_invalid_plan_cannot_succeed_without_testing(tmp_path, tests):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"tests": tests}), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid or empty"):
        selection.run_plan(path)
