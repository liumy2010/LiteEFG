"""Conservatively select pytest modules from a push or pull-request diff.

This module uses only the standard library, so selection runs before pip install.
Unreviewed source paths and unavailable diffs always select the full suite.
"""

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SELF_TEST = "tests/test_ci_selection.py"
SHA = re.compile(r"[0-9a-fA-F]{40}(?:[0-9a-fA-F]{24})?")
DRL_SMOKE = {
    "tests/test_drl_dispatch.py", "tests/test_drl_environment.py",
    "tests/test_graph_context.py", "tests/test_graph_owner_checkpoint.py",
    "tests/test_type_stubs.py",
}


def test_sources(root):
    return {path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
            for path in sorted((root / "tests").glob("*.py"))}


def dependent_tests(seeds, sources):
    """Include transitive users of test helpers, including subprocess strings.

Matching module names anywhere in the source deliberately over-selects rather
than missing imports embedded in subprocess scripts or importlib calls.
"""
    affected = set(seeds)
    pending = list(seeds)
    while pending:
        pattern = re.compile(r"\b" + re.escape(PurePosixPath(pending.pop()).stem) + r"\b")
        for path, source in sources.items():
            if path not in affected and pattern.search(source):
                affected.add(path)
                pending.append(path)
    return {path for path in affected if PurePosixPath(path).name.startswith("test_")}


def full_plan(reason, changed=()):
    return {"mode": "full", "changed_files": sorted(set(changed)),
            "reasons": [reason], "tests": ["tests"]}


def select_tests(changed, root=ROOT):
    changed = sorted(set(changed))
    if not changed:
        return full_plan("No changed paths available; run the full suite.")
    sources = test_sources(root)
    tests = {path for path in sources if PurePosixPath(path).name.startswith("test_")}
    docs = {path for path in tests if path.startswith("tests/test_docs_")}
    drl = {path for path in tests if path.startswith("tests/test_drl_")} | docs | DRL_SMOKE
    native = {path for path in tests if not path.startswith("tests/test_drl_")
              and path != "tests/test_docs_drl.py"}
    selected = {SELF_TEST}
    reasons = []

    for path in changed:
        if (PurePosixPath(path).is_absolute() or ".." in PurePosixPath(path).parts
                or "\\" in path):
            return full_plan(f"Unrecognized path: {path!r}.", changed)
        if path in {"tests/select_ci_tests.py", SELF_TEST}:
            return full_plan(f"Test selection logic changed: {path}.", changed)
        if path.startswith("docs/") or PurePosixPath(path).name == "README.md":
            group, label = docs, "documentation examples"
        elif path == "tests/docs_examples.py":
            group, label = docs, "documentation harness and its users"
        elif path in sources and PurePosixPath(path).name.startswith("test_"):
            group, label = {path}, "changed test and transitive helper users"
        elif path.startswith("tests/baselines/") or path == "tests/reference_output_check.py":
            group, label = native, "native reference output checks, checkpoints, parallelism and docs"
        elif path in {"LiteEFG/drl/env/dark_chess.py", "LiteEFG/drl/env/_dark_chess.py",
                      "LiteEFG/drl/env/goofspiel.py", "LiteEFG/baselines/drl/PPO.py"}:
            symbols = (r"DarkChess|dark_chess" if "dark_chess" in path else
                       r"Goofspiel|LiteEFG\.drl\.env\.goofspiel" if "goofspiel" in path else
                       r"PPO|baselines\.drl")
            group = {name for name, source in sources.items() if re.search(symbols, source)}
            group |= docs | DRL_SMOKE
            label = "game/algorithm users, documentation and DRL integration checks"
        elif path.startswith(("LiteEFG/drl/", "LiteEFG/baselines/drl/")):
            group, label = drl, "all DRL and integration tests"
        elif (path.startswith("LiteEFG/baselines/") and path.endswith(".py")
              and PurePosixPath(path).name != "__init__.py"):
            group, label = native, "native baselines and their integration tests"
        else:
            return full_plan(f"Shared code, configuration or unreviewed path: {path}.", changed)
        selected |= dependent_tests(group, sources)
        reasons.append(f"{path}: {label}")

    # A stale rule must not silently lose coverage when tests are removed/renamed.
    if not selected <= tests:
        return full_plan("A selected test is missing; review the routing rules.", changed)
    return {"mode": "selected", "changed_files": changed,
            "reasons": reasons, "tests": sorted(selected)}


def git(root, *args):
    result = subprocess.run(["git", *args], cwd=root, check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    return result.stdout


def event_changes(event_name, event, root=ROOT):
    """Return both sides of renames; use PR merge-base or the whole push range."""
    if event_name == "pull_request":
        base = event["pull_request"]["base"]["sha"]
        head = event["pull_request"]["head"]["sha"]
    elif event_name == "push":
        base, head = event["before"], event["after"]
    else:
        raise ValueError(f"{event_name or 'missing event'} requests a full run")
    if any(not isinstance(sha, str) or not SHA.fullmatch(sha) or set(sha) == {"0"}
           for sha in (base, head)):
        raise ValueError("Missing comparison commit (including a new branch)")
    if event_name == "pull_request":
        base = git(root, "merge-base", base, head).decode().strip()
    raw = git(root, "diff", "--no-renames", "--name-only", "-z", base, head, "--")
    return [path.decode("utf-8") for path in raw.split(b"\0") if path]


def plan_event(event_name, event_path, root=ROOT):
    if event_name == "workflow_dispatch":
        return full_plan("Manual workflow runs always run the full suite.")
    try:
        event = json.loads(Path(event_path).read_text(encoding="utf-8"))
        return select_tests(event_changes(event_name, event, root), root)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        return full_plan(f"Cannot safely determine the affected tests: {error}")


def write_plan(plan, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(plan, indent=2, ensure_ascii=True) + "\n"
    output.write_text(payload, encoding="utf-8", newline="\n")
    print(payload, end="", flush=True)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write("## Reference output checks: test selection\n\n```json\n" + payload + "```\n")


def run_plan(path, root=ROOT):
    plan = json.loads(path.read_text(encoding="utf-8"))
    tests = plan["tests"]
    valid = {p.relative_to(root).as_posix() for p in (root / "tests").glob("test_*.py")}
    if tests != ["tests"] and (not tests or not set(tests) <= valid):
        raise ValueError("Invalid or empty test selection")
    report = root / "artifacts/ci-selection/junit.xml"
    report.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "pytest", "-vv", "--durations=20",
               f"--junitxml={report}", *tests]
    print("Running:", " ".join(command), flush=True)
    return subprocess.run(command, cwd=root).returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--changed-files", nargs="+", help="Preview routing for explicit paths")
    mode.add_argument("--run-plan", type=Path, help="Execute an existing selection")
    parser.add_argument("--output", type=Path, default=Path("artifacts/ci-selection/plan.json"))
    args = parser.parse_args()
    if args.run_plan:
        return run_plan(args.run_plan)
    if args.changed_files:
        plan = select_tests(args.changed_files)
    else:
        plan = plan_event(os.environ.get("GITHUB_EVENT_NAME", ""),
                          os.environ.get("GITHUB_EVENT_PATH", ""))
    write_plan(plan, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
