"""Execute the documented CLI recipes at their original iteration counts."""

import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from docs_examples import ROOT, extract_fenced_blocks


# Use a fresh interpreter for each native graph, keep generated games/CSVs in
# pytest's temporary directory, and emulate Python's script-directory imports.
SCRIPT_RUNNER = """
import os
import runpy
import sys
from pathlib import Path

expanduser = os.path.expanduser
os.path.expanduser = lambda path: os.getcwd() if path == '~' else expanduser(path)
sys.argv = sys.argv[1:]
sys.path.insert(0, str(Path(sys.argv[0]).parent))
runpy.run_path(sys.argv[0], run_name='__main__')
"""


def run_script(argv, tmp_path):
    environ = os.environ.copy()
    environ["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(ROOT), environ.get("PYTHONPATH")])
    )
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT_RUNNER, *argv],
        cwd=tmp_path, env=environ, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


COMMANDS = extract_fenced_blocks("docs/guide/examples.md", "sh")[1:]


@pytest.mark.parametrize("command", COMMANDS, ids=["CFR", "External-CFR", "OS-MCCFR", "QFR", "CFRplus-Leduc"])
def test_documented_baseline_command(command, tmp_path):
    argv = shlex.split(command.replace("\\\n", ""))
    assert argv.pop(0) == "python"
    argv[0] = str(ROOT / argv[0])
    result = run_script(argv, tmp_path)
    reports = re.findall(r"Exploitability: ([^,\s]+)", result.stderr)
    assert reports, result.stderr
    gaps = np.array(reports, dtype=float)
    assert np.isfinite(gaps).all()
    assert (gaps >= -1e-8).all()
    if Path(argv[0]).stem in {"CFR", "OS_MCCFR"}:
        for player in range(2):
            frame = pd.read_csv(tmp_path / f"strategy_{player}.csv", index_col=0)
            assert len(frame) == 6
            assert "Infoset" in frame
            probabilities = frame.drop(columns="Infoset").to_numpy(dtype=float)
            assert np.isfinite(probabilities).all()
            assert (probabilities >= 0).all()
            np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-10)


def test_documented_quick_start_command(tmp_path):
    path = tmp_path / "quick_start.py"
    path.write_text("\n".join(extract_fenced_blocks("docs/guide/quick-start.md")))
    command, = extract_fenced_blocks("docs/guide/quick-start.md", "sh")
    executable, script = shlex.split(command)
    assert executable == "python"
    result = run_script([str(tmp_path / script)], tmp_path)
    assert "1000 per-player exploitability:" in result.stdout
    for player in range(2):
        assert len(pd.read_csv(tmp_path / f"kuhn_player_{player}.csv")) == 6


def test_documented_import_check(tmp_path):
    command = next(
        block for block in extract_fenced_blocks("docs/guide/installation.md", "sh")
        if block.startswith('python -c ')
    )
    argv = shlex.split(command)
    assert argv[:2] == ["python", "-c"]
    environ = os.environ.copy()
    environ["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(ROOT), environ.get("PYTHONPATH")])
    )
    result = subprocess.run(
        [sys.executable, *argv[1:]], cwd=tmp_path, env=environ,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert str(ROOT / "LiteEFG" / "__init__.py") in result.stdout
    assert "kuhn_poker" in result.stdout


def test_drl_install_commands_are_explicit_setup_steps():
    # Installations are setup steps. Check the documented dependencies without
    # running pip; test_shell_syntax checks each complete shell fence with bash -n.
    commands = extract_fenced_blocks("docs/guide/installation.md", "sh")
    drl_commands = [
        command.strip() for command in commands
        if command.startswith(("python -m pip install LiteEFG", "python -m pip install ."))
    ]
    assert drl_commands == [
        "python -m pip install LiteEFG",
        "python -m pip install LiteEFG 'jax[cuda12]>=0.6.2,<0.8'",
        "python -m pip install LiteEFG 'jax[cuda13]>=0.7.2,<0.8'",
        "python -m pip install .",
        "python -m pip install . 'jax[cuda12]>=0.6.2,<0.8'",
        "python -m pip install . 'jax[cuda13]>=0.7.2,<0.8'",
    ]


SHELL_BLOCKS = [
    (path.relative_to(ROOT), index, source)
    for path in sorted((ROOT / "docs").rglob("*.md"))
    for index, source in enumerate(extract_fenced_blocks(path, "sh"))
]


@pytest.mark.parametrize(
    "path,index,source", SHELL_BLOCKS,
    ids=[f"{path}:{index + 1}" for path, index, _ in SHELL_BLOCKS],
)
def test_shell_syntax(path, index, source):
    # Installation/server recipes are checked separately as environment setup;
    # do not reinstall dependencies or start indefinite servers inside pytest.
    result = subprocess.run(
        ["bash", "-n"], input=source, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, f"{path} block {index + 1}: {result.stderr}"
