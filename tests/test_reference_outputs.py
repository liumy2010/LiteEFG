"""Compare each workspace-engine baseline layer with its reviewed reference."""

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("baseline_source", ["public", "workspace"])
def test_reference_outputs(baseline_source):
    artifacts = ROOT / "artifacts/reference-output-checks" / baseline_source
    result = subprocess.run(
        [sys.executable, str(ROOT / "tests/reference_output_check.py"), "check",
         "--baseline-source", baseline_source, "--artifacts", str(artifacts)],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
