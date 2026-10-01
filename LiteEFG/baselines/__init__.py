"""Tabular baselines; neural baselines are available in the drl subpackage."""

import importlib
from pathlib import Path

for _source in Path(__file__).parent.glob("*.py"):
    if _source.stem not in ("__init__", "utils"):
        importlib.import_module(f".{_source.stem}", package=__name__)
