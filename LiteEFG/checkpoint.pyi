from __future__ import annotations

import os

from ._LiteEFG import Environment, Graph

__all__ = ["save_checkpoint", "load_checkpoint"]

def save_checkpoint(path: str | os.PathLike[str], env: Environment, graph: Graph) -> None: ...
def load_checkpoint(path: str | os.PathLike[str]) -> tuple[Environment, Graph]: ...
