from __future__ import annotations
import LiteEFG as leg
from LiteEFG.baselines.baseline import _baseline
__all__ = ['leg', 'graph']
class graph(leg.baselines.baseline._baseline):
    def __init__(self):
        ...
    def current_strategy(self) -> leg._LiteEFG.GraphNode:
        ...
    def update_graph(self, env: leg._LiteEFG.Environment) -> None:
        ...
