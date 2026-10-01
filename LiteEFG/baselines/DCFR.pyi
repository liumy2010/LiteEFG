from __future__ import annotations
import LiteEFG as leg
from LiteEFG.baselines.baseline import _baseline
__all__ = ['leg', 'graph']
class graph(leg.baselines.baseline._baseline):
    def __init__(self, alpha = 1.5, beta = 0, gamma = 2):
        ...
    def current_strategy(self, type_name = 'last-iterate') -> leg.GraphNode:
        ...
    def update_graph(self, env: leg._LiteEFG.Environment) -> None:
        ...
