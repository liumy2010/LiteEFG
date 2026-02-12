import LiteEFG as LiteEFG
from LiteEFG.baselines.baseline import _baseline
from __future__ import annotations
__all__ = ['LiteEFG', 'graph']
class graph(LiteEFG.baselines.baseline._baseline):
    def __init__(self, eta = 0.01, noise_type = 'exponential'):
        ...
    def current_strategy(self) -> LiteEFG._LiteEFG.GraphNode:
        ...
    def update_graph(self, env: LiteEFG._LiteEFG.Environment) -> None:
        ...
