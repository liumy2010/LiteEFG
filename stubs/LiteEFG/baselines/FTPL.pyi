import LiteEFG as LiteEFG
from __future__ import annotations
__all__ = ['LiteEFG', 'graph']
class graph(LiteEFG._LiteEFG.Graph):
    def __init__(self, eta = 0.01):
        ...
    def current_strategy(self) -> LiteEFG._LiteEFG.GraphNode:
        ...
    def update_graph(self, env: LiteEFG._LiteEFG.Environment) -> None:
        ...
