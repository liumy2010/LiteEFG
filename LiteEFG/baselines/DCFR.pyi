import LiteEFG as LiteEFG
from LiteEFG.baselines.baseline import _baseline
from __future__ import annotations
__all__ = ['LiteEFG', 'graph']
class graph(LiteEFG.baselines.baseline._baseline):
    def __init__(self, alpha = 1.5, beta = 0, gamma = 2):
        ...
    def current_strategy(self, type_name = 'last-iterate'):
        ...
    def update_graph(self, env: LiteEFG._LiteEFG.Environment) -> None:
        ...
