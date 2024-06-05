import LiteEFG as LiteEFG
from __future__ import annotations
__all__ = ['LiteEFG', 'graph']
class graph(LiteEFG._LiteEFG.Graph):
    def __init__(self, eta = 0.1, inner_epoch = 10, regularizer: typing.Literal['Euclidean', 'Entropy'] = 'Entropy', weighted = False):
        ...
    def _update(self, gradient, upd_u, ref_u):
        ...
    def current_strategy(self) -> LiteEFG._LiteEFG.GraphNode:
        ...
    def get_ev(self, gradient, ev, strategy, ref_strategy):
        ...
    def update_graph(self, env: LiteEFG._LiteEFG.Environment) -> None:
        ...
