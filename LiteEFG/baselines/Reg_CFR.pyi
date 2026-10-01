from __future__ import annotations
import typing
import LiteEFG as leg
from LiteEFG.baselines.baseline import _baseline
import math as math
__all__ = ['leg', 'graph', 'math']
class graph(leg.baselines.baseline._baseline):
    def __init__(self, kappa = 1.0, tau = 0.001, gamma = 0.001, regularizer: typing.Literal['Euclidean', 'Entropy'] = 'Euclidean', weighted = False, shrink_iter = 100000, out_reg = False):
        ...
    def _get_ev(self, gradient, ev, strategy, ref_strategy):
        ...
    def _update(self, gradient, upd_u, ref_u, is_stabilize = False):
        ...
    def current_strategy(self) -> leg._LiteEFG.GraphNode:
        ...
    def update_graph(self, env: leg._LiteEFG.Environment) -> None:
        ...
