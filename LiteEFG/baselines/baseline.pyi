import LiteEFG as LiteEFG
from __future__ import annotations
__all__ = ['LiteEFG']
class _baseline(LiteEFG._LiteEFG.Graph):
    def __init__(self):
        ...
    def current_strategy(self) -> LiteEFG._LiteEFG.GraphNode:
        """
        
                    return the node representing the current strategy, which will be used to sample / compute the utility.
                
        """
    def update_graph(self, env: LiteEFG._LiteEFG.Environment) -> None:
        """
        
                    update the graph.
                
        """
