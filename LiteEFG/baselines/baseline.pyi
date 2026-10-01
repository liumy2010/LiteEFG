from __future__ import annotations
import LiteEFG as leg
__all__ = ['leg']
class _baseline(leg._LiteEFG.Graph):
    def __init__(self):
        ...
    def current_strategy(self) -> leg._LiteEFG.GraphNode:
        """
        
                    return the node representing the current strategy, which will be used to sample / compute the utility.
                
        """
    def update_graph(self, env: leg._LiteEFG.Environment) -> None:
        """
        
                    update the graph.
                
        """
