import LiteEFG as leg

class _baseline(leg.Graph):
    def __init__(self):
        super().__init__()

    def current_strategy(self) -> leg.GraphNode:
        """
            return the node representing the current strategy, which will be used to sample / compute the utility.
        """
        raise NotImplementedError("current_strategy method should be implemented by the baseline")

    def update_graph(self, env: leg.Environment) -> None:
        """
            update the graph.
        """
        raise NotImplementedError("update_graph method should be implemented by the baseline")
