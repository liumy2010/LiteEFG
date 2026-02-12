from __future__ import annotations

from LiteEFG._LiteEFG import GraphNode

__all__ = ['uniform', 'normal', 'exponential']


def uniform(node: GraphNode, lower: float = 0.0, upper: float = 1.0) -> GraphNode:
    ...


def normal(node: GraphNode, mean: float = 0.0, stddev: float = 1.0) -> GraphNode:
    ...


def exponential(node: GraphNode, lambda_: float = 1.0) -> GraphNode:
    ...
