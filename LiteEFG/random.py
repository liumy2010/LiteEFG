from __future__ import annotations

from . import _LiteEFG

GraphNode = _LiteEFG.GraphNode

__all__ = ["uniform", "normal", "exponential"]


def _get_binding(name: str):
    binding = getattr(_LiteEFG, name, None)
    if binding is None:
        raise RuntimeError(
            "LiteEFG random bindings are unavailable. Rebuild LiteEFG to use LiteEFG.random."
        )
    return binding


def uniform(node: GraphNode, lower: float = 0.0, upper: float = 1.0) -> GraphNode:
    return _get_binding("_uniform")(node, lower, upper)


def normal(node: GraphNode, mean: float = 0.0, stddev: float = 1.0) -> GraphNode:
    return _get_binding("_normal")(node, mean, stddev)


def exponential(node: GraphNode, lambda_: float = 1.0) -> GraphNode:
    return _get_binding("_exponential")(node, lambda_)
