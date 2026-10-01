"""Optional JAX backend for transient local graphs and batched game sampling."""

from .graph import (
    DRLGraph, Node, Model, ModelList, model, minimize, const, exp, log, minimum, maximum,
    clip, stop_gradient, where, dot, mse, gather, masked_softmax, entropy, normalize, aggregate,
)
from .environment import Environment
from .env import Goofspiel, DarkChess
from .data import dataloader
from .runtime import Trainer, Policy, average_utility, uniform_policy
from .resident import PackedRollout

__all__ = [
    "DRLGraph", "Node", "Model", "ModelList", "Environment", "model", "minimize", "const", "exp", "log",
    "minimum", "maximum", "clip", "stop_gradient", "where", "dot", "mse",
    "gather", "masked_softmax", "entropy", "normalize", "aggregate", "Goofspiel", "DarkChess", "Trainer",
    "Policy", "PackedRollout", "average_utility", "uniform_policy", "dataloader",
]
