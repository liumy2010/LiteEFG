from __future__ import annotations
from typing import overload
from LiteEFG._LiteEFG import Environment
from LiteEFG._LiteEFG import FileEnv
from LiteEFG._LiteEFG import Graph
from LiteEFG._LiteEFG import GraphNode
from LiteEFG._LiteEFG import GraphNodeStatus
from LiteEFG._LiteEFG import Vector
from LiteEFG._LiteEFG import argmax
from LiteEFG._LiteEFG import argmin
from LiteEFG._LiteEFG import backward
from LiteEFG._LiteEFG import cat
from LiteEFG._LiteEFG import euclidean
from LiteEFG._LiteEFG import forward
from LiteEFG._LiteEFG import get_threads
from LiteEFG._LiteEFG import negative_entropy
from LiteEFG._LiteEFG import project
from LiteEFG._LiteEFG import set_seed
from LiteEFG._LiteEFG import set_threads
from LiteEFG.src.Environment.OpenSpiel.OpenSpielToGameFile import OpenSpielEnv
from LiteEFG.checkpoint import load_checkpoint as load_checkpoint
from LiteEFG.checkpoint import save_checkpoint as save_checkpoint
from LiteEFG.cpp_game import CppEnv as CppEnv
from LiteEFG.cpp_game import compile_cpp_game as compile_cpp_game
from LiteEFG.drl.graph import Node as _DRLNode
from LiteEFG.drl.graph import DRLGraph as DRLGraph
from LiteEFG.drl.graph import Model as Model
from LiteEFG.drl.graph import ModelList as ModelList
from LiteEFG.drl.graph import model as model
from LiteEFG.drl.graph import minimize as minimize
from LiteEFG.drl.graph import clip as clip
from LiteEFG.drl.graph import stop_gradient as stop_gradient
from LiteEFG.drl.graph import masked_softmax as masked_softmax
from LiteEFG.drl.graph import entropy as entropy
from LiteEFG.drl.graph import gather as gather
from LiteEFG.drl.graph import mse as mse
from LiteEFG.drl.graph import where as where
from LiteEFG.drl.env import Goofspiel as Goofspiel
from LiteEFG.drl.env import DarkChess as DarkChess
from LiteEFG.drl.data import dataloader as dataloader
from LiteEFG.drl.runtime import Trainer as Trainer
from LiteEFG.drl.runtime import Policy as Policy
from LiteEFG.drl.resident import PackedRollout as PackedRollout
from LiteEFG.drl.runtime import average_utility as average_utility
from LiteEFG.drl.runtime import uniform_policy as uniform_policy
from . import _LiteEFG
from . import baselines
from . import random
from . import src

@overload
def aggregate(arg0: GraphNode, aggregator: str = 'sum', object: str = 'children', player: str = 'self', padding: float = 0.0) -> GraphNode: ...
@overload
def aggregate(arg0: _DRLNode, aggregator: str = 'sum', object: str = 'children', player: str = 'self', padding: float = 0.0, *, discount: float = 1.0, decay: float = 1.0, include_self: bool = False) -> _DRLNode: ...
@overload
def const(size: GraphNode, val: int | float | GraphNode) -> GraphNode: ...
@overload
def const(size: int, val: GraphNode) -> GraphNode: ...
@overload
def const(size: int, val: _DRLNode) -> _DRLNode: ...
@overload
def const(size: int, val: int | float) -> GraphNode | _DRLNode: ...

@overload
def copy(arg0: GraphNode) -> GraphNode: ...
@overload
def copy(arg0: _DRLNode) -> _DRLNode: ...
@overload
def dot(arg0: GraphNode, arg1: GraphNode) -> GraphNode: ...
@overload
def dot(arg0: _DRLNode, arg1: _DRLNode) -> _DRLNode: ...
@overload
def exp(arg0: GraphNode) -> GraphNode: ...
@overload
def exp(arg0: _DRLNode) -> _DRLNode: ...
@overload
def log(arg0: GraphNode) -> GraphNode: ...
@overload
def log(arg0: _DRLNode) -> _DRLNode: ...
@overload
def max(arg0: GraphNode) -> GraphNode: ...
@overload
def max(arg0: _DRLNode) -> _DRLNode: ...
@overload
def maximum(arg0: GraphNode, arg1: int | float | GraphNode) -> GraphNode: ...
@overload
def maximum(arg0: _DRLNode, arg1: int | float | _DRLNode) -> _DRLNode: ...
@overload
def maximum(arg0: int | float, arg1: _DRLNode) -> _DRLNode: ...
@overload
def mean(arg0: GraphNode) -> GraphNode: ...
@overload
def mean(arg0: _DRLNode) -> _DRLNode: ...
@overload
def min(arg0: GraphNode) -> GraphNode: ...
@overload
def min(arg0: _DRLNode) -> _DRLNode: ...
@overload
def minimum(arg0: GraphNode, arg1: int | float | GraphNode) -> GraphNode: ...
@overload
def minimum(arg0: _DRLNode, arg1: int | float | _DRLNode) -> _DRLNode: ...
@overload
def minimum(arg0: int | float, arg1: _DRLNode) -> _DRLNode: ...
@overload
def normalize(arg0: GraphNode, p_norm: float, ignore_negative: bool = False) -> GraphNode: ...
@overload
def normalize(arg0: _DRLNode, p_norm: float = 1.0, ignore_negative: bool = False) -> _DRLNode: ...
@overload
def sum(arg0: GraphNode) -> GraphNode: ...
@overload
def sum(arg0: _DRLNode) -> _DRLNode: ...

__all__ = [
    'CppEnv', 'Environment', 'FileEnv', 'Graph', 'GraphNode', 'GraphNodeStatus',
    'OpenSpielEnv', 'Vector', 'aggregate', 'argmax', 'argmin', 'backward',
    'baselines', 'cat', 'compile_cpp_game', 'const', 'copy', 'dot', 'euclidean',
    'exp', 'forward', 'get_threads', 'load_checkpoint', 'log', 'max', 'maximum',
    'mean', 'min', 'minimum', 'negative_entropy', 'normalize', 'project',
    'random', 'save_checkpoint', 'set_seed', 'set_threads', 'src', 'sum',
    'DRLGraph', 'Model', 'ModelList', 'model', 'minimize', 'clip', 'stop_gradient',
    'masked_softmax', 'entropy', 'gather', 'mse', 'where', 'Goofspiel', 'DarkChess',
    'Trainer', 'Policy', 'PackedRollout', 'average_utility', 'uniform_policy', 'dataloader',
]
