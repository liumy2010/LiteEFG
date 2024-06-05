import LiteEFG as LiteEFG
from __future__ import annotations
import numpy as np
import open_spiel.python.policy
from open_spiel.python.policy import TabularPolicy
import os as os
import pandas as pd
import pyspiel as pyspiel
import typing as typing
__all__ = ['InfosetName', 'LiteEFG', 'NodeName', 'OpenSpielEnv', 'TabularPolicy', 'np', 'os', 'pd', 'pyspiel', 'typing']
class OpenSpielEnv(LiteEFG._LiteEFG.FileEnv):
    def __init__(self, game: pyspiel.Game, traverse_type = 'Enumerate', regenerate = False):
        ...
    def get_strategy(self, strategy_node: LiteEFG._LiteEFG.GraphNode, type_name = 'default') -> typing.Tuple[open_spiel.python.policy.TabularPolicy, typing.List[pandas.core.frame.DataFrame]]:
        ...
    def get_value(self, player: int, node: LiteEFG._LiteEFG.GraphNode) -> typing.List[typing.Tuple[str, float]]:
        ...
    def interact(self, policy: open_spiel.python.policy.TabularPolicy, controlled_player = 0, reveal_private = True, epochs = 1000) -> None:
        ...
def InfosetName(node, idx):
    ...
def NodeName(node):
    ...
