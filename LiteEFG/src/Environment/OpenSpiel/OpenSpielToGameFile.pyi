from __future__ import annotations
import LiteEFG as leg
import numpy as np
import open_spiel.python.policy
from open_spiel.python.policy import TabularPolicy
import os as os
import pandas as pd
import pyspiel as pyspiel
import typing as typing
__all__ = ['InfosetName', 'leg', 'NodeName', 'OpenSpielEnv', 'TabularPolicy', 'np', 'os', 'pd', 'pyspiel', 'typing']
class OpenSpielEnv(leg._LiteEFG.FileEnv):
    def __init__(self, game: pyspiel.Game, traverse_type = 'Enumerate', regenerate = False):
        ...
    def get_strategy(self, strategy_node: leg._LiteEFG.GraphNode, type_name = 'default') -> typing.Tuple[open_spiel.python.policy.TabularPolicy, typing.List[pd.DataFrame]]:
        ...
    def get_value(self, player: int, node: leg._LiteEFG.GraphNode) -> typing.List[typing.Tuple[str, typing.List[float]]]:
        ...
    def interact(self, policy: open_spiel.python.policy.TabularPolicy, controlled_player = 0, reveal_private = True, epochs = 1000) -> None:
        ...
    def set_value(self, player: int, node: leg._LiteEFG.GraphNode, values: typing.List[typing.List]) -> None:
        ...
def InfosetName(node, idx):
    ...
def NodeName(node):
    ...
