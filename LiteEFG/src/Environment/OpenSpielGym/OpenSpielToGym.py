import sys
sys.path.append("../")
import pyspiel
import LiteEFG
from OpenSpiel.OpenSpielToGameFile import OpenSpielEnv
import numpy as np

class TabularStrategy(LiteEFG.Graph):
    def __init__(self):
        super().__init__()
        
        with LiteEFG.backward(is_static=True):
            self.strategy = LiteEFG.const(self.action_set_size, 1.0 / self.action_set_size)

class OpenSpielGymEnv(OpenSpielEnv):
    def __init__(self, game: pyspiel.Game, traverse_type="Enumerate", regenerate=False):
        super().__init__(game, traverse_type, regenerate, is_gym=True)
        self.graph = TabularStrategy()
        self.set_graph(self.graph)
        x = LiteEFG.FileEnv.get_value(self, 1, self.graph.strategy)
        print(x)
        length = 0
        for _ in x:
            length += len(_[1])
        w = np.zeros(length)
        #LiteEFG.FileEnv.set_value(self, 1, self.graph.strategy, w)
        print(LiteEFG.FileEnv.get_value(self, 1, self.graph.strategy))
        print(LiteEFG.FileEnv.exploitability(self, self.graph.strategy))

if __name__ == "__main__":
    env = OpenSpielGymEnv(pyspiel.load_game("kuhn_poker"))
    