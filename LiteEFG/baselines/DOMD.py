#######################################################
# Dilated Optimistic Mirror Descent (DOMD)
# Lee, Chung-Wei, Christian Kroer, and Haipeng Luo. "Last-iterate convergence in extensive-form games." 
#      Advances in Neural Information Processing Systems 34 (2021): 14293-14305.
#######################################################

import LiteEFG
from typing import Literal

class graph(LiteEFG.Graph):
    def __init__(self, eta=0.1, regularizer: Literal["Euclidean", "Entropy"]="Entropy", weighted=False):
        super().__init__()
        self.eta = eta
        self.regularizer = regularizer

        # Create a new graph for CFR
        with LiteEFG.backward(is_static=True):
        
            self.alpha = 1.0
            if weighted:
                self.alpha = LiteEFG.const(1, 1.0)
                self.alpha.inplace(LiteEFG.aggregate(self.alpha, "sum"))
                self.alpha.inplace((self.alpha.max() + 1) * 2)

            ev = LiteEFG.const(size=1, val=0.0)
            bar_ev = LiteEFG.const(size=1, val=0.0)

            self.coef = self.alpha / eta
            self.u = LiteEFG.const(self.action_set_size, 1.0 / self.action_set_size)
            self.bar_u = self.u.copy()

        with LiteEFG.backward():

            bar_gradient = LiteEFG.aggregate(bar_ev, "sum") + self.utility
            gradient = LiteEFG.aggregate(ev, "sum") + self.utility

            self.prev_bar_u =  self.bar_u.copy()
            self._update(bar_gradient, self.bar_u, self.bar_u)
            self._update(gradient, self.u, self.bar_u)

            self.get_ev(bar_gradient, bar_ev, self.bar_u, self.prev_bar_u)
            self.get_ev(gradient, ev, self.u, self.bar_u)

        print("===============Graph is ready for DOMD===============")
        print("eta: %f, regularizer: %s" % (self.eta, self.regularizer))
        print("=====================================================\n")
    
    def get_ev(self, gradient, ev, strategy, ref_strategy):
        if self.regularizer == "Euclidean":
            ev.inplace(LiteEFG.dot(gradient, strategy) - LiteEFG.euclidean(strategy - ref_strategy) * self.coef)
        else:
            kl = LiteEFG.dot((strategy / ref_strategy).log(), strategy) * self.coef
            ev.inplace(LiteEFG.dot(gradient, strategy) - kl)

    def _update(self, gradient, upd_u, ref_u):
        gradient_div = gradient / self.coef
        
        if self.regularizer == "Euclidean":
            upd_u.inplace(ref_u + gradient_div)
            upd_u.inplace(upd_u.project(distance="L2"))

        else:
            upd_u.inplace(ref_u.log() + gradient_div)
            upd_u.inplace(upd_u - upd_u.max())
            upd_u.inplace(upd_u.exp())
            upd_u.inplace(upd_u.project(distance="KL"))
    
    def update_graph(self, env : LiteEFG.Environment) -> None:
        env.update(self.u)
    
    def current_strategy(self) -> LiteEFG.GraphNode:
        return self.u
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="leduc_poker")
    parser.add_argument("--traverse_type", type=str, choices=["Enumerate", "External", "Outcome"], default="Enumerate")
    parser.add_argument("--iter", type=int, default=100000)
    parser.add_argument("--print_freq", type=int, default=1000)

    parser.add_argument("--eta", help="learning rate", type=float, default=0.1)
    parser.add_argument("--regularizer", type=str, choices=["Euclidean", "Entropy"], default="Euclidean")
    parser.add_argument("--weighted", help="weighted dilated regularizer or not", action="store_true")

    args = parser.parse_args()

    from utils import train
    train(graph(args.eta, args.regularizer, args.weighted), args.traverse_type, "last-iterate", args.iter, args.print_freq, args.game)
