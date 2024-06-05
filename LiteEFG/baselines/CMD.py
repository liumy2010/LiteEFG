#######################################################
# Clairvoyant Mirror Descent (CMD)
# Piliouras, Georgios, Ryann Sim, and Stratis Skoulakis. "Beyond time-average convergence: Near-optimal uncoupled online learning via clairvoyant multiplicative weights update." 
#      Advances in Neural Information Processing Systems 35 (2022): 22258-22269.
#######################################################

import LiteEFG
from typing import Literal

class graph(LiteEFG.Graph):
    def __init__(self, eta=0.1, inner_epoch=10, regularizer: Literal["Euclidean", "Entropy"]="Entropy", weighted=False):
        super().__init__()
        self.eta = eta
        self.regularizer = regularizer
        self.timestep = 0
        self.inner_epoch = inner_epoch

        # Create a new graph for CFR
        with LiteEFG.backward(is_static=True):
        
            self.alpha = 1.0
            if weighted:
                self.alpha = LiteEFG.const(1, 1.0)
                self.alpha.inplace(LiteEFG.aggregate(self.alpha, "sum"))
                self.alpha.inplace((self.alpha.max() + 1) * 2)

            ev = LiteEFG.const(size=1, val=0.0)

            self.coef = self.alpha / eta
            self.u = LiteEFG.const(self.action_set_size, 1.0 / self.action_set_size)
            self.bar_u = self.u.copy()

        with LiteEFG.backward():

            gradient = LiteEFG.aggregate(ev, "sum") + self.utility

            self._update(gradient, self.u, self.bar_u)
            self.get_ev(gradient, ev, self.u, self.bar_u)
        
        with LiteEFG.backward(color=1):
            
            self.bar_u.inplace(self.u.copy())

        print("===============Graph is ready for CMD===============")
        print("eta: %f, inner epoch: %d, regularizer: %s" % (self.eta, self.inner_epoch, self.regularizer))
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
        self.timestep += 1
        if self.timestep % self.inner_epoch == 0:
            env.update(self.u, upd_color=[0, 1])
        else:
            env.update(self.u, upd_color=[0])

    
    def current_strategy(self) -> LiteEFG.GraphNode:
        return self.bar_u
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="leduc_poker")
    parser.add_argument("--traverse_type", type=str, choices=["Enumerate", "External", "Outcome"], default="Enumerate")
    parser.add_argument("--iter", type=int, default=100000)
    parser.add_argument("--print_freq", type=int, default=1000)

    parser.add_argument("--eta", help="learning rate", type=float, default=0.1)
    parser.add_argument("--inner_epoch", type=int, default=10)
    parser.add_argument("--regularizer", type=str, choices=["Euclidean", "Entropy"], default="Euclidean")
    parser.add_argument("--weighted", help="weighted dilated regularizer or not", action="store_true")

    args = parser.parse_args()

    from utils import train
    train(graph(args.eta, args.inner_epoch, args.regularizer, args.weighted), args.traverse_type, "last-iterate", args.iter, args.print_freq, args.game)
