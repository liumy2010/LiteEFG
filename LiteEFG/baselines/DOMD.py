#######################################################
# Dilated Optimistic Mirror Descent (DOMD)
# Lee, Chung-Wei, Christian Kroer, and Haipeng Luo. "Last-iterate convergence in extensive-form games." 
#      Advances in Neural Information Processing Systems 34 (2021): 14293-14305.
#######################################################

import LiteEFG as leg
import argparse
from typing import Literal
from tqdm import tqdm

class DOMD(leg.Graph):
    def __init__(self, eta, regularizer: Literal["Euclidean", "Entropy"], weighted):
        super().__init__()
        self.eta = eta
        self.regularizer = regularizer

        # Create a new graph for CFR
        self.backward(is_static=True)
        
        self.alpha = 1.0
        if weighted:
            self.alpha = leg.const(1, 1.0)
            self.alpha.inplace(leg.aggregate(self.alpha, "sum"))
            self.alpha.inplace((self.alpha.max() + 1) * 2)

        ev = leg.const(size=1, val=0.0)
        bar_ev = leg.const(size=1, val=0.0)

        self.coef = self.alpha / eta
        self.u = leg.const(self.action_set_size, 1.0 / self.action_set_size)
        self.bar_u = self.u.copy()

        self.backward()

        bar_gradient = leg.aggregate(bar_ev, "sum") + self.utility
        gradient = leg.aggregate(ev, "sum") + self.utility

        self.prev_bar_u =  self.bar_u.copy()
        self.Update(bar_gradient, self.bar_u, self.bar_u)
        self.Update(gradient, self.u, self.bar_u)

        self.GetEv(bar_gradient, bar_ev, self.bar_u, self.prev_bar_u)
        self.GetEv(gradient, ev, self.u, self.bar_u)

        print("===============Graph is ready for DOMD===============")
        print("eta: %f, regularizer: %s" % (self.eta, self.regularizer))
        print("=====================================================\n")
    
    def GetEv(self, gradient, ev, strategy, ref_strategy):
        if self.regularizer == "Euclidean":
            ev.inplace(leg.dot(gradient, strategy) - leg.euclidean(strategy - ref_strategy) * self.coef)
        else:
            kl = leg.dot((strategy / ref_strategy).log(), strategy) * self.coef
            ev.inplace(leg.dot(gradient, strategy) - kl)

    def Update(self, gradient, upd_u, ref_u):
        gradient_div = gradient / self.coef
        
        if self.regularizer == "Euclidean":
            upd_u.inplace(ref_u + gradient_div)
            upd_u.inplace(upd_u.project(distance="L2"))

        else:
            upd_u.inplace(ref_u.log() + gradient_div)
            upd_u.inplace(upd_u - upd_u.max())
            upd_u.inplace(upd_u.exp())
            upd_u.inplace(upd_u.project(distance="KL"))
    
    def UpdateGraph(self, env):
        env.Update(self.u)
    
    def Strategy(self):
        return self.u
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traverse_type", type=str, choices=["Enumerate", "External", "Outcome"], default="Enumerate")
    parser.add_argument("--iter", type=int, default=100000)
    parser.add_argument("--print_freq", type=int, default=1000)

    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--regularizer", type=str, choices=["Euclidean", "Entropy"], default="Euclidean")
    parser.add_argument("--weighted", help="weighted dilated regularizer or not", action="store_true")

    args = parser.parse_args()

    env = leg.FileEnv("GameInstances/leduc.game", traverse_type=args.traverse_type)
    alg = DOMD(args.eta, args.regularizer, args.weighted)
    env.GetGraph(alg)
    for i in tqdm(range(args.iter)):
        alg.UpdateGraph(env)
        env.UpdateStrategy(alg.Strategy())
        if i % args.print_freq == 0:
            print(i, env.Exploitability(alg.Strategy(), "last-iterate"))