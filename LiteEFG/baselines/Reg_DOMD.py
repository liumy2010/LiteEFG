#######################################################
# Regularized Dilated Optimistic Mirror Descent (Reg-DOMD)
# Mingyang Liu, Asuman Ozdaglar, Tiancheng Yu, and Kaiqing Zhang
#   "The power of regularization in solving extensive-form games."
#   International Conference on Learning Representations (2023)
#######################################################

import LiteEFG
from typing import Literal

class graph(LiteEFG.Graph):
    def __init__(self, eta=0.1, tau=0.1, regularizer: Literal["Euclidean", "Entropy"]="Entropy", 
                        weighted=False, shrink_iter=100000, out_reg=False):
        super().__init__()
        self.eta = eta
        self.regularizer = regularizer
        self.timestep = 0
        self.shrink_iter = shrink_iter
        self.out_reg = out_reg

        # Create a new graph for CFR
        with LiteEFG.backward(is_static=True):
        
            self.alpha = 1.0
            if weighted:
                self.alpha = LiteEFG.const(1, 1.0)
                self.alpha.inplace(LiteEFG.aggregate(self.alpha, "sum"))
                self.alpha.inplace((self.alpha.max() + 1) * 2)

            ev = LiteEFG.const(size=1, val=0.0)
            bar_ev = LiteEFG.const(size=1, val=0.0)

            self.tau = LiteEFG.const(1, tau)
            self.eta_coef = self.alpha / eta
            self.coef = self.alpha * self.tau
            self.u = LiteEFG.const(self.action_set_size, 1.0 / self.action_set_size)
            self.bar_u = self.u.copy()

        with LiteEFG.backward(color=0):

            bar_gradient = LiteEFG.aggregate(bar_ev, "sum") + self.utility
            gradient = LiteEFG.aggregate(ev, "sum") + self.utility

            self.prev_bar_u =  self.bar_u.copy()
            self._update(bar_gradient, self.bar_u, self.bar_u)
            self._update(gradient, self.u, self.bar_u)

            self.get_ev(bar_gradient, bar_ev, self.bar_u, self.prev_bar_u)
            self.get_ev(gradient, ev, self.u, self.bar_u)
        
        with LiteEFG.backward(color=1):
            self.tau.inplace(self.tau * 0.5)
            self.coef.inplace(self.alpha * self.tau)

        print("===============Graph is ready for Reg-DOMD===============")
        print("eta: %f, tau: %f, regularizer: %s" % (self.eta, tau, self.regularizer))
        print("=====================================================\n")
    
    def get_ev(self, gradient, ev, strategy, ref_strategy):
        if not self.out_reg:
            if self.regularizer == "Euclidean":
                ev.inplace(LiteEFG.dot(gradient, strategy) - LiteEFG.euclidean(strategy - ref_strategy) * self.eta_coef
                                                            + LiteEFG.euclidean(ref_strategy) * self.coef)
            else:
                kl = LiteEFG.dot((strategy / ref_strategy).log(), strategy) * self.eta_coef
                ev.inplace(LiteEFG.dot(gradient, strategy) - kl + self.coef)
        else:
            if self.regularizer == "Euclidean":
                ev.inplace(LiteEFG.dot(gradient, strategy) - LiteEFG.euclidean(strategy - ref_strategy) * self.eta_coef
                                                            - LiteEFG.euclidean(strategy) * self.coef)
            else:
                kl = LiteEFG.dot((strategy / ref_strategy).log(), strategy) * self.eta_coef
                ev.inplace(LiteEFG.dot(gradient, strategy) - kl - LiteEFG.negative_entropy(strategy) * self.coef)

    def _update(self, gradient, upd_u, ref_u):
        if not self.out_reg:
            if self.regularizer == "Euclidean":
                gradient.inplace(gradient - ref_u * self.coef)
                gradient_div = gradient / self.eta_coef
                upd_u.inplace(ref_u + gradient_div)
                upd_u.inplace(upd_u.project(distance="L2"))

            else:
                gradient.inplace(gradient - (ref_u.log() + 1) * self.coef)
                gradient_div = gradient / self.eta_coef
                upd_u.inplace(ref_u.log() + gradient_div)
                upd_u.inplace(upd_u - upd_u.max())
                upd_u.inplace(upd_u.exp())
                upd_u.inplace(upd_u.project(distance="KL"))
        else:
            if self.regularizer == "Euclidean":
                gradient_div = gradient / self.eta_coef
                upd_u.inplace((ref_u + gradient_div) / (1.0 + self.coef * self.eta))
                upd_u.inplace(upd_u.project(distance="L2"))

            else:
                gradient_div = gradient / self.eta_coef
                upd_u.inplace((ref_u.log() + gradient_div) / (1.0 + self.coef * self.eta))
                upd_u.inplace(upd_u - upd_u.max())
                upd_u.inplace(upd_u.exp())
                upd_u.inplace(upd_u.project(distance="KL"))
    
    def update_graph(self, env : LiteEFG.Environment) -> None:
        self.timestep += 1
        env.update(self.u, upd_color=[0, 1]) if self.timestep % self.shrink_iter == 0 else env.update(self.u, upd_color=[0])
    
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
    parser.add_argument("--tau", help="regularization coefficient", type=float, default=0.1)
    parser.add_argument("--shrink-iter", help="shrink tau by half every shrink-iter iterations", type=int, default=100000)
    parser.add_argument("--regularizer", type=str, choices=["Euclidean", "Entropy"], default="Euclidean")
    parser.add_argument("--weighted", help="weighted dilated regularizer or not", action="store_true")
    parser.add_argument("--out-reg", help="whether the update-rule is \\argmin <g, x+\\nabla\\psi(x_0)+D(x, x_0)> (default) or \\argmin <g, x>+\\psi(x)+D(x, x_0)", action="store_true")

    args = parser.parse_args()

    from utils import train
    train(graph(args.eta, args.tau, args.regularizer, args.weighted, args.shrink_iter, args.out_reg), args.traverse_type, "last-iterate", args.iter, args.print_freq, args.game)
