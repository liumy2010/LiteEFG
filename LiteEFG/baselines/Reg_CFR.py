#######################################################
# Regularized Counterfactual Regret Minimization (Reg-CFR)
# Mingyang Liu, Asuman Ozdaglar, Tiancheng Yu, and Kaiqing Zhang
#   "The power of regularization in solving extensive-form games."
#   International Conference on Learning Representations (2023)
#######################################################

import LiteEFG
from typing import Literal
import math

class graph(LiteEFG.Graph):
    def __init__(self, kappa=1.0, tau=0.1, regularizer: Literal["Euclidean", "Entropy"]="Entropy", 
                        weighted=False, shrink_iter=100000, out_reg=False):
        super().__init__()
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

            self.sqr_lambda = LiteEFG.const(1, kappa)
            self.tau = LiteEFG.const(1, tau)
            self.coef = self.alpha * self.tau
            self.u = LiteEFG.const(self.action_set_size, 1.0 / self.action_set_size)
            self.bar_u = self.u.copy()
            prev_gradient = LiteEFG.const(size=self.action_set_size, val=0.0)
            self.prev_eta = LiteEFG.const(1, math.sqrt(1.0 / kappa))
            self.eta = LiteEFG.const(1, math.sqrt(1.0 / kappa))
            self.init_strategy = self.u.copy()

        with LiteEFG.backward(color=0):
            gradient = LiteEFG.aggregate(ev, "sum") + self.utility

        with LiteEFG.backward(color=2):
            self.prev_eta.inplace(self.eta.copy())
            self.sqr_lambda.inplace(self.sqr_lambda + 2.0 * LiteEFG.euclidean(gradient - prev_gradient))
            self.eta.inplace(1.0 / (self.sqr_lambda ** 0.5))

        with LiteEFG.backward(color=0):

            prev_gradient.inplace(gradient.copy())
            self.eta_coef = self.alpha / self.eta

            self.prev_bar_u =  self.bar_u.copy()
            self._update(gradient, self.bar_u, self.bar_u, True)
            self._update(gradient, self.u, self.bar_u, False)

            self.get_ev(gradient, ev, self.u, self.bar_u)
        
        with LiteEFG.backward(color=1):
            self.tau.inplace(self.tau * 0.5)
            self.coef.inplace(self.alpha * self.tau)

        print("===============Graph is ready for Reg-DOMD===============")
        print("kappa: %f, tau: %f, regularizer: %s" % (kappa, tau, self.regularizer))
        print("=====================================================\n")
    
    def get_ev(self, gradient, ev, strategy, ref_strategy):
        if self.regularizer == "Euclidean":
            ev.inplace(LiteEFG.dot(gradient, strategy) - LiteEFG.euclidean(strategy) * self.coef)
        else:
            ev.inplace(LiteEFG.dot(gradient, strategy) - LiteEFG.negative_entropy(strategy) * self.coef)

    def _update(self, gradient, upd_u, ref_u, is_stabilize=False):
        mix_coef = 1.0 if not is_stabilize else self.eta / self.prev_eta
        if not self.out_reg:
            if self.regularizer == "Euclidean":
                gradient_div = (gradient - ref_u * self.coef) / self.eta_coef
                upd_u.inplace((ref_u * mix_coef + self.init_strategy * (1.0 - mix_coef)) + gradient_div)
                upd_u.inplace(upd_u.project(distance="L2"))

            else:
                gradient_div = (gradient - (ref_u.log() + 1) * self.coef) / self.eta_coef
                upd_u.inplace((ref_u.log() * mix_coef + self.init_strategy.log() * (1.0 - mix_coef)) + gradient_div)
                upd_u.inplace((upd_u - upd_u.max()).exp().project(distance="KL"))
        else:
            if self.regularizer == "Euclidean":
                gradient_div = gradient / self.eta_coef
                upd_u.inplace(((ref_u * mix_coef + self.init_strategy * (1.0 - mix_coef)) + gradient_div) / (1.0 + self.coef * self.eta))
                upd_u.inplace(upd_u.project(distance="L2"))

            else:
                gradient_div = gradient / self.eta_coef
                upd_u.inplace(((ref_u.log() * mix_coef + self.init_strategy.log() * (1.0 - mix_coef)) + gradient_div) / (1.0 + self.coef * self.eta))
                upd_u.inplace((upd_u - upd_u.max()).exp().project(distance="KL"))
    
    def update_graph(self, env : LiteEFG.Environment) -> None:
        self.timestep += 1
        if self.timestep == 1:
            env.update(self.u, upd_color=[0])
            return
        env.update(self.u, upd_color=[0, 1, 2]) if self.timestep % self.shrink_iter == 0 else env.update(self.u, upd_color=[0, 2])
        
        '''
        lambda_dict = env.get_value(0, self.sqr_lambda)
        lambda_sum = 0
        for k in lambda_dict:
            lambda_sum += sum(k[1])
        print(lambda_sum / len(lambda_dict))
        '''
    
    def current_strategy(self) -> LiteEFG.GraphNode:
        return self.u
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="leduc_poker")
    parser.add_argument("--traverse_type", type=str, choices=["Enumerate", "External", "Outcome"], default="Enumerate")
    parser.add_argument("--iter", type=int, default=100000)
    parser.add_argument("--print_freq", type=int, default=1000)

    parser.add_argument("--kappa", help="initial reciprocal of learning rate", type=float, default=1.0)
    parser.add_argument("--tau", help="regularization coefficient", type=float, default=0.1)
    parser.add_argument("--shrink-iter", help="shrink tau by half every shrink-iter iterations", type=int, default=100000)
    parser.add_argument("--regularizer", type=str, choices=["Euclidean", "Entropy"], default="Euclidean")
    parser.add_argument("--weighted", help="weighted dilated regularizer or not", action="store_true")
    parser.add_argument("--out-reg", help="whether the update-rule is \\argmin <g, x+\\nabla\\psi(x_0)+D(x, x_0)> (default) or \\argmin <g, x>+\\psi(x)+D(x, x_0)", action="store_true")

    args = parser.parse_args()

    from utils import train
    train(graph(args.kappa, args.tau, args.regularizer, args.weighted, args.shrink_iter, args.out_reg), args.traverse_type, "last-iterate", args.iter, args.print_freq, args.game)
