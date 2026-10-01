#######################################################
# Q-Function based Regret Minimization (QFR)
# Mingyang Liu, Gabriele Farina, and Asuman Ozdaglar.
# "A Policy-Gradient Approach to Solving Imperfect-Information Games with Iterate Convergence."
# International Conference on Learning Representations (2025)
#######################################################

import LiteEFG as leg
from LiteEFG.baselines.baseline import _baseline
from typing import Literal

class graph(_baseline):
    def __init__(self, eta=0.1, tau=0.001, gamma=0.001, regularizer: Literal["Euclidean", "Entropy"]="Entropy", feedback="Q", weighted=False):
        super().__init__()

        self.eta = eta
        self.tau = tau
        self.gamma = gamma
        self.regularizer = regularizer
        self.feedback = feedback

        with leg.backward(is_static=True):

            self.alpha = 1.0
            if weighted:
                self.alpha = leg.const(1, 1.0)
                self.alpha.inplace(leg.aggregate(self.alpha, "sum"))
                self.alpha.inplace((self.alpha.max() + 1) * 2)

            self.ev = leg.const(1, 0.0)
            self.coef = self.tau
            self.mu = self.subtree_size.normalize(p_norm=1.0, ignore_negative=True)
            self.eta_coef = self.eta * self.coef
            
            self.u = leg.const(self.action_set_size, 1.0 / self.action_set_size)
            self.bar_u = self.u.copy()

        with leg.backward():

            if(self.feedback in ["Q", "traj-Q", "counterfactual"]):
                self._full_information()
            elif self.feedback == "Outcome":
                self._outcome_sampling()

        print("===============Graph is ready for QFR===============")
        print("eta: %f, tau: %f, gamma: %f, regularizer: %s, feedback: %s" % (self.eta, self.tau, self.gamma, self.regularizer, self.feedback))
        print("====================================================\n")
    
    def _full_information(self):
        if self.feedback == "counterfactual":
            self.m_th = 1.0
        elif self.feedback == "traj-Q":
            self.m_th = 1.0 / self.reach_prob
        else:
            self.m_th = self.opponent_reach_prob
        
        self.eta_tau = self.eta_coef / self.m_th * self.opponent_reach_prob + 1 # 1 + eta * tau / m_th * mu_{-p}(s)
        
        if self.regularizer == "Euclidean":
            reg = leg.euclidean(self.u) * self.coef
        else:
            reg = leg.negative_entropy(self.u, shifted=True) * self.coef

        bidilated_reg = reg * self.reach_prob
        gradient = leg.aggregate(self.ev, "sum") + self.utility \
                                                    + leg.aggregate(bidilated_reg, "sum", player="opponents")
        
        self.ev.inplace(leg.dot(gradient, self.u) - reg * self.opponent_reach_prob)

        gradient.inplace(gradient / self.m_th * self.eta / self.alpha)

        self._update(self.bar_u, self.bar_u, gradient)
        self._update(self.u, self.bar_u, gradient)
    
    def _outcome_sampling(self):
        #self.m_th = 1.0 / self.reach_prob
        self.eta_tau = self.eta_coef + 1 # 1 + eta * tau

        if self.regularizer == "Euclidean":
            reg = leg.euclidean(self.u) * self.coef
        else:
            reg = leg.negative_entropy(self.u, shifted=True) * self.coef

        gradient = leg.aggregate(self.ev, "sum") + self.utility \
                                                    + leg.aggregate(reg, "sum", player="opponents")
        self.ev.inplace(leg.sum(gradient) - reg)

        gradient.inplace(gradient / self.u * self.eta / self.alpha)

        self._update(self.bar_u, self.bar_u, gradient)
        self._update(self.u, self.bar_u, gradient)

    def _update(self, upd_u, ref_u, gradient):
        if self.regularizer == "Euclidean":
            upd_u.inplace((ref_u + gradient) / self.eta_tau)
            upd_u.inplace(upd_u.project(distance="L2", gamma=self.gamma, mu=self.mu))
        else:
            upd_u.inplace((ref_u.log() + gradient) / self.eta_tau)
            upd_u.inplace(upd_u - upd_u.max())
            upd_u.inplace(upd_u.exp())
            upd_u.inplace(upd_u.project(distance="KL", gamma=self.gamma, mu=self.mu))

    def update_graph(self, env : leg.Environment) -> None:
        env.update(self.u)

    def current_strategy(self) -> leg.GraphNode:
        return self.u

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="leduc_poker")
    parser.add_argument("--iter", type=int, default=100000)
    parser.add_argument("--print_freq", type=int, default=1000)

    parser.add_argument("--eta", help="learning rate", type=float, default=0.1)
    parser.add_argument("--tau", help="regularization coefficient", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.001)
    parser.add_argument("--regularizer", type=str, choices=["Euclidean", "Entropy"], default="Entropy")
    parser.add_argument("--feedback", type=str, choices=["Q", "traj-Q", "counterfactual", "Outcome"], default="Q")
    parser.add_argument("--weighted", help="weighted dilated regularizer or not", action="store_true")

    args = parser.parse_args()

    traverse_type = "Enumerate" if(args.feedback != "Outcome") else "Outcome"
    
    from utils import train
    train(graph(args.eta, args.tau, args.gamma, args.regularizer, args.feedback, args.weighted), traverse_type, "default", args.iter, args.print_freq, args.game)
    # do not need to update strategy for algorithms only need last-iterate
    # it will be faster since LiteEFG do not need to maintain the sequence-form strategy
    # update_strategy() will enumerate all infosets. When using outcome-sampling, at each iteration,
    # the algorithm will only update a trajectory of length height
