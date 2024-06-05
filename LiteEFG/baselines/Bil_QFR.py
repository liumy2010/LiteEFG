import LiteEFG
from typing import Literal

class graph(LiteEFG.Graph):
    def __init__(self, eta=0.001, tau=0.001, gamma=0.001, regularizer: Literal["Euclidean", "Entropy"]="Entropy", feedback="Q", weighted=False, bidilated=True):
        super().__init__()

        self.eta = eta
        self.tau = tau
        self.gamma = gamma
        self.regularizer = regularizer
        self.feedback = feedback

        with LiteEFG.backward(is_static=True):

            self.alpha = 1.0
            if weighted:
                self.alpha = LiteEFG.const(1, 1.0)
                self.alpha.inplace(LiteEFG.aggregate(self.alpha, "sum"))
                self.alpha.inplace((self.alpha.max() + 1) * 2)

            self.ev = LiteEFG.const(1, 0.0)
            self.coef = self.alpha * self.tau
            self.mu = self.subtree_size.normalize(p_norm=1.0, ignore_negative=True)
            self.eta_coef = self.eta * self.coef
            
            self.u = LiteEFG.const(self.action_set_size, 1.0 / self.action_set_size)
            self.bar_u = self.u.copy()

        with LiteEFG.backward():

            if(self.feedback in ["Q", "traj-Q", "counterfactual"]):
                self._full_information()
            elif self.feedback == "Outcome":
                self._outcome_sampling()

        print("===============Graph is ready for Bidilated QFR===============")
        print("eta: %f, tau: %f, gamma: %f, regularizer: %s, feedback: %s" % (self.eta, self.tau, self.gamma, self.regularizer, self.feedback))
        print("====================================================\n")
    
    def _full_information(self):
        if self.feedback == "counterfactual":
            self.m_th = 1.0
        elif self.feedback == "traj-Q":
            self.m_th = 1.0 / self.reach_prob
        else:
            self.m_th = self.opponent_reach_prob
        
        self.eta_tau = self.eta_coef / self.m_th * self.opponent_reach_prob + 1 # 1 + eta * tau * alpha / m_th * mu_{-p}(s)
        
        if self.regularizer == "Euclidean":
            reg = LiteEFG.euclidean(self.u) * self.coef
        else:
            reg = LiteEFG.negative_entropy(self.u, shifted=True) * self.coef

        bidilated_reg = reg * self.reach_prob
        gradient = LiteEFG.aggregate(self.ev, "sum") + self.utility \
                                                    + LiteEFG.aggregate(bidilated_reg, "sum", player="opponents")
        
        self.ev.inplace(LiteEFG.dot(gradient, self.u) - reg * self.opponent_reach_prob)

        gradient.inplace(gradient / self.m_th * self.eta)

        self._update(self.bar_u, self.bar_u, gradient)
        self._update(self.u, self.bar_u, gradient)
    
    def _outcome_sampling(self):
        #self.m_th = 1.0 / self.reach_prob
        self.eta_tau = self.eta_coef + 1 # 1 + eta * tau * alpha

        if self.regularizer == "Euclidean":
            reg = LiteEFG.euclidean(self.u) * self.coef
        else:
            reg = LiteEFG.negative_entropy(self.u, shifted=True) * self.coef

        gradient = LiteEFG.aggregate(self.ev, "sum") + self.utility \
                                                    + LiteEFG.aggregate(reg, "sum", player="opponents")
        self.ev.inplace(LiteEFG.sum(gradient) - reg)

        gradient.inplace(gradient / self.u * self.eta)

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
    
    def update_graph(self, env : LiteEFG.Environment) -> None:
        env.update(self.u)
    
    def current_strategy(self) -> LiteEFG.GraphNode:
        return self.u

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="leduc_poker(suit_isomorphism=True)")
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
    # update_strategy() will enumerate all infosets. However, when using outcome-sampling, at each iteration,
    # the algorithm will only update a trajectory of length height
    # Solution: lazy-update, TBD