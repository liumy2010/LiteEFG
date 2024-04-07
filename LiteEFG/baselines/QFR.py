import LiteEFG as leg
import argparse
from typing import Literal
from tqdm import tqdm

class QFR(leg.Graph):
    def __init__(self, eta, tau, gamma, regularizer: Literal["Euclidean", "Entropy"], feedback="Q", weighted=False):
        super().__init__()

        self.eta = eta
        self.tau = tau
        self.gamma = gamma
        self.regularizer = regularizer
        self.feedback = feedback

        self.backward(is_static=True)

        self.alpha = 1.0
        if weighted:
            self.alpha = leg.const(1, 1.0)
            self.alpha.inplace(leg.aggregate(self.alpha, "sum"))
            self.alpha.inplace((self.alpha.max() + 1) * 2)


        self.ev = leg.const(1, 0.0)
        self.coef = self.alpha * self.tau
        self.mu = self.subtree_size.normalize(p_norm=1.0, ignore_negative=True)
        self.eta_coef = self.eta * self.coef
        
        self.u = leg.const(self.action_set_size, 1.0 / self.action_set_size)
        self.bar_u = self.u.copy()

        self.backward()

        if(self.feedback in ["Q", "traj-Q", "counterfactual"]):
            self.FullInformation()
        elif self.feedback == "Outcome":
            self.OutcomeSampling()

        print("===============Graph is ready for QFR===============")
        print("eta: %f, tau: %f, gamma: %f, regularizer: %s, feedback: %s" % (self.eta, self.tau, self.gamma, self.regularizer, self.feedback))
        print("====================================================\n")
    
    def FullInformation(self):
        if self.feedback == "counterfactual":
            self.m_th = 1.0
        elif self.feedback == "traj-Q":
            self.m_th = 1.0 / self.reach_prob
        else:
            self.m_th = self.opponent_reach_prob
        
        self.eta_tau = self.eta_coef / self.m_th + 1 # 1 + eta * tau * alpha / m_th
        gradient = leg.aggregate(self.ev, "sum") + self.utility
        self.ev.inplace(leg.dot(gradient, self.u))

        if self.regularizer == "Euclidean":
            reg = leg.euclidean(self.u)
        else:
            reg = leg.negative_entropy(self.u, shifted=True)
        
        self.ev.inplace(self.ev - reg * self.coef)

        gradient.inplace(gradient / self.m_th * self.eta)

        self.Update(self.bar_u, self.bar_u, gradient)
        self.Update(self.u, self.bar_u, gradient)
    
    def OutcomeSampling(self):
        self.m_th = 1.0 / self.reach_prob
        self.eta_tau = self.eta_coef / self.m_th + 1 # 1 + eta * tau * alpha / m_th
        gradient = leg.aggregate(self.ev, "sum") + self.utility
        self.ev.inplace(leg.sum(gradient))

        if self.regularizer == "Euclidean":
            reg = leg.euclidean(self.u)
        else:
            reg = leg.negative_entropy(self.u, shifted=True)
        
        self.ev.inplace(self.ev - reg * self.coef / self.opponent_reach_prob)

        gradient.inplace(gradient / self.u * self.eta)

        self.Update(self.bar_u, self.bar_u, gradient)
        self.Update(self.u, self.bar_u, gradient)

    def Update(self, upd_u, ref_u, gradient):
        if self.regularizer == "Euclidean":
            upd_u.inplace((ref_u + gradient) / self.eta_tau)
            upd_u.inplace(upd_u.project(distance="L2", gamma=self.gamma, mu=self.mu))
        else:
            upd_u.inplace((ref_u.log() + gradient) / self.eta_tau)
            upd_u.inplace(upd_u - upd_u.max())
            upd_u.inplace(upd_u.exp())
            upd_u.inplace(upd_u.project(distance="KL", gamma=self.gamma, mu=self.mu))
    
    def UpdateGraph(self, env):
        env.Update(self.u)
    
    def Strategy(self):
        return self.u

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iter", type=int, default=100000)
    parser.add_argument("--print_freq", type=int, default=1000)

    parser.add_argument("--eta", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--regularizer", type=str, choices=["Euclidean", "Entropy"], default="Euclidean")
    parser.add_argument("--feedback", type=str, choices=["Q", "traj-Q", "counterfactual", "Outcome"], default="Q")
    parser.add_argument("--weighted", help="weighted dilated regularizer or not", action="store_true")

    args = parser.parse_args()

    traverse_type = "Enumerate" if(args.feedback != "Outcome") else "Outcome"

    env = leg.FileEnv("GameInstances/liars_dice.game", traverse_type=traverse_type)
    alg = QFR(args.eta, args.tau, args.gamma, args.regularizer, args.feedback, args.weighted)
    env.GetGraph(alg)

    for i in tqdm(range(args.iter)):
        alg.UpdateGraph(env)
        if i % args.print_freq == 0:
            print(i, env.Exploitability(alg.Strategy()))
            # do not need to update strategy for algorithms only need last-iterate
            # it will be faster since LiteEFG do not need to maintain the sequence-form strategy
            # UpdateStrategy() will enumerate all infosets. However, when using outcome-sampling, at each iteration,
            # the algorithm will only update a trajectory of length height
            # Solution: lazy-update, TBD