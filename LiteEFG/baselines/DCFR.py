#######################################################
# Discounted Counterfactual Regret Minimization (DCFR)
# Brown, Noam, and Tuomas Sandholm. "Solving imperfect-information games via discounted regret minimization." 
# AAAI 2019.
#######################################################

import LiteEFG as leg
import argparse
from tqdm import tqdm

class DCFR(leg.Graph):
    def __init__(self, alpha=1.5, beta=0, gamma=2):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.threshold = 10

        self.backward(is_static=True)

        self.timestep = leg.const(size=1, val=0.0)
        ev = leg.const(size=1, val=0.0)
        self.strategy = leg.const(self.action_set_size, 1.0 / self.action_set_size)
        self.avg_seq_strategy = leg.const(self.action_set_size, 0.0)
        self.regret_buffer = leg.const(self.action_set_size, 0.0)
        self.avg_strategy = self.strategy.copy()

        self.backward()

        self.pos_coeff = (self.timestep ** self.alpha) / (self.timestep ** self.alpha + 1) if abs(self.alpha) < self.threshold else (1 if self.alpha>0 else 0)
        self.neg_coeff = (self.timestep ** self.beta) / (self.timestep ** self.beta + 1) if abs(self.beta) < self.threshold else (1 if self.beta>0 else 0)
        self.strategy_coef = (self.timestep / (self.timestep + 1)) ** self.gamma
        self.timestep.inplace(self.timestep + 1)

        gradient = leg.aggregate(ev, aggregator="sum")
        gradient.inplace(gradient + self.utility)
        ev.inplace(leg.dot(gradient, self.strategy))

        self.neg_regret = self.regret_buffer < 0
        self.pos_regret = self.regret_buffer >= 0

        self.regret_buffer.inplace((self.neg_regret * self.regret_buffer) * self.neg_coeff + (self.pos_regret * self.regret_buffer) * self.pos_coeff)

        self.regret_buffer.inplace(self.regret_buffer + gradient - ev)
        self.avg_seq_strategy.inplace(self.avg_seq_strategy * self.strategy_coef + self.strategy * self.reach_prob)
        self.avg_strategy.inplace(self.avg_seq_strategy.normalize(p_norm=1.0, ignore_negative=True))
        self.strategy.inplace(leg.normalize(self.regret_buffer, p_norm=1.0, ignore_negative=True))

    def UpdateGraph(self, env):
        env.Update(self.strategy, upd_player=1)
        env.Update(self.strategy, upd_player=2)
    
    def Strategy(self):
        return self.avg_strategy
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traverse_type", type=str, choices=["Enumerate", "External"], default="Enumerate")
    parser.add_argument("--alpha", type=int, default=1.5)
    parser.add_argument("--beta", type=int, default=0)
    parser.add_argument("--gamma", type=int, default=2)
    parser.add_argument("--iter", type=int, default=100000)
    parser.add_argument("--print_freq", type=int, default=1000)

    args = parser.parse_args()

    env = leg.FileEnv("GameInstances/leduc.game", traverse_type=args.traverse_type)
    alg = DCFR(args.alpha, args.beta, args.gamma)
    env.GetGraph(alg)

    for i in tqdm(range(args.iter)):
        alg.UpdateGraph(env)
        env.UpdateStrategy(alg.Strategy())
        if i % args.print_freq == 0:
            print(i, env.Exploitability(alg.Strategy(), "last-iterate"))