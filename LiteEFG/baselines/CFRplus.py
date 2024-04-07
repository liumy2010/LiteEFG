#######################################################
# Counterfactual Regret Minimization+ (CFR+)
# Zinkevich, Martin, et al. "Regret minimization in games with incomplete information." 
#       Advances in neural information processing systems 20 (2007).
#######################################################

import LiteEFG as leg
import argparse
from tqdm import tqdm

class CFR_Plus(leg.Graph):
    def __init__(self):
        super().__init__()
        self.backward(is_static=True)
        ev = 1.0 * leg.const(1, 0.0)
        self.strategy = leg.const(self.action_set_size, 1.0 / self.action_set_size)
        self.regret_buffer = leg.const(self.action_set_size, 0.0)

        # RM+
        self.backward()

        gradient = leg.aggregate(ev, aggregator="sum")
        gradient.inplace(gradient + self.utility)
        ev.inplace(leg.dot(gradient, self.strategy))
        self.regret_buffer.inplace(leg.maximum(self.regret_buffer + gradient - ev, 0.0))
        self.strategy.inplace(leg.normalize(self.regret_buffer, p_norm=1.0, ignore_negative=True))
    
    def UpdateGraph(self, env):
        env.Update(self.strategy, upd_player=1)
        env.Update(self.strategy, upd_player=2)
    
    def Strategy(self):
        return self.strategy
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traverse_type", type=str, choices=["Enumerate", "External"], default="Enumerate")
    parser.add_argument("--iter", type=int, default=100000)
    parser.add_argument("--print_freq", type=int, default=1000)

    args = parser.parse_args()

    env = leg.FileEnv("GameInstances/leduc.game", traverse_type=args.traverse_type)
    alg = CFR_Plus()
    env.GetGraph(alg)
    for i in tqdm(range(args.iter)):
        alg.UpdateGraph(env)
        env.UpdateStrategy(alg.Strategy())
        if i % args.print_freq == 0:
            print(i, env.Exploitability(alg.Strategy(), "linear-avg-iterate"))
