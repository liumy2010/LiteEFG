#######################################################
# Outcome Sampling Monte-Carlo Counterfactual Regret Minimization (OS-MCCFR)
# Lanctot, Marc, et al. "Monte Carlo sampling for regret minimization in extensive games." 
#       Advances in neural information processing systems 22 (2009).
#######################################################

import LiteEFG as leg
import argparse
from tqdm import tqdm

class OS_MCCFR(leg.Graph):
    def __init__(self, delta, rm_plus, balanced=True):
        super().__init__()
        self.forward(is_static=True)

        ev = leg.const(size=1, val=0.0)
        self.strategy = leg.const(self.action_set_size, 1.0 / self.action_set_size)
        self.sequence_form_strategy = self.strategy.copy()
        self.prev_strategy = self.strategy.copy()
        
        self.strategy_reach = leg.aggregate(self.sequence_form_strategy, aggregator="sum", object="parent", padding=1.0)
        # return pad when the object not found. Here it is return 1.0 when no parent exists (root infoset)
        self.sequence_form_strategy.inplace(self.strategy_reach * self.strategy)

        if balanced:
            self.uniform = leg.normalize(self.subtree_size, p_norm=1.0, ignore_negative=True)
        else:
            self.uniform = self.strategy.copy()
        self.sequence_form_uniform = self.uniform.copy()
        self.explore_strategy = self.uniform.copy()
    
        self.uniform_reach = leg.aggregate(self.sequence_form_uniform, aggregator="sum", object="parent", padding=1.0)
        self.sequence_form_uniform.inplace(self.uniform_reach * self.uniform)

        self.regret_buffer = leg.const(self.action_set_size, 0.0)

        self.backward()

        gradient = leg.aggregate(ev, aggregator="sum")
        gradient.inplace(gradient + self.utility / (self.reach_prob * self.explore_strategy))
        ev.inplace(leg.dot(gradient, self.strategy))

        self.regret_buffer.inplace(self.regret_buffer + gradient - ev)
        self.prev_strategy.inplace(self.strategy.copy())
        self.strategy.inplace(leg.normalize(self.regret_buffer, p_norm=1.0, ignore_negative=True))
        if rm_plus:
            self.regret_buffer.inplace(leg.maximum(self.regret_buffer, 0.0))

        self.forward()

        self.strategy_reach.inplace(leg.aggregate(self.sequence_form_strategy, aggregator="sum", object="parent", padding=1.0))
        self.sequence_form_strategy.inplace(self.strategy_reach * self.strategy)
        self.explore_strategy.inplace(leg.normalize(delta * self.sequence_form_uniform + (1 - delta) * self.sequence_form_strategy, p_norm=1.0))
    
    def UpdateGraph(self, env):
        env.Update([self.explore_strategy, self.strategy], upd_player=1)
        env.Update([self.prev_strategy, self.explore_strategy], upd_player=2)
    
    def Strategy(self):
        return self.strategy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rm+", dest="rm_plus", help="regret matching+ or not", action="store_true")
    parser.add_argument("--balanced", help="balance exploration or not", action="store_true")
    parser.add_argument("--delta", help="exploration rate", type=float, default=0.1)
    parser.add_argument("--iter", type=int, default=100000)
    parser.add_argument("--print_freq", type=int, default=1000)

    args = parser.parse_args()

    env = leg.FileEnv("GameInstances/liars_dice.game", traverse_type="Outcome")
    alg = OS_MCCFR(args.delta, args.rm_plus, args.balanced)
    env.GetGraph(alg)

    for i in tqdm(range(args.iter)):
        alg.UpdateGraph(env)
        env.UpdateStrategy(alg.strategy)
        if i % args.print_freq == 0:
            print(i, env.Exploitability(alg.Strategy(), "avg-iterate"))