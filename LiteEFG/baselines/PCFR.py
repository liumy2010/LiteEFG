#######################################################
# Predictive Counterfactual Regret Minimization+ (PCFR+)
# Farina, Gabriele, Christian Kroer, and Tuomas Sandholm.
# "Faster game solving via predictive blackwell approachability: Connecting regret matching and mirror descent." 
# Proceedings of the AAAI Conference on Artificial Intelligence (2021).
#######################################################

import LiteEFG as leg
from LiteEFG.baselines.baseline import _baseline

class graph(_baseline):
    def __init__(self):
        super().__init__()
        with leg.backward(is_static=True):
            
            ev = 1.0 * leg.const(1, 0.0)
            self.strategy = leg.const(self.action_set_size, 1.0 / self.action_set_size)
            self.regret_buffer = leg.const(self.action_set_size, 0.0)

        # RM+
        with leg.backward():

            gradient = leg.aggregate(ev, aggregator="sum") + self.utility
            ev.inplace(leg.dot(gradient, self.strategy))
            self.regret_buffer.inplace(leg.maximum(self.regret_buffer + gradient - ev, 0.0))
            self.strategy.inplace(leg.normalize(self.regret_buffer + gradient - ev, p_norm=1.0, ignore_negative=True))
        
        print("===============Graph is ready for PCFR+===============")
        print()
        print("======================================================\n")

    def update_graph(self, env : leg.Environment) -> None:
        env.update(self.strategy, upd_player=1)
        env.update(self.strategy, upd_player=2)

    def current_strategy(self) -> leg.GraphNode:
        return self.strategy

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="leduc_poker")
    parser.add_argument("--traverse_type", type=str, choices=["Enumerate", "External"], default="Enumerate")
    parser.add_argument("--iter", type=int, default=100000)
    parser.add_argument("--print_freq", type=int, default=1000)

    args = parser.parse_args()

    from utils import train
    train(graph(), args.traverse_type, "linear-avg-iterate", args.iter, args.print_freq, args.game)
