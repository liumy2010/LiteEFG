#######################################################
# Outcome Sampling Monte-Carlo Counterfactual Regret Minimization (OS-MCCFR)
# Lanctot, Marc, et al. "Monte Carlo sampling for regret minimization in extensive games." 
#       Advances in neural information processing systems 22 (2009).
#######################################################

import LiteEFG

class graph(LiteEFG.Graph):
    def __init__(self, delta=0.1, rm_plus=False, balanced=True):
        super().__init__()

        self.delta = delta
        self.rm_plus = rm_plus

        with LiteEFG.forward(is_static=True):

            expectation = LiteEFG.const(size=1, val=0.0)
            self.strategy = LiteEFG.const(self.action_set_size, 1.0 / self.action_set_size)
            self.sequence_form_strategy = self.strategy.copy()
            self.prev_strategy = self.strategy.copy()
            
            self.strategy_reach = LiteEFG.aggregate(self.sequence_form_strategy, aggregator="sum", object="parent", padding=1.0)
            # return pad when the object not found. Here it is return 1.0 when no parent exists (root infoset)
            self.sequence_form_strategy.inplace(self.strategy_reach * self.strategy)

            if balanced:
                self.uniform = LiteEFG.normalize(self.subtree_size, p_norm=1.0, ignore_negative=True)
            else:
                self.uniform = self.strategy.copy()
            self.sequence_form_uniform = self.uniform.copy()
            self.explore_strategy = self.uniform.copy()
        
            self.uniform_reach = LiteEFG.aggregate(self.sequence_form_uniform, aggregator="sum", object="parent", padding=1.0)
            self.sequence_form_uniform.inplace(self.uniform_reach * self.uniform)

            self.regret_buffer = LiteEFG.const(self.action_set_size, 0.0)

        with LiteEFG.backward():

            counterfactual_value = LiteEFG.aggregate(expectation, aggregator="sum")
            counterfactual_value.inplace(counterfactual_value + self.utility / (self.reach_prob * self.explore_strategy))
            expectation.inplace(LiteEFG.dot(counterfactual_value, self.strategy))

            self.regret_buffer.inplace(self.regret_buffer + counterfactual_value - expectation)
            self.prev_strategy.inplace(self.strategy.copy())
            self.strategy.inplace(LiteEFG.normalize(self.regret_buffer, p_norm=1.0, ignore_negative=True))
            if rm_plus:
                self.regret_buffer.inplace(LiteEFG.maximum(self.regret_buffer, 0.0))

        with LiteEFG.forward():

            self.strategy_reach.inplace(LiteEFG.aggregate(self.sequence_form_strategy, aggregator="sum", object="parent", padding=1.0))
            self.sequence_form_strategy.inplace(self.strategy_reach * self.strategy)
            self.explore_strategy.inplace(LiteEFG.normalize(delta * self.sequence_form_uniform + (1 - delta) * self.sequence_form_strategy, p_norm=1.0))

        print("===============Graph is ready for Outcome-Sampling MCCFR===============")
        print("delta: %f, rm_plus: %s" % (self.delta, self.rm_plus))
        print("====================================================\n")
    
    def update_graph(self, env : LiteEFG.Environment) -> None:
        env.update([self.explore_strategy, self.strategy], upd_player=1)
        env.update([self.prev_strategy, self.explore_strategy], upd_player=2)
    
    def current_strategy(self) -> LiteEFG.GraphNode:
        return self.strategy

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="leduc_poker")
    parser.add_argument("--rm+", dest="rm_plus", help="regret matching+ or not", action="store_true")
    parser.add_argument("--balanced", help="balance exploration or not", action="store_true")
    parser.add_argument("--delta", help="exploration rate", type=float, default=0.1)
    parser.add_argument("--iter", type=int, default=100000)
    parser.add_argument("--print_freq", type=int, default=1000)

    args = parser.parse_args()

    from utils import train
    train(graph(args.delta, args.rm_plus, args.balanced), "Outcome", "avg-iterate", args.iter, args.print_freq, args.game, output_strategy=True)
