#######################################################
# Discounted Counterfactual Regret Minimization (DCFR)
# Brown, Noam, and Tuomas Sandholm. "Solving imperfect-information games via discounted regret minimization." 
# AAAI 2019.
#######################################################

import LiteEFG

class graph(LiteEFG.Graph):
    def __init__(self, alpha=1.5, beta=0, gamma=2):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.threshold = 10

        with LiteEFG.backward(is_static=True):

            self.timestep = LiteEFG.const(size=1, val=0.0)
            expectation = LiteEFG.const(size=1, val=0.0)
            self.strategy = LiteEFG.const(self.action_set_size, 1.0 / self.action_set_size)
            self.avg_seq_strategy = LiteEFG.const(self.action_set_size, 0.0)
            self.regret_buffer = LiteEFG.const(self.action_set_size, 0.0)
            self.avg_strategy = self.strategy.copy()

            self.pos_coeff = LiteEFG.const(1, 1.0) if self.alpha>self.threshold else LiteEFG.const(1, 0.0)
            self.neg_coeff = LiteEFG.const(1, 1.0) if self.beta>self.threshold else LiteEFG.const(1, 0.0)

        with LiteEFG.backward():

            self.strategy_coef = (self.timestep / (self.timestep + 1)) ** self.gamma
            self.timestep.inplace(self.timestep + 1)
            
            counterfactual_value = LiteEFG.aggregate(expectation, aggregator="sum") + self.utility
            expectation.inplace(LiteEFG.dot(counterfactual_value, self.strategy))

            self.neg_regret = self.regret_buffer < 0
            self.pos_regret = self.regret_buffer >= 0

            self.regret_buffer.inplace((self.neg_regret * self.regret_buffer) * self.neg_coeff + (self.pos_regret * self.regret_buffer) * self.pos_coeff)

            self.regret_buffer.inplace(self.regret_buffer + counterfactual_value - expectation)
            self.avg_seq_strategy.inplace(self.avg_seq_strategy * self.strategy_coef + self.strategy * self.reach_prob)
            self.avg_strategy.inplace(self.avg_seq_strategy.normalize(p_norm=1.0, ignore_negative=True))
            self.strategy.inplace(LiteEFG.normalize(self.regret_buffer, p_norm=1.0, ignore_negative=True))

            if abs(self.alpha) < self.threshold:
                self.pos_coeff.inplace((self.timestep ** self.alpha) / (self.timestep ** self.alpha + 1))
            if abs(self.beta) < self.threshold:
                self.neg_coeff.inplace((self.timestep ** self.beta) / (self.timestep ** self.beta + 1))

    def update_graph(self, env : LiteEFG.Environment) -> None:
        env.update(self.strategy, upd_player=1)
        env.update(self.strategy, upd_player=2)
    
    def current_strategy(self, type_name="last-iterate"):
        assert(type_name in ["last-iterate", "average-iterate"])
        return self.strategy if type_name == "average-iterate" else self.avg_strategy

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="leduc_poker")
    parser.add_argument("--traverse_type", type=str, choices=["Enumerate", "External"], default="Enumerate")
    parser.add_argument("--alpha", type=int, default=1.5)
    parser.add_argument("--beta", type=int, default=0)
    parser.add_argument("--gamma", type=int, default=2)
    parser.add_argument("--iter", type=int, default=100000)
    parser.add_argument("--print_freq", type=int, default=1000)

    args = parser.parse_args()
    from utils import train
    train(graph(args.alpha, args.beta, args.gamma), args.traverse_type, "last-iterate", args.iter, args.print_freq, args.game)
