#######################################################
# Follow the Perturbed Leader (FTPL)
# Hazan, Elad. 
# "Introduction to online convex optimization."
# Foundations and Trends® in Optimization 2.3-4 (2016): 157-325.
#######################################################

import LiteEFG
from LiteEFG.baselines.baseline import _baseline

class graph(_baseline):
    def __init__(self, eta=0.01, noise_type="exponential"):
        super().__init__()

        self.eta = eta
        self.noise_type = noise_type
        
        with LiteEFG.backward(is_static=True):

            expectation = LiteEFG.const(size=1, val=0.0)
            self.strategy = LiteEFG.const(self.action_set_size, 1.0 / self.action_set_size)
            self.cumulative_utility = LiteEFG.const(self.action_set_size, 0.0)

        with LiteEFG.backward():

            if self.noise_type == "uniform":
                noise = LiteEFG.random.uniform(self.action_set_size, 0.0, 1.0 / self.eta)
            elif self.noise_type == "normal":
                noise = LiteEFG.random.normal(self.action_set_size, 0.0, 1.0 / self.eta)
            else:
                noise = LiteEFG.random.exponential(self.action_set_size, self.eta)

            counterfactual_value = LiteEFG.aggregate(expectation, aggregator="sum") + self.utility
            expectation.inplace(LiteEFG.dot(counterfactual_value, self.strategy))
            self.cumulative_utility.inplace(self.cumulative_utility + counterfactual_value)

            self.strategy.inplace(LiteEFG.argmax(self.cumulative_utility + noise))

        print("===============Graph is ready for FTPL===============")
        print("eta: %f, noise: %s" % (self.eta, self.noise_type))
        print("====================================================\n")

    def update_graph(self, env : LiteEFG.Environment) -> None:
        env.update(self.strategy)

    def current_strategy(self) -> LiteEFG.GraphNode:
        return self.strategy

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="leduc_poker")
    parser.add_argument("--traverse_type", type=str, choices=["Enumerate", "External"], default="Enumerate")
    parser.add_argument("--eta", type=float, default=0.01)
    parser.add_argument("--noise", type=str, choices=["uniform", "normal", "exponential"], default="exponential")
    parser.add_argument("--iter", type=int, default=100000)
    parser.add_argument("--print_freq", type=int, default=1000)

    args = parser.parse_args()
    from utils import train
    train(graph(eta=args.eta, noise_type=args.noise), args.traverse_type, "avg-iterate", args.iter, args.print_freq, args.game, True)
