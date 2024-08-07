#######################################################
# Implicit Exploration Online Mirror Descent (IXOMD)
# Kozuno, Tadashi, Pierre Ménard, Rémi Munos, and Michal Valko
# "Learning in two-player zero-sum partially observable Markov games with perfect recall."
# Advances in Neural Information Processing Systems (2021)
#######################################################

import LiteEFG

class graph(LiteEFG.Graph):
    def __init__(self, eta=0.001, gamma=0.0005):
        super().__init__()

        self.eta = eta
        self.gamma = gamma
        self.initialized = False

        with LiteEFG.forward(is_static=True):
            self.strategy = LiteEFG.const(self.action_set_size, 1.0/self.action_set_size)
            expectation = LiteEFG.const(size=1, val=0.0)
            self.Z = LiteEFG.const(size=1, val=1.0)
        
        with LiteEFG.backward():
            gradient = LiteEFG.aggregate(expectation, aggregator="sum") + self.utility / (self.reach_prob * self.strategy + self.gamma) * self.eta
            # gradient = \eta * r_h^t + \log Z_{h+1}^t

            logit_max = LiteEFG.max(gradient)
            self.Z.inplace(LiteEFG.log(LiteEFG.sum(self.strategy * LiteEFG.exp(gradient - logit_max))) + logit_max) # Z_h^t
            self._update(gradient, self.strategy, self.strategy)
            expectation.inplace(self.Z.copy())

        print("===============Graph is ready for IXOMD===============")
        print("eta: %f, gamma: %f" % (self.eta, self.gamma))
        print("======================================================\n")
    
    def _update(self, gradient, upd_u, ref_u):
        upd_u.inplace(ref_u.log() + gradient)
        upd_u.inplace(upd_u - upd_u.max())
        upd_u.inplace(upd_u.exp())
        upd_u.inplace(upd_u.project(distance="KL"))
    
    def update_graph(self, env : LiteEFG.Environment) -> None:
        env.update(self.strategy)

    def current_strategy(self) -> LiteEFG.GraphNode:
        return self.strategy

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default="leduc_poker")
    parser.add_argument("--eta", help="learning rate", type=float, default=0.001)
    parser.add_argument("--gamma", help="IX parameter", type=float, default=0.0005)
    parser.add_argument("--iter", type=int, default=1000000)
    parser.add_argument("--print_freq", type=int, default=1000)

    args = parser.parse_args()

    from utils import train
    train(graph(args.eta, args.gamma), "Outcome", "avg-iterate", args.iter, args.print_freq, args.game, output_strategy=True)
