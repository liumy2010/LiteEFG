#######################################################
# Balanced Online Mirror Descent (Balanced-OMD)
# Bai, Yu, Chi Jin, Song Mei, and Tiancheng Yu
# "Near-optimal learning of extensive-form games with imperfect information."
# International Conference on Machine Learning (2022)
#######################################################

import LiteEFG
import numpy as np

class graph(LiteEFG.Graph):
    def __init__(self, eta=0.001, gamma=0.0005):
        super().__init__()

        self.eta = eta
        self.gamma = gamma
        self.initialized = False

        with LiteEFG.forward(is_static=True):
            self.depth = LiteEFG.const(size=1, val=1.0)
            self.depth.inplace(LiteEFG.aggregate(self.depth, "max", object="parent", player="self", padding=0) + 1.0)
            self.strategy = LiteEFG.const(self.action_set_size, 1.0 / self.action_set_size)
            expectation = LiteEFG.const(size=1, val=0.0)
            self.visit_prob = LiteEFG.const(size=1, val=0.0)
        
        with LiteEFG.backward():
            self.visit_action_prob = self.visit_prob / self.action_set_size
            gradient = LiteEFG.aggregate(expectation, aggregator="sum") + self.utility / (self.reach_prob * self.strategy + self.gamma * self.visit_action_prob)

            self.prev_strategy =  self.strategy.copy()
            self._update(gradient, self.strategy, self.prev_strategy)

            kl = LiteEFG.dot((self.strategy / self.prev_strategy).log(), self.strategy) / (self.eta * self.visit_action_prob)
            expectation.inplace(LiteEFG.dot(gradient, self.strategy) - kl)

        print("===============Graph is ready for Balanced OMD===============")
        print("eta: %f, gamma: %f" % (self.eta, self.gamma))
        print("=============================================================\n")
    
    def _update(self, gradient, upd_u, ref_u):
        gradient_div = gradient * self.eta * self.visit_action_prob
        upd_u.inplace(ref_u.log() + gradient_div)
        upd_u.inplace(upd_u - upd_u.max())
        upd_u.inplace(upd_u.exp())
        upd_u.inplace(upd_u.project(distance="KL"))
    
    def update_graph(self, env : LiteEFG.Environment) -> None:
        if not self.initialized:
            self.initialized = True
            for i in range(1, 3): # compute \mu^{*,h}_{1:h} for each infoset
                depth = env.get_value(i, self.depth)
                max_depth = 0
                for j in depth:
                    max_depth = max(max_depth, np.round(j[1][0]).astype(int))
                depth_infoset_count = np.zeros(max_depth + 1)

                for j in depth:
                    depth_infoset_count[np.round(j[1][0]).astype(int)] += 1
                
                visit_prob = [[0] for _ in range(len(depth))]
                for j in range(len(depth)):
                    visit_prob[j][0] = 1.0# / depth_infoset_count[np.round(depth[j][1][0]).astype(int)]
                env.set_value(i, self.visit_prob, visit_prob)
            
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
