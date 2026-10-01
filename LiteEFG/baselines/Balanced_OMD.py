#######################################################
# Balanced Online Mirror Descent (Balanced-OMD)
# Bai, Yu, Chi Jin, Song Mei, and Tiancheng Yu.
# "Near-optimal learning of extensive-form games with imperfect information."
# International Conference on Machine Learning (2022).
#######################################################

import LiteEFG as leg
from LiteEFG.baselines.baseline import _baseline
import numpy as np

class graph(_baseline):
    def __init__(self, eta=0.001, gamma=0.0005):
        super().__init__()

        self.eta = eta
        self.gamma = gamma
        self.initialized = False

        with leg.forward(is_static=True):
            self.depth = leg.const(size=1, val=1.0)
            self.depth.inplace(leg.aggregate(self.depth, "max", object="parent", player="self", padding=0) + 1.0)
            self.strategy = leg.const(self.action_set_size, 1.0 / self.action_set_size)
            expectation = leg.const(size=1, val=0.0)
            self.visit_prob = leg.const(size=1, val=0.0)

            self.target_depth = leg.const(size=1, val=1.0)
            self.subtree_infoset_number = leg.const(size=1, val=0.0)
            self.subtree_infoset_number_vector = leg.const(self.action_set_size, 0.0)
            self.parent_subtree_size = leg.const(size=1, val=0.0)
            self.transition_prob = leg.const(size=1, val=1.0)
            self.zero = leg.const(size=1, val=0.0)
            self.is_root = leg.aggregate(self.zero, "sum", object="parent", player="self", padding=1.0)

        # Reuse Balanced_FTRL's subtree counting and transition recursion,
        # counting only infosets at the selected depth instead of all actions.
        with leg.backward(color=1):
            self.subtree_infoset_number.inplace(leg.aggregate(self.subtree_infoset_number, "sum", object="children", player="self", padding=0))
            self.subtree_infoset_number_vector.inplace(self.subtree_infoset_number.copy())
            self.subtree_infoset_number.inplace(self.subtree_infoset_number.sum() + (self.depth == self.target_depth))

        with leg.forward(color=1):
            self.transition_prob.inplace(leg.aggregate(self.transition_prob, "sum", object="parent", player="self", padding=1.0))
            self.parent_subtree_size.inplace(leg.aggregate(self.subtree_infoset_number_vector, "sum", object="parent", player="self", padding=0) *\
                                              (1.0 - self.is_root) + self.parent_subtree_size * self.is_root)
            # A branch without target-depth descendants has zero weight.
            self.transition_prob.inplace(self.transition_prob / leg.maximum(self.parent_subtree_size, 1.0) * self.subtree_infoset_number)
        
        with leg.backward(color=0):
            self.visit_action_prob = self.visit_prob / self.action_set_size
            gradient = leg.aggregate(expectation, aggregator="sum") + self.utility / (self.reach_prob * self.strategy + self.gamma * self.visit_action_prob)

            self.prev_strategy =  self.strategy.copy()
            self._update(gradient, self.strategy, self.prev_strategy)

            kl = leg.dot((self.strategy / self.prev_strategy).log(), self.strategy) / (self.eta * self.visit_action_prob)
            expectation.inplace(leg.dot(gradient, self.strategy) - kl)

        print("===============Graph is ready for Balanced OMD===============")
        print("eta: %f, gamma: %f" % (self.eta, self.gamma))
        print("=============================================================\n")
    
    def _update(self, gradient, upd_u, ref_u):
        gradient_div = gradient * self.eta * self.visit_action_prob
        upd_u.inplace(ref_u.log() + gradient_div)
        upd_u.inplace(upd_u - upd_u.max())
        upd_u.inplace(upd_u.exp())
        upd_u.inplace(upd_u.project(distance="KL"))
    
    def update_graph(self, env : leg.Environment) -> None:
        if not self.initialized:
            for i in range(1, 3): # compute \mu^{*,h}_{1:h} for each infoset
                depth = env.get_value(i, self.depth)
                is_root = env.get_value(i, self.is_root)
                max_depth = 0
                for j in depth:
                    max_depth = max(max_depth, np.round(j[1][0]).astype(int))
                depth_infoset_count = np.zeros(max_depth + 1)

                for j in depth:
                    depth_infoset_count[np.round(j[1][0]).astype(int)] += 1
                
                visit_prob = [[0.0] for _ in range(len(depth))]
                for h in range(1, max_depth + 1):
                    parent_subtree_size = [[0.0] for _ in range(len(depth))]
                    for j in range(len(depth)):
                        if is_root[j][1][0] > 0.5:
                            parent_subtree_size[j][0] = depth_infoset_count[h]
                    env.set_value(i, self.target_depth, [[float(h)] for _ in range(len(depth))])
                    env.set_value(i, self.parent_subtree_size, parent_subtree_size)
                    env.update(self.strategy, upd_player=i, upd_color=[1], traverse_type="Enumerate")

                    transition_prob = env.get_value(i, self.transition_prob)
                    for j in range(len(depth)):
                        if np.round(depth[j][1][0]).astype(int) == h:
                            # Lemma C.6: mu^{*,h}_{1:h} = 1 / (X_h A p^{*,h}_{1:h}).
                            # The action factor is applied by visit_action_prob.
                            visit_prob[j][0] = 1.0 / (depth_infoset_count[h] * transition_prob[j][1][0])
                env.set_value(i, self.visit_prob, visit_prob)
            self.initialized = True
            
        env.update(self.strategy, upd_color=[0])

    def current_strategy(self) -> leg.GraphNode:
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
