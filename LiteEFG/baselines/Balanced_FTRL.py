#######################################################
# Fiegel, Côme, Pierre Ménard, Tadashi Kozuno, Rémi Munos, Vianney Perchet, and Michal Valko. 
# "Adapting to game trees in zero-sum imperfect information games." 
# International Conference on Machine Learning (2023).
#######################################################

import LiteEFG

class graph(LiteEFG.Graph):
    def __init__(self, eta=0.001, gamma=0.0005):
        super().__init__()

        self.eta = eta
        self.gamma = gamma
        self.initialized = False

        with LiteEFG.backward(is_static=True):
            self.subtree_action_number = self.action_set_size.copy()

            self.subtree_action_number.inplace(LiteEFG.aggregate(self.subtree_action_number, "sum", object="children", player="self", padding=0))
            self.strategy = LiteEFG.normalize(self.subtree_action_number + 1, p_norm=1.0)
            self.parent_subtree_size = LiteEFG.const(1, 0.0)

            self.subtree_action_number_vector = self.subtree_action_number.copy()
            self.subtree_action_number.inplace(self.subtree_action_number.sum() + self.action_set_size)
            
        with LiteEFG.forward(is_static=True):
            self.zero = LiteEFG.const(size=1, val=0.0)
            self.is_root = LiteEFG.aggregate(self.zero, "sum", object="parent", player="self", padding=1.0)

        with LiteEFG.forward(color=1):
            expectation = LiteEFG.const(size=1, val=0.0)
            self.visit_prob = LiteEFG.const(size=1, val=1.0)
            self.Z = LiteEFG.const(size=1, val=0.0)

            self.visit_prob.inplace(LiteEFG.aggregate(self.visit_prob, "sum", object="parent", player="self", padding=1.0))
            self.parent_subtree_size.inplace(LiteEFG.aggregate(self.subtree_action_number_vector, "sum", object="parent", player="self", padding=0) *\
                                              (1.0 - self.is_root) + self.parent_subtree_size * self.is_root)
            self.visit_prob.inplace(self.visit_prob / self.parent_subtree_size * self.subtree_action_number)
        
        with LiteEFG.backward(color=0):
            gradient = LiteEFG.aggregate(expectation, aggregator="sum") + self.utility / (self.reach_prob * self.strategy + self.gamma / self.visit_prob) * self.eta

            logit_max = LiteEFG.max(gradient)
            self.Z.inplace(LiteEFG.log(LiteEFG.sum(self.strategy * LiteEFG.exp(gradient - logit_max))) + logit_max) # Z_h^t
            self._update(gradient, self.strategy, self.strategy)
            expectation.inplace(self.Z.copy())

        print("===============Graph is ready for Balanced FTRL===============")
        print("eta: %f, gamma: %f" % (self.eta, self.gamma))
        print("==============================================================\n")
    
    def _update(self, gradient, upd_u, ref_u):
        upd_u.inplace(ref_u.log() + gradient)
        upd_u.inplace(upd_u - upd_u.max())
        upd_u.inplace(upd_u.exp())
        upd_u.inplace(upd_u.project(distance="KL"))
    
    def update_graph(self, env : LiteEFG.Environment) -> None:
        if not self.initialized:
            self.initialized = True
            for i in range(1, 3):
                subtree_action_number = env.get_value(i, self.subtree_action_number)
                is_root = env.get_value(i, self.is_root)

                parent_subtree_size = [[0.0] for _ in range(len(is_root))]

                total_actions = 0
                for j in range(len(is_root)):
                    if is_root[j][1][0] > 0.5: # root, >0.5 to avoid numerical error
                        total_actions += subtree_action_number[j][1][0]
                for j in range(len(is_root)):
                    if is_root[j][1][0] > 0.5:
                        parent_subtree_size[j][0] = total_actions
                env.set_value(i, self.parent_subtree_size, parent_subtree_size)
                env.update(self.strategy, upd_color=[1], traverse_type="Enumerate")

        env.update(self.strategy, upd_color=[0])

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
