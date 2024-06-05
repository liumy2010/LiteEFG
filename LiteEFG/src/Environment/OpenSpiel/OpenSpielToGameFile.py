import pyspiel
import os
import LiteEFG
import numpy as np
import pandas as pd
from open_spiel.python.policy import TabularPolicy
import typing

def NodeName(node):
    if isinstance(node, str):
        return node.replace("\n", "/")
    return node.serialize().replace("\n", "/").replace(" ", "_")

def InfosetName(node, idx):
    infoset = "pl%d_%d__"%(0, idx) + node.replace("\n", "/") if isinstance(node, str) \
                    else "pl%d_%d__"%(node.current_player()+1, idx) + node.information_state_string().replace("\n", "/")
    return infoset.replace(" ", "_")

class OpenSpielEnv(LiteEFG.FileEnv):
    def __init__(self, game: pyspiel.Game, traverse_type="Enumerate", regenerate=False):
        if not isinstance(game, pyspiel.Game):
            raise ValueError("game must be an instance of pyspiel.Game")
        
        if game.get_type().dynamics == pyspiel.GameType.Dynamics.SEQUENTIAL:
            self.game = game
        elif game.get_type().dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS: 
            game = pyspiel.convert_to_turn_based(game)
            self.game = game
        else:
            raise ValueError("The game must be either sequential or simultaneous")

        policy = TabularPolicy(game)
        self.state_lookup = {}
        for k in policy.state_lookup.keys():
            infoset = InfosetName(k, 0)
            infoset = infoset[infoset.find('__')+2:]
            self.state_lookup[infoset] = policy.state_lookup[k]

        game_name = game.get_type().short_name
        infosets = {}
        num_infosets = [0 for _ in range(game.num_players())]
        queue = [game.new_initial_state()]

        game_full_name = game_name
        for k in game.get_parameters():
            game_full_name += "_%s=%s"%(k, game.get_parameters()[k])

        current_directory = os.path.expanduser('~')#os.path.dirname(os.path.abspath(__file__))
        #for i in range(3):
        #    current_directory = os.path.dirname(current_directory)
        os.makedirs(os.path.join(current_directory, "game_instances"), exist_ok=True)
        file_name = os.path.join(current_directory, "game_instances", game_full_name + ".openspiel")

        if os.path.exists(file_name) and not regenerate:
            super().__init__(file_name, traverse_type=traverse_type)
            return
        
        print("Generating %s.openspiel instance from OpenSpiel"%(game_full_name))
        file = open(file_name, "w")

        print("# %s instance with parameters:"%game_name, file=file)
        print("#", file=file)
        print("# Opt {", file=file)
        print("#     openspiel,", file=file)
        for k in game.get_parameters():
            print("#     %s: %s,"%(k, game.get_parameters()[k]), file=file)
        if "players" not in game.get_parameters():
            print("#     players: %d,"%game.num_players(), file=file)
        print("# }", file=file)
        print("#", file=file)

        while len(queue) > 0:
            node = queue.pop()
            
            if node.is_terminal():
                print("node %s leaf payoffs"%NodeName(node), end='', file=file)
                for player, reward in enumerate(node.returns()):
                    print(" %d=%f"%(player+1, reward), end='', file=file)
                print(file=file)
                continue

            if node.is_chance_node():
                print("node %s chance actions"%NodeName(node), end='', file=file)
                for action, prob in node.chance_outcomes():
                    child = node.clone()
                    child.apply_action(action)
                    queue.append(child)
                    print(" %s=%.8f"%(NodeName(child.serialize()), prob), end='', file=file)
                print(file=file)
            else:
                print("node %s player %d actions"%(NodeName(node), node.current_player()+1), end='', file=file)
                for action in node.legal_actions():
                    child = node.clone()
                    child.apply_action(action)
                    queue.append(child)
                    print(" %s"%NodeName(child.serialize()), end='', file=file)
                print(file=file)

                try:
                    infoset = node.information_state_string()
                except:
                    raise ValueError("The game %s does not have information state implemented by OpenSpiel \
                                        (typically such games are also too large to run tabular algorithms)"%game_name)
                if infoset not in infosets:
                    infosets[infoset] = [InfosetName(node, num_infosets[node.current_player()])]
                    num_infosets[node.current_player()] += 1
                infosets[infoset].append(node.serialize())

        for infoset in infosets:
            print("infoset %s nodes"%infosets[infoset][0], end='', file=file)
            for node in infosets[infoset][1:]:
                print(" %s"%NodeName(node), end='', file=file)
            print(file=file)
        
        file.close()
        super().__init__(file_name, traverse_type=traverse_type)
    
    def get_value(self, player: int, node: LiteEFG.GraphNode) -> typing.List[typing.Tuple[str, float]]:
        values = super().get_value(player+1, node)
        policy = TabularPolicy(self.game)
        ret = []
        for k, _ in values:
            idx = self.state_lookup[k[k.find('__')+2:]]
            infoset = list(policy.state_lookup.keys())[idx]
            ret.append((infoset, _))
        return ret

    def get_strategy(self, strategy_node: LiteEFG.GraphNode, type_name="default") -> typing.Tuple[TabularPolicy, typing.List[pd.DataFrame]]:
        df_list = []
        policy = TabularPolicy(self.game)

        for player in range(self.game.num_players()):
            #print("Player %d strategy"%player)
            df = pd.DataFrame(columns=["Infoset"] + [self.game.action_to_string(player, _) for _ in range(self.game.num_distinct_actions())])
            strategy = super().get_strategy(player+1, strategy_node, type_name)
            data_list = []
            for infoset, probs in strategy:
                idx = self.state_lookup[infoset[infoset.find('__')+2:]]
                policy.action_probability_array[idx][policy.legal_actions_mask[idx]>0.5] = probs
                new_row = pd.DataFrame([[list(policy.state_lookup.keys())[idx]] + list(policy.action_probability_array[idx])], columns=df.columns)
                data_list.append(new_row)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)
                df = pd.concat([df]+data_list, ignore_index=True)
            df_list.append(df)
        
        #from open_spiel.python.algorithms import exploitability
        #expl = exploitability.exploitability(self.game, policy)
        #print("Exploitability: %f"%expl)
        return policy, df_list

    def interact(self, policy: TabularPolicy, controlled_player=0, reveal_private=True, epochs=1000) -> None:

        print("\nYou are player: %d in %s (players indexed from 0 to %d)"%(controlled_player, self.game.get_type().short_name, self.game.num_players()-1))

        accumulated_payoff = np.zeros(self.game.num_players())
        for epoch in range(epochs):
            print("\n========== Epoch %d ==========\n"%epoch)
            s = self.game.new_initial_state()
            while not s.is_terminal():
                if s.is_chance_node():
                    chance_outcomes = s.chance_outcomes()
                    prob = [_[1] for _ in chance_outcomes]
                    actions = [_[0] for _ in chance_outcomes]
                    action = np.random.choice(actions, p=prob)
                    s.apply_action(action)
                elif s.current_player() == controlled_player:
                    print(s.information_state_string())
                    actions = s.legal_actions()
                    while True:
                        print("Valid Actions: ", end='')
                        for i, action in enumerate(actions):
                            print("%d="%action + self.game.action_to_string(controlled_player, action), end=" ")
                        action = input("\nYour Choice: ")
                        if int(action) not in actions:
                            print("Invalid action! Please choose from the valid actions.")
                            continue
                        break
                    s.apply_action(int(action))
                else:
                    probs = policy.action_probabilities(s)
                    action = np.random.choice(list(probs.keys()), p=list(probs.values()))
                    s.apply_action(action)
            
            print("\n========== Epoch %d Summary ==========\n"%epoch)
            if not reveal_private:
                print("Your last observation is: ", s.information_state_string(controlled_player))
            else:
                print("The final outcome of the game is:\n")
                print(s)
            print("You are player: %d (players indexed from 0 to %d)"%(controlled_player, self.game.num_players()-1))
            print("Your Payoff: ", s.returns()[controlled_player])
            accumulated_payoff += s.returns()
            print("Accumulated Payoff of Each Player: ", accumulated_payoff)
            print("Average Payoff of Each Player: ", accumulated_payoff / (epoch+1))

if __name__ == "__main__":
    env = OpenSpielEnv(pyspiel.load_game("leduc_poker(suit_isomorphism=True)"))
    import LiteEFG.baselines.CFR as CFR
    import LiteEFG.baselines.CFRplus as CFRplus

    graph = CFR.graph()
    env.set_graph(graph)
    for i in range(10000):
        graph.update_graph(env)
        env.update_strategy(graph.current_strategy())
    policy, _ = env.get_strategy(graph.current_strategy(), "avg-iterate")
    env.interact(policy)
