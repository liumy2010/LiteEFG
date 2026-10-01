import pyspiel
import os
import LiteEFG as leg
import numpy as np
import pandas as pd
from open_spiel.python.policy import TabularPolicy
import typing
import tempfile
import hashlib
from urllib.parse import quote


_CACHE_HEADER = "# LiteEFG OpenSpiel cache format 2\n"


def NodeName(node):
    text = node if isinstance(node, str) else node.serialize()
    # Prefix even empty serializations, and escape delimiters without merging
    # distinct strings such as "a b" and "a_b" or "a\nb" and "a/b".
    return "node_" + quote(text, safe="")

def InfosetName(node, idx):
    player = 0 if isinstance(node, str) else node.current_player() + 1
    text = node if isinstance(node, str) else node.information_state_string()
    return "pl%d_%d__%s" % (player, idx, quote(text, safe=""))

class OpenSpielEnv(leg.FileEnv):
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

        if not game.get_type().provides_information_state_string:
            raise ValueError("The game must provide information-state strings for tabular conversion")
        policy = TabularPolicy(game)
        self.state_lookup = {quote(key, safe=""): index
                             for key, index in policy.state_lookup.items()}
        self._information_states = list(policy.state_lookup)

        game_full_name = game.get_type().short_name
        for k in game.get_parameters():
            game_full_name += "_%s=%s"%(k, game.get_parameters()[k])

        current_directory = os.path.expanduser('~')
        os.makedirs(os.path.join(current_directory, "game_instances"), exist_ok=True)
        cache_name = quote(game_full_name, safe="=_-")
        if len(cache_name) > 200:
            # Nested simultaneous-game parameters can exceed a filesystem's
            # filename limit after escaping. Keep their identity in the digest.
            digest = hashlib.sha256(game_full_name.encode("utf-8")).hexdigest()[:32]
            prefix = quote(game.get_type().short_name, safe="_-")[:48]
            cache_name = prefix + "_" + digest
        file_name = os.path.join(current_directory, "game_instances",
                                 cache_name + ".openspiel")

        if os.path.exists(file_name) and not regenerate:
            with open(file_name, encoding="utf-8") as cached:
                compatible_cache = cached.readline() == _CACHE_HEADER
            if compatible_cache:
                super().__init__(file_name, traverse_type=traverse_type)
                return
        
        print("Generating %s.openspiel instance from OpenSpiel"%(game_full_name))
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode="w", encoding="utf-8", newline="\n",
                    dir=os.path.dirname(file_name), prefix=".openspiel-",
                    suffix=".tmp", delete=False) as file:
                temporary_path = file.name
                self._write_game(file)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, file_name)
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                os.unlink(temporary_path)
        super().__init__(file_name, traverse_type=traverse_type)

    def _write_game(self, file):
        game = self.game
        game_name = game.get_type().short_name
        infosets = {}
        num_infosets = [0 for _ in range(game.num_players())]
        queue = [game.new_initial_state()]

        file.write(_CACHE_HEADER)
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
                    print(" %d=%.17g"%(player+1, reward), end='', file=file)
                print(file=file)
                continue

            if node.is_chance_node():
                print("node %s chance actions"%NodeName(node), end='', file=file)
                for action, prob in node.chance_outcomes():
                    child = node.clone()
                    child.apply_action(action)
                    queue.append(child)
                    print(" %s=%.17g"%(NodeName(child), prob), end='', file=file)
                print(file=file)
            else:
                print("node %s player %d actions"%(NodeName(node), node.current_player()+1), end='', file=file)
                for action in node.legal_actions():
                    child = node.clone()
                    child.apply_action(action)
                    queue.append(child)
                    print(" %s"%NodeName(child), end='', file=file)
                print(file=file)

                try:
                    infoset = (node.current_player(), node.information_state_string())
                except RuntimeError as error:
                    raise ValueError("The game %s does not have information state implemented by OpenSpiel \
                                        (typically such games are also too large to run tabular algorithms)"%game_name) from error
                if infoset not in infosets:
                    infosets[infoset] = [InfosetName(node, num_infosets[node.current_player()])]
                    num_infosets[node.current_player()] += 1
                infosets[infoset].append(node.serialize())

        for infoset in infosets:
            print("infoset %s nodes"%infosets[infoset][0], end='', file=file)
            for node in infosets[infoset][1:]:
                print(" %s"%NodeName(node), end='', file=file)
            print(file=file)
        
    def get_value(self, player: int, node: leg.GraphNode) -> typing.List[typing.Tuple[str, typing.List[float]]]:
        values = super().get_value(player, node)
        ret = []
        for k, vector in values:
            idx = self.state_lookup[k[k.find('__')+2:]]
            ret.append((self._information_states[idx], vector))
        return ret

    def get_strategy(self, strategy_node: leg.GraphNode, type_name="default") -> typing.Tuple[TabularPolicy, typing.List[pd.DataFrame]]:
        df_list = []
        policy = TabularPolicy(self.game)

        for player in range(self.game.num_players()):
            columns = ["Infoset"] + [self.game.action_to_string(player, action)
                                      for action in range(self.game.num_distinct_actions())]
            strategy = super().get_strategy(player+1, strategy_node, type_name)
            data_list = []
            for infoset, probs in strategy:
                idx = self.state_lookup[infoset[infoset.find('__')+2:]]
                policy.action_probability_array[idx][policy.legal_actions_mask[idx]>0.5] = probs
                data_list.append([self._information_states[idx]] +
                                 list(policy.action_probability_array[idx]))
            df_list.append(pd.DataFrame(data_list, columns=columns))
        return policy, df_list

    def set_value(self, player: int, node: leg.GraphNode, values: typing.List[typing.List]) -> None:
        super().set_value(player, node, values)

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
    env = OpenSpielEnv(pyspiel.load_game("kuhn_poker"))
    import LiteEFG.baselines.CFR as CFR
    import LiteEFG.baselines.CFRplus as CFRplus

    graph = CFR.graph()
    env.set_graph(graph)
    for i in range(10000):
        graph.update_graph(env)
        env.update_strategy(graph.current_strategy())
    policy, _ = env.get_strategy(graph.current_strategy(), "avg-iterate")
    env.interact(policy)
