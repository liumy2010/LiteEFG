import LiteEFG
from tqdm import tqdm
import pyspiel
import csv

def train(graph, traverse_type, convergence_type, iter, print_freq, game_env="leduc_poker", output_strategy=False):
    game = pyspiel.load_game(game_env)
    #game = pyspiel.create_matrix_game([[0,1],[1,0]], [[0,1],[1,0]])
    env = LiteEFG.OpenSpielEnv(game, traverse_type=traverse_type, regenerate=True)
    #env = LiteEFG.FileEnv("GameInstances/kuhn_poker.openspiel", traverse_type=args.traverse_type)
    env.set_graph(graph)

    pbar = tqdm(total=iter)
    best_exp = 1e9

    for i in range(iter):
        graph.update_graph(env)
        env.update_strategy(graph.current_strategy(), update_best=(convergence_type == "best-iterate"))
            
        if i % print_freq == 0:
            exploitability = sum(env.exploitability(graph.current_strategy(), convergence_type))
            best_exp = min(best_exp, exploitability)
            pbar.set_description(f'Exploitability: {exploitability:.8f}, Best: {best_exp:.8f}')
            pbar.update(print_freq)

    if output_strategy:
        _, df_list = env.get_strategy(graph.current_strategy(), "avg-iterate")
        for i, df in enumerate(df_list):
            df['Infoset'] = df['Infoset'].apply(lambda x: x.replace('\n', '\\n'))
            df.to_csv("strategy_" + str(i) + ".csv", quoting=csv.QUOTE_MINIMAL, quotechar='"')

def test():
    import CFR
    import CFRplus
    import DCFR
    import DOMD
    import MMD
    import OS_MCCFR
    import QFR
    import Bil_QFR
    import CMD
    alg_list = [CFR.graph(), CFRplus.graph(), DCFR.graph(), DOMD.graph(), MMD.graph(), OS_MCCFR.graph(), QFR.graph(), Bil_QFR.graph(), CMD.graph()]
    traverse_type_list = ["Enumerate", "Enumerate", "Enumerate", "Enumerate", "Enumerate", "Outcome", "Enumerate", "Enumerate", "Enumerate"]
    convergence_type_list = ["avg-iterate", "linear-avg-iterate", "last-iterate", "last-iterate", "default", "avg-iterate", "default", "default", "last-iterate"]
    alg_name_list = ["CFR", "CFR+", "DCFR", "DOMD", "MMD", "OS-MCCFR", "QFR", "Bil-QFR", "CMD"]

    for i, alg in enumerate(alg_list):
        train(alg, traverse_type_list[i], convergence_type_list[i], 100000, 1000, "kuhn_poker", True)
        print(alg_name_list[i] + " finished training")
    print("Test Success")

if __name__ == "__main__":
    test()
