import LiteEFG
from tqdm import tqdm
import pyspiel
import csv

def train(graph, traverse_type, convergence_type, iter, print_freq, game_env="leduc_poker", output_strategy=False):
    game = pyspiel.load_game(game_env)
    env = LiteEFG.OpenSpielEnv(game, traverse_type=traverse_type, regenerate=False)
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
    import sys
    import os
    import runpy

    package_dir = os.path.dirname(__file__)
    python_files = [file for file in os.listdir(package_dir) if file.endswith('.py') and file != '__init__.py' and file != 'utils.py']

    original_argv = sys.argv
    for file in python_files:
        module_name = f'{file[:-3]}'
        sys.argv = [module_name, '--iter', '1000', '--print_freq', '100', '--game', 'leduc_poker(suit_isomorphism=True)']
        runpy.run_module(module_name, run_name='__main__')
    
    sys.argv = original_argv

    print("Test Success")

if __name__ == "__main__":
    test()
