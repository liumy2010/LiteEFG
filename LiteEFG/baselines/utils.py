import LiteEFG as leg
from tqdm import tqdm
import pyspiel
import csv

def train(graph, traverse_type, convergence_type, iter, print_freq, game_env="leduc_poker", output_strategy=False,
          *, strategy_type=None):
    """Train and measure an environment convergence type for the selected policy.

    strategy_type is passed to graph.current_strategy; None preserves the
    graph's default selector. convergence_type selects the environment's
    averaging of that policy, independently of the graph's own averages.
    """
    if iter <= 0:
        raise ValueError("iter must be positive")
    if print_freq <= 0:
        raise ValueError("print_freq must be positive")
    game = pyspiel.load_game(game_env)
    env = leg.OpenSpielEnv(game, traverse_type=traverse_type, regenerate=False)
    env.set_graph(graph)

    def current_strategy():
        if strategy_type is None:
            return graph.current_strategy()
        return graph.current_strategy(strategy_type)

    best_exp = float("inf")

    with tqdm(total=iter) as pbar:
        for i in range(iter):
            graph.update_graph(env)
            env.update_strategy(current_strategy(), update_best=(convergence_type == "best-iterate"))

            if (i + 1) % print_freq == 0 or i + 1 == iter:
                exploitability = sum(env.exploitability(current_strategy(), convergence_type))
                best_exp = min(best_exp, exploitability)
                pbar.set_description(f'Exploitability: {exploitability:.8f}, Best: {best_exp:.8f}')
            pbar.update(1)

    if output_strategy:
        _, df_list = env.get_strategy(current_strategy(), "avg-iterate")
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
        sys.argv = [module_name, '--iter', '1000', '--print_freq', '100', '--game', 'leduc_poker']
        runpy.run_module(module_name, run_name='__main__')
    
    sys.argv = original_argv

    print("Test Success")

if __name__ == "__main__":
    test()
