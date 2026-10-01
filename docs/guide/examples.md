---
description: Run the checked-in algorithms, evaluate current and averaged policies, inspect infoset values, and export OpenSpiel policies.
---

# Examples and experiment recipes

Run baseline algorithms from the scripts in [`LiteEFG/baselines`](https://github.com/liumy2010/LiteEFG/tree/main/LiteEFG/baselines). Start with the [quick start](./quick-start).

## Run the baseline scripts

First [install the current repository](./installation#build-the-current-repository) so the scripts and installed native bindings come from the same checkout. From the repository root, install the runner's progress-bar dependency:

```sh
python -m pip install tqdm
```

The examples below use Counterfactual Regret Minimization (CFR),
Monte Carlo Counterfactual Regret Minimization (MCCFR), and
Q-Function based Regret Minimization (QFR). Execute the scripts by path
from the repository root:

::: code-group

```sh [CFR]
python LiteEFG/baselines/CFR.py \
  --game kuhn_poker \
  --traverse_type Enumerate \
  --iter 10000 \
  --print_freq 1000
```

```sh [External-sampling CFR]
python LiteEFG/baselines/CFR.py \
  --game kuhn_poker \
  --traverse_type External \
  --iter 10000 \
  --print_freq 1000
```

```sh [Outcome-sampling MCCFR]
python LiteEFG/baselines/OS_MCCFR.py \
  --game kuhn_poker \
  --delta 0.1 \
  --balanced \
  --iter 10000 \
  --print_freq 1000
```

```sh [QFR]
python LiteEFG/baselines/QFR.py \
  --game kuhn_poker \
  --feedback Q \
  --regularizer Entropy \
  --eta 0.1 \
  --tau 0.001 \
  --gamma 0.001 \
  --iter 10000 \
  --print_freq 1000
```

:::

Quote game strings that contain parameters so the shell passes them unchanged:

```sh
python LiteEFG/baselines/CFRplus.py \
  --game 'leduc_poker(suit_isomorphism=True)' \
  --iter 10000 \
  --print_freq 1000
```

Use a script's `--help` to see its complete flags. The scripts do not all expose the same options: Outcome-Sampling MCCFR (OS-MCCFR), Implicit Exploration Online Mirror Descent (IXOMD), Balanced Online Mirror Descent (Balanced OMD), and Balanced Follow the Regularized Leader (Balanced FTRL) select `Outcome` internally; Magnetic Mirror Descent (MMD) and QFR infer traversal from `--feedback`. See the [algorithm table](./algorithms) for exact defaults and evaluation choices.

### What the runner reports and saves

The helper creates an `OpenSpielEnv`, attaches the graph, and calls `algorithm.update_graph(env)` followed by `env.update_strategy(...)` on every iteration. At reporting intervals, it sums the values returned by `env.exploitability(...)` and displays that total as `Exploitability`. Each component is one player's improvement from a best response. The total is not divided by the number of players.

`Best` in the progress bar is the smallest reported total seen so far. It does not mean a best strategy has been saved: none of the bundled algorithm scripts selects the helper's `"best-iterate"` mode. Reports occur after every `print_freq` completed iterations and after the final iteration. Both `iter` and `print_freq` must be positive.

CFR, OS-MCCFR, IXOMD, Balanced OMD, Balanced FTRL, and Follow the Perturbed Leader (FTPL) enable comma-separated values (CSV) export in their script entry points. They write `strategy_0.csv`, `strategy_1.csv`, and so on in the current directory, replacing files with the same names. The shared helper always exports `"avg-iterate"`, escapes newlines in infoset labels, and includes the pandas row index. The filenames use zero-based player indices.

::: tip Sampling and measurement cost
An outcome-sampling update visits one trajectory, but recording a strategy and computing exploitability still traverse all infosets or the full game as needed. The helper records history on every iteration, including for MMD and QFR. For current-iterate experiments, the loop below avoids that history work.
:::

## Evaluate a current iterate

Save the standalone Python examples below outside the checkout if you installed non-editably, as in the [quick start](./quick-start). An editable installation also lets you run them from the checkout.

`type_name="default"` evaluates the strategy currently stored in the graph. It does not require `update_strategy`. This example uses MMD's Python defaults explicitly and evaluates only every 100 updates:

```python
import LiteEFG as leg
import pyspiel
from LiteEFG.baselines import MMD

leg.set_seed(0)
env = leg.OpenSpielEnv(
    pyspiel.load_game("kuhn_poker"),
    traverse_type="Enumerate",
)
algorithm = MMD.graph(
    eta=0.005,
    tau=0.05,
    gamma=0.0,
    regularizer="Entropy",
    feedback="Q",
)
env.set_graph(algorithm)

for iteration in range(1, 1001):
    algorithm.update_graph(env)
    if iteration % 100 == 0:
        improvements = env.exploitability(
            algorithm.current_strategy(), "default"
        )
        print(iteration, sum(improvements))
```

This evaluates the unregularized game's best-response improvement of the current policy. Positive regularization can change the policy the algorithm approaches, so the measured quantity need not converge to zero.

For a sampled MMD experiment, pair `feedback="Outcome"` with `traverse_type="Outcome"`. A traversal change alone does not change the estimator constructed inside the graph.

## Inspect and export an average policy

For CFR, explicitly record a strategy after every update you want included in the average. The environment averages sequence-form strategies, then converts the requested result back to local action probabilities when exporting.

```python
import LiteEFG as leg
import pyspiel
from LiteEFG.baselines import CFR

env = leg.OpenSpielEnv(pyspiel.load_game("kuhn_poker"))
algorithm = CFR.graph()
env.set_graph(algorithm)
strategy = algorithm.current_strategy()

for _ in range(1000):
    algorithm.update_graph(env)
    env.update_strategy(strategy)

print("Average-policy improvement:", env.exploitability(strategy, "avg-iterate"))
print("Average-policy utility:", env.utility(strategy, "avg-iterate"))

# LiteEFG's numerical environment methods index players from 1.
for infoset, regrets in env.get_value(1, algorithm.regret_buffer)[:3]:
    print(infoset, regrets)

# The OpenSpiel adapter returns a TabularPolicy and one DataFrame per player.
policy, frames = env.get_strategy(strategy, "avg-iterate")
for player, frame in enumerate(frames):
    print(frame.head())
    frame.to_csv(f"kuhn-player-{player}.csv", index=False)
```

Each DataFrame has an `Infoset` column and columns for the game's action names. The returned OpenSpiel `TabularPolicy` can be passed to OpenSpiel tools that accept tabular policies. The adapter also provides a terminal interaction loop:

```python
# Continue after the previous example in an interactive terminal.
env.interact(policy, controlled_player=0, reveal_private=True, epochs=3)
```

`controlled_player` follows OpenSpiel's zero-based indexing. The human chooses actions for that player; the remaining players use the supplied policy.

## Keep the best recorded policy

To select a policy by the lowest total exploitability seen at recording times, replace the history call in the CFR loop with:

```python
env.update_strategy(strategy, update_best=True)
```

Then retrieve it with:

```python
best_policy, best_frames = env.get_strategy(strategy, "best-iterate")
```

`update_best=True` evaluates exploitability at each recording call, which adds full-game work even when the algorithm samples. `"best-iterate"` is only available after such a call. For complete selector semantics and the distinction between raw environment exports and the OpenSpiel wrapper, see the [environment application programming interface (API)](environments.md#api-reference).

## Save and resume training

After attaching a graph to an environment, use `graph.save(path)` and
`graph.load(path)` to save and restore its complete training state. For example,
this runs 50 updates with a save/load cycle every 10 updates:

```python
import LiteEFG as leg
import pyspiel
from LiteEFG.baselines.CFR import graph as CFR

env = leg.OpenSpielEnv(pyspiel.load_game("kuhn_poker"))
graph = CFR()
env.set_graph(graph)

for step in range(1, 51):
    graph.update_graph(env)
    env.update_strategy(graph.current_strategy())
    if step % 10 == 0:
        graph.save("cfr.checkpoint")
        graph.load("cfr.checkpoint")
```

`graph.load` restores the existing graph and environment in place. It checks the
Python graph class, native computation structure, and game tree before accepting
the checkpoint. Saved state includes graph operations, values at every
information set, strategy averages, Python attributes such as update counters,
and global Python, NumPy, and native random streams, so schedules and stochastic
updates continue from the saved point.

To resume in a fresh process, import LiteEFG and load both objects directly:

```python
import LiteEFG as leg

env, graph = leg.load_checkpoint("cfr.checkpoint")
graph.update_graph(env)
env.update_strategy(graph.current_strategy())
```

The restored environment is already initialized; calling `env.set_graph(graph)`
again resets its state. Fresh-process loading returns a native environment;
OpenSpiel adapter methods and Python environment attributes are not restored.
The graph class must remain importable with compatible code, and the runtime
must be compatible with the saved Python, NumPy, platform, and C++ random
implementation. Checkpoint files use pickle; load only files from trusted sources.
