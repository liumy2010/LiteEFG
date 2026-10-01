---
description: Run Counterfactual Regret Minimization on Kuhn poker, measure exploitability, and export a policy.
---

# Quick start

Solve Kuhn poker with Counterfactual Regret Minimization (CFR) in a short training loop.
This is the tabular workflow. For neural policies, start with
[Deep learning with JAX](./deep-learning.md).

Complete [installation](./installation.md), then save the following as `quick_start.py` outside the repository if you installed the package non-editably.

## Run CFR on Kuhn poker

```python
import LiteEFG as leg
import pyspiel
from LiteEFG.baselines import CFR

# Create the environment before constructing the computation graph.
env = leg.OpenSpielEnv(
    pyspiel.load_game("kuhn_poker"),
    traverse_type="Enumerate",
)
algorithm = CFR.graph()
env.set_graph(algorithm)
strategy = algorithm.current_strategy()

for iteration in range(1, 1001):
    algorithm.update_graph(env)
    env.update_strategy(strategy)

    if iteration % 200 == 0:
        gaps = env.exploitability(strategy, "avg-iterate")
        print(iteration, "per-player exploitability:", gaps, "Exploitability:", sum(gaps))

policy, tables = env.get_strategy(strategy, "avg-iterate")
print(tables[0].head())
```

```sh
python quick_start.py
```

The loop prints each player’s best-response improvement every 200 iterations
and a few strategy rows. The average strategy’s total gap should become small
with further training, though it need not decrease at every report.

::: tip Reading the metric
`env.exploitability(...)` returns a **list**, one best-response improvement per player. LiteEFG reports `sum(gaps)` as **Exploitability**. Do not equate that sum with a differently normalized scalar metric from another library. See [evaluation](environments.md#api-reference).
:::

## What the loop does

| Step | Purpose |
| --- | --- |
| `OpenSpielEnv(...)` | Converts or loads the game and selects a traversal mode. |
| `CFR.graph()` | Builds the algorithm’s local graph, including strategy and regret buffers. |
| `env.set_graph(algorithm)` | Attaches the graph to the environment and executes its static initialization. |
| `algorithm.update_graph(env)` | Calls the baseline’s update procedure; for CFR this is `env.update(self.strategy)`. |
| `env.update_strategy(strategy)` | Records the current sequence-form strategy and updates its stored averages. |
| `env.exploitability(strategy, "avg-iterate")` | Evaluates the average of the strategies recorded so far. |

`strategy` is a graph-node handle. Its values live separately at each information set and change as the graph executes, so you do not need to fetch a new handle each iteration.

Recording strategies is a separate operation from updating the graph. Omitting `update_strategy` means there is no stored average to evaluate. It also does not track a best iterate by default: use `update_best=True` when you need that history mode.

## Inspect or export the result

Continue in the same script:

```python
# The OpenSpiel wrapper returns a TabularPolicy and one DataFrame per player.
policy, tables = env.get_strategy(strategy, "avg-iterate")
tables[0].to_csv("kuhn_player_0.csv", index=False)
tables[1].to_csv("kuhn_player_1.csv", index=False)

# Native environment access uses player IDs starting at 1.
regrets = env.get_value(1, algorithm.regret_buffer)
print(regrets[:2])
```

The DataFrame list uses normal zero-based Python indexing; the native `player` argument uses IDs starting at 1. `OpenSpielEnv.get_strategy` returns all players together and has a different signature from `FileEnv.get_strategy`.

To play against the exported policy interactively, use `env.interact(policy, controlled_player=0, epochs=1)`. This method requests input in your terminal.

## Choose the next step

- [Core concepts](./concepts.md) explains information sets, local values, and strategy averaging.
- [Computation graph](./computation-graph.md) walks through the actual CFR update rule.
- [Algorithms](./algorithms.md) compares the implementations and their traversal choices.
- [Examples](./examples.md) shows the checked-in runners and other experiment patterns.
