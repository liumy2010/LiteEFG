---
description: Configure Balanced Follow the Regularized Leader and understand its update schedule, traversal, and evaluation.
---

# Balanced Follow the Regularized Leader (Balanced FTRL) {#balanced-ftrl}

Balanced follow the regularized leader uses subtree-based visitation quantities
for outcome sampling. The bundled implementation initializes data for players
1 and 2 and assumes a two-player game.

Import the module with `from LiteEFG.baselines import Balanced_FTRL`, then use the
[common baseline interface](../algorithms.md#one-interface-different-update-schedules)
to attach its graph and train.

## Configuration

Construct `Balanced_FTRL.graph(...)` with these options:

| Option | Default | Purpose |
| --- | --- | --- |
| `eta` | `0.001` | Learning rate. |
| `gamma` | `0.0005` | Implicit-exploration parameter in the utility estimator. |

## Traversal and evaluation

| Runner setting | Value |
| --- | --- |
| Traversal | `Outcome` after initialization |
| Evaluation | `avg-iterate` |

The first update includes an enumerating graph pass to initialize the
subtree-based visitation quantities. The runner defaults to 1,000,000 iterations.

See [running baselines](../algorithms.md#running-baselines) for shared runner
defaults and [examples](../examples.md) for command-line and Python workflows.

[Implementation](https://github.com/liumy2010/LiteEFG/blob/main/LiteEFG/baselines/Balanced_FTRL.py)
