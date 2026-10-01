---
description: Configure Balanced Online Mirror Descent and understand its update schedule, traversal, and evaluation.
---

# Balanced Online Mirror Descent (Balanced OMD) {#balanced-omd}

Balanced online mirror descent uses exploration weights based on the game
structure. The bundled implementation initializes data for players 1 and 2
and assumes a two-player game.

Import the module with `from LiteEFG.baselines import Balanced_OMD`, then use the
[common baseline interface](../algorithms.md#one-interface-different-update-schedules)
to attach its graph and train.

## Configuration

Construct `Balanced_OMD.graph(...)` with these options:

| Option | Default | Purpose |
| --- | --- | --- |
| `eta` | `0.001` | Learning rate. |
| `gamma` | `0.0005` | Implicit-exploration parameter in the utility estimator. |

## Traversal and evaluation

| Runner setting | Value |
| --- | --- |
| Traversal | `Outcome` after initialization |
| Evaluation | `avg-iterate` |

Initialization computes exploration weights separately for each infoset depth,
using one enumerating pass per player and target depth. Subsequent updates use
the environment's sampling traversal.

The runner defaults to 1,000,000 iterations.

See [running baselines](../algorithms.md#running-baselines) for shared runner
defaults and [examples](../examples.md) for command-line and Python workflows.

[Implementation](https://github.com/liumy2010/LiteEFG/blob/main/LiteEFG/baselines/Balanced_OMD.py)
