---
description: Configure Implicit Exploration Online Mirror Descent and understand its update schedule, traversal, and evaluation.
---

# Implicit Exploration Online Mirror Descent (IXOMD) {#implicit-exploration-omd}

Implicit exploration online mirror descent (IXOMD) learns from outcome-sampled
trajectories. The implicit-exploration parameter appears in the denominator
of its utility estimator.

Import the module with `from LiteEFG.baselines import IXOMD`, then use the
[common baseline interface](../algorithms.md#one-interface-different-update-schedules)
to attach its graph and train.

## Configuration

Construct `IXOMD.graph(...)` with these options:

| Option | Default | Purpose |
| --- | --- | --- |
| `eta` | `0.001` | Learning rate. |
| `gamma` | `0.0005` | Implicit-exploration parameter in the utility estimator. |

## Traversal and evaluation

| Runner setting | Value |
| --- | --- |
| Traversal | `Outcome` |
| Evaluation | `avg-iterate` |

The runner defaults to 1,000,000 iterations.

See [running baselines](../algorithms.md#running-baselines) for shared runner
defaults and [examples](../examples.md) for command-line and Python workflows.

[Implementation](https://github.com/liumy2010/LiteEFG/blob/main/LiteEFG/baselines/IXOMD.py)
