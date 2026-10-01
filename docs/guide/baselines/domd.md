---
description: Configure Dilated optimistic mirror descent and understand its update schedule, traversal, and evaluation.
---

# Dilated optimistic mirror descent

Dilated optimistic mirror descent (DOMD) updates the strategy using a
regularized mirror-descent rule. Choose the step size and regularizer when
constructing the graph.

Import the module with `from LiteEFG.baselines import DOMD`, then use the
[common baseline interface](../algorithms.md#one-interface-different-update-schedules)
to attach its graph and train.

## Configuration

Construct `DOMD.graph(...)` with these options:

| Option | Default | Purpose |
| --- | --- | --- |
| `eta` | `0.1` | Learning rate. |
| `regularizer` | `"Euclidean"` | `"Euclidean"` or `"Entropy"`. |
| `weighted` | `False` | Enable recursive infoset weights. |

## Traversal and evaluation

| Runner setting | Value |
| --- | --- |
| Traversal | `Enumerate` (default), `External`, `Outcome` |
| Evaluation | `last-iterate` |

`weighted=True` changes the infoset weights without changing the environment's
traversal mode.

See [running baselines](../algorithms.md#running-baselines) for shared runner
defaults and [examples](../examples.md) for command-line and Python workflows.

[Implementation](https://github.com/liumy2010/LiteEFG/blob/main/LiteEFG/baselines/DOMD.py)
