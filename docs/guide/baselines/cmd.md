---
description: Configure Clairvoyant mirror descent and understand its update schedule, traversal, and evaluation.
---

# Clairvoyant mirror descent

Clairvoyant mirror descent (CMD) runs inner updates before publishing a new
strategy. Its iteration count therefore counts inner updates.

Import the module with `from LiteEFG.baselines import CMD`, then use the
[common baseline interface](../algorithms.md#one-interface-different-update-schedules)
to attach its graph and train.

## Configuration

Construct `CMD.graph(...)` with these options:

| Option | Default | Purpose |
| --- | --- | --- |
| `eta` | `0.1` | Learning rate. |
| `inner_epoch` | `10` | Number of inner updates between strategy copies. |
| `regularizer` | `"Euclidean"` | `"Euclidean"` or `"Entropy"`. |
| `weighted` | `False` | Enable recursive infoset weights. |

## Traversal and evaluation

| Runner setting | Value |
| --- | --- |
| Traversal | `Enumerate` (default), `External`, `Outcome` |
| Evaluation | `last-iterate` |

Every `update_graph(env)` call advances the inner update. Every `inner_epoch`
calls, the result is copied to `bar_u`, which `current_strategy()` returns.
Use `update_graph` to preserve this schedule.

`weighted=True` changes the infoset weights without changing the environment's
traversal mode.

See [running baselines](../algorithms.md#running-baselines) for shared runner
defaults and [examples](../examples.md) for command-line and Python workflows.

[Implementation](https://github.com/liumy2010/LiteEFG/blob/main/LiteEFG/baselines/CMD.py)
