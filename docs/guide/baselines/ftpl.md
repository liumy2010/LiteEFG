---
description: Configure Follow the perturbed leader and understand its update schedule, traversal, and evaluation.
---

# Follow the perturbed leader

Follow the perturbed leader (FTPL) adds random noise to cumulative utility and
selects a pure action using the floating-point argmax.

Import the module with `from LiteEFG.baselines import FTPL`, then use the
[common baseline interface](../algorithms.md#one-interface-different-update-schedules)
to attach its graph and train.

## Configuration

Construct `FTPL.graph(...)` with these options:

| Option | Default | Purpose |
| --- | --- | --- |
| `eta` | `0.01` | Control the noise scale. |
| `noise_type` | `"exponential"` | `"uniform"`, `"normal"`, or `"exponential"`. |

## Traversal and evaluation

| Runner setting | Value |
| --- | --- |
| Traversal | `Enumerate` (default), `External` |
| Evaluation | `avg-iterate` |

| Noise | Distribution |
| --- | --- |
| `"uniform"` | Interval `[0, 1 / eta)`. |
| `"normal"` | Standard deviation `1 / eta`. |
| `"exponential"` | Rate `eta`. |

Use `leg.set_seed(...)` to seed LiteEFG's random generator before training.

See [running baselines](../algorithms.md#running-baselines) for shared runner
defaults and [examples](../examples.md) for command-line and Python workflows.

[Implementation](https://github.com/liumy2010/LiteEFG/blob/main/LiteEFG/baselines/FTPL.py)
