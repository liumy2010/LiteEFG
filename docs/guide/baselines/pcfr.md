---
description: Configure Predictive Counterfactual Regret Minimization+ and understand its update schedule, traversal, and evaluation.
---

# Predictive Counterfactual Regret Minimization+ (PCFR+) {#predictive-cfr}

PCFR+ adds a prediction term when forming the next strategy.
The bundled implementation explicitly updates players 1 and 2 and assumes a
two-player game.

Import the module with `from LiteEFG.baselines import PCFR`, then use the
[common baseline interface](../algorithms.md#one-interface-different-update-schedules)
to attach its graph and train.

## Configuration

`PCFR.graph()` has no algorithm-specific constructor options.

## Traversal and evaluation

| Runner setting | Value |
| --- | --- |
| Traversal | `Enumerate` (default), `External` |
| Evaluation | `linear-avg-iterate` |

See [running baselines](../algorithms.md#running-baselines) for shared runner
defaults and [examples](../examples.md) for command-line and Python workflows.

[Implementation](https://github.com/liumy2010/LiteEFG/blob/main/LiteEFG/baselines/PCFR.py)
