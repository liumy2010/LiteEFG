---
description: Configure Counterfactual Regret Minimization and understand its update schedule, traversal, and evaluation.
---

# Counterfactual Regret Minimization (CFR) {#cfr}

Counterfactual regret minimization accumulates counterfactual regrets and
normalizes their positive part into a strategy. The same graph supports
external-sampling Monte Carlo Counterfactual Regret Minimization (MCCFR) with the `External` traversal.

Import the module with `from LiteEFG.baselines import CFR`, then use the
[common baseline interface](../algorithms.md#one-interface-different-update-schedules)
to attach its graph and train.

## Configuration

`CFR.graph()` has no algorithm-specific constructor options.

## Traversal and evaluation

| Runner setting | Value |
| --- | --- |
| Traversal | `Enumerate` (default), `External` |
| Evaluation | `avg-iterate` |

For outcome sampling, use [Outcome-Sampling MCCFR (OS-MCCFR)](./os-mccfr.md). Switching this graph to
`Outcome` does not provide the required importance-corrected estimator.

See [running baselines](../algorithms.md#running-baselines) for shared runner
defaults and [examples](../examples.md) for command-line and Python workflows.

[Implementation](https://github.com/liumy2010/LiteEFG/blob/main/LiteEFG/baselines/CFR.py)
