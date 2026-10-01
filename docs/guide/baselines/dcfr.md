---
description: Configure Discounted Counterfactual Regret Minimization and understand its update schedule, traversal, and evaluation.
---

# Discounted Counterfactual Regret Minimization (DCFR) {#discounted-cfr}

DCFR discounts positive and negative regrets separately and
maintains its own discounted average strategy. The bundled implementation
explicitly updates players 1 and 2 and assumes a two-player game.

Import the module with `from LiteEFG.baselines import DCFR`, then use the
[common baseline interface](../algorithms.md#one-interface-different-update-schedules)
to attach its graph and train.

## Configuration

Construct `DCFR.graph(...)` with these options:

| Option | Default | Purpose |
| --- | --- | --- |
| `alpha` | `1.5` | Exponent for discounting positive regrets. |
| `beta` | `0` | Exponent for discounting negative regrets. |
| `gamma` | `2` | Exponent for discounting the internal average. |

## Traversal and evaluation

| Runner setting | Value |
| --- | --- |
| Traversal | `Enumerate` (default), `External` |
| Evaluation | `last-iterate` of the internal discounted average; see below. |

The regret discount is computed for exponents in `[-10, 10]`, including both
endpoints. Outside this interval, the implementation uses a limiting
approximation: an exponent above 10 gives coefficient 1, and one below -10
gives coefficient 0.

## Select a strategy

`current_strategy()` defaults to `"last-iterate"` and returns the current
`strategy`. Passing `"average-iterate"` returns `avg_strategy`, the internally
maintained discounted average.

The bundled runner selects `"average-iterate"` and evaluates its
`"last-iterate"` snapshot. In a custom loop, evaluate either node directly
with the environment's `"default"` selector. The environment's own average
selector is spelled `"avg-iterate"`; applying it to DCFR's internal average
would average that node again.

See [running baselines](../algorithms.md#running-baselines) for shared runner
defaults and [examples](../examples.md) for command-line and Python workflows.

[Implementation](https://github.com/liumy2010/LiteEFG/blob/main/LiteEFG/baselines/DCFR.py)
