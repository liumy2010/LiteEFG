---
description: Configure Outcome-Sampling Monte Carlo Counterfactual Regret Minimization and understand its update schedule, traversal, and evaluation.
---

# Outcome-Sampling Monte Carlo Counterfactual Regret Minimization (OS-MCCFR) {#outcome-sampling-mccfr}

OS-MCCFR samples one path through the game. It constructs an
exploration strategy and applies the required importance correction. The
bundled implementation explicitly updates players 1 and 2 and assumes a
two-player game.

Import the module with `from LiteEFG.baselines import OS_MCCFR`, then use the
[common baseline interface](../algorithms.md#one-interface-different-update-schedules)
to attach its graph and train.

## Configuration

Construct `OS_MCCFR.graph(...)` with these options:

| Option | Default | Purpose |
| --- | --- | --- |
| `delta` | `0.1` | Exploration rate. |
| `rm_plus` | `False` | Clip the regret buffer at zero. |
| `balanced` | `True` | Use balanced exploration. |

## Traversal and evaluation

| Runner setting | Value |
| --- | --- |
| Traversal | `Outcome` |
| Evaluation | `avg-iterate` |

The runner enables balanced exploration by default. Use `--no-balanced` to
disable it; `--balanced` explicitly enables it.

See [running baselines](../algorithms.md#running-baselines) for shared runner
defaults and [examples](../examples.md) for command-line and Python workflows.

[Implementation](https://github.com/liumy2010/LiteEFG/blob/main/LiteEFG/baselines/OS_MCCFR.py)
