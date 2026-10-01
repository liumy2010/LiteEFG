---
description: Configure Q-function based regret minimization and understand its update schedule, traversal, and evaluation.
---

# Q-function based regret minimization

Q-function based regret minimization (QFR) supports several utility-feedback
estimates. Its update always includes the bidilated regularization term.

Import the module with `from LiteEFG.baselines import QFR`, then use the
[common baseline interface](../algorithms.md#one-interface-different-update-schedules)
to attach its graph and train.

## Configuration

Construct `QFR.graph(...)` with these options:

| Option | Default | Purpose |
| --- | --- | --- |
| `eta` | `0.1` | Learning rate. |
| `tau` | `0.001` | Regularization strength. |
| `gamma` | `0.001` | Simplex perturbation coefficient. |
| `regularizer` | `"Entropy"` | `"Entropy"` or `"Euclidean"`. |
| `feedback` | `"Q"` | Reach weighting and sampling estimator; see [choose feedback](#choose-feedback). |
| `weighted` | `False` | Enable recursive infoset weights. |

## Traversal and evaluation

| Runner setting | Value |
| --- | --- |
| Traversal | Selected by `feedback`; see below. |
| Evaluation | `default` |

### Choose feedback

Let `g` be the action feedback before reach normalization. It combines local
utility, continuation values, and the opponent's bidilated regularization
contribution. Let `p` be this player's `reach_prob`, `r` the
`opponent_reach_prob` (including chance), and `u` the current local strategy.

| `feedback` | Runner traversal | Feedback supplied to the update |
| --- | --- | --- |
| `"Q"` | `Enumerate` | `g / r`: remove chance-and-opponent reach weighting. |
| `"traj-Q"` | `Enumerate` | `p * g`: additionally weight counterfactual feedback by this player's own reach. |
| `"counterfactual"` | `Enumerate` | `g`: retain chance-and-opponent reach weighting without own-reach weighting. |
| `"Outcome"` | `Outcome` | Divide sampled action feedback elementwise by `u`. |

Both strategy updates scale this feedback by `eta / alpha`, where `alpha` is 1
unless `weighted=True` enables recursive infoset weights. Under enumeration,
their regularization divisor is `1 + eta * tau * r / m`, with `m=r` for `Q`,
`m=1/p` for `traj-Q`, and `m=1` for `counterfactual`. Under outcome sampling,
the divisor is `1 + eta * tau`; the continuation value is the sum of the
sampled feedback **before** division or learning-rate scaling, minus the local
regularization term. Thus `feedback` changes both the feedback
scaling and, for the enumerating modes, the local regularization scaling.
The displayed reach formulas use the
[native numerical safeguards](../computation-graph.md#numerical-behavior)
for zero or very small probabilities.

A custom Python loop must choose the environment traversal explicitly.
`weighted=True` enables recursive infoset weights without changing that traversal.

See [running baselines](../algorithms.md#running-baselines) for shared runner
defaults and [examples](../examples.md) for command-line and Python workflows.

[Implementation](https://github.com/liumy2010/LiteEFG/blob/main/LiteEFG/baselines/QFR.py)
