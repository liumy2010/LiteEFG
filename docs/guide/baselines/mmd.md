---
description: Configure Magnetic mirror descent and understand its update schedule, traversal, and evaluation.
---

# Magnetic mirror descent

Magnetic mirror descent (MMD) uses a regularized mirror-descent update.
Its feedback setting determines the utility estimate and the traversal used
by the command-line runner.

Import the module with `from LiteEFG.baselines import MMD`, then use the
[common baseline interface](../algorithms.md#one-interface-different-update-schedules)
to attach its graph and train.

## Configuration

Construct `MMD.graph(...)` with these options:

| Option | Default | Purpose |
| --- | --- | --- |
| `eta` | `0.005` | Learning rate. |
| `tau` | `0.05` | Regularization strength. |
| `gamma` | `0.0` | Simplex perturbation coefficient. |
| `regularizer` | `"Entropy"` | `"Entropy"` or `"Euclidean"`. |
| `feedback` | `"Q"` | Reach weighting and sampling estimator; see [choose feedback](#choose-feedback). |
| `weighted` | `False` | Enable recursive infoset weights. |

## Traversal and evaluation

| Runner setting | Value |
| --- | --- |
| Traversal | Selected by `feedback`; see below. |
| Evaluation | `default` |

### Choose feedback

Let `g` be the action feedback obtained by adding the local utility contribution
to the aggregated continuation values. Under enumeration, it already includes
chance-and-opponent reach weighting. Let `p` be this player's `reach_prob`,
`r` the `opponent_reach_prob` (including chance), and `u` the current local
strategy. The modes differ as follows:

| `feedback` | Runner traversal | Feedback supplied to the update |
| --- | --- | --- |
| `"Q"` | `Enumerate` | `g / r`: remove chance-and-opponent reach weighting to obtain conditional action values. |
| `"traj-Q"` | `Enumerate` | `p * g`: additionally weight counterfactual feedback by this player's own reach. |
| `"counterfactual"` | `Enumerate` | `g`: retain chance-and-opponent reach weighting without own-reach weighting. |
| `"Outcome"` | `Outcome` | Divide sampled action feedback elementwise by `u`. |

The update scales this feedback by `eta / alpha`, where `alpha` is 1 unless
`weighted=True` enables recursive infoset weights. Its regularization divisor
is `1 + eta * tau` in every mode. For `Outcome`, the continuation value is the
sum of the sampled feedback **before** division by `u` or learning-rate scaling.
The displayed reach formulas use the
[native numerical safeguards](../computation-graph.md#numerical-behavior)
for zero or very small probabilities.

A custom Python loop must choose the environment traversal explicitly.
`weighted=True` enables recursive infoset weights without changing that traversal.

See [running baselines](../algorithms.md#running-baselines) for shared runner
defaults and [examples](../examples.md) for command-line and Python workflows.

[Implementation](https://github.com/liumy2010/LiteEFG/blob/main/LiteEFG/baselines/MMD.py)
