---
description: Configure Regularized Dilated Optimistic Mirror Descent and understand its update schedule, traversal, and evaluation.
---

# Regularized Dilated Optimistic Mirror Descent (Reg-DOMD) {#regularized-domd}

Reg-DOMD adds a regularization term to the optimistic mirror-descent
update and periodically reduces its strength.

Import the module with `from LiteEFG.baselines import Reg_DOMD`, then use the
[common baseline interface](../algorithms.md#one-interface-different-update-schedules)
to attach its graph and train.

## Configuration

Construct `Reg_DOMD.graph(...)` with these options:

| Option | Default | Purpose |
| --- | --- | --- |
| `eta` | `0.1` | Learning rate. |
| `tau` | `0.001` | Regularization strength. |
| `regularizer` | `"Euclidean"` | `"Euclidean"` or `"Entropy"`. |
| `weighted` | `False` | Enable recursive infoset weights. |
| `shrink_iter` | `100000` | Number of calls between halvings of `tau`. |
| `out_reg` | `False` | Choose how the update applies regularization; see below. |

### Choose how to apply regularization

Let u be utility feedback, π₀ the reference strategy, η the step size, ψ the convex regularizer, and D<sub>ψ</sub> its Bregman divergence. Suppressing infoset weights, the maximization over feasible strategies π is:

- `False`: argmax<sub>π</sub> ⟨u − τ∇ψ(π₀), π⟩ − η<sup>−1</sup>D<sub>ψ</sub>(π, π₀).
- `True`: argmax<sub>π</sub> ⟨u, π⟩ − τψ(π) − η<sup>−1</sup>D<sub>ψ</sub>(π, π₀).

The command-line interface (CLI) flag `--out-reg` selects `True`.

## Traversal and evaluation

| Runner setting | Value |
| --- | --- |
| Traversal | `Enumerate` (default), `External`, `Outcome` |
| Evaluation | `last-iterate` |

`tau` is halved after the strategy update on calls divisible by `shrink_iter`.
With a sampled traversal, only the infosets visited on that call have their
local `tau` halved.

`weighted=True` changes the infoset weights without changing the environment's
traversal mode.

See [running baselines](../algorithms.md#running-baselines) for shared runner
defaults and [examples](../examples.md) for command-line and Python workflows.

[Implementation](https://github.com/liumy2010/LiteEFG/blob/main/LiteEFG/baselines/Reg_DOMD.py)
