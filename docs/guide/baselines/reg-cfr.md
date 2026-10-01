---
description: Configure Regularized Counterfactual Regret Minimization and understand its update schedule, traversal, and evaluation.
---

# Regularized Counterfactual Regret Minimization (Reg-CFR) {#regularized-cfr}

Reg-CFR combines regret minimization with an adaptive learning rate
and a regularization term whose strength decreases during training.

Import the module with `from LiteEFG.baselines import Reg_CFR`, then use the
[common baseline interface](../algorithms.md#one-interface-different-update-schedules)
to attach its graph and train.

## Configuration

Construct `Reg_CFR.graph(...)` with these options:

| Option | Default | Purpose |
| --- | --- | --- |
| `kappa` | `1.0` | Initialize the adaptive learning rate to `1 / sqrt(kappa)`. |
| `tau` | `0.001` | Regularization strength. |
| `gamma` | `0.001` | Simplex perturbation coefficient. |
| `regularizer` | `"Euclidean"` | `"Euclidean"` or `"Entropy"`. |
| `weighted` | `False` | Enable recursive infoset weights. |
| `shrink_iter` | `100000` | Number of calls between halvings of `tau`. |
| `out_reg` | `False` | Choose how the update applies regularization; see below. |

### Choose how to apply regularization

Let u be utility feedback, π₀ the reference strategy, η the current adaptive step size, ψ the convex regularizer, and D<sub>ψ</sub> its Bregman divergence. Suppressing infoset weights, the maximization over feasible strategies π is:

- `False`: argmax<sub>π</sub> ⟨u − τ∇ψ(π₀), π⟩ − η<sup>−1</sup>D<sub>ψ</sub>(π, π₀).
- `True`: argmax<sub>π</sub> ⟨u, π⟩ − τψ(π) − η<sup>−1</sup>D<sub>ψ</sub>(π, π₀).

Reg-CFR also applies adaptive-step stabilization and the `gamma` simplex perturbation. The command-line interface (CLI) flag `--out-reg` selects `True`.

## Traversal and evaluation

| Runner setting | Value |
| --- | --- |
| Traversal | `Enumerate` (default), `External`, `Outcome` |
| Evaluation | `last-iterate` |

`tau` is halved after the strategy update on calls divisible by `shrink_iter`.
The first call skips this step, even when `shrink_iter=1`. With a sampled
traversal, only the infosets visited on that call have their local `tau` halved.

`weighted=True` changes the infoset weights without changing the environment's
traversal mode. Use `update_graph(env)` to preserve the algorithm's schedule.

See [running baselines](../algorithms.md#running-baselines) for shared runner
defaults and [examples](../examples.md) for command-line and Python workflows.

[Implementation](https://github.com/liumy2010/LiteEFG/blob/main/LiteEFG/baselines/Reg_CFR.py)
