---
description: Choose and configure the bundled deep reinforcement learning baselines.
---

# Deep Reinforcement Learning (DRL) baselines {#drl-baselines}

DRL baselines define learning rules with `DRLGraph` and train neural policies
from sampled games. Import them from `LiteEFG.baselines.drl`, supply
[model resources](./models.md), and bind the graph to an environment. For
algorithms that store a strategy at each information set, see
[tabular baselines](../algorithms.md).

| Baseline | Module | Learning rule |
| --- | --- | --- |
| [Proximal Policy Optimization (PPO)](./baselines/ppo.md) | `PPO` | Clipped policy objective with a learned critic and entropy bonus. |

The bundled DRL baseline is currently PPO. The [quick start](../deep-learning.md)
provides a complete example; [custom objectives](./objectives.md) explains how
to implement another algorithm using the same graph and trainer interfaces.

## Algorithm references

See the [DRL baseline README](https://github.com/liumy2010/LiteEFG/blob/main/LiteEFG/baselines/drl/README.md#references)
for the papers associated with these algorithms.
