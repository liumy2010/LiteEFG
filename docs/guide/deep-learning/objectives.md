---
description: Define a custom DRLGraph loss, schedule graph colors, and aggregate sampled trajectories.
---

# Custom objectives

Subclass `leg.DRLGraph` to define your own Deep Reinforcement Learning (DRL) loss or algorithm.
The graph describes one sampled decision; the trainer supplies batches and
updates the [model resources](./models.md) selected by each loss.

## Write a custom objective

This example uses the [Proximal Policy Optimization (PPO) objective](./baselines/ppo.md) to demonstrate the
graph interface. Color `0` computes
advantages and targets. Color `1` evaluates the actor and critic losses against
those fixed targets. In this example, both networks read `information_set`.

```python
import LiteEFG as leg
from flax import linen as nn
from LiteEFG.drl import aggregate, clip, entropy, exp, gather, masked_softmax, minimum, stop_gradient

def network(output_size):
    return nn.Sequential([
        nn.Dense(16), nn.tanh, nn.Dense(16), nn.tanh, nn.Dense(output_size)
    ])

class PPO(leg.DRLGraph):
    def __init__(self, policy_network, critic_network, gamma=1.0, gae_lambda=0.95):
        super().__init__()
        self.actor = leg.ModelList(policy_network)
        self.critic = leg.ModelList(critic_network)
        with leg.backward(color=0):
            baseline = self.critic[self.player](self.env.information_set).squeeze(-1)
            reward = aggregate(self.reward, "sum", object="segment", discount=gamma)
            next_value = aggregate(baseline, "sum", object="children", discount=gamma)
            delta = reward + next_value - baseline
            raw_advantage = aggregate(
                delta, "sum", object="descendants", discount=gamma,
                decay=gae_lambda, include_self=True,
            )
            target = stop_gradient(raw_advantage + baseline)
            mean = aggregate(raw_advantage, "mean", object="batch")
            variance = aggregate((raw_advantage - mean) ** 2, "mean", object="batch")
            advantage = stop_gradient((raw_advantage - mean) / (variance + 1e-8) ** 0.5)

        with leg.backward(color=1):
            self.strategy = masked_softmax(
                self.actor[self.player](self.env.information_set), self.legal_action_mask
            )
            self.value = self.critic[self.player](self.env.information_set).squeeze(-1)
            log_prob = clip(gather(self.strategy, self.action), 1e-30, 1.0).log()
            ratio = exp(log_prob - stop_gradient(self.log_sampling_prob))
            policy_loss = -minimum(
                ratio * advantage, clip(ratio, 0.8, 1.2) * advantage
            )
            value_loss = (self.value - target) ** 2
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy(self.strategy)
            self.minimize(loss, models=[*self.actor, *self.critic])

algorithm = PPO(
    [leg.model(network(7)) for _ in range(2)],
    [leg.model(network(1)) for _ in range(2)],
)
```

Bind `algorithm` to `leg.Goofspiel(num_cards=7)`, then use the
[training loop](../deep-learning.md#train-ppo-and-evaluate-two-policies) with
`algorithm.trainer`.

## Schedule the graph with colors

Call `trainer.update(data, upd_color=[0])` once on complete trajectories, then
call `trainer.optimize(batch, upd_color=[1])` for each optimization minibatch.
The first call caches the targets; the second uses the current network
parameters to compute losses against them.

Both `leg.backward(color=...)` and `leg.forward(color=...)` label DRL
expressions; `aggregate` specifies the sampled-path relationship to traverse.
`update` refreshes every node in the selected colors, including ordinary
arithmetic and model calls. Dependencies in other colors must already have
cached values in the supplied data. Missing prerequisites raise an error;
selecting a color never implicitly executes another color. `upd_color=[-1]`
selects all colors. DRL expressions do not support `is_static=True`.

## Inputs and local shapes

The graph inherits `current_strategy()` and `current_value()`, which read `self.strategy`
and `self.value`. Training objectives and metrics must be scalar per sampled
decision. Feature and action reductions describe only the local vector axes;
the runtime adds batch dimensions and masks inactive players and terminal
padding automatically.

The standard feature and runtime sample inputs are:

| Input | Meaning |
| --- | --- |
| `self.env.information_set` | The acting player's information encoded for the model |
| `self.legal_action_mask` | Boolean vector of legal actions |
| `self.action_set_size` | Number of legal actions, derived from the mask |
| `self.action` | Action selected while sampling this decision |
| `self.log_sampling_prob` | Log probability assigned to that action by the sampling policy |
| `self.reward` | The player's reward from the current environment transition |
| `self.player` | Zero-based player index |

### Construct nodes and register losses

Call `super().__init__()` before creating nodes. Operations such as
`leg.const` and `leg.dot` work inside a `DRLGraph` constructor. All node
arguments to an operation must belong to the same graph.

For DRL graphs, `leg.const(size, val)` requires a nonnegative, static integer
`size`. The value is a number or a scalar/length-one DRL node; a node value
determines the owning graph. For example, `leg.const(3, self.reward)` broadcasts
the local reward to three entries. `self.action_set_size` is the current number
of legal actions, computed from `self.legal_action_mask`; it is a runtime value
and cannot be used as a DRL constant's size. Use a fixed Python action count
for action-vector widths and use the legal-action mask to handle unavailable
actions. `self.as_node(value)` also creates a constant with an explicit graph
owner.

Register a scalar loss per decision with `self.minimize(loss, models=[...])`
or `model_handle.minimize(loss)`. Only the selected model parameters receive
gradients from that objective. A model call produces a temporary node;
`node.inplace(expression)` changes a local expression, not model parameters
or persistent infoset storage. Complete the graph before binding it.

## Aggregate sampled trajectories

`leg.aggregate(x, aggregator="sum", object="children", player="self", padding=0.0,
discount=1.0, decay=1.0, include_self=False)` combines values across sampled
decisions or environment steps. The arguments following `padding` are
keyword-only. `object` selects which samples supply values; `aggregator` selects
how to combine them. The operation preserves the local tensor shape of `x`.

| `object` | Values combined at one decision |
| --- | --- |
| `"children"` | The next sampled decision of the same player, discounted once per intervening environment step |
| `"parent"` | The previous sampled decision of the same player, with the same environment-step discount convention |
| `"segment"` | Environment-step values from this decision through the step before the next decision of this player, including rewards on inactive steps |
| `"descendants"` | Future sampled decisions of this player, with `discount` per environment step and `decay` per own-decision transition; `include_self=True` includes the current decision |
| `"batch"` | All valid decisions of this player in the collected batch, reduced and broadcast back to each sample |

`segment` and `descendants` support only summation. The batch
object supports `"sum"`, `"mean"`, `"min"`, and `"max"`, with `"sum"` as the
default. These four reductions are equivalent for the single source selected
by `children` or `parent`. Only `player="self"` is
supported: it selects the player whose decisions are being trained. `padding`
supplies the result when there is no matching source; its default zero gives
terminal value bootstrapping for `children`. With `object="batch"`, padding and
inactive samples never contribute to the reduction.

### Decision boundaries and batch reductions

A `segment` follows one player's decision boundaries: it begins at the current
decision and ends immediately before that player's next decision, or at
termination. Optimization minibatch size does not change these boundaries.

Outcome sampling produces one path per episode, and each path can contain
multiple decisions. `object="batch"` combines this player's valid decisions
across both time and all episodes in the collected batch, before it is split
into optimization minibatches. Repeated visits to an information set contribute
once per visit. The reducers therefore differ even with one sampled path: for
values `[2, 4, 6]`, the sum is 12, mean is 4, minimum is 2, and maximum is 6.
The PPO variance expression uses the mean of squared deviations over these
decision samples.

In a native C++ graph, aggregation follows the game-tree relationships selected
by its traversal. A DRL graph aggregates only the sampled trajectory and the
collected batch. It neither enumerates unsampled branches nor supplies
counterfactual reach weighting. In the PPO example, `segment` collects rewards,
`children` supplies the next value estimate, and `descendants` combines Temporal Difference (TD) errors
into Generalized Advantage Estimation (GAE). Batch aggregates then normalize the advantages.

### Cache aggregates before optimization

Call `trainer.update(data, upd_color=[0])` on complete trajectories before
shuffling to evaluate the aggregates in color 0. It uses the current model
parameters and caches all nodes in that color. Aggregate results are frozen and
have no gradient through their source expressions. Nested aggregates are
evaluated in dependency order; their cached values accompany samples through
every minibatch and epoch. Direct `graph.evaluate()` calls can evaluate ordinary
local nodes; trajectory aggregates require the complete trajectory context
supplied by `update`.

See [PPO's targets and update schedule](./baselines/ppo.md#targets-and-update-schedule)
for the meaning of the example's GAE expressions, and [training](./training.md)
to configure collection and minibatching.
