---
description: Implement a JAX game or add named observation and critic features to an existing environment.
---

# Custom environments

To add a game, implement the environment contract below. To add an input to an
existing game, override `features(state)`, as in the Goofspiel example on this
page. Start with [Goofspiel](./goofspiel.md) or [Dark chess](./dark-chess.md) if
either already fits your experiment.

## Implement a game

A custom environment subclasses `LiteEFG.drl.Environment` and implements
integer `num_players`, `num_actions`, and `max_steps` properties, plus these
pure JAX methods. Feature dimensions are inferred from `features(state)`:

| Method | Result for one game |
| --- | --- |
| `init(key)` | Fixed-shape state pytree |
| `features(state)` | Dictionary containing `information_set: [players, vector_size]` and any additional named arrays `[players, *local_shape]` |
| `legal_action_mask(state)` | Boolean legal actions `[players, actions]` |
| `active_players(state)` | Boolean decision mask `[players]` |
| `step(state, actions, key)` | `(next_state, reward[players], done)` |

The runtime supplies one action per player; `step` ignores inactive players.
Terminal states must be absorbing. State and feature shapes remain fixed
throughout sampling. `information_set` must contain only the player's
information, including the visible history needed by the policy. The
environment implementation must itself support JAX tracing for device-side
sampling. `average_utility` currently accepts two-player environments.

## Define model inputs

Every environment implements `features(state)` to return a dictionary of named
JAX arrays, including the required `"information_set"` vector for each player.
Read these inputs through `self.env` while constructing a graph:
`self.env.information_set` or `self.env.full_info_feature`. `Trainer` checks
that every referenced feature exists and binds its local shape and dtype.

Feature names must be public Python identifiers and must not collide
with runtime trajectory fields. The `self.env` namespace is read-only and
separate from graph attributes: for example, `self.env.critic` can supply an
environment feature while `self.critic` holds a model.

The `information_set` array has shape `[num_players, vector_size]`, with a
positive vector size. It encodes only information visible to that player;
states in the same information set must give that player the same vector.

Other arrays have shape `[num_players, *local_shape]` for one game. A per-player
scalar has shape `[num_players]`; vectors and matrices are supported as well.

Names, shapes, and dtypes remain fixed throughout an episode. Features describe
the state before each action. The graph sees only `local_shape`, with the
player and batch axes supplied by the runtime.

## Add a critic feature to Goofspiel

This example supplies current scores and remaining hands to an external Flax
critic. The actor uses the player's `information_set` vector.
The local actor-critic objective constructs a Monte Carlo return target with
trajectory aggregates and subtracts a pre-update critic estimate to obtain its
advantage:

```python
import jax.numpy as jnp
import LiteEFG as leg
from flax import linen as nn
from LiteEFG.drl import aggregate, clip, entropy, gather, masked_softmax, stop_gradient

class FeatureGoofspiel(leg.Goofspiel):
    def features(self, state):
        features = super().features(state)
        full = jnp.concatenate((
            state.scores, state.hands.astype(jnp.float32).reshape(-1)
        ))
        features["full_info_feature"] = jnp.broadcast_to(full, (self.num_players, full.size))
        return features

class FeatureActorCritic(leg.DRLGraph):
    def __init__(self, policy_network, critic_network):
        super().__init__()
        self.actor = leg.ModelList(policy_network)
        self.critic = leg.ModelList(critic_network)
        with leg.backward(color=0):
            baseline = self.critic[self.player](self.env.full_info_feature).squeeze(-1)
            reward = aggregate(self.reward, "sum", object="segment")
            target = aggregate(reward, "sum", object="descendants", include_self=True)
            advantage = stop_gradient(target - baseline)
        with leg.backward(color=1):
            self.strategy = masked_softmax(
                self.actor[self.player](self.env.information_set), self.legal_action_mask
            )
            self.value = self.critic[self.player](self.env.full_info_feature).squeeze(-1)
            log_prob = clip(gather(self.strategy, self.action), 1e-30, 1.0).log()
            policy_loss = -log_prob * advantage
            value_loss = (self.value - stop_gradient(target)) ** 2
            self.minimize(
                policy_loss + 0.5 * value_loss - 0.1 * entropy(self.strategy),
                models=[*self.actor, *self.critic],
            )

feature_env = FeatureGoofspiel(num_cards=4, imp_info=True)
feature_graph = FeatureActorCritic(
    [leg.model(nn.Dense(4)) for _ in range(feature_env.num_players)],
    [leg.model(nn.Dense(1)) for _ in range(feature_env.num_players)],
)
feature_graph.bind(feature_env, batch_size=8, seed=0)
feature_trainer = feature_graph.trainer
feature_data = feature_trainer.collect()
feature_data = feature_trainer.update(feature_data, upd_color=[0])
for feature_batch in leg.dataloader(feature_data, batch_size=32):
    feature_metrics = feature_trainer.optimize(feature_batch, upd_color=[1])
feature_match = leg.average_utility(
    feature_env, feature_trainer.policy(0), feature_trainer.policy(1),
    episodes=32, batch_size=16, seed=0,
)
print(feature_metrics["iteration"], feature_match["mean_utility"])
```

In the private-bid game, the critic's feature includes the opponent's remaining
hand. Keep such privileged inputs out of the actor when the policy must respect
the player's information. Environment state can also contain unrevealed chance
draws; the example's feature includes only current scores and hands.

## Use features during inference

Pass the features together when calling a policy directly:
`policy({"information_set": vectors, "full_info_feature": values}, masks, player=0)`.
Their leading batch dimensions must match. A policy call requires
`information_set` and any other features used by its actor; the example's
critic-only feature need not be supplied for direct action inference.
`average_utility` obtains the dictionary from the environment automatically
for `Policy` instances. External policy callables accept
`(information_set, legal_action_mask)` arrays.

For convenience, a direct `Policy` call also accepts the information-set array
as its first argument: `policy(vectors, masks, features={"full_info_feature": values})`.
The additional dictionary must not contain `information_set`, since the first
argument already supplies it.
