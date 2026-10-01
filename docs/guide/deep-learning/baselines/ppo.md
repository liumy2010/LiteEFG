---
description: Configure Proximal Policy Optimization networks, targets, training, and metrics.
---

# Proximal Policy Optimization (PPO) {#ppo}

PPO alternates between collecting games and making several optimization passes
over them. An **actor** chooses actions, while a **critic** estimates future
rewards. The policy loss clips the probability ratio relative to the policy
that collected the games. The total loss also includes the critic's squared
error and an entropy bonus.

## Networks and parameter sharing

Construct `PPO.graph(policy_network, critic_network, num_actions=env.num_actions)`
with an actor that returns one logit per action and a critic that returns a
length-one value vector. Each network argument accepts a model resource, a
list or tuple of resources, or a `ModelList`.

| Argument | Parameter sharing |
| --- | --- |
| One resource | Every player uses the same parameters. |
| Distinct resources in a sequence | Each player has independent parameters. |
| A sequence that repeats a resource | Players selecting that resource share parameters. |

For independent actors and a shared critic, pass a list of actor resources and
one critic resource. Set learning rates and gradient clipping in the resources'
[Optax optimizers](../models.md#wrap-a-network).

## Constructor options

| Option | Default | Purpose |
| --- | --- | --- |
| `num_actions` | `7` | Actor output width; pass the game's action count. |
| `clip_epsilon` | `0.2` | Allowed deviation of the probability ratio from 1 before clipping. |
| `value_coef` | `0.5` | Weight of the critic's squared error. |
| `entropy_coef` | `0.01` | Weight of the policy entropy bonus. |
| `gamma` | `1.0` | Reward discount per environment step. |
| `gae_lambda` | `0.95` | Generalized Advantage Estimation (GAE) decay per transition between this player's decisions. |
| `normalize_advantage` | `True` | Normalize advantages over the collected batch. |
| `critic_feature` | `"full_info"` | Environment feature supplied to the critic. |

## Actor and critic inputs

The actor reads `self.env.information_set`. The critic reads `self.env.full_info`
by default; pass `critic_feature="information_set"` to use the actor's observation
vector instead. The environment must supply the critic's named feature through
`features(state)`, with one vector per player.

This allows the critic to see privileged state while the actor uses only the
player's information set. Inference through `Policy` needs only the actor's
features, so it does not require the critic's `full_info` vector.

## Targets and update schedule

PPO uses two graph colors:

| Color | When it runs | What it computes |
| --- | --- | --- |
| `0` | Once on each complete collection, before minibatching. | GAE advantages, advantage normalization, and critic targets. |
| `1` | On each optimization minibatch. | Actor and critic losses against the cached targets. |

Targets and recorded sampling probabilities stay fixed across all optimization
epochs on that collection. The [quick start](../../deep-learning.md#read-the-training-loop)
shows the explicit trainer calls for this schedule.

GAE follows successive decisions of the same
player. Rewards and gamma discounts accumulate across inactive environment
steps; lambda applies once per own-decision transition. Terminal trajectories
use zero bootstrap.

A shared critic that predicts the current mover's return uses a positive
bootstrap at the next decision of the same player. It does not negate the next
own-turn value or bootstrap from the opponent's intervening turn.

## Run the training helper

After binding the graph, call
`graph.train(iterations=1, epochs=4, minibatch_size=128, microbatch_size=-1,
microbatch_unit="trajectories", compiled_updates=True)`.

Each iteration collects fresh games, computes color-0 targets once, and
optimizes color 1 for the requested epochs. `iterations`, `epochs`, and
`minibatch_size` must be positive integers. Each call adds `iterations` new
collections; its epoch and minibatch settings apply to that call.

By default, the epochs run in a compiled JAX loop. Set `compiled_updates=False`
to use the Python dataloader and individual `optimize` calls. See
[training](../training.md) for batch sizes and
[parallelism and memory](../scaling.md) for microbatching either schedule.

## Training metrics

In addition to the [trainer's standard metrics](../training.md#read-the-metrics),
PPO reports `policy_loss`, `value_loss`, `entropy`, `approx_kl`, and
`clip_fraction`.

`train` returns loss and graph metrics averaged over all minibatches in the
call, weighted by each player's valid decision count. It sums valid decision
counts, averages utilities across the call's collections, and reports the
trainer's final cumulative counters.

## Examples and source

- [Quick start](../../deep-learning.md): train PPO on Goofspiel and evaluate policies.
- [Dark chess](../dark-chess.md): choose actor and critic inputs for a larger game.
- [PPO implementation](https://github.com/liumy2010/LiteEFG/blob/main/LiteEFG/baselines/drl/PPO.py).
