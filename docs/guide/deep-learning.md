---
description: Train neural policies with JAX, follow a Proximal Policy Optimization example, and choose the next guide.
---

# Deep learning with JAX

Use Deep Reinforcement Learning (DRL) to train neural policies by sampling games and updating networks with JAX. Define
a learning rule with `DRLGraph`, or start from a bundled [DRL baseline](./deep-learning/baselines.md).

This quick start uses Proximal Policy Optimization (PPO) on seven-card Goofspiel. Each player has an **actor**
that chooses actions and a **critic** that estimates future rewards.

Complete the [installation steps](./installation.md#deep-learning-with-jax)
before running the example. Central processing unit (CPU) execution is enough for this small run.

## Train PPO and evaluate two policies

The script creates a game, wraps Flax networks with `leg.model`, and passes them
to `PPO.graph`. Binding the graph to the game creates a trainer. The loop then
collects games, prepares targets, and updates the networks.

Each player gets its own actor and critic. `batch_size=16` collects 16 complete
games; the dataloader groups them into minibatches of eight games.

```python
import jax
import optax
import LiteEFG as leg
from flax import linen as nn
from LiteEFG.baselines.drl import PPO

def network(output_size):
    return nn.Sequential([
        nn.Dense(16), nn.tanh, nn.Dense(16), nn.tanh, nn.Dense(output_size)
    ])

def resource(output_size):
    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(3e-4))
    return leg.model(network(output_size), optimizer=optimizer)

env = leg.Goofspiel(num_cards=7)
policy_network = [resource(env.num_actions) for _ in range(env.num_players)]
critic_network = [resource(1) for _ in range(env.num_players)]
algorithm = PPO.graph(
    policy_network, critic_network, num_actions=env.num_actions,
    gamma=1.0, gae_lambda=0.95, normalize_advantage=True,
)
algorithm.bind(env, batch_size=16, seed=0)
trainer = algorithm.trainer
for iteration in range(2):
    data = trainer.collect()
    data = trainer.update(data, upd_color=[0])
    trainer.key, key = jax.random.split(trainer.key)
    dataloader = leg.dataloader(data, batch_size=8, shuffle=True, generator=key)
    for epoch in range(1):
        for batch in dataloader:
            metrics = trainer.optimize(batch, upd_color=[1])

policy0 = trainer.policy(0)
policy1 = trainer.policy(1)
match = leg.average_utility(
    env, policy0, policy1, episodes=64, batch_size=32, seed=0
)
print(metrics["iteration"], match["mean_utility"])
print(match["confidence_interval_95"])
```

The script prints the collection count, each player's mean utility, and a 95%
confidence interval from the sampled match. Two training iterations are enough
to check the workflow; use more training and evaluation games for experiments.

## Read the training loop

| Step | What happens |
| --- | --- |
| `trainer.collect()` | Play complete games using the current policies. |
| `trainer.update(data, upd_color=[0])` | Compute advantages and critic targets once for those games. |
| `leg.dataloader(...)` | Group complete trajectories into optimization minibatches. |
| `trainer.optimize(batch, upd_color=[1])` | Update the networks using the cached targets. |
| `trainer.policy(player)` | Take a frozen snapshot for evaluation. |

In PPO, colors `0` and `1` label target computation and loss computation.
Targets stay fixed while the networks are updated. Both players train on the
same selected games, with independent parameters in this example.

PPO's [training helper](./deep-learning/baselines/ppo.md#run-the-training-helper)
can also run this schedule for you:
`algorithm.train(iterations=2, epochs=1, minibatch_size=8)`.
Use the explicit loop when you need control over individual stages. See
[training](./deep-learning/training.md) for batching, epochs, and metrics.

`average_utility` measures the two policies playing against each other. It
does not compute exploitability. See [policies and evaluation](./deep-learning/policies.md)
to interpret the results or save the policies.

## Choose the next step

| I want to… | Read |
| --- | --- |
| Choose or configure a baseline | [DRL baselines](./deep-learning/baselines.md) |
| Change the networks or optimizer | [Models and optimizers](./deep-learning/models.md) |
| Control collection, minibatches, and epochs | [Training](./deep-learning/training.md) |
| Save policies or compare their performance | [Policies and evaluation](./deep-learning/policies.md) |
| Define another loss or algorithm | [Custom objectives](./deep-learning/objectives.md) |
| Use several devices or reduce training memory | [Parallelism and memory](./deep-learning/scaling.md) |
| Choose a game | [Goofspiel](./deep-learning/goofspiel.md) or [Dark chess](./deep-learning/dark-chess.md) |
| Supply another game or model input | [Custom environments](./deep-learning/environments.md) |
