---
description: Freeze and save policies, evaluate matchups, and compute exact exploitability for small Goofspiel games.
---

# Policies and evaluation

Use policies to evaluate trained networks or save them for later use. The
[Proximal Policy Optimization (PPO) example](../deep-learning.md) creates two snapshots and evaluates a match.

## Take a snapshot

`trainer.policy(player)` freezes a zero-based player's current parameters.
Further training leaves that snapshot unchanged. A policy keeps its selected
network when evaluated in another seat.

## Evaluate a matchup

`average_utility(env, policy0, policy1, ...)` places the first policy in seat 0
and the second in seat 1. The per-player results follow that seat order:

| Result | Meaning |
| --- | --- |
| `mean_utility` | Average game reward, without the PPO entropy bonus. |
| `standard_error` | Estimated standard error of that average. |
| `confidence_interval_95` | Mean plus or minus 1.96 standard errors. |
| `win_rate` | Fraction of sampled games won. |

The report also includes episode count, seed, and tie rate. The confidence
interval uses a normal approximation; a small sample can give an imprecise
estimate. This measures the specified matchup, not exploitability or a best
response.

## Call a policy directly

A `Policy` accepts a feature dictionary and a legal-action mask, such as
`policy({"information_set": vectors}, masks)`, and returns legal action
probabilities with shape `[batch, actions]`. External policy callables accept
batched `(information_set, legal_action_mask)` arrays and must support JAX
tracing. `leg.uniform_policy` supplies a uniform distribution over legal
actions and can replace either argument in a matchup.

A graph can read its actual zero-based playing seat through `self.player`.
`Trainer` and `average_utility` supply that seat automatically. When calling a
`Policy` directly, pass `policy(features, legal_action_mask, player=1)` for
seat 1. Without `player`, the call uses the seat saved in the snapshot.
Changing the playing seat does not change the snapshot's selected model.

## Save and load a policy

Run the [PPO example](../deep-learning.md#train-ppo-and-evaluate-two-policies)
first, then continue in the same Python session. This uses its `resource`
factory, `env`, and policy snapshots:

```python
policy0.save("goofspiel-player0.npz")
policy1.save("goofspiel-player1.npz")

loaded0 = leg.Policy.load(
    PPO.graph([resource(7) for _ in range(2)], [resource(1) for _ in range(2)],
              num_actions=7), "goofspiel-player0.npz"
)
loaded1 = leg.Policy.load(
    PPO.graph([resource(7) for _ in range(2)], [resource(1) for _ in range(2)],
              num_actions=7), "goofspiel-player1.npz"
)
loaded_match = leg.average_utility(
    env, loaded0, loaded1, episodes=64, batch_size=32, seed=0
)
print(loaded_match["mean_utility"])
```

Supply the same graph and model architecture when loading. The checkpoint
preserves model parameters, feature shapes and dtypes, the default seat, and
the selected model. `Policy.load` returns a frozen snapshot for inference and
evaluation; it does not restore training state or require an environment.

Policy files retain every registered model's parameters, including the critic,
even though action inference evaluates only the actor.

## Exact evaluation for small games

Use `average_utility` to estimate returns by sampling two policies defined by
external models. For a small Goofspiel game, the
[exact evaluation example](https://github.com/liumy2010/LiteEFG/blob/main/examples/goofspiel_exact.py)
also evaluates their full strategy profile with the native `FileEnv` evaluator.
It preserves the neural policies' complete visible history and supports the
public-bid game with random prizes and `point_difference` rewards.

Create four-card checkpoints with
`python examples/goofspiel_ppo.py --cards 4 --output artifacts/drl/goofspiel-4-ppo.json`.
Then run
`python examples/goofspiel_exact.py --cards 4 --player0 artifacts/drl/goofspiel-4-ppo-player0.npz --player1 artifacts/drl/goofspiel-4-ppo-player1.npz --output artifacts/drl/goofspiel-4-exact.json`.
The evaluator uses the training example's external Flax network architecture;
pass the same `--hidden-size` to both scripts when overriding its default of 64.
Run `python examples/goofspiel_exact.py --help` for the batch and budget options.

The JavaScript Object Notation (JSON) report includes each seat's exact `utility`, `best_response_utility`,
and `deviation_gain`. `nash_conv` is the sum of the two deviation gains, and
`exploitability` is half that sum. The sampled confidence interval applies only
to matchup utility. The script checks
the required tree size against `--max-nodes` before construction and rejects
games that exceed the budget, including seven-card Goofspiel at the default
budget of 1,000,000 nodes.
