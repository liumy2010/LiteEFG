---
description: Configure Goofspiel rules, visible information, rewards, and the Proximal Policy Optimization example.
---

# Goofspiel

Use `leg.Goofspiel` for the [Proximal Policy Optimization (PPO) quick start](../deep-learning.md). The game has
two players who bid cards for public prizes.

## Rules and defaults

`Goofspiel()` defaults to `num_cards=7`, `imp_info=False`,
`points_order="random"`, and `returns_type="point_difference"`. Each round,
both players bid an unused card for the current public prize. The higher bid
wins the prize; tied bids discard it. An action is the bid value minus one.
The final forced bid remains an explicit round. Rewards are zero until the
terminal round, when each player receives its score minus the players' mean
score: half the score difference in this two-player game.

## What each network can observe

The `information_set` vectors preserve the ordered visible history and hide future prize
draws and the opponent's current simultaneous bid. With `imp_info=False`,
completed bids and both remaining hands are visible. With `imp_info=True`,
the opponent's bids and remaining hand are hidden; own bids, prizes, winners,
and scores remain visible. `points_order` also accepts `"ascending"` and
`"descending"`.

`Goofspiel.features(state)` also supplies a `full_info` vector containing the
complete state, including the entire prize order with future prizes, both
remaining hands, completed bids, winners, scores, and the current round. PPO
uses this vector for its critic when configured with `critic_feature="full_info"`.
The actor continues to read only `information_set`, which excludes hidden
information and future prizes.

## Run an experiment

The executable [Goofspiel PPO example](https://github.com/liumy2010/LiteEFG/blob/main/examples/goofspiel_ppo.py)
provides configurable training and evaluation batches, a `--require-gpu` check,
JavaScript Object Notation (JSON) reports, and saved policies. Its default game has seven cards.
Run `python examples/goofspiel_ppo.py --iterations 10000 --require-gpu --devices 2 --compiled-updates`
to sample and optimize each training batch on two GPUs for one 10,000-iteration
PPO run. Use `--parallel sampling` or `--parallel optimizing` to parallelize
only that stage.

See [policies and evaluation](./policies.md#exact-evaluation-for-small-games)
for exact utility and exploitability on small games, and
[parallelism and memory](./scaling.md) for device configuration.
