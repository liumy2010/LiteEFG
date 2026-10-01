---
description: Choose and configure LiteEFG's implemented regret minimization, mirror descent, and sampling baselines.
---

# Tabular baselines

LiteEFG includes 15 tabular baseline modules. Each implements an infoset computation graph in Python and delegates game traversal and graph execution to the C++ backend. Start with Counterfactual Regret Minimization (CFR) to learn the interface, then choose the update rule and sampling scheme appropriate to your experiment.

For algorithms that train neural policies from sampled games, see
[Deep Reinforcement Learning (DRL) baselines](./deep-learning/baselines.md).

## One interface, different update schedules

Import a baseline module and instantiate its lowercase `graph` class. With an
already constructed environment:

```python
from LiteEFG.baselines import CFR

algorithm = CFR.graph()
env.set_graph(algorithm)  # env is an already constructed environment

algorithm.update_graph(env)
strategy = algorithm.current_strategy()
```

`update_graph(env)` performs one call of the algorithm's Python update schedule. Depending on the baseline, that may update all players together, perform two alternating player updates, or select particular graph colors. Call this method instead of replacing it with `env.update(...)`: algorithms such as Clairvoyant Mirror Descent (CMD) and Regularized Counterfactual Regret Minimization (Reg-CFR) depend on their own schedule.

`current_strategy()` returns a `GraphNode` handle. Its numeric values live separately at each infoset in the attached environment. Use [environment methods](environments.md#api-reference) to inspect or evaluate them.

The traversal column below records the options exposed by each command-line
runner. See [environments](./environments.md) for what `Enumerate`,
`External`, and `Outcome` visit. Each algorithm page describes its
configuration, player assumptions, and update schedule.

## Choose an algorithm

| Baseline | Runner traversal | Runner evaluation |
| --- | --- | --- |
| [(external-sampling Monte Carlo) Counterfactual Regret Minimization (CFR / MCCFR)](./baselines/cfr.md) | `Enumerate` (default), `External` | `avg-iterate` |
| [Counterfactual Regret Minimization+ (CFR+)](./baselines/cfr-plus.md) | `Enumerate` (default), `External` | `linear-avg-iterate` |
| [Discounted CFR](./baselines/dcfr.md) | `Enumerate` (default), `External` | `last-iterate` of its [discounted average](./baselines/dcfr.md#select-a-strategy) |
| [Predictive CFR+](./baselines/pcfr.md) | `Enumerate` (default), `External` | `linear-avg-iterate` |
| [Outcome-sampling MCCFR](./baselines/os-mccfr.md) | `Outcome` | `avg-iterate` |
| [Dilated optimistic mirror descent](./baselines/domd.md) | `Enumerate` (default), `External`, `Outcome` | `last-iterate` |
| [Clairvoyant mirror descent](./baselines/cmd.md) | `Enumerate` (default), `External`, `Outcome` | `last-iterate` |
| [Regularized Dilated Optimistic Mirror Descent (Reg-DOMD)](./baselines/reg-domd.md) | `Enumerate` (default), `External`, `Outcome` | `last-iterate` |
| [Regularized CFR](./baselines/reg-cfr.md) | `Enumerate` (default), `External`, `Outcome` | `last-iterate` |
| [Magnetic mirror descent](./baselines/mmd.md) | Determined by `feedback` | `default` |
| [Q-function based regret minimization](./baselines/qfr.md) | Determined by `feedback` | `default` |
| [Implicit Exploration Online Mirror Descent (IXOMD)](./baselines/ixomd.md) | `Outcome` | `avg-iterate` |
| [Balanced Online Mirror Descent (Balanced OMD)](./baselines/balanced-omd.md) | `Outcome` | `avg-iterate` |
| [Balanced Follow the Regularized Leader (Balanced FTRL)](./baselines/balanced-ftrl.md) | `Outcome` | `avg-iterate` |
| [Follow the perturbed leader](./baselines/ftpl.md) | `Enumerate` (default), `External` | `avg-iterate` |

## Running baselines

Command-line runners use the defaults listed on each algorithm's page.

All runners default to `leduc_poker`. Most use 100,000 iterations; IXOMD, Balanced OMD, and Balanced FTRL use 1,000,000. Every runner defaults to `--print_freq 1000`.

The runner's `Exploitability` label reports the sum of all players' deviation gains. For a two-player zero-sum game, divide this value by two to obtain two-player exploitability.

For reproducible experiments, pass the important parameters explicitly and record the source revision, game string, traversal, iteration count, seed, and evaluation selector. The [examples](./examples) show how to run the checked-in scripts and how to control evaluation in a Python loop.

## Algorithm references

See the [Tabular baseline README](https://github.com/liumy2010/LiteEFG/blob/main/LiteEFG/baselines/README.md#references)
for the papers associated with these algorithms.
