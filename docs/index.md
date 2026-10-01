---
layout: home
title: Solving extensive-form games
description: A Python interface and C++ engine for expressing game-solving algorithms as local computation graphs.
hero:
  name: LiteEFG
  text: Game-solving algorithms, expressed locally.
  tagline: Define an update rule at one information set in Python. Let the C++ engine carry it through the game.
  image:
    src: /local-to-global.svg
    alt: One Python update rule distributed to information sets with separate local values
  actions:
    - theme: brand
      text: Run your first solver
      link: /guide/quick-start
    - theme: alt
      text: Understand the graph
      link: /guide/computation-graph
    - theme: alt
      text: GitHub
      link: https://github.com/liumy2010/LiteEFG
features:
  - title: Python rules. C++ execution.
    details: Compose local vector operations and information-set aggregation. The engine handles traversal and keeps each information set’s state.
    link: /guide/concepts
    linkText: Explore the model
  - title: A starting point for research
    details: Study 15 baseline implementations, from Counterfactual Regret Minimization (CFR) and its variants to mirror descent and sampled learning algorithms.
    link: /guide/algorithms
    linkText: Browse the algorithms
  - title: Bring the game you need
    details: Convert compatible OpenSpiel games or load the repository’s text game format. Choose full enumeration or sampled traversal.
    link: /guide/environments
    linkText: Choose an environment
---

<div class="home-content">

<div class="section-label">From a game to a strategy</div>

<div class="home-grid">
<div>

## A small loop. A full experiment.

Start with the existing CFR baseline on Kuhn poker. Attach its graph, run updates, and evaluate the average strategy. Then open the implementation to see the update rule behind it.

For neural policies, start with the [Proximal Policy Optimization (PPO) training example](./guide/deep-learning.md).

[Installation →](./guide/installation.md) &nbsp; [Complete quick start →](./guide/quick-start.md)

</div>
<div>

```python
import LiteEFG as leg
import pyspiel
from LiteEFG.baselines import CFR

env = leg.OpenSpielEnv(pyspiel.load_game("kuhn_poker"))
algorithm = CFR.graph()
env.set_graph(algorithm)

for _ in range(1000):
    algorithm.update_graph(env)
    env.update_strategy(algorithm.current_strategy())

gaps = env.exploitability(algorithm.current_strategy(), "avg-iterate")
print(sum(gaps))
```

</div>
</div>

<div class="home-note">

Built for extensive-form game research. Open source under the MIT license. [Read the paper and cite LiteEFG →](./research.md)

</div>
</div>
