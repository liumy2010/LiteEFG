<p align="center">
  <img src="resources/logo.webp" alt="Logo" width="120" height="120">
</p>

## <h1 align="center">LiteEFG</h1>

<p align="center">
    An effective and easy-to-use library for solving Extensive-form games.
</p>

## Table of Contents

- [About the Project](#about-the-project)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Baselines](#baselines)

## About the Project

**LiteEFG** is an efficient `Python` library for solving Extensive-form games (EFGs) based on computation graph. By defining the local update-rule for an information set (infoset) with `Python`, `LiteEFG` will automatically distribute the computation graph to all infosets of the game. Compared to implementing purely by `Python`, `LiteEFG` can provide up to **100x** aaceleration.

### Speed of `LiteEFG`

To demonstrate the efficiency of `LiteEFG`, the computation time of running Counterfactual Regret Minimization (CFR) [[1]](#1) for $100,000$ iterations, **without sampling**, in the following environments are provided.

The results are tested on Ubuntu with CPU: `13th Gen Intel(R) Core(TM) i7-13700K   3.40 GHz`.

- Kuhn Poker: $\leq 0.5s$
- Leduc Poker: $10s$
- Goofspiel: $10s$
- 4-sided Liar's Dice: $45s$
- Liar's Dice: $2$ hours

## Installation

   ```sh
   git clone --recursive https://github.com/liumy2010/LiteEFG.git
   cd LiteEFG
   pip install .
   ```

### Prerequisites
- A compiler with C++17 support
- Pip 10+ or CMake >= 3.4 (or 3.14+ on Windows, which was the first version to support VS 2019)
- Ninja or Pip 10+

## Usage

### Structure of LiteEFG

  To use LiteEFG, the user first need to specify the computation graph of the algorithm. The computation graph has 2 parts: *backward*, and *forward*.

  #### Graphs

  - **Backward Graph.** LiteEFG will enumerate the infosets in the reversed breadth-first order of infosets, and the computation defined in backward graph will be executed. The variables of backward graph are stored **locally** in each infoset.
  - **Forward Graph.** LiteEFG will enumerate the infosets in the breadth-first order of infosets, and the computation defined in forward graph will be executed. The variables of forward graph are stored **locally** in each infoset.

  #### Order of Update

  - **Initialize.** The static computation will be executed once when initialize the environment. The order is still `backward->forward`. Initialization will be automatically done when calling `env.GetGraph(graph)`.
  - **Update.** At every iteration when calling `environment.Update`, the backward graph will be updated first. Then, the forward graph will be updated.

  #### Default Variables
  LiteEFG prepare several default variables for easier implementation of algorithms. Note that all variables are stored locally for each infoset.
  - `utility`: Updated every iteration. It stores the expected utility of each action of an infoset. For a two-player zero-sum game $\max_{\mathbf{x}}\min_{\mathbf{y}}\mathbf{x^\top A y}$, where $\mathbf{A}$ is the utility matrix, `utility`=$(\mathbf{Ay}^{tr})_{(I,a)}$ for max-player $\mathbf{x}$. $(\mathbf{x}^{tr}, \mathbf{y}^{tr})$ is the strategy fed into `env.Update`
  - `action_set_size`: The size of action set of an infoset
  - `reach_prob`: The reach probabiltiy of player $p$ of an infoset, where $p$ is the player to take actions at that infoset
  - `opponent_reach_prob`: The reach probability of players other than $p$ of an infoset
  - `subtree_size`: A vector with size `action_set_size`, which stores the number of terminal (infoset, action) pairs in the subtree when taking an action

### Operations

#### Basic Operations
- `GraphNode.inplace(expression)`: The value of the expression will be stored to the original address of GraphNode
- `GraphNode.sum() / LiteEFG.sum(x)`: Return sum of all elements of `GraphNode`
- `GraphNode.mean() / LiteEFG.mean(x)`: Return mean of all elements of `GraphNode`
- `GraphNode.max() / LiteEFG.max(x)`: Return max of all elements of `GraphNode`
- `GraphNode.min() / LiteEFG.min(x)`: Return min of all elements of `GraphNode`
- `GraphNode.copy()`: Return the copy of `GraphNode`
- `GraphNode.exp() / LiteEFG.exp(x)`: Return exponential of all elements of `GraphNode`. For every index $i$, let $x_i\to e^{x_i}$
- `GraphNode.log() / LiteEFG.log(x)`: Return log of all elements of `GraphNode`. For every index $i$, let $x_i\to \ln(x_i)$
- `GraphNode.argmax() / LiteEFG.argmax(x)`: Return argmax of  `GraphNode`
- `GraphNode.argmin() / LiteEFG.argmin(x)`: Return argmin of `GraphNode`
- `GraphNode.euclidean() / LiteEFG.euclidean(x)`: Return $\frac{1}{2}\sum_i x_i^2$, where $\mathbf{x}$ is the vector stored at `GraphNode`
`GraphNode.negative_entropy(shifted=False) / LiteEFG.negative_entropy(x, shifted=False)`: Return $\sum_i x_i\ln(x_i)+\mathbf{1}(\text{shifted})\log N$, where $\mathbf{x}\in\mathbf{R}^N$ is the vector stored at `GraphNode`
- `GraphNode.normalize(p_norm, ignore_negative=False) / LiteEFG.normalize(x, p_norm, ignore_negative=False)`: Normalize $\mathbf{x}$ by its $p$-norm. Specifically, $x_i\to \frac{x_i}{(\sum_j |x_j|^p)^{1/p}}$. When `ignore_negative=True`, $x_i\to \frac{[x_i]^+}{(\sum_j ([x_j]^+)^p)^{1/p}}$, where $[x_i]^+=x_i\cdot \mathbf{1}(x_i\geq 0)$.
- `GraphNode.dot(y) / LiteEFG.dot(x, y)`: Return $\langle \mathbf{x}, \mathbf{y}\rangle$
- `GraphNode.maximum(y) / LiteEFG.maximum(x, y)`: Return $\mathbf{z}$ with $z_i=\max(x_i,y_i)$. Also supports `GraphNode.maximum(scalar) / LiteEFG.maximum(x, scalar)`
- `GraphNode.minimum(y) / LiteEFG.minimum(x, y)`: Return $\mathbf{z}$ with $z_i=\min(x_i,y_i)$. Also supports `GraphNode.minimum(scalar) / LiteEFG.minimum(x, scalar)`
- `LiteEFG.pow(x, y)`: Return $x_i\to x_i^y$

#### Game Specific Operations
- `GraphNode.project(distance_name : ["L2", "KL"], gamma=0.0, mu=uniform_distribution)`: Project $\mathbf{x}\in\mathbf{R}^N$ to the perturbed simplex $\Delta_N:=\\{\mathbf{v}\succeq \gamma\mathbf{\mu}\colon \sum_i v_i=1\\}$, with respect to either Euclidean distance or `KL`-Divergence. By default, $\gamma=0.0$ and $\mathbf{\mu}=\frac{1}{N}\mathbf{1}$
- `LiteEFG.project(x, distance_name, gamma=0.0, mu=uniform_distribution)`: Return `x.project(distance_name, gamma, mu)`
- `LiteEFG.aggregate(x, aggregator_name : ["sum", "mean", "max", "min"], object_name : ["parent", "children"], padding=0.0)`: Aggregate variables from either `object_name` infoset. By default, `object_name="children"`
  - `object_name="children"`: For each action `a` of infoset `I`, the vector stored in `GraphNode` `x` of subsequent infosets under `(I,a)` will be concatenated into one vector. Then, the vector will be aggregated via `aggregator_name`. At last, the resulted scalars of each `(I,a)` pair will be concatenated again and returned. Therefore, the returned `GraphNode` stores a vector of size `action_set_size`. If there's no subsequent infoset under an `(I,a)` pair, use `padding` as the scalar for `(I,a)`
  - `object_name="parent"`: Assume `I` is a children of infoset-action pair `(I',a')`. If the vector stored at `GraphNode` `x` is of size `action_set_size` of infoset `I'`, then its $(a')^{th}$ component will be returned. Otherwise, if `x` is simply a scalar, `x` will be returned. If no parent exists, return `padding`

### Environments

- `Environment.Update(strategies, upd_player=-1)`: Update the computation graph stored in the environment. `strategies` is a list of length `num_players` which specify the strategy used to traverse the game for each player. `upd_player=-1` means that the graph of all players will be updated. Otherwise, only update the graph of `upd_player`.
- `Environment.Update(strategy, upd_player=-1)`: Same as `Environment.Update([strategy, strategy, ..., strategy], upd_player)`, *i.e.* all players use `strategy` to traverse the game.
- `Environment.UpdateStrategy(strategy, update_best=False)`: Store the sequence-form strategy corresponding to the behavior-form strategy stored in `strategy`
  - Last-iterate: $\mathbf{x}_T$
  - Average-iterate: $\frac{1}{T} \sum\limits_{t=1}^{T} \mathbf{x}_t$
  - Linear average-iterate: $\frac{2}{T(T+1)} \sum\limits_{t=1}^{T} t\cdot \mathbf{x}_t$ 
  - When `update_best=True`, compute the exploitability and store the sequence-form strategy with the lowest exploitability
- `Environment.Exploitability(strategy, type_name="default")`: Return the exploitability of each player when all players use `strategy`
  - `type_name="default"`: Compute the sequence-form strategy in real-time using the behavior-form strategy stored at `strategy`
  - `type_name="last-iterate"`: Need to call `Environment.UpdateStrategy(strategy)` first. Then, compute the exploitability corresponding to the last-iterate of the stored sequence-form strategy
  - `type_name="avg-iterate"`: Need to call `Environment.UpdateStrategy(strategy)` first. Then, compute the exploitability corresponding to the average-iterate of the stored sequence-form strategy
  - `type_name="linear-avg-iterate"`: Need to call `Environment.UpdateStrategy(strategy)` first. Then, compute the exploitability corresponding to the linear average-iterate of the stored sequence-form strategy
  `type_name="last-iterate"`: Need to call `Environment.UpdateStrategy(strategy, update_best=True)` first. Then, compute the exploitability corresponding to the best-iterate of the stored sequence-form strategy
- `Environment.Utility(strategy, type_name="default")`: Similar to `Environment.Exploitability` above, but returns the utility of each player when all players use `strategy`

#### File Environment

User can call initialize environments from game files stored in `LiteEFG/LiteEFG/GameInstances` by `LiteEFG.FileEnv(file_path, traverse_type : ["Enumerate", "External", "Outcome"])`. The traverse type specifies whether the environment will be explored via enumerating all nodes at each iteration, external-sampling, or outcome sampling [[2]](#2).

Currently, `LiteEFG` supports the following environments.
- Kuhn Poker [[3]](#3)
- Leduc Poker [[4]](#4)
- Liar's Dice
- 4-sided Liar's Dice
- Goofspiel

#### Parameterized Environment

To be more flexible, `LiteEFG` also supports writing new environments by `c++`. There are examples provided in `LiteEFG/LiteEFG/src/Environment/Leduc/` and `LiteEFG/LiteEFG/src/Environment/NFG/`.

## Example

  ```python
  class CFR(leg.Graph): # inherits the computation graph
    def __init__(self):
        super().__init__() # initialize computation graph
        self.backward(is_static=True)
        # In the following, we will define static backward graph variables

        ev = leg.const(size=1, val=0.0) # initialize expected value as a scalar 0.0
        self.strategy = leg.const(self.action_set_size, 1.0 / self.action_set_size) # initialize the strategy as uniform strategy
        self.regret_buffer = leg.const(self.action_set_size, 0.0) # initialize the regret buffer as all 0 vector

        self.backward()
        # In the following, we will define the backward graph variables

        gradient = leg.aggregate(ev, aggregator="sum")
        # For current infoset I and action a, aggregate will first concatenate the object variable "ev" of all infosets with parent equal to (I,a). Then, the aggregator "sum" will be called to aggregate the vector to 1 scalar. At last, a vector of size "action_set_size" will be returned, since aggregate maintain a scalar for each action
        gradient.inplace(gradient + self.utility)
        # does not change the address of gradient, replace it with new value
        ev.inplace(leg.dot(gradient, self.strategy))
        # replace expected value
        self.regret_buffer.inplace(self.regret_buffer + gradient - ev)
        # update regret_buffer
        self.strategy.inplace(leg.normalize(self.regret_buffer, p_norm=1.0, ignore_negative=True))
        # normalize the strategy by p_norm. ignore_negative means that negative componenents will be treated as 0 during normalization. When the p_norm is 0, a uniform distribution will be returned (all one vector divided by dimension)
      
    def UpdateGraph(self, env):
      env.Update(self.strategy)
      # update the graph for both players. self.strategy is the traversal strategy to 
    
    def Strategy(self):
      return self.strategy

  if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traverse_type", type=str, choices=["Enumerate", "External"], default="Enumerate")
    parser.add_argument("--iter", type=int, default=100000)
    parser.add_argument("--print_freq", type=int, default=1000)

    args = parser.parse_args()

    env = leg.FileEnv("GameInstances/leduc.game", traverse_type=args.traverse_type) # load the environment from files
    alg = CFR()
    env.GetGraph(alg) # pass graph to the environment

    for i in tqdm(range(args.iter)):
        alg.UpdateGraph(env) # update the graph
        env.UpdateStrategy(alg.Strategy()) # automatically update the last-iterate, best-iterate, avg-iterate, linear-avg-iterate of the strategy
        if i % args.print_freq == 0:
            print(i, env.Exploitability(alg.Strategy(), "avg-iterate")) # output the exploitabiltiy
  ```

## Baselines

In `LiteEFG`, the baselines are stored in `LiteEFG/LiteEFG/baselines`. Currently, the following algorithms are implemented.

- Counterfactual Regret Minimization (CFR) [[1]](#1)
- Monte-Carlo Counterfactual Regret Minimization (MCCFR) [[2]](#2)
- Counterfactual Regret Minimization+ (CFR+) [[5]](#5)
- Dilated Optimistic Mirror Descent (DOMD) [[6]](#6)
- Magnetic Mirror Descent (MMD) [[7]](#7)
- Q-Function based Regret Minimization (QFR)

## References
<a id="1">[1]</a> 
Zinkevich, Martin, et al. "Regret minimization in games with incomplete information." Advances in neural information processing systems 20 (2007).

<a id="2">[2]</a>
Lanctot, Marc, et al. "Monte Carlo sampling for regret minimization in extensive games." Advances in neural information processing systems 22 (2009).

<a id="3">[3]</a> 
Harold W Kuhn. A simplified two-person poker. Contributions to the Theory of Games, 1(417):
97–103, 1950.

<a id="4">[4]</a> 
Finnegan Southey, Michael Bowling, Bryce Larson, Carmelo Piccione, Neil Burch, Darse Billings,
and Chris Rayner. Bayes’ bluff: opponent modelling in poker. In Proceedings of the Twenty-First
Conference on Uncertainty in Artificial Intelligence, pages 550–558, 2005.

<a id="5">[5]</a> 
Tammelin, Oskari. "Solving large imperfect information games using CFR+." arXiv preprint arXiv:1407.5042 (2014).

<a id="6">[6]</a> 
Lee, Chung-Wei, Christian Kroer, and Haipeng Luo. "Last-iterate convergence in extensive-form games. Advances in Neural Information Processing Systems 34 (2021): 14293-14305.

<a id="7">[7]</a> 
Samuel Sokota, Ryan D'Orazio, J. Zico Kolter, Nicolas Loizou, Marc Lanctot, Ioannis Mitliagkas, Noam Brown, and Christian Kroer. "A Unified Approach to Reinforcement Learning, Quantal Response Equilibria, and Two-Player Zero-Sum Games." International Conference on Learning Representations (2023)