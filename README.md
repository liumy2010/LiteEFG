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
  ```
  pip install LiteEFG
  ```

  or

   ```sh
   git clone --recursive https://github.com/liumy2010/LiteEFG.git
   cd LiteEFG
   pip install .
   ```

### Prerequisites
- open_spiel >= 1.2
- A compiler with C++17 support
- Pip 10+ or CMake >= 3.4 (or 3.14+ on Windows, which was the first version to support VS 2019)
- Ninja or Pip 10+

## Usage

### Structure of LiteEFG

  To use LiteEFG, the user first need to specify the computation graph of the algorithm. The computation graph has 2 parts: *backward*, and *forward*.

  #### Graphs

  - **Backward Graph.** LiteEFG will enumerate the infosets in the reversed breadth-first order of infosets, and the computation defined in backward graph will be executed. The variables of backward graph are stored **locally** in each infoset
  - **Forward Graph.** LiteEFG will enumerate the infosets in the breadth-first order of infosets, and the computation defined in forward graph will be executed. The variables of forward graph are stored **locally** in each infoset

  To define graph nodes for backward / forward graph, user may use the following two class
  - `LiteEFG.backward(is_static=False, color=0)`: Within the `with` statement of this class, all graph nodes in the following will be defined in the backward graph
    - `is_static` indicates whether the graph node is static. If it is, then it will only be updated at initialization, otherwise update every time when calling `Environment.update`
    - `color`: Indicates the color of the nodes defined below. Then, during each update, the user may specify the node color that will be updated
- `LiteEFG.forward(is_static=False, color=0)`: Within the `with` statement of this class, all graph nodes in the following will be defined in the forward graph

Here's an example of computing the Fibonacci sequence.
  ```python
  with LiteEFG.backward(is_static=True):
      a = LiteEFG.const(size=1, val=1.0)
      b = LiteEFG.const(size=1, val=1.0)
  
  with LiteEFG.backward(is_static=False):
      c = a + b
      a.inplace(b.copy()) # change the value of a to b
      b.inplace(c.copy()) # change the value of b to c
      # The details of inplace and copy can be found in 
      # the following
  ```

  #### Order of Update

  - **Initialize.** The static computation will be executed once when initialize the environment. The order is still `backward->forward`. Initialization will be automatically done when calling `env.set_graph(graph)`
  - **Update.** At every iteration when calling `environment.update`, the backward graph will be updated first. Then, the forward graph will be updated

  #### Default Variables
  LiteEFG prepare several default variables for easier implementation of algorithms. Note that all variables are stored locally for each infoset.
  - `utility`: Updated every iteration. It stores the expected utility of each action of an infoset. For a two-player zero-sum game $\max_{\mathbf{x}}\min_{\mathbf{y}}\mathbf{x^\top A y}$, where $\mathbf{A}$ is the utility matrix, `utility`=$(\mathbf{Ay}^{tr})_{(I,a)}$ for max-player $\mathbf{x}$. $(\mathbf{x}^{tr}, \mathbf{y}^{tr})$ is the strategy fed into `env.update`
  - `action_set_size`: The size of action set of an infoset
  - `reach_prob`: The reach probabiltiy of player $p$ of an infoset, where $p$ is the player to take actions at that infoset
  - `opponent_reach_prob`: The reach probability of players other than $p$ of an infoset
  - `subtree_size`: A vector with size `action_set_size`, which stores the number of terminal (infoset, action) pairs in the subtree when taking an action

### Operations

#### Basic Operations
- `LiteEFG.set_seed`: Set the seed for the pseudo-random number generator
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
- `LiteEFG.aggregate(x, aggregator_name : ["sum", "mean", "max", "min"], object_name : ["children", "parent"]="children", player : ["self", "opponents"]="self", padding=0.0)`: Aggregate variables from either `object_name` infoset. By default, `object_name="children"`
  - `object_name="children"`: For each action `a` of infoset `I`, the vector stored in `GraphNode` `x` of subsequent infosets under `(I,a)` will be concatenated into one vector. Then, the vector will be aggregated via `aggregator_name`. At last, the resulted scalars of each `(I,a)` pair will be concatenated again and returned. Therefore, the returned `GraphNode` stores a vector of size `action_set_size`. If there's no subsequent infoset under an `(I,a)` pair, use `padding` as the scalar for `(I,a)`
  - `object_name="parent"`: Assume `I` is a children of infoset-action pair `(I',a')`. If the vector stored at `GraphNode` `x` is of size `action_set_size` of infoset `I'`, then its $(a')^{th}$ component will be returned. Otherwise, if `x` is simply a scalar, `x` will be returned. If no parent exists, return `padding`
  - `player` indicates whether to aggregate infosets belong to the player herself or aggregate infosets belong to opponents. An example can be found in `LiteEFG/baselines/Bil_QFR.py`

### Environments

- `Environment.update(strategies, upd_player=-1, upd_color=[-1])`: Update the computation graph stored in the environment. `strategies` is a list of length `num_players` which specify the strategy used to traverse the game for each player. `upd_player=-1` means that the graph of all players will be updated. Otherwise, only update the graph of `upd_player`. `upd_color=[-1]` means that all nodes will be updated. Otherwise, only node with the color in `upd_color` will be updated. An example can be found in `LiteEFG/baselines/CMD.py`
- `Environment.update(strategy, upd_player=-1, upd_color=[-1])`: Same as `Environment.update([strategy, strategy, ..., strategy], upd_player, upd_color)`, *i.e.* all players use `strategy` to traverse the game
- `Environment.update_strategy(strategy, update_best=False)`: Store the sequence-form strategy corresponding to the behavior-form strategy stored in `strategy`
  - Last-iterate: $\mathbf{x}_T$
  - Average-iterate: $\frac{1}{T} \sum\limits_{t=1}^{T} \mathbf{x}_t$
  - Linear average-iterate: $\frac{2}{T(T+1)} \sum\limits_{t=1}^{T} t\cdot \mathbf{x}_t$ 
  - When `update_best=True`, compute the exploitability and store the sequence-form strategy with the lowest exploitability
- `Environment.exploitability(strategy, type_name="default")`: Return the exploitability of each player when all players use `strategy`
  - `type_name="default"`: Compute the sequence-form strategy in real-time using the behavior-form strategy stored at `strategy`
  - `type_name="last-iterate"`: Need to call `Environment.update_strategy(strategy)` first. Then, compute the exploitability corresponding to the last-iterate of the stored sequence-form strategy
  - `type_name="avg-iterate"`: Need to call `Environment.update_strategy(strategy)` first. Then, compute the exploitability corresponding to the average-iterate of the stored sequence-form strategy
  - `type_name="linear-avg-iterate"`: Need to call `Environment.update_strategy(strategy)` first. Then, compute the exploitability corresponding to the linear average-iterate of the stored sequence-form strategy
  `type_name="last-iterate"`: Need to call `Environment.update_strategy(strategy, update_best=True)` first. Then, compute the exploitability corresponding to the best-iterate of the stored sequence-form strategy
- `Environment.utility(strategy, type_name="default")`: Similar to `Environment.exploitability` above, but returns the utility of each player when all players use `strategy`
- `Environment.get_value(player, node)`: Return a list of `(infoset, vector)` pairs, where vector is the value of `node` in the infoset
- `Environment.get_strategy(player, strategy, type_name="default")`: Return a list of `(infoset, vector)` pairs, where vector is the value of `strategy` in the infoset. `type_name` is the same meaning as that in `Environment.exploitability`

#### OpenSpiel Environment

`LiteEFG` is compatible with `OpenSpiel`[[8]](#8). To use the environment implemented in OpenSpiel, one can call `LiteEFG.OpenSpielEnv(game: pyspiel.game, traverse_type="Enumerate", regenerate=False)`. The game should be an `pyspiel.game` instance, *e.g.* `pyspiel.load_game("kuhn_poker")`. The traverse type specifies whether the environment will be explored via enumerating all nodes at each iteration, external-sampling, or outcome sampling [[2]](#2). `regenerate` denotes whether generating the game file again regardless of its existence

- `OpenSpielEnv.get_strategy(strategy, type_name="default")`: Return `(open_spiel.python.policy.TabularPolicy, [pandas.DataFrame])`. The function wrap up the strategy given by `Environment.get_strategy` to the `open_spiel.python.policy.TabularPolicy`. Also, for each player, her strategy is stored in a `pandas.DataFrame`, and the strategy of each player will be concatenated into a list of `pandas.DataFrame`
- `OpenSpiel.interact(policy: TabularPolicy, controlled_player=0, reveal_private=True, epochs=1000)`: Interact with the policy stored in `policy`. `controlled_player` indicate which player the user want to control, and the rest of the players will apply their policy stored in `policy`. `reveal_private` indicates whether to reveal the private information of all players at the end of each round, such as revealing the private cards of all players in poker games. `epochs` indicates how many rounds the user want to play

The example can be found in `LiteEFG/LiteEFG/src/Environment/OpenSpiel/OpenSpielToGameFile.py`

#### Game File Environment
To support games not implemented by OpenSpiel, `LiteEFG` also supports games described in a specific game format. An example can be found in `LiteEFG/game_instances/*.game`.

#### Parameterized Environment

To be more flexible, `LiteEFG` also supports writing new environments by `c++`. There are examples provided in `LiteEFG/LiteEFG/src/Environment/Leduc/` and `LiteEFG/LiteEFG/src/Environment/NFG/`

## Example

  ```python
  import LiteEFG as leg
  import pyspiel
  import argparse
  from tqdm import tqdm

  class CFR(leg.Graph): # inherits the computation graph
      def __init__(self):
          super().__init__() # initialize computation graph
          with leg.backward(is_static=True):
          # In the following, we will define static backward graph variables

              ev = leg.const(size=1, val=0.0) # initialize expected value as a scalar 0.0
              self.strategy = leg.const(self.action_set_size, 1.0 / self.action_set_size)
              # initialize the strategy as uniform strategy
              self.regret_buffer = leg.const(self.action_set_size, 0.0)
              # initialize the regret buffer as all 0 vector

          with leg.backward():
          # In the following, we will define the backward graph variables

              gradient = leg.aggregate(ev, aggregator="sum")
              # For current infoset I and action a, aggregate will first concatenate the object variable "ev" of all infosets with parent equal to (I,a). 
              # Then, the aggregator "sum" will be called to aggregate the vector to 1 scalar. 
              # At last, a vector of size "action_set_size" will be returned, 
              # since aggregate maintain a scalar for each action
              gradient.inplace(gradient + self.utility)
              # does not change the address of gradient, 
              # replace it with new value
              ev.inplace(leg.dot(gradient, self.strategy))
              # replace expected value
              self.regret_buffer.inplace(self.regret_buffer + gradient - ev)
              # update regret_buffer
              self.strategy.inplace(leg.normalize(self.regret_buffer, p_norm=1.0, ignore_negative=True))
              # Normalize the strategy by p_norm.
              # ignore_negative means that negative componenents will be treated as 0 during normalization. 
              # When the p_norm is 0, a uniform distribution will be returned (all one vector divided by dimension)
        
      def update_graph(self, env : leg.Environment) -> None:
          env.update(self.strategy)
          # update the graph for both players. 
          # self.strategy is the traversal strategy to compute the expected utility
      
      def current_strategy(self) -> leg.GraphNode:
          return self.strategy

  if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      parser.add_argument("--traverse_type", type=str, choices=["Enumerate", "External"], default="Enumerate")
      parser.add_argument("--iter", type=int, default=100000)
      parser.add_argument("--print_freq", type=int, default=1000)

      args = parser.parse_args()

      leduc_poker = pyspiel.load_game("leduc_poker")
      env = leg.OpenSpielEnv(leduc_poker, traverse_type=args.traverse_type) # load the environment from files
      alg = CFR()
      env.set_graph(alg) # pass graph to the environment

      for i in tqdm(range(args.iter)):
          alg.update_graph(env) # update the graph
          env.update_strategy(alg.current_strategy()) 
          # automatically update the last-iterate, best-iterate, avg-iterate, linear-avg-iterate of the strategy
          if i % args.print_freq == 0:
              print(i, env.exploitability(alg.current_strategy(), "avg-iterate")) # output the exploitabiltiy
  ```

## Baselines

In `LiteEFG`, the baselines are stored in `LiteEFG/LiteEFG/baselines`. Currently, the following algorithms are implemented.

- Counterfactual Regret Minimization (CFR) [[1]](#1)
- Monte-Carlo Counterfactual Regret Minimization (MCCFR) [[2]](#2)
- Counterfactual Regret Minimization+ (CFR+) [[5]](#5)
- Dilated Optimistic Mirror Descent (DOMD) [[6]](#6)
- Magnetic Mirror Descent (MMD) [[7]](#7)
- Clairvoyant Mirror Descent (CMD) [[9]](#9)
- Discounted Counterfactual Regret Minimization (DCFR) [[10]](#10)
- Predictive Counterfactual Regret Minimization (PCFR) [[11]](#11)
- Q-Function based Regret Minimization (QFR)
- Regularized Dilated Optimistic Mirror Descent (Reg-DOMD)
- Regularized Counterfactual Regret Minimization (Reg-CFR)

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

<a id="8">[8]</a> 
Lanctot, Marc, et al. "OpenSpiel: A framework for reinforcement learning in games." arXiv preprint arXiv:1908.09453 (2019).

<a id="9">[9]</a> 
Piliouras, Georgios, Ryann Sim, and Stratis Skoulakis. "Beyond time-average convergence: Near-optimal uncoupled online learning via clairvoyant multiplicative weights update." Advances in Neural Information Processing Systems 35 (2022): 22258-22269.

<a id="10">[10]</a> 
Brown, Noam, and Tuomas Sandholm. "Solving imperfect-information games via discounted regret minimization." Proceedings of the AAAI Conference on Artificial Intelligence. 2019.

<a id="11">[11]</a> 
Farina, Gabriele, Christian Kroer, and Tuomas Sandholm. "Faster game solving via predictive blackwell approachability: Connecting regret matching and mirror descent." Proceedings of the AAAI Conference on Artificial Intelligence. 2021.
