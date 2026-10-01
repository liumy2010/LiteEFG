---
description: Choose FileEnv, OpenSpiel, or CppEnv for a game using LiteEFG's native solver.
---

# Tabular environments

Choose how to provide a game to LiteEFG's native solver:

| Interface | Input | Use it when |
| --- | --- | --- |
| [FileEnv](./environments/file-env.md) | A path to a game description | You have a bundled instance or a custom game-tree file. |
| [OpenSpiel](./environments/open-spiel.md) | A `pyspiel.Game` | You want an OpenSpiel game and policy export. |
| [CppEnv](./environments/cpp-env.md) | A C++ game implementation | You want to generate states during traversal. |

`FileEnv` and `OpenSpielEnv` store an explicit game tree. `CppEnv` generates
states as needed and stores information sets as they are discovered; see its
[graph support](./environments/cpp-env.md#graph-support) before choosing a baseline.

Attach a graph with `env.set_graph(graph)`, then follow the
[quick start](./quick-start.md) or a [tabular baseline](./algorithms.md).
See the [environment application programming interface (API)](#api-reference) for shared methods.
For neural policies, use the [JAX environments](./deep-learning/environments.md).

## Choose a traversal

Set `traverse_type` when constructing `FileEnv` or `OpenSpielEnv`. Names are case-sensitive.

| Mode | Work in an update |
| --- | --- |
| `"Enumerate"` | Visit the full tree and compute reach-weighted utility contributions. |
| `"External"` | For each updating player, enumerate that player's actions and sample chance and the other players. |
| `"Outcome"` | Sample one path through the tree using the supplied traversal strategy and chance probabilities. |

Choose a graph designed for the traversal. For example, the Counterfactual Regret Minimization (CFR) baseline supports `Enumerate` and `External`; outcome sampling uses algorithm-specific estimators in baselines such as `OS_MCCFR` and `IXOMD`. See [algorithms](./algorithms.md).

`env.update(..., traverse_type=...)` can override the traversal for one call. Both node selection and utility weighting use that mode; later calls with `traverse_type="default"` use the constructor's setting. The graph's estimator must support the selected traversal.

## API reference

Import LiteEFG with `import LiteEFG as leg`. `leg.Environment` supplies
shared methods inherited by `FileEnv` and `OpenSpielEnv`; it has no
public Python constructor. The environment pages describe their
constructors and policy-export methods.

### Player numbering

| Context | Convention |
| --- | --- |
| `player` and `upd_player` arguments | Players `1` through the number of players; `upd_player=-1` updates all players. |
| Lists of strategy nodes | Element `i - 1` supplies the strategy for player `i`; no chance-player entry. |
| `utility()` and `exploitability()` results | Element `i - 1` is the result for player `i`. |
| OpenSpiel policies, exported tables, `interact(controlled_player=...)` | OpenSpiel uses players `0` through the number of players minus one. |

Chance is internally numbered `0` in LiteEFG, but it is not a valid player argument for value or strategy access.

### Graph lifecycle

#### `set_graph`

```python
env.set_graph(graph)
```

Attach a `leg.Graph`. Initialize game metadata when needed, create graph values at each information set, and execute static backward and static forward operations. Finish defining the graph before attaching it. Call this once before the training loop; it is not an iteration step.

#### `update`

```python
env.update(strategy, upd_player=-1, upd_color=[-1], traverse_type="default")
env.update(strategies, upd_player=-1, upd_color=[-1], traverse_type="default")
```

Execute dynamic graph operations over the selected traversal.

| Argument | Meaning |
| --- | --- |
| `strategy` | A `GraphNode` holding a valid behavior strategy at each information set. All players use that node as their traversal strategy. |
| `strategies` | A list of `GraphNode` objects, exactly one per player, in player order. This is the alternative overload. |
| `upd_player` | `-1` for all players, or a one-based player number. |
| `upd_color` | Graph colors to execute. `[-1]` selects all colors; an unknown color raises an error. |
| `traverse_type` | `"default"` uses the constructor's mode. The binding also accepts `"Enumerate"`, `"External"`, and `"Outcome"`. |

The supplied node determines traversal probabilities; the graph determines how its stored variables are updated. `update` does **not** record a strategy history. A Python baseline's `update_graph(env)` may call this method multiple times with different players, colors, or traversal strategy nodes. See [computation graph](computation-graph.md).

An explicit `traverse_type` applies to both the visited nodes and utility weighting for this call. It does not change the constructor's default for subsequent calls. Choose a graph whose estimator supports the selected [traversal](environments.md#choose-a-traversal).

#### `update_strategy`

```python
env.update_strategy(strategy, update_best=False)
```

Convert the behavior strategy stored at `strategy` into sequence form for each player and record a snapshot. Every call updates the last, average, and linearly weighted average iterates. Use it at the sampling frequency you want represented in those averages, typically once after each algorithm iteration.

With `update_best=True`, evaluate the profile and keep a snapshot whenever the **sum of players' exploitability values** improves. This adds a full-tree evaluation. The default `False` does not maintain the best iterate.

Pass a single `GraphNode`. It holds separate local values at each player's information sets; using one node does not require players to use identical probability vectors.

### Strategy versions

`utility`, `exploitability`, and `get_strategy` accept `type_name`:

| Value | Strategy used | Required recording |
| --- | --- | --- |
| `"default"` | Current behavior strategy converted to sequence form at the time of the call. | None. |
| `"last-iterate"` | Most recently recorded snapshot. | `update_strategy(strategy)` |
| `"avg-iterate"` | Uniform average of recorded sequence-form strategies. | `update_strategy(strategy)` |
| `"linear-avg-iterate"` | Recorded sequence-form strategies weighted `1, 2, …, T`. | `update_strategy(strategy)` |
| `"best-iterate"` | Recorded profile with the lowest sum of exploitability values among snapshots evaluated for best tracking. | `update_strategy(strategy, update_best=True)` |

History is maintained separately for each strategy node. Requesting an unavailable stored version raises an error. Averages are formed in **sequence form**, not by independently averaging local behavior probabilities.

### Evaluate a profile

#### `utility`

```python
env.utility(strategy, type_name="default")  # -> list[float]
```

Accepts a single `GraphNode` or a list of one node per player. Returns each player's expected payoff under the supplied profile. Evaluation traverses the full game tree, even when training uses a sampling environment.

#### `exploitability`

```python
env.exploitability(strategy, type_name="default")  # -> list[float]
```

Accepts the same single-node or list forms as `utility`. Returns each player's **exploitability (best-response gain)**: their best-response payoff against the others minus their payoff under the supplied profile. It returns a list, not a scalar.

```python
strategy = graph.current_strategy()
exploitabilities = env.exploitability(strategy, type_name="avg-iterate")
print("Per-player exploitability:", exploitabilities)
print("Exploitability:", sum(exploitabilities))
```

LiteEFG reports this sum of players' best-response gains as **Exploitability**. In two-player zero-sum games, conventions that report half this sum differ by a factor of two from the sum printed here. Compare experiments using the same convention.

Evaluation checks the validity of the sequence-form strategy and computes best responses over the full tree. For frequent progress logging on larger games, account for this work separately from sampled updates.

### Read and write local values

#### `get_value`

```python
env.get_value(player, node)  # -> list[tuple[str, list[float]]]
```

Return `(information_set_name, value_vector)` pairs for a one-based `player`. Values are raw contents of the `GraphNode`, including one-element vectors for scalars. Attach the graph before calling this method.

`OpenSpielEnv` overrides this method to return OpenSpiel information-state strings. The vectors have the same shape as those returned by the base method.

#### `set_value`

```python
env.set_value(player, node, values)  # -> None
```

The binding accepts either:

- A list of vectors: one vector per information set, with lengths matching the node's stored vectors.
- A flat list of floats: the concatenation of those vectors in the same order.

Read the order from `get_value`; do not assume alphabetical order or file declaration order. Mismatched numbers of values or vector lengths raise errors.

```python
rows = env.get_value(1, node)
values = [vector for _, vector in rows]
# Modify the values while preserving their shapes and any strategy constraints.
env.set_value(1, node, values)
```

This writes the graph's current values; it does not rewrite previously recorded strategy snapshots.
