---
description: Load and cache OpenSpiel games, choose a traversal, and export policies.
---

# OpenSpiel

`OpenSpielEnv` converts an OpenSpiel game into an explicit tree for
LiteEFG's native solver. Start with a small game that provides
information-state strings.

## Load a game

```python
import LiteEFG as leg
import pyspiel

game = pyspiel.load_game("kuhn_poker")
env = leg.OpenSpielEnv(game, traverse_type="Enumerate")
```

The adapter accepts sequential games and converts simultaneous games using `pyspiel.convert_to_turn_based`. The resulting game is available as `env.game`; exported policies correspond to that game.

## Conversion and caching

On the first load, the adapter enumerates the game tree, writes a text description, and loads it through `FileEnv`. It stores the description in:

```text
~/game_instances/<game-short-name>_<parameter>=<value>....openspiel
```

The filename identifies the game and its parameters. Later loads reuse the
cached description. To regenerate it, pass `regenerate=True`:

```python
env = leg.OpenSpielEnv(game, regenerate=True)
```

Regenerate the file after changing the game's rules or upgrading OpenSpiel. The cache filename does not include library versions; games with the same name and parameters share a cache, including custom games whose rules are not fully represented by those parameters.

::: info Tabular game requirements
The adapter builds an OpenSpiel `TabularPolicy` and materializes the full tree. Choosing a sampling traversal reduces work during updates; it does not avoid this initial enumeration or tree storage. Tabular conversion requires information-state strings. Start with a small game that implements information-state strings.
:::


## Choose a traversal

Set `traverse_type` when constructing the environment. For example,
use external sampling with the game above:

```python
env = leg.OpenSpielEnv(game, traverse_type="External")
```

See [traversal modes](../environments.md#choose-a-traversal) to choose a
mode supported by your baseline. Sampling reduces work during updates;
the adapter still enumerates the full game tree when loading it.

## Export and inspect policies

After training an `OpenSpielEnv` and recording strategies with `env.update_strategy(...)`:

```python
policy, tables = env.get_strategy(
    graph.current_strategy(), type_name="avg-iterate"
)
print(tables[0])  # Player 1 in LiteEFG; player 0 in OpenSpiel.
```

`policy` is an OpenSpiel `TabularPolicy`; `tables` contains one pandas `DataFrame` per player, with information-state strings and action probabilities.

You can play against an exported OpenSpiel policy in a terminal:

```python
env.interact(policy, controlled_player=0, epochs=10)
```

`controlled_player` follows OpenSpiel's **zero-based** numbering. LiteEFG's `get_value`, `set_value`, and base `get_strategy` methods use **one-based** player numbers. See the [application programming interface (API) reference](../environments.md#player-numbering) for the full convention.

## API reference

### `OpenSpielEnv`

```python
leg.OpenSpielEnv(game, traverse_type="Enumerate", regenerate=False)
```

`game` must be a `pyspiel.Game` that provides information-state strings. Sequential games are used directly; simultaneous games are converted to turn-based games. `env.game` holds the game used by the adapter. `regenerate=True` replaces its cached game description after successful conversion; incompatible cache formats are regenerated automatically. See [conversion and caching](open-spiel.md#conversion-and-caching) for resource requirements and cache location.

### `OpenSpielEnv.get_strategy`

```python
env.get_strategy(strategy_node, type_name="default")
# -> tuple[TabularPolicy, list[pandas.DataFrame]]
```

The OpenSpiel override has **no `player` argument**. It exports all players into a new OpenSpiel `TabularPolicy` and one pandas `DataFrame` per player. A table contains an `Infoset` column followed by action labels from OpenSpiel. Action probabilities are placed in the policy's legal-action positions.

```python
policy, tables = env.get_strategy(
    graph.current_strategy(), type_name="avg-iterate"
)
print(tables[0])
```

To explicitly request the base representation on an `OpenSpielEnv`, call:

```python
rows = leg.FileEnv.get_strategy(
    env, 1, graph.current_strategy(), "avg-iterate"
)
```

This returns the file's internal information-set labels, whereas the adapter's `get_value` translates labels back to OpenSpiel strings.

### `OpenSpielEnv.interact`

```python
env.interact(policy, controlled_player=0, reveal_private=True, epochs=1000)
```

Run terminal interaction against an OpenSpiel `TabularPolicy`. You choose legal action IDs for the **zero-based** `controlled_player`; other players sample actions from the policy, and chance is sampled from the game. The method prints payoffs and accumulated statistics.

`epochs` is the number of games. `reveal_private=True` prints the terminal state; `False` prints the controlled player's final information state instead. This is an interactive console helper.
