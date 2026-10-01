---
description: Load bundled game files or describe a custom tree for FileEnv.
---

# FileEnv

`FileEnv` loads a complete game tree from LiteEFG's text format.
Use a bundled instance or describe your own tree. See
[traversal modes](../environments.md#choose-a-traversal) to choose
how the solver visits that tree during training.

## Load a bundled game

From the repository root, after an editable install (`python -m pip install -e .`):

```python
import LiteEFG as leg

env = leg.FileEnv(
    "LiteEFG/game_instances/kuhn.game",
    traverse_type="Enumerate",
)
```

With a non-editable installation, run Python outside the checkout and pass an absolute path to the game file instead. This avoids the source directory shadowing the installed extension; see the [installation check](../installation.md#check-the-installation).

The following table describes the five bundled game instances.

| File | Instance |
| --- | --- |
| [kuhn.game](https://github.com/liumy2010/LiteEFG/blob/main/LiteEFG/game_instances/kuhn.game) | Two-player Kuhn poker, three ranks. |
| [leduc.game](https://github.com/liumy2010/LiteEFG/blob/main/LiteEFG/game_instances/leduc.game) | Two-player Leduc poker, three ranks and two copies. |
| [goofspiel.game](https://github.com/liumy2010/LiteEFG/blob/main/LiteEFG/game_instances/goofspiel.game) | Two players, hand size three, `limited_information: false`. |
| [D24.game](https://github.com/liumy2010/LiteEFG/blob/main/LiteEFG/game_instances/D24.game) | Two-player liar's dice, one die each, four faces. |
| [liars_dice.game](https://github.com/liumy2010/LiteEFG/blob/main/LiteEFG/game_instances/liars_dice.game) | Two-player liar's dice, one die each, six faces. |

## Describe your own game

Use a bundled `.game` file as the starting point. This is LiteEFG's text format, not a general parser for Gambit or arbitrary game files. Its three sections are a comment header, all nodes, and information-set groups.

The following are excerpts from [kuhn.game](https://github.com/liumy2010/LiteEFG/blob/main/LiteEFG/game_instances/kuhn.game); the linked file supplies the complete tree.

### Header

```text
# Kuhn instance with parameters:
#
# Opt {
#     game_tree: true,
#     num_players: 2,
#     num_ranks: 3,
#     output_file: "GameInstances/kuhn.txt",
# }
#
```

The parser requires a player count in the header, accepting `num_players:` or `players:`. Preserve the trailing comma on its value and the `# }` terminator. Other descriptive fields are not instructions to generate a tree or write an output file.

### Nodes and actions

```text
node / chance actions 12=0.16666667 13=0.16666667 21=0.16666667 23=0.16666667 31=0.16666667 32=0.16666667
node /C:12 player 1 actions k b
node /C:12/P1:k player 2 actions k b
node /C:12/P1:k/P2:k leaf payoffs 1=-1 2=1
```

- A chance node lists `action=probability` entries. The parser normalizes the supplied chance weights.
- A player node lists its player number and ordered action names. Players are numbered from **1**.
- A leaf lists `player=payoff` entries. Specify every player's payoff.
- A child path appends `/C:<action>` for chance or `/P<player>:<action>` for a player. The root `/` already provides the separating slash.

Use unique node names and provide every referenced child. Each information set must have consistent actions and action ordering across its nodes. The algorithms and sequence-form representation expect a perfect-recall game.

### Information sets

After defining **all nodes**, group indistinguishable decision nodes:

```text
infoset pl1_0__1?/ nodes /C:12 /C:13
infoset pl1_1__1?/1:k/2:b nodes /C:12/P1:k/P2:b /C:13/P1:k/P2:b
```

An information-set name is a whitespace-free label. Each group belongs to one player. A player decision node omitted from these groups becomes a singleton information set automatically. Do not group chance or terminal nodes.

Use space-separated tokens, Unix newlines, and one record per line. Follow the
bundled `.game` format when writing files by hand; generated `.openspiel` files
use a different child-reference convention. The parser does not comprehensively
validate malformed input.


## Inspect policies

After training, use `env.get_strategy(player, strategy, type_name)`
to obtain information-set names and probability vectors. Players are
numbered from 1. See the [environment application programming interface (API)](#get-strategy)
for strategy selectors and return values.

## API reference

### `FileEnv`

```python
leg.FileEnv(file_name, traverse_type="Enumerate")
```

Read a complete game tree from the path `file_name`. The traversal is one of `"Enumerate"`, `"External"`, or `"Outcome"`. See [game files](file-env.md#describe-your-own-game) for the format.

### `get_strategy`

```python
env.get_strategy(player, strategy, type_name="default")
# -> list[tuple[str, list[float]]]
```

On `FileEnv`, return named local behavior probability vectors for the chosen one-based `player`. Stored sequence-form versions are converted back to behavior form. If an information set has effectively zero own reach probability, this conversion returns a uniform local distribution there.

This differs from `get_value`: `get_strategy` interprets the node as a strategy, supports historical versions, and normalizes the returned local probabilities.
