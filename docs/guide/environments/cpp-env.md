---
description: Implement procedural games in C++ and load them with CppEnv.
---

# CppEnv

`LiteEFG.CppEnv` loads a C++ implementation that generates game states as the
solver traverses them, without requiring a complete game-tree file.
The implementation supports finite, sequential, perfect-recall games with any
number of players and arbitrary utilities. Algorithm convergence guarantees
depend on the chosen algorithm and game class.

## Loading and training

Create an environment with
`env = LiteEFG.CppEnv("my_game.cpp", parameters={}, traverse_type="External")`.
The factory compiles the source into a shared library and loads it into the
native extension. Parameters are passed to the game constructor as JavaScript Object Notation (JSON).

Attach a graph with `env.set_graph(graph)`. For the Counterfactual Regret Minimization (CFR) baseline, construct
`graph = LiteEFG.baselines.CFR.graph()`, then call `graph.update_graph(env)`
after each iteration. Call `env.update_strategy(graph.current_strategy())`
when recording strategy snapshots for averaged or last-iterate evaluation.
External sampling enumerates the updating player's actions and samples chance
and the other players. The updating player's branching can still make an
iteration expensive. Information sets are initialized when first encountered.

`update_strategy` scans the discovered information sets and actions to accumulate
exact realization-weighted averages over the recorded snapshots. Its cost is
linear in that stored action count. It is not a sampled averaging operation.
When evaluating the current strategy with `type_name="default"`, snapshots can
be omitted. `last-iterate` refers to the most recently recorded snapshot.

`get_strategy`, `get_value`, `utility`, and `exploitability` accept the same
graph nodes and player numbering as the explicit environment. `stats()` reports
the procedural environment's counters. `inspect(actions)` replays an action
sequence and returns state information for diagnostics.

## C++ interface

Include `LiteEFG/Game.h`, implement `liteefg::Game` and `liteefg::State`, and use
`LITEEFG_REGISTER_GAME(MyGame)` to export the factory. `MyGame` must accept a
`const std::string&` containing JSON parameters. The installed package includes
the [complete public interface](https://github.com/liumy2010/LiteEFG/blob/main/LiteEFG/include/LiteEFG/Game.h); the loader
supplies its include directory automatically. The
[Dark Hex implementation](https://github.com/liumy2010/LiteEFG/blob/main/examples/cpp_games/dark_hex.cpp) is a complete
example.

| Method | Contract |
| --- | --- |
| `Game::NumPlayers()` | Number of players. |
| `Game::NewInitialState()` | A fresh state owned by the caller. |
| `Game::Name()` | Descriptive game name. |
| `State::CurrentPlayer()` | Players are numbered from 1; chance is 0 and terminal is -1. |
| `State::LegalActions()` | Ordered integer action identifiers at the current state. |
| `State::ApplyAction(action)` / `UndoAction()` | Apply an action and restore the entire preceding state, including private observations. |
| `State::InformationSet()` | Stable key for the acting player's full information history. |
| `State::Returns()` | Terminal utility vector, with player 1 at index 0. |
| `State::ChanceOutcomes()` | Action/probability pairs for exact traversal. Probabilities must form a distribution. |
| `State::SampleChance(uniform)` | Optionally sample chance directly using the supplied uniform value in [0, 1). |
| `State::ChanceProbability(action)` | Probability of a sampled chance action. Override alongside direct sampling to avoid enumeration during training. |

All states with the same player's information-set key must have the same ordered
legal actions and the same preceding own information-set/action sequence.
The key must preserve the player's remembered actions and observations.
Different hidden states within an information set share solver variables; their
transitions and terminal utilities are still computed from their full state.

The default `SampleChance` and `ChanceProbability` enumerate `ChanceOutcomes`.
Override both when direct sampling is cheaper. Exact evaluation still requires
`ChanceOutcomes`, even when training uses the direct sampler.

## Graph support

The implicit backend supports local graph operations, own-player dynamic parent
and child aggregation, and static parent aggregation. CFR uses this supported
subset. Graphs requiring static child aggregation, `subtree_size`, or opponent
aggregation are rejected: these operations require structural information that
is unavailable from sampled histories alone. Choose a compatible graph rather
than assuming every baseline works with the procedural backend.

## Exact evaluation

`env.exploitability(strategy, type_name="default")` returns each player's exact
unilateral deviation gain. The sum is NashConv. In a two-player zero-sum game,
half of that sum is the usual exploitability convention. `env.utility` also
evaluates exactly. `env.evaluate(strategy, type_name="default")` returns
`utility`, `deviation_gain`, and `nash_conv` together using one evaluation;
prefer it when both utility and exploitability are needed.

Exact evaluation can be expensive and may discover information
sets that training has not visited yet, increasing the stored information-set
count. Lowering evaluation frequency reduces how often this cost is incurred.

The [Dark Hex runner](https://github.com/liumy2010/LiteEFG/blob/main/examples/dark_hex.py) trains the 3 by 3 board with
External CFR using `python examples/dark_hex.py`. It defaults to 10,000 updates,
evaluates exactly every 10,000 updates, and evaluates the final strategy.
Use `--iterations` and `--eval-every` to control these intervals.
`--strategy` selects the current or averaged strategy, and `--output` writes a
compact JSON report of metrics and runtime counters. The game uses perfect
recall, classical failed-move handling, and
reveals no opponent actions.

## Compilation and cache

The loader requires Linux, including Ubuntu under Windows Subsystem for Linux (WSL), or macOS with a
GCC/Clang toolchain that supports C++17 and is compatible with LiteEFG's C++ standard-library application binary interface (ABI). On
Windows, run both LiteEFG and the compiler inside WSL.
`CXX` selects the compiler and may include a compiler wrapper. `CXXFLAGS` and
`extra_compile_args` supply compiler arguments; relative include paths are
resolved from the source directory.

`compile_cpp_game(source, cache_dir=None, extra_compile_args=())` compiles a game
without constructing an environment and returns the shared-library path.
The default cache is `$XDG_CACHE_HOME/LiteEFG/cpp-games`, or
`~/.cache/LiteEFG/cpp-games` when `XDG_CACHE_HOME` is unset. `cache_dir` can also
be passed to `CppEnv`.

Changes to the source, included headers, or build configuration trigger a
rebuild. Compiler failures include the invocation and diagnostics.
