---
description: Bind a graph, collect games, prepare targets, choose minibatches, and interpret training metrics.
---

# Training

The trainer collects complete games, evaluates graph computations, and
optimizes model resources. The algorithm defines its loss, any targets, and
the order in which to run them. This page explains the shared trainer methods;
see [Deep Reinforcement Learning (DRL) baselines](./baselines.md) for algorithm-specific schedules.

## Bind the graph

Call `graph.bind(env, batch_size=1024, seed=0)` after completing the graph,
then access `graph.trainer`. Binding returns the graph and can be done only
once. Shared model resources keep their current state; use fresh resources
and a new graph for an independent run.

By default, collection and optimization use one JAX device. See
[parallelism and memory](./scaling.md) for `devices`, `parallel`,
`sampling_devices`, and `compact_batches`.

## Choose batch sizes

A **trajectory** is one complete sampled game. A **decision** is one player's
action within that game. Games can have different numbers of decisions.

| Setting | What it counts |
| --- | --- |
| `graph.bind(..., batch_size=512)` | Games collected in one call to `collect()`. |
| `leg.dataloader(..., batch_size=128)` | Complete trajectories in one optimizer minibatch. |
| `trainer.optimize_epochs(..., minibatch_size=128)` | Complete trajectories per minibatch in a multi-epoch call. |
| `epochs=4` | Passes over the collected trajectories before collecting again. |

A collection of 512 games with minibatches of 128 gives four optimizer updates
per epoch. All players use the same selected trajectories. A final smaller
minibatch keeps any remainder unless `drop_last=True` is set on the dataloader.

## Collect and prepare targets

`trainer.collect()` returns a tuple of player dictionaries containing JAX
arrays with leading dimensions `[max_steps, batch_size]`. Successful collection
increments `trainer.iteration`. Episodes must finish within `max_steps`;
truncated trajectories are unsupported.

If the graph computes trajectory aggregates or batch-wide targets, call
`trainer.update(data, upd_color=[0])` on the complete collection before creating
minibatches. It computes and caches the nodes in the selected colors. Cached
targets stay fixed throughout optimization.

Colors are labels chosen by the graph, not fixed trainer stages. The examples
use color `0` for targets and color `1` for the loss. A local loss without
trajectory aggregates can be optimized directly; graphs without explicit color
scopes use color `0`. See [custom objectives](./objectives.md) to define colors
and cached computations.

## Iterate over minibatches

Use `leg.dataloader(dataset, batch_size=1, shuffle=False, *, drop_last=False,
generator=None)` after preparing the targets.

| Option | Behavior |
| --- | --- |
| `batch_size` | Complete trajectories per minibatch. |
| `shuffle=True` | A new episode permutation on each epoch, shared by all players. |
| `generator` | JAX pseudorandom number generator (PRNG) key for the loader's random state; the default uses seed 0. |
| `drop_last=True` | Omit a final incomplete trajectory group. |

Iterating over the same loader starts a new epoch. `len(loader)` gives the
number of groups in an epoch. Each batch stays on the JAX device and contains
all time steps of the selected episodes, flattened in episode order and then
time order. Features, cached targets, and masks remain aligned.

Pass a batch to `trainer.optimize(batch, upd_color=[1])` to update the models
using the losses in color `1`; select the colors defined by your graph.
Inactive turns and terminal padding have zero loss weight. Each resource's
loss mean uses its valid decisions; a resource with no valid decisions does
not advance its optimizer. Shared resources pool decisions from their players
and receive one update.

### Working with flattened records

The loader uses a two-dimensional `_valid` mask to identify `[time, episode]`
data and preserves trajectory boundaries in its output. Plain dictionaries
with a one-dimensional mask treat each row as a one-step trajectory. Keep
the loader's batch objects when [microbatching complete trajectories](./scaling.md#reduce-activation-memory).
A smaller final batch may cause JAX to compile an additional optimization
function for its shape.

## Run multiple epochs in one call

`trainer.optimize_epochs(data, epochs=4, minibatch_size=128, upd_color=[1],
generator=None, microbatch_size=-1, microbatch_unit="trajectories",
precompute_color=None)` shuffles and optimizes the supplied collection.

Pass `precompute_color=[0]` to evaluate color `0` once on the complete collection
before the epochs. The default `None` uses caches already present in `data`.
Cached targets and batch aggregates remain fixed during the call.

An explicit `generator` controls shuffling without changing the trainer's
random key. With `None`, the method splits and advances the trainer's key once
after target preparation succeeds, before optimizer updates.

Algorithms can wrap these trainer methods in their own `train` method, as
[Proximal Policy Optimization (PPO) does](./baselines/ppo.md#run-the-training-helper). The base `DRLGraph.train`
raises `NotImplementedError`; a custom graph must define its schedule or use
the trainer methods directly.

## Read the metrics

| Report | Meaning |
| --- | --- |
| `loss` and graph metrics | Per-player means over valid decisions; each graph defines its own additional metrics. |
| `valid_samples` | Valid decision counts per player. Repeated training visits across epochs count again. |
| `iteration` | Successful collection count. |
| `trajectories` | Cumulative games collected. |
| `mean_utility` | Sampled game utilities, without the entropy bonus. |

`optimize` reports metrics for its minibatch and selected colors, plus direct
input or constant metrics; its utility comes from the latest collection.
When combining minibatch loss or graph metrics yourself, weight them by each
player's `valid_samples`.
