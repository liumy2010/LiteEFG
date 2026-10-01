---
description: Choose JAX devices and reduce activation and rollout memory with microbatches and compaction.
---

# Parallelism and memory

Use these options after the [training loop](./training.md) works on one device.
They control where work runs and how much data a network processes at once.

| Goal | Setting |
| --- | --- |
| Use multiple devices | `devices` and `parallel` in `graph.bind(...)`. |
| Reduce network activation memory | `microbatch_size` and `microbatch_unit` during optimization. |
| Reduce padded feature processing | `compact_batches` in `graph.bind(...)`, or `trainer.compact(data)`. |

## Choose devices and stages

`devices=None` uses the current default JAX device. Pass an explicit sequence
from `jax.local_devices()` to select multiple local devices. For example,
`graph.bind(env, batch_size=1024, seed=0, devices=jax.local_devices()[:2])`
uses two devices for both sampling and optimization. `parallel=None` selects
both stages; it does not select additional devices automatically.

The `parallel` option selects which stages share the chosen devices:

| `parallel` | Execution |
| --- | --- |
| `None` or `("sampling", "optimizing")` | Parallel sampling and synchronized data-parallel optimization |
| `("sampling",)` | Parallel sampling, with targets and optimization on one device |
| `("optimizing",)` | Sampling on one device and synchronized data-parallel optimization |
| `()` | Sampling and optimization on one device |

A single stage can also be passed as a string, such as `parallel="sampling"`.
`sampling_devices=[device0, device1]` selects sampling-only parallelism and
cannot be combined with `devices` or `parallel`.

Parallel sampling divides complete games evenly across the selected devices.
The global `batch_size` must divide evenly by the sampling device count; a
batch of 1024 games on two devices assigns 512 games to each. Every collection
uses the current policies and separate random keys for its games.

Target computation keeps complete trajectories together. Advantage
normalization and other batch aggregates use the complete global collection.

Parallel optimization partitions training inputs across devices.
Model parameters and optimizer states are replicated. Gradients are combined
using the global valid-sample counts before gradient clipping and the optimizer
update, so every replica performs the same update.

An optimization minibatch need not divide evenly by the device count: masked
padding fills the remaining slots. Device selection preserves the global batch
size, trajectory grouping, shuffle order, and optimizer update count. These
options work with both explicit `trainer.optimize` calls and compiled
`trainer.optimize_epochs` calls.

## Reduce activation memory

A microbatch is a smaller group within one optimizer minibatch. It reduces
memory used by network activations while accumulating gradients for the same
optimizer update.

| Example | One optimizer update processes |
| --- | --- |
| `minibatch_size=128`, `microbatch_size=-1` | All 128 trajectories in one pass. |
| `minibatch_size=128`, `microbatch_size=8` | 16 groups of eight trajectories. |
| `microbatch_unit="points"`, `microbatch_size=256` | Groups of valid decisions per player; see device rounding below for packed rollouts. |

Pass these options to `trainer.optimize` or `trainer.optimize_epochs`.
[Proximal Policy Optimization (PPO)'s training helper](./baselines/ppo.md#run-the-training-helper) also accepts them.

`microbatch_size=-1` uses the whole optimizer minibatch in one forward/backward
pass with either unit. A positive integer selects smaller groups within that
minibatch, counted globally across optimizing devices. The default
`microbatch_unit="trajectories"` counts complete trajectories shared by all
players. With `microbatch_unit="points"`, the size counts valid decisions per
player, not the sum across players. Point selection applies even with
`compact_batches=False`.

For `PackedRollout`, positive point microbatches use at most
`ceil(microbatch_size / number_of_optimizing_devices)` points per player on each
device. A size not divisible by the device count rounds the aggregate capacity
up. Each device selects only its assigned trajectories from the original
global minibatch.

Both modes accumulate gradients at fixed parameters, using the original sample
weights. Clipping and the optimizer run once per original minibatch. A final
smaller group keeps every remaining valid decision; padding has zero weight.

Cached targets, shuffle order, sample counts, and optimizer-step counts are
preserved. Smaller groups reduce activation memory at the cost of more passes.

## Compact padded data

Complete-game batches include inactive turns and padding after termination.
Compaction selects valid decisions for network evaluation. It is enabled by
default with `graph.bind(..., compact_batches=True)`.

Whole-minibatch and trajectory-microbatch optimization compact each selected
group. Positive point microbatches can pack features once and reuse them:

| Call | Data packed when trajectories divide evenly across optimizing devices |
| --- | --- |
| `trainer.optimize(batch, ...)` | The supplied batch. |
| `trainer.optimize_epochs(data, ...)` | The whole rollout, reused across all epochs and minibatches. |

When trajectories do not divide evenly, point microbatches use the ordinary
trajectory dictionaries. An existing `PackedRollout` is reused. Dataloader
batches retain complete trajectory boundaries; a plain rank-1 tuple treats
each record as a one-step trajectory.

Compaction preserves the selected trajectories, valid decisions, loss weights,
and optimizer schedule. It does not shorten games. Set `compact_batches=False`
when binding to disable packing for whole-minibatch and trajectory-microbatch
optimization. Point microbatches always select valid decisions. This boolean
setting applies to both explicit optimization and compiled `optimize_epochs`
calls.

Changing network batch dimensions can select different graphics processing unit (GPU) kernels and
floating-point reduction orders. Matching samples and optimizer updates
therefore does not guarantee bitwise-identical parameters.

### Pack a rollout explicitly

`trainer.compact(data, target_chunk_size=128)` returns a `PackedRollout` when
the episode count divides evenly across optimizing devices. Pass it to
`trainer.update` and `trainer.optimize_epochs`, or to `trainer.optimize` for
one update using the entire rollout. It belongs to the trainer that created it;
it is not an input to `dataloader`. Use `packed.block_until_ready()` when a
synchronous measurement is needed.

`target_chunk_size` controls the chunk size used when preparing targets.
With `PackedRollout`, target
colors must produce scalar or small vector caches. Use ordinary trajectory
dictionaries for graph updates that need full-timeline large features.

Use `precompute_color=[0]` with `optimize_epochs` to prepare targets before
optimization; see [training](./training.md#run-multiple-epochs-in-one-call).
