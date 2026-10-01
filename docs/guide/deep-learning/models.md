---
description: Wrap networks and optimizers as model resources, and share parameters between players.
---

# Models and optimizers

A model resource holds a network's parameters and optimizer state. The graph
uses those resources to define the learning rule. Create resources outside
the graph, then pass them to its constructor.

## Wrap a network

`leg.model(module, optimizer=None, name=None)` accepts a module implementing
`init(key, example_input)` and `apply(params, input)`. Alternatively, pass the
two functions directly with `leg.model(init=init_fn, apply=apply_fn)`. Both
interfaces use explicit parameter pytrees and JAX-traceable calculations.
The model must reshape features as needed and return the tensors consumed by
the algorithm; see the [baseline's requirements](./baselines.md).

The default optimizer is Optax Adam with learning rate `3e-4`; pass an Optax
transformation to configure the learning rate, gradient clipping, or optimizer.
The parameters returned by `init` must form a trainable pytree accepted by
`apply`, as with Flax Linen modules that use only the `params` collection.

The trainer does not manage mutable layer collections or extra per-call random
keys. Such models need an adapter that exposes the supported pure init/apply
interface.

## Share or separate parameters

A fresh `leg.model(...)` creates independent parameters, even when it wraps the
same network architecture. Reusing a resource shares its parameters, optimizer
state, and update counter, including across graphs.

Inside a custom graph, `leg.ModelList(resources)[self.player](inputs)` selects
a resource by the symbolic player index. A list of distinct resources gives
each player independent parameters; repeating a resource shares parameters
among the players that select it. Ordinary integer indexing, such as
`models[0]`, returns the resource itself.

Each optimization step pools a shared resource's gradients over the valid
decisions that selected it, then applies its optimizer once. A resource with
no valid samples does not advance its optimizer.

## Inspect model state

After initialization, `model_handle.state` exposes `params`, `opt_state`, and
`step`. See [policies and evaluation](./policies.md) to take a frozen policy
snapshot or save it for later use.

Continue with [training](./training.md) to collect games and optimize the
models, or [custom objectives](./objectives.md) to define the learning rule.
