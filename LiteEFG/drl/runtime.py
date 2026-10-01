"""Device-resident trajectory sampling, graph optimization, and policy matches."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .environment import Environment
from .graph import DRLGraph, ModelState, _validate_feature_name


def _feature_mapping(features):
    """Validate the public feature namespace without materializing device data."""
    if not isinstance(features, Mapping):
        raise TypeError("Environment features must be a mapping of names to JAX arrays")
    for name in features:
        _validate_feature_name(name)
    return dict(features)


def _environment_features(env, state):
    """Read named features, including the mandatory information-set vector."""
    method = getattr(env, "features", None)
    if not callable(method):
        raise ValueError("Environment must define features(state) with an 'information_set' vector")
    features = _feature_mapping(method(state))
    if "information_set" not in features:
        raise ValueError("Environment features must define 'information_set'")
    for name, value in features.items():
        value = jnp.asarray(value)
        if value.ndim < 1 or value.shape[0] != env.num_players:
            raise ValueError(f"Environment feature {name!r} must have shape "
                             f"[num_players, ...], with num_players={env.num_players}")
        features[name] = value
    information_set = features["information_set"]
    if information_set.ndim != 2 or information_set.shape[1] == 0:
        raise ValueError("Environment feature 'information_set' must have shape "
                         "[num_players, feature_size] with a nonempty vector per player")
    return features


def _environment_feature_specs(env):
    return jax.eval_shape(lambda key: _environment_features(env, env.init(key)),
                          jax.random.PRNGKey(0))


def example_inputs(observation_size, num_actions, *, features=None):
    """Construct one local input record, optionally with custom feature arrays."""
    inputs = {
        "information_set": jnp.zeros((observation_size,), jnp.float32),
        "legal_action_mask": jnp.ones((num_actions,), bool),
        "action": jnp.array(0, jnp.int32),
        "log_sampling_prob": jnp.array(0.0), "sampling_value": jnp.array(0.0),
        "reward": jnp.array(0.0), "player": jnp.array(0, jnp.int32),
    }
    if features is not None:
        inputs.update(_feature_mapping(features))
    if jnp.shape(inputs["information_set"]) != (observation_size,):
        raise ValueError("Example information_set must be a vector of the specified size")
    return inputs


def _params(states):
    return tuple(state.params for state in states)


def _distribution(probabilities, legal, active):
    """Validate policies while supplying a harmless action for padded records."""
    if probabilities.shape != legal.shape:
        raise ValueError("Policy output shape must match the legal-action mask")
    legal_count = legal.sum(-1, keepdims=True)
    uniform = jnp.where(legal_count > 0, legal / jnp.maximum(legal_count, 1),
                        jnp.ones_like(probabilities) / probabilities.shape[-1])
    mass = jnp.where(legal, probabilities, 0).sum(-1, keepdims=True)
    correct = (jnp.all(jnp.isfinite(probabilities) & (probabilities >= 0), axis=-1)
               & (jnp.abs(probabilities.sum(-1) - 1) < 1e-4)
               & jnp.all(jnp.where(legal, True, probabilities == 0), axis=-1)
               & (legal_count[..., 0] > 0))
    normalized = jnp.where(legal, probabilities, 0) / jnp.maximum(mass, 1e-30)
    safe = jnp.where((active & correct)[..., None], normalized, uniform)
    return safe, ~active | correct


def rollout(env: Environment, policies, keys, *, feature_policies=False):
    """One complete sampled path per key, with all state and records on device.

    An environment implements init/features/legal_action_mask/active_players/
    step using fixed-shape JAX pytrees. Actions are a vector of player actions;
    the environment ignores inactive players. ``features(state)``
    returns named arrays with shape ``[num_players, *local_shape]``. These
    include the required ``information_set`` vector. Pre-action features are
    included in the records. Each policy returns
    (prob,value); with ``feature_policies=True`` it also receives a third
    argument containing the current player's batched feature dictionary.
    """
    batch_size = keys.shape[0]
    episode_keys = jax.vmap(lambda key: jax.random.split(key, 2))(keys)
    init_keys, rollout_keys = episode_keys[:, 0], episode_keys[:, 1]
    states = jax.vmap(env.init)(init_keys)

    def transition(carry, _):
        states, random_keys, finished = carry
        features = jax.vmap(lambda state: _environment_features(env, state))(states)
        information_sets = features["information_set"]
        legal = jax.vmap(env.legal_action_mask)(states)
        active = jax.vmap(env.active_players)(states) & ~finished[:, None]
        split = jax.vmap(lambda key: jax.random.split(key, env.num_players + 2))(random_keys)
        actions, log_probs, values = [], [], []
        valid_policy = jnp.ones(batch_size, bool)
        for player, policy in enumerate(policies):
            if feature_policies:
                local_features = {name: value[:, player] for name, value in features.items()}
                probabilities, value = policy(information_sets[:, player], legal[:, player], local_features)
            else:
                probabilities, value = policy(information_sets[:, player], legal[:, player])
            probabilities, correct = _distribution(probabilities, legal[:, player], active[:, player])
            valid_policy &= correct
            logits = jnp.where(probabilities > 0, jnp.log(jnp.maximum(probabilities, 1e-30)), -jnp.inf)
            action = jax.vmap(jax.random.categorical)(split[:, player + 1], logits)
            probability = jnp.take_along_axis(probabilities, action[:, None], axis=-1)[:, 0]
            actions.append(action)
            log_probs.append(jnp.log(jnp.maximum(probability, 1e-30)))
            values.append(jnp.broadcast_to(value, (batch_size,)))
        actions = jnp.stack(actions, axis=1)
        next_states, reward, done = jax.vmap(env.step)(states, actions, split[:, -1])
        reward = jnp.where(finished[:, None], 0, reward)
        record = {
            "legal_action_mask": legal,
            "action": actions, "log_sampling_prob": jnp.stack(log_probs, axis=1),
            "sampling_value": jnp.stack(values, axis=1), "reward": reward,
            "_valid": active, "_done": done | finished, "_alive": ~finished,
            "player": jnp.broadcast_to(jnp.arange(env.num_players), active.shape),
            "_policy_valid": valid_policy,
            **features,
        }
        return (next_states, split[:, 0], finished | done), record

    (_, _, finished), records = jax.lax.scan(
        transition, (states, rollout_keys, jnp.zeros(batch_size, bool)), None, length=env.max_steps
    )
    return records, finished


class Policy:
    """A graph and a frozen parameter snapshot, suitable for GPU matches."""

    def __init__(self, graph, params, observation_size, num_actions, *, player=0,
                 model_player=None):
        self._validate_player(graph, player, route=model_player is None)
        if model_player is not None:
            self._validate_player(graph, model_player, route=True)
        self.graph, self.params = graph, params
        self.observation_size, self.num_actions = observation_size, num_actions
        self.player = int(player)
        self.model_player = None if model_player is None else int(model_player)

    @staticmethod
    def _validate_player(graph, player, *, route):
        if isinstance(player, bool) or not isinstance(player, (int, np.integer)) or player < 0:
            raise ValueError("Policy player must be a nonnegative integer")
        if route and any(player >= len(models) for models in graph.model_lists):
            raise ValueError("Policy model player is outside the ModelList range")

    def __call__(self, information_set, legal_action_mask, *, player=None, features=None):
        """Infer from a feature mapping, or an information-set array plus extras."""
        if isinstance(player, (int, np.integer)):
            self._validate_player(self.graph, player, route=self.model_player is None)
        if isinstance(information_set, Mapping):
            if features is not None:
                raise ValueError("Supply features once, using the first argument's mapping")
            inputs = _feature_mapping(information_set)
            if "information_set" not in inputs:
                raise ValueError("Policy features must define 'information_set'")
        else:
            inputs = {} if features is None else _feature_mapping(features)
            if "information_set" in inputs:
                raise ValueError("Supply information_set once, using a feature mapping or an array")
            inputs["information_set"] = information_set
        inputs.update(legal_action_mask=legal_action_mask, player=self.player if player is None else player)
        if self.model_player is not None:
            inputs["_model_player"] = self.model_player
        parameters = self.params
        if not any(isinstance(value, jax.core.Tracer) for value in jax.tree_util.tree_leaves(inputs)):
            leaves = [value for value in jax.tree_util.tree_leaves(inputs)
                      if isinstance(value, jax.Array)]
            if leaves:
                # Preserve the caller's array placement, replicating the small
                # parameter snapshot across a distributed input's mesh. Inside
                # a JIT the enclosing computation controls placement instead.
                from jax.sharding import NamedSharding, PartitionSpec
                source = inputs["information_set"]
                source = source if isinstance(source, jax.Array) else leaves[0]
                sharding = source.sharding
                target = (NamedSharding(sharding.mesh, PartitionSpec())
                          if isinstance(sharding, NamedSharding) else sharding)
                parameters = jax.device_put(parameters, target)
                inputs = jax.tree_util.tree_map(
                    lambda value: jax.device_put(value, target)
                    if isinstance(value, jax.Array) and value.devices() != source.devices()
                    else value, inputs)
        probabilities, = self.graph.evaluate(
            parameters, inputs,
            (self.graph.current_strategy(),),
        )
        return probabilities

    def save(self, path):
        """Write portable numerical parameters; no pickle or executable objects."""
        leaves = jax.tree_util.tree_leaves(self.params)
        specs = dict(self.graph.input_specs)
        specs.setdefault("information_set", jax.ShapeDtypeStruct((self.observation_size,), jnp.float32))
        metadata = {"format_version": 4, "num_actions": self.num_actions, "leaves": len(leaves),
                    "player": self.player,
                    "model_player": self.model_player,
                    "input_specs": {name: {"shape": list(spec.shape), "dtype": str(spec.dtype)}
                                    for name, spec in specs.items()}}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as output:
            np.savez_compressed(output, metadata=np.array(json.dumps(metadata)),
                                **{f"array_{i}": np.asarray(x) for i, x in enumerate(leaves)})

    @classmethod
    def load(cls, graph, path):
        """Load weights into an explicitly supplied, matching graph definition."""
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            version = metadata.get("format_version")
            if version not in (1, 2, 3, 4):
                raise ValueError("Unsupported policy checkpoint format")
            player = metadata.get("player", 0)
            if type(player) is not int or player < 0:
                raise ValueError("Checkpoint player must be a nonnegative integer")
            model_player = metadata.get("model_player") if version == 4 else None
            if model_player is not None and (type(model_player) is not int or model_player < 0):
                raise ValueError("Checkpoint model_player must be a nonnegative integer or null")
            actions = metadata["num_actions"]
            specs = _feature_mapping(metadata.get("input_specs", {}))
            if version in (1, 2):
                specs.setdefault("information_set", {
                    "shape": [metadata["observation_size"]], "dtype": "float32",
                })
            if "information_set" not in specs:
                raise ValueError("Policy checkpoint must define an information_set specification")
            feature_templates = {}
            for name, spec in specs.items():
                try:
                    shape = spec["shape"]
                    if (not isinstance(shape, list)
                            or any(type(size) is not int or size < 0 for size in shape)):
                        raise ValueError("Invalid feature shape")
                    feature_templates[name] = jnp.zeros(tuple(shape), dtype=spec["dtype"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"Invalid checkpoint specification for feature {name!r}") from error
            info = feature_templates["information_set"]
            if info.ndim != 1 or info.size == 0:
                raise ValueError("Checkpoint information_set must be a nonempty vector")
            obs = info.shape[0]
            if version in (1, 2) and obs != metadata["observation_size"]:
                raise ValueError("Checkpoint information_set shape disagrees with its input dimension")
            skeleton = _params(graph.init(
                jax.random.PRNGKey(0), example_inputs(obs, actions, features=feature_templates)))
            leaves, definition = jax.tree_util.tree_flatten(skeleton)
            if len(leaves) != metadata["leaves"]:
                raise ValueError("Policy checkpoint does not match the model architecture")
            restored = []
            for i, template in enumerate(leaves):
                array = archive[f"array_{i}"]
                if array.shape != template.shape or array.dtype != template.dtype or not np.isfinite(array).all():
                    raise ValueError("Policy checkpoint shape, dtype, or numerical values are invalid")
                restored.append(jnp.asarray(array))
        return cls(graph, jax.tree_util.tree_unflatten(definition, restored), obs, actions,
                   player=player, model_player=model_player)


def uniform_policy(information_set, legal_action_mask):
    del information_set
    mask = legal_action_mask.astype(jnp.float32)
    return mask / jnp.maximum(mask.sum(-1, keepdims=True), 1)


class Trainer:
    """Execute a graph using the live states of its caller-owned model resources.

    Each model object owns one parameter and optimizer state, including when
    used by multiple players or graphs. Each optimization step pools gradients
    and normalizes by the valid decisions that actually use that resource.
    The caller schedules collect(), update(), and optimize(); no training loop or
    aggregate-freezing schedule is inferred from the graph.
    """

    def __init__(self, graph, env: Environment, *, batch_size=1024, seed=0,
                 devices=None, parallel=None, sampling_devices=None, compact_batches=True):
        if not isinstance(graph, DRLGraph) or not graph.objectives:
            raise ValueError("Trainer requires a DRLGraph with a training objective")
        if graph._trainer is not None:
            raise RuntimeError("This DRLGraph is already bound to a trainer; "
                               "create a new graph instance for independent training")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(compact_batches, bool):
            raise ValueError("compact_batches must be a boolean")
        self.graph, self.env = graph, env
        self._compact_batches = compact_batches
        declared_actions = ("num_actions" in vars(graph)
                            or any("num_actions" in cls.__dict__ for cls in type(graph).__mro__))
        if declared_actions and graph.num_actions != env.num_actions:
            raise ValueError("Graph num_actions must match the environment")
        if any(len(models) != env.num_players for models in graph.model_lists):
            raise ValueError("Every ModelList must contain one entry per environment player")
        self.batch_size = batch_size
        self.iteration = 0
        key = jax.random.PRNGKey(seed)
        legacy_sampling = sampling_devices is not None
        if legacy_sampling and (devices is not None or parallel is not None):
            raise ValueError("sampling_devices cannot be combined with devices or parallel")
        selected = sampling_devices if legacy_sampling else devices
        try:
            selected = (key.device,) if selected is None else tuple(selected)
        except TypeError as error:
            raise ValueError("devices must be a sequence of distinct local JAX devices") from error
        if (not selected or any(not isinstance(device, jax.Device) for device in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError("devices must contain distinct local JAX devices")
        if len({device.platform for device in selected}) != 1:
            raise ValueError("devices must use the same platform")
        local_devices = jax.local_devices(backend=selected[0].platform)
        if any(device not in local_devices for device in selected):
            raise ValueError("devices must contain distinct local JAX devices")
        if parallel is None:
            parallel = ("sampling",) if legacy_sampling else ("sampling", "optimizing")
        try:
            parallel = (parallel,) if isinstance(parallel, str) else tuple(parallel)
        except TypeError as error:
            raise ValueError("parallel must select sampling and/or optimizing") from error
        if (any(stage not in ("sampling", "optimizing") for stage in parallel)
                or len(set(parallel)) != len(parallel)):
            raise ValueError("parallel must contain distinct sampling and/or optimizing stages")
        self.devices = selected
        self.parallel = tuple(stage for stage in ("sampling", "optimizing") if stage in parallel)
        if not legacy_sampling:
            key = jax.device_put(key, selected[0])
        self.sampling_devices = selected if "sampling" in parallel else (key.device,)
        self.optimizing_devices = selected if "optimizing" in parallel else (key.device,)
        self._parallel_layout = None
        if len(self.optimizing_devices) > 1:
            from .parallel import _ParallelLayout
            self._parallel_layout = _ParallelLayout(self.optimizing_devices)
        if batch_size % len(self.sampling_devices):
            raise ValueError("batch_size must be divisible by the number of sampling_devices")
        self.key, init_key = jax.random.split(key)
        specs = _environment_feature_specs(env)
        self._information_set_size = specs["information_set"].shape[1]
        inputs = example_inputs(
            self._information_set_size, env.num_actions,
            features={name: jnp.zeros(spec.shape[1:], spec.dtype) for name, spec in specs.items()},
        )
        graph.init(init_key, inputs)
        self._state_view = None
        self._compiled = {}
        self._resident_optimizers = {}
        self._metrics = tuple(graph.metrics)
        self._metric_nodes = tuple(graph.metrics.values())
        self._objectives = tuple(graph.objectives)
        self._mean_utility = np.zeros(env.num_players)
        # Publish ownership only after all environment and model initialization
        # succeeds. A failed binding must not leave a partially built trainer.
        graph._trainer = self

    def _model_states(self):
        return tuple(resource.state for resource in self.graph.models)

    def _resident_optimizer(self, target_chunk_size=128):
        from .resident import ResidentOptimizer
        if (isinstance(target_chunk_size, bool) or not isinstance(target_chunk_size, int)
                or target_chunk_size < 1):
            raise ValueError("target_chunk_size must be a positive integer")
        if target_chunk_size not in self._resident_optimizers:
            self._resident_optimizers[target_chunk_size] = ResidentOptimizer(
                self, target_chunk_size=target_chunk_size)
        return self._resident_optimizers[target_chunk_size]

    def compact(self, data, *, target_chunk_size=128):
        """Pack a rollout's valid features once on each optimizing device.

        Return a trainer-owned ``PackedRollout`` for ``update``, ``optimize``,
        and ``optimize_epochs``. Small reward and mask arrays keep their full
        time axes; large features retain their trajectory owner after packing.
        Episodes must divide evenly across optimizing devices. Target model
        evaluation uses at most ``target_chunk_size`` local points per call.
        The packed features are reused across all epochs and minibatches.
        """
        return self._resident_optimizer(target_chunk_size).pack(data)

    def _prepare_optimization_data(self, data, microbatch_size, microbatch_unit,
                                   trajectory_steps=None):
        from .resident import PackedRollout
        if isinstance(data, PackedRollout):
            return data
        episodes = (data[0]["_valid"].shape[1] if trajectory_steps is None
                    else data[0]["_valid"].size // trajectory_steps)
        if (self.compact_batches and microbatch_size > 0 and microbatch_unit == "points"
                and episodes % len(self.optimizing_devices) == 0):
            if trajectory_steps is not None:
                data = tuple({name: value.reshape((episodes, trajectory_steps) + value.shape[1:])
                              .swapaxes(0, 1) for name, value in row.items()} for row in data)
            return self.compact(data)
        return data

    @property
    def compact_batches(self):
        """Whether optimizer minibatches omit invalid sample slots."""
        return self._compact_batches

    @property
    def states(self):
        """Per-player views of the same unique resource table, in graph model order."""
        states = self._model_states()
        if self._state_view is None or any(current is not previous
                                          for current, previous in zip(states, self._state_view[0])):
            self._state_view = tuple(states for _ in range(self.env.num_players))
        return self._state_view

    def model_state(self, resource):
        """Read a caller-owned resource's current parameters and optimizer state."""
        self.graph.model_index(resource)
        return resource.state

    def _selected_objectives(self, upd_color):
        colors = None if upd_color is None else self.graph._normalize_colors(upd_color)
        objectives = tuple((node, selected) for node, selected in self._objectives
                           if colors is None or self.graph._colors[node.index] in colors)
        if not objectives:
            raise ValueError("Selected colors have no training objective")
        return objectives

    def _minibatch_gradients(self, states, batch, upd_color=None):
        graph = self.graph
        selected_colors = None if upd_color is None else graph._normalize_colors(upd_color)
        objectives = self._selected_objectives(upd_color)
        metric_nodes = tuple(node for node in self._metric_nodes
                             if selected_colors is None or graph._colors[node.index] is None
                             or graph._colors[node.index] in selected_colors)
        valid = batch["_valid"].astype(jnp.float32)
        count = valid.sum()
        denominator = jnp.maximum(count, 1)
        # Invalid records must not introduce undefined derivatives, such as
        # sqrt(0) on zero-filled padding. Evaluate them using a valid record's
        # inputs and cached aggregates, then exclude them from every reduction.
        # An entirely inactive minibatch never commits an optimizer update.
        representative = jnp.argmax(valid)
        batch = {
            name: jnp.where((valid > 0).reshape(valid.shape + (1,) * (value.ndim - 1)),
                            value, value[representative])
            for name, value in batch.items() if name != "_valid"
        }

        def masked_mean(value):
            if value.shape != valid.shape:
                raise ValueError("Training losses and metrics must be scalar per sample")
            return jnp.where(valid > 0, value, 0).sum() / denominator

        def objective(params):
            total = jnp.array(0.0)
            for node, selected in objectives:
                differentiable = tuple(p if i in selected else jax.lax.stop_gradient(p)
                                       for i, p in enumerate(params))
                value, = graph.evaluate(differentiable, batch, (node,), upd_color=upd_color)
                total += masked_mean(value)
            metrics = graph.evaluate(params, batch, metric_nodes, upd_color=upd_color)
            return total, jnp.stack((total, *(masked_mean(x) for x in metrics)))

        (_, metrics), gradients = jax.value_and_grad(objective, has_aux=True)(_params(states))
        usage = graph.model_usage(_params(states), batch, upd_color=upd_color)
        participation = jnp.stack([(mask & (valid > 0)).sum() for mask in usage])
        return gradients, (jnp.where(count > 0, metrics * count, 0), count), participation

    @staticmethod
    def _apply_gradient(state, gradient, resource, count):
        def optimize(_):
            updates, optimizer_state = resource.optimizer.update(gradient, state.opt_state, state.params)
            return ModelState(optax.apply_updates(state.params, updates), optimizer_state, state.step + 1)
        return jax.lax.cond(count > 0, optimize, lambda _: state, operand=None)

    def _update_minibatch(self, states, batch, upd_color=None):
        updated, sums, counts, _ = self._update_models(states, (batch,), upd_color)
        return updated, (sums[0], counts[0])

    def _update_models(self, states, records, upd_color):
        """Pool gradients before updating each distinct resource exactly once."""
        if self._parallel_layout is not None:
            # Constrain the actual model inputs after shuffling and compaction.
            # Replicated output states require globally reduced gradients before
            # clipping/Adam; the existing valid-decision weights remain global.
            records = self._parallel_layout.minibatch(records)
        results = tuple(self._minibatch_gradients(states, batch, upd_color) for batch in records)
        counts = jnp.stack([statistics[1] for _, statistics, _ in results])
        sums = jnp.stack([statistics[0] for _, statistics, _ in results])
        participation = jnp.stack([usage for _, _, usage in results]).sum(0)
        updated = []
        for index, resource in enumerate(self.graph.models):
            gradients = [gradient[index] for gradient, _, _ in results]
            participating_seats = [usage[index] for _, _, usage in results]
            gradient = jax.tree_util.tree_map(
                lambda *leaves: sum(jnp.where(used > 0, leaf * count, 0)
                                    for leaf, count, used in zip(leaves, counts, participating_seats))
                                / jnp.maximum(participation[index], 1), *gradients)
            if self._parallel_layout is not None:
                gradient = jax.tree_util.tree_map(lambda value: jax.lax.with_sharding_constraint(
                    value, self._parallel_layout.replicated), gradient)
            updated.append(self._apply_gradient(states[index], gradient, resource, participation[index]))
        return tuple(updated), sums, counts, participation

    @staticmethod
    def _validate_microbatch_size(microbatch_size):
        if (isinstance(microbatch_size, bool) or not isinstance(microbatch_size, int)
                or (microbatch_size != -1 and microbatch_size <= 0)):
            raise ValueError("microbatch_size must be -1 or a positive integer")

    @staticmethod
    def _validate_microbatch_unit(microbatch_unit):
        if (not isinstance(microbatch_unit, str)
                or microbatch_unit not in ("trajectories", "points")):
            raise ValueError("microbatch_unit must be 'trajectories' or 'points'")

    def _update_models_microbatched(self, states, records, orders, upd_color,
                                   microbatch_size, trajectory_steps, capacity,
                                   microbatch_unit="trajectories"):
        """Accumulate fixed-parameter gradients, then update each resource once.

        Trajectory chunks split before compaction. Point chunks compact only
        integer indices once per original group, then gather bounded features.
        Participation and gradient weights span the original minibatch.
        """
        size = orders[0].shape[0]
        if microbatch_unit == "points":
            width = min(size, microbatch_size)
            point_counts, point_orders = [], []
            for row, order in zip(records, orders):
                valid = row["_valid"][order]
                point_counts.append(valid.sum())
                positions = jnp.nonzero(valid, size=size, fill_value=0)[0]
                point_orders.append(order[positions])
            orders = tuple(point_orders)
            groups = (jnp.stack(point_counts).max() + width - 1) // width
        else:
            width = min(size, microbatch_size * trajectory_steps)
            capacity = min(capacity, width)
            groups = (size + width - 1) // width

        def differentiate(index):
            positions = index * width + jnp.arange(width)
            selected = jnp.minimum(positions, size - 1)
            batches = []
            for player, (row, order) in enumerate(zip(records, orders)):
                indices = order[selected]
                if microbatch_unit == "points":
                    valid = positions < point_counts[player]
                else:
                    valid = row["_valid"][indices] & (positions < size)
                    if capacity < width:
                        compact = jnp.nonzero(valid, size=capacity, fill_value=0)[0]
                        indices = indices[compact]
                        valid = jnp.arange(capacity) < valid.sum()
                batch = jax.tree_util.tree_map(lambda value: value[indices], row)
                batch["_valid"] = valid
                batches.append(batch)
            if self._parallel_layout is not None:
                batches = self._parallel_layout.minibatch(batches)
            results = tuple(self._minibatch_gradients(states, batch, upd_color)
                            for batch in batches)
            counts = jnp.stack([statistics[1] for _, statistics, _ in results])
            sums = jnp.stack([statistics[0] for _, statistics, _ in results])
            participation = jnp.stack([usage for _, _, usage in results]).sum(0)
            numerators = []
            for index in range(len(self.graph.models)):
                gradients = [gradient[index] for gradient, _, _ in results]
                usage = [used[index] for _, _, used in results]
                numerators.append(jax.tree_util.tree_map(
                    lambda *leaves: sum(jnp.where(used > 0, leaf * count, 0)
                                        for leaf, count, used in zip(leaves, counts, usage)),
                    *gradients))
            return tuple(numerators), sums, counts, participation

        # Initialize from the first chunk so arbitrary model/metric dtypes are
        # preserved without staging a full-minibatch forward or backward pass.
        totals = differentiate(0)

        def accumulate(index, totals):
            return jax.tree_util.tree_map(jnp.add, totals, differentiate(index))

        if microbatch_unit == "points" or groups > 1:
            totals = jax.lax.fori_loop(1, groups, accumulate, totals)
        numerators, sums, counts, participation = totals
        updated = []
        for index, resource in enumerate(self.graph.models):
            gradient = jax.tree_util.tree_map(
                lambda value: value / jnp.maximum(participation[index], 1), numerators[index])
            if self._parallel_layout is not None:
                gradient = jax.tree_util.tree_map(lambda value: jax.lax.with_sharding_constraint(
                    value, self._parallel_layout.replicated), gradient)
            updated.append(self._apply_gradient(states[index], gradient, resource, participation[index]))
        return tuple(updated), sums, counts, participation

    def _collect(self, parameters, sample_keys):
        graph, env = self.graph, self.env
        policies = []
        for player in range(env.num_players):
            def policy(obs, legal, features, params=parameters, player=player):
                strategy, = graph.evaluate(params, {**features, "legal_action_mask": legal,
                                                     "player": player},
                                            (graph.current_strategy(),))
                return strategy, jnp.zeros(obs.shape[0], jnp.float32)
            policies.append(policy)
        records, complete = rollout(env, policies, sample_keys, feature_policies=True)
        data = tuple({
            **{name: value[:, :, player] for name, value in records.items()
               if not name.startswith("_") and name != "sampling_value"},
            "_valid": records["_valid"][:, :, player], "_alive": records["_alive"],
        } for player in range(env.num_players))
        status = {
            "mean_utility": records["reward"].sum(0).mean(0),
            "complete": complete.all(), "valid_policies": records["_policy_valid"].all(),
        }
        return jax.lax.stop_gradient(data), status

    def collect(self):
        """Sample complete episodes using current policies, without evaluating critics.

        Return one record dictionary per player, with leading [time, episode]
        dimensions. A successful collection advances the sampling counter and
        RNG. Invalid trajectories leave both unchanged. Sampling devices split
        the global episode batch evenly and use the same current parameters.
        With parallel optimization, trajectories retain their episode sharding
        for target computation. Sampling-only execution gathers on the learner.
        """
        if "collect" not in self._compiled:
            self._compiled["collect"] = jax.jit(self._collect)
        key, sample_key = jax.random.split(self.key)
        sample_keys = jax.random.split(sample_key, self.batch_size)
        parameters = _params(self._model_states())
        shard_size = self.batch_size // len(self.sampling_devices)
        # Dispatch all devices before waiting, using the latest policy on each.
        shards = [self._compiled["collect"](
            jax.device_put(parameters, device),
            jax.device_put(sample_keys[index * shard_size:(index + 1) * shard_size], device),
        ) for index, device in enumerate(self.sampling_devices)]
        statuses = jax.device_get([status for _, status in shards])
        if not all(status["complete"] for status in statuses):
            raise ValueError("Environment max_steps did not finish every trajectory; truncated returns are unsupported")
        if not all(status["valid_policies"] for status in statuses):
            raise ValueError("A model produced invalid or illegal action probabilities")
        utilities = np.stack([status["mean_utility"] for status in statuses])
        if not np.isfinite(utilities).all():
            raise FloatingPointError("Non-finite sampled utility")
        layout = self._parallel_layout
        if layout is not None and self.sampling_devices == layout.devices:
            data = layout.join_trajectories([data for data, _ in shards])
        else:
            data = self._gather_trajectories(shards)
            if layout is not None:
                data = layout.put(data, layout.trajectory_sharding(self.batch_size))
        self.key = key
        self.iteration += 1
        self._mean_utility = utilities.mean(axis=0)
        return data

    def _gather_trajectories(self, shards):
        records = [jax.device_put(data, self.key.device) for data, _ in shards]
        if len(records) == 1:
            data = records[0]
        else:
            if "gather" not in self._compiled:
                self._compiled["gather"] = jax.jit(lambda *parts: jax.tree_util.tree_map(
                    lambda *values: jnp.concatenate(values, axis=1), *parts))
            data = self._compiled["gather"](*records)
        return data

    def _validate_data(self, data, rank):
        if not isinstance(data, (tuple, list)) or len(data) != self.env.num_players:
            raise ValueError("Data must contain one record dictionary per player")
        shape = None
        for records in data:
            if not isinstance(records, Mapping) or "_valid" not in records:
                raise ValueError("Each player's records must include a _valid mask")
            valid = records["_valid"]
            if valid.ndim != rank or valid.dtype != jnp.bool_ or any(size == 0 for size in valid.shape):
                raise ValueError(f"Expected a nonempty rank-{rank} boolean _valid mask")
            if shape is not None and valid.shape != shape:
                raise ValueError("Players must have matching sample dimensions")
            shape = valid.shape
            if any(value.shape[:rank] != shape for value in records.values()):
                raise ValueError("Record fields must have matching sample dimensions")
        return shape

    @staticmethod
    def _seat_records(data):
        return tuple(dict(records, player=jnp.full(records["_valid"].shape, player, jnp.int32))
                     for player, records in enumerate(data))

    @staticmethod
    def _compact_capacity(count, width):
        # Bounded powers of two avoid compiling a new model shape for every
        # possible valid count. Never enlarge an already dense minibatch.
        return min(width, 1 << (max(int(count), 1) - 1).bit_length())

    @staticmethod
    def _gather_minibatch(records, orders, capacity):
        """Gather valid rows within each original minibatch, retaining order.

        The caller selects a capacity that covers every player's valid count.
        Gather indices before features so a compiled epoch never needs to copy
        all padded feature arrays into a shuffled intermediate buffer.
        """
        result = []
        for row, order in zip(records, orders):
            if capacity < order.shape[0]:
                valid = row["_valid"][order]
                positions = jnp.nonzero(valid, size=capacity, fill_value=0)[0]
                selected = jax.tree_util.tree_map(lambda value: value[order[positions]], row)
                # nonzero's filler may repeat a valid row; explicitly mask it.
                selected["_valid"] = jnp.arange(capacity) < valid.sum()
            else:
                selected = jax.tree_util.tree_map(lambda value: value[order], row)
            result.append(selected)
        return tuple(result)

    def _minibatch_capacity(self, records):
        width = records[0]["_valid"].size
        if not self.compact_batches:
            return width
        key = "minibatch_valid_count"
        if key not in self._compiled:
            self._compiled[key] = jax.jit(lambda *masks: jnp.stack(
                [mask.sum() for mask in masks]).max())
        count = self._compiled[key](*(row["_valid"] for row in records))
        return self._compact_capacity(jax.device_get(count), width)

    @staticmethod
    def _microbatch_valid_count(masks, minibatch_width, microbatch_width):
        """Count valid rows without crossing original trajectory-group boundaries."""
        players, size = masks.shape
        masks = jnp.pad(masks, ((0, 0), (0, (-size) % minibatch_width)))
        masks = masks.reshape(players, masks.shape[1] // minibatch_width, minibatch_width)
        masks = jnp.pad(masks, ((0, 0), (0, 0),
                                (0, (-minibatch_width) % microbatch_width)))
        return masks.reshape(players, masks.shape[1],
                             masks.shape[2] // microbatch_width, microbatch_width).sum(-1).max()

    def _microbatch_capacity(self, records, trajectory_steps, microbatch_size):
        size = records[0]["_valid"].size
        width = min(size, trajectory_steps * microbatch_size)
        if not self.compact_batches:
            return width
        key = ("microbatch_valid_count", trajectory_steps, microbatch_size)
        if key not in self._compiled:
            self._compiled[key] = jax.jit(lambda *masks: self._microbatch_valid_count(
                jnp.stack(masks), masks[0].size, min(masks[0].size,
                                                      trajectory_steps * microbatch_size)))
        count = self._compiled[key](*(row["_valid"] for row in records))
        return self._compact_capacity(jax.device_get(count), width)

    def _plan_minibatches(self, data, epochs, minibatch_size, generator, microbatch_size=-1,
                          microbatch_unit="trajectories"):
        """Shuffle complete trajectories and select one capacity per call.

        All players use the same trajectory groups. Indices address the
        time-major source records in trajectory-major order, so each group
        contains every step of its selected episodes. Only masks and indices
        participate in planning; compaction needs one scalar host transfer.
        """
        key = ("minibatch_plan", epochs, minibatch_size)
        point_microbatches = microbatch_size != -1 and microbatch_unit == "points"
        if microbatch_size != -1:
            key += ("microbatch", microbatch_size)
            if point_microbatches:
                key += ("unit", "points")
        if key not in self._compiled:
            def plan(masks, loader_key):
                players, horizon, episodes = masks.shape
                size = horizon * episodes
                width = horizon * min(episodes, minibatch_size)
                flat_masks = masks.reshape(players, size)

                def epoch(key, _):
                    key, shuffle_key = jax.random.split(key)
                    trajectories = jax.random.permutation(shuffle_key, episodes)
                    indices = (trajectories[:, None]
                               + jnp.arange(horizon)[None, :] * episodes).reshape(-1)
                    orders = jnp.broadcast_to(indices, (players, size))
                    count = jnp.array(0, jnp.int32)
                    if self.compact_batches and not point_microbatches:
                        valid = jnp.take_along_axis(flat_masks, orders, axis=1)
                        if microbatch_size == -1:
                            valid = jnp.pad(valid, ((0, 0), (0, (-size) % width)))
                            count = valid.reshape(players, -1, width).sum(-1).max()
                        else:
                            micro_width = horizon * min(episodes, minibatch_size, microbatch_size)
                            count = self._microbatch_valid_count(valid, width, micro_width)
                    return key, (orders, count)

                _, (orders, counts) = jax.lax.scan(epoch, loader_key, None, length=epochs)
                return orders, counts.max()
            self._compiled[key] = jax.jit(plan)
        masks = jnp.stack([row["_valid"] for row in data])
        # Only masks and shuffle indices are planned on the coordinator; large
        # feature arrays stay distributed through target and model evaluation.
        masks, generator = jax.device_put((masks, generator), self.key.device)
        orders, count = self._compiled[key](masks, generator)
        width = masks.shape[1] * min(masks.shape[2], minibatch_size)
        if point_microbatches:
            # Only integer indices occupy this capacity. Stable raw width keeps
            # point-model executable shapes independent of valid-point counts.
            return orders, width
        if microbatch_size != -1:
            width = min(width, masks.shape[1] * microbatch_size)
        capacity = (self._compact_capacity(jax.device_get(count), width)
                    if self.compact_batches else width)
        return orders, capacity

    def update(self, data, *, upd_color):
        """Execute selected colors once on full trajectories and return their caches.

        This does not change model parameters. Cross-color prerequisites must
        already be present in data; selecting a color refreshes its results.
        """
        from .resident import PackedRollout
        if isinstance(data, PackedRollout):
            return self._resident_optimizer(data.target_chunk_size).update(
                data, upd_color=upd_color)
        _, episodes = self._validate_data(data, 2)
        colors = self.graph._normalize_colors(upd_color)
        layout = self._parallel_layout
        sharding = layout.trajectory_sharding(episodes) if layout is not None else None
        key = ("update", colors, sharding) if layout is not None else ("update", colors)
        if key not in self._compiled:
            def execute(states, records):
                return tuple(self.graph.update(
                    _params(states), inputs, upd_color=colors,
                    valid=inputs["_valid"], alive=inputs.get("_alive", inputs["_valid"]),
                ) for inputs in records)
            self._compiled[key] = (jax.jit(execute, in_shardings=(layout.replicated, sharding),
                                           out_shardings=sharding)
                                   if layout is not None else jax.jit(execute))
        states, records = self._model_states(), self._seat_records(data)
        if layout is not None:
            states, records = layout.put(states, layout.replicated), layout.put(records, sharding)
        else:
            states, records = jax.device_put((states, records), self.key.device)
        return self._compiled[key](states, records)

    def optimize(self, batch, *, upd_color, microbatch_size=-1,
                 microbatch_unit="trajectories"):
        """Evaluate selected loss colors, differentiate, and apply one optimizer step.

        Other colors are read from the batch's saved values. All players' updates
        are committed together after checking loss and optimizer-state finiteness.
        Positive ``microbatch_size`` counts complete trajectories by default,
        or valid decisions per player when ``microbatch_unit='points'``. Both
        counts are global across devices. Point chunks pack valid indices even
        when ``compact_batches=False``. Gradients accumulate before one optimizer
        step; ``-1`` evaluates the full minibatch at once in either unit. Use
        ``dataloader`` batches to preserve multi-step trajectory boundaries;
        plain rank-1 record tuples describe one-step trajectories. With
        compaction enabled, positive point microbatches pack the supplied
        batch once on each optimizing device when its trajectory count divides
        evenly across those devices. Existing target caches are preserved.
        """
        self._validate_microbatch_size(microbatch_size)
        self._validate_microbatch_unit(microbatch_unit)
        from .resident import PackedRollout
        colors = self.graph._normalize_colors(upd_color)
        if not isinstance(batch, PackedRollout):
            self._validate_data(batch, 1)
            trajectory_steps = getattr(batch, "trajectory_steps", 1)
            if (isinstance(trajectory_steps, bool) or not isinstance(trajectory_steps, int)
                    or trajectory_steps < 1 or batch[0]["_valid"].size % trajectory_steps):
                raise ValueError("Batch trajectory_steps must describe complete trajectories")
            batch = self._prepare_optimization_data(
                batch, microbatch_size, microbatch_unit, trajectory_steps)
        if isinstance(batch, PackedRollout):
            return self._resident_optimizer(batch.target_chunk_size).optimize(
                batch, upd_color=colors, microbatch_size=microbatch_size,
                microbatch_unit=microbatch_unit)
        if microbatch_size == -1:
            capacity = self._minibatch_capacity(batch)
        elif microbatch_unit == "points":
            capacity = batch[0]["_valid"].size
        else:
            capacity = self._microbatch_capacity(batch, trajectory_steps, microbatch_size)
        key = ("optimize", colors, capacity)
        if microbatch_size != -1:
            key += ("microbatch", microbatch_size, trajectory_steps)
            if microbatch_unit == "points":
                key += ("unit", "points")
        layout = self._parallel_layout
        if key not in self._compiled:
            def execute(states, records):
                if microbatch_size == -1:
                    if capacity < records[0]["_valid"].size:
                        records = self._gather_minibatch(
                            records, (jnp.arange(row["_valid"].size) for row in records), capacity)
                    new_states, sums, counts, participation = self._update_models(states, records, colors)
                else:
                    orders = tuple(jnp.arange(row["_valid"].size) for row in records)
                    if microbatch_unit == "points":
                        new_states, sums, counts, participation = self._update_models_microbatched(
                            states, records, orders, colors, microbatch_size, trajectory_steps,
                            capacity, microbatch_unit="points")
                    else:
                        new_states, sums, counts, participation = self._update_models_microbatched(
                            states, records, orders, colors, microbatch_size, trajectory_steps, capacity)
                finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf))
                                           for leaf in jax.tree_util.tree_leaves(new_states)]))
                return new_states, sums / jnp.maximum(counts[:, None], 1), counts, participation, finite
            self._compiled[key] = (jax.jit(execute, out_shardings=layout.replicated)
                                   if layout is not None else jax.jit(execute))
        previous = self._model_states()
        states, records = previous, self._seat_records(batch)
        if layout is not None:
            sharding = (layout.samples if batch[0]["_valid"].size % len(layout.devices) == 0
                        else layout.replicated)
            states, records = layout.put(states, layout.replicated), layout.put(records, sharding)
        else:
            states, records = jax.device_put((states, records), self.key.device)
        states, values, counts, participation, finite = self._compiled[key](states, records)
        values, counts, participation, finite = jax.device_get((values, counts, participation, finite))
        if not np.isfinite(values).all():
            raise FloatingPointError("Non-finite graph loss or training metric")
        if not finite:
            raise FloatingPointError("Non-finite model parameters or optimizer state; update was not committed")
        if any(resource.state is not state for resource, state in zip(self.graph.models, previous)):
            raise RuntimeError("A model resource changed during optimization; stale update was not committed")
        for resource, state, samples in zip(self.graph.models, states, participation):
            if samples > 0:
                resource.state = state
        names = ("loss", *(name for name, node in zip(self._metrics, self._metric_nodes)
                           if self.graph._colors[node.index] is None
                           or self.graph._colors[node.index] in colors))
        report = {name: values[:, index].tolist() for index, name in enumerate(names)}
        report.update(iteration=self.iteration, trajectories=self.iteration * self.batch_size,
                      mean_utility=self._mean_utility.tolist(), valid_samples=counts.astype(int).tolist())
        return report

    def optimize_epochs(self, data, *, epochs, minibatch_size, upd_color, generator=None,
                        microbatch_size=-1, microbatch_unit="trajectories",
                        precompute_color=None):
        """Optimize groups of ``minibatch_size`` complete trajectories.

        The shared trajectory shuffle and final partial group match ``dataloader``.
        A failed update retains all preceding valid updates, just as repeated
        calls to ``optimize`` do, and no subsequent updates are committed.
        ``microbatch_unit`` selects complete trajectories or valid decision
        points per player for positive ``microbatch_size``. Both accumulate
        gradients inside each original group before its single optimizer step;
        ``-1`` processes the original group at once in either unit.

        With compaction enabled, positive point microbatches pack the complete
        rollout once on each optimizing device when its episode count divides
        evenly. ``precompute_color`` optionally executes target colors once on
        this complete rollout, before shuffling or updating model parameters.
        Prepared features and targets are reused across every epoch. The default
        ``None`` uses only the supplied caches without recomputing targets.
        An explicit ``generator`` leaves the trainer's key unchanged. Omitting
        it splits and advances that key after target preparation succeeds,
        before attempting optimizer updates.
        """
        self._validate_microbatch_size(microbatch_size)
        self._validate_microbatch_unit(microbatch_unit)
        for name, value in (("epochs", epochs), ("minibatch_size", minibatch_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        colors = self.graph._normalize_colors(upd_color)
        if precompute_color is not None:
            precompute_color = self.graph._normalize_colors(precompute_color)
        from .resident import PackedRollout
        if isinstance(data, PackedRollout):
            self._resident_optimizer(data.target_chunk_size)._validate_packed(data)
        else:
            self._validate_data(data, 2)
            data = self._prepare_optimization_data(data, microbatch_size, microbatch_unit)
        if precompute_color is not None:
            data = self.update(data, upd_color=precompute_color)
        if generator is None:
            self.key, generator = jax.random.split(self.key)
        if isinstance(data, PackedRollout):
            return self._resident_optimizer(data.target_chunk_size).optimize_epochs(
                data, epochs=epochs, minibatch_size=minibatch_size,
                generator=generator, upd_color=colors,
                microbatch_size=microbatch_size, microbatch_unit=microbatch_unit)
        _, episodes = self._validate_data(data, 2)
        if microbatch_size != -1 and microbatch_unit == "points":
            orders, capacity = self._plan_minibatches(data, epochs, minibatch_size, generator,
                microbatch_size=microbatch_size, microbatch_unit="points")
        else:
            orders, capacity = self._plan_minibatches(data, epochs, minibatch_size, generator,
                                                     microbatch_size=microbatch_size)
        layout = self._parallel_layout
        sharding = layout.trajectory_sharding(episodes) if layout is not None else None
        key = ("optimize_epochs", colors, epochs, minibatch_size, capacity)
        if layout is not None:
            key += (sharding,)
        if microbatch_size != -1:
            key += ("microbatch", microbatch_size)
            if microbatch_unit == "points":
                key += ("unit", "points")
        if key not in self._compiled:
            def execute(states, records, planned_orders):
                horizon, episodes = records[0]["_valid"].shape
                size = horizon * episodes
                width = horizon * min(episodes, minibatch_size)
                records = jax.tree_util.tree_map(
                    lambda value: value.reshape((size,) + value.shape[2:]), records)
                full_batches, remainder = divmod(size, width)

                def update(carry, order):
                    states, failure, participated = carry
                    if microbatch_size == -1:
                        batch = self._gather_minibatch(records, order, capacity)
                        candidate, sums, counts, participation = self._update_models(
                            states, batch, colors)
                    else:
                        if microbatch_unit == "points":
                            candidate, sums, counts, participation = self._update_models_microbatched(
                                states, records, order, colors, microbatch_size, horizon,
                                capacity, microbatch_unit="points")
                        else:
                            candidate, sums, counts, participation = self._update_models_microbatched(
                                states, records, order, colors, microbatch_size, horizon, capacity)
                    values = sums / jnp.maximum(counts[:, None], 1)
                    finite_states = jnp.all(jnp.stack([
                        jnp.all(jnp.isfinite(leaf))
                        for leaf in jax.tree_util.tree_leaves(candidate)]))
                    new_failure = jnp.where(jnp.all(jnp.isfinite(values)),
                                            jnp.where(finite_states, 0, 2), 1)
                    failure = jnp.where(failure != 0, failure, new_failure)
                    states = jax.lax.cond(failure == 0, lambda _: candidate,
                                          lambda _: states, operand=None)
                    participated = participated | ((participation > 0) & (failure == 0))
                    return (states, failure, participated), (values, counts)

                def epoch(carry, order):
                    stats = None
                    if full_batches:
                        batches = order[:, :full_batches * width].reshape(
                            len(records), full_batches, width).swapaxes(0, 1)
                        carry, stats = jax.lax.scan(update, carry, batches)
                    if remainder:
                        carry, last = update(carry, order[:, full_batches * width:])
                        last = jax.tree_util.tree_map(lambda value: value[None], last)
                        stats = last if stats is None else jax.tree_util.tree_map(
                            lambda prefix, suffix: jnp.concatenate((prefix, suffix)), stats, last)
                    return carry, stats

                (states, failure, participated), (values, counts) = jax.lax.scan(
                    epoch, (states, jnp.array(0, jnp.int32),
                            jnp.zeros(len(states), dtype=bool)), planned_orders,
                    length=epochs)
                return states, values, counts, failure, participated

            self._compiled[key] = (jax.jit(execute,
                in_shardings=(layout.replicated, sharding, layout.replicated),
                out_shardings=layout.replicated) if layout is not None else jax.jit(execute))
        previous = self._model_states()
        states, records = previous, self._seat_records(data)
        if layout is not None:
            states, records = layout.put(states, layout.replicated), layout.put(records, sharding)
            orders = layout.put(orders, layout.replicated)
        else:
            states, records, orders = jax.device_put((states, records, orders), self.key.device)
        states, values, counts, failure, participated = self._compiled[key](
            states, records, orders)
        values, counts, failure, participated = jax.device_get((values, counts, failure, participated))
        if any(resource.state is not state for resource, state in zip(self.graph.models, previous)):
            raise RuntimeError("A model resource changed during optimization; stale update was not committed")
        # Publish only resources touched by an accepted update, including valid
        # steps preceding a failure. Untouched resources keep their identity.
        for resource, state, active in zip(self.graph.models, states, participated):
            if active:
                resource.state = state
        if failure == 1:
            raise FloatingPointError("Non-finite graph loss or training metric")
        if failure == 2:
            raise FloatingPointError("Non-finite model parameters or optimizer state; update was not committed")

        # Match the Python-double weighted accumulation of the explicit loop;
        # transfer only small reports, once after all device optimizer steps.
        totals = np.zeros(values.shape[2:], dtype=np.float64)
        valid_samples = np.zeros(self.env.num_players, dtype=np.int64)
        for step_values, step_counts in zip(values.reshape((-1,) + values.shape[2:]),
                                            counts.reshape((-1, counts.shape[-1]))):
            totals += step_values.astype(np.float64) * step_counts[:, None]
            valid_samples += step_counts.astype(np.int64)
        means = totals / np.maximum(valid_samples[:, None], 1)
        names = ("loss", *(name for name, node in zip(self._metrics, self._metric_nodes)
                           if self.graph._colors[node.index] is None
                           or self.graph._colors[node.index] in colors))
        report = {name: means[:, index].tolist() for index, name in enumerate(names)}
        report.update(iteration=self.iteration, trajectories=self.iteration * self.batch_size,
                      mean_utility=self._mean_utility.tolist(),
                      valid_samples=valid_samples.tolist())
        return report

    def policy(self, player):
        """Freeze a zero-based player's current graph parameters."""
        if not isinstance(player, int) or not 0 <= player < self.env.num_players:
            raise ValueError("player is a zero-based player index")
        parameters = jax.device_put(_params(self._model_states()), self.key.device)
        return Policy(self.graph, parameters, self._information_set_size,
                      self.env.num_actions, player=player, model_player=player)


def average_utility(env: Environment, policy0, policy1, *, episodes=10000, batch_size=1024, seed=0):
    """Estimate raw utility for a fixed two-policy matchup, with uncertainty.

    Policy objects receive the environment's batched feature mapping; external
    callables receive (information_set, legal_action_mask). Both return legal
    action probabilities. No training occurs. Episode seeds are independent
    of chunk size, and the final partial chunk is counted exactly.
    """
    if env.num_players != 2:
        raise ValueError("average_utility currently accepts two-player environments")
    for name, value in (("episodes", episodes), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    information_set_size = _environment_feature_specs(env)["information_set"].shape[1]
    for policy in (policy0, policy1):
        if isinstance(policy, Policy) and (policy.observation_size != information_set_size or policy.num_actions != env.num_actions):
            raise ValueError("Policy and environment input/output dimensions differ")

    def wrap(policy, player):
        def evaluate(obs, mask, features):
            probabilities = (policy(features, mask, player=player) if isinstance(policy, Policy)
                             else policy(obs, mask))
            return probabilities, jnp.zeros(obs.shape[0], jnp.float32)
        return evaluate

    @jax.jit
    def sample(keys):
        records, complete = rollout(env, (wrap(policy0, 0), wrap(policy1, 1)), keys,
                                    feature_policies=True)
        return records["reward"].sum(0), complete, records["_policy_valid"].all(axis=0)

    key = jax.random.PRNGKey(seed)
    total = np.zeros(2, np.float64)
    squares = np.zeros(2, np.float64)
    wins = np.zeros(2, np.int64)
    ties = 0
    for start in range(0, episodes, batch_size):
        count = min(batch_size, episodes - start)
        # Fixed-size final chunk avoids a second compilation; ignore padding.
        identifiers = jnp.arange(start, start + batch_size, dtype=jnp.uint32)
        keys = jax.vmap(lambda index: jax.random.fold_in(key, index))(identifiers)
        rewards, complete, valid = jax.device_get(sample(keys))
        if not complete[:count].all():
            raise ValueError("Environment max_steps did not finish the evaluation episodes")
        if not valid[:count].all():
            raise ValueError("An evaluation policy produced invalid or illegal probabilities")
        rewards = rewards[:count].astype(np.float64)
        if not np.isfinite(rewards).all():
            raise FloatingPointError("Non-finite evaluation utility")
        total += rewards.sum(0)
        squares += (rewards ** 2).sum(0)
        wins += (rewards > 0).sum(0)
        ties += int(np.all(rewards == 0, axis=1).sum())
    mean = total / episodes
    variance = np.maximum((squares - total * mean) / max(episodes - 1, 1), 0)
    stderr = np.sqrt(variance / episodes) if episodes > 1 else np.full(2, np.nan)
    return {"episodes": episodes, "seed": seed, "mean_utility": mean.tolist(),
            "standard_error": stderr.tolist(),
            "confidence_interval_95": np.stack((mean - 1.96 * stderr, mean + 1.96 * stderr), axis=-1).tolist(),
            "win_rate": (wins / episodes).tolist(), "tie_rate": ties / episodes,
            "estimator": "Monte Carlo mean of raw terminal utilities; normal-approximation 95% interval"}
