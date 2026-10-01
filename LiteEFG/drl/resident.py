"""Owner-local packed trajectories and pooled optimizer execution.

Features are compacted once, while small complete timelines retain rewards and
temporal aggregate semantics. Explicit shard maps keep packed features local;
only gradient numerators, participation counts, and metrics are pooled.
"""
from dataclasses import dataclass, field, replace
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from .graph import Node, _aggregate_values


_TIMELINE_COMPONENT_LIMIT = 16


@dataclass(frozen=True)
class PackedRollout:
    """Compacted records owned by one trainer, with complete small timelines.

    ``records`` is a per-player tuple of dictionaries partitioned by device.
    Each device owns ``capacity`` rows. Valid rows retain global trajectory IDs,
    time indices, and local timeline indices. ``timeline`` retains fields with
    at most sixteen local components, including full rewards and alive masks.
    The arrays are immutable; consumers must preserve masks and index metadata.
    """
    records: tuple
    timeline: tuple
    valid_counts: np.ndarray
    capacity: int
    horizon: int
    episodes: int
    target_chunk_size: int
    _owner: object = field(repr=False, compare=False)

    def block_until_ready(self):
        """Wait for all packed records and cached timeline values."""
        jax.block_until_ready((self.records, self.timeline))
        return self


def _global_aggregate(values, valid, alive, spec):
    if spec.object != 'batch':
        return _aggregate_values(values, valid, alive, spec)
    values = jnp.asarray(values, dtype=jnp.result_type(values, jnp.float32))
    valid = valid & alive
    mask = valid.reshape(valid.shape + (1,) * (values.ndim - 2))
    count = jax.lax.psum(valid.sum(), 'learners')
    if spec.aggregator in ('sum', 'mean'):
        value = jax.lax.psum(jnp.where(mask, values, 0).sum((0, 1)), 'learners')
        if spec.aggregator == 'mean':
            value /= jnp.maximum(count, 1)
    elif spec.aggregator == 'min':
        value = jax.lax.pmin(jnp.where(mask, values, jnp.inf).min((0, 1)), 'learners')
    else:
        value = jax.lax.pmax(jnp.where(mask, values, -jnp.inf).max((0, 1)), 'learners')
    return jnp.broadcast_to(jnp.where(count > 0, value, spec.padding), values.shape)


class ResidentOptimizer:
    """Internal execution engine for trainer-owned packed rollouts.

    A positive point microbatch size is a global budget, rounded up to equal
    local widths across devices. Trajectory microbatches retain the original
    global trajectory groups. All modes pool gradients and apply each resource
    optimizer only once per original minibatch.
    """

    def __init__(self, trainer, target_chunk_size=128):
        if (isinstance(target_chunk_size, bool) or not isinstance(target_chunk_size, int)
                or target_chunk_size < 1):
            raise ValueError('target_chunk_size must be a positive integer')
        self.trainer = trainer
        self.devices = tuple(trainer.optimizing_devices)
        self.mesh = Mesh(np.asarray(self.devices, dtype=object), ('learners',))
        self.target_chunk_size = target_chunk_size
        self.executables = {}
        self.last_arguments = {}
        self.capture_arguments = False

    def _validate_packed(self, packed):
        if not isinstance(packed, PackedRollout):
            raise TypeError('Expected a PackedRollout from trainer.compact')
        if packed._owner is not self.trainer:
            raise ValueError('PackedRollout belongs to a different trainer')
        if (len(packed.records) != self.trainer.env.num_players
                or len(packed.timeline) != self.trainer.env.num_players
                or packed.horizon < 1 or packed.episodes < 1 or packed.capacity < 1
                or packed.episodes % len(self.devices)):
            raise ValueError('PackedRollout has inconsistent trajectory metadata')
        for row, timeline in zip(packed.records, packed.timeline):
            if (row['_valid'].shape != (len(self.devices) * packed.capacity,)
                    or timeline['_valid'].shape != (packed.horizon, packed.episodes)):
                raise ValueError('PackedRollout arrays do not match their trajectory metadata')
        return packed

    def _microbatch_options(self, packed, minibatch_size, microbatch_size, microbatch_unit):
        self.trainer._validate_microbatch_size(microbatch_size)
        self.trainer._validate_microbatch_unit(microbatch_unit)
        if microbatch_size == -1:
            return min(packed.capacity, minibatch_size * packed.horizon), 'full'
        if microbatch_unit == 'points':
            return (microbatch_size + len(self.devices) - 1) // len(self.devices), 'points'
        return min(packed.capacity, microbatch_size * packed.horizon), 'trajectories'

    def _put(self, tree, spec):
        sharding = NamedSharding(self.mesh, spec)
        return jax.tree_util.tree_map(lambda x: jax.device_put(x, sharding), tree)

    def _mapped(self, name, function, in_specs, out_specs):
        if name not in self.executables:
            mapped = shard_map(function, mesh=self.mesh, in_specs=in_specs,
                               out_specs=out_specs, check_rep=False)
            self.executables[name] = jax.jit(mapped)
        return self.executables[name]

    def _call(self, name, executable, *arguments):
        if self.capture_arguments:
            self.last_arguments[name] = arguments
        return executable(*arguments)

    def pack(self, data):
        """Compact each player's valid records without changing their owner."""
        if isinstance(data, PackedRollout):
            return self._validate_packed(data)
        horizon, episodes = self.trainer._validate_data(data, 2)
        if episodes % len(self.devices):
            raise ValueError('Episodes must divide evenly across resident devices')
        data = self._put(self.trainer._seat_records(data), P(None, 'learners'))
        def counts(rows):
            return jnp.stack([row['_valid'].sum() for row in rows])[None, :]
        function = self._mapped('pack_counts', counts, (P(None, 'learners'),), P('learners', None))
        valid_counts = np.asarray(jax.device_get(self._call('pack_counts', function, data)))
        local_size = horizon * (episodes // len(self.devices))
        maximum = max(int(valid_counts.max()), 1)
        capacity = min(local_size, 1 << (maximum - 1).bit_length())
        name = f'pack_{horizon}_{episodes}_{capacity}'
        def execute(rows):
            packed, timeline = [], []
            for row in rows:
                local_steps, local_episodes = row['_valid'].shape
                order = (jnp.arange(local_episodes)[:, None]
                         + jnp.arange(local_steps)[None, :] * local_episodes).reshape(-1)
                valid = row['_valid'].reshape(-1)[order]
                positions = jnp.nonzero(valid, size=capacity, fill_value=0)[0]
                indices = order[positions]
                active = jnp.arange(capacity) < valid.sum()
                chosen = {key: value.reshape((local_size,) + value.shape[2:])[indices]
                          for key, value in row.items()}
                chosen['_valid'] = active
                chosen['_alive'] = active
                chosen['_timeline_index'] = indices
                chosen['_time_index'] = indices // local_episodes
                chosen['_trajectory_id'] = (indices % local_episodes
                    + jax.lax.axis_index('learners') * local_episodes)
                packed.append(chosen)
                timeline.append({key: value for key, value in row.items()
                                 if math.prod(value.shape[2:]) <= _TIMELINE_COMPONENT_LIMIT})
            return tuple(packed), tuple(timeline)
        function = self._mapped(name, execute, (P(None, 'learners'),),
                                (P('learners'), P(None, 'learners')))
        records, timeline = self._call(name, function, data)
        return PackedRollout(records, timeline, valid_counts, capacity, horizon, episodes,
                             self.target_chunk_size, self.trainer)

    def _model_values(self, params, row, dependencies, operation, kind):
        graph = self.trainer.graph
        shape = row['_valid'].shape
        feature_index = dependencies[0] if kind == 'model' else dependencies[1]
        values, = graph._evaluate(params, row, (Node(graph, feature_index),), _batch_shape=shape)
        if kind == 'model':
            resource = graph.models[operation]
            return jax.lax.map(lambda x: resource.apply(params[operation], x), values,
                               batch_size=self.target_chunk_size)
        selector, = graph._evaluate(params, row, (Node(graph, dependencies[0]),), _batch_shape=shape)
        if graph._expressions[dependencies[0]] == ('input', (), 'player'):
            selector = row.get('_model_player', selector)
        selector = jnp.broadcast_to(selector, shape)
        branches = tuple(lambda x, model=graph.models[i], weights=params[i]: model.apply(weights, x)
                         for i in operation)
        def evaluate(pair):
            selected, x = pair
            value = jax.lax.switch(selected, branches, x)
            return jnp.where((selected >= 0) & (selected < len(branches)), value,
                             jnp.full_like(value, jnp.nan))
        return jax.lax.map(evaluate, (selector, values), batch_size=self.target_chunk_size)

    def update(self, packed, *, upd_color):
        """Refresh selected cached colors using current parameters.

        Full temporal aggregates require small timeline values. A segment
        aggregate cannot depend on a model: its inactive-turn values were not
        retained. Such graphs must use the trainer's dense update path.
        Other-color prerequisites must already be cached, as in dense update.
        """
        self._validate_packed(packed)
        graph = self.trainer.graph
        colors = graph._normalize_colors(upd_color)
        indices = tuple(i for i, color in enumerate(graph._colors) if color in colors)
        for row in packed.records:
            graph._color_plan(indices, graph._cached_nodes(row, colors), colors)
        for index in indices:
            kind, dependencies, operation = graph._expressions[index]
            if kind == 'aggregate' and operation.object == 'segment':
                ancestors, pending = set(), list(dependencies)
                while pending:
                    dependency = pending.pop()
                    if dependency in ancestors:
                        continue
                    ancestors.add(dependency)
                    source_kind, source_dependencies, _ = graph._expressions[dependency]
                    if source_kind in ('model', 'model_select'):
                        raise ValueError('Resident segment aggregates cannot depend on model values '
                                         'at inactive turns; use dense trajectory data')
                    pending.extend(source_dependencies)
        name = f'update_{packed.horizon}_{packed.episodes}_{packed.capacity}_{colors}'
        def execute(params, rows, timelines):
            updated_rows, updated_timelines = [], []
            for row, original in zip(rows, timelines):
                prepared = dict(original)
                target_row = dict(row)
                for index in indices:
                    prepared.pop(graph._node_key(index), None)
                    target_row.pop(graph._node_key(index), None)
                valid = prepared['_valid']
                alive = prepared.get('_alive', valid)
                shape = valid.shape
                for index in indices:
                    kind, dependencies, operation = graph._expressions[index]
                    if kind in ('model', 'model_select'):
                        values = self._model_values(params, target_row, dependencies, operation, kind)
                        if math.prod(values.shape[1:]) > _TIMELINE_COMPONENT_LIMIT:
                            raise ValueError('Resident cached model outputs must have at most '
                                             'sixteen local components; use dense trajectory data')
                        masked = jnp.where(row['_valid'].reshape((-1,) + (1,) * (values.ndim - 1)),
                                           values, 0)
                        destination = jnp.where(row['_valid'], row['_timeline_index'], valid.size)
                        result = jnp.zeros((valid.size,) + values.shape[1:], values.dtype)
                        result = result.at[destination].set(masked, mode='drop')
                        result = result.reshape(shape + values.shape[1:])
                    elif kind == 'aggregate':
                        source, = graph._evaluate(params, prepared, (Node(graph, dependencies[0]),),
                                                  _batch_shape=shape)
                        result = _global_aggregate(source, valid, alive, operation)
                    else:
                        try:
                            result, = graph._evaluate(params, prepared, (Node(graph, index),),
                                                      _batch_shape=shape)
                        except ValueError as error:
                            if 'Missing graph inputs:' not in str(error):
                                raise
                            raise ValueError('Resident target expression requires a large feature '
                                             'on the full timeline; use dense trajectory data') from error
                    if math.prod(result.shape[2:]) > _TIMELINE_COMPONENT_LIMIT:
                        raise ValueError('Resident cached target nodes must have at most sixteen '
                                         'local components; use dense trajectory data')
                    result = jax.lax.stop_gradient(result)
                    key = graph._node_key(index)
                    spec = jax.ShapeDtypeStruct(result.shape[2:], result.dtype)
                    previous = graph._node_specs.get(key)
                    if previous is not None and (previous.shape != spec.shape or previous.dtype != spec.dtype):
                        raise ValueError('Cached target node changed its local shape or dtype')
                    graph._node_specs[key] = spec
                    prepared[key] = result
                    target_row[key] = result.reshape((valid.size,) + result.shape[2:])[row['_timeline_index']]
                updated_rows.append(target_row)
                updated_timelines.append(prepared)
            return tuple(updated_rows), tuple(updated_timelines)
        function = self._mapped(name, execute, (P(), P('learners'), P(None, 'learners')),
                                (P('learners'), P(None, 'learners')))
        params = self._put(tuple(resource.state.params for resource in graph.models), P())
        records, timeline = self._call(name, function, params, packed.records, packed.timeline)
        return replace(packed, records=records, timeline=timeline)

    def plan(self, packed, *, epochs, minibatch_size, generator,
             microbatch_size=-1, microbatch_unit='trajectories'):
        """Plan the dense API's global shuffle without moving packed features."""
        self._validate_packed(packed)
        for name, value in (('epochs', epochs), ('minibatch_size', minibatch_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        width, mode = self._microbatch_options(packed, minibatch_size,
                                              microbatch_size, microbatch_unit)
        episodes = packed.episodes
        name = f'plan_{epochs}_{episodes}_{minibatch_size}'
        if name not in self.executables:
            def generate(key):
                def epoch(key, _):
                    key, shuffle = jax.random.split(key)
                    order = jax.random.permutation(shuffle, episodes)
                    rank = jnp.zeros(episodes, jnp.int32).at[order].set(jnp.arange(episodes))
                    return key, (order, rank, rank // minibatch_size)
                return jax.lax.scan(epoch, key, None, length=epochs)[1]
            self.executables[name] = jax.jit(generate)
        orders, ranks, groups = self._call(name, self.executables[name],
                                           jax.device_put(generator, self.trainer.key.device))
        ranks, groups = self._put((ranks, groups), P())
        group_count = (episodes + minibatch_size - 1) // minibatch_size
        count_name = f'plan_counts_{epochs}_{group_count}_{packed.capacity}'
        def count(rows, groups):
            result = []
            for group_ids in groups:
                result.append(jnp.stack([jnp.bincount(group_ids[row['_trajectory_id']],
                    weights=row['_valid'].astype(jnp.int32), length=group_count) for row in rows], axis=-1))
            return jnp.stack(result)[None, ...]
        function = self._mapped(count_name, count, (P('learners'), P()), P('learners', None, None, None))
        counts = self._call(count_name, function, packed.records, groups)
        if mode == 'points':
            micro_counts = jnp.maximum(1, (counts.max(-1) + width - 1) // width)
        elif mode == 'trajectories':
            sizes = jnp.minimum(minibatch_size, episodes - jnp.arange(group_count) * minibatch_size)
            micro_counts = jnp.broadcast_to((sizes + microbatch_size - 1) // microbatch_size,
                                             counts.shape[:-1])
        else:
            micro_counts = jnp.ones(counts.shape[:-1], jnp.int32)
        return {'trajectory_orders': orders, 'rank_for_trajectory': ranks,
                'group_for_trajectory': groups, 'device_group_counts': counts,
                'device_microbatch_counts': micro_counts, 'local_microbatch_size': width,
                'microbatch_size': microbatch_size, 'microbatch_unit': microbatch_unit}

    def _local_accumulate(self, states, rows, rank, first_rank, stop_rank, colors,
                          horizon, width, mode, microbatch_size):
        capacity = rows[0]['_valid'].size
        point_orders, point_counts = [], []
        for row in rows:
            point_rank = rank[row['_trajectory_id']]
            valid = row['_valid'] & (point_rank >= first_rank) & (point_rank < stop_rank)
            sorting = jnp.where(valid, point_rank * horizon + row['_time_index'],
                                rank.size * horizon + jnp.arange(capacity))
            point_orders.append(jnp.argsort(sorting, stable=True))
            point_counts.append(valid.sum())
        if mode == 'points':
            groups = (jnp.stack(point_counts).max() + width - 1) // width
        elif mode == 'trajectories':
            groups = (stop_rank - first_rank + microbatch_size - 1) // microbatch_size
        else:
            groups = 1
        def differentiate(index):
            results = []
            for row, order, count in zip(rows, point_orders, point_counts):
                if mode == 'trajectories':
                    first = first_rank + index * microbatch_size
                    stop = jnp.minimum(first + microbatch_size, stop_rank)
                    ordered_ranks = rank[row['_trajectory_id'][order]]
                    valid = row['_valid'][order] & (ordered_ranks >= first) & (ordered_ranks < stop)
                    selected = jnp.nonzero(valid, size=width, fill_value=0)[0]
                    active = jnp.arange(width) < valid.sum()
                else:
                    positions = index * width + jnp.arange(width)
                    selected = jnp.minimum(positions, capacity - 1)
                    active = positions < count
                batch = jax.tree_util.tree_map(lambda value: value[order[selected]], row)
                batch['_valid'] = active
                results.append(self.trainer._minibatch_gradients(states, batch, colors))
            counts = jnp.stack([statistics[1] for _, statistics, _ in results])
            sums = jnp.stack([statistics[0] for _, statistics, _ in results])
            participation = jnp.stack([used for _, _, used in results]).sum(0)
            numerators = []
            for model in range(len(self.trainer.graph.models)):
                gradients = [gradient[model] for gradient, _, _ in results]
                usage = [used[model] for _, _, used in results]
                numerators.append(jax.tree_util.tree_map(lambda *leaves: sum(
                    jnp.where(used > 0, leaf * count, 0) for leaf, count, used in zip(leaves, counts, usage)),
                    *gradients))
            return tuple(numerators), sums, counts, participation
        totals = differentiate(0)
        return jax.lax.fori_loop(1, groups, lambda i, carry: jax.tree_util.tree_map(
            jnp.add, carry, differentiate(i)), totals)

    def _global_gradients(self, states, rows, rank, first_rank, stop_rank, colors,
                          horizon, width, mode, microbatch_size):
        local = self._local_accumulate(states, rows, rank, first_rank, stop_rank, colors,
                                       horizon, width, mode, microbatch_size)
        numerators, sums, counts, participation = jax.lax.psum(local, 'learners')
        gradients = tuple(jax.tree_util.tree_map(lambda x: x / jnp.maximum(participation[i], 1), value)
                          for i, value in enumerate(numerators))
        return gradients, sums, counts, participation

    def gradient_probe(self, packed, trajectory_ids, upd_color=None, *,
                       microbatch_size=-1, microbatch_unit='trajectories'):
        """Inspect pooled pre-clip gradients without publishing any update."""
        self._validate_packed(packed)
        if upd_color is None:
            upd_color = tuple(color for node, _ in self.trainer._objectives
                              if (color := self.trainer.graph._colors[node.index]) is not None)
        colors = self.trainer.graph._normalize_colors(upd_color)
        ids = np.asarray(trajectory_ids)
        if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
                or not len(ids) or len(np.unique(ids)) != len(ids)
                or ids.min() < 0 or ids.max() >= packed.episodes):
            raise ValueError('trajectory_ids must contain distinct valid global episode IDs')
        ids = ids.astype(np.int32)
        width, mode = self._microbatch_options(packed, len(ids), microbatch_size, microbatch_unit)
        rank = jnp.full(packed.episodes, packed.episodes, jnp.int32).at[jnp.asarray(ids)].set(jnp.arange(len(ids)))
        horizon = packed.horizon
        name = f'probe_{horizon}_{packed.episodes}_{packed.capacity}_{len(ids)}_{colors}_{mode}_{width}_{microbatch_size}'
        def execute(states, rows, ranks):
            gradients, sums, counts, participation = self._global_gradients(
                states, rows, ranks, 0, len(ids), colors, horizon, width, mode, microbatch_size)
            return dict(gradients=gradients, sums=sums, counts=counts, participation=participation)
        function = self._mapped(name, execute, (P(), P('learners'), P()), P())
        return self._call(name, function, self._put(self.trainer._model_states(), P()),
                          packed.records, self._put(rank, P()))

    def optimize(self, packed, *, upd_color, microbatch_size=-1,
                 microbatch_unit='trajectories'):
        """Update once using every supplied trajectory in its original order."""
        self._validate_packed(packed)
        ranks = self._put(jnp.arange(packed.episodes, dtype=jnp.int32)[None, :], P())
        return self._optimize_ranks(packed, ranks, packed.episodes, upd_color,
                                     microbatch_size, microbatch_unit)

    def optimize_epochs(self, packed, *, epochs, minibatch_size, generator, upd_color,
                        microbatch_size=-1, microbatch_unit='trajectories'):
        """Optimize unchanged global trajectory minibatches on their owners."""
        planning = self.plan(packed, epochs=epochs, minibatch_size=minibatch_size,
                             generator=generator, microbatch_size=microbatch_size,
                             microbatch_unit=microbatch_unit)
        return self._optimize_ranks(packed, planning['rank_for_trajectory'], minibatch_size,
                                     upd_color, microbatch_size, microbatch_unit)

    def _optimize_ranks(self, packed, ranks, minibatch_size, upd_color,
                        microbatch_size, microbatch_unit):
        trainer = self.trainer
        colors = trainer.graph._normalize_colors(upd_color)
        width, mode = self._microbatch_options(packed, minibatch_size,
                                              microbatch_size, microbatch_unit)
        episodes, horizon = packed.episodes, packed.horizon
        epochs = ranks.shape[0]
        group_count = (episodes + minibatch_size - 1) // minibatch_size
        name = f'optimize_{epochs}_{minibatch_size}_{horizon}_{episodes}_{packed.capacity}_{colors}_{mode}_{width}_{microbatch_size}'
        def execute(states, rows, ranks):
            def epoch(carry, rank):
                def update(carry, group):
                    states, failure, participated = carry
                    gradients, sums, counts, participation = self._global_gradients(
                        states, rows, rank, group * minibatch_size,
                        jnp.minimum((group + 1) * minibatch_size, episodes), colors,
                        horizon, width, mode, microbatch_size)
                    candidate = tuple(trainer._apply_gradient(state, gradient, resource, participation[i])
                        for i, (state, gradient, resource) in enumerate(zip(states, gradients, trainer.graph.models)))
                    values = sums / jnp.maximum(counts[:, None], 1)
                    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree_util.tree_leaves(candidate)]))
                    current_failure = jnp.where(jnp.all(jnp.isfinite(values)), jnp.where(finite, 0, 2), 1)
                    current_failure = jax.lax.pmax(current_failure, 'learners')
                    failure = jnp.where(failure != 0, failure, current_failure)
                    states = jax.lax.cond(failure == 0, lambda _: candidate, lambda _: states, None)
                    participated |= (participation > 0) & (failure == 0)
                    return (states, failure, participated), (values, counts)
                return jax.lax.scan(update, carry, jnp.arange(group_count))
            return jax.lax.scan(epoch, (states, jnp.array(0, jnp.int32),
                jnp.zeros(len(states), bool)), ranks)
        function = self._mapped(name, execute, (P(), P('learners'), P()), P())
        previous = trainer._model_states()
        (states, failure, participated), (values, counts) = self._call(
            name, function, self._put(previous, P()), packed.records, ranks)
        values, counts, failure, participated = jax.device_get((values, counts, failure, participated))
        if any(resource.state is not old for resource, old in zip(trainer.graph.models, previous)):
            raise RuntimeError('A model resource changed during resident optimization')
        for resource, state, active in zip(trainer.graph.models, states, participated):
            if active:
                resource.state = state
        if failure == 1:
            raise FloatingPointError('Non-finite graph loss or training metric')
        if failure == 2:
            raise FloatingPointError('Non-finite model parameters or optimizer state; update was not committed')
        totals = np.zeros(values.shape[2:], np.float64)
        valid_samples = np.zeros(trainer.env.num_players, np.int64)
        for value, count in zip(values.reshape((-1,) + values.shape[2:]), counts.reshape((-1, counts.shape[-1]))):
            totals += value.astype(np.float64) * count[:, None]
            valid_samples += count.astype(np.int64)
        means = totals / np.maximum(valid_samples[:, None], 1)
        names = ('loss', *(key for key, node in zip(trainer._metrics, trainer._metric_nodes)
            if trainer.graph._colors[node.index] is None or trainer.graph._colors[node.index] in colors))
        report = {key: means[:, i].tolist() for i, key in enumerate(names)}
        report.update(iteration=trainer.iteration, trajectories=trainer.iteration * trainer.batch_size,
            mean_utility=trainer._mean_utility.tolist(), valid_samples=valid_samples.tolist())
        return report
