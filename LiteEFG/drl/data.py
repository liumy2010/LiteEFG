"""Small iterable batches of aligned per-player trajectory records."""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec


@jax.tree_util.register_pytree_node_class
class _TrajectoryBatch(tuple):
    """Tuple-compatible records with static whole-trajectory boundaries."""

    def __new__(cls, records, trajectory_steps):
        batch = super().__new__(cls, records)
        batch._trajectory_steps = trajectory_steps
        return batch

    @property
    def trajectory_steps(self):
        return self._trajectory_steps

    def tree_flatten(self):
        return tuple(self), self.trajectory_steps

    @classmethod
    def tree_unflatten(cls, trajectory_steps, records):
        return cls(records, trajectory_steps)


def dataloader(dataset, batch_size=1, shuffle=False, *, drop_last=False, generator=None):
    """Batch complete trajectories and flatten each group's records.

    ``dataset`` is a nonempty tuple or list of record dictionaries. Each player
    has a boolean ``_valid`` mask of shape ``[samples]`` or ``[time, episode]``;
    all fields share those leading dimensions, including across players.
    ``batch_size`` counts episodes for rank-2 data and individual records for
    rank-1 data. A batch contains every time step of its selected episodes,
    ordered by episode and then time. Invalid records remain present so the
    training graph can apply its masks. Returned batches retain their complete
    trajectory boundaries as static metadata for optimizer microbatching;
    tuple indexing, iteration, and JAX tree transformations remain supported.

    Each traversal optionally shuffles episodes with the same order for all
    players, keeping complete trajectories and their fields aligned.
    ``generator`` is a JAX PRNG key, copied into the loader; omitting it uses
    seed 0. Repeated traversals advance
    the loader's key without changing the supplied key or dataset. The final
    batch keeps any remaining complete episodes unless ``drop_last=True``.
    """
    return _DataLoader(dataset, batch_size, shuffle, drop_last, generator)


class _DataLoader:
    def __init__(self, dataset, batch_size, shuffle, drop_last, generator):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(shuffle, bool) or not isinstance(drop_last, bool):
            raise ValueError("shuffle and drop_last must be booleans")
        if not isinstance(dataset, (tuple, list)) or not dataset:
            raise ValueError("dataset must contain one record dictionary per player")
        if isinstance(dataset, _TrajectoryBatch):
            steps = dataset.trajectory_steps
            episodes = dataset[0]["_valid"].shape[0] // steps
            dataset = tuple({name: value.reshape((episodes, steps) + value.shape[1:]).swapaxes(0, 1)
                             for name, value in records.items()} for records in dataset)

        shape = None
        players = []
        for records in dataset:
            if not isinstance(records, Mapping) or "_valid" not in records:
                raise ValueError("Each player's records must include a _valid mask")
            try:
                records = {name: jnp.asarray(value) for name, value in records.items()}
            except (TypeError, ValueError) as error:
                raise ValueError("Record fields must be arrays with matching sample dimensions") from error
            valid = records["_valid"]
            if valid.ndim not in (1, 2) or valid.dtype != jnp.bool_ or any(
                size == 0 for size in valid.shape
            ):
                raise ValueError("Expected a nonempty rank-1 or rank-2 boolean _valid mask")
            if shape is not None and valid.shape != shape:
                raise ValueError("Players must have matching sample dimensions")
            shape = valid.shape
            if any(value.shape[:valid.ndim] != shape for value in records.values()):
                raise ValueError("Record fields must have matching sample dimensions")
            size = valid.size
            players.append({name: value.reshape((size,) + value.shape[valid.ndim:])
                            for name, value in records.items()})

        key = jax.random.PRNGKey(0) if generator is None else generator
        try:
            key = jnp.array(key, copy=True)
            if jax.random.key_data(key).ndim != 1:
                raise ValueError("Expected a single key")
        except (TypeError, ValueError) as error:
            raise ValueError("generator must be a single JAX PRNG key") from error

        self._data = tuple(players)
        self._steps = shape[0] if len(shape) == 2 else 1
        self._size = shape[-1]
        self._key = key
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __len__(self):
        return (self._size + (0 if self.drop_last else self.batch_size - 1)) // self.batch_size

    def __iter__(self):
        data = self._data
        order = jnp.arange(self._size)
        if self.shuffle:
            self._key, key = jax.random.split(self._key)
            order = jax.random.permutation(key, self._size)
        indices = (order[:, None] + jnp.arange(self._steps)[None, :] * self._size).reshape(-1)
        stop = len(self) * self.batch_size if self.drop_last else self._size
        width = self.batch_size * self._steps
        def batch(start):
            selected = indices[start:start + width]
            placed = {}

            def take(value):
                # A sharded dataset and a coordinator-owned RNG must use the
                # same device set for indexing. Indices are small replicated
                # arrays; feature storage retains its distributed layout.
                sharding = value.sharding
                target = (NamedSharding(sharding.mesh, PartitionSpec())
                          if isinstance(sharding, NamedSharding) else sharding)
                if target not in placed:
                    placed[target] = jax.device_put(selected, target)
                return value[placed[target]]

            return _TrajectoryBatch(jax.tree.map(take, data), self._steps)

        return (batch(start) for start in range(0, stop * self._steps, width))
