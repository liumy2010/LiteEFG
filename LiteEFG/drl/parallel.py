"""Internal device layouts for globally shaped trajectory and training arrays."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec


class _ParallelLayout:
    """Replicate model state and partition complete episodes or sample rows."""

    def __init__(self, devices):
        self.devices = tuple(devices)
        self.mesh = Mesh(np.asarray(self.devices, dtype=object), ("devices",))
        self.replicated = NamedSharding(self.mesh, PartitionSpec())
        self.trajectories = NamedSharding(self.mesh, PartitionSpec(None, "devices"))
        self.samples = NamedSharding(self.mesh, PartitionSpec("devices"))

    @staticmethod
    def put(tree, sharding):
        """Place every array leaf using a prefix partition specification."""
        return jax.tree_util.tree_map(lambda value: jax.device_put(value, sharding), tree)

    def trajectory_sharding(self, episodes):
        """Choose episode partitioning when the global shape divides evenly."""
        return self.trajectories if episodes % len(self.devices) == 0 else self.replicated

    def join_trajectories(self, shards):
        """Join per-device record trees without gathering their episode arrays.

        ``shards`` follows ``self.devices`` order. Each leaf has shape
        ``[time, local_episodes, ...]`` with equal local shapes across devices.
        The returned leaves have their usual global ``[time, episodes, ...]``
        shape and retain each device's existing episode block.
        """
        if len(shards) != len(self.devices):
            raise ValueError("Expected one trajectory shard per layout device")

        def join(*arrays):
            shape = arrays[0].shape
            if len(shape) < 2 or any(array.shape != shape for array in arrays):
                raise ValueError("Trajectory shards must have matching [time, episode, ...] shapes")
            global_shape = (shape[0], shape[1] * len(self.devices), *shape[2:])
            by_device = dict(zip(self.devices, arrays))
            local_arrays = [jax.device_put(by_device[device], device)
                            for device in self.trajectories.addressable_devices_indices_map(global_shape)]
            return jax.make_array_from_single_device_arrays(
                global_shape, self.trajectories, local_arrays)

        return jax.tree_util.tree_map(join, *shards)

    def minibatch(self, records):
        """Pad and constrain flattened records before model evaluation in a JIT.

        Padding changes neither valid sample membership nor loss weights.
        Padded fields repeat a valid representative when one exists, while
        ``_valid`` and ``_alive`` are false. The leading sample dimension is
        partitioned; all trailing feature dimensions remain local to a sample.
        """
        result = []
        for row in records:
            padding = (-row["_valid"].shape[0]) % len(self.devices)
            if padding:
                representative = jnp.argmax(row["_valid"])
                padded = {}
                for name, value in row.items():
                    shape = (padding, *value.shape[1:])
                    extra = (jnp.zeros(shape, dtype=value.dtype)
                             if name in ("_valid", "_alive") else
                             jnp.broadcast_to(value[representative], shape))
                    padded[name] = jnp.concatenate((value, extra), axis=0)
                row = padded
            result.append(jax.tree_util.tree_map(
                lambda value: jax.lax.with_sharding_constraint(value, self.samples), row))
        return tuple(result)
