"""A small fully connected network for DRL examples."""

from numbers import Integral

from flax import linen as nn
from jax import Array
import jax.numpy as jnp
from jax.typing import ArrayLike


class MLP(nn.Module):
    """Two ReLU hidden layers followed by a linear output layer.

    The final input dimension contains features; leading batch dimensions
    are preserved. Use one output for a value model or one per action for
    policy logits.
    """

    output_size: int
    hidden_size: int = 128

    @nn.compact
    def __call__(self, inputs: ArrayLike) -> Array:
        for name in ("output_size", "hidden_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        values = jnp.asarray(inputs, dtype=jnp.float32)
        values = nn.relu(nn.Dense(self.hidden_size)(values))
        values = nn.relu(nn.Dense(self.hidden_size)(values))
        return nn.Dense(self.output_size)(values)
