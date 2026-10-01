"""Abstract environment interface for JAX trajectory sampling."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import jax


class Environment(ABC):
    """A game with explicit state and pure, fixed-shape JAX operations.

    Concrete games in ``LiteEFG.drl.env`` inherit this class. State is a JAX
    pytree whose structure, array shapes, and dtypes remain fixed throughout
    a trajectory. Methods must support ``jax.jit`` and ``jax.vmap``.
    Player indices are zero-based, and every episode ends within ``max_steps``.
    """

    @property
    @abstractmethod
    def num_players(self) -> int:
        """Number of players, shared by all player-indexed outputs."""
        raise NotImplementedError

    @property
    @abstractmethod
    def num_actions(self) -> int:
        """Fixed action-space size used by each player's legal-action mask."""
        raise NotImplementedError

    @property
    @abstractmethod
    def max_steps(self) -> int:
        """Maximum number of transitions needed to finish any episode."""
        raise NotImplementedError

    @abstractmethod
    def init(self, key: jax.Array) -> Any:
        """Return the initial state using the supplied JAX random key."""
        raise NotImplementedError

    @abstractmethod
    def features(self, state: Any) -> Mapping[str, jax.Array]:
        """Return named arrays with shape ``[num_players, *local_shape]``.

        The required ``information_set`` feature has shape
        ``[num_players, vector_size]`` and contains only player-visible
        information. Additional features may supply other algorithm inputs.
        """
        raise NotImplementedError

    @abstractmethod
    def legal_action_mask(self, state: Any) -> jax.Array:
        """Return a boolean array of shape ``[num_players, num_actions]``."""
        raise NotImplementedError

    @abstractmethod
    def active_players(self, state: Any) -> jax.Array:
        """Return a boolean decision mask ``[num_players]``, false at terminal."""
        raise NotImplementedError

    @abstractmethod
    def step(self, state: Any, actions: jax.Array, key: jax.Array
             ) -> tuple[Any, jax.Array, jax.Array]:
        """Return ``(next_state, reward[num_players], done)`` for one game.

        ``actions`` contains one legal action per active player; ignore
        inactive players' entries. ``done`` is a boolean scalar. Terminal
        states are absorbing and produce zero further reward.
        """
        raise NotImplementedError


__all__ = ["Environment"]
