"""Save and resume complete LiteEFG training state.

Checkpoint files use pickle and must only be loaded from trusted sources. The
graph's class must remain importable, with compatible code, when loading. Native
random streams require the same platform, runtime and C++ standard library.
"""

from __future__ import annotations

import os
import pickle
import platform
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

from . import _LiteEFG
from ._graph_context import activate as _activate_graph

__all__ = ["save_checkpoint", "load_checkpoint"]

_MAGIC = b"LiteEFG checkpoint\x00"
_FORMAT_VERSION = 1


def _runtime():
    return {
        "python": tuple(sys.version_info[:2]),
        "platform": sys.platform,
        "machine": platform.machine(),
        "numpy": np.__version__,
    }


def save_checkpoint(path, env, graph):
    """Atomically save a graph and its initialized environment to ``path``.

    Saves operations, per-information-set values, strategy histories, Python
    graph attributes (including baseline counters), and the global Python,
    NumPy and native random states. Call after completing an update. The source
    game file is included through the native game tree and is not needed to
    resume. Existing checkpoint files are replaced only after a successful save.
    """
    if not isinstance(env, _LiteEFG.Environment):
        raise TypeError("env must be a LiteEFG Environment")
    if not isinstance(graph, _LiteEFG.Graph):
        raise TypeError("graph must be a LiteEFG Graph")

    payload = {
        "format_version": _FORMAT_VERSION,
        "runtime": _runtime(),
        "environment": _LiteEFG._checkpoint_save_environment(env, graph),
        "graph": pickle.dumps(graph, protocol=pickle.HIGHEST_PROTOCOL),
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "native_random": _LiteEFG._checkpoint_get_random_state(),
    }
    path = Path(path)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="wb", dir=path.parent, prefix=f".{path.name}.",
                suffix=".tmp", delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(_MAGIC)
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def load_checkpoint(path):
    """Return ``(env, graph)`` restored from a trusted checkpoint file.

    The graph retains its Python subclass and attributes without rerunning its
    constructor. The environment is a native ``Environment``: training and
    strategy inspection work without the original game file, but OpenSpiel
    adapter methods and Python environment attributes are not restored. Global
    Python, NumPy and native random streams resume from the saved point.

    Do not call ``env.set_graph(graph)`` afterwards; that would reset the state.
    Checkpoints require compatible graph code and the same runtime as the save.
    Invalid or unsupported files raise ``ValueError``; filesystem errors retain
    their usual exception types.
    """
    with open(path, "rb") as stream:
        if stream.read(len(_MAGIC)) != _MAGIC:
            raise ValueError("Not a LiteEFG checkpoint file")
        try:
            payload = pickle.load(stream)
        except (pickle.UnpicklingError, EOFError) as error:
            raise ValueError("Invalid or truncated LiteEFG checkpoint") from error
        if stream.read(1):
            raise ValueError("Unexpected trailing data in LiteEFG checkpoint")

    if not isinstance(payload, dict):
        raise ValueError("Invalid LiteEFG checkpoint payload")
    if payload.get("format_version") != _FORMAT_VERSION:
        raise ValueError("Unsupported LiteEFG checkpoint format version")
    if payload.get("runtime") != _runtime():
        raise ValueError("Checkpoint runtime does not match this Python/NumPy/platform runtime")
    required = {"environment", "graph", "python_random", "numpy_random", "native_random"}
    if not required.issubset(payload):
        raise ValueError("Incomplete LiteEFG checkpoint payload")

    # Validate Python RNG states privately before changing any global stream.
    random.Random().setstate(payload["python_random"])
    np.random.RandomState().set_state(payload["numpy_random"])
    old_python = random.getstate()
    old_numpy = np.random.get_state()
    old_native = _LiteEFG._checkpoint_get_random_state()
    try:
        graph = pickle.loads(payload["graph"])
        if not isinstance(graph, _LiteEFG.Graph):
            raise ValueError("Checkpoint does not contain a LiteEFG Graph")
        env = _LiteEFG._checkpoint_load_environment(payload["environment"])
        _LiteEFG._checkpoint_set_random_state(payload["native_random"])
        random.setstate(payload["python_random"])
        np.random.set_state(payload["numpy_random"])
        graph._checkpoint_environment = env
        _activate_graph(graph)
    except Exception:
        random.setstate(old_python)
        np.random.set_state(old_numpy)
        _LiteEFG._checkpoint_set_random_state(old_native)
        raise
    return env, graph


def _graph_save_checkpoint(self, path, env):
    """Save this graph and ``env``; see :func:`save_checkpoint`."""
    save_checkpoint(path, env, self)


def _associated_environment(graph):
    env = getattr(graph, "_checkpoint_environment", None)
    if not isinstance(env, _LiteEFG.Environment):
        raise ValueError("Call env.set_graph(graph) before saving or loading a checkpoint")
    return env


def _graph_save(self, path):
    """Atomically save this graph, its environment and random states to ``path``.

    Associate the graph with its environment using ``env.set_graph(graph)``
    first. Only load checkpoints produced by trusted code.
    """
    save_checkpoint(path, _associated_environment(self), self)


def _graph_load(self, path):
    """Restore this graph and its environment in place, and return this graph.

    The saved Python graph class, native operations and game tree must match
    this graph and its associated environment. A mismatch raises ``ValueError``
    before training state is changed. Python counters, values, strategy
    histories and global random streams resume from the saved point. The graph
    constructor is not rerun. Only load trusted checkpoint files.
    """
    env = _associated_environment(self)
    old_python = random.getstate()
    old_numpy = np.random.get_state()
    old_native = _LiteEFG._checkpoint_get_random_state()
    try:
        saved_env, saved_graph = load_checkpoint(path)
        if type(saved_graph) is not type(self):
            raise ValueError("Checkpoint graph class does not match this graph")
        _LiteEFG._checkpoint_restore(env, self, saved_env, saved_graph)
        attributes = saved_graph.__dict__.copy()
        attributes["_checkpoint_environment"] = env
        _LiteEFG._graph_bind_attributes(self, attributes)
        self.__dict__.clear()
        self.__dict__.update(attributes)
        _activate_graph(self)
    except Exception:
        random.setstate(old_python)
        np.random.set_state(old_numpy)
        _LiteEFG._checkpoint_set_random_state(old_native)
        _activate_graph(self)
        raise
    return self


_LiteEFG.Graph.save = _graph_save
_LiteEFG.Graph.load = _graph_load
_LiteEFG.Graph.save_checkpoint = _graph_save_checkpoint
_LiteEFG.Graph.load_checkpoint = staticmethod(load_checkpoint)
