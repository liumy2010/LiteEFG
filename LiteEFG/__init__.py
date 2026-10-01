from ._LiteEFG import *
from . import _LiteEFG as _native
from ._graph_context import current_graph as _current_graph
from ._graph_context import install_native_context as _install_native_context
from ._graph_context import scoped_status_type as _scoped_status_type

_install_native_context()

from .src.Environment.OpenSpiel.OpenSpielToGameFile import OpenSpielEnv
from . import baselines
from . import random
from .checkpoint import load_checkpoint, save_checkpoint
from .cpp_game import CppEnv, compile_cpp_game
from functools import wraps as _wraps


def _dispatch_operator(name, native):
    @_wraps(native)
    def dispatch(*args, **kwargs):
        values = (*args, *kwargs.values())
        has_drl = any(getattr(value, "_liteefg_drl_node", False) for value in values)
        has_native = any(isinstance(value, _native.GraphNode) for value in values)
        if has_drl and has_native:
            raise TypeError("Cannot mix native Graph and DRLGraph nodes")
        active = _current_graph()
        use_current = (not has_drl and not has_native and active is not None
                       and not isinstance(active, _native.Graph))
        if not has_drl and not use_current:
            return native(*args, **kwargs)
        if use_current:
            # Only operands become nodes. Options such as p_norm remain static.
            count = 2 if name in {"minimum", "maximum", "dot"} else 1
            args = tuple(active.as_node(value) if index < count else value
                         for index, value in enumerate(args))
            kwargs = {key: active.as_node(value) if key in
                      ({"arg0", "x", "arg1", "y"} if count == 2 else {"arg0", "x"})
                      else value for key, value in kwargs.items()}
        if name in {"copy", "sum", "mean", "min", "max"}:
            def apply(arg0):
                return getattr(arg0, name)()
            return apply(*args, **kwargs)
        from .drl import graph
        # pybind's unnamed arguments are exposed as arg0/arg1; preserve those
        # spellings when the same public operation receives a DRL node.
        translated = dict(kwargs)
        for source, target in (("arg0", "x"), ("arg1", "y")):
            if source in translated:
                if target in translated:
                    raise TypeError(f"{name} received duplicate values for {target}")
                translated[target] = translated.pop(source)
        return getattr(graph, name)(*args, **translated)
    return dispatch


for _operator_name in ("exp", "log", "minimum", "maximum", "dot", "normalize",
                       "copy", "sum", "mean", "min", "max"):
    globals()[_operator_name] = _dispatch_operator(_operator_name, globals()[_operator_name])
del _operator_name


forward = _scoped_status_type(_native.forward)
backward = _scoped_status_type(_native.backward)


def const(size, val):
    """Create a vector in the operands' graph, or the active construction graph."""
    values = (size, val)
    has_drl = any(getattr(value, "_liteefg_drl_node", False) for value in values)
    has_native = any(isinstance(value, _native.GraphNode) for value in values)
    if has_drl and has_native:
        raise TypeError("Cannot mix native Graph and DRLGraph nodes")
    active = _current_graph()
    if has_drl or (not has_native and active is not None
                   and not isinstance(active, _native.Graph)):
        from .drl.graph import const as drl_const
        return drl_const(size, val)
    return _native.const(size, val)


def aggregate(arg0, aggregator="sum", object="children", player="self", padding=0.0,
              **kwargs):
    """Aggregate native information sets or a DRL sampled trajectory."""
    values = (arg0, padding, *kwargs.values())
    has_drl = any(getattr(value, "_liteefg_drl_node", False) for value in values)
    has_native = any(isinstance(value, _native.GraphNode) for value in values)
    if has_drl and has_native:
        raise TypeError("Cannot mix native Graph and DRLGraph nodes")
    active = _current_graph()
    if has_drl or (not has_native and active is not None
                   and not isinstance(active, _native.Graph)):
        from .drl.graph import aggregate as drl_aggregate
        if not getattr(arg0, "_liteefg_drl_node", False):
            arg0 = active.as_node(arg0)
        return drl_aggregate(arg0, aggregator, object, player, padding, **kwargs)
    return _native.aggregate(arg0, aggregator, object, player, padding, **kwargs)


def __getattr__(name):
    # Keep JAX optional and preserve import time for the tabular package.
    if name in {"DRLGraph", "Model", "ModelList", "model", "Trainer", "Policy", "PackedRollout", "Goofspiel", "DarkChess",
                "average_utility", "uniform_policy", "dataloader", "minimize", "clip",
                "stop_gradient", "masked_softmax", "entropy", "gather", "mse", "where"}:
        from . import drl
        value = getattr(drl, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
