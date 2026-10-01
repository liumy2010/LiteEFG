"""Construction scopes shared by native and JAX graphs."""

from contextvars import ContextVar
from functools import wraps
from threading import RLock
import weakref

from . import _LiteEFG as native


_current = ContextVar("liteefg_current_graph", default=None)
_constructing = ContextVar("liteefg_constructing_graphs", default=())
_status = ContextVar("liteefg_construction_status", default=(False, 0))
# Native construction state is process-wide. Serialize complete constructors;
# execution of already constructed graphs does not acquire this lock.
_construction_lock = RLock()


def current_graph():
    reference = _current.get()
    return None if reference is None else reference()


def current_status():
    """Return the lexical static flag and color, independent of node ownership."""
    return _status.get()


def scoped_status_type(native_status):
    """Keep native scope behavior while sharing its color with DRL builders."""
    class Status(native_status):
        def __init__(self, is_static=False, color=0):
            super().__init__(is_static=is_static, color=color)
            self._liteefg_status = (is_static, color)

        def __enter__(self):
            super().__enter__()
            self._liteefg_status_token = _status.set(self._liteefg_status)
            return self

        def __exit__(self, *args):
            try:
                return super().__exit__(*args)
            finally:
                _status.reset(self._liteefg_status_token)

    Status.__name__ = native_status.__name__
    Status.__qualname__ = native_status.__name__
    Status.__module__ = "LiteEFG"
    return Status


def activate(graph):
    """Select an explicitly initialized graph for new, unowned expressions."""
    if isinstance(graph, native.Graph):
        native._graph_activate(graph)
    else:
        native._graph_clear_context()
    _status.set((False, 0))
    _current.set(weakref.ref(graph))


def scope_subclass_init(cls):
    """Restore both backends after the complete most-derived constructor."""
    initialize = cls.__init__
    if getattr(initialize, "_liteefg_scoped_init", False):
        return

    @wraps(initialize)
    def scoped(self, *args, **kwargs):
        # super().__init__ calls for the same instance are part of this scope.
        if any(graph is self for graph in _constructing.get()):
            return initialize(self, *args, **kwargs)
        with _construction_lock:
            previous_native = native._graph_capture_context()
            graph_token = _current.set(None)
            stack_token = _constructing.set((*_constructing.get(), self))
            status_token = _status.set((False, 0))
            native._graph_clear_context()
            try:
                return initialize(self, *args, **kwargs)
            finally:
                try:
                    native._graph_restore_context(previous_native)
                finally:
                    _status.reset(status_token)
                    _constructing.reset(stack_token)
                    _current.reset(graph_token)

    scoped._liteefg_scoped_init = True
    cls.__init__ = scoped


def install_native_context():
    """Keep the bound native Graph type, including its pickle identity."""
    if native.Graph.__dict__.get("_liteefg_context_installed", False):
        return
    initialize = native.Graph.__init__

    @wraps(initialize)
    def init(self, *args, **kwargs):
        initialize(self, *args, **kwargs)
        activate(self)

    def init_subclass(cls, **kwargs):
        super(native.Graph, cls).__init_subclass__(**kwargs)
        scope_subclass_init(cls)

    native.Graph.__init__ = init
    native.Graph.__init_subclass__ = classmethod(init_subclass)
    native.Graph._liteefg_context_installed = True
