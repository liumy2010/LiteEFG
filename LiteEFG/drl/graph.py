"""Local, transient computation graphs lowered to pure JAX functions."""

from __future__ import annotations

from typing import Any, NamedTuple
import keyword
import math
import numbers
import operator

from .. import _LiteEFG as _native
from .._graph_context import activate, current_graph, current_status, scope_subclass_init

try:
    import jax
    import jax.numpy as jnp
    import optax
except ImportError as error:
    raise ImportError("DRLGraph requires JAX and Optax: python -m pip install LiteEFG") from error


class Node:
    """A symbolic value. Its numerical result is local to one execution.

    Feature/action axes are explicit; sample axes are supplied by the runtime.
    The graph owns immutable expression versions, so inplace preserves reads
    recorded before the write without storing an information-set table.
    """

    _liteefg_drl_node = True
    __array_priority__ = 1000
    __hash__ = object.__hash__

    def __init__(self, graph, index):
        self.graph, self.index = graph, index

    def _binary(self, other, operation):
        return operation_node(operation, self, other)

    def __add__(self, other): return self._binary(other, operator.add)
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self._binary(other, operator.sub)
    def __rsub__(self, other): return operation_node(operator.sub, other, self)
    def __mul__(self, other): return self._binary(other, operator.mul)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self._binary(other, operator.truediv)
    def __rtruediv__(self, other): return operation_node(operator.truediv, other, self)
    def __pow__(self, other): return self._binary(other, operator.pow)
    def __neg__(self): return operation_node(operator.neg, self)
    def __lt__(self, other): return self._binary(other, operator.lt)
    def __le__(self, other): return self._binary(other, operator.le)
    def __gt__(self, other): return self._binary(other, operator.gt)
    def __ge__(self, other): return self._binary(other, operator.ge)
    def __eq__(self, other): return self._binary(other, operator.eq)
    def __ne__(self, other): return self._binary(other, operator.ne)
    def __and__(self, other): return self._binary(other, operator.and_)
    def __or__(self, other): return self._binary(other, operator.or_)
    def __invert__(self): return operation_node(operator.invert, self)
    def __getitem__(self, item): return operation_node(lambda x: x[..., item], self)

    def __bool__(self):
        raise TypeError("A DRL node is symbolic; use where() for data-dependent selection")

    def copy(self): return operation_node(lambda x: x, self)
    def exp(self): return exp(self)
    def log(self): return log(self)
    def clip(self, lower, upper): return clip(self, lower, upper)
    def stop_gradient(self): return stop_gradient(self)
    def sum(self, axis=-1, keepdims=False):
        return operation_node(lambda x: jnp.sum(x, axis=axis, keepdims=keepdims), self)
    def mean(self, axis=-1, keepdims=False):
        return operation_node(lambda x: jnp.mean(x, axis=axis, keepdims=keepdims), self)
    def max(self, axis=-1, keepdims=False):
        return operation_node(lambda x: jnp.max(x, axis=axis, keepdims=keepdims), self)
    def min(self, axis=-1, keepdims=False):
        return operation_node(lambda x: jnp.min(x, axis=axis, keepdims=keepdims), self)
    def squeeze(self, axis=-1):
        return operation_node(lambda x: jnp.squeeze(x, axis=axis), self)
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return operation_node(lambda x: jnp.reshape(x, shape), self)
    def astype(self, dtype):
        return operation_node(lambda x: x.astype(dtype), self)
    def aggregate(self, aggregator="sum", object="children", player="self", padding=0.0,
                  *, discount=1.0, decay=1.0, include_self=False):
        return aggregate(self, aggregator, object, player, padding,
                         discount=discount, decay=decay, include_self=include_self)

    def inplace(self, expression):
        """Assign a new local expression version; never train a model implicitly."""
        if self.graph._sealed:
            raise RuntimeError("Cannot change a graph after initialization")
        value = self.graph.as_node(expression)
        self.index = value.index


def operation_node(operation, *arguments):
    nodes = [x for x in arguments if isinstance(x, Node)]
    if not nodes:
        return operation(*arguments)
    graph = nodes[0].graph
    dependencies = tuple(graph.as_node(x).index for x in arguments)
    return graph._append("operation", dependencies, operation)


def exp(x): return operation_node(jnp.exp, x)
def log(x): return operation_node(jnp.log, x)
def minimum(x, y): return operation_node(jnp.minimum, x, y)
def maximum(x, y): return operation_node(jnp.maximum, x, y)
def clip(x, lower, upper): return operation_node(jnp.clip, x, lower, upper)
def stop_gradient(x): return operation_node(jax.lax.stop_gradient, x)
def where(condition, x, y): return operation_node(jnp.where, condition, x, y)
def dot(x, y): return operation_node(lambda a, b: jnp.sum(a * b, axis=-1), x, y)
def mse(prediction, target): return (prediction - target) ** 2


class _Aggregate(NamedTuple):
    aggregator: str
    object: str
    padding: float
    discount: float
    decay: float
    include_self: bool


def aggregate(x, aggregator="sum", object="children", player="self", padding=0.0,
              *, discount=1.0, decay=1.0, include_self=False):
    """Aggregate a sampled trajectory when its graph color is updated.

    Children/parent read the next/previous decision of the same player and
    preserve local tensor axes. Descendants sum later decisions, optionally
    including this decision. Segment sums environment steps from this decision
    up to (excluding) the next decision, including rewards on inactive steps.
    Batch reduces all valid decisions and broadcasts the result to each sample.
    The default reduction is sum; descendants and segment support only sum.

    Discount applies once per environment step; descendants additionally apply
    decay once per own-decision transition. Empty sources return padding.
    These operations require a complete sampled trajectory in ``graph.update``.
    They never enumerate unvisited actions or information sets.
    """
    if not isinstance(x, Node):
        raise TypeError("DRL aggregate requires a graph node")
    if player != "self":
        raise ValueError("DRL aggregate supports only player='self'")
    if object not in {"children", "parent", "descendants", "segment", "batch"}:
        raise ValueError("Invalid DRL aggregate object")
    if aggregator not in {"sum", "mean", "min", "max"}:
        raise ValueError("Invalid DRL aggregator")
    if object in {"descendants", "segment"} and aggregator != "sum":
        raise ValueError(f"DRL aggregate object={object!r} supports only sum")
    for name, value in (("discount", discount), ("decay", decay), ("padding", padding)):
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
            raise ValueError(f"DRL aggregate {name} must be a finite static scalar")
        if name != "padding" and not 0 <= value <= 1:
            raise ValueError(f"DRL aggregate {name} must be between zero and one")
    if not isinstance(include_self, bool):
        raise ValueError("DRL aggregate include_self must be a bool")
    if object != "descendants" and (decay != 1 or include_self):
        raise ValueError("decay and include_self apply only to descendants aggregation")
    if object == "batch" and discount != 1:
        raise ValueError("discount does not apply to batch aggregation")
    spec = _Aggregate(aggregator, object, float(padding), float(discount), float(decay), include_self)
    return x.graph._append("aggregate", (x.index,), spec)


def _aggregate_values(values, valid, alive, spec):
    """Pure JAX reductions on [time, episode, *local_shape] arrays."""
    values = jnp.asarray(values, dtype=jnp.result_type(values, jnp.float32))
    local_axes = (1,) * (values.ndim - 2)
    def expand(mask):
        return mask.reshape(mask.shape + local_axes)

    valid = valid & alive
    padding = jnp.full_like(values[0], spec.padding)
    zeros = jnp.zeros_like(values[0])
    present = jnp.zeros(valid.shape[1:], bool)
    if spec.object == "batch":
        mask = expand(valid)
        count = valid.sum()
        if spec.aggregator in {"sum", "mean"}:
            result = jnp.where(mask, values, 0).sum(axis=(0, 1))
            if spec.aggregator == "mean":
                result = result / jnp.maximum(count, 1)
        elif spec.aggregator == "min":
            result = jnp.where(mask, values, jnp.inf).min(axis=(0, 1))
        else:
            result = jnp.where(mask, values, -jnp.inf).max(axis=(0, 1))
        result = jnp.where(count > 0, result, spec.padding)
        return jnp.broadcast_to(result, values.shape)

    if spec.object == "segment":
        def segment(pending, record):
            value, acts, live = record
            total = jnp.where(expand(live), value + spec.discount * pending, 0)
            return jnp.where(expand(acts), 0, total), jnp.where(expand(acts), total, padding)
        _, result = jax.lax.scan(segment, zeros, (values, valid, alive), reverse=True)
        return result

    def traverse(carry, record):
        accumulated, found = carry
        value, acts, live = record
        following = spec.discount * accumulated
        if spec.object == "descendants":
            following *= jnp.where(expand(acts), spec.decay, 1)
            total = following + jnp.where(expand(acts), value, 0)
            selected = total if spec.include_self else following
            exists = found | acts if spec.include_self else found
        else:
            total = jnp.where(expand(acts), value, following)
            selected, exists = following, found
        output = jnp.where(expand(acts & exists), selected, padding)
        return (jnp.where(expand(live), total, 0), live & (acts | found)), output

    _, result = jax.lax.scan(
        traverse, (zeros, present), (values, valid, alive), reverse=spec.object != "parent")
    return result


def const(size, val):
    """Create a fixed-width local vector; its scalar fill may be a graph node."""
    if isinstance(size, Node):
        raise ValueError("DRLGraph const size must be a static nonnegative integer; "
                         "a sampled action_set_size cannot define a JAX array shape")
    if isinstance(size, bool):
        raise ValueError("DRLGraph const size must be a static nonnegative integer")
    try:
        size = operator.index(size)
    except TypeError as error:
        raise ValueError("DRLGraph const size must be a static nonnegative integer") from error
    if size < 0:
        raise ValueError("DRLGraph const size must be a static nonnegative integer")
    graph = val.graph if isinstance(val, Node) else current_graph()
    if not isinstance(graph, DRLGraph):
        raise ValueError("Create a DRLGraph before creating a DRL constant")
    value = graph.as_node(val)

    def fill(scalar):
        if scalar.size != 1:
            raise ValueError("DRLGraph const val must be scalar or a length-one vector")
        return jnp.full((size,), scalar.reshape(()), dtype=scalar.dtype)

    return operation_node(fill, value)


def gather(values, actions):
    return operation_node(
        lambda v, a: jnp.take_along_axis(v, a.astype(jnp.int32)[..., None], axis=-1)[..., 0],
        values, actions,
    )


def masked_softmax(logits, mask):
    def apply(x, legal):
        # Terminal padding may have no legal actions. Return zeros there.
        safe = jnp.where(legal, x, jnp.finfo(x.dtype).min)
        probabilities = jax.nn.softmax(safe, axis=-1) * legal
        return probabilities / jnp.maximum(probabilities.sum(-1, keepdims=True), 1e-30)
    return operation_node(apply, logits, mask)


def entropy(probabilities):
    return operation_node(
        lambda p: -jnp.sum(p * jnp.log(jnp.maximum(p, 1e-30)), axis=-1), probabilities
    )


def normalize(x, p_norm=1.0, ignore_negative=False):
    def apply(value):
        value = jnp.maximum(value, 0) if ignore_negative else value
        norm = jnp.sum(jnp.abs(value) ** p_norm, axis=-1, keepdims=True) ** (1 / p_norm)
        return jnp.where(norm > 0, value / jnp.maximum(norm, 1e-30), jnp.ones_like(value) / value.shape[-1])
    return operation_node(apply, x)


class ModelState(NamedTuple):
    params: Any
    opt_state: Any
    step: Any


class Model:
    """Independent parameter and optimizer resource, usable in multiple graphs.

    Wrap a module exposing ``init(key, example_input)`` and
    ``apply(parameters, input)``, or supply those callables explicitly.
    The network architecture and implementation belong to the caller.
    Object identity determines sharing. Create distinct handles for independent
    networks, or reuse a handle across calls, players, and graph definitions.
    ``state`` is initialized once and updated at successful training boundaries;
    numerical graph execution always receives explicit parameter snapshots.
    """

    def __init__(self, module=None, *, init=None, apply=None, optimizer=None, name=None):
        if module is not None:
            if init is not None or apply is not None:
                raise TypeError("Supply either a module or explicit init/apply functions")
            init, apply = getattr(module, "init", None), getattr(module, "apply", None)
        if not callable(init) or not callable(apply):
            raise TypeError("A model requires init(key, input) and apply(params, input)")
        self.module = module
        self.init, self.apply = init, apply
        self.optimizer = optimizer if optimizer is not None else optax.adam(3e-4)
        self.name = name
        self.state = None
        self.input_spec = None

    def __call__(self, inputs):
        if not isinstance(inputs, Node):
            raise TypeError("Call a graph model with a DRLGraph input node")
        graph = inputs.graph
        index = graph._register_model(self, inputs.index)
        return graph._append("model", (inputs.index,), index)

    def minimize(self, loss):
        if not isinstance(loss, Node):
            raise TypeError("A training objective must be a graph node")
        loss.graph.minimize(loss, models=[self])


def model(module=None, *, init=None, apply=None, optimizer=None, name=None):
    """Create a reusable model handle; registration occurs when a graph calls it."""
    return Model(module, init=init, apply=apply, optimizer=optimizer, name=name)


class ModelList:
    """Immutable model handles with ordinary or symbolic indexing.

    ``models[self.player](inputs)`` selects the player's network in a graph.
    Repeating the same handle shares its parameters and optimizer. Candidate
    networks may have different architectures but must return matching shapes
    and dtypes. Player-indexed lists must match the environment's player count.
    """

    def __init__(self, models):
        self.models = tuple(models)
        if not self.models or any(not isinstance(item, Model) for item in self.models):
            raise TypeError("ModelList requires a nonempty sequence of Model handles")

    def __len__(self):
        return len(self.models)

    def __iter__(self):
        return iter(self.models)

    def __getitem__(self, index):
        if isinstance(index, Node):
            return _ModelSelection(self, index)
        return self.models[index]


class _ModelSelection:
    def __init__(self, models, selector):
        self.models, self.selector = models, selector

    def __call__(self, inputs):
        if not isinstance(inputs, Node) or inputs.graph is not self.selector.graph:
            raise ValueError("Model selection and inputs must be nodes of the same graph")
        graph = inputs.graph
        indices = tuple(graph._register_model(resource, inputs.index) for resource in self.models)
        node = graph._append("model_select", (self.selector.index, inputs.index), indices)
        if graph._expressions[self.selector.index] == ("input", (), "player"):
            if not any(existing is self.models for existing in graph.model_lists):
                graph.model_lists.append(self.models)
        return node


_RESERVED_FEATURE_NAMES = frozenset({
    "information_state", "legal_action_mask", "action", "log_sampling_prob",
    "sampling_value", "reward", "player", "valid", "advantage", "return_target",
    "old_log_prob", "old_value", "done", "policy_valid",
})


def _validate_feature_name(name):
    """Keep environment features distinct from flat runtime input records."""
    if (not isinstance(name, str) or not name.isidentifier()
            or name.startswith("_") or keyword.iskeyword(name)):
        raise ValueError(f"Invalid environment feature name: {name!r}")
    if name in _RESERVED_FEATURE_NAMES:
        raise ValueError(f"Environment feature name {name!r} is reserved")


class _FeatureNamespace:
    """Read-only symbolic environment inputs, without live environment state."""

    __slots__ = ("_graph", "_inputs")

    def __init__(self, graph):
        self._graph = graph
        self._inputs = {"information_set": graph._append("input", (), "information_set")}

    def __getattr__(self, name):
        try:
            _validate_feature_name(name)
        except ValueError as error:
            raise AttributeError(str(error)) from None
        if name in self._inputs:
            return self._inputs[name]
        graph = self._graph
        if graph._sealed:
            raise AttributeError(f"Environment feature {name!r} was not declared before graph initialization")
        node = graph._append("input", (), name)
        graph._custom_inputs[name] = node
        self._inputs[name] = node
        return node


class DRLGraph:
    """An infoset-local program with transient values and explicit models.

    No numerical values are kept per information set. ``evaluate`` lowers
    only the dependencies of the requested outputs to JAX operations. The
    trainer supplies sample dimensions and masked loss reductions. Attributes
    of ``self.env`` declare environment inputs during construction; their local
    shapes and dtypes are bound by ``init`` from one example record.
    """

    _core_input_ranks = {
        "information_set": 1, "legal_action_mask": 1, "action": 0,
        "log_sampling_prob": 0, "sampling_value": 0,
        "reward": 0, "player": 0,
    }
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        scope_subclass_init(cls)

    def __init__(self):
        self._expressions = []
        self._colors = []
        self.models = []
        self.model_lists = []
        self._model_indices = {}
        self._model_inputs = []
        self.objectives = []
        self.metrics = {}
        self._plans = {}
        self._node_specs = {}
        self._sealed = False
        self._trainer = None
        self._custom_inputs = {}
        self.input_specs = {}
        activate(self)
        self._env = _FeatureNamespace(self)
        for name in self._core_input_ranks:
            if name != "information_set":
                setattr(self, name, self._append("input", (), name))
        self.action_set_size = operation_node(
            lambda mask: mask.astype(jnp.float32).sum(-1), self.legal_action_mask
        )
        # Like inputs, this built-in feature is available in every color.
        self._colors[self.action_set_size.index] = None

    @property
    def env(self):
        """Read-only namespace for ``information_set`` and named environment features."""
        return self._env

    @property
    def trainer(self):
        """The training state owned by this graph after bind()."""
        if self._trainer is None:
            raise RuntimeError("Call graph.bind(env, ...) before accessing its trainer or training")
        return self._trainer

    def bind(self, env, *, batch_size=1024, seed=0, devices=None, parallel=None,
             sampling_devices=None, compact_batches=True):
        """Bind one environment and initialize this graph's training state.

        Call after constructing the complete graph. A graph can be bound once;
        create a separate graph instance for an independent training run.
        Return this graph so binding can be chained with its construction.
        ``compact_batches`` omits invalid rows from optimizer model evaluation
        while preserving each original minibatch's valid examples and weights.
        ``devices`` selects local JAX devices (one default device when omitted).
        ``parallel`` selects ``sampling`` and/or ``optimizing``; both are enabled
        by default. An empty sequence executes both stages on the first device.
        ``sampling_devices`` selects the sampling-only interface and cannot be
        combined with ``devices`` or ``parallel``.
        """
        from .runtime import Trainer
        Trainer(self, env, batch_size=batch_size, seed=seed,
                devices=devices, parallel=parallel,
                sampling_devices=sampling_devices, compact_batches=compact_batches)
        return self

    def __getattr__(self, name):
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}; "
                             "access environment features through self.env")

    def _append(self, kind, dependencies, operation):
        if self._sealed:
            raise RuntimeError("Construct the complete DRLGraph before initializing or compiling it")
        is_static, color = current_status()
        if kind in {"input", "constant"}:
            color = None
        else:
            if is_static:
                raise ValueError("DRL graph scopes do not support is_static=True; update a color explicitly")
            if isinstance(color, bool) or not isinstance(color, numbers.Integral) or color < 0:
                raise ValueError("DRL graph color must be a nonnegative integer")
            color = int(color)
        index = len(self._expressions)
        self._expressions.append((kind, dependencies, operation))
        self._colors.append(color)
        return Node(self, index)

    def _selected_colors(self, upd_color):
        """Normalize native-style color selectors; -1 selects every color."""
        if isinstance(upd_color, numbers.Integral) and not isinstance(upd_color, bool):
            upd_color = (upd_color,)
        try:
            colors = tuple(upd_color)
        except TypeError as error:
            raise ValueError("upd_color must be an integer or a sequence of integers") from error
        if any(isinstance(color, bool) or not isinstance(color, numbers.Integral)
               or color < -1 for color in colors):
            raise ValueError("upd_color must contain nonnegative integers or -1")
        if -1 in colors:
            return frozenset(color for color in self._colors if color is not None)
        return frozenset(int(color) for color in colors)

    def _normalize_colors(self, upd_color):
        return tuple(sorted(self._selected_colors(upd_color)))

    def as_node(self, value):
        if isinstance(value, Node):
            if value.graph is not self:
                raise ValueError("Cannot mix nodes from different graphs")
            return value
        if isinstance(value, _native.GraphNode):
            raise TypeError("Cannot mix native Graph and DRLGraph nodes")
        return self._append("constant", (), jnp.asarray(value))

    def minimize(self, loss, models=None):
        if self._sealed:
            raise RuntimeError("Cannot add objectives after graph initialization")
        node = self.as_node(loss)
        selected = tuple(dict.fromkeys(self.models if models is None else models))
        if not selected or any(resource not in self._model_indices for resource in selected):
            raise ValueError("An objective needs models called by this graph")
        # Capture the expression version, not the mutable Python node handle.
        self.objectives.append((Node(self, node.index), tuple(self.model_index(m) for m in selected)))

    def _register_model(self, resource, input_index):
        if self._sealed:
            raise RuntimeError("Cannot add model calls after graph initialization")
        if resource not in self._model_indices:
            self._model_indices[resource] = len(self.models)
            self.models.append(resource)
            self._model_inputs.append([])
        index = self._model_indices[resource]
        if input_index not in self._model_inputs[index]:
            self._model_inputs[index].append(input_index)
        return index

    def model_index(self, resource):
        """Return a handle's index in this graph's unique parameter table."""
        try:
            return self._model_indices[resource]
        except KeyError:
            raise ValueError("Model has not been called by this graph") from None

    def current_strategy(self):
        return self.strategy

    def current_value(self):
        return self.value

    def train(self, **kwargs):
        """Optional algorithm-defined schedule using self.trainer.

        Subclasses may provide their own training arguments and color schedule.
        Graphs without this method's implementation remain usable through
        collect(), update(), and optimize(), with dataloader for batching.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define a training schedule; "
            "use self.trainer's execution methods explicitly"
        )

    def _plan(self, indices, cached=()):
        indices = tuple(indices)
        cached = frozenset(cached)
        key = (indices, cached)
        if key not in self._plans:
            required = set()
            def visit(index):
                if index in required:
                    return
                required.add(index)
                if index in cached:
                    return
                for dependency in self._expressions[index][1]:
                    visit(dependency)
            for index in indices:
                visit(index)
            self._plans[key] = tuple(sorted(required))
        return self._plans[key]

    @staticmethod
    def _node_key(index):
        return f"_drl_node_{index}"

    def _cached_nodes(self, inputs, colors=None):
        cached = {}
        for index in range(len(self._expressions)):
            if colors is not None and self._colors[index] in colors:
                continue
            name = self._node_key(index)
            if name in inputs:
                cached[index] = name
        return cached

    def _color_plan(self, indices, cached, colors):
        plan = self._plan(indices, cached)
        if colors is not None:
            for index in plan:
                color = self._colors[index]
                if color is not None and color not in colors and index not in cached:
                    raise ValueError(f"Graph node {index} requires cached color {color}; "
                                     "update that color before executing dependent colors")
        return plan

    def update(self, params, inputs, *, upd_color, valid=None, alive=None):
        """Execute selected colors and return inputs with detached node values.

        All expressions in selected colors run in construction order. A value
        from any other color must already be cached in ``inputs``. Selecting a
        color again refreshes its values, including ordinary expressions and
        model calls. Aggregates require complete [time, episode] trajectories.
        """
        colors = self._selected_colors(upd_color)
        indices = tuple(index for index, color in enumerate(self._colors) if color in colors)
        cached = self._cached_nodes(inputs, colors)
        plan = self._color_plan(indices, cached, colors)
        has_aggregates = any(self._expressions[index][0] == "aggregate" for index in indices)
        batch_shape = None
        if has_aggregates or valid is not None or alive is not None:
            if valid is None or alive is None:
                raise ValueError("Trajectory aggregates require valid/alive masks in graph.update")
            valid, alive = jnp.asarray(valid), jnp.asarray(alive)
            if (valid.ndim != 2 or alive.shape != valid.shape or 0 in valid.shape
                    or valid.dtype != jnp.bool_ or alive.dtype != jnp.bool_):
                raise ValueError("Trajectory valid/alive masks must be nonempty boolean [time, episode] arrays")
            batch_shape = valid.shape
        required = {self._expressions[index][2] for index in plan
                    if self._expressions[index][0] == "input"}
        required.update(cached[index] for index in plan if index in cached)
        _, batch_shape = self._local_records(inputs, required, batch_shape)
        prepared = dict(inputs)
        for index in indices:
            prepared.pop(self._node_key(index), None)
        for index in indices:
            kind, dependencies, operation = self._expressions[index]
            if kind == "aggregate":
                source, = self._evaluate(params, prepared, (Node(self, dependencies[0]),),
                                         _batch_shape=batch_shape)
                result = _aggregate_values(source, valid, alive, operation)
            else:
                result, = self._evaluate(params, prepared, (Node(self, index),),
                                         _batch_shape=batch_shape)
            result = jax.lax.stop_gradient(result)
            name = self._node_key(index)
            spec = jax.ShapeDtypeStruct(result.shape[len(batch_shape):], result.dtype)
            previous = self._node_specs.get(name)
            if previous is not None and (previous.shape != spec.shape or previous.dtype != spec.dtype):
                raise ValueError("A cached graph node must retain its local shape and dtype")
            self._node_specs[name] = spec
            prepared[name] = result
        return prepared

    def evaluate(self, params, inputs, outputs, *, upd_color=None):
        """Evaluate local outputs, optionally recomputing only selected colors.

        A selected color ignores its own cached values. Other colors are read
        from explicit node caches without executing their dependencies. Without
        a selector, inference traverses ordinary dependencies and uses any
        available caches. Aggregates execute only in ``update``.
        """
        return self._evaluate(params, inputs, outputs, upd_color=upd_color)

    def model_usage(self, params, inputs, *, upd_color=None):
        """Return one participation mask per resource for selected objectives.

        A sample counts once even if a handle is called several times. Cached
        detached expressions do not contribute, and a selected model counts
        only on records routed to one of its slots.
        """
        colors = None if upd_color is None else self._selected_colors(upd_color)
        cached = self._cached_nodes(inputs, colors)
        _, batch_shape = self._local_records(inputs, set())
        usage = [jnp.zeros(batch_shape, bool) for _ in self.models]
        selectors = {}
        for objective, selected in self.objectives:
            if colors is not None and self._colors[objective.index] not in colors:
                continue
            for index in self._color_plan((objective.index,), cached, colors):
                if index in cached:
                    continue
                kind, dependencies, operation = self._expressions[index]
                if kind == "model" and operation in selected:
                    usage[operation] = jnp.ones(batch_shape, bool)
                elif kind == "model_select":
                    selector_index = dependencies[0]
                    if selector_index not in selectors:
                        if ("_model_player" in inputs
                                and self._expressions[selector_index] == ("input", (), "player")):
                            selectors[selector_index] = jnp.asarray(inputs["_model_player"])
                        else:
                            selectors[selector_index], = self._evaluate(
                                params, inputs, (Node(self, selector_index),), upd_color=upd_color)
                    selector = selectors[selector_index]
                    for slot, model_index in enumerate(operation):
                        if model_index in selected:
                            usage[model_index] = usage[model_index] | (selector == slot)
        return tuple(usage)

    def _evaluate(self, params, inputs, outputs, *, upd_color=None, _batch_shape=None):
        if any(not isinstance(node, Node) or node.graph is not self for node in outputs):
            raise ValueError("Outputs must be nodes of this graph")
        indices = tuple(node.index for node in outputs)
        colors = None if upd_color is None else self._selected_colors(upd_color)
        cached = self._cached_nodes(inputs, colors)
        plan = self._color_plan(indices, cached, colors)
        unresolved = [index for index in plan
                      if self._expressions[index][0] == "aggregate" and index not in cached]
        if unresolved:
            raise ValueError("Trajectory aggregate requires prepared inputs from an earlier color update; "
                             "use graph.update to execute its color, not local evaluation")
        required_inputs = {self._expressions[index][2] for index in plan
                           if self._expressions[index][0] == "input" and index not in cached}
        required_inputs.update(cached[index] for index in plan if index in cached)
        if "_model_player" in inputs and any(
                self._expressions[index][0] == "model_select"
                and self._expressions[self._expressions[index][1][0]] == ("input", (), "player")
                for index in plan if index not in cached):
            required_inputs.add("_model_player")
        local_inputs, batch_shape = self._local_records(inputs, required_inputs, _batch_shape)

        def execute(record):
            values = {}
            for index in plan:
                kind, dependencies, operation = self._expressions[index]
                if index in cached:
                    values[index] = jax.lax.stop_gradient(record[cached[index]])
                elif kind == "input":
                    values[index] = record[operation]
                elif kind == "constant":
                    values[index] = operation
                elif kind == "model":
                    values[index] = self.models[operation].apply(params[operation], values[dependencies[0]])
                elif kind == "model_select":
                    selector = values[dependencies[0]]
                    if self._expressions[dependencies[0]] == ("input", (), "player"):
                        selector = record.get("_model_player", selector)
                    if selector.ndim != 0 or not jnp.issubdtype(selector.dtype, jnp.integer):
                        raise ValueError("ModelList index must be a scalar integer")
                    branches = tuple(
                        lambda x, resource=self.models[i], weights=params[i]: resource.apply(weights, x)
                        for i in operation)
                    output = jax.lax.switch(selector, branches, values[dependencies[1]])
                    # lax.switch clamps out-of-range indices. Expose invalid
                    # dynamic indices instead of silently choosing a network.
                    values[index] = jnp.where((selector >= 0) & (selector < len(operation)),
                                              output, jnp.full_like(output, jnp.nan))
                else:
                    values[index] = operation(*(values[dependency] for dependency in dependencies))
            return tuple(values[index] for index in indices)

        for size in reversed(batch_shape):
            execute = jax.vmap(execute, axis_size=size)
        return execute(local_inputs)

    def _local_records(self, inputs, required_inputs, batch_shape=None):
        missing = required_inputs.difference(inputs)
        if missing:
            raise ValueError(f"Missing graph inputs: {', '.join(sorted(missing))}")
        local_inputs = {name: jnp.asarray(inputs[name]) for name in required_inputs}
        # All operators have single-infoset semantics. vmap supplies sample
        # axes, including scalar/vector broadcasting that ordinary [B,D]*[B]
        # array arithmetic would either reject or silently misinterpret.
        feature_inputs = {"information_set", "legal_action_mask"}
        batch_shape = () if batch_shape is None else tuple(batch_shape)
        # Observation/mask axes also define the batch when the selected output
        # is constant or depends only on a scalar such as the player seat.
        shape_inputs = dict(local_inputs)
        shape_inputs.update({name: jnp.asarray(inputs[name]) for name in feature_inputs if name in inputs})
        local_ranks = {}
        for name, value in shape_inputs.items():
            if name == "_model_player":
                spec = None
            elif name.startswith("_drl_node_"):
                spec = self._node_specs.get(name)
            else:
                spec = self.input_specs.get(name)
            if spec is not None:
                local_rank = len(spec.shape)
                suffix = value.shape[-local_rank:] if local_rank else ()
                if value.ndim < local_rank or suffix != spec.shape:
                    raise ValueError(f"Graph input {name!r} must have local shape {spec.shape}; "
                                     f"received {value.shape}")
                if value.dtype != spec.dtype:
                    raise ValueError(f"Graph input {name!r} must have dtype {spec.dtype}; "
                                     f"received {value.dtype}")
            elif name in self._core_input_ranks or name == "_model_player":
                local_rank = self._core_input_ranks.get(name, 0)
                if value.ndim < local_rank:
                    raise ValueError(f"Graph input {name!r} requires {local_rank} local dimension(s)")
            else:
                raise ValueError(f"Graph input {name!r} has no local shape; call graph.init first")
            local_ranks[name] = local_rank
            prefix = value.shape[:-local_rank] if local_rank else value.shape
            if prefix:
                if batch_shape and batch_shape != prefix:
                    raise ValueError("Graph inputs have inconsistent sample dimensions")
                batch_shape = prefix
        if batch_shape:
            for name, value in local_inputs.items():
                rank = local_ranks[name]
                if value.ndim == rank:
                    local_inputs[name] = jnp.broadcast_to(value, batch_shape + value.shape)

        return local_inputs, batch_shape

    def init(self, key, example_inputs):
        """Initialize new resources and reuse existing ones from any graph.

        Each custom input is required, including declarations outside model
        dependencies. The standard information_set vector is bound when
        provided. Every call site must preserve the handle's local input shape
        and dtype. Resources are published only after all checks succeed.
        """
        inference_nodes = []
        for method in (self.current_strategy, self.current_value):
            try:
                node = method()
            except AttributeError:
                continue
            if isinstance(node, Node):
                inference_nodes.append(node.index)
        inference_nodes.extend(index for indices in self._model_inputs for index in indices)
        if any(self._expressions[index][0] == "aggregate" for index in self._plan(inference_nodes)):
            raise ValueError("Model initialization and sampling strategy/value cannot depend on trajectory aggregates")
        missing = self._custom_inputs.keys() - example_inputs.keys()
        if missing:
            raise ValueError(f"Missing example graph inputs: {', '.join(sorted(missing))}")
        for name, rank in self._core_input_ranks.items():
            if name in example_inputs and jnp.asarray(example_inputs[name]).ndim != rank:
                raise ValueError(f"Example graph input {name!r} must be one local record "
                                 f"with rank {rank}, without sample dimensions")
        specs = {}
        names = list(self._custom_inputs)
        if "information_set" in example_inputs:
            names.append("information_set")
        for name in names:
            example = jnp.asarray(example_inputs[name])
            spec = jax.ShapeDtypeStruct(example.shape, example.dtype)
            previous = self.input_specs.get(name)
            if previous is not None and (spec.shape != previous.shape or spec.dtype != previous.dtype):
                raise ValueError(f"Example graph input {name!r} must retain shape {previous.shape} "
                                 f"and dtype {previous.dtype}; received {spec.shape}, {spec.dtype}")
            specs[name] = spec
        # Model input expressions may already depend on these shapes while
        # their parameters are being initialized in dependency order.
        previous_specs = dict(self.input_specs)
        self.input_specs.update(specs)
        previous_states = tuple(resource.state for resource in self.models)
        states, model_specs = [], []
        keys = jax.random.split(key, max(1, len(self.models)))
        try:
            for index, resource in enumerate(self.models):
                inputs, = self.evaluate(tuple(state.params for state in states), example_inputs,
                                        (Node(self, self._model_inputs[index][0]),))
                spec = jax.ShapeDtypeStruct(inputs.shape, inputs.dtype)
                if resource.input_spec is not None and resource.input_spec != spec:
                    raise ValueError(f"Model input must retain shape and dtype {resource.input_spec}; received {spec}")
                state = previous_states[index]
                if state is None:
                    params = resource.init(keys[index], inputs)
                    state = ModelState(params, resource.optimizer.init(params), jnp.array(0, jnp.int32))
                states.append(state)
                model_specs.append(spec)
            params = tuple(state.params for state in states)
            for index, call_inputs in enumerate(self._model_inputs):
                for input_index in call_inputs:
                    value, = jax.eval_shape(lambda: self.evaluate(params, example_inputs,
                                                                 (Node(self, input_index),)))
                    if value.shape != model_specs[index].shape or value.dtype != model_specs[index].dtype:
                        raise ValueError(f"All calls to a model must have the same input shape and dtype; "
                                         f"expected {model_specs[index]}, received {value}")
            for index, (kind, _, _) in enumerate(self._expressions):
                if kind == "model_select":
                    jax.eval_shape(lambda: self.evaluate(params, example_inputs, (Node(self, index),)))
            if any(resource.state is not previous
                   for resource, previous in zip(self.models, previous_states)):
                raise RuntimeError("A model resource changed during initialization; initialization was not committed")
        except Exception:
            self.input_specs = previous_specs
            raise
        for resource, state, spec, previous in zip(self.models, states, model_specs, previous_states):
            if previous is None:
                resource.state = state
            resource.input_spec = spec
        self._sealed = True
        return tuple(states)


def minimize(loss, *, models):
    if not isinstance(loss, Node):
        raise TypeError("A training objective must be a graph node")
    loss.graph.minimize(loss, models=models)
