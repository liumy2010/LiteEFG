"""Checkpoint restoration preserves ownership inside stored Python containers."""

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import LiteEFG as leg
from LiteEFG import _LiteEFG as native


GAME = Path(__file__).resolve().parents[1] / "LiteEFG/game_instances/kuhn.game"


class AttributeContainer:
    def __init__(self, node):
        self.node = node
        self.cycle = self

    @property
    def computed(self):
        raise AssertionError("Binding checkpoint nodes must not evaluate properties")


class SlotBase:
    __slots__ = ("node", "__private")

    def __init__(self, node):
        self.node = node
        self.__private = node

    def private_node(self):
        return self.__private


class SlotContainer(SlotBase):
    __slots__ = ("cycle", "unassigned")

    def __init__(self, node):
        super().__init__(node)
        self.cycle = self

    @property
    def computed(self):
        raise AssertionError("Binding checkpoint slots must not evaluate properties")


class ContainerGraph(leg.Graph):
    def __init__(self):
        super().__init__()
        with leg.backward(is_static=True):
            node = leg.const(1, 2.0)
            self.box = SimpleNamespace(
                attributes=AttributeContainer(node), slots=SlotContainer(node))
            self.strategy = leg.const(self.action_set_size, 1.0 / self.action_set_size)


class ContainerEnvironment(leg.FileEnv):
    pass


def assert_owned_arithmetic(graph):
    node = graph.box.attributes.node
    assert graph.box.attributes.cycle is graph.box.attributes
    assert graph.box.slots.cycle is graph.box.slots
    assert graph.box.slots.node is node
    assert graph.box.slots.private_node() is node
    with leg.backward(is_static=True):
        result = node + graph.box.slots.node
    environment = leg.FileEnv(str(GAME))
    environment.set_graph(graph)
    for _, value in environment.get_value(1, result):
        np.testing.assert_array_equal(value, [4.0])


def test_pickle_rebinds_namespaces_instance_dictionaries_and_inherited_slots():
    original = ContainerGraph()
    restored = pickle.loads(pickle.dumps(original))
    assert_owned_arithmetic(restored)
    with pytest.raises(ValueError, match="different graphs"):
        restored.box.attributes.node + original.box.attributes.node


def test_checkpoint_load_in_place_rebinds_custom_container_nodes(tmp_path):
    graph = ContainerGraph()
    environment = leg.FileEnv(str(GAME))
    environment.set_graph(graph)
    checkpoint = tmp_path / "container.ckpt"
    graph.save(checkpoint)
    graph.load(checkpoint)
    assert_owned_arithmetic(graph)


def test_attribute_binding_preserves_nested_graph_and_environment_boundaries():
    outer = ContainerGraph()
    child = ContainerGraph()
    environment = ContainerEnvironment(str(GAME))
    environment.set_graph(child)
    environment.stored_node = child.box.attributes.node
    outer.box.child = child
    outer.box.environment = environment
    native._graph_bind_attributes(outer, outer.__dict__)
    # The nested graph and native environment belong to their own graph scope.
    environment.stored_node + child.box.slots.node
    with pytest.raises(ValueError, match="different graphs"):
        outer.box.attributes.node + child.box.attributes.node


def test_node_from_destroyed_graph_fails_without_accessing_freed_storage():
    graph = ContainerGraph()
    node = graph.box.attributes.node
    del graph
    with pytest.raises(ValueError, match="live owning graph"):
        node + 1.0
