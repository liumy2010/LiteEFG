"""Check that shipped type declarations are usable and match the public API."""

import importlib
import inspect
from pathlib import Path
import typing

import pytest

import LiteEFG as leg


PACKAGE = Path(__file__).resolve().parents[1] / "LiteEFG"
STUBS = sorted(PACKAGE.rglob("*.pyi"))
BASELINE_STUBS = sorted(
    path for path in (PACKAGE / "baselines").glob("*.pyi")
    if path.stem not in {"__init__", "baseline"}
)


def load_stub(path):
    namespace = {"__name__": "_liteefg_stub_check"}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("path", STUBS, ids=lambda path: str(path.relative_to(PACKAGE)))
def test_type_stubs_have_valid_python_syntax(path):
    compile(path.read_text(encoding="utf-8"), str(path), "exec")


@pytest.mark.parametrize("path", BASELINE_STUBS, ids=lambda path: path.stem)
def test_baseline_stub_constructor_matches_runtime(path):
    module = importlib.import_module("LiteEFG.baselines." + path.stem)
    namespace = load_stub(path)
    declared = namespace["graph"].__init__
    actual = inspect.signature(module.graph.__init__).parameters
    expected = inspect.signature(declared).parameters
    assert list(expected) == list(actual)
    for name in expected:
        assert expected[name].default == actual[name].default
        assert expected[name].kind == actual[name].kind
    # Resolving annotations also detects unimported typing.Literal and aliases.
    hints = typing.get_type_hints(declared, namespace, namespace)
    runtime_hints = typing.get_type_hints(module.graph.__init__)
    for name, hint in hints.items():
        assert hint == runtime_hints[name]


def test_openspiel_stub_preserves_vector_values_and_dataframe_exports():
    import pandas as pd
    from open_spiel.python.policy import TabularPolicy

    namespace = load_stub(PACKAGE / "src/Environment/OpenSpiel/OpenSpielToGameFile.pyi")
    adapter = namespace["OpenSpielEnv"]
    values = typing.get_type_hints(adapter.get_value, namespace, namespace)
    exports = typing.get_type_hints(adapter.get_strategy, namespace, namespace)
    assert values["return"] == typing.List[typing.Tuple[str, typing.List[float]]]
    assert exports["return"] == typing.Tuple[TabularPolicy, typing.List[pd.DataFrame]]


def test_const_stub_excludes_floating_point_sizes():
    namespace = load_stub(PACKAGE / "_LiteEFG.pyi")
    hints = typing.get_type_hints(namespace["const"], namespace, namespace)
    assert set(typing.get_args(hints["size"])) == {int, namespace["GraphNode"]}
    with pytest.raises(ValueError, match="size"):
        leg.const(1.5, 0.0)
