"""C++ code-cache correctness, failure recovery, and safe compiler invocation."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import threading
import time

import pytest

import LiteEFG
from LiteEFG import cpp_game


@pytest.fixture
def compiler(tmp_path, monkeypatch):
    source = tmp_path / "a game #$ with spaces.cpp"
    source.write_text("// procedural game\n", encoding="utf-8")
    included = tmp_path / "nested header #$.h"
    included.write_text("// transitive dependency\n", encoding="utf-8")
    executable = tmp_path / "c++"
    executable.write_bytes(b"fake compiler executable")
    state = {"compiles": 0, "commands": [], "version": "test compiler 1", "error": None, "edit": False}
    mutex = threading.Lock()

    def escape(path):
        return str(path).replace("\\", "\\\\").replace("$", "$$").replace("#", "\\#").replace(" ", "\\ ")

    def run(command, **kwargs):
        assert isinstance(command, list)
        assert not kwargs.get("shell", False)
        assert kwargs["cwd"] == source.parent
        with mutex:
            state["commands"].append(command)
        if "--version" in command:
            output = state["version"]
        elif "-dumpmachine" in command:
            output = "test-linux-gnu"
        elif "-M" in command:
            output = "_liteefg_cpp_game: {} ".format(escape(source)) + "\\\n " + escape(included) + "\n"
        else:
            assert command[-3] == str(source)
            assert command[-2] == "-o"
            with mutex:
                state["compiles"] += 1
            time.sleep(0.02)
            if state["error"]:
                return subprocess.CompletedProcess(command, 1, "", state["error"])
            Path(command[-1]).write_bytes((state["version"] + source.read_text() + included.read_text()).encode())
            if state["edit"]:
                state["edit"] = False
                source.write_text("// changed during compilation\n")
            output = ""
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(cpp_game.sys, "platform", "linux")
    monkeypatch.setattr(cpp_game.shutil, "which", lambda _: str(executable))
    monkeypatch.setattr(cpp_game.subprocess, "run", run)
    monkeypatch.setenv("CXX", "c++")
    monkeypatch.delenv("CXXFLAGS", raising=False)
    return source, included, tmp_path / "cache", state


def test_reuses_compiled_code_without_creating_a_game_file(compiler):
    source, _, cache, state = compiler
    first = cpp_game.compile_cpp_game(source, cache)
    second = cpp_game.compile_cpp_game(source, cache)
    assert first == second
    assert state["compiles"] == 1
    metadata = json.loads((Path(first).parent / "metadata.json").read_text())
    assert metadata["compiler_version"] == "test compiler 1"
    assert len(metadata["dependencies"]) >= 3
    assert not list(cache.rglob("*.game"))
    assert not list(cache.glob(".build-*"))


@pytest.mark.parametrize("changed", ["source", "header", "flags", "compiler"])
def test_cache_invalidates_for_compilation_inputs(compiler, monkeypatch, changed):
    source, included, cache, state = compiler
    first = cpp_game.compile_cpp_game(source, cache)
    if changed == "source":
        source.write_text("// new game\n")
    elif changed == "header":
        included.write_text("// new included rules\n")
    elif changed == "flags":
        monkeypatch.setenv("CXXFLAGS", "-DNEW_RULE=1")
    else:
        state["version"] = "test compiler 2"
    second = cpp_game.compile_cpp_game(source, cache)
    assert first != second
    assert state["compiles"] == 2


def test_repairs_corrupt_library_and_incomplete_metadata(compiler):
    source, _, cache, state = compiler
    library = Path(cpp_game.compile_cpp_game(source, cache))
    expected = library.read_bytes()
    library.write_bytes(b"incomplete")
    assert cpp_game.compile_cpp_game(source, cache) == str(library)
    assert library.read_bytes() == expected
    (library.parent / "metadata.json").unlink()
    cpp_game.compile_cpp_game(source, cache)
    assert state["compiles"] == 3


def test_compile_error_preserves_diagnostics_and_can_be_retried(compiler):
    source, _, cache, state = compiler
    state["error"] = "game.cpp:14: invalid game definition"
    with pytest.raises(RuntimeError, match="game.cpp:14: invalid game definition"):
        cpp_game.compile_cpp_game(source, cache)
    assert not list(cache.rglob("game.so"))
    assert not list(cache.glob(".build-*"))
    state["error"] = None
    assert Path(cpp_game.compile_cpp_game(source, cache)).is_file()


def test_concurrent_callers_share_one_complete_build(compiler):
    source, _, cache, state = compiler
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda _: cpp_game.compile_cpp_game(source, cache), range(4)))
    assert len(set(paths)) == 1
    assert state["compiles"] == 1
    assert Path(paths[0]).read_bytes()


def test_retries_when_source_changes_during_compilation(compiler):
    source, _, cache, state = compiler
    state["edit"] = True
    library = Path(cpp_game.compile_cpp_game(source, cache))
    assert state["compiles"] == 2
    assert b"changed during compilation" in library.read_bytes()
    assert len(list(cache.rglob("game.so"))) == 1


def test_factory_passes_json_parameters_without_materializing_a_tree(monkeypatch):
    captured = {}
    sentinel = object()

    def compile_game(source, cache, flags):
        captured["compile"] = (source, cache, flags)
        return "/tmp/plugin.so"

    def native(library, parameters, traversal):
        captured["native"] = (library, json.loads(parameters), traversal)
        return sentinel

    monkeypatch.setattr(cpp_game, "compile_cpp_game", compile_game)
    monkeypatch.setattr(LiteEFG._LiteEFG, "_CppEnv", native, raising=False)
    assert cpp_game.CppEnv("game.cpp", {"size": 3}, cache_dir="cache") is sentinel
    assert captured["compile"] == ("game.cpp", "cache", ())
    assert captured["native"] == ("/tmp/plugin.so", {"size": 3}, "External")
    with pytest.raises(ValueError):
        cpp_game.CppEnv("game.cpp", {"size": float("nan")})


def test_rejects_native_windows_with_wsl_guidance(monkeypatch):
    monkeypatch.setattr(cpp_game.sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="on Windows use WSL"):
        cpp_game.compile_cpp_game("game.cpp")
