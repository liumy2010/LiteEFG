"""Compile and load procedural C++ games without serializing their game trees."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

__all__ = ["CppEnv", "compile_cpp_game"]


def _run(command, cwd):
    try:
        result = subprocess.run(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", check=False,
        )
    except OSError as error:
        raise RuntimeError("Could not run the C++ compiler: {}".format(error)) from error
    if result.returncode:
        invocation = " ".join(shlex.quote(str(arg)) for arg in command)
        raise RuntimeError(
            "C++ game compilation failed (exit {}):\n{}\n{}{}".format(
                result.returncode, invocation, result.stdout, result.stderr,
            )
        )
    return result.stdout


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dependency_paths(output):
    """Read GCC/Clang make dependencies, including escaped spaces and dollars."""
    output = re.sub(r"\\\r?\n", "", output)
    prefix = "_liteefg_cpp_game:"
    if not output.startswith(prefix):
        raise RuntimeError("C++ compiler returned an invalid dependency listing")
    tokens, token = [], []
    index = len(prefix)
    while index < len(output):
        char = output[index]
        if char == "\\" and index + 1 < len(output):
            index += 1
            token.append(output[index])
        elif char == "$" and output[index:index + 2] == "$$":
            token.append("$")
            index += 1
        elif char.isspace():
            if token:
                tokens.append("".join(token))
                token = []
        else:
            token.append(char)
        index += 1
    if token:
        tokens.append("".join(token))
    return tokens


def _dependencies(compiler, flags, source, api_header):
    output = _run(
        compiler + flags + ["-M", "-MT", "_liteefg_cpp_game", str(source)],
        source.parent,
    )
    paths = {source, api_header}
    for item in _dependency_paths(output):
        path = Path(item)
        paths.add((source.parent / path).resolve() if not path.is_absolute() else path.resolve())
    return {str(path): _sha256(path) for path in sorted(paths)}


@contextmanager
def _cache_lock(path):
    # Process-scoped advisory locks are released even when a compiler is killed.
    import fcntl

    with path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _valid_cached_library(directory, name, fingerprint):
    try:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        return (
            metadata["fingerprint"] == fingerprint
            and metadata["library_sha256"] == _sha256(directory / name)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def compile_cpp_game(source, cache_dir=None, extra_compile_args=()):
    """Return the cached shared-library path for a C++17 game implementation.

    Requires GCC or Clang on Linux (including WSL) or macOS. ``CXX`` selects
    the compiler and ``CXXFLAGS`` supplies additional flags. Relative include
    paths in flags are resolved from the source file's directory. The cache
    contains compiled code and compiler metadata, never game-tree data.
    """
    if not (sys.platform.startswith("linux") or sys.platform == "darwin"):
        raise RuntimeError("C++ game plugins require Linux/WSL or macOS; on Windows use WSL")
    source = Path(source).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError("C++ game source must be a file: {}".format(source))
    include = Path(__file__).resolve().parent / "include"
    api_header = include / "LiteEFG" / "Game.h"
    if not api_header.is_file():
        raise RuntimeError("LiteEFG's C++ game API header is missing: {}".format(api_header))
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler:
        raise ValueError("CXX must name a C++ compiler")
    executable = shutil.which(compiler[0])
    if executable is None:
        raise RuntimeError("C++ compiler not found: {} (set CXX to a GCC/Clang compiler)".format(compiler[0]))
    # Preserve the invocation name: clang++ and clang can be symlinks to the
    # same binary but choose different standard-library linkage.
    compiler[0] = str(Path(executable).absolute())
    if isinstance(extra_compile_args, (str, bytes)):
        raise TypeError("extra_compile_args must be a sequence of individual arguments")
    flags = shlex.split(os.environ.get("CXXFLAGS", "")) + [
        "-std=c++17", "-O3", "-fPIC", "-I", str(include),
    ] + list(extra_compile_args)
    link_flags = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared"]
    identity = {
        "format": 1,
        "compiler": compiler,
        "compiler_version": _run(compiler + ["--version"], source.parent).strip(),
        "compiler_target": _run(compiler + ["-dumpmachine"], source.parent).strip(),
        "compiler_sha256": _sha256(compiler[0]),
        "flags": flags + link_flags,
        "platform": sys.platform,
        "machine": platform.machine(),
        "source": str(source),
        "compiler_environment": {
            name: os.environ.get(name) for name in (
                "CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "LIBRARY_PATH",
                "SDKROOT", "MACOSX_DEPLOYMENT_TARGET", "COMPILER_PATH",
            )
        },
    }
    if cache_dir is None:
        cache_dir = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "LiteEFG" / "cpp-games"
    cache = Path(cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    name = "game.dylib" if sys.platform == "darwin" else "game.so"

    for _ in range(3):
        dependencies = _dependencies(compiler, flags, source, api_header)
        inputs = dict(identity, dependencies=dependencies)
        fingerprint = hashlib.sha256(
            json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        destination = cache / fingerprint
        with _cache_lock(cache / (fingerprint + ".lock")):
            if _valid_cached_library(destination, name, fingerprint):
                return str(destination / name)
            with tempfile.TemporaryDirectory(prefix=".build-", dir=str(cache)) as temporary:
                build = Path(temporary)
                library = build / name
                _run(compiler + flags + link_flags + [str(source), "-o", str(library)], source.parent)
                # Do not label a binary with stale inputs if an editor changed
                # a source/header while the compiler was reading it.
                if _dependencies(compiler, flags, source, api_header) != dependencies:
                    continue
                metadata = dict(inputs, fingerprint=fingerprint, library_sha256=_sha256(library))
                (build / "metadata.json").write_text(
                    json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                )
                destination.mkdir(exist_ok=True)
                # Readers take the same lock. Publish metadata last so an
                # interrupted build is always detected and rebuilt.
                os.replace(library, destination / name)
                os.replace(build / "metadata.json", destination / "metadata.json")
                return str(destination / name)
    raise RuntimeError("C++ game sources changed repeatedly during compilation; retry with stable source files")


def CppEnv(source, parameters=None, traverse_type="External", cache_dir=None, extra_compile_args=()):
    """Create an implicit game environment from a C++ implementation.

    ``parameters`` is a JSON-serializable mapping passed to the game's factory.
    States are generated during traversal; no game file is read or written.
    Exact evaluation visits the full game and should be requested infrequently
    for large games. See ``docs/guide/cpp-games.md`` for graph restrictions.
    """
    from . import _LiteEFG

    factory = getattr(_LiteEFG, "_CppEnv", None)
    if factory is None:
        raise RuntimeError("The installed LiteEFG extension lacks CppEnv; rebuild or install a matching extension")
    encoded_parameters = json.dumps(
        {} if parameters is None else dict(parameters), sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )
    library = compile_cpp_game(source, cache_dir, extra_compile_args)
    return factory(library, encoded_parameters, traverse_type)
