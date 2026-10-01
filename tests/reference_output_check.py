"""Record and compare numerical baseline outputs in isolated worker processes."""

import argparse
import concurrent.futures
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests/baselines/manifest.json"
ARTIFACTS = ROOT / "artifacts/reference-output-checks"
REFERENCE = ROOT / "tests/baselines" / ("reference-" + sys.platform + ".json")
PUBLIC_BASELINE_REFERENCE = ROOT / "tests/baselines" / (
    "reference-public-baselines-" + sys.platform + ".json")
PYPI_REFERENCE = ROOT / "tests/baselines" / ("reference-pypi-" + sys.platform + ".json")
PUBLIC_BASELINES = ROOT / "tests/baselines/public"
sys.path.insert(0, str(ROOT))


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write_json(path, data):
    """Never leave a truncated reference or permit non-finite JSON numbers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_manifest(path=MANIFEST):
    manifest = read_json(path)
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported manifest schema")
    checkpoints = manifest["checkpoints"]
    if (not checkpoints or checkpoints != sorted(set(checkpoints))
            or checkpoints[0] < 1 or checkpoints[-1] != manifest["iterations"]):
        raise ValueError("Checkpoints must be increasing and end at iterations")
    kinds = manifest["strategy_types"]
    if (not kinds or len(kinds) != len(set(kinds))
            or not set(kinds) <= {"last-iterate", "avg-iterate", "linear-avg-iterate"}):
        raise ValueError("strategy_types must be nonempty, unique supported selectors")
    for key in ("games", "baselines"):
        ids = [item["id"] for item in manifest[key]]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Empty or duplicate IDs in " + key)
    for tolerance in manifest["tolerances"].values():
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Tolerances must be finite and nonnegative")
    return manifest


def public_source():
    source = read_json(PUBLIC_BASELINES / "source.json")
    actual = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
              for path in PUBLIC_BASELINES.glob("*.py")}
    if actual != source["files_sha256"]:
        raise ValueError("Frozen public baseline sources do not match source.json")
    return source


def activate_package(oracle_root=None, baseline_source="workspace"):
    """Select the engine and baseline layer without silently mixing their origins."""
    if baseline_source not in {"workspace", "public"}:
        raise ValueError("Unknown baseline source: " + str(baseline_source))
    if oracle_root and baseline_source != "public":
        raise ValueError("The PyPI oracle requires frozen public baselines")
    package_root = Path(oracle_root).resolve() if oracle_root else ROOT
    if oracle_root:
        if package_root == ROOT or ROOT in package_root.parents:
            raise ValueError("The PyPI oracle must be installed outside this checkout")
        distributions = {dist.metadata["Name"].lower(): dist
                         for dist in importlib.metadata.distributions(path=[str(package_root)])}
        if "liteefg" not in distributions:
            raise ValueError("--oracle-root must contain a pip-installed LiteEFG distribution")
    sys.path[:] = [entry for entry in sys.path if Path(entry or os.getcwd()).resolve() != ROOT]
    sys.path.insert(0, str(package_root))
    import LiteEFG as leg
    if Path(leg.__file__).resolve().parent != package_root / "LiteEFG":
        raise ValueError("Wrong LiteEFG imported: {}. Use a fresh process.".format(leg.__file__))
    if oracle_root and Path(leg._LiteEFG.__file__).resolve().parent != package_root / "LiteEFG":
        raise ValueError("Oracle native extension was imported from the wrong package")
    directory = PUBLIC_BASELINES if baseline_source == "public" else ROOT / "LiteEFG/baselines"
    if baseline_source == "public":
        public_source()
    # Public baseline modules use absolute LiteEFG.baselines imports. Give them
    # their original package name against the explicitly selected engine.
    for name in list(sys.modules):
        if name == "LiteEFG.baselines" or name.startswith("LiteEFG.baselines."):
            del sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        "LiteEFG.baselines", directory / "__init__.py",
        submodule_search_locations=[str(directory)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    leg.baselines = module
    spec.loader.exec_module(module)
    return directory


def baseline_defaults(manifest, directory=None):
    """A newly added/removed baseline or changed constructor default must fail."""
    from LiteEFG.baselines.baseline import _baseline

    discovered = {}
    directory = directory or ROOT / "LiteEFG/baselines"
    for source in sorted(directory.glob("*.py")):
        if source.name == "__init__.py":
            continue
        module = importlib.import_module("LiteEFG.baselines." + source.stem)
        implementations = [value for value in vars(module).values()
                           if inspect.isclass(value) and value is not _baseline
                           and issubclass(value, _baseline)
                           and value.__module__ == module.__name__]
        if implementations:
            if len(implementations) != 1 or implementations[0].__name__ != "graph":
                raise ValueError("Register the new baseline entry point in " + source.name)
            parameters = inspect.signature(implementations[0]).parameters
            discovered[source.stem] = {
                name: parameter.default for name, parameter in parameters.items()
                if parameter.default is not inspect.Parameter.empty
            }
    selected = {case["module"] for case in manifest["baselines"]}
    if selected != set(discovered):
        raise ValueError("Baseline coverage changed: untested={}, missing={}".format(
            sorted(set(discovered) - selected), sorted(selected - set(discovered))))
    return discovered


def runtime_contract():
    # std::default_random_engine and distributions vary by C++ standard library.
    return {
        "platform": sys.platform,
        "python": "{}.{}".format(*sys.version_info[:2]),
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("open_spiel", "numpy", "pandas")},
    }


def provenance(baseline_source, oracle_root=None):
    import LiteEFG as leg
    package = Path(leg.__file__).resolve().parent
    sources = sorted(package.rglob("*.py"))
    sources += sorted((package / "src").rglob("*.cpp"))
    sources += sorted((package / "src").rglob("*.h"))
    digest = hashlib.sha256()
    for source in sources:
        digest.update(str(source.relative_to(package)).encode())
        digest.update(source.read_bytes())
    extension = Path(leg._LiteEFG.__file__)
    data = {
        "engine_origin": "pypi" if oracle_root else "workspace",
        "engine_version": getattr(leg._LiteEFG, "__version__", "unavailable"),
        "baseline_source": baseline_source,
        "public_baselines": public_source(),
        "source_sha256": digest.hexdigest(),
        "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest(),
        "platform_description": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }
    if oracle_root:
        distribution = next(dist for dist in importlib.metadata.distributions(path=[str(oracle_root)])
                            if dist.metadata["Name"].lower() == "liteefg")
        data["pypi_distribution_version"] = distribution.version
        record = distribution.read_text("RECORD")
        data["distribution_record_sha256"] = hashlib.sha256(record.encode()).hexdigest()
    return data


def ensure_local_build():
    """Reject stale extension binaries so C++ edits cannot accidentally pass."""
    import LiteEFG as leg
    extension = Path(leg._LiteEFG.__file__).resolve()
    if extension.parent != ROOT / "LiteEFG":
        raise ValueError("Build the checkout's extension with setup.py build_ext --inplace")
    inputs = [ROOT / "CMakeLists.txt", ROOT / "setup.py"]
    inputs += list((ROOT / "LiteEFG/src").rglob("*.cpp"))
    inputs += list((ROOT / "LiteEFG/src").rglob("*.h"))
    newer = [str(path.relative_to(ROOT)) for path in inputs
             if path.stat().st_mtime_ns > extension.stat().st_mtime_ns]
    if newer:
        raise ValueError("Rebuild the native extension before testing; newer sources: "
                         + ", ".join(newer[:5]))


def compare_values(expected, actual, absolute, relative, path="output"):
    """Report structural and numerical differences; NaN/Inf always fail."""
    differences = []
    if isinstance(expected, dict) and isinstance(actual, dict):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        if missing or extra:
            differences.append("{}: missing={}, extra={}".format(path, missing, extra))
        for key in sorted(set(expected) & set(actual)):
            differences.extend(compare_values(expected[key], actual[key], absolute,
                                              relative, path + "." + key))
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            differences.append("{}: length {} != {}".format(path, len(expected), len(actual)))
        for index, (left, right) in enumerate(zip(expected, actual)):
            differences.extend(compare_values(left, right, absolute, relative,
                                              "{}[{}]".format(path, index)))
    elif (type(expected) in (int, float) and type(actual) in (int, float)
          and (isinstance(expected, float) or isinstance(actual, float))):
        limit = absolute + relative * abs(expected)
        if (not math.isfinite(expected) or not math.isfinite(actual)
                or abs(expected - actual) > limit):
            differences.append("{}: expected={:.17g}, actual={:.17g}, abs_diff={:.3g}, "
                               "limit={:.3g}".format(path, expected, actual,
                                                    abs(expected - actual), limit))
    elif type(expected) is not type(actual) or expected != actual:
        differences.append("{}: expected={!r}, actual={!r}".format(path, expected, actual))
    return differences


def compare_snapshots(expected, actual):
    # Configuration changes are exact mismatches, not tolerated numeric drift.
    differences = compare_values(expected["contract"], actual["contract"], 0.0, 0.0,
                                 "contract")
    tolerance = expected["contract"]["manifest"]["tolerances"]
    differences.extend(compare_values(expected["results"], actual["results"],
                                      tolerance["absolute"], tolerance["relative"],
                                      "results"))
    return differences


def validate_snapshot(snapshot):
    """Two equally empty or incomplete runs must never become a reference."""
    manifest = snapshot["contract"]["manifest"]
    expected_cases = {case["id"] + "/" + game["id"] for case in manifest["baselines"]
                      for game in manifest["games"]}
    differences = []

    def keys(value, expected, path):
        if not isinstance(value, dict) or set(value) != expected:
            differences.append(path + ": incomplete or unexpected output fields")
            return False
        return True

    if not keys(snapshot["results"], expected_cases, "results"):
        return differences
    for case, checkpoints in snapshot["results"].items():
        if not keys(checkpoints, {str(i) for i in manifest["checkpoints"]}, case):
            continue
        for iteration, strategies in checkpoints.items():
            path = case + "/" + iteration
            if not keys(strategies, set(manifest["strategy_types"]), path):
                continue
            for kind, metrics in strategies.items():
                metric_path = path + "/" + kind
                if not keys(metrics, {"utility", "deviation_gain", "nash_conv",
                                      "exploitability", "information_sets"}, metric_path):
                    continue
                for key, value in metrics.items():
                    values = value if isinstance(value, list) else [value]
                    if key in {"utility", "deviation_gain", "information_sets"}:
                        if not isinstance(value, list) or len(value) != 2:
                            differences.append(metric_path + "/" + key + ": expected two players")
                    if not values or any(type(v) not in (float, int) or not math.isfinite(v)
                                         for v in values):
                        differences.append(metric_path + "/" + key + ": invalid numeric metric")
    return differences


def measure(env, strategy, strategy_type):
    import LiteEFG as leg

    utility = list(env.utility(strategy, strategy_type))
    gains = list(env.exploitability(strategy, strategy_type))
    if len(utility) != 2 or len(gains) != 2:
        raise ValueError("Benchmarks must have exactly two players")
    if not all(math.isfinite(value) for value in utility + gains):
        raise ValueError("Non-finite utility or exploitability")
    if abs(math.fsum(utility)) > 1e-8 or min(gains) < -1e-8:
        raise ValueError("Invalid zero-sum utility or negative deviation gain")
    infosets = []
    for player in (1, 2):
        # Use the bound native API to inspect probabilities without constructing
        # pandas frames. The wrapper still handles environment creation/training.
        rows = leg.Environment.get_strategy(env, player, strategy, strategy_type)
        if not rows:
            raise ValueError("Player has no information sets")
        for name, probabilities in rows:
            if (not probabilities or not all(math.isfinite(p) for p in probabilities)
                    or min(probabilities) < -1e-10 or max(probabilities) > 1 + 1e-10
                    or abs(math.fsum(probabilities) - 1) > 1e-8):
                raise ValueError("Invalid strategy at player {} / {}".format(player, name))
        infosets.append(len(rows))
    nash_conv = math.fsum(gains)
    return {"utility": utility, "deviation_gain": gains, "nash_conv": nash_conv,
            "exploitability": nash_conv / 2, "information_sets": infosets}


def validate_checkpoint_interval(checkpoint_interval, oracle_root=None):
    if checkpoint_interval is not None:
        if type(checkpoint_interval) is not int or checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be a positive integer")
        if oracle_root:
            raise ValueError("Checkpoint comparisons require the workspace engine")


def run_worker(manifest, baseline_id, game_id, oracle_root=None, baseline_source="workspace",
               checkpoint_interval=None):
    validate_checkpoint_interval(checkpoint_interval, oracle_root)
    activate_package(oracle_root, baseline_source)
    import numpy as np
    import pyspiel
    import LiteEFG as leg

    case = next(case for case in manifest["baselines"] if case["id"] == baseline_id)
    game_spec = next(game for game in manifest["games"] if game["id"] == game_id)
    random.seed(manifest["seed"])
    np.random.seed(manifest["seed"])
    leg.set_seed(manifest["seed"])
    module = importlib.import_module("LiteEFG.baselines." + case["module"])
    game = pyspiel.load_game(game_spec["spec"])
    if game.num_players() != 2 or game.get_type().utility != pyspiel.GameType.Utility.ZERO_SUM:
        raise ValueError("Only two-player zero-sum benchmark games are supported")

    # OpenSpielEnv currently hardcodes ~/game_instances. Redirect only its
    # construction-time expanduser('~') lookup; don't change HOME or reuse caches.
    expanduser = os.path.expanduser
    with tempfile.TemporaryDirectory(prefix="liteefg-reference-outputs-") as cache:
        with patch("os.path.expanduser", side_effect=lambda path:
                   cache if path == "~" else expanduser(path)):
            env = leg.OpenSpielEnv(game, traverse_type=case["traversal"], regenerate=True)
        graph = module.graph(**case["kwargs"])
        env.set_graph(graph)
        checkpoints = {}
        checkpoint_path = Path(cache) / "training.checkpoint"
        for iteration in range(1, manifest["iterations"] + 1):
            graph.update_graph(env)
            strategy = graph.current_strategy()
            env.update_strategy(strategy)
            if checkpoint_interval and iteration % checkpoint_interval == 0:
                graph.save(checkpoint_path)
                graph.load(checkpoint_path)
                # Measure restored values even at the final (50th) update.
                strategy = graph.current_strategy()
            if iteration in manifest["checkpoints"]:
                checkpoints[str(iteration)] = {
                    kind: measure(env, strategy, kind) for kind in manifest["strategy_types"]
                }
    return checkpoints


def collect_snapshot(manifest, jobs=4, oracle_root=None, baseline_source="workspace",
                     checkpoint_interval=None):
    validate_checkpoint_interval(checkpoint_interval, oracle_root)
    directory = activate_package(oracle_root, baseline_source)
    if not oracle_root:
        ensure_local_build()
    contract = {"manifest": manifest, "runtime": runtime_contract(),
                "constructor_defaults": baseline_defaults(manifest, directory),
                "public_baselines": public_source()}
    results, errors = {}, []
    with tempfile.TemporaryDirectory(prefix="liteefg-results-") as directory:
        directory = Path(directory)
        config_path = directory / "manifest.json"
        write_json(config_path, manifest)

        def run(case, game):
            key = case["id"] + "/" + game["id"]
            output_path = directory / (case["id"] + "-" + game["id"] + ".json")
            command = [sys.executable, str(Path(__file__).resolve()), "_worker",
                       "--manifest", str(config_path), "--baseline", case["id"],
                       "--game", game["id"], "--output", str(output_path),
                       "--baseline-source", baseline_source]
            if oracle_root:
                command.extend(["--oracle-root", str(Path(oracle_root).resolve())])
            if checkpoint_interval:
                command.extend(["--checkpoint-interval", str(checkpoint_interval)])
            worker_env = dict(os.environ, PYTHONHASHSEED=str(manifest["seed"]),
                              OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                              MKL_NUM_THREADS="1")
            try:
                process = subprocess.run(command, cwd=ROOT, env=worker_env,
                                         capture_output=True, text=True, timeout=120)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(key + ": timed out after 120 seconds") from exc
            if process.returncode:
                raise RuntimeError("{}: worker exit {}\n{}\n{}".format(
                    key, process.returncode, process.stdout[-4000:], process.stderr[-4000:]))
            return key, read_json(output_path)

        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(run, case, game) for case in manifest["baselines"]
                       for game in manifest["games"]]
            for future in concurrent.futures.as_completed(futures):
                try:
                    key, values = future.result()
                    results[key] = values
                except Exception as exc:
                    errors.append(str(exc))
    source = provenance(baseline_source, oracle_root)
    if checkpoint_interval:
        source["checkpoint_interval"] = checkpoint_interval
    return {"contract": contract, "provenance": source,
            "results": dict(sorted(results.items())), "errors": errors}


def default_reference(baseline_source="workspace", engine_origin="workspace"):
    if baseline_source not in {"workspace", "public"}:
        raise ValueError("Unknown baseline source: " + str(baseline_source))
    if engine_origin == "pypi" and baseline_source == "public":
        return PYPI_REFERENCE
    if engine_origin != "workspace":
        raise ValueError("The PyPI oracle requires frozen public baselines")
    return REFERENCE if baseline_source == "workspace" else PUBLIC_BASELINE_REFERENCE


def validate_source(snapshot, baseline_source, engine_origin=None):
    """Origin is a layer guard; source/binary hashes remain descriptive provenance.

    Checks may explicitly use the historic PyPI/public reference for either local
    layer. Recording must match the exact requested engine and baseline source.
    """
    source = snapshot.get("provenance", {})
    if not isinstance(source, dict):
        raise ValueError("Missing or invalid reference provenance")
    engine, baselines = source.get("engine_origin"), source.get("baseline_source")
    if (engine not in {"workspace", "pypi"} or baselines not in {"workspace", "public"}
            or (engine == "pypi" and baselines != "public")):
        raise ValueError("Invalid reference provenance: expected a workspace layer or "
                         "the PyPI engine with frozen public baselines")
    if engine_origin is not None and engine != engine_origin:
        raise ValueError("Reference engine_origin mismatch: expected {}, got {}".format(
            engine_origin, engine))
    if (engine_origin is not None or engine == "workspace") and baselines != baseline_source:
        raise ValueError("Reference baseline_source mismatch: expected {}, got {}".format(
            baseline_source, baselines))


def check(reference=None, artifacts=ARTIFACTS, manifest_path=MANIFEST, jobs=4,
          baseline_source="workspace", checkpoint_interval=None):
    validate_checkpoint_interval(checkpoint_interval)
    reference = Path(reference) if reference is not None else default_reference(baseline_source)
    artifacts = Path(artifacts)
    if not reference.is_file():
        raise ValueError("No reference for this platform: {}. Review and record a "
                         "reference explicitly; check never creates one.".format(reference))
    expected = read_json(reference)
    validate_source(expected, baseline_source)
    invalid = expected["errors"] + validate_snapshot(expected)
    if invalid:
        raise ValueError("Invalid reference:\n" + "\n".join(invalid))
    manifest = load_manifest(manifest_path)
    # Stop with an actionable error before launching workers in the wrong runtime.
    runtime_diff = compare_values(expected["contract"]["runtime"], runtime_contract(),
                                  0.0, 0.0, "runtime")
    if runtime_diff:
        raise ValueError("Reference environment mismatch:\n" + "\n".join(runtime_diff))
    actual = collect_snapshot(manifest, jobs, baseline_source=baseline_source,
                              checkpoint_interval=checkpoint_interval)
    validate_source(actual, baseline_source, "workspace")
    write_json(artifacts / "actual.json", actual)
    differences = actual["errors"] + validate_snapshot(actual) + compare_snapshots(expected, actual)
    write_json(artifacts / "comparison.json", {"passed": not differences,
               "reference": str(reference), "case_count": len(actual["results"]),
               "differences": differences})
    if differences:
        raise AssertionError("Reference output checks failed ({} differences):\n{}\n"
                             "Full results: {}".format(len(differences),
                             "\n".join(differences[:20]), artifacts))
    return len(actual["results"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "record", "record-workspace", "list", "_worker"])
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--reference", type=Path,
                        help="Reference path; defaults to the selected engine/baseline layer")
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--checkpoint-interval", type=int,
                        help="For check: save and reload training after every N updates")
    parser.add_argument("--oracle-root", type=Path,
                        help="Isolated pip --target installation (required for record)")
    parser.add_argument("--baseline-source", choices=["workspace", "public"],
                        help="Baseline layer (default: workspace; record always requires public)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly replace an existing reference after reviewing changes")
    parser.add_argument("--baseline", help=argparse.SUPPRESS)
    parser.add_argument("--game", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.checkpoint_interval is not None:
        if args.checkpoint_interval < 1:
            parser.error("--checkpoint-interval must be positive")
        if args.command not in {"check", "_worker"} or args.oracle_root:
            parser.error("--checkpoint-interval is only valid for workspace-engine checks")
    if args.overwrite and args.command not in {"record", "record-workspace"}:
        parser.error("--overwrite is only valid for record or record-workspace")
    if args.oracle_root and args.command not in {"record", "_worker"}:
        parser.error(args.command + " uses the workspace engine; --oracle-root is only for record")
    if args.command == "record" and args.baseline_source not in {None, "public"}:
        parser.error("record requires frozen public baselines; use record-workspace for local baselines")
    args.baseline_source = args.baseline_source or (
        "public" if args.command == "record" else "workspace")
    if args.command == "_worker":
        if not all((args.baseline, args.game, args.output)):
            parser.error("_worker requires --baseline, --game and --output")
        if args.oracle_root and args.baseline_source != "public":
            parser.error("The PyPI oracle requires frozen public baselines")
    elif any((args.baseline, args.game, args.output)):
        parser.error("--baseline, --game and --output are only valid for _worker")
    args.reference = args.reference or default_reference(
        args.baseline_source, "pypi" if args.command == "record" else "workspace")
    manifest = load_manifest(args.manifest)
    if args.command == "list":
        print(json.dumps(manifest, indent=2))
    elif args.command == "_worker":
        write_json(args.output, run_worker(manifest, args.baseline, args.game,
                                          args.oracle_root, args.baseline_source,
                                          args.checkpoint_interval))
    elif args.command in {"record", "record-workspace"}:
        if args.reference.exists() and not args.overwrite:
            parser.error("Reference exists. Review the change and pass --overwrite to replace it.")
        if args.command == "record" and not args.oracle_root:
            parser.error("record requires --oracle-root: install public LiteEFG with pip --target first")
        engine_origin = "pypi" if args.command == "record" else "workspace"
        first = collect_snapshot(manifest, args.jobs, args.oracle_root, args.baseline_source)
        validate_source(first, args.baseline_source, engine_origin)
        write_json(args.artifacts / "record-first.json", first)
        second = collect_snapshot(manifest, args.jobs, args.oracle_root, args.baseline_source)
        validate_source(second, args.baseline_source, engine_origin)
        write_json(args.artifacts / "record-second.json", second)
        differences = (first["errors"] + second["errors"] + validate_snapshot(first)
                       + validate_snapshot(second) + compare_snapshots(first, second))
        if differences:
            raise AssertionError("Refusing unstable/failed reference:\n" + "\n".join(differences))
        write_json(args.reference, first)
        print("Recorded {} cases after two independent runs: {}".format(
            len(first["results"]), args.reference))
    else:
        count = check(args.reference, args.artifacts, args.manifest, args.jobs,
                      args.baseline_source, args.checkpoint_interval)
        print("PASS: {} baseline/game cases; results: {}".format(count, args.artifacts))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AssertionError, ValueError, ImportError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
