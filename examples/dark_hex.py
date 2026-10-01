"""Train Dark Hex from C++ states and periodically evaluate exact NashConv."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import LiteEFG as leg
from LiteEFG.baselines import CFR


def revision():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def workspace_dirty():
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True,
        check=False,
    )
    return bool(result.stdout) if result.returncode == 0 else None


def peak_rss_bytes():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss if sys.platform == "darwin" else rss * 1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-size", type=int, choices=(2, 3, 4), default=3)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--strategy", choices=(
        "default", "last-iterate", "avg-iterate", "linear-avg-iterate"), default="default")
    parser.add_argument("--source", type=Path, default=ROOT / "examples/cpp_games/dark_hex.cpp")
    parser.add_argument("--output", type=Path, help="Optional JSON benchmark report path")
    args = parser.parse_args()
    if args.iterations < 0:
        parser.error("--iterations must be nonnegative")
    if args.eval_every <= 0:
        parser.error("--eval-every must be positive")

    source = args.source.resolve()
    started = time.perf_counter()
    leg.set_seed(args.seed)
    env = leg.CppEnv(source, parameters={"board_size": args.board_size}, traverse_type="External")
    with redirect_stdout(sys.stderr):
        graph = CFR.graph()
    env.set_graph(graph)
    strategy = graph.current_strategy()
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine": "workspace LiteEFG CppEnv",
        "engine_commit": revision(),
        "engine_workspace_dirty": workspace_dirty(),
        "engine_binary_sha256": hashlib.sha256(Path(leg._LiteEFG.__file__).read_bytes()).hexdigest(),
        "algorithm": "workspace LiteEFG.baselines.CFR (External)",
        "baseline_source_sha256": hashlib.sha256(Path(CFR.__file__).read_bytes()).hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "game_source": str(source),
        "game_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rules": "classical Dark Hex, reveal nothing, perfect recall of own attempts and outcomes",
        "parameters": {"board_size": args.board_size},
        "seed": args.seed,
        "iterations": args.iterations,
        "eval_every": args.eval_every,
        "strategy": args.strategy,
        "evaluation": "exact per-player best-response gain; exploitability = NashConv / 2",
        "setup_seconds": time.perf_counter() - started,
        "checkpoints": [],
    }
    training_seconds = 0.0

    def evaluate(iteration):
        print(f"Iteration {iteration}: beginning exact {args.strategy} evaluation", file=sys.stderr,
              flush=True)
        before = time.perf_counter()
        metrics = env.evaluate(strategy, args.strategy)
        gains = metrics["deviation_gain"]
        checkpoint = {
            "iteration": iteration,
            "training_seconds": training_seconds,
            "evaluation_seconds": time.perf_counter() - before,
            "utility": metrics["utility"],
            "deviation_gain": gains,
            "nash_conv": sum(gains),
            "exploitability": sum(gains) / 2.0,
            "peak_rss_bytes": peak_rss_bytes(),
            "stats": env.stats(),
        }
        result["checkpoints"].append(checkpoint)
        print(json.dumps(checkpoint, sort_keys=True), file=sys.stderr, flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    for iteration in range(1, args.iterations + 1):
        before = time.perf_counter()
        graph.update_graph(env)
        if args.strategy != "default":
            env.update_strategy(strategy)
        training_seconds += time.perf_counter() - before
        if iteration % args.eval_every == 0:
            evaluate(iteration)
    if args.iterations == 0 or args.iterations % args.eval_every:
        evaluate(args.iterations)
    result["wall_seconds"] = time.perf_counter() - started
    result["peak_rss_bytes"] = peak_rss_bytes()
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
