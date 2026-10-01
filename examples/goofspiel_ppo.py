"""Train entropy-regularized PPO and estimate two-model Goofspiel utilities."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import optax
import flax
from flax import linen as nn

import LiteEFG as leg
from LiteEFG.baselines.drl import PPO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128,
                        help="Complete trajectories per optimizer minibatch")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=100000)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--require-gpu", action="store_true")
    device_options = parser.add_mutually_exclusive_group()
    device_options.add_argument("--devices", type=int,
                                help="Number of local devices for the selected parallel stages (default: 1)")
    device_options.add_argument("--sampling-devices", type=int,
                                help="Number of local sampling devices; optimization uses one device")
    parser.add_argument("--parallel", nargs="*", choices=("sampling", "optimizing"),
                        help="Stages to parallelize (default: sampling optimizing); no values selects neither")
    parser.add_argument("--compiled-updates", action="store_true",
                        help="Compile the PPO epoch/minibatch loop on the selected optimization devices")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/drl/goofspiel-7-ppo.json")
    args = parser.parse_args()
    if any(value <= 0 for value in (args.iterations, args.log_every, args.hidden_size,
                                    args.epochs, args.minibatch_size)):
        parser.error("iterations, log-every, hidden-size, epochs and minibatch-size must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    devices = jax.devices()
    if args.require_gpu and any(device.platform != "gpu" for device in devices):
        parser.error("A JAX GPU device is required; install the CUDA extra and check the NVIDIA driver")
    if args.sampling_devices is not None and args.parallel is not None:
        parser.error("sampling-devices cannot be combined with parallel; use devices instead")
    device_count = (args.sampling_devices if args.sampling_devices is not None
                    else args.devices if args.devices is not None else 1)
    if not 1 <= device_count <= len(jax.local_devices()):
        parser.error("device count must be between 1 and the number of local JAX devices")
    stages = ("sampling", "optimizing") if args.parallel is None else tuple(args.parallel)
    if len(set(stages)) != len(stages):
        parser.error("parallel stages must be distinct")
    sampling_count = device_count if "sampling" in stages else 1
    if args.batch_size <= 0 or args.batch_size % sampling_count:
        parser.error("batch-size must be positive and divisible by the sampling device count")
    env = leg.Goofspiel(num_cards=args.cards)
    def network(outputs):
        module = nn.Sequential([
            nn.Dense(args.hidden_size), nn.tanh,
            nn.Dense(args.hidden_size), nn.tanh, nn.Dense(outputs),
        ])
        optimizer = optax.chain(optax.clip_by_global_norm(0.5),
                                optax.adam(args.learning_rate))
        return leg.model(module, optimizer=optimizer)

    policy_network = [network(args.cards) for _ in range(env.num_players)]
    critic_network = [network(1) for _ in range(env.num_players)]
    graph = PPO.graph(policy_network, critic_network, num_actions=args.cards,
                      entropy_coef=args.entropy_coef)
    selected = jax.local_devices()[:device_count]
    options = ({"sampling_devices": selected} if args.sampling_devices is not None
               else {"devices": selected, "parallel": stages})
    graph.bind(env, batch_size=args.batch_size, seed=args.seed, **options)
    trainer = graph.trainer
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
    source_paths = ["LiteEFG/drl/graph.py", "LiteEFG/drl/runtime.py", "LiteEFG/drl/parallel.py",
                    "LiteEFG/drl/environment.py", "LiteEFG/drl/env/goofspiel.py",
                    "LiteEFG/baselines/drl/PPO.py",
                    "examples/goofspiel_ppo.py"]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "engine": "workspace LiteEFG DRLGraph / JAX", "algorithm": "PPO + entropy regularizer",
        "engine_commit": revision.stdout.strip(), "workspace_dirty": bool(dirty.stdout.strip()),
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths},
        "jax_version": jax.__version__, "optax_version": optax.__version__,
        "flax_version": flax.__version__,
        "devices": [{"platform": d.platform, "kind": d.device_kind} for d in devices],
        "sampling_devices": [{"id": d.id, "platform": d.platform, "kind": d.device_kind}
                             for d in trainer.sampling_devices],
        "optimizing_devices": [{"id": d.id, "platform": d.platform, "kind": d.device_kind}
                               for d in trainer.optimizing_devices],
        "parallel": list(trainer.parallel),
        "learner_device": {"id": trainer.key.device.id, "platform": trainer.key.device.platform,
                           "kind": trainer.key.device.device_kind},
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "game": {"name": "goofspiel", "num_cards": args.cards, "imp_info": False,
                 "points_order": "random", "returns_type": "point_difference",
                 "utility": "Each player's score minus the players' mean score (half the two-player score difference)"},
        "evaluation": "Sample mean of original game utility, without entropy bonus; not exploitability",
        "training_metrics": "Valid-decision-weighted means across optimizer minibatches in each logging chunk",
        "training": [], "matchups": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def match(label, first, second, seed):
        started = time.perf_counter()
        result = leg.average_utility(env, first, second, episodes=args.eval_episodes,
                                     batch_size=args.eval_batch_size, seed=seed)
        result["seconds_including_compile"] = time.perf_counter() - started
        report["matchups"][label] = result
        print(json.dumps({"matchup": label, **result}), flush=True)
        save()

    match("initial_player0_vs_uniform", trainer.policy(0), leg.uniform_policy, args.seed + 1)
    initial_player0 = trainer.policy(0)
    elapsed = 0.0
    while trainer.iteration < args.iterations:
        count = min(args.log_every, args.iterations - trainer.iteration)
        started = time.perf_counter()
        metrics = graph.train(iterations=count, epochs=args.epochs,
                              minibatch_size=args.minibatch_size,
                              compiled_updates=args.compiled_updates)
        seconds = time.perf_counter() - started
        elapsed += seconds
        metrics.update(chunk_iterations=count, chunk_seconds=seconds,
                       trajectories_per_second=count * args.batch_size / seconds,
                       training_seconds_including_compile=elapsed)
        report["training"].append(metrics)
        print(json.dumps(metrics), flush=True)
        save()
    first, second = trainer.policy(0), trainer.policy(1)
    first_path = args.output.with_name(args.output.stem + "-player0.npz")
    second_path = args.output.with_name(args.output.stem + "-player1.npz")
    first.save(first_path)
    second.save(second_path)
    report["policies"] = [str(first_path), str(second_path)]
    match("trained_player0_vs_player1", first, second, args.seed + 2)
    match("trained_player0_vs_uniform", first, leg.uniform_policy, args.seed + 1)
    match("uniform_vs_trained_player1", leg.uniform_policy, second, args.seed + 3)
    match("initial_player0_vs_trained_player1", initial_player0, second, args.seed + 4)
    save()
    print(f"Report: {args.output}", flush=True)


if __name__ == "__main__":
    main()
