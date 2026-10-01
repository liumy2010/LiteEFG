"""Train simple MLP actors and one shared MLP critic for DarkChess."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flax import serialization
import jax
import optax

import LiteEFG as leg
from LiteEFG.baselines.drl import PPO
from LiteEFG.drl.models import MLP


def build_graph(env, *, learning_rate=3e-4, hidden_size=128):
    """Independent actors by seat, with a single current-mover critic."""
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size < 1:
        raise ValueError("hidden_size must be a positive integer")

    def optimizer():
        return optax.chain(optax.clip_by_global_norm(0.5), optax.adam(learning_rate))

    policy_network = [leg.model(
        MLP(output_size=env.num_actions, hidden_size=hidden_size),
        optimizer=optimizer(),
    ) for _ in range(env.num_players)]
    critic_network = leg.model(MLP(output_size=1, hidden_size=hidden_size),
                               optimizer=optimizer())
    return PPO.graph(
        policy_network, critic_network, num_actions=env.num_actions,
        critic_feature="full_info",
    )


def save_models(graph, output):
    """Export actor-only Flax variables separately from the shared critic."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for player in range(2):
        path = output / f"actor-player{player}.msgpack"
        path.write_bytes(serialization.to_bytes(graph.actor[player].state.params))
        paths.append(path)
    critic_path = output / "critic.msgpack"
    critic_path.write_bytes(serialization.to_bytes(graph.critic.state.params))
    return paths, critic_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-length", type=int, default=8)
    parser.add_argument("--max-plies", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--minibatch-size", type=int, default=128,
                        help="Complete trajectories per optimizer minibatch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128,
                        help="Units in each of the two actor and critic hidden layers")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/dark-chess/ppo")
    args = parser.parse_args()
    for name in ("history_length", "max_plies", "iterations", "batch_size", "epochs", "minibatch_size", "hidden_size"):
        if getattr(args, name) < 1:
            parser.error(f"{name.replace('_', '-')} must be positive")
    env = leg.DarkChess(max_plies=args.max_plies, history_length=args.history_length)
    graph = build_graph(env, learning_rate=args.learning_rate, hidden_size=args.hidden_size)
    graph.bind(env, batch_size=args.batch_size, seed=args.seed)
    metrics = graph.train(iterations=args.iterations, epochs=args.epochs,
                          minibatch_size=args.minibatch_size)
    actors, critic = save_models(graph, args.output)
    report = {
        "environment": "MIT Fog of War chess",
        "configuration": {name: str(value) if isinstance(value, Path) else value
                          for name, value in vars(args).items()},
        "critic": {"shared": True, "perspective": "current mover", "architecture": "MLP",
                   "full_info_size": env.full_info_size, "repetition_encoding": "current count only",
                   "hidden_sizes": [args.hidden_size, args.hidden_size], "activation": "relu"},
        "actor": {"inputs": ["information_set"], "shared": False, "architecture": "MLP",
                  "hidden_sizes": [args.hidden_size, args.hidden_size], "activation": "relu"},
        "metrics": metrics, "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "actor_variables": [str(path) for path in actors], "critic_variables": str(critic),
    }
    (args.output / "run.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                          encoding="utf-8")
    print(json.dumps(metrics, allow_nan=False))


if __name__ == "__main__":
    main()
