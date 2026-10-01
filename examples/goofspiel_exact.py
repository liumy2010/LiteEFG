"""Import frozen JAX policies into the C++ evaluator for small Goofspiel games."""

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
import tempfile
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

import LiteEFG as leg
from LiteEFG.drl.env.goofspiel import Goofspiel, GoofspielState
from LiteEFG.drl.runtime import Policy, _environment_features


def _size(num_cards):
    """Count the full tree algebraically, eliding the final forced round."""
    chance = infosets = sequences = 0
    prefix = 1
    for remaining in range(num_cards, 1, -1):
        chance += prefix
        infosets += prefix * remaining
        sequences += prefix * remaining * remaining
        prefix *= remaining ** 3
    terminals = math.factorial(num_cards) ** 3
    return {
        "num_cards": num_cards,
        "nodes": chance + infosets + sequences + terminals,
        "terminal_nodes": terminals,
        "information_sets": [infosets, infosets],
        "total_information_sets": 2 * infosets,
        "sequences": [sequences, sequences],
        "forced_final_round_elided": True,
    }


def _write_tree(path, num_cards):
    """Preserve public histories and hide P0's current bid from P1."""
    contexts, information_sets = [], []
    node_count = 0

    def new_node():
        nonlocal node_count
        name = f"n{node_count}"
        node_count += 1
        return name

    def score(scores, card, first, second):
        return (scores[0] + (card if first > second else 0),
                scores[1] + (card if second > first else 0))

    with Path(path).open("w", encoding="utf-8", newline="\n") as output:
        # 'openspiel' selects FileEnv's explicit child-name syntax only. No
        # OpenSpiel game, information-state key or tabular policy is used.
        output.write("# Opt {\n# openspiel,\n# players: 2,\n# }\n")

        def chance_node(name, history, prizes, first_hand, second_hand, scores):
            if len(prizes) == 1:
                final = score(scores, prizes[0], first_hand[0], second_hand[0])
                utility = (final[0] - final[1]) / 2
                output.write(f"node {name} leaf payoffs 1={utility:.17g} 2={-utility:.17g}\n")
                return
            children = [new_node() for _ in prizes]
            probability = 1 / len(prizes)
            actions = " ".join(f"{child}={probability:.17g}" for child in children)
            output.write(f"node {name} chance actions {actions}\n")
            for card, first_node in zip(prizes, children):
                context = len(contexts)
                contexts.append((history, card))
                second_nodes = [new_node() for _ in first_hand]
                output.write(f"node {first_node} player 1 actions {' '.join(second_nodes)}\n")
                information_sets.append((f"p1_h{context}", [first_node]))
                # All hidden first bids belong to the same P1 information set.
                information_sets.append((f"p2_h{context}", second_nodes))
                for first, second_node in zip(first_hand, second_nodes):
                    next_nodes = [new_node() for _ in second_hand]
                    output.write(f"node {second_node} player 2 actions {' '.join(next_nodes)}\n")
                    for second, next_node in zip(second_hand, next_nodes):
                        chance_node(
                            next_node, history + ((card, first, second),),
                            tuple(value for value in prizes if value != card),
                            tuple(value for value in first_hand if value != first),
                            tuple(value for value in second_hand if value != second),
                            score(scores, card, first, second),
                        )

        chance_node(new_node(), (), tuple(range(1, num_cards + 1)),
                    tuple(range(num_cards)), tuple(range(num_cards)), (0, 0))
        for name, members in information_sets:
            output.write(f"infoset {name} nodes {' '.join(members)}\n")
    return contexts, node_count


def _states_for_contexts(contexts, num_cards):
    """Create exactly the JAX environment state seen before either current bid."""
    batch = len(contexts)
    rounds = np.zeros(batch, np.int32)
    cards = np.zeros((batch, num_cards), np.int32)
    hands = np.ones((batch, 2, num_cards), bool)
    bids = np.full((batch, num_cards, 2), -1, np.int32)
    winners = np.full((batch, num_cards), -1, np.int32)
    scores = np.zeros((batch, 2), np.float32)
    for index, (history, current_card) in enumerate(contexts):
        rounds[index] = len(history)
        revealed = [record[0] for record in history] + [current_card]
        cards[index] = revealed + [card for card in range(1, num_cards + 1)
                                   if card not in revealed]
        for turn, (card, first, second) in enumerate(history):
            hands[index, 0, first] = False
            hands[index, 1, second] = False
            bids[index, turn] = (first, second)
            winner = 2 if first == second else (0 if first > second else 1)
            winners[index, turn] = winner
            if winner < 2:
                scores[index, winner] += card
    return GoofspielState(*(jnp.asarray(value) for value in
                            (rounds, cards, hands, bids, winners, scores)))


class _ImportedStrategy(leg.Graph):
    def __init__(self):
        super().__init__()
        with leg.backward(is_static=True):
            self.strategy = leg.const(self.action_set_size, 1.0 / self.action_set_size)


def exact_exploitability(env, policy0, policy1, *, max_nodes=1_000_000, batch_size=4096):
    """Compute exact full-history exploitability using native C++ best response.

    This supports public-bid, random-prize, point-difference Goofspiel. The
    complete tree is counted before construction and rejected above max_nodes.
    Models are frozen callable policies returning legal action probabilities;
    features use the same encoder as training. No probability is truncated.
    Native calculations use double precision and double-normalized JAX outputs.
    """
    if not isinstance(env, Goofspiel) or env.imp_info or env.points_order != "random":
        raise ValueError("Exact evaluation requires public-bid Goofspiel with random prizes")
    if env.returns_type != "point_difference":
        raise ValueError("Exact evaluation requires point_difference returns")
    for name, value in (("max_nodes", max_nodes), ("batch_size", batch_size)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    size = _size(env.num_cards)
    if size["nodes"] > max_nodes:
        raise ValueError(f"Exact tree has {size['nodes']:,} nodes, above max_nodes={max_nodes:,}; "
                         "use a smaller game or an explicitly larger evaluation budget")
    feature_specs = jax.eval_shape(
        lambda key: _environment_features(env, env.init(key)), jax.random.key(0))
    information_set_size = feature_specs["information_set"].shape[-1]
    for policy in (policy0, policy1):
        if isinstance(policy, Policy) and (policy.observation_size != information_set_size
                                          or policy.num_actions != env.num_actions):
            raise ValueError("Policy and environment input/output dimensions differ")
    if env.num_cards == 1:
        return {"utility": [0.0, 0.0], "deviation_gain": [0.0, 0.0],
                "best_response_utility": [0.0, 0.0], "nash_conv": 0.0,
                "exploitability": 0.0, "size": size,
                "evaluator": "Analytic forced single-card tie"}

    with tempfile.TemporaryDirectory(prefix="liteefg-goofspiel-exact-") as temporary:
        path = Path(temporary) / "goofspiel.game"
        contexts, node_count = _write_tree(path, env.num_cards)
        if node_count != size["nodes"] or len(contexts) != size["information_sets"][0]:
            raise RuntimeError("Generated tree does not match the independent size formula")
        native = leg.FileEnv(str(path))
        graph = _ImportedStrategy()
        native.set_graph(graph)
        for player, policy in enumerate((policy0, policy1)):
            order = native.get_value(player + 1, graph.strategy)

            @jax.jit
            def predict(states):
                features = jax.vmap(lambda state: _environment_features(env, state))(states)
                features = {name: value[:, player] for name, value in features.items()}
                masks = jax.vmap(env.legal_action_mask)(states)[:, player]
                probabilities = (policy(features, masks, player=player)
                                 if isinstance(policy, Policy)
                                 else policy(features["information_set"], masks))
                return probabilities, masks

            imported = []
            for start in range(0, len(order), batch_size):
                entries = order[start:start + batch_size]
                selected = [contexts[int(name.split("_h", 1)[1])] for name, _ in entries]
                states = _states_for_contexts(selected, env.num_cards)
                probabilities, masks = jax.device_get(predict(states))
                probabilities = np.asarray(probabilities, dtype=np.float64)
                if probabilities.shape != masks.shape or not np.isfinite(probabilities).all():
                    raise ValueError("Policy returned invalid probability shape or non-finite values")
                if (np.any(probabilities < 0) or np.any(probabilities[~masks] != 0)
                        or np.any(np.abs(probabilities.sum(axis=-1) - 1) > 1e-4)):
                    raise ValueError("Policy returned negative, unnormalized or illegal probabilities")
                for row, mask, (_, template) in zip(probabilities, masks, entries):
                    legal = row[mask]
                    if len(legal) != len(template) or legal.sum() <= 0:
                        raise ValueError("Policy actions do not match the native information set")
                    imported.append((legal / legal.sum()).tolist())
            native.set_value(player + 1, graph.strategy, imported)
        utility = np.asarray(native.utility(graph.strategy), dtype=np.float64)
        gains = np.asarray(native.exploitability(graph.strategy), dtype=np.float64)
    if not np.isfinite(utility).all() or not np.isfinite(gains).all() or np.any(gains < -1e-8):
        raise FloatingPointError("Native exact evaluation returned invalid results")
    return {"utility": utility.tolist(), "deviation_gain": gains.tolist(),
            "best_response_utility": (utility + gains).tolist(),
            "nash_conv": float(gains.sum()), "exploitability": float(gains.sum() / 2),
            "size": size, "evaluator": "LiteEFG FileEnv C++ exact sequence-form best response"}


def main():
    from flax import linen as nn
    from LiteEFG.baselines.drl import PPO

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player0", required=True, type=Path)
    parser.add_argument("--player1", required=True, type=Path)
    parser.add_argument("--cards", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-nodes", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/drl/goofspiel-4-exact.json")
    args = parser.parse_args()
    if args.hidden_size < 1:
        parser.error("hidden-size must be positive")

    def new_graph():
        def network(outputs):
            return nn.Sequential([nn.Dense(args.hidden_size), nn.tanh,
                                  nn.Dense(args.hidden_size), nn.tanh, nn.Dense(outputs)])
        policy_network = [leg.model(network(args.cards)) for _ in range(2)]
        critic_network = [leg.model(network(1)) for _ in range(2)]
        return PPO.graph(policy_network, critic_network, num_actions=args.cards)

    policies = [Policy.load(new_graph(), path) for path in (args.player0, args.player1)]
    started = time.perf_counter()
    report = exact_exploitability(Goofspiel(args.cards), *policies,
                                  max_nodes=args.max_nodes, batch_size=args.batch_size)
    sources = ["examples/goofspiel_exact.py", "LiteEFG/drl/environment.py",
               "LiteEFG/drl/env/goofspiel.py",
               "LiteEFG/drl/graph.py", "LiteEFG/drl/runtime.py", "LiteEFG/baselines/drl/PPO.py",
               "LiteEFG/src/Environment/Environment.cpp", "LiteEFG/src/Environment/SequenceForm.cpp"]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    report.update({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seconds_including_compile": time.perf_counter() - started,
        "engine_commit": revision.stdout.strip(),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                          for name in sources},
        "native_binary_sha256": hashlib.sha256(Path(leg._LiteEFG.__file__).read_bytes()).hexdigest(),
        "policies": [{"path": str(path.resolve()),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                     for path in (args.player0, args.player1)],
        "configuration": {name: str(value) if isinstance(value, Path) else value
                          for name, value in vars(args).items()},
        "game": {"name": "goofspiel", "num_cards": args.cards, "imp_info": False,
                 "points_order": "random", "returns_type": "point_difference",
                 "information": "Complete ordered public prize and bid history"},
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
