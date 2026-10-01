"""Per-update traversal overrides must also select the matching payoff weights."""

from itertools import permutations

import pytest

import LiteEFG as leg


TRAVERSALS = ["Enumerate", "Outcome", "External"]


@pytest.fixture
def chance_game(tmp_path):
    # Chance picks one of two payoff matrices, unseen by either player.
    # Both players use uniform strategies, with a single infoset each.
    matrices = [[[4, -2], [6, 0]], [[2, 8], [-4, 10]]]
    records = [
        "# Opt {", "#     openspiel,", "#     players: 2,", "# }",
        "node start chance actions root0=0.25 root1=0.75",
    ]
    for chance, matrix in enumerate(matrices):
        records.append(f"node root{chance} player 1 actions a{chance} b{chance}")
        for action, row in zip("ab", matrix):
            records.append(
                f"node {action}{chance} player 2 actions {action}{chance}0 {action}{chance}1")
            for reply, payoff in enumerate(row):
                records.append(
                    f"node {action}{chance}{reply} leaf payoffs 1={payoff} 2={-payoff}")
    records += [
        "infoset p1 nodes root0 root1",
        "infoset p2 nodes a0 b0 a1 b1",
    ]
    path = tmp_path / "chance.game"
    path.write_text("\n".join(records) + "\n")
    return path


def utility_graph():
    graph = leg.Graph()
    with leg.backward(is_static=True):
        strategy = leg.const(graph.action_set_size, 1.0 / graph.action_set_size)
        observed = leg.const(graph.action_set_size, 0.0)
    with leg.backward():
        observed.inplace(graph.utility.copy())
    return graph, strategy, observed


def snapshot(env, observed):
    return [env.get_value(player, observed)[0][1] for player in (1, 2)]


@pytest.mark.parametrize("configured", TRAVERSALS)
def test_enumerate_override_matches_hand_calculated_counterfactual_values(
        chance_game, configured):
    graph, strategy, observed = utility_graph()
    env = leg.FileEnv(str(chance_game), traverse_type=configured)
    env.set_graph(graph)
    env.update(strategy, traverse_type="Enumerate")
    # P1: .25 * [1, 3] + .75 * [5, 3].
    # P2: .25 * [-5, 1] + .75 * [1, -9].
    actual = snapshot(env, observed)
    assert actual[0] == pytest.approx([4.0, 3.0])
    assert actual[1] == pytest.approx([-0.5, -6.5])


@pytest.mark.parametrize("configured,override", list(permutations(TRAVERSALS, 2)))
@pytest.mark.parametrize("updated_player", [-1, 1, 2])
def test_override_matches_native_mode_and_preserves_default(
        chance_game, configured, override, updated_player):
    graph, strategy, observed = utility_graph()
    env = leg.FileEnv(str(chance_game), traverse_type=configured)
    reference = leg.FileEnv(str(chance_game), traverse_type=override)
    env.set_graph(graph)
    reference.set_graph(graph)

    for current in (override, configured):
        leg.set_seed(0)
        env.update(strategy, upd_player=updated_player,
                   **({"traverse_type": current} if current == override else {}))
        leg.set_seed(0)
        reference.update(strategy, upd_player=updated_player, traverse_type=current)
        for actual, expected in zip(snapshot(env, observed), snapshot(reference, observed)):
            assert actual == pytest.approx(expected)
