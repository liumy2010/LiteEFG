"""Independent Goofspiel rule, information and device-execution checks."""

import itertools

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from LiteEFG.drl.env import Goofspiel


def _play(env, point_cards, actions):
    state = env.init(jax.random.key(0))._replace(point_cards=point_cards)

    def advance(current, bids):
        next_state, reward, done = env.step(current, bids, jax.random.key(0))
        return next_state, (reward, done)

    return jax.lax.scan(advance, state, actions)


@pytest.mark.parametrize("imp_info", [False, True])
@pytest.mark.parametrize("points_order", ["random", "ascending", "descending"])
def test_all_three_card_games_match_openspiel(imp_info, points_order):
    """Enumerate bid permutations and chance permutations, including all ties."""
    pyspiel = pytest.importorskip("pyspiel")
    n = 3
    env = Goofspiel(n, imp_info=imp_info, points_order=points_order)
    oracle_game = pyspiel.load_game("goofspiel", {
        "num_cards": n,
        "players": 2,
        "imp_info": imp_info,
        "points_order": points_order,
        "returns_type": "point_difference",
    })
    bid_orders = list(itertools.permutations(range(n)))
    if points_order == "random":
        prizes = list(itertools.permutations(range(1, n + 1)))
    elif points_order == "ascending":
        prizes = [tuple(range(1, n + 1))]
    else:
        prizes = [tuple(range(n, 0, -1))]
    cases = list(itertools.product(prizes, bid_orders, bid_orders))
    cards = np.asarray([case[0] for case in cases], dtype=np.int32)
    actions = np.asarray([
        list(zip(case[1], case[2])) for case in cases
    ], dtype=np.int32)
    expected = []
    for prize_order, first_bids, second_bids in cases:
        oracle = oracle_game.new_initial_state()
        for round_index in range(n):
            if oracle.is_terminal():
                # OpenSpiel resolves the last forced bid automatically.
                assert round_index == n - 1
                break
            if oracle.is_chance_node():
                oracle.apply_action(prize_order[round_index] - 1)
            for player, bids in enumerate((first_bids, second_bids)):
                assert oracle.legal_actions(player) == sorted(bids[round_index:])
            oracle.apply_actions([first_bids[round_index], second_bids[round_index]])
        assert oracle.is_terminal()
        expected.append(oracle.returns())

    batched_play = jax.jit(jax.vmap(lambda c, a: _play(env, c, a)))
    state, (rewards, done) = batched_play(jnp.asarray(cards), jnp.asarray(actions))
    np.testing.assert_array_equal(rewards[:, :-1], 0)
    np.testing.assert_array_equal(rewards[:, -1], expected)
    np.testing.assert_array_equal(done[:, :-1], False)
    np.testing.assert_array_equal(done[:, -1], True)
    np.testing.assert_array_equal(state.hands, False)
    np.testing.assert_array_equal(state.bids, actions)
    np.testing.assert_array_equal(np.sum(rewards, axis=(1, 2)), 0)


def test_ties_discard_prize_and_point_difference_is_centered_score():
    env = Goofspiel(3, points_order="descending")
    state, (rewards, _) = jax.jit(lambda: _play(
        env, jnp.array([3, 2, 1]), jnp.array([[2, 2], [1, 0], [0, 1]])
    ))()
    np.testing.assert_array_equal(state.scores, [2, 1])
    np.testing.assert_array_equal(state.winners, [2, 0, 1])
    np.testing.assert_array_equal(rewards[-1], [0.5, -0.5])


@pytest.mark.parametrize("imp_info", [False, True])
def test_future_prizes_do_not_leak(imp_info):
    env = Goofspiel(4, imp_info=imp_info)
    state = env.init(jax.random.key(0))._replace(point_cards=jnp.array([1, 2, 3, 4]))
    other = state._replace(point_cards=jnp.array([1, 2, 4, 3]))
    np.testing.assert_array_equal(env.features(state)["information_set"],
                                  env.features(other)["information_set"])
    assert not np.array_equal(env.features(state)["full_info"],
                              env.features(other)["full_info"])
    state, _, _ = env.step(state, jnp.array([0, 1]), jax.random.key(0))
    other, _, _ = env.step(other, jnp.array([0, 1]), jax.random.key(0))
    np.testing.assert_array_equal(env.features(state)["information_set"],
                                  env.features(other)["information_set"])
    assert not np.array_equal(env.features(state)["full_info"],
                              env.features(other)["full_info"])
    state, _, _ = env.step(state, jnp.array([1, 0]), jax.random.key(0))
    other, _, _ = env.step(other, jnp.array([1, 0]), jax.random.key(0))
    assert not np.array_equal(env.features(state)["information_set"],
                              env.features(other)["information_set"])


def test_imperfect_information_hides_opponent_bids_and_hand():
    private = Goofspiel(4, imp_info=True, points_order="descending")
    public = Goofspiel(4, imp_info=False, points_order="descending")
    initial = private.init(jax.random.key(0))
    first, _, _ = private.step(initial, jnp.array([3, 0]), jax.random.key(0))
    second, _, _ = private.step(initial, jnp.array([3, 1]), jax.random.key(0))
    np.testing.assert_array_equal(private.features(first)["information_set"][0],
                                  private.features(second)["information_set"][0])
    assert not np.array_equal(private.features(first)["information_set"][1],
                              private.features(second)["information_set"][1])
    assert not np.array_equal(public.features(first)["information_set"][0],
                              public.features(second)["information_set"][0])
    assert not np.array_equal(private.features(first)["full_info"][0],
                              private.features(second)["full_info"][0])
    for state in (first, second):
        np.testing.assert_array_equal(private.features(state)["full_info"],
                                      public.features(state)["full_info"])


@pytest.mark.parametrize("imp_info", [False, True])
def test_full_info_encodes_complete_state_relative_to_each_player(imp_info):
    env = Goofspiel(4, imp_info=imp_info)
    state = env.init(jax.random.key(0))._replace(point_cards=jnp.array([2, 4, 1, 3]))
    for actions in ([3, 0], [1, 2]):
        state, _, _ = env.step(state, jnp.array(actions), jax.random.key(0))
    features = env.features(state)
    assert set(features) == {"information_set", "full_info"}
    assert features["information_set"].shape == (2, 55 if imp_info else 75)
    full_info = np.asarray(features["full_info"])
    assert full_info.shape == (2, 75)
    assert full_info.dtype == np.float32
    # Check every block independently, including private opponent history and
    # prizes that have not yet been revealed to either player.
    bids = np.zeros((4, 2, 4), dtype=np.float32)
    bids[0, 0, 3] = bids[0, 1, 0] = 1
    bids[1, 0, 1] = bids[1, 1, 2] = 1
    for player in range(2):
        vector = full_info[player]
        opponent = 1 - player
        np.testing.assert_array_equal(vector[:5], [0, 0, 1, 0, 0])
        np.testing.assert_array_equal(vector[5:9], state.hands[player])
        np.testing.assert_array_equal(vector[9:25].reshape(4, 4),
                                      np.eye(4)[[1, 3, 0, 2]])
        np.testing.assert_array_equal(vector[25:41].reshape(4, 4), bids[:, player])
        winners = np.zeros((4, 3), dtype=np.float32)
        winners[0, player] = winners[1, opponent] = 1
        np.testing.assert_array_equal(vector[41:53].reshape(4, 3), winners)
        np.testing.assert_array_equal(vector[53:55],
                                      np.asarray(state.scores)[[player, opponent]] / 10)
        np.testing.assert_array_equal(vector[55:59], state.hands[opponent])
        np.testing.assert_array_equal(vector[59:75].reshape(4, 4), bids[:, opponent])
    if not imp_info:
        # In public games, the sole encoding difference is future prizes.
        visible = full_info.copy()
        visible[:, 9:25].reshape(2, 4, 4)[:, 3] = 0
        np.testing.assert_array_equal(features["information_set"], visible)


def test_full_info_preserves_private_opponent_bid_order():
    env = Goofspiel(4, imp_info=True, points_order="descending")
    states = []
    for history in ([[3, 0], [2, 1]], [[3, 1], [2, 0]]):
        state = env.init(jax.random.key(0))
        for actions in history:
            state, _, _ = env.step(state, jnp.array(actions), jax.random.key(0))
        states.append(state)
    for field in ("round", "point_cards", "hands", "winners", "scores"):
        np.testing.assert_array_equal(getattr(states[0], field), getattr(states[1], field))
    first, second = map(env.features, states)
    np.testing.assert_array_equal(first["information_set"][0], second["information_set"][0])
    for player in range(2):
        assert not np.array_equal(first["full_info"][player], second["full_info"][player])


@pytest.mark.parametrize("imp_info", [False, True])
def test_observation_preserves_own_ordered_history(imp_info):
    env = Goofspiel(4, imp_info=imp_info, points_order="descending")
    initial = env.init(jax.random.key(0))
    histories = ([[0, 3], [1, 2]], [[1, 3], [0, 2]])
    states = []
    for history in histories:
        current = initial
        for actions in history:
            current, _, _ = env.step(current, jnp.array(actions), jax.random.key(0))
        states.append(current)
    np.testing.assert_array_equal(states[0].hands, states[1].hands)
    np.testing.assert_array_equal(states[0].scores, states[1].scores)
    np.testing.assert_array_equal(states[0].winners, states[1].winners)
    assert not np.array_equal(env.features(states[0])["information_set"][0],
                              env.features(states[1])["information_set"][0])


@pytest.mark.parametrize("imp_info", [False, True])
def test_batched_seven_card_rollout_jits_and_terminates(imp_info):
    env = Goofspiel(7, imp_info=imp_info)
    batch = 32
    keys = jax.random.split(jax.random.key(0), batch)

    def rollout(key):
        initial = env.init(key)

        def advance(state, unused):
            del unused
            legal = env.legal_action_mask(state)
            first = 6 - jnp.argmax(legal[0, ::-1])
            second = jnp.argmax(legal[1])
            next_state, reward, _ = env.step(state, jnp.stack((first, second)), key)
            return next_state, (reward, env.features(state), legal)

        return jax.lax.scan(advance, initial, None, length=env.max_steps)

    state, (reward, features, legal) = jax.jit(jax.vmap(rollout))(keys)
    observation = features["information_set"]
    # Round (8), own hand (7), prizes (49), own bids (49), winners (21),
    # scores (2), plus opponent hand (7) and bids (49) in public-bid games.
    assert observation.shape == (batch, 7, 2, 136 if imp_info else 192)
    assert observation.dtype == jnp.float32
    assert np.isfinite(observation).all()
    assert features["full_info"].shape == (batch, 7, 2, 192)
    assert features["full_info"].dtype == jnp.float32
    assert np.isfinite(features["full_info"]).all()
    np.testing.assert_array_equal(np.sum(legal, axis=-1),
                                  np.broadcast_to(np.arange(7, 0, -1)[None, :, None],
                                                  (batch, 7, 2)))
    np.testing.assert_array_equal(state.round, 7)
    np.testing.assert_array_equal(np.sum(reward, axis=-1), 0)
    np.testing.assert_array_equal(jax.vmap(env.active_players)(state), False)
    np.testing.assert_array_equal(jax.vmap(env.legal_action_mask)(state), False)
    assert np.unique(np.asarray(state.point_cards), axis=0).shape[0] > 1


def test_terminal_state_absorbs_invalid_dummy_actions_without_rewards():
    env = Goofspiel(1)
    state, reward, done = jax.jit(env.step)(
        env.init(jax.random.key(0)), jnp.array([0, 0]), jax.random.key(0)
    )
    assert bool(done)
    np.testing.assert_array_equal(reward, [0, 0])
    after, reward, done = jax.jit(env.step)(
        state, jnp.array([-100, 100]), jax.random.key(0)
    )
    for before_leaf, after_leaf in zip(jax.tree.leaves(state), jax.tree.leaves(after)):
        np.testing.assert_array_equal(before_leaf, after_leaf)
    np.testing.assert_array_equal(reward, [0, 0])
    assert bool(done)
    assert np.isfinite(env.features(after)["information_set"]).all()


@pytest.mark.parametrize("kwargs", [
    {"num_cards": 0}, {"num_cards": -1}, {"num_cards": 2.5}, {"num_cards": True},
    {"imp_info": "False"}, {"points_order": "invalid"}, {"returns_type": "win_loss"},
])
def test_invalid_configurations_fail_before_compilation(kwargs):
    with pytest.raises(ValueError):
        Goofspiel(**kwargs)
