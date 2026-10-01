"""Dark chess API, private history, and JAX sampler integration."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from LiteEFG.drl.env import DarkChess
from LiteEFG.drl.env.dark_chess import action_from_uci, action_to_uci
from LiteEFG.drl.runtime import rollout, uniform_policy


def _advance(env, state, move):
    actions = jnp.full(2, -1, dtype=jnp.int32).at[state.core.turn].set(action_from_uci(move))
    return env.step(state, actions, jax.random.key(0))


@pytest.mark.parametrize("kwargs", [
    {"max_plies": 0}, {"max_plies": True}, {"max_plies": 2.5},
    {"history_length": 0}, {"history_length": False}, {"history_length": 1.5},
    {"history_length": None},
])
def test_configuration_rejects_invalid_sizes(kwargs):
    with pytest.raises(ValueError):
        DarkChess(**kwargs)


@pytest.mark.parametrize("move", ["e2e4", "g1f3", "e1g1"])
def test_host_uci_round_trip(move):
    env = DarkChess()
    state = env.init(jax.random.key(0))
    assert action_to_uci(action_from_uci(move), state) == move


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("piece", ["q", "r", "b", "n"])
def test_all_promotion_action_round_trips(side, piece):
    env = DarkChess()
    state = env.init(jax.random.key(0))
    source, target = (48, 56) if side == 0 else (8, 0)
    board = state.core.board.at[source].set(1 - 2 * side)
    state = state._replace(core=state.core._replace(board=board))
    move = ("a7a8" if side == 0 else "a2a1") + piece
    assert action_to_uci(action_from_uci(move), state) == move
    assert (action_from_uci(move) < 4096) == (piece == "q")


@pytest.mark.parametrize("move", ["", "E2E4", "a0a1", "a1a1", "a2a3q", "a7c8n", "e2e4x", None])
def test_invalid_uci_rejected(move):
    with pytest.raises(ValueError):
        action_from_uci(move)


@pytest.mark.parametrize("action", [-1, 4228, True, 1.5])
def test_invalid_action_id_rejected(action):
    env = DarkChess()
    with pytest.raises(ValueError):
        action_to_uci(action, env.init(jax.random.key(0)))


def test_turns_horizon_reward_and_absorbing_history():
    env = DarkChess(max_plies=2, history_length=3)
    state = env.init(jax.random.key(0))
    np.testing.assert_array_equal(env.active_players(state), [True, False])
    np.testing.assert_array_equal(env.legal_action_mask(state).sum(-1), [20, 0])
    state, reward, done = _advance(env, state, "e2e4")
    assert not done
    np.testing.assert_array_equal(reward, [0, 0])
    np.testing.assert_array_equal(env.active_players(state), [False, True])
    state, reward, done = _advance(env, state, "e7e5")
    assert done
    np.testing.assert_array_equal(reward, [0, 0])
    assert not env.active_players(state).any()
    assert not env.legal_action_mask(state).any()
    after, reward, done = env.step(state, jnp.array([-100, -100]), jax.random.key(0))
    assert done
    for old, new in zip(jax.tree.leaves(state), jax.tree.leaves(after)):
        np.testing.assert_array_equal(old, new)
    np.testing.assert_array_equal(reward, [0, 0])


def test_king_danger_pays_once_and_after_players_response():
    env = DarkChess(history_length=2)
    state = env.init(jax.random.key(0))
    for move in ("f2f3", "e7e5", "g2g4", "d8h4"):
        state, reward, done = _advance(env, state, move)
        assert not done
        np.testing.assert_array_equal(reward, 0)
    # White can play into danger, but immediately loses after that move.
    state, reward, done = _advance(env, state, "a2a3")
    assert done
    np.testing.assert_array_equal(reward, [-1, 1])
    _, reward, _ = env.step(state, jnp.array([0, 0]), jax.random.key(0))
    np.testing.assert_array_equal(reward, 0)


def test_both_players_observe_every_ply_and_history_is_bounded():
    env = DarkChess(history_length=3)
    state = env.init(jax.random.key(0))
    snapshots = [np.asarray(env.observation(state))]
    for move in ("e2e4", "e7e5", "g1f3", "b8c6"):
        state, _, _ = _advance(env, state, move)
        snapshots.insert(0, np.asarray(env.observation(state)))
        count = min(3, len(snapshots))
        np.testing.assert_array_equal(state.observation_history[:count], snapshots[:count])
    assert state.observation_history.shape == (3, 2, 64)


@pytest.mark.parametrize("history_length", [1, 3, 8])
def test_feature_shapes_channels_orientation_and_padding(history_length):
    env = DarkChess(history_length=history_length)
    state = env.init(jax.random.key(0))
    features = jax.jit(env.features)(state)
    assert set(features) == {"information_set", "full_info", "history_mask"}
    assert features["information_set"].shape == (2, history_length * 896 + 1)
    assert features["full_info"].shape == (2, (1 + 2 * history_length) * 896 + 2 * history_length + 73)
    assert features["full_info"].shape[-1] == env.full_info_size
    assert env.critic_board_count == 1 + 2 * history_length
    assert env.critic_metadata_size == 73
    assert features["history_mask"].shape == (2, history_length)
    np.testing.assert_array_equal(features["information_set"][:, -1], [1, 0])
    assert features["information_set"].dtype == features["full_info"].dtype == jnp.float32
    assert features["history_mask"].dtype == jnp.bool_
    boards = np.asarray(features["information_set"][:, :-1]).reshape(2, history_length, 14, 8, 8)
    np.testing.assert_array_equal(boards, env.board_features(state))
    np.testing.assert_array_equal(features["history_mask"][:, 0], True)
    np.testing.assert_array_equal(features["history_mask"][:, 1:], False)
    np.testing.assert_array_equal(boards[:, 1:], 0)
    np.testing.assert_array_equal(boards[:, 0].sum(1), 1)
    # The initial views are identical after swapping colors and flipping ranks.
    np.testing.assert_array_equal(boards[0, 0], boards[1, 0])
    np.testing.assert_array_equal(boards[:, 0, 0, 1], 1)  # Own pawns.
    np.testing.assert_array_equal(boards[:, 0, 5, 0, 4], 1)  # Own king on e-file.
    np.testing.assert_array_equal(boards[:, 0, 12, 2:4], 1)  # Seen empty ranks.
    np.testing.assert_array_equal(boards[:, 0, 13, 4:], 1)  # Unseen ranks.
    complete = np.asarray(env.full_board_features(state))
    np.testing.assert_array_equal(complete.sum(1), 1)
    np.testing.assert_array_equal(complete[:, 13], 0)
    np.testing.assert_array_equal(complete[:, 6, 6], 1)  # Opponent pawns.
    np.testing.assert_array_equal(complete[:, 11, 7, 4], 1)
    np.testing.assert_array_equal(features["full_info"][0], features["full_info"][1])
    board_end = env.critic_board_count * 896
    critic_boards = features["full_info"][0, :board_end].reshape(-1, 14, 8, 8)
    np.testing.assert_array_equal(critic_boards, env.critic_board_features(state))
    np.testing.assert_array_equal(critic_boards[0], complete[0])
    np.testing.assert_array_equal(features["full_info"][0, board_end:board_end + 2 * history_length],
                                  env.critic_history_mask(state).reshape(-1))
    np.testing.assert_array_equal(features["full_info"][0, -73:], env.critic_metadata(state))


def test_board_encoding_distinguishes_every_observable_square_category():
    values = np.array([1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6, 0, 7])
    absolute = np.resize(values, (2, 2, 64))
    encoded = np.asarray(DarkChess._encode_boards(jnp.asarray(absolute)))
    for player in range(2):
        for time in range(2):
            for rank in range(8):
                for file in range(8):
                    raw_rank = rank if player == 0 else 7 - rank
                    raw = absolute[player, time, raw_rank * 8 + file]
                    own = raw * (1 if player == 0 else -1)
                    channel = (13 if raw == 7 else 12 if raw == 0 else
                               own - 1 if own > 0 else 5 - own)
                    np.testing.assert_array_equal(encoded[player, time, :, rank, file],
                                                  np.eye(14)[channel])


def test_both_actor_and_critic_inputs_preserve_observation_history():
    env = DarkChess(history_length=3)
    state = env.init(jax.random.key(0))
    frames = [np.asarray(env.board_features(state)[:, 0])]
    for move in ("e2e4", "e7e5", "g1f3"):
        state, _, _ = _advance(env, state, move)
        features = env.features(state)
        frames.insert(0, np.asarray(env.board_features(state)[:, 0]))
        boards = np.asarray(features["information_set"][:, :-1]).reshape(2, 3, 14, 8, 8)
        np.testing.assert_array_equal(features["information_set"][:, -1], [1, 0])
        count = min(len(frames), 3)
        np.testing.assert_array_equal(boards[:, :count], np.stack(frames[:count], axis=1))
        np.testing.assert_array_equal(features["history_mask"][:, :count], True)
    changed_history = state._replace(observation_history=state.observation_history.at[1].set(7))
    assert not np.array_equal(env.features(state)["information_set"],
                              env.features(changed_history)["information_set"])
    assert not np.array_equal(env.features(state)["full_info"],
                              env.features(changed_history)["full_info"])
    np.testing.assert_array_equal(env.full_board_features(state), env.full_board_features(changed_history))


@pytest.mark.parametrize("moves", [("e2e4",), ("e2e4", "e7e5")])
def test_critic_sequence_uses_mover_perspective_and_preserves_opponent_visibility(moves):
    env = DarkChess(history_length=4)
    state = env.init(jax.random.key(0))
    for move in moves:
        state, _, _ = _advance(env, state, move)
    mover = int(state.core.turn)
    actor = np.asarray(env.board_features(state))
    critic = np.asarray(jax.jit(env.critic_board_features)(state))
    np.testing.assert_array_equal(critic[0], env.full_board_features(state)[mover])
    np.testing.assert_array_equal(critic[1:5], actor[mover])
    # Swap own/opponent piece channels and reverse ranks only. Empty and unseen
    # squares retain the opponent's information, not the mover's visibility.
    channel_order = np.r_[6:12, 0:6, 12:14]
    opponent_view = actor[1 - mover][:, channel_order, ::-1, :]
    np.testing.assert_array_equal(critic[5:], opponent_view)
    assert not np.array_equal(critic[1, 13], critic[5, 13])
    np.testing.assert_array_equal(critic[1:].sum(1),
                                  np.broadcast_to(env.critic_history_mask(state).reshape(-1, 1, 1),
                                                  (8, 8, 8)))
    np.testing.assert_array_equal(env.features(state)["full_info"][0],
                                  env.features(state)["full_info"][1])


def test_opponent_history_enters_only_privileged_input():
    env = DarkChess(history_length=3)
    state = env.init(jax.random.key(0))
    state, _, _ = _advance(env, state, "e2e4")
    mover = int(state.core.turn)
    changed = state._replace(
        observation_history=state.observation_history.at[1, 1 - mover].set(7),
    )
    before, after = env.features(state), env.features(changed)
    for name in ("information_set", "history_mask"):
        np.testing.assert_array_equal(before[name][mover], after[name][mover])
    assert not np.array_equal(before["full_info"], after["full_info"])
    boards_before, boards_after = env.critic_board_features(state), env.critic_board_features(changed)
    np.testing.assert_array_equal(boards_before[:4], boards_after[:4])
    assert not np.array_equal(boards_before[4:], boards_after[4:])


def test_padding_excludes_stale_storage_from_both_model_inputs():
    env = DarkChess(history_length=3)
    state = env.init(jax.random.key(0))
    changed = state._replace(observation_history=state.observation_history.at[1:].set(1))
    for name in env.features(state):
        np.testing.assert_array_equal(env.features(state)[name], env.features(changed)[name])


@pytest.mark.parametrize("mover", [0, 1])
def test_critic_metadata_encodes_color_rights_ep_and_draw_bookkeeping(mover):
    env = DarkChess(max_plies=200, history_length=2)
    state = env.init(jax.random.key(0))
    rights = jnp.array([[True, False], [False, True]])
    state = state._replace(core=state.core._replace(
        turn=jnp.array(mover, jnp.int32), castling_rights=rights,
        ep_square=jnp.array(43, jnp.int32), halfmove_clock=jnp.array(88, jnp.int32),
        ply=jnp.array(50, jnp.int32), repetition_count=jnp.array(2, jnp.int32),
    ))
    metadata = np.asarray(jax.jit(env.critic_metadata)(state))
    assert metadata.shape == (73,)
    assert metadata.dtype == np.float32
    assert metadata[0] == 1 - mover
    np.testing.assert_array_equal(metadata[1:5], np.asarray(rights)[[mover, 1 - mover]].reshape(-1))
    expected_ep = 43 if mover == 0 else 19  # d6 becomes d3 for black.
    np.testing.assert_array_equal(metadata[5:70], np.eye(65)[expected_ep])
    np.testing.assert_allclose(metadata[70:], [0.88, 0.75, 2 / 3])
    no_ep = state._replace(core=state.core._replace(ep_square=jnp.array(-1, jnp.int32)))
    np.testing.assert_array_equal(env.critic_metadata(no_ep)[5:70], np.eye(65)[64])


def test_compact_critic_metadata_keeps_repetition_history_approximation_explicit():
    env = DarkChess(history_length=2)
    state = env.init(jax.random.key(0))
    changed = state._replace(core=state.core._replace(repetition_keys=jnp.zeros_like(state.core.repetition_keys)))
    np.testing.assert_array_equal(env.features(state)["full_info"], env.features(changed)["full_info"])


def test_features_support_batched_jit_with_different_current_movers():
    env = DarkChess(history_length=2)
    white = env.init(jax.random.key(0))
    black, _, _ = _advance(env, white, "e2e4")
    states = jax.tree.map(lambda a, b: jnp.stack((a, b)), white, black)
    batch = jax.jit(jax.vmap(env.features))(states)
    # The final actor feature is the receiving player's color, independent of
    # which player currently moves or the perspective used by the critic.
    np.testing.assert_array_equal(batch["information_set"][..., -1], [[1, 0], [1, 0]])
    for index, state in enumerate((white, black)):
        for name, expected in env.features(state).items():
            if name == "full_info":
                np.testing.assert_array_equal(batch[name][index, :, :-3], expected[:, :-3])
                # JIT may replace clock division by a reciprocal multiply.
                np.testing.assert_allclose(batch[name][index, :, -3:], expected[:, -3:],
                                           rtol=1e-6, atol=1e-7)
            else:
                np.testing.assert_array_equal(batch[name][index], expected)


def test_hidden_engine_bookkeeping_is_not_an_actor_feature():
    env = DarkChess(history_length=2)
    state = env.init(jax.random.key(0))
    # White cannot observe these internal values or hidden black back-rank pieces.
    changed = state._replace(core=state.core._replace(
        board=state.core.board.at[57].set(-3).at[58].set(-2),
        castling_rights=state.core.castling_rights.at[1].set(False),
        halfmove_clock=jnp.array(88, jnp.int32),
        repetition_count=jnp.array(2, jnp.int32),
    ))
    np.testing.assert_array_equal(env.observation(state)[0], env.observation(changed)[0])
    np.testing.assert_array_equal(env.features(state)["information_set"][0],
                                  env.features(changed)["information_set"][0])
    assert not np.array_equal(env.features(state)["full_info"][0],
                              env.features(changed)["full_info"][0])


def test_jit_vmap_complete_rollout_uses_alternating_legal_actions():
    env = DarkChess(max_plies=6, history_length=2)

    def policy(features, legal):
        return uniform_policy(features, legal), jnp.zeros(features.shape[0])

    keys = jax.random.split(jax.random.key(11), 2)
    records, done = jax.jit(lambda keys: rollout(env, (policy, policy), keys))(keys)
    assert np.asarray(done).all()
    assert records["legal_action_mask"].shape == (6, 2, 2, 4228)
    assert np.asarray(records["_policy_valid"]).all()
    masks = np.asarray(records["legal_action_mask"])
    chosen = np.take_along_axis(masks, np.asarray(records["action"])[..., None], axis=-1)[..., 0]
    assert chosen[np.asarray(records["_valid"])].all()
    np.testing.assert_array_equal(np.asarray(records["reward"]).sum(-1), 0)


def test_mlp_actor_and_privileged_critic_train():
    pytest.importorskip("flax.linen")
    import LiteEFG as leg
    from LiteEFG.baselines.drl import PPO
    from LiteEFG.drl.models import MLP

    env = DarkChess(max_plies=4, history_length=2)
    actors = [leg.model(MLP(env.num_actions, hidden_size=8)) for _ in range(2)]
    critic = leg.model(MLP(1, hidden_size=8))
    graph = PPO.graph(actors, critic, num_actions=env.num_actions, critic_feature="full_info")
    graph.bind(env, batch_size=2, seed=17)
    assert graph.input_specs["information_set"].shape == (1793,)
    assert graph.input_specs["full_info"].shape == (env.full_info_size,)
    before = jax.tree.map(np.asarray, graph.trainer.states)
    metrics = graph.train(iterations=1, epochs=1, minibatch_size=4)
    assert metrics["iteration"] == 1
    assert np.isfinite(metrics["loss"]).all()
    for resource in (*actors, critic):
        index = graph.model_index(resource)
        old = jax.tree.leaves(before[0][index].params)
        new = jax.tree.leaves(resource.state.params)
        assert any(not np.array_equal(a, b) for a, b in zip(old, new))
