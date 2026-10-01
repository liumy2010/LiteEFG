"""Independent checks of public record batching and sample alignment."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pytest
import LiteEFG as leg

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("optax")
pytest.importorskip("flax")


def records(shape=(4, 5)):
    ids = jnp.arange(np.prod(shape)).reshape(shape)
    return tuple({
        "_valid": (ids + player) % 3 != 0,
        "_alive": ids % 4 != 0,
        "id": ids,
        "features": jnp.stack((ids, ids + 100 * (player + 1)), axis=-1),
        "_drl_node_7": 10 * ids + player,
        "local": jnp.zeros(shape + (0,)),
    } for player in range(2))


def joined(batches, player, field):
    return np.concatenate([np.asarray(batch[player][field]) for batch in batches])


def assert_aligned(batches):
    for batch in batches:
        for player, record in enumerate(batch):
            ids = np.asarray(record["id"])
            np.testing.assert_array_equal(record["_valid"], (ids + player) % 3 != 0)
            np.testing.assert_array_equal(record["_alive"], ids % 4 != 0)
            np.testing.assert_array_equal(record["features"][:, 0], ids)
            np.testing.assert_array_equal(record["features"][:, 1], ids + 100 * (player + 1))
            np.testing.assert_array_equal(record["_drl_node_7"], 10 * ids + player)
            assert record["local"].shape == (len(ids), 0)


@pytest.mark.parametrize("shape,batch_size,batch_lengths", [
    ((4, 5), 2, [8, 8, 4]),
    ((20,), 6, [6, 6, 6, 2]),
])
def test_batches_cover_all_records_preserve_alignment_and_keep_tail(shape, batch_size, batch_lengths):
    dataset = records(shape)
    original = jax.tree.map(lambda value: np.asarray(value).copy(), dataset)
    loader = leg.dataloader(dataset, batch_size=batch_size)
    batches = list(loader)
    assert len(loader) == len(batch_lengths)
    assert [batch[0]["_valid"].shape for batch in batches] == [(size,) for size in batch_lengths]
    expected = np.arange(20).reshape(shape).T.reshape(-1)
    for player in range(2):
        np.testing.assert_array_equal(joined(batches, player, "id"), expected)
    assert_aligned(batches)
    assert_aligned(list(loader))
    for before, after in zip(jax.tree.leaves(original), jax.tree.leaves(dataset)):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize("shape,batch_size,expected_episodes", [
    ((4, 5), 2, 4), ((4, 5), 5, 5), ((4, 5), 6, 0),
    ((20,), 6, 18), ((20,), 20, 20), ((20,), 21, 0),
])
def test_drop_last_omits_only_the_incomplete_batch(shape, batch_size, expected_episodes):
    loader = leg.dataloader(list(records(shape)), batch_size, drop_last=True)
    batches = list(loader)
    steps = shape[0] if len(shape) == 2 else 1
    expected_records = expected_episodes * steps
    assert len(loader) == len(batches) == expected_episodes // batch_size
    assert sum(batch[0]["id"].size for batch in batches) == expected_records
    assert_aligned(batches)
    if expected_records:
        expected = np.arange(np.prod(shape)).reshape(shape).T.reshape(-1)[:expected_records]
        np.testing.assert_array_equal(joined(batches, 0, "id"), expected)


@pytest.mark.parametrize("key_factory", [jax.random.PRNGKey, jax.random.key])
@pytest.mark.parametrize("shape", [(32,), (4, 11)])
def test_shuffle_is_reproducible_reiterable_and_keeps_player_fields_aligned(key_factory, shape):
    key = key_factory(42)
    original_key = np.asarray(jax.random.key_data(key)).copy()
    dataset = records(shape)
    left = leg.dataloader(dataset, 7, shuffle=True, generator=key)
    right = leg.dataloader(dataset, 7, shuffle=True, generator=key)
    reference_key = key
    episodes = shape[-1]
    episode_records = np.arange(np.prod(shape)).reshape(shape).T.reshape(episodes, -1)
    previous = None
    for _ in range(2):
        reference_key, shuffle_key = jax.random.split(reference_key)
        expected = episode_records[np.asarray(jax.random.permutation(shuffle_key, episodes))].reshape(-1)
        batches, repeated = list(left), list(right)
        assert_aligned(batches)
        for player in range(2):
            order = joined(batches, player, "id")
            np.testing.assert_array_equal(np.sort(order), np.arange(np.prod(shape)))
            np.testing.assert_array_equal(order, joined(repeated, player, "id"))
            np.testing.assert_array_equal(order, expected)
        order = joined(batches, 0, "id")
        np.testing.assert_array_equal(order, joined(batches, 1, "id"))
        if previous is not None:
            assert not np.array_equal(previous, order)
        previous = order
    np.testing.assert_array_equal(jax.random.key_data(key), original_key)
    np.testing.assert_array_equal(dataset[0]["id"], np.arange(np.prod(shape)).reshape(shape))


def test_variable_length_trajectories_keep_shared_episode_groups_and_all_time_steps():
    steps, episodes = 7, 5
    lengths = np.array([[7, 0, 1, 4, 2], [1, 6, 0, 3, 5]])
    times = jnp.broadcast_to(jnp.arange(steps)[:, None], (steps, episodes))
    episode_ids = jnp.broadcast_to(jnp.arange(episodes)[None, :], (steps, episodes))
    dataset = tuple({
        "_valid": times < player_lengths[None, :],
        "time": times,
        "episode": episode_ids,
        "target": episode_ids * 100 + times * 10 + player,
    } for player, player_lengths in enumerate(lengths))
    batches = list(leg.dataloader(dataset, 2, shuffle=True, generator=jax.random.PRNGKey(13)))
    assert [batch[0]["_valid"].size for batch in batches] == [14, 14, 7]
    visited = []
    for batch in batches:
        chosen = np.asarray(batch[0]["episode"])[::steps]
        visited.extend(chosen)
        for player, record in enumerate(batch):
            np.testing.assert_array_equal(record["episode"], np.repeat(chosen, steps))
            np.testing.assert_array_equal(record["time"], np.tile(np.arange(steps), len(chosen)))
            valid = np.asarray(record["_valid"]).reshape(-1, steps)
            np.testing.assert_array_equal(valid.sum(axis=1), lengths[player, chosen])
            np.testing.assert_array_equal(
                record["target"], np.repeat(chosen, steps) * 100
                + np.tile(np.arange(steps), len(chosen)) * 10 + player,
            )
    np.testing.assert_array_equal(np.sort(visited), np.arange(episodes))


@pytest.mark.parametrize("steps", [1, 3, 9])
def test_512_trajectories_produce_four_groups_of_128_independent_of_length(steps):
    loader = leg.dataloader(records((steps, 512)), batch_size=128)
    batches = list(loader)
    assert len(loader) == len(batches) == 4
    for batch in batches:
        assert batch[0]["_valid"].shape == (steps * 128,)
        np.testing.assert_array_equal(batch[0]["id"], batch[1]["id"])
        assert np.unique(np.asarray(batch[0]["id"]) % 512).size == 128


def test_default_seed_and_interleaved_iterators_are_deterministic():
    dataset = records()
    loader = leg.dataloader(dataset, 3, shuffle=True)
    expected = leg.dataloader(dataset, 3, shuffle=True, generator=jax.random.PRNGKey(0))
    first, second = iter(loader), iter(loader)
    first_batches, second_batches = [], []
    for first_batch, second_batch in zip(first, second):
        first_batches.append(first_batch)
        second_batches.append(second_batch)
    np.testing.assert_array_equal(joined(first_batches, 0, "id"), joined(list(expected), 0, "id"))
    np.testing.assert_array_equal(joined(second_batches, 0, "id"), joined(list(expected), 0, "id"))


@pytest.mark.parametrize("steps", [1, 4])
@pytest.mark.parametrize("transform", ["tree_map", "jit", "device_get"])
def test_trajectory_metadata_survives_tree_transforms_and_rebatching(steps, transform):
    dataset = records((steps, 7))
    batch = next(iter(leg.dataloader(dataset, batch_size=5)))
    assert isinstance(batch, tuple)
    assert batch.trajectory_steps == steps
    mapped = {"tree_map": lambda value: jax.tree.map(lambda leaf: leaf, value),
              "jit": jax.jit(lambda value: value),
              "device_get": jax.device_get}[transform](batch)
    assert mapped.trajectory_steps == steps
    assert len(mapped) == len(batch) == 2
    key = jax.random.PRNGKey(13)
    rebatched = list(leg.dataloader(mapped, batch_size=2, shuffle=True, generator=key))
    assert [value[0]["_valid"].size for value in rebatched] == [2 * steps, 2 * steps, steps]
    assert all(value.trajectory_steps == steps for value in rebatched)
    assert_aligned(rebatched)
    _, shuffle_key = jax.random.split(key)
    order = np.asarray(jax.random.permutation(shuffle_key, 5))
    expected = np.concatenate([np.arange(steps) * 7 + episode for episode in order])
    np.testing.assert_array_equal(joined(rebatched, 0, "id"), expected)
    np.testing.assert_array_equal(joined(rebatched, 1, "id"), expected)


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5, "2"])
def test_invalid_batch_size_fails_at_construction(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        leg.dataloader(records(), batch_size=batch_size)


@pytest.mark.parametrize("options", [{"shuffle": 1}, {"drop_last": "yes"}])
def test_invalid_flags_fail_at_construction(options):
    with pytest.raises(ValueError, match="booleans"):
        leg.dataloader(records(), **options)


@pytest.mark.parametrize("dataset, message", [
    ([], "record dictionary"),
    ({"_valid": jnp.ones(2, bool)}, "record dictionary"),
    (({},), "_valid"),
    (({"_valid": jnp.ones(2)},), "boolean"),
    (({"_valid": jnp.ones((2, 2, 2), bool)},), "rank-1 or rank-2"),
    (({"_valid": jnp.ones((0, 2), bool)},), "nonempty"),
    (({"_valid": jnp.ones(2, bool)}, {"_valid": jnp.ones(3, bool)}), "Players"),
    (({"_valid": jnp.ones((2, 3), bool), "feature": jnp.ones((2, 4))},), "fields"),
    (({"_valid": jnp.ones(2, bool), "feature": jnp.array(1)},), "fields"),
    (({"_valid": jnp.ones(2, bool), "feature": {"nested": jnp.ones(2)}},), "fields"),
    (({"_valid": jnp.ones(2, bool), "feature": None},), "fields"),
])
def test_invalid_datasets_fail_at_construction(dataset, message):
    with pytest.raises(ValueError, match=message):
        leg.dataloader(dataset)


@pytest.mark.parametrize("key", [42, jnp.ones(2), jax.random.split(jax.random.key(0), 2)])
def test_invalid_generator_fails_at_construction(key):
    with pytest.raises(ValueError, match="JAX PRNG key"):
        leg.dataloader(records(), generator=key)
