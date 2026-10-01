"""Compaction preserves minibatch membership, frozen targets, and optimizer state."""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
optax = pytest.importorskip("optax")
nn = pytest.importorskip("flax.linen")

import LiteEFG as leg
from LiteEFG.baselines.drl.PPO import graph as PPO
from LiteEFG.drl.env import Goofspiel
from LiteEFG.drl.runtime import Trainer


class _RegressionEnvironment:
    num_players = 2
    num_actions = 2
    max_steps = 1

    def init(self, key):
        return jnp.array(0, jnp.int32)

    def features(self, state):
        return {"information_set": jnp.ones((2, 3)), "target": jnp.zeros(2),
                "sample_id": jnp.zeros(2)}

    def legal_action_mask(self, state):
        return jnp.ones((2, 2), bool)

    def active_players(self, state):
        return jnp.ones(2, bool)

    def step(self, state, actions, key):
        return state + 1, jnp.zeros(2), jnp.array(True)


class _Regression(leg.DRLGraph):
    def __init__(self, sharing="independent"):
        super().__init__()

        def model():
            return leg.model(
                init=lambda key, inputs: jnp.linspace(0.04, 0.16, inputs.shape[-1]),
                apply=lambda weights, inputs: jnp.dot(inputs, weights),
                optimizer=optax.chain(optax.clip_by_global_norm(0.7),
                                      optax.adamw(0.013, weight_decay=0.03)),
            )

        resources = [model(), model()]
        if sharing == "shared":
            resources[1] = resources[0]
        self.local = leg.ModelList(resources)
        self.common = model() if sharing == "partial" else None
        with leg.backward(color=1):
            self.prediction = self.local[self.player](self.env.information_set)
            if self.common is not None:
                self.prediction = self.prediction + self.common(self.env.information_set)
                resources.append(self.common)
            self.loss = (self.prediction - self.env.target) ** 2
            self.minimize(self.loss, models=resources)
            self.strategy = self.legal_action_mask / self.action_set_size
            self.metrics = {"prediction": self.prediction, "sample_id": self.env.sample_id}


def _trainer(sharing="independent", **options):
    return _Regression(sharing).bind(_RegressionEnvironment(), batch_size=7,
                                    seed=13, **options).trainer


def _assert_close_tree(actual, expected):
    actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_tree = jax.tree_util.tree_flatten(expected)
    assert actual_tree == expected_tree
    for left, right in zip(actual_leaves, expected_leaves):
        if np.issubdtype(np.asarray(left).dtype, np.inexact):
            np.testing.assert_allclose(left, right, rtol=5e-5, atol=3e-6)
        else:
            np.testing.assert_array_equal(left, right)


def _records(shape, masks, *, poison=False):
    slots = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)
    records = []
    for player, mask in enumerate(masks):
        features = jnp.stack((1 + slots / 20, jnp.sin(slots + player),
                              jnp.cos(slots / 3 + player)), axis=-1)
        target = 0.2 * slots / 7 + 0.3 * jnp.sin(slots * 0.7 + player)
        inputs = {"information_set": features, "target": target,
                  "sample_id": slots + 100 * player, "_valid": jnp.asarray(mask)}
        if poison:
            inputs = {name: value if name == "_valid" else jnp.where(
                mask.reshape(shape + (1,) * (value.ndim - len(shape))), value,
                jnp.inf if name == "target" else jnp.nan)
                      for name, value in inputs.items()}
        records.append(inputs)
    return tuple(records)


def _manual_epochs(trainer, data, *, epochs, minibatch_size, generator):
    loader = leg.dataloader(data, batch_size=minibatch_size, shuffle=True,
                            generator=generator)
    reports = [trainer.optimize(batch, upd_color=[1])
               for _ in range(epochs) for batch in loader]
    counts = np.asarray([report["valid_samples"] for report in reports])
    result = dict(reports[-1], valid_samples=counts.sum(axis=0).tolist())
    for name in ("loss", *trainer.graph.metrics):
        values = np.asarray([report[name] for report in reports])
        result[name] = ((values * counts).sum(axis=0)
                        / np.maximum(counts.sum(axis=0), 1)).tolist()
    return result


def _oracle_epoch_batches(data, epochs, minibatch_size, generator):
    """Select whole source columns independently of loader and runtime helpers."""
    episodes = data[0]["_valid"].shape[1]
    for _ in range(epochs):
        generator, shuffle_key = jax.random.split(generator)
        order = np.asarray(jax.random.permutation(shuffle_key, episodes))
        for start in range(0, episodes, minibatch_size):
            selected = order[start:start + minibatch_size]
            yield tuple({name: jnp.concatenate([value[:, episode] for episode in selected])
                         for name, value in records.items()} for records in data)


@pytest.mark.parametrize("compact_batches", [False, True])
@pytest.mark.parametrize("sharing", ["independent", "shared", "partial"])
def test_compiled_trajectory_groups_match_independent_source_column_oracle(
        compact_batches, sharing):
    shape = (5, 7)
    times, episodes = np.indices(shape)
    lengths = np.asarray([1, 2, 3, 4, 5, 1, 4])
    # One seat participates in every episode, the other in just one episode.
    # This distinguishes a shared episode shuffle from independent seat shuffles
    # and exercises optimizer skips in groups with no samples for the second seat.
    masks = (times < lengths, (episodes == 3) & (times < 2))
    data = _records(shape, masks, poison=True)
    reference = _trainer(sharing, compact_batches=False)
    actual = _trainer(sharing, compact_batches=compact_batches)
    options = dict(epochs=2, minibatch_size=3, generator=jax.random.PRNGKey(31))
    reports, expected_steps = [], np.zeros(len(reference.graph.models), dtype=int)
    for batch in _oracle_epoch_batches(data, **options):
        active = [bool(np.asarray(records["_valid"]).any()) for records in batch]
        for index, resource in enumerate(reference.graph.models):
            expected_steps[index] += (
                any(resource is reference.graph.local[player] and active[player]
                    for player in range(2))
                or (resource is reference.graph.common and any(active)))
        reports.append(reference.optimize(batch, upd_color=[1]))
    assert len(reports) == 6
    result = actual.optimize_epochs(data, upd_color=[1], **options)
    _assert_close_tree(actual.states, reference.states)
    counts = np.asarray([report["valid_samples"] for report in reports])
    np.testing.assert_array_equal(result["valid_samples"], counts.sum(0))
    for name in ("loss", *actual.graph.metrics):
        values = np.asarray([report[name] for report in reports])
        expected = (values * counts).sum(0) / np.maximum(counts.sum(0), 1)
        np.testing.assert_allclose(result[name], expected, rtol=5e-5, atol=3e-6)
    np.testing.assert_array_equal(
        [int(resource.state.step) for resource in actual.graph.models], expected_steps)


@pytest.mark.parametrize("compact_batches", [False, True])
@pytest.mark.parametrize("horizon", [1, 5, 128])
def test_512_trajectories_take_four_updates_at_minibatch_128(
        compact_batches, horizon):
    shape = (horizon, 512)
    times, episodes = np.indices(shape)
    lengths = 1 + episodes % min(horizon, 20)
    masks = (times < lengths, times < np.maximum(1, lengths // 2))
    data = _records(shape, masks)
    trainer = _trainer(compact_batches=compact_batches)
    report = trainer.optimize_epochs(
        data, epochs=1, minibatch_size=128, generator=jax.random.PRNGKey(11), upd_color=[1])
    assert [int(resource.state.step) for resource in trainer.graph.models] == [4, 4]
    assert report["valid_samples"] == [int(mask.sum()) for mask in masks]
    for player, mask in enumerate(masks):
        expected_id = np.asarray(data[player]["sample_id"])[mask].mean()
        np.testing.assert_allclose(report["sample_id"][player], expected_id, rtol=1e-6)


@pytest.mark.parametrize("sharing", ["independent", "shared", "partial"])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("mode", ["mixed", "sparse"])
def test_compaction_preserves_shuffled_updates_and_optimizer_states(sharing, compiled, mode):
    shape = (5, 7)
    slots = np.arange(np.prod(shape)).reshape(shape)
    masks = ((slots % 3 == 0, slots % 5 < 2) if mode == "mixed"
             else (slots == 17, np.zeros(shape, dtype=bool)))
    data = _records(shape, masks)
    reference = _trainer(sharing, compact_batches=False)
    compact = _trainer(sharing, compact_batches=True)
    before = tuple(resource.state for resource in compact.graph.models)
    initial_key = np.asarray(compact.key).copy()
    options = dict(epochs=2, minibatch_size=3, generator=jax.random.PRNGKey(21))
    expected = _manual_epochs(reference, data, **options)
    actual = (compact.optimize_epochs(data, upd_color=[1], **options) if compiled
              else _manual_epochs(compact, data, **options))
    _assert_close_tree(actual, expected)
    _assert_close_tree(compact.states, reference.states)
    np.testing.assert_array_equal(compact.key, initial_key)
    np.testing.assert_array_equal(compact.key, reference.key)
    assert actual["valid_samples"] == [2 * int(mask.sum()) for mask in masks]
    assert all(np.isfinite(np.asarray(leaf)).all()
               for leaf in jax.tree_util.tree_leaves(compact.states))

    # Compute resource participation from the public loader's original groups.
    # Adam must advance once per active group, even when only one sample remains.
    loader = leg.dataloader(data, batch_size=3, shuffle=True,
                            generator=jax.random.PRNGKey(21))
    expected_steps = np.zeros(len(compact.graph.models), dtype=int)
    widths = []
    for _ in range(2):
        for batch in loader:
            widths.append(batch[0]["_valid"].size)
            active = [bool(np.asarray(records["_valid"]).any()) for records in batch]
            for index, resource in enumerate(compact.graph.models):
                participates = [compact.graph.local[player] is resource and active[player]
                                for player in range(2)]
                expected_steps[index] += (any(participates)
                                          or (resource is compact.graph.common and any(active)))
    assert widths == [15, 15, 5] * 2
    for resource, initial, steps in zip(compact.graph.models, before, expected_steps):
        assert int(resource.state.step) == steps
        if steps == 0:
            assert resource.state is initial


@pytest.mark.parametrize("sharing", ["independent", "shared", "partial"])
def test_nonfinite_padding_empty_seats_and_dense_capacity_preserve_updates(sharing):
    reference = _trainer(sharing, compact_batches=False)
    compact = _trainer(sharing, compact_batches=True)
    slots = np.arange(33)
    # Cross power-of-two boundaries, include asymmetric seats, and finish with
    # an all-invalid update after Adam's moments are already nonzero.
    for counts in ((0, 0), (1, 3), (16, 8), (17, 31), (33, 33), (8, 0), (0, 0)):
        masks = tuple((slots * 7 % 33) < count for count in counts)
        batch = _records((33,), masks, poison=True)
        before = tuple(resource.state for resource in compact.graph.models)
        expected = reference.optimize(batch, upd_color=[1])
        actual = compact.optimize(batch, upd_color=[1])
        assert actual["valid_samples"] == list(counts)
        _assert_close_tree(actual, expected)
        _assert_close_tree(compact.states, reference.states)
        if not any(counts):
            assert actual["loss"] == [0.0, 0.0]
            assert all(resource.state is state
                       for resource, state in zip(compact.graph.models, before))


@pytest.mark.parametrize("mode", ["empty", "dense", "uneven"])
def test_compiled_compaction_handles_empty_dense_and_remainder_batches(mode):
    shape = (3, 11)
    slots = np.arange(np.prod(shape)).reshape(shape)
    masks = {"empty": (slots < 0, slots < 0),
             "dense": (slots >= 0, slots >= 0),
             "uneven": (slots % 3 != 0, slots == 20)}[mode]
    data = _records(shape, masks, poison=True)
    reference = _trainer("partial", compact_batches=False)
    compact = _trainer("partial", compact_batches=True)
    before = tuple(resource.state for resource in compact.graph.models)
    options = dict(epochs=2, minibatch_size=4, generator=jax.random.PRNGKey(4))
    expected = _manual_epochs(reference, data, **options)
    actual = compact.optimize_epochs(data, upd_color=[1], **options)
    _assert_close_tree(actual, expected)
    _assert_close_tree(compact.states, reference.states)
    if mode == "empty":
        assert all(resource.state is state
                   for resource, state in zip(compact.graph.models, before))


def test_compiled_capacity_cache_handles_changing_short_dense_inputs():
    reference = _trainer("shared", compact_batches=False)
    compact = _trainer("shared", compact_batches=True)
    for iteration, width in enumerate((9, 13, 17, 9)):
        masks = (np.ones((1, width), dtype=bool),) * 2
        data = _records((1, width), masks)
        options = dict(epochs=1, minibatch_size=32,
                       generator=jax.random.PRNGKey(iteration))
        expected = _manual_epochs(reference, data, **options)
        actual = compact.optimize_epochs(data, upd_color=[1], **options)
        assert actual["valid_samples"] == [width, width]
        _assert_close_tree(actual, expected)
        _assert_close_tree(compact.states, reference.states)
        assert all(int(resource.state.step) == iteration + 1
                   for resource in compact.graph.models)


def test_compaction_keeps_full_trajectory_ppo_targets_and_normalization(monkeypatch):
    algo = PPO([leg.model(nn.Dense(3)) for _ in range(2)],
               [leg.model(nn.Dense(1)) for _ in range(2)], num_actions=3)
    with leg.backward(color=1):
        algo.metrics["frozen_advantage"] = algo.advantage + 0.0
        algo.metrics["frozen_target"] = algo.return_target + 0.0
    trainer = algo.bind(Goofspiel(num_cards=3), batch_size=32, seed=7,
                        compact_batches=True).trainer
    prepared = trainer.update(trainer.collect(), upd_color=[0])
    batch = tuple({name: value.reshape((96,) + value.shape[2:])[:64]
                   for name, value in player.items()} for player in prepared)
    masks = tuple(jnp.arange(64) % (9 + player) == player for player in range(2))
    batch = tuple(dict(records, _valid=mask) for records, mask in zip(batch, masks))
    params = tuple(resource.state.params for resource in algo.models)
    expected = [algo.evaluate(params, records, (algo.advantage, algo.return_target),
                              upd_color=[1]) for records in batch]
    means = np.asarray([[np.asarray(values)[np.asarray(mask)].mean() for values in outputs]
                        for outputs, mask in zip(expected, masks)])
    # A fresh normalization on these sparse minibatches would have zero mean.
    assert np.max(np.abs(means[:, 0])) > 0.01
    before = jax.tree_util.tree_map(lambda value: np.asarray(value).copy(), batch)

    def unexpected_update(*args, **kwargs):
        pytest.fail("Compaction must use the cached trajectory targets")

    monkeypatch.setattr(algo, "update", unexpected_update)
    for _ in range(2):
        report = trainer.optimize(batch, upd_color=[1])
        np.testing.assert_allclose(report["frozen_advantage"], means[:, 0],
                                   rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(report["frozen_target"], means[:, 1],
                                   rtol=2e-6, atol=2e-7)
    for current, original in zip(jax.tree_util.tree_leaves(batch),
                                 jax.tree_util.tree_leaves(before)):
        np.testing.assert_array_equal(current, original)


def test_compaction_is_enabled_by_default_and_can_be_disabled():
    assert _trainer().compact_batches is True
    assert _trainer(compact_batches=False).compact_batches is False
    assert Trainer(_Regression(), _RegressionEnvironment(), batch_size=1).compact_batches is True


@pytest.mark.parametrize("compiled", [False, True])
def test_sparse_records_reach_the_model_in_a_smaller_batch(monkeypatch, compiled):
    slots = np.arange(32)
    data = _records((32,), (slots == 19, slots == 23))
    compact = _trainer()
    observed_widths = []
    gradients = compact._minibatch_gradients

    def record_width(states, batch, upd_color=None):
        observed_widths.append(batch["_valid"].size)
        return gradients(states, batch, upd_color)

    monkeypatch.setattr(compact, "_minibatch_gradients", record_width)
    if compiled:
        trajectories = jax.tree_util.tree_map(lambda value: value[None], data)
        report = compact.optimize_epochs(
            trajectories, epochs=1, minibatch_size=32,
            generator=jax.random.PRNGKey(8), upd_color=[1])
    else:
        report = compact.optimize(data, upd_color=[1])
    assert report["valid_samples"] == [1, 1]
    assert observed_widths
    assert max(observed_widths) < 32


@pytest.mark.parametrize("option", [None, 0, 1, "true"])
def test_compaction_option_requires_a_boolean(option):
    with pytest.raises((ValueError, TypeError), match="compact_batches.*boolean"):
        _trainer(compact_batches=option)


@pytest.mark.parametrize("implementation", ["threefry2x32", "rbg", "unsafe_rbg"])
def test_compiled_shuffle_matches_loader_for_each_prng_implementation(implementation):
    shape = (5, 7)
    slots = np.arange(np.prod(shape)).reshape(shape)
    data = _records(shape, (slots % 5 == 0, slots % 7 == 0))
    trainer = _trainer()
    for seed in (17, 29):
        generator = jax.random.key(seed, impl=implementation)
        original_key = np.asarray(jax.random.key_data(generator)).copy()
        orders, _ = trainer._plan_minibatches(data, 2, 3, generator)
        loader = leg.dataloader(data, batch_size=3, shuffle=True, generator=generator)
        oracle = iter(_oracle_epoch_batches(data, 2, 3, generator))
        for epoch in range(2):
            np.testing.assert_array_equal(orders[epoch, 0], orders[epoch, 1])
            for index, batch in enumerate(loader):
                independent_batch = next(oracle)
                for player, records in enumerate(batch):
                    # Compare every row ID, including the final short batch.
                    expected = np.asarray(records["sample_id"]) - 100 * player
                    actual = orders[epoch, player, index * 15:(index + 1) * 15]
                    np.testing.assert_array_equal(actual, expected)
                    for name, value in records.items():
                        np.testing.assert_array_equal(value, independent_batch[player][name])
                    # A trajectory's five records are adjacent in time order.
                    grouped = np.asarray(actual).reshape(-1, shape[0])
                    np.testing.assert_array_equal(grouped // shape[1],
                                                  np.broadcast_to(np.arange(shape[0]), grouped.shape))
                    np.testing.assert_array_equal(grouped % shape[1],
                                                  np.broadcast_to(grouped[:, :1] % shape[1], grouped.shape))
        np.testing.assert_array_equal(jax.random.key_data(generator), original_key)
