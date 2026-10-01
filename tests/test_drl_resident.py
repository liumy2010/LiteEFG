"""Resident rollout packing preserves targets and whole-minibatch updates."""
import os
from pathlib import Path
import re
import subprocess
import sys

if __name__ == '__main__' and hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
optax = pytest.importorskip("optax")
pytest.importorskip("flax.linen")

import LiteEFG as leg
from LiteEFG.baselines.drl.PPO import graph as PPO
from LiteEFG.drl.resident import PackedRollout, ResidentOptimizer
from test_drl_compact_batches import (
    _Regression, _RegressionEnvironment, _assert_close_tree, _oracle_epoch_batches,
)
from test_drl_microbatch import _hard_records, _oracle_step
from test_drl_parallel import _replicated
from test_drl_point_microbatch import _equations


class _ResidentAPI:
    """Exercise public dispatch while inspecting a separate owner-bound probe."""
    def __init__(self, trainer, local_microbatch_size=3, target_chunk_size=128):
        self.trainer = trainer
        self.target_chunk_size = target_chunk_size
        self.engine = ResidentOptimizer(trainer, target_chunk_size=target_chunk_size)
        self.point_size = local_microbatch_size * len(trainer.optimizing_devices)

    @property
    def executables(self):
        return self.engine.executables

    @property
    def last_arguments(self):
        return self.engine.last_arguments

    @property
    def capture_arguments(self):
        return self.engine.capture_arguments

    @capture_arguments.setter
    def capture_arguments(self, value):
        self.engine.capture_arguments = value

    def pack(self, data):
        packed = self.trainer.compact(data, target_chunk_size=self.target_chunk_size)
        assert isinstance(packed, PackedRollout)
        return packed

    def targets(self, packed):
        return self.trainer.update(packed, upd_color=[0])

    def plan(self, packed, **options):
        return self.engine.plan(packed, microbatch_size=self.point_size,
                                microbatch_unit='points', **options)

    def gradient_probe(self, packed, trajectory_ids, upd_color=(1,)):
        return self.engine.gradient_probe(packed, trajectory_ids, upd_color=upd_color,
                                          microbatch_size=self.point_size, microbatch_unit='points')

    def optimize_epochs(self, packed, **options):
        return self.trainer.optimize_epochs(packed, microbatch_size=self.point_size,
                                             microbatch_unit='points', **options)


def _devices():
    return tuple(jax.local_devices(backend='cpu'))


def _regression(sharing='partial', *, compact_batches=True):
    return _Regression(sharing).bind(
        _RegressionEnvironment(), batch_size=4 * len(_devices()), seed=13,
        devices=_devices(), parallel=('optimizing',), compact_batches=compact_batches).trainer


def _regression_data():
    devices = len(_devices())
    horizon, episodes = 5, 4 * devices
    times, columns = np.indices((horizon, episodes))
    owners = columns // 4
    first = (owners == 0) | ((owners == 2) & (times == columns % 5))
    first |= (owners >= 3) & (times < 1 + columns % 4)
    second = ((owners == 0) & (times < columns % 3))
    second |= (owners >= 2) & (times % 2 == 0) & (columns % 3 == 0)
    rows = _hard_records((horizon, episodes), (first, second))
    result = []
    for row in rows:
        bulky = jnp.broadcast_to(jnp.arange(37), (horizon, episodes, 37)).astype(jnp.float32)
        result.append(dict(row, _alive=jnp.ones((horizon, episodes), bool),
                           reward=jnp.zeros((horizon, episodes)), bulky_feature=bulky))
    return tuple(result)


def _selected(data, trajectory_ids):
    return tuple({name: np.concatenate([np.asarray(value)[:, episode] for episode in trajectory_ids])
                  for name, value in row.items()} for row in data)


def _gradient_oracle(trainer, batch):
    """NumPy differentiation; no graph differentiation or runtime reductions."""
    graph = trainer.graph
    states = tuple(resource.state for resource in graph.models)
    numerators = [np.zeros_like(np.asarray(state.params)) for state in states]
    participation = np.zeros(len(states), np.int32)
    sums, counts = [], []
    for player, row in enumerate(batch):
        valid = np.asarray(row['_valid'])
        features = np.asarray(row['information_set'])[valid]
        target = np.asarray(row['target'])[valid]
        indices = [graph.model_index(graph.local[player])]
        if graph.common is not None:
            indices.append(graph.model_index(graph.common))
        predicted = sum(features @ np.asarray(states[index].params) for index in indices)
        error = predicted - target
        sums.append([np.sum(error ** 2), np.sum(predicted),
                     np.asarray(row['sample_id'])[valid].sum()])
        counts.append(int(valid.sum()))
        for index in indices:
            numerators[index] += 2 * features.T @ error
            participation[index] += int(valid.sum())
    gradients = tuple(value / max(int(count), 1)
                      for value, count in zip(numerators, participation))
    return dict(gradients=gradients, sums=np.asarray(sums),
                counts=np.asarray(counts), participation=participation)


def _orders(episodes, epochs, key):
    orders = []
    for _ in range(epochs):
        key, shuffle = jax.random.split(key)
        orders.append(np.asarray(jax.random.permutation(shuffle, episodes)))
    return np.asarray(orders)


def _assert_report(report, reports):
    counts = np.asarray([value['valid_samples'] for value in reports])
    np.testing.assert_array_equal(report['valid_samples'], counts.sum(0))
    for name in ('loss', 'prediction', 'sample_id'):
        means = np.asarray([value[name] for value in reports])
        expected = (means * counts).sum(0) / np.maximum(counts.sum(0), 1)
        np.testing.assert_allclose(report[name], expected, rtol=5e-5, atol=3e-6)


def test_pack_is_a_local_bijection_and_preserves_the_full_small_timeline():
    data = _regression_data()
    trainer = _regression()
    resident = _ResidentAPI(trainer, local_microbatch_size=3)
    before = tuple(resource.state for resource in trainer.graph.models)
    packed = resident.pack(data)
    devices = len(_devices())
    assert (packed.horizon, packed.episodes) == data[0]['_valid'].shape
    assert packed.valid_counts.shape == (devices, 2)
    for player, (row, source, timeline) in enumerate(zip(packed.records, data, packed.timeline)):
        assert row['_valid'].shape == (devices * packed.capacity,)
        assert 'bulky_feature' not in timeline
        for name in ('_valid', '_alive', 'reward', 'sample_id', 'target'):
            np.testing.assert_array_equal(timeline[name], source[name])
        for shard in row['_valid'].addressable_shards:
            assert shard.data.shape == (packed.capacity,)
        for owner in range(devices):
            section = slice(owner * packed.capacity, (owner + 1) * packed.capacity)
            active = np.asarray(row['_valid'])[section]
            ids = np.asarray(row['_trajectory_id'])[section][active]
            times = np.asarray(row['_time_index'])[section][active]
            expected = [(time, episode) for episode in range(owner * 4, (owner + 1) * 4)
                        for time in range(packed.horizon) if np.asarray(source['_valid'])[time, episode]]
            assert list(zip(times, ids)) == expected
            assert len(expected) == packed.valid_counts[owner, player]
            np.testing.assert_array_equal(np.asarray(row['_timeline_index'])[section][active],
                                          times * 4 + ids % 4)
            for name in source:
                np.testing.assert_array_equal(np.asarray(row[name])[section][active],
                                              np.asarray(source[name])[times, ids])
            actual_device = row['_valid'].addressable_shards[owner].device
            assert actual_device == _devices()[owner]
    assert all(resource.state is state for resource, state in zip(trainer.graph.models, before))


def test_plan_uses_one_global_shuffle_and_exact_original_minibatch_membership():
    data = _regression_data()
    resident = _ResidentAPI(_regression(), local_microbatch_size=3)
    packed = resident.pack(data)
    epochs, minibatch, key = 3, 3, jax.random.PRNGKey(17)
    plan = resident.plan(packed, epochs=epochs, minibatch_size=minibatch, generator=key)
    orders = _orders(packed.episodes, epochs, key)
    np.testing.assert_array_equal(plan['trajectory_orders'], orders)
    groups = (packed.episodes + minibatch - 1) // minibatch
    expected_counts = np.zeros((len(_devices()), epochs, groups, 2), np.int32)
    for epoch, order in enumerate(orders):
        np.testing.assert_array_equal(np.asarray(plan['rank_for_trajectory'])[epoch, order],
                                      np.arange(packed.episodes))
        np.testing.assert_array_equal(np.asarray(plan['group_for_trajectory'])[epoch, order],
                                      np.arange(packed.episodes) // minibatch)
        for group in range(groups):
            ids = order[group * minibatch:(group + 1) * minibatch]
            for owner in range(len(_devices())):
                local_ids = ids[ids // 4 == owner]
                for player, row in enumerate(data):
                    expected_counts[owner, epoch, group, player] = np.asarray(row['_valid'])[:, local_ids].sum()
    np.testing.assert_array_equal(plan['device_group_counts'], expected_counts)
    np.testing.assert_array_equal(plan['device_microbatch_counts'],
                                  np.maximum(1, (expected_counts.max(-1) + 2) // 3))
    np.testing.assert_array_equal(expected_counts.sum((0, 2)),
                                  np.tile([np.asarray(row['_valid']).sum() for row in data], (epochs, 1)))
    if len(_devices()) > 1:
        assert not expected_counts[1].any()
        assert np.ptp(np.asarray(plan['device_microbatch_counts']), axis=0).max() > 0


@pytest.mark.parametrize('sharing', ['independent', 'shared', 'partial'])
@pytest.mark.parametrize('width', [2, 7])
def test_gradient_probe_matches_independent_pooled_numerators(sharing, width):
    trainer = _regression(sharing)
    resident = _ResidentAPI(trainer, local_microbatch_size=width)
    data = _regression_data()
    packed = resident.pack(data)
    before = tuple(resource.state for resource in trainer.graph.models)
    key = np.asarray(trainer.key).copy()
    groups = [np.arange(packed.episodes)[::-1], np.array([3, 0, 1 if packed.episodes == 4 else packed.episodes - 1])]
    if len(_devices()) > 1:
        groups += [np.array([4, 7, 6])]
    for ids in groups:
        expected = _gradient_oracle(trainer, _selected(data, ids))
        actual = resident.gradient_probe(packed, ids, upd_color=[1])
        _assert_close_tree(actual, expected)
    assert all(resource.state is old for resource, old in zip(trainer.graph.models, before))
    np.testing.assert_array_equal(trainer.key, key)


@pytest.mark.parametrize('sharing', ['independent', 'shared', 'partial'])
def test_epochs_match_numpy_adam_oracle_and_native_global_points(sharing):
    actual, reference = [_regression(sharing) for _ in range(2)]
    native = _regression(sharing, compact_batches=False)
    resident = _ResidentAPI(actual, local_microbatch_size=3)
    data = _regression_data()
    packed = resident.pack(data)
    options = dict(epochs=2, minibatch_size=3, generator=jax.random.PRNGKey(31))
    reports = []
    for batch in _oracle_epoch_batches(data, **options):
        expected, metrics, participation = _oracle_step(reference, batch)
        for resource, state in zip(reference.graph.models, expected):
            resource.state = state
        reports.append(dict(zip(('loss', 'prediction', 'sample_id'), metrics.T.tolist()),
                            valid_samples=[int(np.asarray(row['_valid']).sum()) for row in batch]))
    actual_report = resident.optimize_epochs(packed, upd_color=[1], **options)
    native_report = native.optimize_epochs(data, upd_color=[1], microbatch_size=3 * len(_devices()),
                                             microbatch_unit='points', **options)
    _assert_close_tree(actual.states, reference.states)
    _assert_close_tree(actual.states, native.states)
    _assert_report(actual_report, reports)
    _assert_close_tree(actual_report, native_report)
    _replicated(actual.states, _devices())
    np.testing.assert_array_equal(actual.key, reference.key)
    previous = tuple(resource.state for resource in actual.graph.models)
    empty = tuple(dict(row, _valid=jnp.zeros_like(row['_valid'])) for row in data)
    report = resident.optimize_epochs(resident.pack(empty), upd_color=[1], **options)
    assert report['valid_samples'] == [0, 0]
    assert report['loss'] == [0.0, 0.0]
    assert all(resource.state is old for resource, old in zip(actual.graph.models, previous))


def test_actual_chunks_never_borrow_points_from_another_group_or_owner(monkeypatch):
    trainer = _regression()
    resident = _ResidentAPI(trainer, local_microbatch_size=3)
    data = _regression_data()
    packed = resident.pack(data)
    observations = []
    differentiate = trainer._minibatch_gradients

    def record(owner, player, sample_ids, valid):
        observations.append((int(np.asarray(owner)), int(np.asarray(player)[0]),
                             np.asarray(sample_ids)[np.asarray(valid)].astype(int)))

    def observe(states, batch, colors):
        assert batch['_valid'].shape == (3,)
        jax.debug.callback(record, jax.lax.axis_index('learners'), batch['player'],
                           batch['sample_id'], batch['_valid'])
        return differentiate(states, batch, colors)

    monkeypatch.setattr(trainer, '_minibatch_gradients', observe)
    order = _orders(packed.episodes, 1, jax.random.PRNGKey(31))[0]
    for start in range(0, packed.episodes, 3):
        ids = order[start:start + 3]
        observations.clear()
        resident.gradient_probe(packed, ids)
        jax.effects_barrier()
        for owner in range(len(_devices())):
            local_ids = ids[ids // 4 == owner]
            counts = [int(np.asarray(row['_valid'])[:, local_ids].sum()) for row in data]
            calls = max(1, (max(counts) + 2) // 3)
            for player, source in enumerate(data):
                chunks = [values for device, seat, values in observations if device == owner and seat == player]
                assert len(chunks) == calls
                expected = [int(np.asarray(source['sample_id'])[time, episode]) for episode in local_ids
                            for time in range(packed.horizon) if np.asarray(source['_valid'])[time, episode]]
                np.testing.assert_array_equal(np.sort(np.concatenate(chunks)), np.sort(expected))
                assert all(len(values) <= 3 for values in chunks)


def test_local_chunk_loops_have_no_collectives_or_full_feature_gathers():
    resident = _ResidentAPI(_regression(), local_microbatch_size=3)
    resident.capture_arguments = True
    packed = resident.pack(_regression_data())
    resident.gradient_probe(packed, np.arange(packed.episodes))
    name = next(name for name in resident.executables if name.startswith('probe_'))
    execute, arguments = resident.executables[name], resident.last_arguments[name]
    equations = list(_equations(jax.make_jaxpr(execute)(*arguments)))
    loops = [equation for equation in equations if equation.primitive.name == 'while']
    assert loops, 'Unequal local work must exercise a dynamic point loop'
    collective = {'psum', 'pmax', 'pmin', 'all_gather', 'all_to_all', 'ppermute'}
    for loop in loops:
        assert not any(equation.primitive.name in collective
                       for equation in _equations(loop.params['body_jaxpr']))
    gathers = [output.aval.shape for equation in equations if equation.primitive.name == 'gather'
               for output in equation.outvars if hasattr(output.aval, 'shape')
               and len(output.aval.shape) == 2 and output.aval.shape[-1] in (3, 37)]
    assert gathers and all(shape[0] <= 3 for shape in gathers), gathers
    assert not any(equation.primitive.name in {'all_gather', 'all_to_all', 'ppermute'} for equation in equations)
    compiled = execute.lower(*arguments).compile()
    assert len(compiled.runtime_executable().local_devices()) == len(_devices())
    if len(_devices()) > 1:
        hlo = compiled.as_text().lower()
        assert not re.search(r'\b(?:all-gather|all-to-all|collective-permute)(?:-start)?\(', hlo)
        reductions = [line.split(' all-reduce', 1)[0] for line in hlo.splitlines()
                      if re.search(r'\ball-reduce(?:-start)?\(', line)]
        assert any(re.search(r'\bf32\[3\](?:\{[^}]*\})?', line) for line in reductions), (
            'Inspect actual float parameter-gradient reduction, not only scalar counts')


class _TargetEnvironment:
    num_players = 2
    num_actions = 2
    max_steps = 4

    def init(self, key):
        return jnp.array(0, jnp.int32)

    def features(self, state):
        return {'information_set': jnp.ones((2, 3)), 'full_info': jnp.ones((2, 33))}

    def legal_action_mask(self, state):
        return jnp.ones((2, 2), bool)

    def active_players(self, state):
        return jnp.ones(2, bool)

    def step(self, state, actions, key):
        return state + 1, jnp.zeros(2), state >= 3


def _ppo(gamma, decay, normalize, shared=True, *, compact_batches=True):
    actors = [leg.model(init=lambda key, inputs: jnp.zeros((3, 2)),
                        apply=lambda weights, inputs: inputs @ weights,
                        optimizer=optax.chain(optax.clip_by_global_norm(.7), optax.adam(.01))) for _ in range(2)]
    critics = [leg.model(init=lambda key, inputs, value=value: jnp.array(value, jnp.float32),
                         apply=lambda weight, inputs: weight * inputs[..., :1],
                         optimizer=optax.chain(optax.clip_by_global_norm(.7), optax.adam(.01)))
               for value in (1., 1.7)]
    graph = PPO(actors, critics[0] if shared else critics, num_actions=2,
                gamma=gamma, gae_lambda=decay, normalize_advantage=normalize)
    return graph.bind(_TargetEnvironment(), batch_size=4 * len(_devices()), seed=13,
                      devices=_devices(), parallel=('sampling', 'optimizing'), compact_batches=compact_batches).trainer


def _target_data():
    episodes = 4 * len(_devices())
    times, ids = np.indices((4, episodes))
    lengths = 4 - np.arange(episodes) % 3
    lengths[0] = 4
    alive = times < lengths
    rows = []
    for player in range(2):
        valid = alive & (times % 2 == player) & (ids // 4 != 1)
        values = .5 + times * .5 + ids * .3 + player * .2
        reward = .1 * (1 + ids) + .2 * times
        reward = reward * (1 if player == 0 else -.7)
        reward[:, 0] = [0., 1., 0., 4.] if player == 0 else [1., 0., 2., -.5]
        features = np.broadcast_to(values[..., None], (4, episodes, 33)).copy()
        features[~valid] = np.nan
        actor = np.stack((np.ones_like(values), ids / 10, times / 4), -1)
        actor[~valid] = np.nan
        row = dict(full_info=jnp.asarray(features, jnp.float32),
                   information_set=jnp.asarray(actor, jnp.float32),
                   reward=jnp.asarray(np.where(alive, reward, np.nan), jnp.float32),
                   _valid=jnp.asarray(valid), _alive=jnp.asarray(alive),
                   legal_action_mask=jnp.ones((4, episodes, 2), bool),
                   action=jnp.zeros((4, episodes), jnp.int32),
                   log_sampling_prob=jnp.full((4, episodes), -np.log(2), jnp.float32))
        rows.append(row)
    return tuple(rows)


def _gae_oracle(data, gamma, decay, normalize, shared=True):
    """Enumerate own decisions and discounted intervening environment rewards."""
    results = []
    for player, row in enumerate(data):
        valid, alive = np.asarray(row['_valid']), np.asarray(row['_alive'])
        values = np.asarray(row['full_info'])[..., 0] * (1 if shared or player == 0 else 1.7)
        rewards = np.asarray(row['reward'])
        advantages = np.zeros(valid.shape, np.float64)
        targets = np.zeros_like(advantages)
        for episode in range(valid.shape[1]):
            decisions = np.flatnonzero(valid[:, episode])
            terminal = int(alive[:, episode].sum())
            future = 0.
            for rank in range(len(decisions) - 1, -1, -1):
                time = int(decisions[rank])
                end = int(decisions[rank + 1]) if rank + 1 < len(decisions) else terminal
                segment = sum(gamma ** (step - time) * float(rewards[step, episode])
                              for step in range(time, end))
                successor = float(values[end, episode]) if rank + 1 < len(decisions) else 0.
                delta = segment + gamma ** (end - time) * successor - float(values[time, episode])
                future = delta + gamma ** (end - time) * decay * future
                advantages[time, episode] = future
                targets[time, episode] = future + float(values[time, episode])
        raw = advantages.copy()
        if normalize and valid.any():
            selected = advantages[valid]
            advantages[valid] = (selected - selected.mean()) / np.sqrt(selected.var() + 1e-8)
        results.append((advantages, targets, raw))
    return results


@pytest.mark.parametrize('gamma,decay,normalize,shared', [
    (.5, 0., False, True), (.5, 1., False, True),
    (.9, .7, False, False), (.9, .7, True, True),
])
def test_targets_match_hand_gae_with_opponent_rewards_and_global_normalization(
        gamma, decay, normalize, shared):
    trainer = _ppo(gamma, decay, normalize, shared)
    resident = _ResidentAPI(trainer, local_microbatch_size=3, target_chunk_size=3)
    data = _target_data()
    oracle = _gae_oracle(data, gamma, decay, normalize, shared)
    if gamma == .5:
        np.testing.assert_allclose(oracle[0][2][[0, 2], 0], [.375 + .125 * decay, .5], atol=1e-8)
        np.testing.assert_allclose(oracle[0][1][[0, 2], 0], [.875 + .125 * decay, 2.], atol=1e-8)
    previous = tuple(resource.state for resource in trainer.graph.models)
    original_key = np.asarray(trainer.key).copy()
    native = trainer.update(data, upd_color=[0])
    packed = resident.targets(resident.pack(data))
    graph = trainer.graph
    for player, (row, timeline, expected) in enumerate(zip(packed.records, packed.timeline, oracle)):
        assert 'full_info' not in timeline
        np.testing.assert_array_equal(timeline['reward'], data[player]['reward'])
        valid = np.asarray(data[player]['_valid'])
        for node, expected_value in zip((graph.advantage, graph.return_target), expected[:2]):
            key = graph._node_key(node.index)
            np.testing.assert_allclose(np.asarray(timeline[key])[valid], expected_value[valid], rtol=3e-6, atol=3e-6)
            np.testing.assert_allclose(np.asarray(native[player][key])[valid], expected_value[valid], rtol=3e-6, atol=3e-6)
            packed_valid = np.asarray(row['_valid'])
            times = np.asarray(row['_time_index'])[packed_valid]
            ids = np.asarray(row['_trajectory_id'])[packed_valid]
            np.testing.assert_allclose(np.asarray(row[key])[packed_valid], expected_value[times, ids], rtol=3e-6, atol=3e-6)
        if normalize:
            normalized = np.asarray(timeline[graph._node_key(graph.advantage.index)])[valid]
            assert abs(normalized.mean()) < 3e-6
            assert abs(normalized.var() - 1) < 3e-6
            if len(_devices()) >= 4:
                local_valid = valid[:, :4]
                local = np.asarray(timeline[graph._node_key(graph.advantage.index)])[:, :4][local_valid]
                assert abs(local.mean()) > .01
    repeated = resident.targets(packed)
    for first, second in zip(jax.tree.leaves((packed.records, packed.timeline)),
                             jax.tree.leaves((repeated.records, repeated.timeline))):
        np.testing.assert_array_equal(first, second)
    assert all(resource.state is old for resource, old in zip(trainer.graph.models, previous))
    np.testing.assert_array_equal(trainer.key, original_key)


def test_ppo_shared_critic_updates_match_native_points_and_keep_targets_frozen():
    actual = _ppo(.9, .7, True, True)
    native = _ppo(.9, .7, True, True, compact_batches=False)
    resident = _ResidentAPI(actual, local_microbatch_size=3, target_chunk_size=3)
    data = _target_data()
    prepared = native.update(data, upd_color=[0])
    packed = resident.targets(resident.pack(data))
    frozen = jax.tree.map(lambda value: np.asarray(value).copy(), (packed.records, packed.timeline))
    options = dict(epochs=2, minibatch_size=3, generator=jax.random.PRNGKey(19), upd_color=[1])
    expected = native.optimize_epochs(prepared, microbatch_size=3 * len(_devices()),
                                       microbatch_unit='points', **options)
    report = resident.optimize_epochs(packed, **options)
    _assert_close_tree(actual.states, native.states)
    _assert_close_tree(report, expected)
    _replicated(actual.states, _devices())
    for before, after in zip(jax.tree.leaves(frozen), jax.tree.leaves((packed.records, packed.timeline))):
        np.testing.assert_array_equal(before, after)


def test_one_failed_global_group_preserves_prior_good_groups_and_all_replicas():
    actual, reference = _regression(), _regression()
    before = tuple(resource.state for resource in actual.graph.models)
    resident = _ResidentAPI(actual, local_microbatch_size=2)
    data = _regression_data()
    episodes = data[0]['_valid'].shape[1]
    key = jax.random.PRNGKey(9)
    order = _orders(episodes, 1, key)[0]
    bad_rank = next(rank for rank, episode in enumerate(order)
                    if rank > 0 and np.asarray(data[0]['_valid'])[:, episode].any())
    bad_episode = int(order[bad_rank])
    bad_time = int(np.flatnonzero(np.asarray(data[0]['_valid'])[:, bad_episode])[0])
    corrupted = list(data)
    corrupted[0] = dict(corrupted[0], target=corrupted[0]['target'].at[bad_time, bad_episode].set(jnp.nan))
    for episode in order[:bad_rank]:
        states, _, _ = _oracle_step(reference, _selected(data, [int(episode)]))
        for resource, state in zip(reference.graph.models, states):
            resource.state = state
    with pytest.raises(FloatingPointError, match='Non-finite'):
        resident.optimize_epochs(resident.pack(tuple(corrupted)), epochs=1, minibatch_size=1,
                                  generator=key, upd_color=[1])
    _assert_close_tree(actual.states, reference.states)
    for resource, original in zip(actual.graph.models, before):
        if int(resource.state.step):
            _replicated(resource.state, _devices())
        else:
            assert resource.state is original


@pytest.mark.parametrize('microbatch_size,microbatch_unit', [
    (-1, 'trajectories'), (-1, 'points'), (1, 'trajectories'),
    (3, 'trajectories'), (3, 'points'), (128, 'points'),
])
def test_public_manual_packed_update_matches_one_numpy_optimizer_step(microbatch_size, microbatch_unit):
    trainer = _regression()
    data = _regression_data()
    expected, metrics, _ = _oracle_step(trainer, _selected(data, np.arange(data[0]['_valid'].shape[1])))
    packed = trainer.compact(data)
    report = trainer.optimize(packed, upd_color=[1], microbatch_size=microbatch_size,
                              microbatch_unit=microbatch_unit)
    _assert_close_tree(tuple(resource.state for resource in trainer.graph.models), expected)
    np.testing.assert_allclose(np.asarray([report[name] for name in
                                         ('loss', 'prediction', 'sample_id')]).T,
                               metrics, rtol=5e-5, atol=3e-6)
    assert report['valid_samples'] == [int(np.asarray(row['_valid']).sum()) for row in data]
    assert all(int(resource.state.step) == 1 for resource in trainer.graph.models)


@pytest.mark.parametrize('microbatch_size,microbatch_unit', [
    (-1, 'trajectories'), (1, 'trajectories'), (3, 'trajectories'), (3, 'points'),
])
def test_packed_units_preserve_original_epoch_groups_and_remainders(microbatch_size, microbatch_unit):
    trainer, reference = _regression(), _regression()
    data = _regression_data()
    options = dict(epochs=2, minibatch_size=3, generator=jax.random.PRNGKey(37))
    reports = []
    for batch in _oracle_epoch_batches(data, **options):
        expected, metrics, _ = _oracle_step(reference, batch)
        for resource, state in zip(reference.graph.models, expected):
            resource.state = state
        reports.append(dict(zip(('loss', 'prediction', 'sample_id'), metrics.T.tolist()),
                            valid_samples=[int(np.asarray(row['_valid']).sum()) for row in batch]))
    report = trainer.optimize_epochs(trainer.compact(data), upd_color=[1],
                                      microbatch_size=microbatch_size,
                                      microbatch_unit=microbatch_unit, **options)
    _assert_close_tree(trainer.states, reference.states)
    _assert_report(report, reports)


def test_packed_point_size_rounds_up_per_device_and_stays_bounded(monkeypatch):
    trainer = _regression()
    engine = ResidentOptimizer(trainer)
    packed = trainer.compact(_regression_data())
    width = (3 + len(_devices()) - 1) // len(_devices())
    widths = []
    differentiate = trainer._minibatch_gradients

    def inspect(states, batch, colors):
        widths.append(batch['_valid'].shape[0])
        return differentiate(states, batch, colors)

    monkeypatch.setattr(trainer, '_minibatch_gradients', inspect)
    expected = _gradient_oracle(trainer, _selected(_regression_data(), np.arange(packed.episodes)))
    actual = engine.gradient_probe(packed, np.arange(packed.episodes), upd_color=[1],
                                   microbatch_size=3, microbatch_unit='points')
    assert widths and set(widths) == {width}
    _assert_close_tree(actual, expected)


def test_packed_default_and_minus_one_units_share_execution_and_keep_full_group(monkeypatch):
    trainer = _regression()
    packed = trainer.compact(_regression_data())
    widths = []
    differentiate = trainer._minibatch_gradients

    def inspect(states, batch, colors):
        widths.append(batch['_valid'].shape[0])
        return differentiate(states, batch, colors)

    monkeypatch.setattr(trainer, '_minibatch_gradients', inspect)
    trainer.optimize(packed, upd_color=[1])
    assert widths and max(widths) >= max(packed.valid_counts.max(axis=1))
    engine = trainer._resident_optimizer()
    executables = dict(engine.executables)
    trainer.optimize(packed, upd_color=[1], microbatch_size=-1, microbatch_unit='points')
    assert engine.executables.keys() == executables.keys()
    assert all(engine.executables[key] is value for key, value in executables.items())
    assert all(int(resource.state.step) == 2 for resource in trainer.graph.models)


def test_packed_trajectory_microgroups_keep_complete_owned_episodes(monkeypatch):
    from collections import Counter
    trainer = _regression()
    data = _regression_data()
    packed = trainer.compact(data)
    observations = []
    differentiate = trainer._minibatch_gradients
    common = trainer.graph.model_index(trainer.graph.common)

    def record(owner, player, sample_ids, valid, step):
        assert int(np.asarray(step)) == 0
        observations.append((int(np.asarray(owner)), int(np.asarray(player)[0]),
                             tuple(np.asarray(sample_ids)[np.asarray(valid)].astype(int))))

    def inspect(states, batch, colors):
        assert batch['_valid'].shape[0] <= 2 * packed.horizon
        jax.debug.callback(record, jax.lax.axis_index('learners'), batch['player'],
                           batch['sample_id'], batch['_valid'], states[common].step)
        return differentiate(states, batch, colors)

    monkeypatch.setattr(trainer, '_minibatch_gradients', inspect)
    trainer.optimize(packed, upd_color=[1], microbatch_size=2, microbatch_unit='trajectories')
    jax.effects_barrier()
    for owner in range(len(_devices())):
        for player, row in enumerate(data):
            expected = []
            for start in range(0, packed.episodes, 2):
                ids = [episode for episode in range(start, min(start + 2, packed.episodes)) if episode // 4 == owner]
                expected.append(tuple(int(np.asarray(row['sample_id'])[time, episode]) for episode in ids
                                      for time in range(packed.horizon) if np.asarray(row['_valid'])[time, episode]))
            actual = [ids for device, seat, ids in observations if device == owner and seat == player]
            assert Counter(ids for ids in actual if ids) == Counter(ids for ids in expected if ids)
    assert all(int(resource.state.step) == 1 for resource in trainer.graph.models)


@pytest.mark.parametrize('operation', ['update', 'optimize', 'epochs', 'probe', 'plan'])
def test_packed_rollout_rejects_a_different_trainer_before_state_mutation(operation):
    first, second = _regression(), _regression()
    packed = first.compact(_regression_data())
    before = tuple(resource.state for resource in second.graph.models)
    options = dict(epochs=1, minibatch_size=3, generator=jax.random.PRNGKey(5))
    engine = ResidentOptimizer(second)
    with pytest.raises(ValueError, match='trainer|owner|bound'):
        if operation == 'update':
            second.update(packed, upd_color=[1])
        elif operation == 'optimize':
            second.optimize(packed, upd_color=[1])
        elif operation == 'epochs':
            second.optimize_epochs(packed, upd_color=[1], **options)
        elif operation == 'probe':
            engine.gradient_probe(packed, [0], upd_color=[1])
        else:
            engine.plan(packed, **options)
    assert all(resource.state is state for resource, state in zip(second.graph.models, before))


@pytest.mark.parametrize('options', [
    {'microbatch_size': None}, {'microbatch_size': 0}, {'microbatch_size': -2},
    {'microbatch_size': True}, {'microbatch_size': 1.0},
    {'microbatch_unit': None}, {'microbatch_unit': 'rows'}, {'microbatch_unit': 1},
])
def test_packed_invalid_microbatch_options_reject_before_update(options):
    trainer = _regression()
    packed = trainer.compact(_regression_data())
    before = tuple(resource.state for resource in trainer.graph.models)
    with pytest.raises(ValueError, match='microbatch'):
        trainer.optimize(packed, upd_color=[1], **options)
    with pytest.raises(ValueError, match='microbatch'):
        trainer.optimize_epochs(packed, upd_color=[1], epochs=1, minibatch_size=3,
                                 generator=jax.random.PRNGKey(0), **options)
    assert all(resource.state is state for resource, state in zip(trainer.graph.models, before))


def test_explicit_compact_validates_rank_ownership_and_chunk_size():
    trainer = _regression()
    data = _regression_data()
    with pytest.raises(ValueError, match='rank|dimensions|trajectory|trajectories'):
        trainer.compact(_selected(data, [0]))
    for invalid in (0, -1, True, 1.5, None):
        with pytest.raises(ValueError, match='target_chunk_size'):
            trainer.compact(data, target_chunk_size=invalid)
    if len(_devices()) > 1:
        uneven = tuple({name: value[:, :-1] for name, value in row.items()} for row in data)
        with pytest.raises(ValueError, match='divid|divisib|even'):
            trainer.compact(uneven)


def test_compiled_packed_updates_reuse_code_without_capturing_previous_data():
    trainer, reference = _regression(), _regression()
    original = _regression_data()
    modified = tuple(dict(row, target=row['target'] + 1.25 * (player + 1),
                          information_set=row['information_set'] * .7,
                          _valid=row['_valid'].at[0, 0].set(False))
                     for player, row in enumerate(original))
    options = dict(epochs=1, minibatch_size=3, generator=jax.random.PRNGKey(31),
                   microbatch_size=3, microbatch_unit='points')
    cached = None
    capacity = None
    for data in (original, modified, original):
        reports = []
        for batch in _oracle_epoch_batches(data, **{key: options[key] for key in ('epochs', 'minibatch_size', 'generator')}):
            expected, metrics, _ = _oracle_step(reference, batch)
            for resource, state in zip(reference.graph.models, expected):
                resource.state = state
            reports.append(dict(zip(('loss', 'prediction', 'sample_id'), metrics.T.tolist()),
                                valid_samples=[int(np.asarray(row['_valid']).sum()) for row in batch]))
        packed = trainer.compact(data)
        assert capacity is None or packed.capacity == capacity
        capacity = packed.capacity
        report = trainer.optimize_epochs(packed, upd_color=[1], **options)
        _assert_close_tree(trainer.states, reference.states)
        _assert_report(report, reports)
        executables = {key: value for key, value in trainer._resident_optimizer().executables.items()
                       if key.startswith('optimize_')}
        assert len(executables) == 1
        if cached is not None:
            assert executables.keys() == cached.keys()
            assert all(executables[key] is value for key, value in cached.items())
        assert all(value._cache_size() == 1 for value in executables.values())
        cached = executables


class _ManyPlayerEnvironment(_RegressionEnvironment):
    def __init__(self, players):
        self.num_players = players

    def features(self, state):
        return {'information_set': jnp.ones((self.num_players, 3)),
                'target': jnp.zeros(self.num_players), 'sample_id': jnp.zeros(self.num_players)}

    def legal_action_mask(self, state):
        return jnp.ones((self.num_players, 2), bool)

    def active_players(self, state):
        return jnp.ones(self.num_players, bool)

    def step(self, state, actions, key):
        return state + 1, jnp.zeros(self.num_players), jnp.array(True)


class _ColoredRegression(leg.DRLGraph):
    def __init__(self, players, *, prepare=False, unsupported_segment=False):
        super().__init__()

        def model():
            return leg.model(init=lambda key, inputs: jnp.linspace(.04, .16, inputs.shape[-1]),
                             apply=lambda weights, inputs: jnp.dot(inputs, weights),
                             optimizer=optax.chain(optax.clip_by_global_norm(.7),
                                                   optax.adamw(.013, weight_decay=.03)))

        resources = [model() for _ in range(players)]
        self.local = leg.ModelList(resources)
        self.common = model()
        if prepare:
            with leg.backward(color=5):
                source = self.local[self.player](self.env.information_set) if unsupported_segment else self.reward
                self.prepared = leg.stop_gradient(leg.aggregate(source, 'sum', object='segment', discount=.9))
        with leg.backward(color=7):
            self.prediction = self.local[self.player](self.env.information_set) + self.common(self.env.information_set)
            self.loss = (self.prediction - (self.prepared if prepare else self.env.target)) ** 2
            self.minimize(self.loss, models=[*resources, self.common])
            self.strategy = self.legal_action_mask / self.action_set_size
            self.metrics = {'prediction': self.prediction, 'sample_id': self.env.sample_id}


def _many_player(prepare=False, unsupported_segment=False):
    players = 3
    return _ColoredRegression(players, prepare=prepare, unsupported_segment=unsupported_segment).bind(
        _ManyPlayerEnvironment(players), batch_size=4 * len(_devices()), seed=13,
        devices=_devices(), parallel=('optimizing',)).trainer


def _many_player_data():
    data = _regression_data()
    third = dict(data[1], _valid=jnp.zeros_like(data[1]['_valid']).at[2, 0].set(True))
    # Build features afresh so this sparse third seat's sole record is finite.
    masks = tuple(np.asarray(row['_valid']) for row in (*data, third))
    data = _hard_records(data[0]['_valid'].shape, masks)
    return tuple(dict(row, _alive=jnp.ones_like(row['_valid']),
                      reward=jnp.arange(row['_valid'].size).reshape(row['_valid'].shape) / 30.)
                 for row in data)


def test_three_players_and_nonstandard_training_color_match_independent_gradient_oracle():
    trainer = _many_player()
    data = _many_player_data()
    batch = _selected(data, np.arange(data[0]['_valid'].shape[1]))
    expected, metrics, _ = _oracle_step(trainer, batch)
    report = trainer.optimize(trainer.compact(data), upd_color=[7],
                              microbatch_size=3, microbatch_unit='points')
    _assert_close_tree(tuple(resource.state for resource in trainer.graph.models), expected)
    np.testing.assert_allclose(np.asarray([report[name] for name in ('loss', 'prediction', 'sample_id')]).T,
                               metrics, rtol=5e-5, atol=3e-6)
    assert report['valid_samples'][-1] == 1


def test_nonstandard_preparation_color_preserves_opponent_turn_rewards_and_cached_targets():
    actual, reference = _many_player(True), _many_player(True)
    data = _many_player_data()
    packed = actual.update(actual.compact(data), upd_color=[5])
    prepared = reference.update(data, upd_color=[5])
    for player, (row, timeline) in enumerate(zip(packed.records, packed.timeline)):
        valid = np.asarray(data[player]['_valid'])
        key = actual.graph._node_key(actual.graph.prepared.index)
        np.testing.assert_allclose(np.asarray(timeline[key])[valid], np.asarray(prepared[player][key])[valid], rtol=1e-6, atol=1e-6)
        points = np.asarray(row['_valid'])
        times, ids = np.asarray(row['_time_index'])[points], np.asarray(row['_trajectory_id'])[points]
        np.testing.assert_allclose(np.asarray(row[key])[points], np.asarray(prepared[player][key])[times, ids], rtol=1e-6, atol=1e-6)
    options = dict(epochs=2, minibatch_size=3, generator=jax.random.PRNGKey(11), upd_color=[7],
                   microbatch_size=3, microbatch_unit='points')
    expected = reference.optimize_epochs(prepared, **options)
    report = actual.optimize_epochs(packed, **options)
    _assert_close_tree(actual.states, reference.states)
    _assert_close_tree(report, expected)


def test_segment_aggregate_cannot_silently_drop_model_values_on_inactive_turns():
    trainer = _many_player(True, unsupported_segment=True)
    before = tuple(resource.state for resource in trainer.graph.models)
    with pytest.raises(ValueError, match='inactive|segment|unsupported|support'):
        trainer.update(trainer.compact(_many_player_data()), upd_color=[5])
    assert all(resource.state is old for resource, old in zip(trainer.graph.models, before))


def test_packed_ppo_missing_preparation_cache_fails_without_publishing_updates():
    trainer = _ppo(.9, .7, True)
    packed = trainer.compact(_target_data())
    before = tuple(resource.state for resource in trainer.graph.models)
    with pytest.raises(ValueError, match='cache|color|update|aggregate'):
        trainer.optimize(packed, upd_color=[1], microbatch_size=3, microbatch_unit='points')
    assert all(resource.state is state for resource, state in zip(trainer.graph.models, before))


def test_cached_target_program_uses_each_new_packed_rollout_values():
    trainer = _ppo(.9, .7, True)
    data = _target_data()
    modified = tuple(dict(row, reward=row['reward'] * (1.3 + player * .2),
                          full_info=row['full_info'] * .8) for player, row in enumerate(data))
    for source in (data, modified, data):
        packed = trainer.update(trainer.compact(source, target_chunk_size=3), upd_color=[0])
        oracle = _gae_oracle(source, .9, .7, True)
        for player, timeline in enumerate(packed.timeline):
            valid = np.asarray(source[player]['_valid'])
            for node, expected in zip((trainer.graph.advantage, trainer.graph.return_target), oracle[player][:2]):
                actual = np.asarray(timeline[trainer.graph._node_key(node.index)])[valid]
                np.testing.assert_allclose(actual, expected[valid], rtol=3e-6, atol=3e-6)


def test_ppo_compiled_positive_points_pack_before_targets_and_match_raw_manual_updates(monkeypatch):
    from test_drl_compiled_updates import _algorithm
    from LiteEFG.drl.env import Goofspiel
    trainers = [_algorithm('partial').bind(Goofspiel(num_cards=3), batch_size=4 * len(_devices()),
                                          seed=11, devices=_devices(), parallel=('optimizing',)).trainer
                for _ in range(2)]
    expected = trainers[0].graph.train(iterations=1, epochs=2, minibatch_size=3,
                                        compiled_updates=False, microbatch_size=3, microbatch_unit='points')
    trainer = trainers[1]
    calls = []
    original_compact, original_update, original_optimize = trainer.compact, trainer.update, trainer.optimize_epochs

    def compact(data, **options):
        calls.append(('compact', isinstance(data, PackedRollout)))
        return original_compact(data, **options)

    def update(data, **options):
        calls.append(('update', isinstance(data, PackedRollout)))
        return original_update(data, **options)

    def optimize(data, **options):
        calls.append(('optimize', isinstance(data, PackedRollout)))
        assert options['precompute_color'] == [0]
        return original_optimize(data, **options)

    monkeypatch.setattr(trainer, 'compact', compact)
    monkeypatch.setattr(trainer, 'update', update)
    monkeypatch.setattr(trainer, 'optimize_epochs', optimize)
    report = trainer.graph.train(iterations=1, epochs=2, minibatch_size=3,
                                 compiled_updates=True, microbatch_size=3, microbatch_unit='points')
    assert calls == [('optimize', False), ('compact', False), ('update', True)]
    _assert_close_tree(trainer.states, trainers[0].states)
    _assert_close_tree(report, expected)
    np.testing.assert_array_equal(trainer.key, trainers[0].key)


@pytest.mark.parametrize('train_options,compact_batches', [
    ({}, True), ({'microbatch_size': 1}, True),
    ({'microbatch_size': 3, 'microbatch_unit': 'points'}, False),
])
def test_ppo_other_modes_keep_raw_dispatch(monkeypatch, train_options, compact_batches):
    from test_drl_compiled_updates import _algorithm
    from LiteEFG.drl.env import Goofspiel
    trainer = _algorithm('partial').bind(Goofspiel(num_cards=3), batch_size=4 * len(_devices()), seed=11,
                                         compact_batches=compact_batches).trainer

    def unexpected(*args, **kwargs):
        pytest.fail('PPO should retain raw dispatch for this mode')

    monkeypatch.setattr(trainer, 'compact', unexpected)
    trainer.graph.train(iterations=1, epochs=1, minibatch_size=3, **train_options)


def test_ppo_uneven_optimizer_ownership_keeps_supported_raw_point_path(monkeypatch):
    if len(_devices()) == 1:
        return
    from test_drl_compiled_updates import _algorithm
    from LiteEFG.drl.env import Goofspiel
    trainer = _algorithm('partial').bind(Goofspiel(num_cards=3), batch_size=5, seed=11,
                                         devices=_devices(), parallel=('optimizing',)).trainer

    def unexpected(*args, **kwargs):
        pytest.fail('Uneven owner counts must retain the supported raw path')

    monkeypatch.setattr(trainer, 'compact', unexpected)
    report = trainer.graph.train(iterations=1, epochs=1, minibatch_size=3,
                                 microbatch_size=3, microbatch_unit='points')
    assert report['valid_samples'] == [15, 15]


@pytest.mark.parametrize('compiled', [False, True])
def test_public_positive_points_auto_pack_and_match_independent_gradient_oracle(monkeypatch, compiled):
    trainer, reference = _regression(), _regression()
    data = _regression_data()
    reports = []
    options = dict(epochs=2, minibatch_size=3, generator=jax.random.PRNGKey(31))
    batches = (_oracle_epoch_batches(data, **options) if compiled else
               [_selected(data, np.arange(data[0]['_valid'].shape[1]))])
    for batch in batches:
        expected, metrics, _ = _oracle_step(reference, batch)
        for resource, state in zip(reference.graph.models, expected):
            resource.state = state
        reports.append(dict(zip(('loss', 'prediction', 'sample_id'), metrics.T.tolist()),
                            valid_samples=[int(np.asarray(row['_valid']).sum()) for row in batch]))
    calls = []
    compact = trainer.compact

    def record(source, **kwargs):
        calls.append(source)
        return compact(source, **kwargs)

    monkeypatch.setattr(trainer, 'compact', record)
    if compiled:
        report = trainer.optimize_epochs(data, upd_color=[1], microbatch_size=3,
                                          microbatch_unit='points', **options)
    else:
        batch = next(iter(leg.dataloader(data, batch_size=data[0]['_valid'].shape[1])))
        report = trainer.optimize(batch, upd_color=[1], microbatch_size=3, microbatch_unit='points')
    assert len(calls) == 1
    assert calls[0][0]['_valid'].shape == data[0]['_valid'].shape
    _assert_close_tree(trainer.states, reference.states)
    _assert_report(report, reports)


def test_manual_auto_pack_restores_shuffled_trajectories_and_preserves_cached_targets(monkeypatch):
    actual = _ppo(.9, .7, False)
    reference = _ppo(.9, .7, False, compact_batches=False)
    data = tuple(dict(row, zero_width=jnp.zeros((*row['_valid'].shape, 0))) for row in _target_data())
    prepared = actual.update(data, upd_color=[0])
    reference.update(data, upd_color=[0])
    episodes = data[0]['_valid'].shape[1]
    key = jax.random.PRNGKey(17)
    batch = next(iter(leg.dataloader(prepared, batch_size=episodes, shuffle=True, generator=key)))
    assert batch.trajectory_steps == 4
    _, shuffled = jax.random.split(key)
    order = np.asarray(jax.random.permutation(shuffled, episodes))
    seen = []
    compact = actual.compact

    def record(restored, **options):
        seen.append(restored)
        for row, expected in zip(restored, prepared):
            for name in expected:
                np.testing.assert_array_equal(row[name], np.asarray(expected[name])[:, order])
        return compact(restored, **options)

    def unexpected_update(*args, **kwargs):
        pytest.fail('Manual optimize must reuse existing trajectory target caches')

    monkeypatch.setattr(actual, 'compact', record)
    monkeypatch.setattr(actual, 'update', unexpected_update)
    expected = reference.optimize(batch, upd_color=[1], microbatch_size=3, microbatch_unit='points')
    report = actual.optimize(batch, upd_color=[1], microbatch_size=3, microbatch_unit='points')
    assert len(seen) == 1 and seen[0][0]['zero_width'].shape == (4, episodes, 0)
    _assert_close_tree(actual.states, reference.states)
    _assert_close_tree(report, expected)


def test_plain_flat_batch_auto_pack_uses_one_step_trajectories(monkeypatch):
    trainer = _regression()
    data = _regression_data()
    wrapped = next(iter(leg.dataloader(data, batch_size=data[0]['_valid'].shape[1])))
    batch = tuple(dict(row) for row in wrapped)
    expected, _, _ = _oracle_step(trainer, batch)
    shapes = []
    compact = trainer.compact

    def record(restored, **options):
        shapes.append(restored[0]['_valid'].shape)
        return compact(restored, **options)

    monkeypatch.setattr(trainer, 'compact', record)
    trainer.optimize(batch, upd_color=[1], microbatch_size=3, microbatch_unit='points')
    assert shapes == [(1, batch[0]['_valid'].size)]
    _assert_close_tree(tuple(resource.state for resource in trainer.graph.models), expected)


@pytest.mark.parametrize('compact_batches', [False, True])
def test_public_epochs_precomputes_targets_once_before_any_optimizer_step(monkeypatch, compact_batches):
    actual = _ppo(.9, .7, True, compact_batches=compact_batches)
    reference = _ppo(.9, .7, True, compact_batches=False)
    data = _target_data()
    options = dict(epochs=2, minibatch_size=3, generator=jax.random.PRNGKey(19), upd_color=[1],
                   microbatch_size=3, microbatch_unit='points')
    prepared = reference.update(data, upd_color=[0])
    expected = reference.optimize_epochs(prepared, **options)
    calls = []
    compact, update = actual.compact, actual.update

    def record_compact(source, **kwargs):
        calls.append(('compact', isinstance(source, PackedRollout)))
        return compact(source, **kwargs)

    def record_update(source, **kwargs):
        calls.append(('update', isinstance(source, PackedRollout)))
        assert tuple(kwargs['upd_color']) == (0,)
        assert all(int(resource.state.step) == 0 for resource in actual.graph.models)
        return update(source, **kwargs)

    monkeypatch.setattr(actual, 'compact', record_compact)
    monkeypatch.setattr(actual, 'update', record_update)
    report = actual.optimize_epochs(data, precompute_color=[0], **options)
    assert calls == ([('compact', False), ('update', True)] if compact_batches else [('update', False)])
    _assert_close_tree(actual.states, reference.states)
    _assert_close_tree(report, expected)


def test_public_epochs_precompute_color_supports_nonstandard_graph_colors():
    actual, reference = _many_player(True), _many_player(True)
    data = _many_player_data()
    prepared = reference.update(data, upd_color=[5])
    options = dict(epochs=2, minibatch_size=3, generator=jax.random.PRNGKey(11), upd_color=[7],
                   microbatch_size=3, microbatch_unit='points')
    expected = reference.optimize_epochs(prepared, **options)
    report = actual.optimize_epochs(data, precompute_color=[5], **options)
    _assert_close_tree(actual.states, reference.states)
    _assert_close_tree(report, expected)


def test_public_epochs_default_precompute_keeps_targets_frozen(monkeypatch):
    trainer = _ppo(.9, .7, True)
    data = trainer.update(_target_data(), upd_color=[0])
    original = jax.tree.map(lambda value: np.asarray(value).copy(), data)

    def unexpected(*args, **kwargs):
        pytest.fail('Default precompute_color=None must reuse existing target caches')

    monkeypatch.setattr(trainer, 'update', unexpected)
    trainer.optimize_epochs(data, epochs=2, minibatch_size=3, generator=jax.random.PRNGKey(19),
                              upd_color=[1], microbatch_size=3, microbatch_unit='points')
    for before, after in zip(jax.tree.leaves(original), jax.tree.leaves(data)):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize('operation', ['manual', 'epochs'])
def test_explicit_packed_inputs_do_not_repack(monkeypatch, operation):
    trainer = _regression()
    packed = trainer.compact(_regression_data())

    def unexpected(*args, **kwargs):
        pytest.fail('A supplied PackedRollout must preserve its current owner allocation')

    monkeypatch.setattr(trainer, 'compact', unexpected)
    if operation == 'manual':
        trainer.optimize(packed, upd_color=[1], microbatch_size=3, microbatch_unit='points')
    else:
        trainer.optimize_epochs(packed, epochs=2, minibatch_size=3, generator=jax.random.PRNGKey(19),
                                  upd_color=[1], microbatch_size=3, microbatch_unit='points')


def test_explicit_packed_precompute_refreshes_current_parameters_without_repacking(monkeypatch):
    trainer = _ppo(.9, .7, True)
    packed = trainer.update(trainer.compact(_target_data()), upd_color=[0])
    critic = trainer.graph.critic
    critic.state = critic.state._replace(params=critic.state.params * .8)
    captured = []
    update = trainer.update

    def unexpected(*args, **kwargs):
        pytest.fail('Explicit packed precomputation must not repack original features')

    def record(source, **options):
        assert source is packed
        result = update(source, **options)
        captured.append(result)
        return result

    monkeypatch.setattr(trainer, 'compact', unexpected)
    monkeypatch.setattr(trainer, 'update', record)
    trainer.optimize_epochs(packed, precompute_color=[0], epochs=1, minibatch_size=3,
                              generator=jax.random.PRNGKey(19), upd_color=[1],
                              microbatch_size=3, microbatch_unit='points')
    assert len(captured) == 1
    original_data = _target_data()
    scaled = tuple(dict(row, full_info=row['full_info'] * .8) for row in original_data)
    expected = _gae_oracle(scaled, .9, .7, True)
    for player, row in enumerate(captured[0].timeline):
        valid = np.asarray(original_data[player]['_valid'])
        for node, values in zip((trainer.graph.advantage, trainer.graph.return_target), expected[player][:2]):
            np.testing.assert_allclose(np.asarray(row[trainer.graph._node_key(node.index)])[valid],
                                       values[valid], rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize('invalid', [True, -2, ['bad']])
def test_public_epochs_invalid_precompute_color_never_updates_models(invalid):
    trainer = _ppo(.9, .7, True)
    before = tuple(resource.state for resource in trainer.graph.models)
    with pytest.raises(ValueError, match='color'):
        trainer.optimize_epochs(_target_data(), precompute_color=invalid,
                                  epochs=1, minibatch_size=3, generator=jax.random.PRNGKey(0),
                                  upd_color=[1], microbatch_size=3, microbatch_unit='points')
    assert all(resource.state is old for resource, old in zip(trainer.graph.models, before))


@pytest.mark.parametrize('compiled', [False, True])
def test_auto_resident_and_raw_trajectory_unit_switches_keep_distinct_code(monkeypatch, compiled):
    trainer = _regression()
    data = _regression_data()
    batch = next(iter(leg.dataloader(data, batch_size=data[0]['_valid'].shape[1])))
    resident_codes = None
    for index, unit in enumerate(('points', 'trajectories', 'points')):
        options = dict(upd_color=[1], microbatch_size=3, microbatch_unit=unit)
        if compiled:
            trainer.optimize_epochs(data, epochs=1, minibatch_size=3,
                                      generator=jax.random.PRNGKey(11), **options)
            stage = 'optimize_epochs'
        else:
            trainer.optimize(batch, **options)
            stage = 'optimize'
        current = {key: value for key, value in trainer._resident_optimizer().executables.items()
                   if key.startswith('optimize_')}
        assert len(current) == 1
        if resident_codes is not None:
            assert current.keys() == resident_codes.keys()
            assert all(current[key] is value for key, value in resident_codes.items())
        resident_codes = current
        raw = [value for key, value in trainer._compiled.items()
               if isinstance(key, tuple) and key[0] == stage]
        assert len(raw) == (0 if index == 0 else 1)


def test_ppo_releases_dense_rollout_wrapper_before_packed_target_evaluation(monkeypatch):
    import gc
    import weakref
    trainer = _ppo(.9, .7, True)
    collect, update = trainer.collect, trainer.update
    references = []

    class WeakRollout(list):
        pass

    def record_collect():
        rollout = WeakRollout(collect())
        references.append(weakref.ref(rollout))
        return rollout

    def record_update(data, **options):
        assert isinstance(data, PackedRollout)
        gc.collect()
        assert references and references[-1]() is None, 'PPO retained the dense rollout during packed targets'
        return update(data, **options)

    monkeypatch.setattr(trainer, 'collect', record_collect)
    monkeypatch.setattr(trainer, 'update', record_update)
    trainer.graph.train(iterations=1, epochs=1, minibatch_size=3,
                         microbatch_size=3, microbatch_unit='points')
    assert len(references) == 1 and references[0]() is None


@pytest.mark.parametrize('compact_batches', [False, True])
def test_public_epochs_implicit_generator_matches_explicit_shuffle_and_advances_key_once(compact_batches):
    actual = _regression(compact_batches=compact_batches)
    reference = _regression(compact_batches=compact_batches)
    data = _regression_data()
    reference_key = np.asarray(reference.key).copy()
    options = dict(epochs=2, minibatch_size=3, upd_color=[1],
                   microbatch_size=3, microbatch_unit='points')
    for _ in range(2):
        next_key, shuffle = jax.random.split(actual.key)
        expected = reference.optimize_epochs(data, generator=shuffle, **options)
        report = actual.optimize_epochs(data, **options)
        np.testing.assert_array_equal(actual.key, next_key)
        np.testing.assert_array_equal(reference.key, reference_key)
        _assert_close_tree(actual.states, reference.states)
        _assert_close_tree(report, expected)


def test_public_epochs_precompute_failure_does_not_advance_implicit_generator(monkeypatch):
    trainer = _ppo(.9, .7, True)
    before_key = np.asarray(trainer.key).copy()
    before = tuple(resource.state for resource in trainer.graph.models)

    def fail(data, **options):
        assert isinstance(data, PackedRollout)
        raise RuntimeError('target evaluation failed for regression')

    monkeypatch.setattr(trainer, 'update', fail)
    with pytest.raises(RuntimeError, match='target evaluation failed'):
        trainer.optimize_epochs(_target_data(), precompute_color=[0], epochs=1, minibatch_size=3,
                                  upd_color=[1], microbatch_size=3, microbatch_unit='points')
    np.testing.assert_array_equal(trainer.key, before_key)
    assert all(resource.state is old for resource, old in zip(trainer.graph.models, before))


@pytest.mark.parametrize('compiled', [False, True])
def test_ppo_target_failure_keeps_the_key_at_its_post_collect_value(monkeypatch, compiled):
    trainer = _ppo(.9, .7, True)
    collect = trainer.collect
    after_collect = []
    before = tuple(resource.state for resource in trainer.graph.models)

    def record_collect():
        data = collect()
        after_collect.append(np.asarray(trainer.key).copy())
        return data

    def fail(*args, **kwargs):
        raise RuntimeError('target evaluation failed for regression')

    monkeypatch.setattr(trainer, 'collect', record_collect)
    monkeypatch.setattr(trainer, 'update', fail)
    with pytest.raises(RuntimeError, match='target evaluation failed'):
        trainer.graph.train(iterations=1, epochs=1, minibatch_size=3, compiled_updates=compiled,
                             microbatch_size=3, microbatch_unit='points')
    assert len(after_collect) == 1
    np.testing.assert_array_equal(trainer.key, after_collect[0])
    assert all(resource.state is old for resource, old in zip(trainer.graph.models, before))


def test_public_epochs_optimizer_failure_advances_the_implicit_generator_once():
    trainer = _regression()
    data = list(_regression_data())
    data[0] = dict(data[0], target=jnp.where(data[0]['_valid'], jnp.nan, data[0]['target']))
    next_key, _ = jax.random.split(trainer.key)
    with pytest.raises(FloatingPointError, match='Non-finite'):
        trainer.optimize_epochs(tuple(data), epochs=1, minibatch_size=3,
                                  upd_color=[1], microbatch_size=3, microbatch_unit='points')
    np.testing.assert_array_equal(trainer.key, next_key)
    assert all(int(resource.state.step) == 0 for resource in trainer.graph.models)


_VIRTUAL_CPU_SHARDS = 4


class _ResidentShard:
    def __init__(self, index):
        if not 0 <= index < _VIRTUAL_CPU_SHARDS:
            raise ValueError(f'Invalid resident test shard: {index}')
        self.index = index

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, config, items):
        # Run after -k excludes the subprocess launchers. Partition collected
        # cases (including all parameters and future tests) without omissions.
        selected = items[self.index::_VIRTUAL_CPU_SHARDS]
        deselected = [item for index, item in enumerate(items)
                      if index % _VIRTUAL_CPU_SHARDS != self.index]
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


@pytest.mark.parametrize('shard', range(_VIRTUAL_CPU_SHARDS),
                         ids=lambda shard: f'shard-{shard + 1}')
@pytest.mark.parametrize('device_count', [2, 4])
def test_virtual_cpu_ownership_gae_and_gradient_collectives(device_count, shard):
    environment = dict(os.environ, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='',
                       XLA_PYTHON_CLIENT_PREALLOCATE='false', OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / 'tests'), os.environ.get('PYTHONPATH', ''))))
    flags = re.sub(r'--xla_force_host_platform_device_count=\d+', '', environment.get('XLA_FLAGS', ''))
    environment['XLA_FLAGS'] = flags + f' --xla_force_host_platform_device_count={device_count}'
    command = [sys.executable, str(Path(__file__).resolve()), str(device_count), str(shard)]
    try:
        result = subprocess.run(command, env=environment, text=True,
                                capture_output=True, timeout=300)
    except subprocess.TimeoutExpired as error:
        output = '\n'.join(value.decode(errors='replace') if isinstance(value, bytes) else value or ''
                           for value in (error.stdout, error.stderr))
        pytest.fail(f'Resident CPU {device_count}, shard {shard + 1}/{_VIRTUAL_CPU_SHARDS} '
                    f'timed out after 300 seconds.\n{output}', pytrace=False)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    assert f'CORE_RESIDENT_CPU_{device_count}_SHARD_{shard}_PASS' in result.stdout


if __name__ == '__main__':
    expected_devices, shard = map(int, sys.argv[1:])
    assert len(_devices()) == expected_devices
    code = pytest.main([str(Path(__file__).resolve()), '-vv', '-p', 'no:capture',
                        '-k', 'not virtual_cpu', '--tb=short', '--durations=10'],
                       plugins=[_ResidentShard(shard)])
    if code == 0:
        print(f'CORE_RESIDENT_CPU_{expected_devices}_SHARD_{shard}_PASS', flush=True)
    raise SystemExit(code)
