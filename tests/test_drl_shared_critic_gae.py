"""A current-mover shared critic bootstraps at the same player's next turn."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
pytest.importorskip("optax")
nn = pytest.importorskip("flax.linen")

import LiteEFG as leg
from LiteEFG.baselines.drl.PPO import graph
from LiteEFG.drl.environment import Environment


class AlternatingGame(Environment):
    """Four forced moves, intermediate rewards, and two padded rollout slots."""

    num_players = 2
    num_actions = 1
    max_steps = 6

    def init(self, key):
        del key
        return jnp.array(0, dtype=jnp.int32)

    def features(self, state):
        # Both player records receive the same current-mover value input.
        # Large opponent-turn values expose accidental one-ply bootstrapping.
        current_value = jnp.array([0.5, -90.0, 1.5, -92.0, 700.0])[state]
        return {
            "information_set": jnp.full((2, 1), state, dtype=jnp.float32),
            "full_info": jnp.full((2, 1), current_value),
        }

    def legal_action_mask(self, state):
        return self.active_players(state)[:, None]

    def active_players(self, state):
        return (jnp.arange(2) == state % 2) & (state < 4)

    def step(self, state, actions, key):
        del actions, key
        reward = jnp.array([0.0, 2.0, 0.0, 4.0, 0.0])[state]
        next_state = jnp.minimum(state + 1, 4)
        return next_state, jnp.stack((reward, -reward)), next_state == 4


class ScaledValue(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        scale = self.param("scale", nn.initializers.ones, ())
        return inputs * scale


@pytest.mark.parametrize("gae_lambda", [0.0, 0.6, 1.0])
def test_shared_current_mover_critic_uses_positive_same_player_bootstrap(gae_lambda):
    algorithm = graph(
        [leg.model(nn.Dense(1)) for _ in range(2)], leg.model(ScaledValue()), num_actions=1,
        critic_feature="full_info",
        gamma=0.5, gae_lambda=gae_lambda, normalize_advantage=False,
    )
    algorithm.bind(AlternatingGame(), batch_size=2, seed=17)
    trainer = algorithm.trainer
    records = trainer.collect()
    prepared = trainer.update(records, upd_color=[0])

    # P0: r_segment(0)=.5*2=1, next value=.5**2*1.5=.375;
    # delta0=.875 and delta2=.5, so A0=.875+.125*lambda.
    # P1: delta1=-2+.5**2*(-92)-(-90)=65 and delta3=88.
    targets = ([1.375 + 0.125 * gae_lambda, 2.0],
               [-25.0 + 22.0 * gae_lambda, -4.0])
    values = ([0.5, 1.5], [-90.0, -92.0])
    for player, decision_times in enumerate(([0, 2], [1, 3])):
        valid = np.asarray(records[player]["_valid"])
        expected_valid = np.zeros((6, 2), dtype=bool)
        expected_valid[decision_times] = True
        np.testing.assert_array_equal(valid, expected_valid)
        params = tuple(model.params for model in trainer.states[player])
        target, advantage = algorithm.evaluate(
            params, prepared[player], (algorithm.return_target, algorithm.advantage),
            upd_color=[1],
        )
        expected_target = np.broadcast_to(np.asarray(targets[player])[:, None], (2, 2))
        expected_advantage = expected_target - np.asarray(values[player])[:, None]
        indices = jnp.asarray(decision_times)
        np.testing.assert_allclose(target[indices], expected_target, rtol=0, atol=1e-5)
        np.testing.assert_allclose(advantage[indices], expected_advantage,
                                   rtol=0, atol=1e-5)

    # Changing estimates on the opponent's valid turns may only change that
    # opponent's bootstraps; it cannot enter P0's own-decision recurrence.
    altered = tuple(dict(player_records) for player_records in records)
    changed_features = altered[0]["full_info"].at[1].set(12345.0).at[3:].set(-12345.0)
    altered[0]["full_info"] = changed_features
    refreshed = trainer.update(altered, upd_color=[0])
    original = prepared[0][algorithm._node_key(algorithm.return_target.index)]
    changed = refreshed[0][algorithm._node_key(algorithm.return_target.index)]
    np.testing.assert_array_equal(changed[jnp.array([0, 2])], original[jnp.array([0, 2])])
