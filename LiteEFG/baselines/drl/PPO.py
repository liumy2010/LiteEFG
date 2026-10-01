#######################################################
# Proximal Policy Optimization (PPO)
# Schulman, John, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.
# "Proximal Policy Optimization Algorithms."
# arXiv preprint arXiv:1707.06347 (2017).
# https://arxiv.org/abs/1707.06347
#######################################################

"""Clipped PPO with an entropy bonus, expressed as an infoset-local DRLGraph."""

import math

import jax

import LiteEFG as leg
from LiteEFG.drl.graph import Model, ModelList, _validate_feature_name


def _network_handles(value, name):
    if isinstance(value, (Model, ModelList)):
        return value
    if isinstance(value, (list, tuple)):
        return ModelList(value)
    raise TypeError(f"{name} must be a Model handle or a sequence of Model handles; use leg.model(network)")


class graph(leg.DRLGraph):
    """PPO computations referencing caller-owned policy and critic resources.

    Color 0 evaluates the critic and freezes GAE advantages and return targets
    on complete trajectories. Color 1 evaluates the actor and critic losses
    against those targets for each minibatch. Pass ``leg.model`` handles or
    per-player sequences of handles. A single handle is shared by all players;
    repeated handles in a sequence define sharing groups. Models own their
    parameters and optimizers independently of this graph.
    The actor returns action logits; the critic returns one value per input.
    ``critic_feature`` selects the environment feature read by the critic.
    The actor reads ``information_set``. Optimizer configuration, including
    learning rate and clipping, belongs to the supplied model resources.
    """

    def __init__(self, policy_network, critic_network, num_actions=7,
                 clip_epsilon=0.2, value_coef=0.5,
                 entropy_coef=0.01, gamma=1.0,
                 gae_lambda=0.95, normalize_advantage=True,
                 critic_feature="full_info"):
        super().__init__()
        if not isinstance(num_actions, int) or isinstance(num_actions, bool) or num_actions < 1:
            raise ValueError("num_actions must be a positive integer")
        if not math.isfinite(clip_epsilon) or not 0 < clip_epsilon < 1:
            raise ValueError("clip_epsilon must be between zero and one")
        for name, value in (("value_coef", value_coef),
                            ("entropy_coef", entropy_coef)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, value in (("gamma", gamma), ("gae_lambda", gae_lambda)):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and between zero and one")
        if not isinstance(normalize_advantage, bool):
            raise ValueError("normalize_advantage must be a boolean")
        _validate_feature_name(critic_feature)

        self.num_actions = num_actions
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_advantage = normalize_advantage
        self.critic_feature = critic_feature
        self.actor = self.policy_network = _network_handles(policy_network, "policy_network")
        self.critic = self.critic_network = _network_handles(critic_network, "critic_network")

        def call(network, inputs):
            return (network[self.player] if isinstance(network, ModelList) else network)(inputs)

        with leg.backward(color=1):
            self.strategy = leg.masked_softmax(call(self.actor, self.env.information_set),
                                               self.legal_action_mask)
            self.value = call(self.critic, getattr(self.env, critic_feature)).squeeze(-1)

        with leg.backward(color=0):
            value = call(self.critic, getattr(self.env, critic_feature)).squeeze(-1)
            reward = leg.aggregate(self.reward, "sum", object="segment", discount=gamma)
            next_value = leg.aggregate(value, "sum", object="children", discount=gamma)
            delta = reward + next_value - value
            # A reverse scan computes A_k = delta_k + gamma**gap * lambda * A_{k+1},
            # where gap is the number of environment steps to our next decision.
            raw_advantage = leg.aggregate(delta, "sum", object="descendants", discount=gamma,
                                         decay=gae_lambda, include_self=True)
            self.return_target = leg.stop_gradient(raw_advantage + value)
            advantage = raw_advantage
            if normalize_advantage:
                mean = leg.aggregate(advantage, "mean", object="batch")
                variance = leg.aggregate((advantage - mean) ** 2, "mean", object="batch")
                advantage = (advantage - mean) / (variance + 1e-8) ** 0.5
            self.advantage = leg.stop_gradient(advantage)

        with leg.backward(color=1):
            log_prob = leg.clip(leg.gather(self.strategy, self.action), 1e-30, 1.0).log()
            log_ratio = log_prob - leg.stop_gradient(self.log_sampling_prob)
            ratio = leg.exp(log_ratio)
            unclipped = ratio * self.advantage
            clipped = leg.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * self.advantage
            self.policy_loss = -leg.minimum(unclipped, clipped)
            self.value_loss = (self.value - self.return_target) ** 2
            self.policy_entropy = leg.entropy(self.strategy)
            self.loss = (self.policy_loss + value_coef * self.value_loss
                         - entropy_coef * self.policy_entropy)
            self.minimize(self.loss, models=[resource for network in (self.actor, self.critic)
                                            for resource in (network if isinstance(network, ModelList)
                                                             else (network,))])

            self.metrics = {
                "policy_loss": self.policy_loss,
                "value_loss": self.value_loss,
                "entropy": self.policy_entropy,
                "approx_kl": ratio - 1 - log_ratio,
                "clip_fraction": ((ratio < 1 - clip_epsilon)
                                  | (ratio > 1 + clip_epsilon)).astype("float32"),
            }

    def current_strategy(self):
        return self.strategy

    def current_value(self):
        return self.value

    def train(self, iterations=1, *, epochs=4, minibatch_size=128,
              microbatch_size=-1, microbatch_unit="trajectories", compiled_updates=True):
        """Run PPO's schedule with this graph's bound trainer.

        Each iteration collects fresh trajectories and executes color 0 once;
        every minibatch update executes color 1. ``minibatch_size`` counts
        complete trajectories shared by all players; their valid decisions
        determine the loss weights. Report loss and graph metrics
        weighted by valid decisions across all updates, and average utility
        across the collected iterations. ``compiled_updates`` runs the same
        shuffled optimizer schedule in device loops with one metrics transfer
        per iteration; model training stays on the learner device.
        ``microbatch_size=-1`` evaluates each whole optimizer minibatch in one
        pass. A positive size counts complete trajectories by default, or valid
        decisions per player with ``microbatch_unit="points"``. Both units count
        globally across optimizing devices. Gradients accumulate within each
        original minibatch before one optimizer update. Compiled positive-point
        schedules with compaction enabled pack features once per collection
        on each optimizing device when the episode partition divides evenly.
        """
        trainer = self.trainer
        for name, value in (("iterations", iterations), ("epochs", epochs),
                            ("minibatch_size", minibatch_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(compiled_updates, bool):
            raise ValueError("compiled_updates must be a boolean")
        if (isinstance(microbatch_size, bool) or not isinstance(microbatch_size, int)
                or microbatch_size == 0 or microbatch_size < -1):
            raise ValueError("microbatch_size must be -1 or a positive integer")
        if (not isinstance(microbatch_unit, str)
                or microbatch_unit not in ("trajectories", "points")):
            raise ValueError('microbatch_unit must be "trajectories" or "points"')
        optimize_options = {
            "microbatch_size": microbatch_size,
            "microbatch_unit": microbatch_unit,
        }

        num_players = trainer.env.num_players
        metric_names = (name for name, node in self.metrics.items()
                        if self._colors[node.index] in (None, 1))
        totals = {name: [0.0] * num_players for name in ("loss", *metric_names)}
        valid_samples = [0] * num_players
        utilities = [0.0] * num_players
        for _ in range(iterations):
            data = trainer.collect()
            if compiled_updates:
                data = [data] # data.pop() removes data to help free up memory.
                reports = (trainer.optimize_epochs(
                    data.pop(), epochs=epochs, minibatch_size=minibatch_size,
                    upd_color=[1], precompute_color=[0], microbatch_size=microbatch_size,
                    microbatch_unit=microbatch_unit),)
            else:
                data = trainer.update(data, upd_color=[0])
                data = [data] # data.pop() removes data to help free up memory.
                trainer.key, key = jax.random.split(trainer.key)
                dataloader = leg.dataloader(data.pop(), batch_size=minibatch_size,
                                           shuffle=True, generator=key)
                reports = (trainer.optimize(batch, upd_color=[1], **optimize_options)
                           for _ in range(epochs) for batch in dataloader)
            for report in reports:
                for player, samples in enumerate(report["valid_samples"]):
                    valid_samples[player] += samples
                    for name, values in totals.items():
                        values[player] += report[name][player] * samples
            for player, utility in enumerate(report["mean_utility"]):
                utilities[player] += utility
        report.update({name: [total / max(samples, 1)
                              for total, samples in zip(values, valid_samples)]
                       for name, values in totals.items()})
        report.update(valid_samples=valid_samples,
                      mean_utility=[utility / iterations for utility in utilities])
        return report
