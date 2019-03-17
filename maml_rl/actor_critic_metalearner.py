from collections import OrderedDict

import torch
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
from torch.distributions.kl import kl_divergence

from maml_rl.utils.torch_utils import (weighted_mean, detach_distribution,
                                       weighted_normalize)
from maml_rl.utils.optimization import conjugate_gradient


class ActorCriticMetaLearner(object):
    """Meta-learner

    The meta-learner is responsible for sampling the trajectories/episodes
    (before and after the one-step adaptation), compute the inner loss, compute
    the updated parameters based on the inner-loss, and perform the meta-update.

    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic
        Meta-Learning for Fast Adaptation of Deep Networks", 2017
        (https://arxiv.org/abs/1703.03400)
    [2] Richard Sutton, Andrew Barto, "Reinforcement learning: An introduction",
        2018 (http://incompleteideas.net/book/the-book-2nd.html)
    [3] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan,
        Pieter Abbeel, "High-Dimensional Continuous Control Using Generalized
        Advantage Estimation", 2016 (https://arxiv.org/abs/1506.02438)
    [4] John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan,
        Pieter Abbeel, "Trust Region Policy Optimization", 2015
        (https://arxiv.org/abs/1502.05477)
    """

    def __init__(self, sampler, policy, critic, gamma=0.95,
                 fast_lr=0.5, tau=1.0, device='cpu'):
        self.sampler = sampler
        self.policy = policy
        self.critic = critic
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        self.gamma = gamma
        self.fast_lr = fast_lr
        self.tau = tau
        self.to(device)

    def inner_loss(self, episodes, params=None):
        """Compute the inner loss for the one-step gradient update. The inner
        loss is REINFORCE with baseline [2], computed on advantages estimated
        with Generalized Advantage Estimation (GAE, [3]).
        """
        values = self.critic(episodes.observations)
        advantages = episodes.gae(values, tau=self.tau) - values.squeeze()
        advantages = weighted_normalize(advantages, weights=episodes.mask)

        pi = self.policy(episodes.observations, params=params)
        log_probs = pi.log_prob(episodes.actions)
        if log_probs.dim() > 2:
            log_probs = torch.sum(log_probs, dim=2)
        loss = -weighted_mean(log_probs * advantages.detach(), dim=0,
                              weights=episodes.mask)

        return loss

    def inner_critic_loss(self, episodes, params=None):
        values = self.critic(episodes.observations)
        advantages = episodes.gae(values, tau=self.tau)
        advantages = weighted_normalize(advantages, weights=episodes.mask)
        value_loss = advantages.pow(2).mean()
        return value_loss

    def adapt(self, episodes, first_order=False):
        """Adapt the parameters of the policy network to a new task, from
        sampled trajectories `episodes`, with a one-step gradient update [1].
        """
        # Adapt the critic, then adapt the policy
        # Get the loss on the training episodes
        critic_loss = self.inner_critic_loss(episodes)
        critic_params = self.critic.update_params(critic_loss, step_size=self.fast_lr,
                                                  first_order=first_order)
        loss = self.inner_loss(episodes)
        # Get the new parameters after a one-step gradient update
        params = self.policy.update_params(loss, step_size=self.fast_lr,
                                           first_order=first_order)

        return params, critic_params

    def sample(self, tasks, first_order=False):
        """Sample trajectories (before and after the update of the parameters)
        for all the tasks `tasks`.
        """
        episodes = []
        for task in tasks:
            self.sampler.reset_task(task)
            train_episodes = self.sampler.sample(self.policy,
                                                 gamma=self.gamma, device=self.device)

            params = self.adapt(train_episodes, first_order=first_order)[0]

            valid_episodes = self.sampler.sample(self.policy, params=params,
                                                 gamma=self.gamma, device=self.device)
            episodes.append((train_episodes, valid_episodes))
        return episodes

    def kl_divergence(self, episodes, old_pis=None):
        kls = []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            params = self.adapt(train_episodes)[0]
            pi = self.policy(valid_episodes.observations, params=params)

            if old_pi is None:
                old_pi = detach_distribution(pi)

            mask = valid_episodes.mask
            if valid_episodes.actions.dim() > 2:
                mask = mask.unsqueeze(2)
            kl = weighted_mean(kl_divergence(pi, old_pi), dim=0, weights=mask)
            kls.append(kl)

        return torch.mean(torch.stack(kls, dim=0))

    def hessian_vector_product(self, episodes, damping=1e-2):
        """Hessian-vector product, based on the Perlmutter method."""

        def _product(vector):
            kl = self.kl_divergence(episodes)
            grads = torch.autograd.grad(kl, self.policy.parameters(),
                                        create_graph=True)
            flat_grad_kl = parameters_to_vector(grads)

            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v, self.policy.parameters())
            flat_grad2_kl = parameters_to_vector(grad2s)

            return flat_grad2_kl + damping * vector

        return _product

    def surrogate_loss(self, episodes, old_pis=None):
        losses, kls, action_dists = [], [], []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_pis):
            policy_params, critic_params = self.adapt(train_episodes)
            with torch.set_grad_enabled(old_pi is None):
                action_dist = self.policy(valid_episodes.observations, params=policy_params)
                action_dists.append(detach_distribution(action_dist))

                if old_pi is None:
                    old_pi = detach_distribution(action_dist)

                values = self.critic(valid_episodes.observations, params=critic_params)
                advantages = valid_episodes.gae(values, tau=self.tau)
                advantages = weighted_normalize(advantages,
                                                weights=valid_episodes.mask)

                log_ratio = (action_dist.log_prob(valid_episodes.actions)
                             - old_pi.log_prob(valid_episodes.actions))
                if log_ratio.dim() > 2:
                    log_ratio = torch.sum(log_ratio, dim=2)
                ratio = torch.exp(log_ratio)

                loss = -weighted_mean(ratio * advantages, dim=0,
                                      weights=valid_episodes.mask)
                losses.append(loss)

                mask = valid_episodes.mask
                if valid_episodes.actions.dim() > 2:
                    mask = mask.unsqueeze(2)
                kl = weighted_mean(kl_divergence(action_dist, old_pi), dim=0,
                                   weights=mask)
                kls.append(kl)

        return (torch.mean(torch.stack(losses, dim=0)),
                torch.mean(torch.stack(kls, dim=0)), action_dists)

    def critic_loss(self, episodes, old_values=None):
        losses, values = [], []
        if old_values is None:
            old_values = [None] * len(episodes)

        for (train_episodes, valid_episodes), old_pi in zip(episodes, old_values):
            critic_params = self.adapt(train_episodes)[1]
            with torch.set_grad_enabled(old_pi is None):
                values = self.critic(valid_episodes.observations, params=critic_params)
                advantages = valid_episodes.gae(values, tau=self.tau)
                advantages = weighted_normalize(advantages,
                                                weights=valid_episodes.mask)
                losses.append(advantages.pow(2).mean())

        return torch.mean(torch.stack(losses, dim=0)), values

    def step(self, episodes, max_kl=1e-3, cg_iters=10, cg_damping=1e-2,
             ls_max_steps=10, ls_backtrack_ratio=0.5):
        """Meta-optimization step (ie. update of the initial parameters), based
        on Trust Region Policy Optimization (TRPO, [4]).
        """
        old_loss, _, old_pis = self.surrogate_loss(episodes)
        grads = torch.autograd.grad(old_loss, self.policy.parameters())
        grads = parameters_to_vector(grads)

        # Compute the step direction with Conjugate Gradient
        hessian_vector_product = self.hessian_vector_product(episodes,
                                                             damping=cg_damping)
        stepdir = conjugate_gradient(hessian_vector_product, grads,
                                     cg_iters=cg_iters)

        # Compute the Lagrange multiplier
        shs = 0.5 * torch.dot(stepdir, hessian_vector_product(stepdir))
        lagrange_multiplier = torch.sqrt(shs / max_kl)

        step = stepdir / lagrange_multiplier

        # Save the old parameters
        old_params = parameters_to_vector(self.policy.parameters())

        # Line search
        step_size = 1.0
        for _ in range(ls_max_steps):
            vector_to_parameters(old_params - step_size * step,
                                 self.policy.parameters())
            loss, kl, _ = self.surrogate_loss(episodes, old_pis=old_pis)
            improve = loss - old_loss
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                break
            step_size *= ls_backtrack_ratio
        else:
            vector_to_parameters(old_params, self.policy.parameters())

        old_critic_loss, old_values = self.critic_loss(episodes)
        grads = torch.autograd.grad(old_critic_loss, self.critic.parameters())
        grads = parameters_to_vector(grads)
        old_critic_params = parameters_to_vector(self.critic.parameters())
        vector_to_parameters(old_critic_params - (0.001 * grads),
                             self.critic.parameters())

    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.critic.to(device, **kwargs)
        self.device = device
