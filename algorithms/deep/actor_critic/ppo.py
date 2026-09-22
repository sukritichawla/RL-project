import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from algorithms.deep.actor_critic.networks import (
    ActorNetwork,
    CriticNetwork
)


class PPOAgent:

    def __init__(
        self,
        state_size=6,
        action_size=6,
        gamma=0.99,
        clip_epsilon=0.2,
        actor_lr=0.0003,
        critic_lr=0.001
    ):

        self.name = "PPO"

        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

        self.actor = ActorNetwork(
            state_size,
            action_size
        )

        self.critic = CriticNetwork(
            state_size
        )

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr
        )

        self.actor_loss = 0
        self.critic_loss = 0

    def get_distribution(self, states):

        logits = self.actor(states)

        return Categorical(
            logits=logits
        )

    def sample_action(self, state):

        state_tensor = torch.FloatTensor(
            state
        ).unsqueeze(0)

        distribution = self.get_distribution(
            state_tensor
        )

        action = distribution.sample()

        log_probability = distribution.log_prob(
            action
        )

        return (
            action.item(),
            log_probability.item()
        )

    def select_action(self, state):

        state_tensor = torch.FloatTensor(
            state
        ).unsqueeze(0)

        with torch.no_grad():

            distribution = self.get_distribution(
                state_tensor
            )

            action = torch.argmax(
                distribution.probs,
                dim=-1
            )

        return action.item()

    def update(
        self,
        states,
        actions,
        rewards,
        old_log_probs
    ):

        states = torch.FloatTensor(
            np.array(states)
        )

        actions = torch.LongTensor(
            actions
        )

        old_log_probs = torch.FloatTensor(
            old_log_probs
        )

        returns = []

        discounted = 0

        for reward in reversed(rewards):

            discounted = (
                reward +
                self.gamma * discounted
            )

            returns.insert(
                0,
                discounted
            )

        returns = torch.FloatTensor(
            returns
        )

        values = self.critic(
            states
        ).squeeze()

        advantages = (
            returns - values.detach()
        )

        distribution = self.get_distribution(
            states
        )

        new_log_probs = distribution.log_prob(
            actions
        )

        ratio = torch.exp(
            new_log_probs - old_log_probs
        )

        clipped_ratio = torch.clamp(
            ratio,
            1 - self.clip_epsilon,
            1 + self.clip_epsilon
        )

        actor_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()

        critic_loss = F.mse_loss(
            values,
            returns
        )

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        self.critic_optimizer.step()

        self.actor_loss = actor_loss.item()
        self.critic_loss = critic_loss.item()