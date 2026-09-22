import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from algorithms.deep.actor_critic.networks import (
    ActorNetwork,
    CriticNetwork
)


class A3CAgent:

    def __init__(
        self,
        state_size=6,
        action_size=6,
        gamma=0.99,
        actor_lr=0.0003,
        critic_lr=0.001
    ):

        self.name = "A3C"

        self.gamma = gamma

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

    def sample_action(self, state):

        state = torch.FloatTensor(
            state
        ).unsqueeze(0)

        logits = self.actor(state)

        distribution = Categorical(
            logits=logits
        )

        action = distribution.sample()

        return action.item()

    def select_action(self, state):

        state = torch.FloatTensor(
            state
        ).unsqueeze(0)

        with torch.no_grad():

            logits = self.actor(state)

            action = torch.argmax(
                logits,
                dim=-1
            )

        return action.item()

    def update(
        self,
        states,
        actions,
        rewards
    ):

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

        states = torch.FloatTensor(
            states
        )

        actions = torch.LongTensor(
            actions
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

        logits = self.actor(states)

        distribution = Categorical(
            logits=logits
        )

        log_probs = distribution.log_prob(
            actions
        )

        actor_loss = -(
            log_probs * advantages
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