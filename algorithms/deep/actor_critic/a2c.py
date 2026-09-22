import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from algorithms.deep.actor_critic.networks import (
    ActorNetwork,
    CriticNetwork
)


class A2CAgent:

    def __init__(
        self,
        state_size=6,
        action_size=6,
        gamma=0.99,
        actor_lr=0.0003,
        critic_lr=0.001
    ):

        self.name = "A2C"

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

        state_tensor = torch.FloatTensor(
            state
        ).unsqueeze(0)

        logits = self.actor(state_tensor)

        distribution = Categorical(
            logits=logits
        )

        action = distribution.sample()

        return action.item()

    def select_action(self, state):

        state_tensor = torch.FloatTensor(
            state
        ).unsqueeze(0)

        with torch.no_grad():

            logits = self.actor(state_tensor)

            action = torch.argmax(
                logits,
                dim=-1
            )

        return action.item()

    def update(
        self,
        states,
        actions,
        rewards,
        next_state,
        done
    ):

        states = torch.FloatTensor(
            np.array(states)
        )

        actions = torch.LongTensor(
            actions
        )

        # --------------------------------
        # Calculate discounted returns
        # --------------------------------

        returns = []

        discounted_return = 0.0

        if done:
            next_value = 0.0

        else:

            next_state_tensor = torch.FloatTensor(
                next_state
            ).unsqueeze(0)

            with torch.no_grad():

                next_value = self.critic(
                    next_state_tensor
                ).item()

        discounted_return = next_value

        for reward in reversed(rewards):

            discounted_return = (
                reward +
                self.gamma * discounted_return
            )

            returns.insert(
                0,
                discounted_return
            )

        returns = torch.FloatTensor(
            returns
        )

        # --------------------------------
        # Critic
        # --------------------------------

        values = self.critic(
            states
        ).squeeze(-1)

        advantages = (
            returns - values.detach()
        )

        # --------------------------------
        # Actor
        # --------------------------------

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

        # --------------------------------
        # Critic loss
        # --------------------------------

        critic_loss = F.mse_loss(
            values,
            returns
        )

        # --------------------------------
        # Update Actor
        # --------------------------------

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            1.0
        )

        self.actor_optimizer.step()

        # --------------------------------
        # Update Critic
        # --------------------------------

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            1.0
        )

        self.critic_optimizer.step()

        self.actor_loss = actor_loss.item()

        self.critic_loss = critic_loss.item()