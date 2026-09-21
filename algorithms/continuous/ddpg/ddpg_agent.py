import random

import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    ALPHA_ACTOR,
    ALPHA_CRITIC,
    GAMMA,
    TAU,
    BUFFER_SIZE,
    BATCH_SIZE,
    EXPLORATION_NOISE
)

from .networks import Actor, Critic
from .replay_buffer import ReplayBuffer


class DDPGAgent:

    def __init__(
        self,
        state_size=6,
        action_size=1
    ):

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = GAMMA
        self.tau = TAU

        self.actor = Actor(
            state_size,
            action_size
        )

        self.actor_target = Actor(
            state_size,
            action_size
        )

        self.critic = Critic(
            state_size,
            action_size
        )

        self.critic_target = Critic(
            state_size,
            action_size
        )

        # Initialize target networks
        self.actor_target.load_state_dict(
            self.actor.state_dict()
        )

        self.critic_target.load_state_dict(
            self.critic.state_dict()
        )

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=ALPHA_ACTOR
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=ALPHA_CRITIC
        )

        self.replay_buffer = ReplayBuffer(
            BUFFER_SIZE
        )

    def select_action(
        self,
        state,
        add_noise=True
    ):

        state_tensor = torch.FloatTensor(
            state
        ).unsqueeze(0)

        with torch.no_grad():

            action = self.actor(
                state_tensor
            ).cpu().numpy()[0]

        if add_noise:

            action += np.random.normal(
                0,
                EXPLORATION_NOISE,
                size=self.action_size
            )

        action = np.clip(
            action,
            -1.0,
            1.0
        )

        return action

    def update(self):

        if len(self.replay_buffer) < BATCH_SIZE:
            return

        batch = self.replay_buffer.sample(
            BATCH_SIZE
        )

        states = torch.FloatTensor(
            np.array([x[0] for x in batch])
        )

        actions = torch.FloatTensor(
            np.array([x[1] for x in batch])
        )

        rewards = torch.FloatTensor(
            np.array([x[2] for x in batch])
        ).unsqueeze(1)

        next_states = torch.FloatTensor(
            np.array([x[3] for x in batch])
        )

        dones = torch.FloatTensor(
            np.array([x[4] for x in batch])
        ).unsqueeze(1)

        # -------------------------
        # Critic update
        # -------------------------

        with torch.no_grad():

            next_actions = self.actor_target(
                next_states
            )

            target_q = self.critic_target(
                next_states,
                next_actions
            )

            target_q = rewards + (
                self.gamma *
                (1 - dones) *
                target_q
            )

        current_q = self.critic(
            states,
            actions
        )

        critic_loss = F.mse_loss(
            current_q,
            target_q
        )

        self.critic_optimizer.zero_grad()

        critic_loss.backward()

        self.critic_optimizer.step()

        # -------------------------
        # Actor update
        # -------------------------

        predicted_actions = self.actor(
            states
        )

        actor_loss = -self.critic(
            states,
            predicted_actions
        ).mean()

        self.actor_optimizer.zero_grad()

        actor_loss.backward()

        self.actor_optimizer.step()

        # -------------------------
        # Soft target updates
        # -------------------------

        self.soft_update(
            self.actor,
            self.actor_target
        )

        self.soft_update(
            self.critic,
            self.critic_target
        )

    def soft_update(
        self,
        source,
        target
    ):

        for target_param, source_param in zip(
            target.parameters(),
            source.parameters()
        ):

            target_param.data.copy_(
                self.tau * source_param.data
                +
                (1 - self.tau) *
                target_param.data
            )

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done
        )