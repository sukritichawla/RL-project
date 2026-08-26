import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from algorithms.deep.dqn import QNetwork
from algorithms.deep.replay_buffer import ReplayBuffer


class DoubleDQNAgent:

    def __init__(
        self,
        state_size=6,
        action_size=6,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        buffer_size=100000,
        batch_size=64,
        target_update_frequency=100
    ):

        self.state_size = state_size
        self.action_size = action_size

        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Online network
        self.online_network = QNetwork(
            state_size,
            action_size
        ).to(self.device)

        # Target network
        self.target_network = QNetwork(
            state_size,
            action_size
        ).to(self.device)

        # Initially both networks are identical
        self.target_network.load_state_dict(
            self.online_network.state_dict()
        )

        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.online_network.parameters(),
            lr=learning_rate
        )

        self.loss_function = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(
            buffer_size
        )

        self.update_count = 0

    def choose_action(self, state):

        # Exploration
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        # Exploitation
        state = torch.FloatTensor(state)
        state = state.unsqueeze(0).to(self.device)

        with torch.no_grad():

            q_values = self.online_network(state)

        return q_values.argmax(dim=1).item()

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


    def train_step(self):

        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(
            np.array(states)
        ).to(self.device)

        actions = torch.LongTensor(
            actions
        ).unsqueeze(1).to(self.device)

        rewards = torch.FloatTensor(
            rewards
        ).unsqueeze(1).to(self.device)

        next_states = torch.FloatTensor(
            np.array(next_states)
        ).to(self.device)

        dones = torch.FloatTensor(
            dones
        ).unsqueeze(1).to(self.device)

        # --------------------------------
        # Q(s,a)
        # --------------------------------

        current_q = self.online_network(states).gather(
            1,
            actions
        )

        # --------------------------------
        # Double DQN
        # --------------------------------

        with torch.no_grad():

            # ONLINE NETWORK selects action
            next_actions = self.online_network(
                next_states
            ).argmax(
                dim=1,
                keepdim=True
            )

            # TARGET NETWORK evaluates action
            next_q = self.target_network(
                next_states
            ).gather(
                1,
                next_actions
            )

            target_q = rewards + (
                self.gamma *
                next_q *
                (1 - dones)
            )

        # --------------------------------
        # Loss
        # --------------------------------

        loss = self.loss_function(
            current_q,
            target_q
        )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        self.update_count += 1

        # Update target network
        if self.update_count % self.target_update_frequency == 0:

            self.target_network.load_state_dict(
                self.online_network.state_dict()
            )

        return loss.item()


    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )