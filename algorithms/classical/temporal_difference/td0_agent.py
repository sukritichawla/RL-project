import random
import numpy as np


class TD0Agent:

    def __init__(
        self,
        state_size=6,
        action_size=6,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    ):

        self.state_size = state_size
        self.action_size = action_size

        self.learning_rate = learning_rate
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table
        self.q_table = {}

    def _state_key(self, state):
        """
        Convert environment state into a hashable key.
        """
        state = np.asarray(state, dtype=np.float32)

        # Round continuous values so the number of states
        # does not become unnecessarily large.
        return tuple(np.round(state, 2))

    def _get_q_values(self, state):

        key = self._state_key(state)

        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.action_size)

        return self.q_table[key]

    def select_action(self, state):

        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        q_values = self._get_q_values(state)

        return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state, done):

        q_values = self._get_q_values(state)

        current_q = q_values[action]

        if done:
            target = reward
        else:
            next_q_values = self._get_q_values(next_state)

            target = reward + self.gamma * np.max(next_q_values)

        # TD(0) update
        td_error = target - current_q

        q_values[action] += self.learning_rate * td_error

        return td_error

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )