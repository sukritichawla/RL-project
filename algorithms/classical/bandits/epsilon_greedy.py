import numpy as np


class EpsilonGreedy:

    def __init__(
        self,
        n_arms,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    ):

        self.name = "Epsilon-Greedy"

        self.n_arms = n_arms

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Estimated value of each arm
        self.q_values = np.zeros(n_arms)

        # Number of times each arm was selected
        self.action_counts = np.zeros(n_arms, dtype=int)

    def select_action(self):

        # Exploration
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)

        # Exploitation
        return int(np.argmax(self.q_values))

    def update(self, action, reward):

        self.action_counts[action] += 1

        count = self.action_counts[action]

        # Incremental average
        self.q_values[action] += (
            reward - self.q_values[action]
        ) / count

        # Decay epsilon
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )