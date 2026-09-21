import numpy as np


class UCB:

    def __init__(self, n_arms, c=2.0):

        self.name = "UCB"

        self.n_arms = n_arms
        self.c = c

        self.q_values = np.zeros(n_arms)

        self.action_counts = np.zeros(
            n_arms,
            dtype=int
        )

        self.total_steps = 0

    def select_action(self):

        self.total_steps += 1

        # Try every arm at least once
        for arm in range(self.n_arms):

            if self.action_counts[arm] == 0:
                return arm

        ucb_values = np.zeros(self.n_arms)

        for arm in range(self.n_arms):

            exploration_bonus = self.c * np.sqrt(
                np.log(self.total_steps)
                / self.action_counts[arm]
            )

            ucb_values[arm] = (
                self.q_values[arm]
                + exploration_bonus
            )

        return int(np.argmax(ucb_values))

    def update(self, action, reward):

        self.action_counts[action] += 1

        count = self.action_counts[action]

        self.q_values[action] += (
            reward - self.q_values[action]
        ) / count