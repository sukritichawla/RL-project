import numpy as np


class ThompsonSampling:

    def __init__(self, n_arms):

        self.name = "Thompson Sampling"

        self.n_arms = n_arms

        # Beta distribution parameters
        self.alpha = np.ones(n_arms)
        self.beta = np.ones(n_arms)

        self.action_counts = np.zeros(
            n_arms,
            dtype=int
        )

    def select_action(self):

        samples = np.random.beta(
            self.alpha,
            self.beta
        )

        return int(np.argmax(samples))

    def update(self, action, reward):

        self.action_counts[action] += 1

        if reward == 1:

            self.alpha[action] += 1

        else:

            self.beta[action] += 1