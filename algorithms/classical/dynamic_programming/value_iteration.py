import numpy as np

from algorithms.base_agent import BaseAgent
from .config import GAMMA, THETA, MAX_ITERATIONS
from .utils import initialize_values, initialize_policy


class ValueIterationAgent(BaseAgent):

    def __init__(self, env):

        self.env = env

        self.model = env.get_model()

        self.num_states = self.model["num_states"]

        self.num_actions = self.model["num_actions"]

        super().__init__(
            state_size=self.num_states,
            action_size=self.num_actions
        )

        self.values = initialize_values(
            self.num_states
        )

        self.policy = initialize_policy(
            self.num_states,
            self.num_actions
        )

        self.name = "Value Iteration"

    def select_action(self, state):

        state = self.env.discretize_state(state)

        return self.policy[state]

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):
        pass

    def value_iteration(self):

        while True:

            delta = 0

            for s in range(self.num_states):

                old_value = self.values[s]

                action_values = np.zeros(self.num_actions)

                for a in range(self.num_actions):

                    for p, ns, r, done in self.model["transitions"][s][a]:

                        action_values[a] += p * (
                            r +
                            GAMMA *
                            self.values[ns] *
                            (not done)
                        )

                self.values[s] = np.max(action_values)

                delta = max(
                    delta,
                    abs(old_value - self.values[s])
                )

            if delta < THETA:

                break

    def extract_policy(self):

        for s in range(self.num_states):

            action_values = np.zeros(self.num_actions)

            for a in range(self.num_actions):

                for p, ns, r, done in self.model["transitions"][s][a]:

                    action_values[a] += p * (
                        r +
                        GAMMA *
                        self.values[ns] *
                        (not done)
                    )

            self.policy[s] = np.argmax(action_values)

    def train(self):

        for _ in range(MAX_ITERATIONS):

            previous = self.values.copy()

            self.value_iteration()

            if np.max(np.abs(previous - self.values)) < THETA:

                break

        self.extract_policy()

        return self.policy

    def save(self, filepath):

        np.save(filepath, self.policy)

    def load(self, filepath):

        self.policy = np.load(filepath)