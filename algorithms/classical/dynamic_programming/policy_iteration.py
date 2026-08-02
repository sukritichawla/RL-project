import numpy as np

from algorithms.base_agent import BaseAgent
from .config import GAMMA, THETA, MAX_ITERATIONS
from .utils import initialize_values, initialize_policy


class PolicyIterationAgent(BaseAgent):

    def __init__(self, env):

        self.env = env

        self.model = env.get_model()

        self.num_states = self.model["num_states"]

        self.num_actions = self.model["num_actions"]

        super().__init__(
            state_size=self.num_states,
            action_size=self.num_actions
        )

        self.value_table = initialize_values(self.num_states)

        self.policy = initialize_policy(
            self.num_states,
            self.num_actions
        )

        self.name = "Policy Iteration"

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

    def policy_evaluation(self):

        while True:

            delta = 0

            for s in range(self.num_states):

                v = self.value_table[s]

                a = self.policy[s]

                new_v = 0

                for p, ns, r, done in self.model["transitions"][s][a]:

                    new_v += p * (
                        r +
                        GAMMA *
                        self.value_table[ns] *
                        (not done)
                    )

                self.value_table[s] = new_v

                delta = max(
                    delta,
                    abs(v - new_v)
                )

            if delta < THETA:

                break

    def policy_improvement(self):

        stable = True

        for s in range(self.num_states):

            old_action = self.policy[s]

            action_values = np.zeros(
                self.num_actions
            )

            for a in range(self.num_actions):

                for p, ns, r, done in self.model["transitions"][s][a]:

                    action_values[a] += p * (

                        r +

                        GAMMA *

                        self.value_table[ns] *

                        (not done)

                    )

            self.policy[s] = np.argmax(action_values)

            if old_action != self.policy[s]:

                stable = False

        return stable

    def train(self):

        for _ in range(MAX_ITERATIONS):

            self.policy_evaluation()

            stable = self.policy_improvement()

            if stable:

                break

        return self.policy

    def save(self, filepath):
        np.save(filepath, self.policy)


    def load(self, filepath):
        self.policy = np.load(filepath)