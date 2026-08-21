import numpy as np

from algorithms.base_agent import BaseAgent
from .config import ALPHA, GAMMA, EPSILON


class SARSAAgent(BaseAgent):

    def __init__(
        self,
        state_size=3,
        action_size=2,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON
    ):
        super().__init__(
            state_size=state_size,
            action_size=action_size
        )

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table = np.zeros(
            (state_size, action_size)
        )

        self.name = "SARSA"

    def select_action(self, state):

        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)

        return np.argmax(self.q_table[state])

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        next_action=None
    ):

        current_q = self.q_table[state, action]

        if done:
            target = reward
        else:
            target = reward + (
                self.gamma *
                self.q_table[next_state, next_action]
            )

        self.q_table[state, action] += self.alpha * (
            target - current_q
        )

    def train(self, env, episodes, max_steps):

        rewards = []

        for episode in range(episodes):

            state_array = env.reset()

            state = env.discretize_state(state_array)

            action = self.select_action(state)

            total_reward = 0

            for _ in range(max_steps):

                next_state_array, reward, done, _ = env.step(action)

                next_state = env.discretize_state(
                    next_state_array
                )

                if done:
                    next_action = None
                else:
                    next_action = self.select_action(
                        next_state
                    )

                self.update(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    next_action
                )

                total_reward += reward

                if done:
                    break

                state = next_state
                action = next_action

            rewards.append(total_reward)

        return rewards

    def save(self, filepath):
        np.save(filepath, self.q_table)

    def load(self, filepath):
        self.q_table = np.load(filepath)