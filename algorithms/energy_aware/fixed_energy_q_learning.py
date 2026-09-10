import random
import numpy as np

from environment.water_env import WaterEnvironment
from environment.actions import Action
from rewards.fixed_energy_reward import calculate_fixed_energy_reward

from .config import (
    ALPHA,
    GAMMA,
    EPSILON,
    EPISODES,
    MAX_STEPS
)


class FixedEnergyQLearningAgent:

    def __init__(
        self,
        state_size=3,
        action_size=2,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON
    ):
        self.state_size = state_size
        self.action_size = action_size

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table = np.zeros(
            (state_size, action_size)
        )

    def select_action(self, state):

        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        target = reward

        if not done:
            target += self.gamma * np.max(
                self.q_table[next_state]
            )

        self.q_table[state, action] += self.alpha * (
            target - self.q_table[state, action]
        )

    def discretize_state(self, state):

        tank_level = state[0]

        if tank_level < 35:
            return 0

        elif tank_level < 70:
            return 1

        return 2

    def action_to_environment_action(self, action):

        if action == 0:
            return Action.PUMP_ON

        return Action.PUMP_OFF

    def train(self):

        episode_rewards = []

        for episode in range(EPISODES):

            env = WaterEnvironment()

            state_array = env.reset()
            state = self.discretize_state(state_array)

            total_reward = 0

            for _ in range(MAX_STEPS):

                # Select PUMP_ON or PUMP_OFF
                action_index = self.select_action(state)

                action = self.action_to_environment_action(
                    action_index
                )

                # Apply action
                next_state_array, _, done, _ = env.step(action)

                # Convert next state to discrete state
                next_state = self.discretize_state(
                    next_state_array
                )

                # Calculate fixed energy-aware reward
                reward, _ = calculate_fixed_energy_reward(
                    env.state
                )

                # Update Q-table
                self.update(
                    state,
                    action_index,
                    reward,
                    next_state,
                    done
                )

                total_reward += reward

                state = next_state

                if done:
                    break

            episode_rewards.append(total_reward)

        return episode_rewards