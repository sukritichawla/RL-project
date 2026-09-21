import numpy as np

from environment.water_env import WaterEnvironment
from environment.actions import Action


class REINFORCEWithBaseline:

    def __init__(
        self,
        state_size=3,
        action_size=2,
        alpha=0.01,
        beta=0.05,
        gamma=0.99,
        episodes=500,
        max_steps=200
    ):
        self.state_size = state_size
        self.action_size = action_size

        # Policy learning rate
        self.alpha = alpha

        # Baseline learning rate
        self.beta = beta

        self.gamma = gamma
        self.episodes = episodes
        self.max_steps = max_steps

        # Policy parameters / preferences
        self.theta = np.zeros(
            (state_size, action_size)
        )

        # State-value baseline V(s)
        self.value = np.zeros(state_size)

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

    def softmax(self, preferences):

        preferences = preferences - np.max(preferences)

        exp_values = np.exp(preferences)

        return exp_values / np.sum(exp_values)

    def get_action_probabilities(self, state):

        return self.softmax(
            self.theta[state]
        )

    def select_action(self, state):

        probabilities = self.get_action_probabilities(state)

        return np.random.choice(
            self.action_size,
            p=probabilities
        )

    def calculate_returns(self, rewards):

        returns = np.zeros(len(rewards))

        G = 0

        for t in reversed(range(len(rewards))):

            G = rewards[t] + self.gamma * G

            returns[t] = G

        return returns

    def update(self, states, actions, returns):

        for state, action, G in zip(
            states,
            actions,
            returns
        ):

            # Baseline value
            baseline = self.value[state]

            # Advantage = Monte Carlo return - baseline
            advantage = G - baseline

            # Update baseline V(s)
            self.value[state] += (
                self.beta * advantage
            )

            # Policy probabilities
            probabilities = self.get_action_probabilities(
                state
            )

            # Gradient of log policy
            gradient = -probabilities

            gradient[action] += 1

            # REINFORCE policy update
            self.theta[state] += (
                self.alpha *
                advantage *
                gradient
            )

    def train(self):

        episode_rewards = []

        for episode in range(self.episodes):

            env = WaterEnvironment()

            state_array = env.reset()

            state = self.discretize_state(
                state_array
            )

            states = []
            actions = []
            rewards = []

            total_reward = 0

            for _ in range(self.max_steps):

                action_index = self.select_action(
                    state
                )

                action = self.action_to_environment_action(
                    action_index
                )

                next_state_array, reward, done, _ = env.step(
                    action
                )

                next_state = self.discretize_state(
                    next_state_array
                )

                states.append(state)
                actions.append(action_index)
                rewards.append(reward)

                total_reward += reward

                state = next_state

                if done:
                    break

            # Monte Carlo returns
            returns = self.calculate_returns(
                rewards
            )

            # Policy + baseline update
            self.update(
                states,
                actions,
                returns
            )

            episode_rewards.append(
                total_reward
            )

        return episode_rewards

    def test(self, steps=10):

        env = WaterEnvironment()

        state_array = env.reset()

        state = self.discretize_state(
            state_array
        )

        print("\nTesting Learned Policy")
        print("-----------------------------")

        for step in range(steps):

            probabilities = self.get_action_probabilities(
                state
            )

            action_index = np.argmax(
                probabilities
            )

            action = self.action_to_environment_action(
                action_index
            )

            next_state_array, reward, done, _ = env.step(
                action
            )

            next_state = self.discretize_state(
                next_state_array
            )

            print(
                f"Step {step + 1}: "
                f"State={state}, "
                f"Action={action_index}, "
                f"Reward={reward}"
            )

            state = next_state

            if done:
                break