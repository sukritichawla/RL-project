from environment.water_env import WaterEnvironment


class Evaluator:

    def __init__(self, agent):

        self.agent = agent

        self.env = WaterEnvironment()

    def evaluate(self, episodes=20):

        rewards = []

        for _ in range(episodes):

            state = self.env.reset()

            done = False

            total = 0

            while not done:

                action = self.agent.select_action(state)

                next_state, reward, done, _ = self.env.step(action)

                total += reward

                state = next_state

            rewards.append(total)

        average = sum(rewards) / len(rewards)

        print(f"Average Reward : {average}")

        return average