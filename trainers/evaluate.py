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

                # Works with MC/TD agents using select_action()
                # and DQN agents using choose_action()
                if hasattr(self.agent, "select_action"):
                    action = self.agent.select_action(state)
                elif hasattr(self.agent, "choose_action"):
                    action = self.agent.choose_action(state)
                else:
                    raise AttributeError(
                        "Agent must have select_action() or choose_action()"
                    )

                next_state, reward, done, _ = self.env.step(action)

                total += reward
                state = next_state

            rewards.append(total)

        average = sum(rewards) / len(rewards)

        print(f"Average Reward : {average:.2f}")

        return average