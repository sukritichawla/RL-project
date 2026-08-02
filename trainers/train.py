from environment.water_env import WaterEnvironment


class Trainer:

    def __init__(self, agent, episodes=500):

        self.agent = agent
        self.episodes = episodes

        self.env = WaterEnvironment()

        self.reward_history = []

    def train(self):

        print(f"Training {self.agent.name}")

        for episode in range(self.episodes):

            state = self.env.reset()

            total_reward = 0

            done = False

            while not done:

                action = self.agent.select_action(state)

                next_state, reward, done, _ = self.env.step(action)

                self.agent.update(
                    state,
                    action,
                    reward,
                    next_state,
                    done
                )

                state = next_state

                total_reward += reward

            self.reward_history.append(total_reward)

            if episode % 50 == 0:
                print(
                    f"Episode {episode} | Reward = {total_reward}"
                )

        return self.reward_history