from environment.water_env import WaterEnvironment
from algorithms.classical.temporal_difference.td0_agent import TD0Agent

def train():

    env = WaterEnvironment()

    agent = TD0Agent(
        state_size=6,
        action_size=6,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )

    episodes = 1000

    rewards_history = []
    td_errors_history = []

    for episode in range(episodes):

        state = env.reset()

        total_reward = 0
        episode_td_errors = []

        done = False

        while not done:

            # 1. Select action
            action = agent.select_action(state)

            # 2. Environment transition
            next_state, reward, done, _ = env.step(action)

            # 3. TD(0) update
            td_error = agent.update(
                state,
                action,
                reward,
                next_state,
                done
            )

            episode_td_errors.append(abs(td_error))

            # 4. Move to next state
            state = next_state

            total_reward += reward

        # 5. Decay epsilon after episode
        agent.decay_epsilon()

        rewards_history.append(total_reward)

        if episode_td_errors:
            td_errors_history.append(
                sum(episode_td_errors) / len(episode_td_errors)
            )

        if (episode + 1) % 10 == 0:

            avg_reward = sum(
                rewards_history[-10:]
            ) / 10

            print(
                f"Episode: {episode + 1} | "
                f"Reward: {total_reward:.2f} | "
                f"Avg Reward: {avg_reward:.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

    return agent, rewards_history, td_errors_history


if __name__ == "__main__":
    train()