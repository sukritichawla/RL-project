import torch

from environment.water_env import WaterEnvironment
from algorithms.deep.double_dqn import DoubleDQNAgent


def train():

    env = WaterEnvironment()

    agent = DoubleDQNAgent(
        state_size=6,
        action_size=6,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        batch_size=64,
        target_update_frequency=100
    )

    episodes = 1000

    rewards_history = []
    losses_history = []

    for episode in range(episodes):

        state = env.reset()

        total_reward = 0
        episode_losses = []

        done = False

        while not done:

            # Choose action
            action = agent.choose_action(state)

            # Environment transition
            next_state, reward, done, _ = env.step(action)

            # Store experience
            agent.remember(
                state,
                action,
                reward,
                next_state,
                done
            )

            # Learn
            loss = agent.train_step()

            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            total_reward += reward

        # Epsilon decay
        agent.decay_epsilon()

        rewards_history.append(total_reward)

        if episode_losses:
            losses_history.append(
                sum(episode_losses) / len(episode_losses)
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

    # SAVE MODEL
    torch.save(
        agent.online_network.state_dict(),
        "double_dqn_model.pth"
    )

    print("\nDouble DQN model saved!")

    return agent, rewards_history, losses_history


if __name__ == "__main__":
    train()