from environment.water_env import WaterEnvironment
from algorithms.deep.dqn_agent import DQNAgent
import torch
from environment.water_env import WaterEnvironment

def train():

    env = WaterEnvironment()

    agent = DQNAgent(
        state_size=6,
        action_size=6,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        buffer_size=100000,
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

            # 1. SELECT ACTION
            action = agent.choose_action(state)

            # 2. ENVIRONMENT STEP
            next_state, reward, done, _ = env.step(action)

            # 3. STORE EXPERIENCE
            agent.remember(
                state,
                action,
                reward,
                next_state,
                done
            )

            # 4. TRAIN DQN
            loss = agent.train_step()

            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            total_reward += reward

        # 5. EPSILON DECAY
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

    # SAVE TRAINED MODEL
    torch.save(
        agent.online_network.state_dict(),
        "dqn_model.pth"
    )

    return agent, rewards_history, losses_history


if __name__ == "__main__":
    train()