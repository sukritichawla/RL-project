import numpy as np

from environment.water_env import WaterEnvironment
from algorithms.deep.actor_critic.ppo import PPOAgent


def train():

    env = WaterEnvironment()

    agent = PPOAgent(
        state_size=6,
        action_size=6
    )

    episodes = 500

    rewards_history = []

    for episode in range(1, episodes + 1):

        state = env.reset()

        states = []
        actions = []
        rewards = []
        old_log_probs = []

        done = False
        total_reward = 0

        while not done:

            action, log_probability = agent.sample_action(
                state
            )

            next_state, reward, done, _ = env.step(
                action
            )

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            old_log_probs.append(log_probability)

            total_reward += reward

            state = next_state

        agent.update(
            states,
            actions,
            rewards,
            old_log_probs
        )

        rewards_history.append(
            total_reward
        )

        if episode % 10 == 0:

            average = np.mean(
                rewards_history[-10:]
            )

            print(
                f"Episode {episode} | "
                f"Reward {total_reward:.2f} | "
                f"Avg Reward {average:.2f} | "
                f"Actor Loss {agent.actor_loss:.4f} | "
                f"Critic Loss {agent.critic_loss:.4f}"
            )

    return agent, rewards_history


if __name__ == "__main__":

    agent, rewards = train()

    print()
    print("PPO training completed.")

    print(
        f"Final Average Reward: "
        f"{np.mean(rewards[-20:]):.2f}"
    )