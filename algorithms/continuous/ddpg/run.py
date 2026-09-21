import numpy as np

from .ddpg_agent import DDPGAgent
from .water_continuous_env import (
    ContinuousWaterEnvironment
)

from .config import (
    EPISODES,
    MAX_STEPS
)


def main():

    print("=" * 60)
    print("DDPG - CONTINUOUS WATER PUMP CONTROL")
    print("=" * 60)

    env = ContinuousWaterEnvironment()

    agent = DDPGAgent(
        state_size=6,
        action_size=1
    )

    episode_rewards = []

    for episode in range(EPISODES):

        state = env.reset()

        total_reward = 0

        for step in range(MAX_STEPS):

            # Actor produces action in [-1, 1]
            action = agent.select_action(
                state,
                add_noise=True
            )

            # Convert [-1, 1] to [0, 1]
            pump_speed = (
                action[0] + 1
            ) / 2.0

            next_state, reward, done, info = (
                env.step(pump_speed)
            )

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done
            )

            agent.update()

            state = next_state

            total_reward += reward

            if done:
                break

        episode_rewards.append(
            total_reward
        )

        if (
            episode + 1
        ) % 10 == 0:

            recent_rewards = (
                episode_rewards[-10:]
            )

            print(
                f"Episode {episode + 1:3d} | "
                f"Average Reward: "
                f"{np.mean(recent_rewards):8.2f}"
            )

    print("\nTraining Results:")
    print("-----------------------------")

    print(
        "Episodes      :",
        len(episode_rewards)
    )

    print(
        "Average Reward:",
        f"{np.mean(episode_rewards):.2f}"
    )

    print(
        "Best Reward   :",
        f"{np.max(episode_rewards):.2f}"
    )

    print("\nTesting Learned Policy")
    print("-----------------------------")

    state = env.reset()

    test_reward = 0

    for step in range(10):

        action = agent.select_action(
            state,
            add_noise=False
        )

        pump_speed = (
            action[0] + 1
        ) / 2.0

        next_state, reward, done, info = (
            env.step(pump_speed)
        )

        print(
            f"Step {step + 1}: "
            f"Pump Speed={pump_speed:.3f}, "
            f"Reward={reward:.2f}"
        )

        test_reward += reward

        state = next_state

        if done:
            break

    print(
        "\nTesting Cumulative Reward:",
        f"{test_reward:.2f}"
    )


if __name__ == "__main__":
    main()