import matplotlib.pyplot as plt
import numpy as np

from environment.water_env import WaterEnvironment

from algorithms.classical.temporal_difference.sarsa import SARSAAgent
from algorithms.classical.temporal_difference.q_learning import QLearningAgent

from algorithms.classical.temporal_difference.config import (
    EPISODES,
    MAX_STEPS
)


def test_agent(agent, env):

    print("\n" + "=" * 50)
    print(f"{agent.name}")
    print("=" * 50)

    rewards = agent.train(
        env,
        EPISODES,
        MAX_STEPS
    )

    print("\nQ-Table:")
    print(agent.q_table)

    print("\nTraining Results:")
    print(f"Episodes       : {EPISODES}")
    print(f"Average Reward : {np.mean(rewards):.2f}")
    print(f"Best Reward    : {np.max(rewards):.2f}")

    # ----------------------------
    # Training Reward Graph
    # ----------------------------

    window = 20

    moving_average = np.convolve(
        rewards,
        np.ones(window) / window,
        mode="valid"
    )

    plt.figure(figsize=(7, 4))
    plt.plot(
        range(window, EPISODES + 1),
        moving_average
    )
    plt.title(f"{agent.name} Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Reward")
    plt.grid(True)
    plt.show()

    # ----------------------------
    # Testing
    # ----------------------------

    print("\nTesting Learned Policy")
    print("-" * 30)

    state_array = env.reset()

    for step in range(10):

        state = env.discretize_state(
            state_array
        )

        action = agent.select_action(state)

        state_array, reward, done, _ = env.step(
            action
        )

        print(
            f"Step {step + 1}: "
            f"State={state}, "
            f"Action={action}, "
            f"Reward={reward}"
        )

        if done:
            break


def main():

    env = WaterEnvironment()

    # ----------------------------
    # SARSA
    # ----------------------------

    sarsa_agent = SARSAAgent()

    test_agent(
        sarsa_agent,
        env
    )

    # ----------------------------
    # Q-Learning
    # ----------------------------

    q_learning_agent = QLearningAgent()

    test_agent(
        q_learning_agent,
        env
    )


if __name__ == "__main__":
    main()