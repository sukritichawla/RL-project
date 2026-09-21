from algorithms.policy_gradient.reinforce_baseline import (
    REINFORCEWithBaseline
)


def main():

    print("=" * 55)
    print("REINFORCE WITH BASELINE")
    print("=" * 55)

    agent = REINFORCEWithBaseline(
        state_size=3,
        action_size=2,
        alpha=0.01,
        beta=0.05,
        gamma=0.99,
        episodes=500,
        max_steps=200
    )

    rewards = agent.train()

    print("\nTraining Results:")
    print("-----------------------------")
    print("Episodes      :", len(rewards))
    print(
        "Average Reward:",
        f"{sum(rewards) / len(rewards):.2f}"
    )
    print(
        "Best Reward   :",
        f"{max(rewards):.2f}"
    )

    print("\nPolicy Parameters:")
    print(agent.theta)

    print("\nState-Value Baseline V(s):")
    print(agent.value)

    agent.test(steps=10)


if __name__ == "__main__":
    main()