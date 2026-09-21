import os
import csv
import numpy as np

from environment.bandit_env import WaterBanditEnvironment

from algorithms.classical.bandits.epsilon_greedy import EpsilonGreedy
from algorithms.classical.bandits.ucb import UCB
from algorithms.classical.bandits.thomson_sampling import ThompsonSampling


class BanditTrainer:

    def __init__(self, agent, steps=10000):

        self.agent = agent
        self.steps = steps

        self.env = WaterBanditEnvironment()

        self.reward_history = []
        self.cumulative_reward_history = []
        self.action_history = []

        self.tank_history = []
        self.pressure_history = []
        self.demand_history = []

    def train(self):

        print()
        print("=" * 70)
        print(f"Training {self.agent.name}")
        print("=" * 70)

        self.env.reset()

        cumulative_reward = 0

        for step in range(1, self.steps + 1):

            # Select pump-control action
            action = self.agent.select_action()

            # Interact with water environment
            _, reward, _, _ = self.env.step(action)

            # Update bandit agent
            self.agent.update(action, reward)

            # Store results
            self.action_history.append(action)
            self.reward_history.append(reward)

            cumulative_reward += reward

            self.cumulative_reward_history.append(
                cumulative_reward
            )

            # Store water-system measurements
            state = self.env.get_last_state()

            self.tank_history.append(
                state["tank_level"]
            )

            self.pressure_history.append(
                state["pressure"]
            )

            self.demand_history.append(
                state["demand"]
            )

            # Display progress
            if step % 1000 == 0:

                average_reward = np.mean(
                    self.reward_history[-1000:]
                )

                print(
                    f"Step {step:5d} | "
                    f"Avg Reward = {average_reward:8.2f} | "
                    f"Cumulative Reward = {cumulative_reward:10.2f}"
                )

        self.print_final_results()

        return {
            "rewards": self.reward_history,
            "cumulative_rewards": self.cumulative_reward_history,
            "actions": self.action_history,
            "tank": self.tank_history,
            "pressure": self.pressure_history,
            "demand": self.demand_history
        }

    def print_final_results(self):

        print()
        print("-" * 70)
        print(f"Results: {self.agent.name}")
        print("-" * 70)

        print(
            f"Total Reward   : "
            f"{sum(self.reward_history):.2f}"
        )

        print(
            f"Average Reward : "
            f"{np.mean(self.reward_history):.2f}"
        )

        print()
        print("Pump-Control Selection Counts:")

        for arm in range(self.env.n_arms):

            count = self.agent.action_counts[arm]

            print(
                f"Arm {arm} "
                f"({self.env.get_arm_name(arm):20s}) : "
                f"{count}"
            )

    def save_results(self, results):

        os.makedirs(
            "results/bandits",
            exist_ok=True
        )

        filename = (
            self.agent.name
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
            + ".csv"
        )

        path = os.path.join(
            "results",
            "bandits",
            filename
        )

        with open(path, "w", newline="") as file:

            writer = csv.writer(file)

            writer.writerow([
                "Step",
                "Reward",
                "Cumulative Reward",
                "Action",
                "Pump Control",
                "Tank Level",
                "Pressure",
                "Demand"
            ])

            for i in range(self.steps):

                action = results["actions"][i]

                writer.writerow([
                    i + 1,
                    results["rewards"][i],
                    results["cumulative_rewards"][i],
                    action,
                    self.env.get_arm_name(action),
                    results["tank"][i],
                    results["pressure"][i],
                    results["demand"][i]
                ])

        print()
        print(f"Results saved to: {path}")


def run_bandit(agent):

    trainer = BanditTrainer(
        agent,
        steps=10000
    )

    results = trainer.train()

    trainer.save_results(results)

    return results


if __name__ == "__main__":

    print()
    print("=" * 70)
    print("URBAN WATER DISTRIBUTION - MULTI-ARMED BANDIT EXPERIMENT")
    print("=" * 70)

    epsilon_agent = EpsilonGreedy(
        n_arms=6
    )

    ucb_agent = UCB(
        n_arms=6
    )

    thompson_agent = ThompsonSampling(
        n_arms=6
    )

    results = {}

    results["Epsilon-Greedy"] = run_bandit(
        epsilon_agent
    )

    results["UCB"] = run_bandit(
        ucb_agent
    )

    results["Thompson Sampling"] = run_bandit(
        thompson_agent
    )

    print()
    print("=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    for name, result in results.items():

        print(
            f"{name:20s} | "
            f"Average Reward = "
            f"{np.mean(result['rewards']):.2f} | "
            f"Total Reward = "
            f"{sum(result['rewards']):.2f}"
        )