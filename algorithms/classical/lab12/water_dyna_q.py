
# algorithms/classical/lab12/water_dyna_q.py

import csv
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from environment.simulator import WaterSimulator
from environment.actions import Action
from algorithms.classical.lab12.water_state_utils import (
    HORIZON,
    NUM_STATES,
    choose_action,
    discretize_state,
)

try:
    from reward import calculate_reward
except ImportError:
    try:
        from environment.reward import calculate_reward
    except ImportError:
        from environment.simulator import calculate_reward


EPISODES = 500
ALPHA = 0.1
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995
PLANNING_STEPS = 10
SEED = 42

RESULTS_DIR = Path("results/lab12")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ACTIONS = list(Action)
NUM_ACTIONS = len(ACTIONS)


def run_experiment(use_dyna, seed=SEED):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    q_table = np.zeros((NUM_STATES, NUM_ACTIONS))

    # Model: (state, action) -> observed transition samples.
    model = defaultdict(list)
    known_pairs = []

    episode_returns = []
    pump_steps = []
    demand_satisfaction = []
    pressure_violations = []
    tank_violations = []

    epsilon = EPSILON_START

    for episode in range(EPISODES):
        simulator = WaterSimulator()
        state_obj = simulator.reset()
        state = discretize_state(state_obj)

        total_reward = 0.0
        pump_count = 0
        satisfied_count = 0
        pressure_bad_count = 0
        tank_bad_count = 0

        for step in range(HORIZON):
            action_idx = choose_action(q_table[state], epsilon, np_rng)
            action = ACTIONS[action_idx]

            next_obj = simulator.simulate(action)
            reward = float(calculate_reward(next_obj))

            next_state = discretize_state(next_obj)
            done = (step == HORIZON - 1)

            # Real Q-Learning update.
            target = reward
            if not done:
                target += GAMMA * np.max(q_table[next_state])

            q_table[state, action_idx] += ALPHA * (
                target - q_table[state, action_idx]
            )

            if use_dyna:
                key = (state, action_idx)

                if key not in model:
                    known_pairs.append(key)

                # Retain observed outcomes to represent stochastic dynamics.
                model[key].append((reward, next_state, done))

                # Planning updates using simulated experience.
                for _ in range(PLANNING_STEPS):
                    simulated_key = rng.choice(known_pairs)
                    simulated_state, simulated_action = simulated_key

                    simulated_reward, simulated_next, simulated_done = rng.choice(
                        model[simulated_key]
                    )

                    simulated_target = simulated_reward
                    if not simulated_done:
                        simulated_target += GAMMA * np.max(
                            q_table[simulated_next]
                        )

                    q_table[simulated_state, simulated_action] += ALPHA * (
                        simulated_target
                        - q_table[simulated_state, simulated_action]
                    )

            total_reward += reward
            pump_count += int(next_obj.pump_status == 1)
            satisfied_count += int(next_obj.demand <= next_obj.tank_level)
            pressure_bad_count += int(
                not (40 <= next_obj.pressure <= 70)
            )
            tank_bad_count += int(next_obj.tank_level <= 30)

            state = next_state

        episode_returns.append(total_reward)
        pump_steps.append(pump_count)
        demand_satisfaction.append(satisfied_count / HORIZON)
        pressure_violations.append(pressure_bad_count / HORIZON)
        tank_violations.append(tank_bad_count / HORIZON)

        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)

        if (episode + 1) % 50 == 0:
            name = "Dyna-Q" if use_dyna else "Q-Learning"
            print(
                f"{name}: episode {episode + 1}/{EPISODES}, "
                f"return={total_reward:.1f}, epsilon={epsilon:.3f}"
            )

    return {
        "returns": episode_returns,
        "pump_steps": pump_steps,
        "demand_satisfaction": demand_satisfaction,
        "pressure_violations": pressure_violations,
        "tank_violations": tank_violations,
        "q_table": q_table,
    }


def main():
    # Use different seeds so the methods are not driven by identical random streams.
    q_results = run_experiment(use_dyna=False, seed=SEED)
    dyna_results = run_experiment(use_dyna=True, seed=SEED + 1)

    metrics = {
        "Q-Learning": q_results,
        "Dyna-Q": dyna_results,
    }

    csv_path = RESULTS_DIR / "dyna_q_metrics.csv"

    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "algorithm",
            "mean_return",
            "mean_pump_steps",
            "demand_satisfaction_percent",
            "pressure_violation_percent",
            "low_tank_percent",
        ])

        for name, result in metrics.items():
            writer.writerow([
                name,
                np.mean(result["returns"]),
                np.mean(result["pump_steps"]),
                100 * np.mean(result["demand_satisfaction"]),
                100 * np.mean(result["pressure_violations"]),
                100 * np.mean(result["tank_violations"]),
            ])

    # Plot smoothed episode returns.
    window = 25
    plt.figure()

    for name, result in metrics.items():
        returns = np.asarray(result["returns"])
        smoothed = np.convolve(
            returns, np.ones(window) / window, mode="valid"
        )
        plt.plot(
            range(window, EPISODES + 1),
            smoothed,
            label=name,
        )

    plt.xlabel("Episode")
    plt.ylabel("Episode return (25-episode moving average)")
    plt.title("Water Environment: Q-Learning vs Dyna-Q")
    plt.legend()
    plt.tight_layout()

    plot_path = RESULTS_DIR / "dyna_q_returns.png"
    plt.savefig(plot_path)
    plt.close()

    np.savez(
        RESULTS_DIR / "dyna_q_tables.npz",
        q_learning=q_results["q_table"],
        dyna_q=dyna_results["q_table"],
    )

    print("\nFinal comparison (averaged over all training episodes)")
    for name, result in metrics.items():
        print(f"\n{name}")
        print(f"Mean return: {np.mean(result['returns']):.2f}")
        print(f"Mean pump-on steps: {np.mean(result['pump_steps']):.2f}")
        print(
            "Demand satisfaction: "
            f"{100 * np.mean(result['demand_satisfaction']):.2f}%"
        )
        print(
            "Pressure outside 40-70: "
            f"{100 * np.mean(result['pressure_violations']):.2f}%"
        )
        print(
            "Tank level <= 30: "
            f"{100 * np.mean(result['tank_violations']):.2f}%"
        )

    print(f"\nMetrics saved to: {csv_path}")
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
