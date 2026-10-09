
# algorithms/classical/lab12/water_td_mc.py

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from environment.simulator import WaterSimulator
from algorithms.classical.lab12.water_state_utils import (
    HORIZON,
    NUM_STATES,
    discretize_state,
)

# Adjust this import to the file where calculate_reward is defined.
try:
    from reward import calculate_reward
except ImportError:
    try:
        from environment.reward import calculate_reward
    except ImportError:
        from environment.simulator import calculate_reward


EPISODES = 500
ALPHA = 0.05
GAMMA = 0.99
SEED = 42

RESULTS_DIR = Path("results/lab12")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_episode(simulator, rng):
    """Run one episode using a uniform random policy."""
    state = simulator.reset()
    trajectory = []
    total_reward = 0.0

    for _ in range(HORIZON):
        action = rng.choice(list(__import__(
            "environment.actions", fromlist=["Action"]
        ).Action))

        next_state = simulator.simulate(action)
        reward = calculate_reward(next_state)

        trajectory.append(
            (discretize_state(state), reward, discretize_state(next_state))
        )
        total_reward += reward
        state = next_state

    return trajectory, total_reward


def train_td_mc(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)

    td_values = np.zeros(NUM_STATES)
    mc_values = np.zeros(NUM_STATES)

    td_returns = []
    mc_returns = []
    td_changes = []
    mc_changes = []

    for episode in range(EPISODES):
        simulator = WaterSimulator()
        trajectory, total_reward = run_episode(simulator, rng)

        # TD(0): update after each transition.
        for i, (state, reward, next_state) in enumerate(trajectory):
            terminal = (i == len(trajectory) - 1)
            target = reward if terminal else (
                reward + GAMMA * td_values[next_state]
            )
            td_values[state] += ALPHA * (target - td_values[state])

        # Constant-alpha MC: calculate full discounted returns first.
        returns = [0.0] * len(trajectory)
        G = 0.0

        for i in range(len(trajectory) - 1, -1, -1):
            _, reward, _ = trajectory[i]
            G = reward + GAMMA * G
            returns[i] = G

        for (state, _, _), G in zip(trajectory, returns):
            mc_values[state] += ALPHA * (G - mc_values[state])

        td_returns.append(total_reward)
        mc_returns.append(total_reward)

        # Track how much the value estimates change this episode.
        # This is a convergence diagnostic, not prediction error.
        td_changes.append(float(np.mean(np.abs(td_values))))
        mc_changes.append(float(np.mean(np.abs(mc_values))))

        if (episode + 1) % 50 == 0:
            print(
                f"Episode {episode + 1}/{EPISODES} | "
                f"Episode return: {total_reward:.1f}"
            )

    window = 25
    td_smoothed = np.convolve(
        td_returns, np.ones(window) / window, mode="valid"
    )
    mc_smoothed = np.convolve(
        mc_returns, np.ones(window) / window, mode="valid"
    )

    plt.figure()
    plt.plot(range(window, EPISODES + 1), td_smoothed, label="TD(0)")
    plt.plot(range(window, EPISODES + 1), mc_smoothed, label="Constant-alpha MC")
    plt.xlabel("Episode")
    plt.ylabel("Episode return (25-episode moving average)")
    plt.title("Water Environment: TD(0) vs Constant-alpha MC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "td_mc_returns.png")
    plt.close()

    np.savez(
        RESULTS_DIR / "td_mc_values.npz",
        td_values=td_values,
        mc_values=mc_values,
        episode_returns=np.array(td_returns),
        td_mean_absolute_value=np.array(td_changes),
        mc_mean_absolute_value=np.array(mc_changes),
    )

    print("\nTD(0) vs MC")
    print(f"Mean episode return: {np.mean(td_returns):.2f}")
    print(f"TD(0) visited-state value estimates: {np.count_nonzero(td_values)}")
    print(f"MC visited-state value estimates: {np.count_nonzero(mc_values)}")
    print(f"Saved plot: {RESULTS_DIR / 'td_mc_returns.png'}")
    print(f"Saved estimates: {RESULTS_DIR / 'td_mc_values.npz'}")


if __name__ == "__main__":
    train_td_mc()
