"""
Commit 1 smoke test.

Runs the existing Q-Learning and SARSA agents through the new generic
runner and prints basic sanity numbers. Not a unit test suite -- just
the "does the plumbing work" check the plan calls for before building
metrics/evaluation on top of it.

Run from the project root:
    python -m experiments.test_runner_smoke
"""

import numpy as np

from environment.water_env import WaterEnvironment
from algorithms.classical.temporal_difference.q_learning import QLearningAgent
from algorithms.classical.temporal_difference.sarsa import SARSAAgent

from experiments.config import STATE_SIZE, ACTION_SIZE, MAX_STEPS
from experiments.runner import run_training


def _summarize(name, result, episodes):
    rewards = result["episode_rewards"]
    assert len(rewards) == episodes, f"{name}: expected {episodes} episode rewards, got {len(rewards)}"
    assert all(np.isfinite(r) for r in rewards), f"{name}: found a non-finite reward"

    q_table = result["agent"].q_table
    assert q_table.shape == (STATE_SIZE, ACTION_SIZE), f"{name}: unexpected Q-table shape {q_table.shape}"
    assert np.any(q_table != 0), f"{name}: Q-table never updated"

    first_avg = np.mean(rewards[:10])
    last_avg = np.mean(rewards[-10:])

    print(f"[{name}] episodes={len(rewards)}  "
          f"first-10 avg reward={first_avg:.2f}  last-10 avg reward={last_avg:.2f}")
    print(f"[{name}] Q-table (state x action):\n{q_table}\n")


def main():
    episodes = 200  # short run, just to prove the wiring works

    q_result = run_training(
        agent_factory=lambda: QLearningAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE),
        env_factory=WaterEnvironment,
        episodes=episodes,
        max_steps=MAX_STEPS,
        seed=0,
    )
    _summarize("Q-Learning", q_result, episodes)

    sarsa_result = run_training(
        agent_factory=lambda: SARSAAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE),
        env_factory=WaterEnvironment,
        episodes=episodes,
        max_steps=MAX_STEPS,
        seed=0,
    )
    _summarize("SARSA", sarsa_result, episodes)

    print("Commit 1 smoke test passed: runner works with both existing agents.")


if __name__ == "__main__":
    main()
