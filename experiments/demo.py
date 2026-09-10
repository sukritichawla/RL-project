"""
Commit 6: live demo.

Runs one greedy (epsilon=0) episode of a chosen algorithm step by step,
printing the action taken and the resulting state after every step, for
a live walkthrough/presentation. If a saved Q-table exists it's loaded;
otherwise the agent is trained fresh first.

Run from the project root, e.g.:
    python -m experiments.demo --algorithm "Q-Learning"
    python -m experiments.demo --algorithm SARSA --train --episodes 300
"""

import argparse
import os
import time

from environment.water_env import WaterEnvironment
from environment.actions import Action, ACTION_NAMES

from algorithms.classical.temporal_difference.q_learning import QLearningAgent
from algorithms.classical.temporal_difference.sarsa import SARSAAgent

from experiments.config import STATE_SIZE, ACTION_SIZE, EPISODES, MAX_STEPS
from experiments.runner import run_training

Q_TABLE_DIR = "results/q_tables"


def build_agent_classes():
    """Same graceful-degradation pattern as compare.py: energy-aware
    agents are included automatically once Person 1's files exist."""

    classes = {
        "Q-Learning": QLearningAgent,
        "SARSA": SARSAAgent,
    }

    try:
        from algorithms.energy_aware.fixed_energy_q_learning import FixedEnergyQLearningAgent
        classes["Fixed Energy Q-Learning"] = FixedEnergyQLearningAgent
    except ImportError:
        pass

    try:
        from algorithms.energy_aware.adaptive_energy_q_learning import AdaptiveEnergyQLearningAgent
        classes["AE-Q"] = AdaptiveEnergyQLearningAgent
    except ImportError:
        pass

    return classes


def q_table_path(algorithm_name):
    safe_name = algorithm_name.lower().replace(" ", "_")
    return os.path.join(Q_TABLE_DIR, f"{safe_name}.npy")


def get_agent(algorithm_name, agent_class, force_train, episodes):
    agent = agent_class(state_size=STATE_SIZE, action_size=ACTION_SIZE)
    path = q_table_path(algorithm_name)

    if not force_train and os.path.exists(path):
        print(f"Loading saved Q-table from {path}")
        agent.load(path)
        return agent

    print(f"Training {algorithm_name} for {episodes} episodes before demo...")
    result = run_training(
        agent_factory=lambda: agent_class(state_size=STATE_SIZE, action_size=ACTION_SIZE),
        env_factory=WaterEnvironment,
        episodes=episodes,
        max_steps=MAX_STEPS,
        seed=0,
    )
    agent = result["agent"]

    os.makedirs(Q_TABLE_DIR, exist_ok=True)
    agent.save(path)
    print(f"Saved Q-table to {path}")

    return agent


def run_demo(agent, env_factory, max_steps, delay=0.3):
    """
    Runs one greedy episode (epsilon forced to 0 for the duration),
    printing each action and the resulting state.
    """

    env = env_factory()
    raw_state = env.reset()
    state = env.discretize_state(raw_state)

    original_epsilon = getattr(agent, "epsilon", None)
    if original_epsilon is not None:
        agent.epsilon = 0.0  # greedy policy for a clean demo, restored after

    total_reward = 0.0
    print("\n--- Demo start ---")
    env.render()

    try:
        for t in range(max_steps):
            action = agent.select_action(state)
            raw_next_state, reward, done, _ = env.step(action)
            state = env.discretize_state(raw_next_state)
            total_reward += reward

            action_name = ACTION_NAMES.get(Action(action), str(action))
            print(f"Step {t + 1}: action={action_name}  reward={reward:.1f}  "
                  f"total_reward={total_reward:.1f}")
            env.render()

            if delay:
                time.sleep(delay)

            if done:
                break
    finally:
        if original_epsilon is not None:
            agent.epsilon = original_epsilon

    print(f"--- Demo end: total_reward={total_reward:.1f} ---\n")
    return total_reward


def main():
    agent_classes = build_agent_classes()

    parser = argparse.ArgumentParser(description="Live demo of a trained agent.")
    parser.add_argument("--algorithm", required=True, choices=list(agent_classes.keys()))
    parser.add_argument("--train", action="store_true",
                         help="Force retraining instead of loading a saved Q-table.")
    parser.add_argument("--episodes", type=int, default=EPISODES,
                         help="Episodes to train for if training is needed.")
    parser.add_argument("--delay", type=float, default=0.3,
                         help="Seconds to pause between steps.")
    args = parser.parse_args()

    agent_class = agent_classes[args.algorithm]
    agent = get_agent(args.algorithm, agent_class, args.train, args.episodes)

    run_demo(agent, WaterEnvironment, MAX_STEPS, delay=args.delay)


if __name__ == "__main__":
    main()
