"""
Interactive live demo for RL-based sustainable urban water distribution.

Algorithms:
    1. Q-Learning
    2. SARSA
    3. Fixed Energy-Aware Q-Learning
    4. Adaptive Energy-Aware Q-Learning (AE-Q)

Run from the project root:

    python -m experiments.demo
"""

import os
import random
import time

import numpy as np

from environment.water_env import WaterEnvironment
from environment.actions import Action

from environment.reward import calculate_reward
from energy import calculate_energy

from rewards.fixed_energy_reward import (
    calculate_fixed_energy_reward
)

from rewards.adaptive_energy_reward import (
    calculate_adaptive_reward
)

from algorithms.classical.temporal_difference.q_learning import (
    QLearningAgent
)

from algorithms.classical.temporal_difference.sarsa import (
    SARSAAgent
)

from algorithms.energy_aware.fixed_energy_q_learning import (
    FixedEnergyQLearningAgent
)

from algorithms.energy_aware.adaptive_energy_q_learning import (
    AdaptiveEnergyQLearningAgent
)

from experiments.config import (
    STATE_SIZE,
    ACTION_SIZE,
    EPISODES,
    MAX_STEPS
)


Q_TABLE_DIR = "results/q_tables"


def _fixed_energy_reward_fn(env, base_reward):
    from rewards.fixed_energy_reward import calculate_fixed_energy_reward
    reward, energy = calculate_fixed_energy_reward(env.state)
    return reward, {"energy": energy}


def _adaptive_energy_reward_fn(env, base_reward):
    from rewards.adaptive_energy_reward import calculate_adaptive_reward
    reward, energy, lambda_t = calculate_adaptive_reward(env.state)
    return reward, {"energy": energy, "lambda_t": lambda_t}


ALGORITHMS = {
    "1": ("Q-Learning", QLearningAgent),
    "2": ("SARSA", SARSAAgent),
    "3": (
        "Fixed Energy-Aware Q-Learning",
        FixedEnergyQLearningAgent
    ),
    "4": (
        "Adaptive Energy-Aware Q-Learning (AE-Q)",
        AdaptiveEnergyQLearningAgent
    ),
}


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_title():
    print()
    print("=" * 72)
    print("        RL FOR SUSTAINABLE URBAN WATER DISTRIBUTION")
    print("                         LIVE DEMO")
    print("=" * 72)


def select_algorithm():
    while True:

        clear_screen()
        print_title()

        print()
        print("Select RL Algorithm")
        print("-" * 72)
        print("  [1] Q-Learning")
        print("  [2] SARSA")
        print("  [3] Fixed Energy-Aware Q-Learning")
        print("  [4] Adaptive Energy-Aware Q-Learning (AE-Q)")
        print("  [0] Exit")
        print("-" * 72)

        choice = input("Enter choice: ").strip()

        if choice == "0":
            return None

        if choice in ALGORITHMS:
            return ALGORITHMS[choice]

        print()
        print("Invalid choice. Please enter 0, 1, 2, 3, or 4.")
        input("Press ENTER to continue...")


def train_standard_agent(agent, env, episodes):
    """
    Train Q-Learning or SARSA using the common environment.
    """

    print()
    print(f"Training {agent.name}...")
    print(f"Episodes: {episodes}")
    print()

    for episode in range(episodes):

        state_array = env.reset()
        state = env.discretize_state(state_array)

        if isinstance(agent, SARSAAgent):
            action = agent.select_action(state)

        total_reward = 0.0

        for _ in range(MAX_STEPS):

            if isinstance(agent, QLearningAgent):
                action = agent.select_action(state)

            next_state_array, reward, done, _ = env.step(action)

            next_state = env.discretize_state(
                next_state_array
            )

            if isinstance(agent, SARSAAgent):

                if done:
                    next_action = None
                else:
                    next_action = agent.select_action(
                        next_state
                    )

                agent.update(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    next_action
                )

            else:

                agent.update(
                    state,
                    action,
                    reward,
                    next_state,
                    done
                )

            total_reward += reward

            if done:
                break

            state = next_state

            if isinstance(agent, SARSAAgent):
                action = next_action

        if (episode + 1) % 100 == 0:
            print(
                f"  Episode {episode + 1:4d}/{episodes} | "
                f"Reward: {total_reward:8.2f}"
            )

    print()
    print("Training complete.")


def train_energy_agent(agent, episodes):
    """
    Train the energy-aware agent using its own training implementation.
    """

    print()
    print(f"Training {agent.__class__.__name__}...")
    print(f"Episodes: {episodes}")
    print()

    result = agent.train()

    if isinstance(result, dict):
        rewards = result.get("rewards", [])
    else:
        rewards = result

    if rewards:
        print(
            f"Final training reward: {rewards[-1]:.2f}"
        )

    print()
    print("Training complete.")


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


def action_to_environment_action(action_index):
    """
    All four algorithms currently use the restricted action space:

        0 -> PUMP_ON
        1 -> PUMP_OFF
    """

    if int(action_index) == 0:
        return Action.PUMP_ON

    return Action.PUMP_OFF


def print_live_state(
    algorithm_name,
    step,
    max_steps,
    state,
    action_name,
    energy,
    water_reward,
    reward,
    energy_penalty,
    lambda_value,
    cumulative_reward,
    cumulative_energy,
    demand_satisfied,
    pressure_violation,
    tank_violation,
    pump_switches
):

    clear_screen()

    print_title()

    print()
    print(f"Algorithm: {algorithm_name}")
    print(f"Step     : {step} / {max_steps}")

    print()
    print("-" * 72)
    print("                    WATER NETWORK STATE")
    print("-" * 72)

    print(
        f"  Tank Level          : {state.tank_level:8.2f}"
    )

    print(
        f"  Pressure            : {state.pressure:8.2f}"
    )

    print(
        f"  Demand              : {state.demand:8.2f}"
    )

    print(
        f"  Pump Status         : "
        f"{'ON' if state.pump_status else 'OFF'}"
    )

    print(
        f"  Valve Status        : "
        f"{'OPEN' if state.valve_status else 'CLOSED'}"
    )

    print()
    print("-" * 72)
    print("                         RL DECISION")
    print("-" * 72)

    print(
        f"  Action              : {action_name}"
    )

    print(
        f"  Energy This Step    : {energy:8.2f}"
    )

    print(
        f"  Water Reward        : {water_reward:8.2f}"
    )

    if lambda_value is not None:

        print(
            f"  Lambda              : {lambda_value:8.3f}"
        )

        print(
            f"  Energy Penalty      : {energy_penalty:8.2f}"
        )

    print(
        f"  Step Reward         : {reward:8.2f}"
    )

    print(
        f"  Cumulative Reward   : {cumulative_reward:8.2f}"
    )

    print(
        f"  Cumulative Energy   : {cumulative_energy:8.2f}"
    )

    print()
    print("-" * 72)
    print("                         PERFORMANCE")
    print("-" * 72)

    print(
        f"  Demand Satisfied    : "
        f"{'YES' if demand_satisfied else 'NO'}"
    )

    print(
        f"  Pressure Violation  : "
        f"{'YES' if pressure_violation else 'NO'}"
    )

    print(
        f"  Tank Violation      : "
        f"{'YES' if tank_violation else 'NO'}"
    )

    print(
        f"  Pump Switches       : {pump_switches}"
    )

    print("-" * 72)

    print()
    print("Running live simulation...")


def run_live_demo(
    agent,
    algorithm_name,
    steps=50,
    delay=0.5
):

    env = WaterEnvironment()

    raw_state = env.reset()

    state = env.discretize_state(raw_state)

    # Force greedy behaviour for demonstration.
    original_epsilon = getattr(
        agent,
        "epsilon",
        None
    )

    if original_epsilon is not None:
        agent.epsilon = 0.0

    cumulative_reward = 0.0
    cumulative_energy = 0.0

    demand_satisfied_count = 0
    pressure_violation_count = 0
    tank_violation_count = 0
    pump_switches = 0

    previous_pump_status = env.state.pump_status

    try:

        for step in range(1, steps + 1):

            # ------------------------------------------------------
            # Select action
            # ------------------------------------------------------

            action_index = agent.select_action(state)

            environment_action = (
                action_to_environment_action(
                    action_index
                )
            )

            action_name = (
                "Pump ON"
                if environment_action == Action.PUMP_ON
                else "Pump OFF"
            )

            # ------------------------------------------------------
            # Environment transition
            # ------------------------------------------------------

            next_state_array, environment_reward, done, _ = (
                env.step(environment_action)
            )

            next_state = env.discretize_state(
                next_state_array
            )

            current_state = env.state

            # ------------------------------------------------------
            # Energy
            # ------------------------------------------------------

            energy = calculate_energy(
                current_state.pump_status
            )

            cumulative_energy += energy

            # ------------------------------------------------------
            # Reward
            # ------------------------------------------------------

            water_reward = calculate_reward(
                current_state
            )

            lambda_value = None
            energy_penalty = 0.0

            if algorithm_name == (
                "Fixed Energy-Aware Q-Learning"
            ):

                reward, _ = calculate_fixed_energy_reward(
                    current_state
                )

                energy_penalty = (
                    water_reward - reward
                )

                lambda_value = 0.5

            elif algorithm_name == (
                "Adaptive Energy-Aware Q-Learning (AE-Q)"
            ):

                reward, _, lambda_value = (
                    calculate_adaptive_reward(
                        current_state
                    )
                )

                energy_penalty = (
                    water_reward - reward
                )

            else:

                reward = environment_reward

            cumulative_reward += reward

            # ------------------------------------------------------
            # Performance metrics
            # ------------------------------------------------------

            demand_satisfied = (
                current_state.demand
                <= current_state.tank_level
            )

            pressure_violation = not (
                40
                <= current_state.pressure
                <= 70
            )

            tank_violation = (
                current_state.tank_level < 30
            )

            if demand_satisfied:
                demand_satisfied_count += 1

            if pressure_violation:
                pressure_violation_count += 1

            if tank_violation:
                tank_violation_count += 1

            if (
                current_state.pump_status
                != previous_pump_status
            ):
                pump_switches += 1

            previous_pump_status = (
                current_state.pump_status
            )

            # ------------------------------------------------------
            # Display
            # ------------------------------------------------------

            print_live_state(
                algorithm_name=algorithm_name,
                step=step,
                max_steps=steps,
                state=current_state,
                action_name=action_name,
                energy=energy,
                water_reward=water_reward,
                reward=reward,
                energy_penalty=energy_penalty,
                lambda_value=lambda_value,
                cumulative_reward=cumulative_reward,
                cumulative_energy=cumulative_energy,
                demand_satisfied=demand_satisfied,
                pressure_violation=pressure_violation,
                tank_violation=tank_violation,
                pump_switches=pump_switches
            )

            state = next_state

            time.sleep(delay)

            if done:
                break

    finally:

        if original_epsilon is not None:
            agent.epsilon = original_epsilon

    completed_steps = step

    satisfaction_rate = (
        demand_satisfied_count
        / completed_steps
        * 100
    )

    print()
    print("=" * 72)
    print("                         DEMO COMPLETE")
    print("=" * 72)

    print(
        f"  Algorithm             : {algorithm_name}"
    )

    print(
        f"  Steps                 : {completed_steps}"
    )

    print(
        f"  Cumulative Reward     : "
        f"{cumulative_reward:.2f}"
    )

    print(
        f"  Total Energy          : "
        f"{cumulative_energy:.2f}"
    )

    print(
        f"  Demand Satisfaction   : "
        f"{satisfaction_rate:.2f}%"
    )

    print(
        f"  Pressure Violations   : "
        f"{pressure_violation_count}"
    )

    print(
        f"  Tank Violations       : "
        f"{tank_violation_count}"
    )

    print(
        f"  Pump Switches         : "
        f"{pump_switches}"
    )

    print("=" * 72)


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