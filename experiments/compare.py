"""
Commit 6: final comparison script.

Runs every available algorithm through the same evaluation pipeline,
exports CSVs, generates comparison plots, prints a summary table, and
(if the energy-aware agents are present) runs an ablation study.

Run from the project root:
    python -m experiments.compare
"""

from environment.water_env import WaterEnvironment

from algorithms.classical.temporal_difference.q_learning import QLearningAgent
from algorithms.classical.temporal_difference.sarsa import SARSAAgent

from experiments.config import (
    STATE_SIZE, ACTION_SIZE, EPISODES, MAX_STEPS, RANDOM_SEEDS,
    MOVING_AVERAGE_WINDOW,
)
from experiments.evaluation import evaluate_all, print_summary
from experiments.export import export_all
from experiments.plots import plot_all
from experiments.ablation import run_ablation


def build_algorithms():
    """
    Builds the algorithm registry. Q-Learning and SARSA are always
    available. The two energy-aware agents are added only if Person 1
    has pushed them -- this script degrades gracefully to a 2-way
    comparison until then, and upgrades to 4-way automatically once
    they exist, without needing edits here.

    NOTE: import paths and constructor kwargs below (energy_weight etc.)
    are best guesses based on the project's naming conventions. If
    Person 1's actual class/kwarg names differ, update the two try
    blocks to match -- everything downstream (evaluation/export/plots)
    is unaffected either way.
    """

    algorithms = {
        "Q-Learning": lambda: QLearningAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE),
        "SARSA": lambda: SARSAAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE),
    }

    try:
        from algorithms.energy_aware.fixed_energy_q_learning import FixedEnergyQLearningAgent
        algorithms["Fixed Energy Q-Learning"] = lambda: FixedEnergyQLearningAgent(
            state_size=STATE_SIZE, action_size=ACTION_SIZE
        )
    except ImportError as e:
        print(f"[compare] Skipping Fixed Energy Q-Learning (not available yet): {e}")

    try:
        from algorithms.energy_aware.adaptive_energy_q_learning import AdaptiveEnergyQLearningAgent
        algorithms["AE-Q"] = lambda: AdaptiveEnergyQLearningAgent(
            state_size=STATE_SIZE, action_size=ACTION_SIZE
        )
    except ImportError as e:
        print(f"[compare] Skipping AE-Q (not available yet): {e}")

    return algorithms


def run_main_comparison():
    algorithms = build_algorithms()
    env_factory = WaterEnvironment

    results = evaluate_all(
        algorithms, env_factory, EPISODES, MAX_STEPS, RANDOM_SEEDS,
        MOVING_AVERAGE_WINDOW,
    )

    print()
    print_summary(results)
    print()

    export_all(results, output_dir="results")
    plot_all(results, MOVING_AVERAGE_WINDOW, output_dir="results/plots")

    return results, algorithms


def run_ablation_study(env_factory):
    """
    Ablation study: sweeps the energy-penalty weight in Fixed Energy
    Q-Learning to show that AE-Q's adaptive weighting isn't just
    reproducing one lucky fixed setting.

    Skipped with a message if Fixed Energy Q-Learning isn't available
    yet, or if its constructor doesn't actually take `energy_weight`
    (update PARAM_GRID / the kwarg name below to match the real
    constructor once you confirm it).
    """

    try:
        from algorithms.energy_aware.fixed_energy_q_learning import FixedEnergyQLearningAgent
    except ImportError as e:
        print(f"[ablation] Skipping ablation study (Fixed Energy Q-Learning not available yet): {e}")
        return None

    PARAM_GRID = [
        {"energy_weight": 0.0},
        {"energy_weight": 0.25},
        {"energy_weight": 0.5},
        {"energy_weight": 1.0},
    ]

    try:
        ablation_results = run_ablation(
            agent_class=FixedEnergyQLearningAgent,
            fixed_kwargs={"state_size": STATE_SIZE, "action_size": ACTION_SIZE},
            param_grid=PARAM_GRID,
            env_factory=env_factory,
            episodes=EPISODES,
            max_steps=MAX_STEPS,
            seeds=RANDOM_SEEDS,
            moving_average_window=MOVING_AVERAGE_WINDOW,
        )
    except TypeError as e:
        print(f"[ablation] FixedEnergyQLearningAgent doesn't accept the kwargs in "
              f"PARAM_GRID ({e}). Update PARAM_GRID in experiments/compare.py to "
              f"match its real constructor.")
        return None

    print()
    print("Ablation study: energy_weight sweep")
    print_summary(ablation_results)

    export_all(ablation_results, output_dir="results/ablation")
    plot_all(ablation_results, MOVING_AVERAGE_WINDOW, output_dir="results/ablation/plots")

    return ablation_results


def main():
    results, algorithms = run_main_comparison()

    if "Fixed Energy Q-Learning" in algorithms:
        run_ablation_study(WaterEnvironment)
    else:
        print("[ablation] Skipped: Fixed Energy Q-Learning not in this run's "
              "algorithm set.")

    print()
    print("Done. See results/ for CSVs and results/plots/ for graphs.")


if __name__ == "__main__":
    main()
