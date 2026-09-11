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


def fixed_energy_reward_fn(env, base_reward):
    """
    FixedEnergyQLearningAgent's own train() ignores env.step()'s reward
    and instead calls calculate_fixed_energy_reward(env.state) -- i.e.
    the baseline reward minus a FIXED penalty (LAMBDA=0.5) times energy
    consumed. Reproduced here so the generic runner trains on the same
    signal Person 1's agent actually uses.
    """
    from rewards.fixed_energy_reward import calculate_fixed_energy_reward
    reward, energy = calculate_fixed_energy_reward(env.state)
    return reward, {"energy": energy}


def adaptive_energy_reward_fn(env, base_reward):
    """
    AdaptiveEnergyQLearningAgent's own train() likewise ignores
    env.step()'s reward and calls calculate_adaptive_reward(env.state),
    which computes lambda_t from the current tank_level/demand (the
    paper's λₜ = λ₀(1 + β·T/Tmax − α·D/Dmax) formula) and subtracts
    lambda_t * energy from the baseline reward.
    """
    from rewards.adaptive_energy_reward import calculate_adaptive_reward
    reward, energy, lambda_t = calculate_adaptive_reward(env.state)
    return reward, {"energy": energy, "lambda_t": lambda_t}


def build_algorithms():
    """
    Builds the algorithm registry. Q-Learning and SARSA are always
    available. The two energy-aware agents are added only if Person 1
    has pushed them -- this script degrades gracefully to a 2-way
    comparison until then, and upgrades to 4-way automatically once
    they exist, without needing edits here.

    Returns:
        algorithms: dict[name -> agent_factory]
        reward_fns: dict[name -> reward_fn] for algorithms that must NOT
            be trained on the environment's baseline reward (see
            fixed_energy_reward_fn / adaptive_energy_reward_fn above).
            Q-Learning/SARSA are intentionally absent from this dict --
            they use the environment's own reward, unchanged.
    """

    algorithms = {
        "Q-Learning": lambda: QLearningAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE),
        "SARSA": lambda: SARSAAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE),
    }
    reward_fns = {}

    try:
        from algorithms.energy_aware.fixed_energy_q_learning import FixedEnergyQLearningAgent
        algorithms["Fixed Energy Q-Learning"] = lambda: FixedEnergyQLearningAgent(
            state_size=STATE_SIZE, action_size=ACTION_SIZE
        )
        reward_fns["Fixed Energy Q-Learning"] = fixed_energy_reward_fn
    except ImportError as e:
        print(f"[compare] Skipping Fixed Energy Q-Learning (not available yet): {e}")

    try:
        from algorithms.energy_aware.adaptive_energy_q_learning import AdaptiveEnergyQLearningAgent
        algorithms["AE-Q"] = lambda: AdaptiveEnergyQLearningAgent(
            state_size=STATE_SIZE, action_size=ACTION_SIZE
        )
        reward_fns["AE-Q"] = adaptive_energy_reward_fn
    except ImportError as e:
        print(f"[compare] Skipping AE-Q (not available yet): {e}")

    return algorithms, reward_fns


def run_main_comparison():
    algorithms, reward_fns = build_algorithms()
    env_factory = WaterEnvironment

    results = evaluate_all(
        algorithms, env_factory, EPISODES, MAX_STEPS, RANDOM_SEEDS,
        MOVING_AVERAGE_WINDOW, reward_fns=reward_fns,
    )

    print()
    print_summary(results)
    print()

    export_all(results, output_dir="results")
    plot_all(results, MOVING_AVERAGE_WINDOW, output_dir="results/plots")

    return results, algorithms, reward_fns


def run_ablation_study(env_factory):
    """
    Ablation study: sweeps the learning rate (alpha) of Fixed Energy
    Q-Learning -- a real constructor parameter -- now correctly trained
    on its actual reward (calculate_fixed_energy_reward) instead of the
    environment's baseline reward.

    This is a learning-rate sensitivity study, not a test of the
    adaptive λ mechanism itself: BASE_LAMBDA/ALPHA/BETA in
    rewards/adaptive_energy_reward.py are module-level constants, not
    constructor kwargs, so sweeping *those* would need a small
    parameterization change to that file (making them optional function
    arguments defaulting to their current values) -- worth doing as a
    follow-up if you want to test the paper's actual claim (adaptive
    weighting vs fixed), rather than this alpha sweep.
    """

    try:
        from algorithms.energy_aware.fixed_energy_q_learning import FixedEnergyQLearningAgent
    except ImportError as e:
        print(f"[ablation] Skipping ablation study (Fixed Energy Q-Learning not available yet): {e}")
        return None

    PARAM_GRID = [
        {"alpha": 0.05},
        {"alpha": 0.10},
        {"alpha": 0.20},
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
            reward_fn=fixed_energy_reward_fn,
        )
    except TypeError as e:
        print(f"[ablation] FixedEnergyQLearningAgent doesn't accept the kwargs in "
              f"PARAM_GRID ({e}). Update PARAM_GRID in experiments/compare.py to "
              f"match its real constructor.")
        return None

    print()
    print("Ablation study: alpha sweep (Fixed Energy Q-Learning, correct reward)")
    print_summary(ablation_results)

    export_all(ablation_results, output_dir="results/ablation")
    plot_all(ablation_results, MOVING_AVERAGE_WINDOW, output_dir="results/ablation/plots")

    return ablation_results


def main():
    results, algorithms, reward_fns = run_main_comparison()

    if "Fixed Energy Q-Learning" in algorithms:
        run_ablation_study(WaterEnvironment)
    else:
        print("[ablation] Skipped: Fixed Energy Q-Learning not in this run's "
              "algorithm set.")

    print()
    print("Done. See results/ for CSVs and results/plots/ for graphs.")


if __name__ == "__main__":
    main()
