"""
Comparison plots for the evaluated algorithms (experiments.evaluation.
evaluate_all output). Requires matplotlib.

Generates:
    learning_curves.png     mean reward per episode, one line per
                             algorithm, ± std shaded band across seeds
    bar_<metric>.png        one bar chart per requested summary metric
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from experiments.metrics import moving_average


def _ensure_dir(filepath):
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)


def plot_learning_curves(results, moving_average_window, filepath,
                          title="Learning Curves"):
    plt.figure(figsize=(9, 5.5))

    for name, result in results.items():
        per_seed = np.array(result["episode_rewards_per_seed"])  # (seeds, episodes)

        smoothed_per_seed = np.array([
            moving_average(run, moving_average_window) for run in per_seed
        ])

        mean_curve = smoothed_per_seed.mean(axis=0)
        std_curve = smoothed_per_seed.std(axis=0)
        episodes = np.arange(len(mean_curve)) + moving_average_window - 1

        plt.plot(episodes, mean_curve, label=name)
        plt.fill_between(episodes, mean_curve - std_curve, mean_curve + std_curve,
                          alpha=0.15)

    plt.xlabel("Episode")
    plt.ylabel(f"Reward ({moving_average_window}-episode moving average)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    _ensure_dir(filepath)
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Wrote {filepath}")


def plot_metric_bar(results, metric_key, ylabel, filepath, title=None):
    names = list(results.keys())
    means = [results[n]["aggregated"][metric_key]["mean"] for n in names]
    stds = [results[n]["aggregated"][metric_key]["std"] for n in names]

    plt.figure(figsize=(7, 5))
    plt.bar(names, means, yerr=stds, capsize=5)
    plt.ylabel(ylabel)
    plt.title(title or metric_key.replace("_", " ").title())
    plt.xticks(rotation=15)
    plt.tight_layout()

    _ensure_dir(filepath)
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Wrote {filepath}")


# Standard set of comparison bar charts required by the plan.
DEFAULT_BAR_METRICS = [
    ("total_energy", "Total energy consumed", "Energy Consumption"),
    ("demand_satisfaction_rate", "Demand satisfaction (%)", "Demand Satisfaction"),
    ("pressure_violations", "Pressure violations (count)", "Pressure Violations"),
    ("tank_violations", "Tank-level violations (count)", "Tank-Level Violations"),
    ("pump_switches", "Pump switches (count)", "Pump Switching Frequency"),
    ("convergence_episode", "Episode reached at convergence", "Convergence Speed"),
]


def plot_all(results, moving_average_window, output_dir="results/plots"):
    plot_learning_curves(
        results, moving_average_window,
        os.path.join(output_dir, "learning_curves.png"),
    )

    for metric_key, ylabel, title in DEFAULT_BAR_METRICS:
        plot_metric_bar(
            results, metric_key, ylabel,
            os.path.join(output_dir, f"bar_{metric_key}.png"),
            title=title,
        )
