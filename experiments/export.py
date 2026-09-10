"""
CSV export for evaluation results (experiments.evaluation.evaluate_all
output). Uses the stdlib csv module only -- no pandas dependency.

Produces three files:
    results_summary.csv     one row per algorithm: mean ± std per metric
    results_per_seed.csv    one row per (algorithm, seed): raw metric values
    learning_curves.csv     one row per episode: mean reward per algorithm
                             (averaged across seeds), for plotting elsewhere
"""

import csv
import os

import numpy as np


def _ensure_dir(filepath):
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)


def export_summary_csv(results, filepath):
    algorithms = list(results.keys())
    metric_names = list(next(iter(results.values()))["aggregated"].keys())

    _ensure_dir(filepath)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["algorithm"]
        for metric in metric_names:
            header += [f"{metric}_mean", f"{metric}_std"]
        writer.writerow(header)

        for name in algorithms:
            aggregated = results[name]["aggregated"]
            row = [name]
            for metric in metric_names:
                row += [aggregated[metric]["mean"], aggregated[metric]["std"]]
            writer.writerow(row)

    print(f"Wrote {filepath}")


def export_per_seed_csv(results, filepath):
    algorithms = list(results.keys())
    metric_names = list(next(iter(results.values()))["per_seed_metrics"][0].keys())

    _ensure_dir(filepath)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["algorithm", "seed"] + metric_names)

        for name in algorithms:
            result = results[name]
            for seed, metrics in zip(result["seeds"], result["per_seed_metrics"]):
                row = [name, seed] + [metrics[m] for m in metric_names]
                writer.writerow(row)

    print(f"Wrote {filepath}")


def export_learning_curves_csv(results, filepath):
    """
    One row per episode, one column per algorithm, holding the reward
    averaged across that algorithm's seeds for that episode. Assumes
    every algorithm was run for the same number of episodes (true when
    driven from experiments.config.EPISODES).
    """

    algorithms = list(results.keys())
    episode_counts = {
        name: len(results[name]["episode_rewards_per_seed"][0])
        for name in algorithms
    }
    num_episodes = min(episode_counts.values())
    if len(set(episode_counts.values())) > 1:
        print(f"Warning: algorithms ran different episode counts {episode_counts}; "
              f"truncating learning_curves.csv to {num_episodes} episodes.")

    mean_curves = {}
    for name in algorithms:
        per_seed = np.array(
            [rewards[:num_episodes] for rewards in results[name]["episode_rewards_per_seed"]]
        )
        mean_curves[name] = per_seed.mean(axis=0)

    _ensure_dir(filepath)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode"] + algorithms)
        for ep in range(num_episodes):
            writer.writerow([ep] + [mean_curves[name][ep] for name in algorithms])

    print(f"Wrote {filepath}")


def export_all(results, output_dir="results"):
    export_summary_csv(results, os.path.join(output_dir, "results_summary.csv"))
    export_per_seed_csv(results, os.path.join(output_dir, "results_per_seed.csv"))
    export_learning_curves_csv(results, os.path.join(output_dir, "learning_curves.csv"))
