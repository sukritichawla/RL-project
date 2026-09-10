"""
Multi-run evaluation: trains each algorithm across every seed in
experiments.config.RANDOM_SEEDS, computes per-run metrics (experiments.
metrics.run_metrics), and aggregates them into mean/std per metric so
algorithms can be compared fairly rather than on a single lucky/unlucky
run.
"""

import numpy as np

from experiments.runner import run_multi_seed
from experiments.metrics import run_metrics


def aggregate_metrics(per_seed_metrics):
    """
    per_seed_metrics: list of dicts (one per seed) as returned by
    experiments.metrics.run_metrics -- all values scalar.

    Returns: {metric_name: {"mean": float, "std": float, "values": [..]}}
    """

    keys = per_seed_metrics[0].keys()
    aggregated = {}
    for key in keys:
        values = np.array([m[key] for m in per_seed_metrics], dtype=np.float64)
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "values": values.tolist(),
        }
    return aggregated


def evaluate_algorithm(name, agent_factory, env_factory, episodes, max_steps,
                        seeds, moving_average_window):
    """
    Trains `name` once per seed and aggregates its metrics.

    Returns:
        {
            "name": str,
            "seeds": [...],
            "episode_rewards_per_seed": [[float, ...], ...],  # for learning curves
            "per_seed_metrics": [dict, ...],
            "aggregated": {metric_name: {"mean", "std", "values"}},
        }
    """

    training_results = run_multi_seed(
        agent_factory, env_factory, episodes, max_steps, seeds,
        collect_steps=True,
    )

    per_seed_metrics = [
        run_metrics(result, moving_average_window) for result in training_results
    ]

    return {
        "name": name,
        "seeds": seeds,
        "episode_rewards_per_seed": [r["episode_rewards"] for r in training_results],
        "per_seed_metrics": per_seed_metrics,
        "aggregated": aggregate_metrics(per_seed_metrics),
    }


def evaluate_all(algorithms, env_factory, episodes, max_steps, seeds,
                  moving_average_window, verbose=True):
    """
    algorithms: dict[name -> agent_factory callable, e.g. lambda: QLearningAgent(...)]

    Returns: dict[name -> evaluate_algorithm(...) result]
    """

    results = {}
    for name, factory in algorithms.items():
        if verbose:
            print(f"Evaluating {name} over {len(seeds)} seeds "
                  f"({episodes} episodes each)...")
        results[name] = evaluate_algorithm(
            name, factory, env_factory, episodes, max_steps, seeds,
            moving_average_window,
        )
    return results


def print_summary(results, metrics_to_show=None):
    """
    Prints a plain-text comparison table (mean ± std) across algorithms.
    """

    if metrics_to_show is None:
        metrics_to_show = [
            "avg_reward", "total_energy", "demand_satisfaction_rate",
            "pressure_violations", "tank_violations", "pump_switches",
            "convergence_episode",
        ]

    names = list(results.keys())
    col_width = max(18, max(len(n) for n in names) + 2)

    header = "metric".ljust(22) + "".join(n.ljust(col_width) for n in names)
    print(header)
    print("-" * len(header))

    for metric in metrics_to_show:
        row = metric.ljust(22)
        for name in names:
            stats = results[name]["aggregated"][metric]
            row += f"{stats['mean']:.2f} ± {stats['std']:.2f}".ljust(col_width)
        print(row)
