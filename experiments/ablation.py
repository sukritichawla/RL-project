"""
Generic ablation runner: sweeps one or more constructor kwargs of a
single agent class and evaluates each configuration exactly like
experiments.evaluation.evaluate_algorithm does, so results are directly
comparable to the main algorithm comparison.

This file is intentionally agent-agnostic -- it doesn't hardcode any
energy-aware agent's parameter names, since those aren't finalized yet.
Point it at whatever agent/parameter you want to study from compare.py.
"""

from experiments.evaluation import evaluate_algorithm, aggregate_metrics


def run_ablation(agent_class, fixed_kwargs, param_grid, env_factory,
                  episodes, max_steps, seeds, moving_average_window,
                  reward_fn=None, name_fn=None, verbose=True):
    """
    agent_class:  the agent class to sweep, e.g. FixedEnergyQLearningAgent
    fixed_kwargs: dict of kwargs held constant across all configurations
                  (e.g. {"state_size": 3, "action_size": 2})
    param_grid:   list of dicts, each holding the kwargs being varied for
                  one configuration, e.g.
                      [{"alpha": 0.05}, {"alpha": 0.1}, {"alpha": 0.2}]
    reward_fn:    optional (env, base_reward) -> (reward, extra_info).
                  Required if agent_class is Fixed/AdaptiveEnergyQLearningAgent
                  -- those train on rewards.fixed_energy_reward /
                  rewards.adaptive_energy_reward, not the environment's
                  baseline reward. Same reward_fn is used for every
                  configuration in param_grid (it doesn't vary per-config
                  unless the parameter being swept lives inside the
                  reward function itself, e.g. BASE_LAMBDA/ALPHA/BETA --
                  see compare.py for that case).
    name_fn:      optional function(varied_kwargs) -> str label. Defaults
                  to a comma-joined "key=value" string.

    Returns: dict[label -> evaluate_algorithm(...) result], in the same
    shape experiments.evaluation.evaluate_all returns, so it can be
    passed straight into experiments.plots / experiments.export.
    """

    if name_fn is None:
        name_fn = lambda kwargs: ", ".join(f"{k}={v}" for k, v in kwargs.items())

    results = {}
    for varied_kwargs in param_grid:
        label = name_fn(varied_kwargs)
        all_kwargs = {**fixed_kwargs, **varied_kwargs}

        if verbose:
            print(f"Ablation: {label}")

        agent_factory = lambda kw=all_kwargs: agent_class(**kw)

        results[label] = evaluate_algorithm(
            label, agent_factory, env_factory, episodes, max_steps, seeds,
            moving_average_window, reward_fn=reward_fn,
        )

    return results
