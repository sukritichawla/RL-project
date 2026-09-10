"""
Generic training/evaluation runner.

Built around the interfaces that already exist in this project, without
modifying them:

    Agent:
        select_action(state) -> action
        update(state, action, reward, next_state, done[, next_action])

    Environment (WaterEnvironment):
        reset() -> raw_state (np.ndarray)
        step(action) -> (raw_next_state, reward, done, info)
        discretize_state(raw_state) -> discrete_state (int)

QLearningAgent.update() takes 5 args (off-policy: bootstraps off max
Q(next_state)). SARSAAgent.update() takes 6 args including next_action
(on-policy). This runner detects which shape an agent uses via its
update() signature, so the same loop drives both today, and will drive
FixedEnergyQLearningAgent / AdaptiveEnergyQLearningAgent later as long as
they follow the same convention -- no changes needed here or to them.
"""

import inspect

import numpy as np


def _wants_next_action(update_fn):
    """True if update() accepts a next_action argument (on-policy, e.g. SARSA)."""
    params = inspect.signature(update_fn).parameters
    return "next_action" in params


def run_episode(agent, env, max_steps, collect_steps=True):
    """
    Runs a single episode.

    The action for the *next* step is always chosen before update() is
    called. On-policy agents (SARSA) use it directly. Off-policy agents
    (Q-Learning) ignore it internally and simply re-derive their own
    action from the current Q-values each step -- so pre-selecting it
    here changes nothing about their behavior or update targets, it only
    lets one loop serve both agent styles.

    Returns:
        total_reward (float)
        steps (list[dict] or None): one record per environment step,
            each holding the raw (undiscretized) state arrays alongside
            the discretized state/action/reward/done. Metrics that need
            more than the episode-total reward (energy, pump switching,
            tank/pressure trajectories, etc.) will be derived from this
            in experiments/metrics.py.
    """

    on_policy = _wants_next_action(agent.update)

    raw_state = env.reset()
    state = env.discretize_state(raw_state)
    action = agent.select_action(state)

    total_reward = 0.0
    steps = [] if collect_steps else None

    for t in range(max_steps):

        raw_next_state, reward, done, info = env.step(action)
        next_state = env.discretize_state(raw_next_state)

        next_action = None
        if not done:
            next_action = agent.select_action(next_state)

        if on_policy:
            agent.update(state, action, reward, next_state, done, next_action)
        else:
            agent.update(state, action, reward, next_state, done)

        total_reward += reward

        if collect_steps:
            steps.append({
                "t": t,
                "raw_state": raw_state,
                "state": state,
                "action": int(action),
                "reward": float(reward),
                "raw_next_state": raw_next_state,
                "next_state": next_state,
                "done": bool(done),
                "info": info,
            })

        if done:
            break

        raw_state = raw_next_state
        state = next_state
        action = next_action

    return total_reward, steps


def run_training(agent_factory, env_factory, episodes, max_steps,
                  seed=None, collect_steps=False):
    """
    Trains one fresh agent for `episodes` episodes on a fresh environment.

    agent_factory: () -> agent instance (bind state_size/action_size/etc.
        with functools.partial or a lambda before passing it in)
    env_factory:   () -> environment instance

    Returns:
        {
            "agent": trained agent instance,
            "episode_rewards": [float, ...],
            "episode_steps": [[step_dict, ...], ...] or None,
        }
    """

    if seed is not None:
        np.random.seed(seed)

    agent = agent_factory()
    env = env_factory()

    episode_rewards = []
    episode_steps = [] if collect_steps else None

    for _ in range(episodes):
        total_reward, steps = run_episode(agent, env, max_steps, collect_steps)
        episode_rewards.append(total_reward)
        if collect_steps:
            episode_steps.append(steps)

    return {
        "agent": agent,
        "episode_rewards": episode_rewards,
        "episode_steps": episode_steps,
    }


def run_multi_seed(agent_factory, env_factory, episodes, max_steps, seeds,
                    collect_steps=False):
    """
    Runs run_training once per seed. This is the entry point Commit 3
    (multi-run evaluation / mean ± std) will build on directly.

    Returns a list of result dicts (see run_training), each tagged with
    its seed.
    """

    results = []
    for seed in seeds:
        result = run_training(
            agent_factory, env_factory, episodes, max_steps,
            seed=seed, collect_steps=collect_steps,
        )
        result["seed"] = seed
        results.append(result)

    return results
