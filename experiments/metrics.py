"""
Algorithm-independent metrics collection.

Built directly from the runner's per-step logs (raw, undiscretized state
arrays: [tank_level, pressure, demand, pump_status, valve_status,
time_step]), plus environment/reward.py and energy.py's own thresholds --
nothing here is guessed:

    - "good pressure" band (40-70) and the tank-level threshold (>30) are
      lifted verbatim from environment/reward.py's own conditions, so a
      "violation" here means exactly what the reward function already
      penalizes. If those literals ever change in reward.py, update the
      constants below to match.
    - "demand satisfied" reuses reward.py's own condition
      (demand <= tank_level).
    - energy uses energy.calculate_energy(pump_status, pump_speed=1.0);
      pump_speed is fixed at 1.0 because the simulator has no pump_speed
      state field yet (INCREASE/DECREASE_PUMP_SPEED currently just moves
      pressure directly).

Works on the `episode_steps` produced by experiments.runner.run_training
(collect_steps=True) -- it does not need to know anything about which
algorithm produced them.
"""

import numpy as np

from energy import calculate_energy

# Thresholds mirrored from environment/reward.py -- keep in sync with it.
PRESSURE_MIN = 40
PRESSURE_MAX = 70
TANK_LOW_THRESHOLD = 30

TANK_IDX, PRESSURE_IDX, DEMAND_IDX, PUMP_IDX, VALVE_IDX, TIME_IDX = range(6)


# ---------------------------------------------------------------------------
# Per-episode metrics
# ---------------------------------------------------------------------------

def episode_metrics(steps):
    """
    Computes one dict of metrics for a single episode's step log
    (as produced by experiments.runner.run_episode/run_training).
    """

    if not steps:
        return {
            "length": 0,
            "total_reward": 0.0,
            "total_energy": 0.0,
            "avg_energy": 0.0,
            "demand_satisfaction_rate": 0.0,
            "shortage_count": 0,
            "pressure_violations": 0,
            "tank_violations": 0,
            "pump_switches": 0,
        }

    total_reward = 0.0
    total_energy = 0.0
    satisfied_count = 0
    shortage_count = 0
    pressure_violations = 0
    tank_violations = 0
    pump_switches = 0

    for step in steps:
        raw_before = step["raw_state"]
        raw_after = step["raw_next_state"]

        total_reward += step["reward"]

        # Energy consumed during this step: driven by the pump status
        # that resulted from this step's action (matches how the
        # simulator uses the *new* pump status when computing dynamics
        # for the same step).
        pump_status_after = int(raw_after[PUMP_IDX])
        total_energy += calculate_energy(pump_status_after, pump_speed=1.0)

        # Demand satisfaction / shortage (reward.py's own definition).
        demand = raw_after[DEMAND_IDX]
        tank = raw_after[TANK_IDX]
        if demand <= tank:
            satisfied_count += 1
        else:
            shortage_count += 1

        # Constraint violations (reward.py's own bands).
        pressure = raw_after[PRESSURE_IDX]
        if not (PRESSURE_MIN <= pressure <= PRESSURE_MAX):
            pressure_violations += 1
        if tank <= TANK_LOW_THRESHOLD:
            tank_violations += 1

        # Pump switching: did pump status actually change this step?
        pump_status_before = int(raw_before[PUMP_IDX])
        if pump_status_before != pump_status_after:
            pump_switches += 1

    length = len(steps)

    return {
        "length": length,
        "total_reward": total_reward,
        "total_energy": total_energy,
        "avg_energy": total_energy / length,
        "demand_satisfaction_rate": 100.0 * satisfied_count / length,
        "shortage_count": shortage_count,
        "pressure_violations": pressure_violations,
        "tank_violations": tank_violations,
        "pump_switches": pump_switches,
    }


# ---------------------------------------------------------------------------
# Convergence speed
# ---------------------------------------------------------------------------

def moving_average(values, window):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def convergence_episode(episode_rewards, window, tolerance=0.05):
    """
    First episode index after which the moving-average reward stays
    within `tolerance` (relative) of its final value for the rest of
    training. Robust to curves that end up decreasing rather than
    increasing -- "converged" just means "stopped changing much",
    which is what matters for comparing algorithms' learning speed.

    Returns len(episode_rewards) - 1 (i.e. "never stabilized" within
    tolerance) if no such point exists.
    """

    ma = moving_average(episode_rewards, window)
    if len(ma) == 0:
        return 0

    final_value = ma[-1]
    band = tolerance * abs(final_value) if final_value != 0 else tolerance

    for i in range(len(ma)):
        if np.all(np.abs(ma[i:] - final_value) <= band):
            # ma[i] corresponds to episode i + window - 1 in the original series
            return i + window - 1

    return len(episode_rewards) - 1


# ---------------------------------------------------------------------------
# Per-run (one seed) aggregation
# ---------------------------------------------------------------------------

def run_metrics(training_result, moving_average_window):
    """
    Aggregates metrics across all episodes of a single training run
    (i.e. a single seed), as returned by experiments.runner.run_training
    with collect_steps=True.
    """

    episode_rewards = training_result["episode_rewards"]
    episode_steps = training_result["episode_steps"]

    per_episode = [episode_metrics(steps) for steps in episode_steps]

    def field(name):
        return np.array([m[name] for m in per_episode], dtype=np.float64)

    return {
        "cumulative_reward": float(np.sum(episode_rewards)),
        "avg_reward": float(np.mean(episode_rewards)),
        "best_reward": float(np.max(episode_rewards)),
        "total_energy": float(np.sum(field("total_energy"))),
        "avg_energy": float(np.mean(field("avg_energy"))),
        "demand_satisfaction_rate": float(np.mean(field("demand_satisfaction_rate"))),
        "shortage_count": float(np.sum(field("shortage_count"))),
        "pressure_violations": float(np.sum(field("pressure_violations"))),
        "tank_violations": float(np.sum(field("tank_violations"))),
        "pump_switches": float(np.sum(field("pump_switches"))),
        "convergence_episode": float(
            convergence_episode(episode_rewards, moving_average_window)
        ),
        "episode_completion_rate": 100.0 * float(
            np.mean([m["length"] > 0 for m in per_episode])
        ),
    }
