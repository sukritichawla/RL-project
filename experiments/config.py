"""
Central experiment configuration.

Every algorithm (Q-Learning, SARSA, Fixed Energy-Aware Q-Learning, AE-Q)
must be evaluated under the exact same conditions, so all shared run
parameters live here rather than being duplicated per-script.
"""

# Number of independent training runs per algorithm (one per seed below).
NUMBER_OF_RUNS = 5

# Training length.
EPISODES = 500
MAX_STEPS = 200  # matches WaterEnvironment's done condition (time_step >= 200)

# One seed per run, reused identically across every algorithm so
# comparisons aren't confounded by different random trajectories.
RANDOM_SEEDS = [0, 1, 2, 3, 4]

assert len(RANDOM_SEEDS) == NUMBER_OF_RUNS, (
    "RANDOM_SEEDS must have exactly NUMBER_OF_RUNS entries"
)

# Window size (in episodes) for moving-average smoothing on learning
# curves and for the convergence-speed metric.
MOVING_AVERAGE_WINDOW = 20

# Discretized state space size produced by WaterEnvironment.discretize_state().
STATE_SIZE = 3

# Restricted action space shared by all four algorithms being compared
# at this stage (0 = PUMP_ON, 1 = PUMP_OFF). See environment/actions.py
# for the full 6-action space — only the first two are used here.
ACTION_SIZE = 2
