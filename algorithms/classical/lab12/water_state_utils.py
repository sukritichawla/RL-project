
# algorithms/classical/lab12/water_state_utils.py

import numpy as np

HORIZON = 200
NUM_BINS = 5
NUM_STATES = 5 * 5 * 5 * 2 * 2 * 5


def _bin(value, low, high, bins=NUM_BINS):
    """Convert a continuous value into a bin index."""
    value = float(value)
    value = max(low, min(high, value))

    if value >= high:
        return bins - 1

    return min(int((value - low) / (high - low) * bins), bins - 1)


def discretize_state(state):
    """
    Accepts a WaterState object and returns an integer in [0, 2499].
    """
    tank = _bin(state.tank_level, 0, 100)
    pressure = _bin(state.pressure, 0, 100)
    demand = _bin(state.demand, 10, 60)

    pump = int(state.pump_status)
    valve = int(state.valve_status)

    timestep = min(
        int(max(0, state.time_step) / (HORIZON / NUM_BINS)),
        NUM_BINS - 1
    )

    # Mixed-radix encoding: six dimensions -> one integer.
    index = tank
    index = index * 5 + pressure
    index = index * 5 + demand
    index = index * 2 + pump
    index = index * 2 + valve
    index = index * 5 + timestep

    return index



def choose_action(q_values, epsilon, rng):
    """Epsilon-greedy action selection; supports Python and NumPy RNGs."""
    if rng.random() < epsilon:
        if hasattr(rng, "integers"):  # NumPy Generator
            return int(rng.integers(len(q_values)))
        return rng.randrange(len(q_values))  # Python random.Random

    best_value = np.max(q_values)
    best_actions = np.flatnonzero(q_values == best_value)

    if hasattr(rng, "integers"):  # NumPy Generator
        return int(rng.choice(best_actions))
    return int(rng.choice(list(best_actions)))

