"""
Adaptive Energy-Aware Reward

The energy penalty changes according to the current
water demand and tank level.

High demand / low tank:
    Lower energy penalty so necessary pumping is not discouraged.

Low demand / high tank:
    Higher energy penalty to discourage unnecessary pumping.
"""

from environment.reward import calculate_reward
from energy import calculate_energy


# Base energy penalty
BASE_LAMBDA = 0.5

# Adaptation parameters
ALPHA = 0.5
BETA = 0.5

# Bounds for the adaptive coefficient
MIN_LAMBDA = 0.1
MAX_LAMBDA = 1.0

# Normalization limits
MAX_DEMAND = 60.0
MAX_TANK_LEVEL = 100.0


def calculate_adaptive_lambda(tank_level, demand):
    """
    Calculate the adaptive energy penalty coefficient.

    High tank + low demand:
        Higher energy penalty.

    Low tank + high demand:
        Lower energy penalty.
    """

    demand_ratio = demand / MAX_DEMAND
    tank_ratio = tank_level / MAX_TANK_LEVEL

    lambda_t = BASE_LAMBDA * (
        1 + BETA * tank_ratio - ALPHA * demand_ratio
    )

    lambda_t = max(MIN_LAMBDA, min(MAX_LAMBDA, lambda_t))

    return lambda_t

def calculate_adaptive_reward(state, pump_speed=1.0):
    """
    Calculate the adaptive energy-aware reward.

    Args:
        state: Current WaterState object.
        pump_speed (float): Normalized pump speed.

    Returns:
        tuple:
            reward (float): Final adaptive reward.
            energy (float): Energy consumed.
            lambda_t (float): Adaptive energy coefficient.
    """

    # Existing water-distribution reward
    water_reward = calculate_reward(state)

    # Energy consumed
    energy = calculate_energy(
        state.pump_status,
        pump_speed
    )

    # Calculate adaptive energy coefficient
    lambda_t = calculate_adaptive_lambda(
        state.tank_level,
        state.demand
    )

    # Adaptive energy penalty
    energy_penalty = lambda_t * energy

    # Final reward
    reward = water_reward - energy_penalty

    return reward, energy, lambda_t