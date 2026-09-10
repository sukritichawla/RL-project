"""
Fixed Energy-Aware Reward

Extends the existing water-distribution reward by adding
a fixed penalty for pump energy consumption.
"""

from environment.reward import calculate_reward
from energy import calculate_energy


# Fixed energy penalty coefficient
LAMBDA = 0.5


def calculate_fixed_energy_reward(state, pump_speed=1.0):
    """
    Calculate the fixed energy-aware reward.

    Args:
        state: Current WaterState object.
        pump_speed (float): Normalized pump speed.

    Returns:
        tuple:
            reward (float): Final reward.
            energy (float): Energy consumed during the step.
    """

    # Existing water-distribution reward
    water_reward = calculate_reward(state)

    # Calculate pump energy
    energy = calculate_energy(
        state.pump_status,
        pump_speed
    )

    # Fixed energy penalty
    energy_penalty = LAMBDA * energy

    # Final reward
    reward = water_reward - energy_penalty

    return reward, energy