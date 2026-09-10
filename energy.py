"""
Energy consumption model for the water distribution simulator.

This is a simple simulation-level model.
EPANET integration will be handled separately later.
"""


# Energy model parameters
PUMP_POWER = 10.0       # Power consumed by pump when running
TIME_STEP = 1.0         # Duration of one simulation step


def calculate_energy(pump_status, pump_speed=1.0):
    """
    Calculate energy consumed by the pump during one time step.

    Args:
        pump_status (int): 1 if pump is ON, 0 if pump is OFF.
        pump_speed (float): Normalized pump speed.

    Returns:
        float: Energy consumed during this step.
    """

    if pump_status == 0:
        return 0.0

    energy = PUMP_POWER * pump_speed * TIME_STEP

    return energy