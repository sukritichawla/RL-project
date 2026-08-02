from dataclasses import dataclass
import numpy as np


@dataclass
class WaterState:
    """
    Represents the current state of the water distribution network.
    """

    tank_level: float
    pressure: float
    demand: float
    pump_status: int
    valve_status: int
    time_step: int

    def to_array(self):
        """
        Convert state to numpy array for RL algorithms.
        """
        return np.array([
            self.tank_level,
            self.pressure,
            self.demand,
            self.pump_status,
            self.valve_status,
            self.time_step
        ], dtype=np.float32)