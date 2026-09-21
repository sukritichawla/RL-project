import random
import numpy as np

from environment.state import WaterState
from environment.reward import calculate_reward


class ContinuousWaterEnvironment:

    def __init__(self):
        self.state = None
        self.time_step = 0

    def reset(self):

        self.time_step = 0

        self.state = WaterState(
            tank_level=80,
            pressure=60,
            demand=30,
            pump_status=1,
            valve_status=1,
            time_step=0
        )

        return self.get_normalized_state()

    def step(self, pump_speed):

        self.time_step += 1

        # Keep pump speed in [0, 1]
        pump_speed = float(
            np.clip(pump_speed, 0.0, 1.0)
        )

        tank = self.state.tank_level
        pressure = self.state.pressure
        demand = self.state.demand

        # Pump speed continuously affects tank level
        tank += 6 * pump_speed

        # Water consumption through open valve
        tank -= 4

        # Small random disturbance
        tank += random.uniform(-2, 2)

        tank = np.clip(
            tank,
            0,
            100
        )

        # Pressure depends continuously on pump speed
        target_pressure = 40 + 30 * pump_speed

        pressure += (
            target_pressure - pressure
        ) * 0.3

        pressure += random.uniform(-2, 2)

        pressure = np.clip(
            pressure,
            0,
            100
        )

        # Demand changes over time
        demand += random.uniform(-5, 5)

        demand = np.clip(
            demand,
            10,
            60
        )

        # Consider pump active when speed > 0
        pump_status = int(
            pump_speed > 0.05
        )

        self.state = WaterState(
            tank_level=float(tank),
            pressure=float(pressure),
            demand=float(demand),
            pump_status=pump_status,
            valve_status=1,
            time_step=self.time_step
        )

        reward = calculate_reward(
            self.state
        )

        done = self.time_step >= 200

        return (
            self.get_normalized_state(),
            reward,
            done,
            {
                "pump_speed": pump_speed
            }
        )

    def get_normalized_state(self):

        return np.array([
            self.state.tank_level / 100.0,
            self.state.pressure / 100.0,
            self.state.demand / 60.0,
            self.state.pump_status,
            self.state.valve_status,
            self.state.time_step / 200.0
        ], dtype=np.float32)