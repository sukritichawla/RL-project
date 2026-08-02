import random

from state import WaterState


class WaterSimulator:

    def __init__(self):

        self.time_step = 0

    def reset(self):

        self.time_step = 0

        return WaterState(
            tank_level=80,
            pressure=60,
            demand=30,
            pump_status=1,
            valve_status=1,
            time_step=0
        )

    def simulate(self, action):

        self.time_step += 1

        tank = random.randint(20, 100)

        pressure = random.randint(30, 80)

        demand = random.randint(10, 60)

        pump = random.randint(0, 1)

        valve = random.randint(0, 1)

        return WaterState(
            tank_level=tank,
            pressure=pressure,
            demand=demand,
            pump_status=pump,
            valve_status=valve,
            time_step=self.time_step
        )