import random
from state import WaterState
from actions import Action


class WaterSimulator:

    def __init__(self):
        self.time_step = 0
        self.state = None

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
        return self.state

    def simulate(self, action):
        self.time_step += 1

        tank = self.state.tank_level
        pressure = self.state.pressure
        pump = self.state.pump_status
        valve = self.state.valve_status

        # --- Apply the action's direct effect ---
        if action == Action.PUMP_ON:
            pump = 1
        elif action == Action.PUMP_OFF:
            pump = 0
        elif action == Action.INCREASE_PUMP_SPEED:
            pressure += 5
        elif action == Action.DECREASE_PUMP_SPEED:
            pressure -= 5
        elif action == Action.OPEN_VALVE:
            valve = 1
        elif action == Action.CLOSE_VALVE:
            valve = 0

        # --- Natural dynamics, influenced by action-driven state ---
        # Pump running raises tank level; valve open drains it toward demand.
        tank += (5 if pump == 1 else -3)
        tank += (-4 if valve == 1 else 2)
        tank += random.randint(-3, 3)  # small noise
        tank = max(0, min(100, tank))

        # Pressure drifts toward a pump-dependent target, with noise.
        target_pressure = 65 if pump == 1 else 40
        pressure += (target_pressure - pressure) * 0.3
        pressure += random.randint(-3, 3)
        pressure = max(0, min(100, pressure))

        # Demand: still mostly exogenous (real-world demand isn't
        # controlled by the utility), but bounded realistically.
        demand = max(10, min(60, self.state.demand + random.randint(-5, 5)))

        self.state = WaterState(
            tank_level=tank,
            pressure=pressure,
            demand=demand,
            pump_status=pump,
            valve_status=valve,
            time_step=self.time_step
        )
        return self.state