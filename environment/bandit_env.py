import numpy as np


class WaterBanditEnvironment:
    """
    Multi-Armed Bandit environment for urban water distribution.

    Each arm represents a different pump operating level.
    The agent repeatedly chooses a pump-control level and receives
    a water-management reward based on:

    - Tank level
    - Pipeline pressure
    - Water demand satisfaction
    - Pump energy consumption
    """

    def __init__(self):
        # Six possible pump-control levels
        self.n_arms = 6

        self.arm_names = [
            "Pump OFF",
            "Very Low Pumping",
            "Low Pumping",
            "Moderate Pumping",
            "High Pumping",
            "Maximum Pumping"
        ]

        # Pump flow produced by each arm
        self.pump_flow = np.array([
            10,
            25,
            40,
            55,
            70,
            85
        ], dtype=np.float32)

        # Target operating ranges
        self.target_tank_level = 60.0
        self.min_tank_level = 30.0

        self.min_pressure = 40.0
        self.max_pressure = 70.0

        self.last_state = None
        self.last_reward = 0.0

    def reset(self):
        """
        Reset the bandit experiment.

        A bandit does not maintain a sequential water-system state.
        Therefore, each pull generates a new water-system condition.
        """

        self.last_state = None
        self.last_reward = 0.0

        return np.zeros(1, dtype=np.float32)

    def step(self, action):

        if action < 0 or action >= self.n_arms:
            raise ValueError("Invalid pump-control action")

        pump_flow = self.pump_flow[action]

        # Simulated urban water demand
        demand = np.random.uniform(35, 80)

        # Simulated initial tank condition
        initial_tank = np.random.uniform(35, 75)

        # Pump operation increases available tank level
        tank_level = initial_tank + (pump_flow * 0.30)

        # Limit tank level to realistic range
        tank_level = min(tank_level, 100.0)

        # Pressure depends on pump operation
        pressure = 30 + (pump_flow * 0.50)

        # Add small stochastic variation
        pressure += np.random.normal(0, 3)

        # Keep pressure within reasonable range
        pressure = max(0, min(100, pressure))

        # --------------------------------------------------
        # WATER DEMAND SATISFACTION
        # --------------------------------------------------

        available_water = tank_level

        if available_water >= demand:
            demand_satisfied = True
        else:
            demand_satisfied = False

        # --------------------------------------------------
        # REWARD CALCULATION
        # --------------------------------------------------

        reward = 0.0

        # 1. Tank level
        if tank_level >= self.min_tank_level:
            reward += 20
        else:
            reward -= 20

        # 2. Pipeline pressure
        if self.min_pressure <= pressure <= self.max_pressure:
            reward += 25
        else:
            reward -= 25

        # 3. Demand satisfaction
        if demand_satisfied:
            reward += 30
        else:
            reward -= 30

        # 4. Energy consumption penalty
        energy_penalty = pump_flow * 0.20
        reward -= energy_penalty

        # Store water-system information
        self.last_state = {
            "tank_level": tank_level,
            "pressure": pressure,
            "demand": demand,
            "pump_flow": pump_flow,
            "demand_satisfied": demand_satisfied
        }

        self.last_reward = reward

        # A bandit has no sequential terminal state.
        done = False

        # Dummy observation because this is a stateless bandit
        observation = np.zeros(1, dtype=np.float32)

        return observation, reward, done, {}

    def get_optimal_arm(self):
        """
        This is determined empirically by the learning algorithms.
        This function is provided only for displaying the best
        observed/known arm after defining the environment.
        """

        return None

    def get_arm_name(self, arm):
        return self.arm_names[arm]

    def get_last_state(self):
        return self.last_state