from simulator import WaterSimulator
from reward import calculate_reward


class WaterEnvironment:

    def __init__(self):

        self.simulator = WaterSimulator()

        self.state = None

    def reset(self):

        self.state = self.simulator.reset()

        return self.state.to_array()

    def step(self, action):

        next_state = self.simulator.simulate(action)

        reward = calculate_reward(next_state)

        done = next_state.time_step >= 200

        self.state = next_state

        return (
            next_state.to_array(),
            reward,
            done,
            {}
        )

    def render(self):

        print(self.state)

    def close(self):

        pass