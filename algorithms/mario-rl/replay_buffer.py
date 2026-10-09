
import random
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append(
            (state, action, reward, next_state, done)
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            states,
            actions,
            rewards,
            next_states,
            dones
        )

    def __len__(self):
        return len(self.buffer)


if __name__ == "__main__":
    memory = ReplayBuffer(capacity=100)

    memory.add("state1", 1, 1.0, "state2", False)
    memory.add("state2", 2, 0.5, "state3", False)

    print("Replay buffer created successfully!")
    print("Stored experiences:", len(memory))

    sample = memory.sample(2)
    print("Sampled actions:", sample[1])
