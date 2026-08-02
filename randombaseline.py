# random_baseline.py
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "environment"))
from environment.water_env import WaterEnvironment
import numpy as np

env = WaterEnvironment()
rng = np.random.default_rng(0)
rewards = []
for ep in range(100):
    state = env.reset()
    total = 0
    done = False
    while not done:
        action = rng.integers(0, 6)
        state, reward, done, _ = env.step(action)
        total += reward
    rewards.append(total)

print("Random policy avg reward:", sum(rewards) / len(rewards))