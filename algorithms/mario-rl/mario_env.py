
import numpy as np
from collections import deque
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv


class MarioEnv:
    def __init__(self):
        # Construct the base environment directly; don't use gym.make()
        self.env = SuperMarioBrosEnv()
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)

        # Four grayscale frames, each resized to 84 x 84
        self.frames = deque(maxlen=4)
        self.action_space = self.env.action_space

    def preprocess(self, observation):
        gray = np.mean(observation, axis=2).astype(np.uint8)
        rows = np.linspace(0, gray.shape[0] - 1, 84).astype(int)
        cols = np.linspace(0, gray.shape[1] - 1, 84).astype(int)
        return gray[rows[:, None], cols]

    def reset(self):
        observation = self.env.reset()
        frame = self.preprocess(observation)
        self.frames.clear()

        for _ in range(4):
            self.frames.append(frame)

        return np.stack(self.frames)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(self.preprocess(observation))
        return np.stack(self.frames), reward, done, info

    def close(self):
        self.env.close()


if __name__ == "__main__":
    env = MarioEnv()
    try:
        obs = env.reset()
        print("Environment ready for RL!")
        print("Observation shape:", obs.shape)
        print("Number of actions:", env.action_space.n)

        obs, reward, done, info = env.step(1)
        print("Step successful!")
        print("Reward:", reward)
        print("New observation shape:", obs.shape)
    finally:
        env.close()
