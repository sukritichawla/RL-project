
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

observation = env.reset()

print("Mario environment created successfully!")
print("Observation shape:", observation.shape)
print("Available actions:", SIMPLE_MOVEMENT)

env.close()
