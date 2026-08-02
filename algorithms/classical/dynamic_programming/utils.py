import numpy as np


def initialize_values(num_states):
    return np.zeros(num_states)


def initialize_policy(num_states, num_actions):
    return np.zeros(num_states, dtype=int)