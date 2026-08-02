"""
agent.py
--------
Tabular, first-visit (or every-visit) Monte Carlo control with an
epsilon-greedy behavior policy, as in Sutton & Barto chapter 5.

Uses per-dimension bin counts (dim_bins) and known environment bounds
from config/, auto-inferring only dimensions left as None (e.g. demand,
which has no fixed range in env_config.py).
"""

import os
import pickle
import numpy as np

from algorithms.base_agent import BaseAgent
from .config import MonteCarloConfig


class MonteCarloAgent(BaseAgent):
    def __init__(self, state_size, action_size, config: MonteCarloConfig = None):
        super().__init__(state_size, action_size)
        self.config = config or MonteCarloConfig()

        self.gamma = self.config.gamma
        self.epsilon = self.config.epsilon_start
        self.epsilon_min = self.config.epsilon_min
        self.epsilon_decay = self.config.epsilon_decay
        self.first_visit = self.config.first_visit

        self.rng = np.random.default_rng(self.config.seed)

        # Per-dimension bin counts (e.g. [5,5,5,2,2,5] — binary dims get 2 bins)
        self.dim_bins = self.config.dim_bins or [self.config.state_bins] * self.state_size

        # state_bounds: list where each entry is (low, high) or None (auto-infer)
        raw_bounds = self.config.state_bounds or [None] * self.state_size
        self.state_bounds = list(raw_bounds)
        self._dims_to_infer = [i for i, b in enumerate(self.state_bounds) if b is None]
        self._bounds_finalized = len(self._dims_to_infer) == 0
        for i in self._dims_to_infer:
            self.state_bounds[i] = (0.0, 1.0)  # placeholder until warmup ends

        self._bound_samples = []
        self._episodes_seen = 0

        # Q table shaped from per-dimension bin counts.
        q_shape = tuple(self.dim_bins) + (self.action_size,)
        self.Q = np.zeros(q_shape, dtype=np.float64)
        self.N = np.zeros(q_shape, dtype=np.int64)

        self._episode_buffer = []

    # ======================================================================
    # Bound auto-inference — only for dimensions with no known config bound
    # ======================================================================
    def _update_bound_warmup(self, raw_state):
        if self._bounds_finalized:
            return
        self._bound_samples.append(np.asarray(raw_state, dtype=np.float64).flatten())

    def _finalize_bounds_if_ready(self):
        if self._bounds_finalized:
            return
        if self._episodes_seen >= self.config.auto_bound_warmup_episodes:
            samples = np.stack(self._bound_samples, axis=0)
            for i in self._dims_to_infer:
                lo, hi = samples[:, i].min(), samples[:, i].max()
                if hi == lo:
                    hi = lo + 1.0
                self.state_bounds[i] = (lo, hi)
            self._bounds_finalized = True
            if self.config.verbose:
                print("[MonteCarloAgent] Finalized state bounds:")
                for i, (lo, hi) in enumerate(self.state_bounds):
                    print(f"  dim {i}: ({lo:.2f}, {hi:.2f})  bins={self.dim_bins[i]}")

    # ======================================================================
    # Discretization
    # ======================================================================
    def _discretize(self, raw_state):
        state_array = np.asarray(raw_state, dtype=np.float64).flatten()
        indices = []
        for i, value in enumerate(state_array):
            low, high = self.state_bounds[i]
            n_bins = self.dim_bins[i]
            clipped = min(max(value, low), high)
            ratio = 0.0 if high == low else (clipped - low) / (high - low)
            idx = min(int(ratio * n_bins), n_bins - 1)
            indices.append(idx)
        return tuple(indices)

    # ======================================================================
    # BaseAgent interface
    # ======================================================================
    def select_action(self, state):
        self._update_bound_warmup(state)
        disc_state = self._discretize(state)

        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.action_size))

        q_values = self.Q[disc_state]
        best = np.flatnonzero(q_values == q_values.max())
        return int(self.rng.choice(best))

    def update(self, state, action, reward, next_state, done):
        disc_state = self._discretize(state)
        self._episode_buffer.append((disc_state, action, reward))

        if done:
            self._run_mc_backup()
            self._episode_buffer = []
            self._episodes_seen += 1
            self._finalize_bounds_if_ready()
            self._decay_epsilon()

    def _run_mc_backup(self):
        G = 0.0
        visited = set()
        for t in reversed(range(len(self._episode_buffer))):
            disc_state, action, reward = self._episode_buffer[t]
            G = reward + self.gamma * G
            key = (disc_state, action)
            if self.first_visit and key in visited:
                continue
            visited.add(key)
            idx = disc_state + (action,)
            self.N[idx] += 1
            self.Q[idx] += (G - self.Q[idx]) / self.N[idx]

    def _decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ======================================================================
    # Persistence
    # ======================================================================
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "Q": self.Q, "N": self.N, "epsilon": self.epsilon,
                "state_bounds": self.state_bounds, "dim_bins": self.dim_bins,
                "state_size": self.state_size, "action_size": self.action_size,
            }, f)

    def load(self, filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.Q = data["Q"]
        self.N = data["N"]
        self.epsilon = data["epsilon"]
        self.state_bounds = data["state_bounds"]
        self.dim_bins = data["dim_bins"]
        self._bounds_finalized = True

    # ======================================================================
    # Notebook helpers
    # ======================================================================
    def greedy_action(self, state):
        disc_state = self._discretize(state)
        q_values = self.Q[disc_state]
        best = np.flatnonzero(q_values == q_values.max())
        return int(self.rng.choice(best))

    def get_q_table_stats(self):
        return {
            "epsilon": self.epsilon,
            "visited_state_action_fraction": float(np.mean(self.N > 0)),
            "q_table_shape": self.Q.shape,
        }