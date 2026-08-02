"""
config.py

Configuration for the Monte Carlo agent. Pulls shared hyperparameters
and known environment bounds from config/ so results stay comparable
across all algorithms.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from config.algorithm_config import DISCOUNT_FACTOR, EPSILON, EPSILON_DECAY, MIN_EPSILON
from config.env_config import MIN_PRESSURE, MAX_PRESSURE, MAX_TANK_LEVEL, EPISODE_LENGTH
from config.training_config import EPISODES, MAX_STEPS, SAVE_INTERVAL


@dataclass
class MonteCarloConfig:
    # Shared across all algorithms (from config/algorithm_config.py)
    gamma: float = DISCOUNT_FACTOR
    epsilon_start: float = EPSILON
    epsilon_min: float = MIN_EPSILON
    epsilon_decay: float = EPSILON_DECAY

    # Monte Carlo specific
    first_visit: bool = True
    max_episode_steps: int = MAX_STEPS       # from training_config.py

    # State discretization — state.py field order:
    # 0 tank_level, 1 pressure, 2 demand, 3 pump_status, 4 valve_status, 5 time_step
    state_bins: int = 5
    # Known bounds from env_config.py; None = auto-infer from data.
    state_bounds: Optional[List[tuple]] = field(default_factory=lambda: [
        (0.0, MAX_TANK_LEVEL),      # tank_level (no MIN given, assume 0)
        (MIN_PRESSURE, MAX_PRESSURE),  # pressure — confirmed exact match
        None,                        # demand — no config value, auto-infer
        (0.0, 1.0),                  # pump_status — binary
        (0.0, 1.0),                  # valve_status — binary
        (0.0, EPISODE_LENGTH - 1),   # time_step
    ])
    # Per-dimension bin overrides: binary flags don't need 5 buckets.
    dim_bins: Optional[List[int]] = field(default_factory=lambda: [5, 5, 5, 2, 2, 5])
    auto_bound_warmup_episodes: int = 5

    # Action space (documentation only — real value comes from Trainer)
    n_actions: int = 6

    save_dir: str = "models/classical/monte_carlo"
    model_filename: str = "monte_carlo_qtable.pkl"

    seed: Optional[int] = 42
    verbose: bool = False