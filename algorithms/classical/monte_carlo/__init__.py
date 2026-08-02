"""
Exposes MonteCarloAgent and MonteCarloConfig at the package level so
the Trainer / notebooks can do:

    from algorithms.classical.monte_carlo import MonteCarloAgent, MonteCarloConfig
"""

from .agent import MonteCarloAgent
from .config import MonteCarloConfig

__all__ = ["MonteCarloAgent", "MonteCarloConfig"]