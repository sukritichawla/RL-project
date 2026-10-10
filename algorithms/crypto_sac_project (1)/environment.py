"""A simple long-only spot crypto trading environment for Gymnasium."""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class CryptoTradingEnv(gym.Env):
    """
    Observation = market features + current allocation + normalized cash/portfolio.
    Action in [-1, 1] is mapped to target allocation [0, 1].
    Trades at the current row's close and marks the portfolio at the next close.
    This is a simplified educational simulator, not an exchange execution model.
    """
    metadata = {"render_modes": []}

    def __init__(self, data, feature_cols, initial_cash=10_000.0,
                 trading_fee=0.001, window_size=30):
        super().__init__()
        self.data = data.reset_index(drop=False).copy()
        self.feature_cols = list(feature_cols)
        self.initial_cash = float(initial_cash)
        self.trading_fee = float(trading_fee)
        self.window_size = int(window_size)
        if len(self.data) <= self.window_size + 2:
            raise ValueError("Not enough rows for the selected window_size.")
        self.features = self.data[self.feature_cols].to_numpy(dtype=np.float32)
        # Clip features to prevent a few extreme observations dominating learning.
        self.features = np.nan_to_num(self.features, nan=0.0, posinf=0.0, neginf=0.0)
        self.close = self.data["Close"].to_numpy(dtype=np.float64)
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )
        obs_dim = len(self.feature_cols) + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.reset()

    def _obs(self):
        # Current-day features only; position information is included.
        market = np.clip(self.features[self.idx], -10, 10)
        return np.concatenate([
            market,
            np.array([self.allocation, self.cash / max(self.portfolio_value, 1e-8)],
                     dtype=np.float32)
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = self.window_size
        self.cash = self.initial_cash
        self.coin = 0.0
        self.portfolio_value = self.initial_cash
        self.allocation = 0.0
        self.prev_value = self.initial_cash
        self.equity_curve = [self.initial_cash]
        self.trades = 0
        return self._obs(), {}

    def step(self, action):
        action = float(np.asarray(action).reshape(-1)[0])
        target_allocation = float(np.clip((action + 1.0) / 2.0, 0.0, 1.0))
        price = float(self.close[self.idx])

        current_value = self.cash + self.coin * price
        desired_coin_value = current_value * target_allocation
        current_coin_value = self.coin * price
        trade_value = desired_coin_value - current_coin_value

        # Fees are paid on the absolute dollar value traded.
        fee = abs(trade_value) * self.trading_fee
        if trade_value > 0:
            spend = min(trade_value + fee, self.cash)
            if spend > 0:
                units = max(0.0, (spend - fee) / price)
                self.cash -= spend
                self.coin += units
                self.trades += 1
        elif trade_value < 0:
            units_to_sell = min(self.coin, abs(trade_value) / price)
            proceeds = units_to_sell * price
            self.coin -= units_to_sell
            self.cash += proceeds - fee
            self.trades += 1

        self.allocation = (self.coin * price) / max(self.cash + self.coin * price, 1e-8)

        # Move one bar forward; reward reflects next-close portfolio change.
        self.idx += 1
        next_price = float(self.close[self.idx])
        self.portfolio_value = self.cash + self.coin * next_price
        self.portfolio_value = max(self.portfolio_value, 1e-8)
        reward = float(np.log(self.portfolio_value / max(current_value, 1e-8)))
        self.equity_curve.append(self.portfolio_value)

        terminated = self.idx >= len(self.close) - 1
        truncated = False
        info = {
            "portfolio_value": self.portfolio_value,
            "price": next_price,
            "allocation": self.allocation,
            "trades": self.trades,
            "fee_paid": fee,
        }
        return self._obs(), reward, terminated, truncated, info
