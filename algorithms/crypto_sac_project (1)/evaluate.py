"""Backtest a trained model and compare against buy-and-hold."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from config import INITIAL_CASH, TRADING_FEE, WINDOW_SIZE, MODEL_DIR, RESULTS_DIR
from environment import CryptoTradingEnv


def metrics(equity, periods_per_year=365):
    equity = np.asarray(equity, dtype=float)
    rets = np.diff(equity) / np.maximum(equity[:-1], 1e-12)
    total_return = equity[-1] / equity[0] - 1
    if len(rets) > 1 and np.std(rets, ddof=1) > 0:
        sharpe = np.sqrt(periods_per_year) * np.mean(rets) / np.std(rets, ddof=1)
    else:
        sharpe = 0.0
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / np.maximum(peaks, 1e-12) - 1
    max_dd = float(np.min(drawdowns))
    return {
        "Final portfolio value": float(equity[-1]),
        "Total return": float(total_return),
        "Annualized Sharpe (approx.)": float(sharpe),
        "Maximum drawdown": max_dd,
    }


def run_backtest(test_df, feature_cols):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    env = CryptoTradingEnv(
        test_df, feature_cols, initial_cash=INITIAL_CASH,
        trading_fee=TRADING_FEE, window_size=WINDOW_SIZE
    )
    model_path = os.path.join(MODEL_DIR, "sac_crypto")
    model = SAC.load(model_path)

    obs, _ = env.reset()
    done = False
    equity = [INITIAL_CASH]
    allocations = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        equity.append(info["portfolio_value"])
        allocations.append(info["allocation"])

    # Buy-and-hold on the same test period, starting at the first test close.
    prices = test_df["Close"].to_numpy(dtype=float)
    prices = prices[WINDOW_SIZE:]
    if len(prices) < 2:
        raise ValueError("Test period is too short for backtesting.")
    bh_equity = INITIAL_CASH * prices / prices[0]
    # Align equity curves to common length.
    n = min(len(equity), len(bh_equity))
    equity = np.asarray(equity[:n])
    bh_equity = np.asarray(bh_equity[:n])

    print("\n=== SAC strategy ===")
    for k, v in metrics(equity).items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) and abs(v) < 10 else f"{k}: {v:,.2f}")
    print("\n=== Buy-and-hold benchmark (fees excluded) ===")
    for k, v in metrics(bh_equity).items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) and abs(v) < 10 else f"{k}: {v:,.2f}")

    plt.figure(figsize=(11, 5))
    plt.plot(equity, label="SAC")
    plt.plot(bh_equity, label="Buy and hold")
    plt.title("BTC-USD test-period equity curve")
    plt.xlabel("Test step (daily)")
    plt.ylabel("Portfolio value ($)")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "equity_curve.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved plot: {plot_path}")

    out = pd.DataFrame({
        "step": np.arange(n),
        "sac_equity": equity,
        "buy_hold_equity": bh_equity,
    })
    out.to_csv(os.path.join(RESULTS_DIR, "backtest.csv"), index=False)
    print(f"Saved results: {os.path.join(RESULTS_DIR, 'backtest.csv')}")


if __name__ == "__main__":
    from data import download_data, add_features
    from config import TRAIN_RATIO
    df = download_data()
    featured, cols = add_features(df)
    split = int(len(featured) * TRAIN_RATIO)
    test_df = featured.iloc[split:].copy()
    run_backtest(test_df, cols)
