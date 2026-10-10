"""Train a Soft Actor-Critic agent on historical crypto data."""
import os
import random
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from data import download_data, add_features
from environment import CryptoTradingEnv
from config import (
    INITIAL_CASH, TRADING_FEE, WINDOW_SIZE, TRAIN_RATIO,
    TOTAL_TIMESTEPS, RANDOM_SEED, MODEL_DIR, RESULTS_DIR
)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print("Downloading/loading historical data...")
    raw = download_data()
    featured, feature_cols = add_features(raw)
    split = int(len(featured) * TRAIN_RATIO)
    train_df = featured.iloc[:split].copy()
    # Keep test data separate; training environment never sees it.
    if len(train_df) <= WINDOW_SIZE + 2:
        raise RuntimeError("Training split too short. Reduce WINDOW_SIZE or use more data.")

    train_env = CryptoTradingEnv(
        train_df, feature_cols, INITIAL_CASH, TRADING_FEE, WINDOW_SIZE
    )
    check_env(train_env, warn=True)
    train_env = Monitor(train_env)

    # Eval on a held-out validation slice taken from the end of training data.
    val_start = int(len(train_df) * 0.80)
    fit_df = train_df.iloc[:val_start].copy()
    val_df = train_df.iloc[val_start:].copy()
    if len(val_df) <= WINDOW_SIZE + 2:
        val_df = train_df.copy()
    train_env = Monitor(CryptoTradingEnv(
        fit_df, feature_cols, INITIAL_CASH, TRADING_FEE, WINDOW_SIZE
    ))
    eval_env = Monitor(CryptoTradingEnv(
        val_df, feature_cols, INITIAL_CASH, TRADING_FEE, WINDOW_SIZE
    ))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_DIR, "best"),
        log_path=os.path.join(RESULTS_DIR, "eval_logs"),
        eval_freq=max(5_000, TOTAL_TIMESTEPS // 10),
        deterministic=True,
        render=False,
    )

    print(f"Rows total={len(featured)}, train={len(train_df)}, test={len(featured)-split}")
    print(f"Features: {feature_cols}")
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        seed=RANDOM_SEED,
        device="auto",
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, progress_bar=False)
    model.save(os.path.join(MODEL_DIR, "sac_crypto"))
    print(f"Saved model to {os.path.join(MODEL_DIR, 'sac_crypto.zip')}")
    print("Now run: python evaluate.py")


if __name__ == "__main__":
    main()
