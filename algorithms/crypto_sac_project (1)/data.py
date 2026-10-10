"""Download and prepare historical OHLCV data."""
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from config import SYMBOL, START_DATE, END_DATE, INTERVAL


def download_data(path="data/btc_usd.csv", refresh=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and not refresh:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        df = yf.download(
            SYMBOL, start=START_DATE, end=END_DATE, interval=INTERVAL,
            auto_adjust=True, progress=False
        )
        if df.empty:
            raise RuntimeError(
                "Yahoo Finance returned no data. Check your internet connection, "
                "symbol, and date range."
            )
        # yfinance may return MultiIndex columns depending on version.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        wanted = ["Open", "High", "Low", "Close", "Volume"]
        missing = [c for c in wanted if c not in df.columns]
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")
        df = df[wanted].copy()
        df.to_csv(path)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 250:
        raise RuntimeError(f"Only {len(df)} rows available; need at least 250.")
    return df


def add_features(df):
    """Create causal features. No future prices are used."""
    out = df.copy()
    close = out["Close"].astype(float)
    out["return_1d"] = close.pct_change()
    out["return_5d"] = close.pct_change(5)
    out["volatility_10d"] = out["return_1d"].rolling(10).std()
    out["sma_ratio_10"] = close / close.rolling(10).mean() - 1
    out["sma_ratio_30"] = close / close.rolling(30).mean() - 1

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi_14"] = (100 - 100 / (1 + rs)) / 100.0
    out["volume_change"] = out["Volume"].pct_change().replace([np.inf, -np.inf], np.nan)
    feature_cols = [
        "return_1d", "return_5d", "volatility_10d",
        "sma_ratio_10", "sma_ratio_30", "rsi_14", "volume_change"
    ]
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan)
    out = out.dropna().copy()
    return out, feature_cols
