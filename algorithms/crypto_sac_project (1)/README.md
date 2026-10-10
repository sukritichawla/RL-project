# Crypto Trading with Soft Actor-Critic (SAC)

Educational, historical-data-only RL project. It does **not** connect to an exchange or place orders.

## Requirements
- Python 3.10 or 3.11 recommended
- Internet connection for the first data download

## Setup

### Windows PowerShell
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
python evaluate.py
```

If PowerShell blocks activation, use Command Prompt:
```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
python evaluate.py
```

### macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
python evaluate.py
```

The first run downloads daily BTC-USD candles from Yahoo Finance and caches them at `data/btc_usd.csv`. Training saves the model in `artifacts/sac_crypto.zip`; evaluation writes `results/backtest.csv` and `results/equity_curve.png`.

## Configuration
Edit `config.py` to change:
- `SYMBOL`: e.g. `ETH-USD`
- `TOTAL_TIMESTEPS`: training duration (start with 50,000; more can take longer)
- `TRADING_FEE`: assumed transaction fee
- `WINDOW_SIZE`: retained for environment compatibility; observations currently use the current feature row rather than a full rolling window
- `TRAIN_RATIO`: chronological train/test split

## Important limitations
- This is a simplified educational backtest, not a profitable-strategy guarantee.
- Historical data can be revised or missing; Yahoo Finance availability and API behavior may change.
- The environment assumes trading at a close price and does not model spread, slippage, exchange outages, taxes, funding, liquidity, or order-book depth.
- Evaluation is one held-out chronological period. For serious research, use walk-forward validation, multiple market regimes, robust baselines, sensitivity tests, and realistic execution costs.
- Do not treat the output as financial advice or deploy it with real funds.
