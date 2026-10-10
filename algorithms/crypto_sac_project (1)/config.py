"""Configuration for the SAC crypto trading demo."""
SYMBOL = "BTC-USD"
START_DATE = "2019-01-01"
END_DATE = None  # None means download through the latest available date
INTERVAL = "1d"

INITIAL_CASH = 10_000.0
TRADING_FEE = 0.001       # 0.1% per traded dollar
WINDOW_SIZE = 30
TRAIN_RATIO = 0.80
TOTAL_TIMESTEPS = 50_000
RANDOM_SEED = 42

MODEL_DIR = "artifacts"
RESULTS_DIR = "results"
