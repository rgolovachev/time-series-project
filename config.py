import os

RANDOM_SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
N_SERIES = 150
SEASON_LENGTH = 12
HORIZON = 18
MIN_CLUSTERS = 2
MAX_CLUSTERS = 10
LAGS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 24]
ROLLING_WINDOWS = [3, 6, 12]
CATBOOST_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.05,
    "depth": 6,
    "verbose": 0,
    "random_seed": RANDOM_SEED,
    "loss_function": "RMSE",
}
TRANSFORMATIONS = ["identity", "log1p", "boxcox", "differencing"]
