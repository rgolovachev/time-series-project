import numpy as np
import pandas as pd
import sys, os
from datasetsforecast.m4 import M4
from config import DATA_DIR, N_SERIES, RANDOM_SEED, HORIZON, LAGS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def load_ds():
    data, *_ = M4.load(directory=DATA_DIR, group="Monthly")

    rng = np.random.RandomState(RANDOM_SEED)
    sample = rng.choice(data["unique_id"].unique(), size=N_SERIES, replace=False)
    df = data[data["unique_id"].isin(sample)].copy()
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    return df


def train_test_split(df):
    train_parts = []
    test_parts = []

    for _, group in df.groupby("unique_id"):
        group = group.sort_values("ds")
        train_parts.append(group.iloc[:-HORIZON])
        test_parts.append(group.iloc[-HORIZON:])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)
    return train_df, test_df


def series2dict(df):
    result = dict()
    for uid, group in df.groupby("unique_id"):
        result[uid] = group.sort_values("ds")["y"].values.astype(float)
    return result


def filter_short_series(series_dict):
    min_len = max(LAGS) + HORIZON + 1
    filtered = dict()
    for key, value in series_dict.items():
        if len(value) >= min_len:
            filtered[key] = value

    return filtered
