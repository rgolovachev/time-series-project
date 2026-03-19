import sys, os
from config import LAGS, ROLLING_WINDOWS, SEASON_LENGTH
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MAX_LAG = max(LAGS)
FEATURE_NAMES = [f"lag_{l}" for l in LAGS] + [f"rolling_mean_{w}" for w in ROLLING_WINDOWS] + [f"rolling_std_{w}" for w in ROLLING_WINDOWS] + ["season_pos"]

def build_ds(series_dict,):
    X_rows = []
    y_rows = []
    uid_rows = []
    for uid, values in series_dict.items():
        for t in range(MAX_LAG, len(values)):
            row = build_row(values, t)
            if row is not None:
                X_rows.append(row)
                y_rows.append(values[t])
                uid_rows.append(uid)
    return np.array(X_rows, dtype=np.float64), np.array(y_rows, dtype=np.float64), uid_rows

def build_row(values, t):
    if t < MAX_LAG:
        return None
    row = [values[t - lag] for lag in LAGS]
    for w in ROLLING_WINDOWS:
        window = values[t - w:t]
        row.append(float(np.mean(window)))
        row.append(float(np.std(window)))
    row.append(t % SEASON_LENGTH)
    return row

def predict_recursive(model, history, horizon, start_time_idx):
    history = list(history)
    preds = []
    for h in range(horizon):
        t = len(history)
        row = []
        for lag in LAGS:
            row.append(history[t - lag])
        for w in ROLLING_WINDOWS:
            window = history[t - w : t]
            row.append(float(np.mean(window)))
            row.append(float(np.std(window)))
        row.append((start_time_idx + h) % SEASON_LENGTH)
        y_hat = float(model.predict([row])[0])
        preds.append(y_hat)
        history.append(y_hat)
    return np.array(preds)
