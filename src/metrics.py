import numpy as np

def mase(y_true, y_pred, y_train, season_length):
    naive_errors = np.abs(y_train[season_length:] - y_train[:-season_length])
    scale = np.mean(naive_errors)
    if scale < 1e-9:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / scale)

def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred))
    mask = denom > 0
    return float(np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)
