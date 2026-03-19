import numpy as np
import pandas as pd
import sys, os
from catboost import CatBoostRegressor
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, AutoTheta, AutoETS
from config import SEASON_LENGTH, HORIZON, CATBOOST_PARAMS
from src.features import build_ds, predict_recursive, MAX_LAG, FEATURE_NAMES
from src.trans import get_transform, DifferencingTransform
from src.metrics import smape, mase
from src.trans import BoxCoxTransform
from scipy import stats as sp_stats


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def run_baselines(train_df):
    sf = StatsForecast(models=[
            Naive(),
            SeasonalNaive(season_length=SEASON_LENGTH),
            AutoTheta(season_length=SEASON_LENGTH),
            AutoETS(season_length=SEASON_LENGTH),
        ],
        freq="MS",
        n_jobs=1,
    )
    sf.fit(train_df)
    forecasts = sf.predict(h=HORIZON)
    forecasts = forecasts.reset_index()
    return forecasts

def eval_baselines(forecasts, test_dict, train_dict):
    model_cols = [c for c in forecasts.columns if c not in ("unique_id", "ds", "index")]
    rows = []
    for uid in test_dict:
        y_true = test_dict[uid]
        y_train = train_dict[uid]
        uid_fcst = forecasts[forecasts["unique_id"] == uid].sort_values("ds")
        for model_name in model_cols:
            y_pred = uid_fcst[model_name].values[:len(y_true)]
            if len(y_pred) < len(y_true):
                continue
            rows.append({
                "unique_id": uid,
                "model": model_name,
                "transformation": "identity",
                "smape": smape(y_true, y_pred),
                "mase": mase(y_true, y_pred, y_train, SEASON_LENGTH),
            })
    return pd.DataFrame(rows)


def transform_dict(series_dict, transform_name):
    fitted = {}
    transformed = {}
    if transform_name == "boxcox":
        lambdas = []
        for uid, y in series_dict.items():
            t = get_transform("boxcox")
            t.fit(y)
            lambdas.append(t.lmbd)
        shared_lambda = float(np.median(lambdas))

        for uid, y in series_dict.items():
            t = BoxCoxTransform()
            t.shift = 0.0
            arr = y.astype(float)
            if np.min(arr) <= 0:
                t.shift = -float(np.min(arr)) + 1.0
            t.lmbd = shared_lambda
            
            bc_vals = sp_stats.boxcox(arr + t.shift, t.lmbd)
            t.bc_min = float(np.min(bc_vals))
            t.bc_max = float(np.max(bc_vals))
            t.orig_max = float(np.max(y))
            transformed[uid] = t.transform(y)
            fitted[uid] = t
    else:
        for uid, y in series_dict.items():
            t = get_transform(transform_name)
            t.fit(y)
            transformed[uid] = t.transform(y)
            fitted[uid] = t

    return transformed, fitted

def run_catboost_expr(train_dict, test_dict, transform_name):
    transformed_train, fitted_transforms = transform_dict(train_dict, transform_name)
    X, y, uid_list = build_ds(transformed_train)

    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y, uid_list = X[mask], y[mask], [u for u, m in zip(uid_list, mask) if m]

    season_idx = FEATURE_NAMES.index("season_pos")
    X[:, season_idx] = X[:, season_idx].astype(int)

    params = {**CATBOOST_PARAMS, "bootstrap_type": "Bernoulli", "subsample": 0.8}
    model = CatBoostRegressor(**params)
    model.fit(X, y)

    rows = []
    for uid in test_dict:
        y_true = test_dict[uid]
        y_train_orig = train_dict[uid]
        t_series = transformed_train[uid]
        transform = fitted_transforms[uid]

        start_idx = len(t_series)
        preds_transformed = predict_recursive(model, t_series, HORIZON, start_idx)
        if transform_name == "differencing":
            last_known = y_train_orig[-1]
            y_pred = transform.inverse_transform_forecast(preds_transformed, last_known)
        else:
            y_pred = transform.inverse_transform(preds_transformed)

        rows.append({
            "unique_id": uid,
            "model": "CatBoost",
            "transformation": transform_name,
            "smape": smape(y_true, y_pred),
            "mase": mase(y_true, y_pred, y_train_orig, SEASON_LENGTH),
        })
    return pd.DataFrame(rows)
