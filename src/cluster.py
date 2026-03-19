import numpy as np
import sys, os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf
from config import SEASON_LENGTH, MIN_CLUSTERS, MAX_CLUSTERS, RANDOM_SEED
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def cluster_series(series_dict):
    feat = get_features(series_dict)
    feat_cols = [c for c in feat.columns if c != "unique_id"]

    scaler = StandardScaler()
    X = scaler.fit_transform(feat[feat_cols].values)

    best_k = get_best_k(feat)
    km = KMeans(n_clusters=best_k, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(X)

    feat["cluster"] = labels
    cluster = feat[["unique_id", "cluster"]].copy()

    print(f"best k is {best_k}")
    print(feat.groupby("cluster")[feat_cols].mean().round(3).to_string())

    return cluster, feat

def get_features(series_dict):
    rows = []
    for uid, y in series_dict.items():
        trend_str, seas_str = stl(y, SEASON_LENGTH)
        mu = np.mean(y)
        sigma = np.std(y)
        cv = 0.0
        if abs(mu) > 1e-9:
            cv = sigma / mu

        acf_vals = acf(y, nlags=SEASON_LENGTH, fft=True)

        acf1 = 0.0
        if len(acf_vals) > 1:
            acf1 = float(acf_vals[1])

        acf_seasonal = 0.0
        if len(acf_vals) > SEASON_LENGTH:
            acf_seasonal = float(acf_vals[SEASON_LENGTH])

        diff_y = np.diff(y)
        diff_acf = acf(diff_y, nlags=1, fft=True)
        diff_acf1 = 0.0
        if len(diff_acf) > 1:
            diff_acf1 = float(diff_acf[1])

        rows.append({
            "unique_id": uid,
            "mean": mu,
            "std": sigma,
            "cv": cv,
            "trend_strength": trend_str,
            "seasonality_strength": seas_str,
            "acf1": acf1,
            "acf_seasonal": acf_seasonal,
            "diff_acf1": diff_acf1,
            "skewness": float(pd.Series(y).skew()),
            "kurtosis": float(pd.Series(y).kurtosis()),
            "length": len(y),
        })

    return pd.DataFrame(rows)

def stl(y, period):
    res = STL(y, period=period, robust=True).fit()
    resid_var = np.var(res.resid)
    detrended_var = np.var(y - res.trend)
    deseasoned_var = np.var(y - res.seasonal)
    trend_str = max(0.0, 1.0 - resid_var / max(deseasoned_var, 1e-9))
    seas_str = max(0.0, 1.0 - resid_var / max(detrended_var, 1e-9))
    return trend_str, seas_str

def get_best_k(feat):
    feat_cols = [c for c in feat.columns if c != "unique_id"]
    X = StandardScaler().fit_transform(feat[feat_cols].values)

    best_k = MIN_CLUSTERS
    best_score = -1.0
    for k in range(MIN_CLUSTERS, MAX_CLUSTERS + 1):
        labels = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10).fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k
