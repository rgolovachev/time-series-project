import os
import time
import warnings
import pandas as pd
from tqdm import tqdm
from config import TRANSFORMATIONS, RESULTS_DIR
from src.data import load_ds, train_test_split, series2dict, filter_short_series
from src.cluster import cluster_series
from src.models import run_baselines, eval_baselines, run_catboost_expr

warnings.filterwarnings("ignore")

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()

    print("dataset is loading...")
    df = load_ds()
    print(f"sampled {df['unique_id'].nunique()} series")
    print(f"total rows: {len(df)}")

    train_df, test_df = train_test_split(df)
    train_dict = series2dict(train_df)
    test_dict = series2dict(test_df)

    train_dict = filter_short_series(train_dict)
    valid_ids = set(train_dict.keys())
    test_dict = {k: v for k, v in test_dict.items() if k in valid_ids}
    train_df = train_df[train_df["unique_id"].isin(valid_ids)].reset_index(drop=True)

    print(f"{len(train_dict)} series remained after filtering")
    print()

    print("clustering series...")
    cluster_df, features_df = cluster_series(train_dict)
    features_df.to_csv(os.path.join(RESULTS_DIR, "cluster_features.csv"), index=False)
    print()

    print("running statistical baselines...")
    baseline_forecasts = run_baselines(train_df)
    baseline_metrics = eval_baselines(baseline_forecasts, test_dict, train_dict)
    print(f"baselines done: {baseline_metrics['model'].unique().tolist()}")

    catboost_results = []
    for tr_name in tqdm(TRANSFORMATIONS, desc="CatBoost transforms"):
        print()
        print(f"  Training CatBoost with transformation: {tr_name}")
        res = run_catboost_expr(train_dict, test_dict, tr_name)
        catboost_results.append(res)
    catboost_metrics = pd.concat(catboost_results, ignore_index=True)

    all_metrics = pd.concat([baseline_metrics, catboost_metrics], ignore_index=True)
    all_metrics = all_metrics.merge(cluster_df, on="unique_id", how="left")
    out_path = os.path.join(RESULTS_DIR, "experiment_results.csv")
    all_metrics.to_csv(out_path, index=False)

    elapsed = time.time() - t0
    print()
    print(f"elapsed time: {elapsed:.1f}s")
    print()

    print("===  RESULTS  ===")
    summary = (
        all_metrics
        .groupby(["model", "transformation"])[["smape", "mase"]]
        .mean()
        .round(3)
    )
    print(summary.to_string())

if __name__ == "__main__":
    main()
