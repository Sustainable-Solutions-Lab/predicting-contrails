"""
Hyperparameter tuning for the model.

Loads from the cached pooled feature parquet, samples 2M flights stratified by
year for the search loop, and uses early stopping (so n_estimators self-tunes).

Two phases:
  1. Structured grid over (max_depth, learning_rate, min_child_weight,
     subsample, colsample_bytree). Score by top-10% forcing capture on a
     held-out validation slice of the 2M sample.
  2. Retrain best config on the full 35M training pool, evaluate on full
     17M test set (pooled + by year).

All artifacts written to experiments/outputs/tune_*.{csv,png}.
"""

from __future__ import annotations

import itertools
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Shared feature list + metric helpers.
from common import (
    MODEL_FEATS, TARGET, OUT, RANDOM_SEED, top_k_metrics,
)

CACHE = OUT / "_pool_cache_all.parquet"
TUNE_SAMPLE = 2_000_000           # rows for the search loop
EARLY_STOP = 30                   # rounds without val improvement → halt
N_BOOST_MAX = 1500                # safety cap on n_estimators


# ──────────────────────────────────────────────────────────────────────────
# Search space — structured grid kept small enough to finish in <20 min
# ──────────────────────────────────────────────────────────────────────────
# 24 configs over the params that move the needle for tree boosters on
# large data. Fix subsample (small effect at scale) and colsample (small
# benefit with 14 features) at sensible defaults.
GRID = {
    "max_depth":         [5, 7, 9, 11],
    "learning_rate":     [0.05, 0.1, 0.2],
    "min_child_weight":  [1, 50],
    "subsample":         [0.85],
    "colsample_bytree":  [1.0],
}

# Fixed params shared by every trial
FIXED = dict(
    n_estimators=N_BOOST_MAX,
    n_jobs=-1,
    tree_method="hist",
    enable_categorical=True,
    random_state=RANDOM_SEED,
    early_stopping_rounds=EARLY_STOP,
    eval_metric="rmse",
)


def cap10(model, X, y_log, y_co2_km):
    yp = model.predict(X)
    _, c = top_k_metrics(y_co2_km, yp, k=0.10)
    return c


def main():
    t0 = time.time()
    print(f"Loading cache {CACHE.name} ...")
    pool = pd.read_parquet(CACHE)
    pool["aircraft_type_icao"] = pool["aircraft_type_icao"].astype("category")
    print(f"  {len(pool):,} rows in {time.time()-t0:.1f}s")

    # 2/3 - 1/3 train:test split, stratified by year.
    feat_union = sorted(set(MODEL_FEATS))
    cols_keep = feat_union + [TARGET, "contrail_CO2_km", "year"]
    train, test = train_test_split(
        pool[cols_keep], test_size=1/3,
        random_state=RANDOM_SEED, stratify=pool["year"],
    )
    del pool

    # Tuning subsample: stratify by year inside train
    print(f"\nSampling {TUNE_SAMPLE:,} rows for tuning loop ...")
    tune = train.sample(TUNE_SAMPLE, random_state=RANDOM_SEED)
    tune_train, tune_val = train_test_split(
        tune, test_size=0.2, random_state=RANDOM_SEED, stratify=tune["year"],
    )
    print(f"  tune_train: {len(tune_train):,}   tune_val: {len(tune_val):,}")

    # ── Phase 1: grid search ──────────────────────────────────────────────
    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    print(f"\n[1] Grid search: {len(combos)} configs × early-stop ...")

    rows = []
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        t1 = time.time()
        m = XGBRegressor(**FIXED, **params)
        m.fit(
            tune_train[MODEL_FEATS], tune_train[TARGET],
            eval_set=[(tune_val[MODEL_FEATS], tune_val[TARGET])],
            verbose=False,
        )
        yp = m.predict(tune_val[MODEL_FEATS])
        rmse = math.sqrt(mean_squared_error(tune_val[TARGET], yp))
        r2 = r2_score(tune_val[TARGET], yp)
        cap = cap10(m, tune_val[MODEL_FEATS], tune_val[TARGET],
                    tune_val["contrail_CO2_km"])
        n_used = m.best_iteration + 1
        elapsed = time.time() - t1
        rows.append(dict(**params, n_used=n_used, val_rmse=rmse,
                         val_r2=r2, val_top10_capture=cap, sec=elapsed))
        print(f"  [{i:2d}/{len(combos)}] depth={params['max_depth']} "
              f"lr={params['learning_rate']:<4} mcw={params['min_child_weight']:<3} "
              f"sub={params['subsample']:<4} col={params['colsample_bytree']:<4} "
              f"→ n={n_used:<4} R²={r2:.3f}  cap10={cap:.3f}  ({elapsed:.0f}s)")

    grid = pd.DataFrame(rows).sort_values("val_top10_capture", ascending=False)
    grid.to_csv(OUT / "tune_grid_results.csv", index=False)
    print("\n[1] Top 10 configs by val top-10% capture:")
    print(grid.head(10).to_string(index=False))

    best = grid.iloc[0].to_dict()
    print(f"\n[1] Best: {best}")

    # ── Phase 2: retrain best on full train, evaluate on full test ────────
    print(f"\n[2] Retraining best on full train ({len(train):,} rows) ...")
    best_params = {k: best[k] for k in GRID}
    # Convert numeric-looking keys back to native types
    best_params["max_depth"] = int(best_params["max_depth"])
    best_params["min_child_weight"] = int(best_params["min_child_weight"])
    best_n = int(best["n_used"])

    final_fixed = {**FIXED, "n_estimators": best_n}
    final_fixed.pop("early_stopping_rounds", None)
    final_fixed.pop("eval_metric", None)

    t2 = time.time()
    final = XGBRegressor(**final_fixed, **best_params)
    final.fit(train[MODEL_FEATS], train[TARGET])
    print(f"  fit: {time.time()-t2:.0f}s")

    # Evaluate
    rows_eval = []
    test_2019 = test[test["year"] == 2019]
    test_2021 = test[test["year"] == 2021]
    for split_name, split in [
        ("test_pooled", test),
        ("test_2019_only", test_2019),
        ("test_2021_only", test_2021),
    ]:
        yp = final.predict(split[MODEL_FEATS])
        rmse = math.sqrt(mean_squared_error(split[TARGET], yp))
        r2 = r2_score(split[TARGET], yp)
        rec5, cap5 = top_k_metrics(split["contrail_CO2_km"], yp, k=0.05)
        rec10, cap10v = top_k_metrics(split["contrail_CO2_km"], yp, k=0.10)
        rows_eval.append(dict(
            split=split_name, n=len(split), rmse=rmse, r2=r2,
            top5_recall=rec5, top5_capture=cap5,
            top10_recall=rec10, top10_capture=cap10v,
        ))

    final_metrics = pd.DataFrame(rows_eval)
    final_metrics.to_csv(OUT / "tune_final_metrics.csv", index=False)
    print("\n[2] Final tuned model metrics:")
    print(final_metrics.to_string(index=False))

    # ── Comparison plot: tuned vs default ─────────────────────────────────
    # Pull pre-tune schedule-only numbers from the original metrics.csv if present
    pre = pd.read_csv(OUT / "metrics.csv")
    pre_lean_ac = pre[pre["name"].str.startswith("schedule-only/")].copy()
    pre_lean_ac["split"] = pre_lean_ac["name"].str.replace("schedule-only/", "", regex=False)
    pre_lean_ac["config"] = "default (depth=7, lr=0.1, n=300)"
    final_metrics["config"] = (
        f"tuned (depth={best_params['max_depth']}, "
        f"lr={best_params['learning_rate']}, n={best_n})"
    )
    cmp = pd.concat([
        pre_lean_ac[["split", "config", "top5_capture", "top10_capture", "r2"]],
        final_metrics[["split", "config", "top5_capture", "top10_capture", "r2"]],
    ], ignore_index=True)
    cmp.to_csv(OUT / "tune_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ["top5_capture", "top10_capture", "r2"]):
        sub = cmp.pivot(index="split", columns="config", values=metric)
        sub = sub.reindex(["test_pooled", "test_2019_only", "test_2021_only"])
        sub.plot(kind="bar", ax=ax, color=["#bbbbbb", "#3a7ca5"], width=0.7)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
        ax.set_xlabel("")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "tune_comparison.png", dpi=160)
    plt.close()

    print(f"\nTotal: {time.time()-t0:.0f}s")
    print(f"Outputs: tune_grid_results.csv, tune_final_metrics.csv, "
          f"tune_comparison.{{csv,png}}")


if __name__ == "__main__":
    main()
