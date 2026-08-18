"""
Train the canonical model and emit publication-ready Fig 3c.

Canonical model = LEAN+AC (14 features), tuned hyperparameters from
the Phase-1 grid search:
    max_depth = 11
    learning_rate = 0.05
    min_child_weight = 50
    subsample = 0.85
    n_estimators = 348

Trains on the full 35M-flight pooled training set, saves the fitted
model, computes permutation importance on a 50k test slice, and writes
a clean two-panel Fig 3c (importance bar + Lorenz curve).
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from feature_pruning import (
    DROPBOX_PLOTS, LEAN_AC_FEATS, OUT, RANDOM_SEED, TARGET, top_k_metrics,
)

CACHE = OUT / "_pool_cache_all.parquet"
MODEL_PATH = OUT / "final_model.joblib"

# Locked from Phase-1 grid search
PARAMS = dict(
    max_depth=11,
    learning_rate=0.05,
    min_child_weight=50,
    subsample=0.85,
    colsample_bytree=1.0,
    n_estimators=348,
    n_jobs=-1,
    tree_method="hist",
    enable_categorical=True,
    random_state=RANDOM_SEED,
)

# Human-readable feature names for the figure
PRETTY = {
    "total_flight_distance_km": "Flight distance",
    "OriginLat": "Origin latitude",
    "DestinationLat": "Destination latitude",
    "OriginLon_sin": "Origin longitude (sin)",
    "OriginLon_cos": "Origin longitude (cos)",
    "DestinationLon_sin": "Destination longitude (sin)",
    "DestinationLon_cos": "Destination longitude (cos)",
    "day_sin": "Day of year (sin)",
    "day_cos": "Day of year (cos)",
    "start_hour_sin": "Departure hour (sin)",
    "start_hour_cos": "Departure hour (cos)",
    "land_score": "Over-land fraction",
    "night_score_full_0": "Night fraction (sun-weighted)",
    "aircraft_type_icao": "Aircraft type (ICAO)",
}


def build_fig3c(perm: pd.DataFrame, co2_km: np.ndarray, yp_full: np.ndarray):
    """Two-panel Fig 3c: Lorenz capture curve + permutation importance."""
    order_pred = np.argsort(yp_full)[::-1]
    order_true = np.argsort(co2_km)[::-1]
    n = len(co2_km)
    pos_total = np.maximum(co2_km, 0).sum()

    def cum_capture(order):
        sorted_pos = np.maximum(co2_km[order], 0)
        return np.cumsum(sorted_pos) / pos_total

    cap_pred = cum_capture(order_pred)
    cap_true = cum_capture(order_true)
    pct_x = np.linspace(1 / n, 100, n)

    # Sub-sample x to 1000 points for plotting
    idx_sub = np.linspace(0, n - 1, 1000).astype(int)

    # wspace must leave room for the right panel's long feature names,
    # which are drawn to the LEFT of the bars (they collided with the
    # left panel's legend at 0.25).
    fig = plt.figure(figsize=(12.5, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.55)

    # (left) Lorenz curve
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(pct_x[idx_sub], cap_true[idx_sub] * 100,
             lw=2.4, color="#1f4e79", label="Perfect knowledge")
    ax0.plot(pct_x[idx_sub], cap_pred[idx_sub] * 100,
             lw=2.4, color="#d1495b", label="LEAN+AC predictions")
    ax0.plot([0, 100], [0, 100], lw=1.5, color="#888888", ls="--",
             label="Random ordering")

    # Annotations: show capture at 5% and 10%
    for k, color in [(0.05, "#d1495b"), (0.10, "#d1495b")]:
        x_pct = k * 100
        idx = int(round(k * n)) - 1
        y_pct = cap_pred[idx] * 100
        ax0.plot([x_pct, x_pct], [0, y_pct], color=color, ls=":", lw=1.0)
        ax0.plot([0, x_pct], [y_pct, y_pct], color=color, ls=":", lw=1.0)
        ax0.scatter([x_pct], [y_pct], color=color, zorder=5, s=22)
        ax0.annotate(f"{y_pct:.0f}% at top {x_pct:.0f}%",
                     xy=(x_pct, y_pct), xytext=(x_pct + 6, y_pct - 6),
                     fontsize=10, color=color)

    ax0.set_xlim(0, 100)
    ax0.set_ylim(0, 100)
    ax0.set_xlabel("Flights ranked by predicted forcing (cumulative %)")
    ax0.set_ylabel("Positive contrail forcing captured (cumulative %)")
    ax0.legend(loc="lower right", frameon=False, fontsize=10)
    ax0.set_title("Concentration of contrail forcing in predicted top flights")
    ax0.grid(alpha=0.25)

    # (right) Permutation importance — GROUPED: each cyclic sin/cos pair
    # is shuffled jointly and shown as one bar, so a variable's
    # importance isn't diluted across its two encoding columns.
    ax1 = fig.add_subplot(gs[0, 1])
    bars = ax1.barh(perm["pretty"], perm["importance_mean"],
                    xerr=perm["importance_std"],
                    color="#3a7ca5", ecolor="#777777")
    # Highlight the four "headline rule" variables in a different color
    headline = {"Flight distance", "Origin latitude",
                "Night fraction (sun-weighted)", "Aircraft type (ICAO)"}
    for bar, name in zip(bars, perm["pretty"]):
        if name in headline:
            bar.set_color("#d1495b")
    ax1.set_xlabel("Permutation importance (Δ R² when shuffled)")
    ax1.set_title("Feature importance — tuned LEAN+AC")
    ax1.grid(alpha=0.25, axis="x")

    plt.savefig(DROPBOX_PLOTS / "fig3c.png", dpi=200, bbox_inches="tight")
    plt.savefig(DROPBOX_PLOTS / "fig3c.pdf", bbox_inches="tight")
    plt.savefig(DROPBOX_PLOTS / "fig3c.eps", bbox_inches="tight")
    plt.close()
    print(f"\nWrote fig3c.{{png,pdf,eps}} to {DROPBOX_PLOTS}")


def replot():
    """Rebuild Fig 3c from the saved model + permutation CSV — no retraining."""
    t0 = time.time()
    print("Plot-only mode: loading saved model and permutation CSV ...")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]

    grouped = OUT / "final_permutation_importance_grouped.csv"
    if grouped.exists():
        perm = pd.read_csv(grouped).rename(columns={"group": "pretty"})
    else:
        perm = pd.read_csv(OUT / "final_permutation_importance.csv")
        perm["pretty"] = perm["feature"].map(PRETTY)
    perm = perm.sort_values("importance_mean", ascending=True)

    cols_keep = LEAN_AC_FEATS + [TARGET, "contrail_CO2_km", "year"]
    pool = pd.read_parquet(CACHE, columns=cols_keep)
    pool["aircraft_type_icao"] = pool["aircraft_type_icao"].astype("category")
    _, test = train_test_split(
        pool, test_size=1/3, random_state=RANDOM_SEED, stratify=pool["year"],
    )
    del pool
    print(f"  test: {len(test):,} rows in {time.time()-t0:.0f}s")

    yp_full = model.predict(test[LEAN_AC_FEATS])
    co2_km = test["contrail_CO2_km"].to_numpy()
    build_fig3c(perm, co2_km, yp_full)

    print(f"\nTotal: {time.time()-t0:.0f}s")


def main():
    t0 = time.time()
    print(f"Loading cache {CACHE.name} ...")
    pool = pd.read_parquet(CACHE)
    pool["aircraft_type_icao"] = pool["aircraft_type_icao"].astype("category")
    print(f"  {len(pool):,} rows in {time.time()-t0:.1f}s")

    cols_keep = LEAN_AC_FEATS + [TARGET, "contrail_CO2_km", "year"]
    train, test = train_test_split(
        pool[cols_keep], test_size=1/3,
        random_state=RANDOM_SEED, stratify=pool["year"],
    )
    del pool
    print(f"  train: {len(train):,}   test: {len(test):,}")

    # ── Train canonical model ──
    print("\n[1] Training canonical LEAN+AC ...")
    t1 = time.time()
    model = XGBRegressor(**PARAMS)
    model.fit(train[LEAN_AC_FEATS], train[TARGET])
    print(f"  fit: {time.time()-t1:.0f}s")

    # ── Save model ──
    joblib.dump({"model": model, "features": LEAN_AC_FEATS, "params": PARAMS},
                MODEL_PATH)
    print(f"  saved to {MODEL_PATH.name}")

    # ── Evaluate on full test (and by year) ──
    rows = []
    test_2019 = test[test["year"] == 2019]
    test_2021 = test[test["year"] == 2021]
    for split_name, split in [
        ("test_pooled", test),
        ("test_2019_only", test_2019),
        ("test_2021_only", test_2021),
    ]:
        yp = model.predict(split[LEAN_AC_FEATS])
        rmse = math.sqrt(mean_squared_error(split[TARGET], yp))
        r2 = r2_score(split[TARGET], yp)
        rec5, cap5 = top_k_metrics(split["contrail_CO2_km"], yp, k=0.05)
        rec10, cap10 = top_k_metrics(split["contrail_CO2_km"], yp, k=0.10)
        rows.append(dict(split=split_name, n=len(split), rmse=rmse, r2=r2,
                         top5_recall=rec5, top5_capture=cap5,
                         top10_recall=rec10, top10_capture=cap10))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "final_metrics.csv", index=False)
    print("\n[2] Final metrics:")
    print(metrics.to_string(index=False))

    # ── Permutation importance on a 50k test sample ──
    print("\n[3] Permutation importance (50k test sample, 5 repeats) ...")
    t2 = time.time()
    pi_sample = test.sample(50_000, random_state=RANDOM_SEED)
    pi = permutation_importance(
        model, pi_sample[LEAN_AC_FEATS], pi_sample[TARGET],
        n_repeats=5, random_state=RANDOM_SEED, n_jobs=-1,
    )
    perm = pd.DataFrame(dict(
        feature=LEAN_AC_FEATS,
        importance_mean=pi.importances_mean,
        importance_std=pi.importances_std,
    )).sort_values("importance_mean", ascending=True)
    perm["pretty"] = perm["feature"].map(PRETTY)
    perm.to_csv(OUT / "final_permutation_importance.csv", index=False)
    print(f"  done in {time.time()-t2:.0f}s")
    print(perm[["feature", "importance_mean", "importance_std"]].to_string(index=False))

    # ── Pre-compute the Lorenz curve from predictions ──
    yp_full = model.predict(test[LEAN_AC_FEATS])
    co2_km = test["contrail_CO2_km"].to_numpy()
    build_fig3c(perm, co2_km, yp_full)

    # Save a JSON manifest of the canonical model for the paper
    manifest = dict(
        model="LEAN+AC (14 features) tuned XGBoost regression",
        params=PARAMS,
        features=LEAN_AC_FEATS,
        n_train=int(len(train)),
        n_test=int(len(test)),
        target="contrail_type (signed log of contrail_CO2_km)",
        metrics=metrics.set_index("split").to_dict(),
        random_seed=RANDOM_SEED,
    )
    with open(OUT / "final_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nTotal: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:
        replot()
    else:
        main()
