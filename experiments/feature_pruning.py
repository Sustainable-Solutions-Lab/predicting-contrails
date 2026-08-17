"""
Feature pruning experiment.

Loads a sample of 2019 + 2021 per-month feature parquets, applies Silas's
exact feature engineering (jul12 trainer), and compares two XGBoost
regressors:

  1. FULL : all 28 schedule-only features as in jul12
  2. LEAN : 12 features chosen to remove redundancy (no end/mid hour, no
            time_span, no crosses_midnight, no bearing, no night_score_bool,
            only night_score_full_0)

Reports test R², RMSE, top-5% recall, top-decile-forcing-capture, and
permutation importance on the lean model.

Trains on a 2019 sample, holds out a 2019 test split, and ALSO evaluates
on a 2021 sample as an out-of-distribution check.

All outputs go to experiments/outputs/.
"""

from __future__ import annotations

import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────
DROPBOX_PARQUETS = Path(
    "/Users/stevedavis/Library/CloudStorage/Dropbox/"
    "Papers/Active Prep/Contrails/WS Corp contrails (w Silas)/adjustedEFs"
)
# All raw generated figures go straight here (Steve's hand-polished
# versions live in the sibling Figures/ folder; scripts stay in git).
DROPBOX_PLOTS = Path(
    "/Users/stevedavis/Library/CloudStorage/Dropbox/"
    "Papers/Active Prep/Contrails/WS Corp contrails (w Silas)/Plots"
)
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# How many shards per year. None = use all 47.
SHARDS_PER_YEAR = None
RANDOM_SEED = 42

# ──────────────────────────────────────────────────────────────────────────
# Feature engineering (mirrors contrail_avoidance_regression_ml_jul12.py)
# ──────────────────────────────────────────────────────────────────────────

AGWP_100 = 82.5e-15  # W/m²/kg
S_EARTH = 5.101e14   # m²
SECONDS_PER_YEAR = 365 * 24 * 3600


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["total_flight_distance_km"].astype(float) >= 0.1].copy()

    # Target
    df["contrail_CO2"] = df["total_contrail_energy_forcing"] / (
        AGWP_100 * SECONDS_PER_YEAR * S_EARTH
    )
    df["contrail_CO2_km"] = df["contrail_CO2"] / df["total_flight_distance_km"]

    df["first_waypoint_time"] = pd.to_datetime(df["first_waypoint_time"])
    df["last_waypoint_time"] = pd.to_datetime(df["last_waypoint_time"])

    # Day-of-year cyclic
    doy = df["first_waypoint_time"].dt.dayofyear.astype(float)
    df["day_sin"] = np.sin(np.pi * doy * 2 / 365)
    df["day_cos"] = np.cos(np.pi * doy * 2 / 365)

    # Hour-of-day cyclic at start, end, mid
    for col, src in [
        ("start", df["first_waypoint_time"].dt.hour),
        ("end", df["last_waypoint_time"].dt.hour),
        (
            "mid",
            (
                df["first_waypoint_time"]
                + (df["last_waypoint_time"] - df["first_waypoint_time"]) / 2
            ).dt.hour,
        ),
    ]:
        h = src.astype(float)
        df[f"{col}_hour_sin"] = np.sin(np.pi * h * 2 / 24)
        df[f"{col}_hour_cos"] = np.cos(np.pi * h * 2 / 24)

    df["time_span_hours"] = (
        df["last_waypoint_time"] - df["first_waypoint_time"]
    ).dt.total_seconds() / 3600
    df["crosses_midnight"] = (
        df["last_waypoint_time"].dt.date != df["first_waypoint_time"].dt.date
    ).astype(int)

    # Cyclic bearing & longitude
    df["bearing_sin"] = np.sin(np.pi * df["bearing"].astype(float) * 2 / 360)
    df["bearing_cos"] = np.cos(np.pi * df["bearing"].astype(float) * 2 / 360)
    for c in ["OriginLon", "DestinationLon"]:
        v = df[c].astype(float)
        df[f"{c}_sin"] = np.sin(np.pi * v * 2 / 360)
        df[f"{c}_cos"] = np.cos(np.pi * v * 2 / 360)

    # Signed-log target (matches Silas)
    x = df["contrail_CO2_km"].astype(float)
    df["contrail_type"] = np.where(
        x > 0, np.log(np.where(x > 0, x, 1)),
        np.where(x < 0, -np.log(np.abs(np.where(x < 0, x, -1))), 0.0),
    )
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


FULL_FEATS = [
    "total_flight_distance_km",
    "day_sin", "day_cos",
    "start_hour_sin", "start_hour_cos",
    "end_hour_sin", "end_hour_cos",
    "mid_hour_sin", "mid_hour_cos",
    "time_span_hours", "crosses_midnight",
    "land_score",
    "bearing_sin", "bearing_cos",
    "OriginLon_sin", "OriginLon_cos",
    "DestinationLon_sin", "DestinationLon_cos",
    "OriginLat", "DestinationLat",
    "night_score_full_0", "night_score_full_1",
    "night_score_full_2", "night_score_full_3",
    "night_score_bool_0", "night_score_bool_1",
    "night_score_bool_2", "night_score_bool_3",
]

LEAN_FEATS = [
    "total_flight_distance_km",
    "day_sin", "day_cos",
    "start_hour_sin", "start_hour_cos",
    "OriginLat", "OriginLon_sin", "OriginLon_cos",
    "DestinationLat", "DestinationLon_sin", "DestinationLon_cos",
    "land_score",
    "night_score_full_0",
]

LEAN_AC_FEATS = LEAN_FEATS + ["aircraft_type_icao"]

CATEGORICAL_FEATS = {"aircraft_type_icao"}

TARGET = "contrail_type"
NEEDED_RAW_COLS = [
    "first_waypoint_time", "last_waypoint_time",
    "total_flight_distance_km", "total_contrail_energy_forcing",
    "bearing", "land_score",
    "OriginLon", "OriginLat", "DestinationLon", "DestinationLat",
    "aircraft_type_icao",
    "night_score_full_0", "night_score_full_1",
    "night_score_full_2", "night_score_full_3",
    "night_score_bool_0", "night_score_bool_1",
    "night_score_bool_2", "night_score_bool_3",
]


FEATURE_OUT_COLS = sorted(set(FULL_FEATS + LEAN_FEATS + LEAN_AC_FEATS)) + [
    TARGET, "contrail_CO2_km",
]


def load_year(year: int, n_shards: int | None) -> pd.DataFrame:
    files = sorted(DROPBOX_PARQUETS.glob(f"features_{year}*_gdf.pq"))
    if n_shards is None or n_shards >= len(files):
        pick = list(range(len(files)))
    else:
        rng = np.random.default_rng(RANDOM_SEED + year)
        pick = sorted(rng.choice(len(files), size=n_shards, replace=False).tolist())

    print(f"  loading {len(pick)} of {len(files)} shards for {year}")
    pieces = []
    total_raw = 0
    for i in pick:
        f = files[i]
        df = pd.read_parquet(f, columns=NEEDED_RAW_COLS)
        total_raw += len(df)
        # Compute features, downcast, drop everything we don't need.
        df = add_features(df).dropna(subset=FULL_FEATS + [TARGET])
        df["aircraft_type_icao"] = df["aircraft_type_icao"].astype("category")
        keep = [c for c in FEATURE_OUT_COLS if c in df.columns]
        df = df[keep]
        for c in df.select_dtypes("float64").columns:
            df[c] = df[c].astype("float32")
        pieces.append(df)
    out = pd.concat(pieces, ignore_index=True)
    # Re-unify the categorical dtype across shards (concat with cat dtypes can
    # create a CategoricalDtype with different categories per shard).
    out["aircraft_type_icao"] = out["aircraft_type_icao"].astype("category")
    print(f"  {year}: {total_raw:,} raw → {len(out):,} clean")
    return out


# ──────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────

def top_k_metrics(y_true_co2_km, y_pred_log, k=0.05):
    """y_true_co2_km is per-km contrail CO2-eq. y_pred_log is signed log.
    Returns (top-k recall on warming flights, fraction of total POSITIVE
    forcing concentrated in the predicted top-k)."""
    arr = np.asarray(y_true_co2_km, dtype=float)
    pred = np.asarray(y_pred_log, dtype=float)
    n = arr.size
    n_top = max(1, int(round(n * k)))
    true_top_idx = np.argsort(arr)[::-1][:n_top]
    pred_top_idx = np.argsort(pred)[::-1][:n_top]
    recall = np.intersect1d(true_top_idx, pred_top_idx).size / n_top
    pos_total = np.maximum(arr, 0).sum()
    captured = np.maximum(arr[pred_top_idx], 0).sum()
    capture = captured / pos_total if pos_total > 0 else float("nan")
    return recall, capture


def evaluate(name, model, X, y_log, y_true_co2_km):
    yp = model.predict(X)
    rmse = math.sqrt(mean_squared_error(y_log, yp))
    r2 = r2_score(y_log, yp)
    mae = mean_absolute_error(y_log, yp)
    rec5, cap5 = top_k_metrics(y_true_co2_km, yp, k=0.05)
    rec10, cap10 = top_k_metrics(y_true_co2_km, yp, k=0.10)
    return dict(name=name, n=len(y_log), rmse=rmse, r2=r2, mae=mae,
                top5_recall=rec5, top5_capture=cap5,
                top10_recall=rec10, top10_capture=cap10)


# ──────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    cache_path = OUT / f"_pool_cache_{SHARDS_PER_YEAR or 'all'}.parquet"
    if cache_path.exists():
        print(f"Loading from cache {cache_path.name} ...")
        pool = pd.read_parquet(cache_path)
        print(f"  cache load: {time.time()-t0:.1f}s   rows: {len(pool):,}")
    else:
        print("Loading 2019 ...")
        df19 = load_year(2019, SHARDS_PER_YEAR)
        df19["year"] = 2019
        print("Loading 2021 ...")
        df21 = load_year(2021, SHARDS_PER_YEAR)
        df21["year"] = 2021
        print(f"  load+features: {time.time()-t0:.1f}s")
        pool = pd.concat([df19, df21], ignore_index=True)
        del df19, df21
        print(f"Caching pooled features to {cache_path.name} ...")
        pool.to_parquet(cache_path)
        print(f"  cache write: {time.time()-t0:.1f}s")

    pool["aircraft_type_icao"] = pool["aircraft_type_icao"].astype("category")
    print(f"  pool memory: {pool.memory_usage(deep=True).sum()/1e9:.2f} GB")

    # ── Correlation matrix of FULL feature set on 2019 ──
    print("\n[1] Correlation matrix on FULL feature set (2019) ...")
    corr = pool[pool["year"] == 2019][FULL_FEATS].corr()
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(corr, annot=False, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, cbar_kws=dict(shrink=0.7), ax=ax)
    ax.set_title("Pairwise correlation of jul12 FULL features (2019)")
    plt.tight_layout()
    plt.savefig(OUT / "correlation_full.png", dpi=160)
    plt.close()
    corr.to_csv(OUT / "correlation_full.csv")
    # Identify the highest off-diagonal correlations
    abs_corr = corr.abs().copy()
    np.fill_diagonal(abs_corr.values, 0)
    top_pairs = (
        abs_corr.where(np.triu(np.ones(abs_corr.shape, dtype=bool), k=1))
        .stack().sort_values(ascending=False).head(20)
    )
    print("  Top 20 |corr| pairs:")
    for (a, b), v in top_pairs.items():
        print(f"    {v:.3f}  {a}  ↔  {b}")

    # ── Random 2/3 train, 1/3 test (stratified by year) ──
    print("\n[2] Random 2/3 - 1/3 split (stratified by year) ...")
    feat_union = sorted(set(FULL_FEATS + LEAN_FEATS + LEAN_AC_FEATS))
    cols_keep = feat_union + [TARGET, "contrail_CO2_km", "year"]
    train, test = train_test_split(
        pool[cols_keep],
        test_size=1/3,
        random_state=RANDOM_SEED,
        stratify=pool["year"],
    )
    del pool
    test_2019 = test[test["year"] == 2019]
    test_2021 = test[test["year"] == 2021]
    print(f"  train: {len(train):,}   test: {len(test):,}  "
          f"(2019: {len(test_2019):,}  2021: {len(test_2021):,})")

    # ── Train FULL, LEAN, LEAN+AC with the same hyperparameters ──
    # Pick hyperparams that won't time out: 300 trees, depth 7, lr 0.1.
    # Silas's grid was [10, 10000] — neither is sensible; use a defensible mid.
    params = dict(
        n_estimators=300, max_depth=7, learning_rate=0.1,
        n_jobs=-1, tree_method="hist", random_state=RANDOM_SEED,
        enable_categorical=True,
    )

    rows = []
    fitted = {}
    for name, feats in [
        ("FULL", FULL_FEATS),
        ("LEAN", LEAN_FEATS),
        ("LEAN+AC", LEAN_AC_FEATS),
    ]:
        print(f"\n[3] Training {name} ({len(feats)} features) ...")
        t1 = time.time()
        model = XGBRegressor(**params)
        model.fit(train[feats], train[TARGET])
        print(f"  fit: {time.time()-t1:.1f}s")
        fitted[name] = (model, feats)
        for split_name, split in [
            ("test_pooled", test),
            ("test_2019_only", test_2019),
            ("test_2021_only", test_2021),
        ]:
            r = evaluate(f"{name}/{split_name}", model,
                         split[feats], split[TARGET], split["contrail_CO2_km"])
            rows.append(r)
            print(f"  {r['name']}: R2={r['r2']:.3f}  RMSE={r['rmse']:.3f}  "
                  f"top5_recall={r['top5_recall']:.3f}  top5_capture={r['top5_capture']:.3f}  "
                  f"top10_capture={r['top10_capture']:.3f}")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    print("\n[3] Metrics summary:")
    print(metrics.to_string(index=False))

    # ── Permutation importance on the LEAN model (2019 test split) ──
    print("\n[4] Permutation importance on LEAN model (2019 test) ...")
    t2 = time.time()
    model, feats = fitted["LEAN"]
    # Subsample to 30k for speed; n_repeats=5
    sample = test.sample(min(30000, len(test)), random_state=RANDOM_SEED)
    pi = permutation_importance(
        model, sample[feats], sample[TARGET],
        n_repeats=5, random_state=RANDOM_SEED, n_jobs=-1,
    )
    perm = pd.DataFrame(dict(
        feature=feats,
        importance_mean=pi.importances_mean,
        importance_std=pi.importances_std,
    )).sort_values("importance_mean", ascending=True)
    print(f"  permutation importance: {time.time()-t2:.1f}s")
    print(perm.to_string(index=False))
    perm.to_csv(OUT / "permutation_importance_lean.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(perm["feature"], perm["importance_mean"],
            xerr=perm["importance_std"], color="#3a7ca5")
    ax.set_xlabel("Permutation importance (Δ R² when shuffled)")
    ax.set_title("LEAN model — permutation importance (2019 test)")
    plt.tight_layout()
    plt.savefig(OUT / "permutation_importance_lean.png", dpi=160)
    plt.close()

    # ── Sklearn (gain) importance for FULL, LEAN, LEAN+AC side by side ──
    fig, axes = plt.subplots(1, len(fitted), figsize=(6 * len(fitted), 7))
    if len(fitted) == 1:
        axes = [axes]
    for ax, (name, (m, fts)) in zip(axes, fitted.items()):
        gi = pd.DataFrame(dict(feature=fts, importance=m.feature_importances_))\
              .sort_values("importance")
        ax.barh(gi["feature"], gi["importance"], color="#3a7ca5")
        ax.set_title(f"{name} ({len(fts)} feats) — XGBoost gain importance")
    plt.tight_layout()
    plt.savefig(OUT / "gain_importance_full_vs_lean.png", dpi=160)
    plt.close()

    print(f"\nTotal: {time.time()-t0:.1f}s")
    print(f"Outputs in {OUT}/")


if __name__ == "__main__":
    main()
