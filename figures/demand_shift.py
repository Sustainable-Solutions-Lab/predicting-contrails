"""
Demand-shift feasibility of the top-10% avoidance rule.

For every flight in the model's predicted top 10% across each customer's
log, we ask: how many same-route alternative departures (same origin and
destination airports) fall *below* the global 90th-percentile threshold,
within +/- {2h, 6h, 24h, 7d} of the original scheduled time?

This is the concrete answer to "if travelers followed the rule, would
the alternatives even exist?" — and bounds the operational disruption
required.

Pipeline:
  1. Build (or load) a 2021-wide predictions parquet — for every 2021
     flight: flight_id, origin/destination airport codes, scheduled UTC,
     and predicted signed-log contrail forcing under the canonical
     schedule-only model.
  2. Compute the global 90th-percentile of predicted log forcing — that
     is the rule's threshold.
  3. Identify each customer's top-10% flagged flights (same procedure
     as fig4_customer.py).
  4. For each flagged flight, query the 2021 directory for same-route
     candidates and report alternative-density statistics.
  5. Plot.

Outputs go straight to the Dropbox Plots/ folder (never into git).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt

# TrueType (Type 42) fonts in EPS/PDF — Illustrator hangs/garbles on
# matplotlib's default Type 3 glyph programs
plt.rcParams.update({"ps.fonttype": 42, "pdf.fonttype": 42})
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from common import (  # noqa: E402
    DROPBOX_PARQUETS, DROPBOX_PLOTS, MODEL_FEATS, NEEDED_RAW_COLS,
    OUT as EXP_OUT, add_features,
)
from fig4_customer import CUSTOMERS  # noqa: E402

OUT = DROPBOX_PLOTS

PRED_CACHE = EXP_OUT / "_2021_predictions.parquet"
MODEL_PATH = EXP_OUT / "final_model.joblib"

# Time-window thresholds for "acceptable nearby alternative"
WINDOWS = [
    ("±2h", pd.Timedelta(hours=2)),
    ("±6h", pd.Timedelta(hours=6)),
    ("±24h", pd.Timedelta(days=1)),
    ("±7d", pd.Timedelta(days=7)),
]


# ──────────────────────────────────────────────────────────────────────────
# Build / load the 2021 predictions cache
# ──────────────────────────────────────────────────────────────────────────

def build_predictions_cache(model, feats: list[str]) -> pd.DataFrame:
    """Stream 2021 monthly parquets, compute features, predict, persist."""
    print(f"Building 2021 predictions cache → {PRED_CACHE.name} ...")
    files = sorted(DROPBOX_PARQUETS.glob("features_2021*_gdf.pq"))
    cols = NEEDED_RAW_COLS + ["flight_id", "origin_airport", "destination_airport"]

    pieces = []
    t0 = time.time()
    for i, f in enumerate(files, 1):
        df = pd.read_parquet(f, columns=cols)
        df = add_features(df).dropna(subset=feats)
        df["aircraft_type_icao"] = df["aircraft_type_icao"].astype("category")
        df["pred_log"] = model.predict(df[feats])
        pieces.append(df[[
            "flight_id", "origin_airport", "destination_airport",
            "first_waypoint_time", "pred_log", "contrail_CO2_km",
            "total_flight_distance_km",
        ]])
        if i % 5 == 0 or i == len(files):
            kept = sum(len(p) for p in pieces)
            print(f"  [{i}/{len(files)}] {kept:,} flights, "
                  f"{(time.time()-t0)/60:.1f} min elapsed", flush=True)

    out = pd.concat(pieces, ignore_index=True)
    print(f"  total: {len(out):,} predicted flights")
    out.to_parquet(PRED_CACHE)
    print(f"  saved {PRED_CACHE.name} ({PRED_CACHE.stat().st_size/1e9:.2f} GB)")
    return out


# ──────────────────────────────────────────────────────────────────────────
# Demand-shift analysis
# ──────────────────────────────────────────────────────────────────────────

def analyze_customer(
    cust_name: str,
    flagged: pd.DataFrame,         # one row per flagged customer flight
    db: pd.DataFrame,              # full 2021 directory
    threshold_log: float,
) -> tuple[pd.DataFrame, dict]:
    """For each flagged flight, count same-route alternatives below threshold
    in each time window."""
    # Index db by (origin, destination) for fast lookup
    db_by_route = {
        key: g.sort_values("first_waypoint_time")
        for key, g in db.groupby(["origin_airport", "destination_airport"])
    }
    # Threshold: only consider alternatives below the rule's threshold
    db_acceptable = db[db["pred_log"] < threshold_log]
    db_acc_by_route = {
        key: g.sort_values("first_waypoint_time")
        for key, g in db_acceptable.groupby(["origin_airport", "destination_airport"])
    }

    rows = []
    for r in flagged.itertuples():
        key = (r.origin_airport, r.destination_airport)
        original_time = r.first_waypoint_time
        same_route_total = len(db_by_route.get(key, pd.DataFrame()))
        candidates = db_acc_by_route.get(key)
        if candidates is None or len(candidates) == 0:
            row = dict(
                flight_id=r.flight_id, origin=r.origin_airport,
                destination=r.destination_airport,
                same_route_total=same_route_total,
                acceptable_total=0,
                min_time_delta_hours=np.nan,
            )
            for label, _ in WINDOWS:
                row[f"n_{label}"] = 0
            rows.append(row)
            continue

        deltas = (candidates["first_waypoint_time"] - original_time).dt.total_seconds() / 3600
        deltas_abs = deltas.abs()
        row = dict(
            flight_id=r.flight_id, origin=r.origin_airport,
            destination=r.destination_airport,
            same_route_total=same_route_total,
            acceptable_total=len(candidates),
            min_time_delta_hours=deltas_abs.min(),
        )
        for label, window in WINDOWS:
            row[f"n_{label}"] = int((deltas_abs <= window.total_seconds() / 3600).sum())
        rows.append(row)

    out = pd.DataFrame(rows)
    out.insert(0, "customer", cust_name)

    pct_with = {
        label: 100 * (out[f"n_{label}"] >= 1).sum() / len(out)
        for label, _ in WINDOWS
    }
    return out, pct_with


# ──────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────

def plot_demand_shift(per_flight: pd.DataFrame, pct_with: dict):
    """3-panel figure summarizing alternative density."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # (a) Alternatives within ±24h, distribution per flagged flight
    ax = axes[0]
    for cust, color in [("cust1", "#d1495b"), ("cust2", "#1f4e79")]:
        sub = per_flight[per_flight["customer"] == cust]
        bins = np.arange(0, max(sub["n_±24h"].max() + 2, 11))
        ax.hist(
            sub["n_±24h"], bins=bins, alpha=0.55, color=color,
            label=f"Customer {cust[-1]} (n={len(sub)})",
        )
    ax.set_xlabel("Same-route acceptable alternatives within ±24h")
    ax.set_ylabel("Number of flagged flights")
    ax.set_title("(a) Density of same-route alternatives", pad=8)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    # (b) Time-delta to nearest acceptable alternative (capped at 7d)
    ax = axes[1]
    cap_h = 24 * 7
    for cust, color in [("cust1", "#d1495b"), ("cust2", "#1f4e79")]:
        sub = per_flight[per_flight["customer"] == cust].copy()
        delta = sub["min_time_delta_hours"].fillna(cap_h * 2)
        delta = np.minimum(delta, cap_h)
        ax.hist(
            delta, bins=np.linspace(0, cap_h, 25), alpha=0.55, color=color,
            label=f"Customer {cust[-1]} (n={len(sub)})",
        )
    ax.set_xlabel("Hours to nearest acceptable same-route alternative")
    ax.set_ylabel("Number of flagged flights")
    ax.set_title("(b) Time delta to nearest alternative (capped at 7d)", pad=8)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    # (c) Cumulative % with ≥1 alternative as function of window
    ax = axes[2]
    labels = [w[0] for w in WINDOWS]
    x = np.arange(len(labels))
    width = 0.36
    cust1_pct = [pct_with["cust1"][lbl] for lbl in labels]
    cust2_pct = [pct_with["cust2"][lbl] for lbl in labels]
    ax.bar(x - width/2, cust1_pct, width, color="#d1495b", label="Customer 1")
    ax.bar(x + width/2, cust2_pct, width, color="#1f4e79", label="Customer 2")
    for xi, v in zip(x - width/2, cust1_pct):
        ax.text(xi, v + 1, f"{v:.0f}%", ha="center", fontsize=9, color="#d1495b")
    for xi, v in zip(x + width/2, cust2_pct):
        ax.text(xi, v + 1, f"{v:.0f}%", ha="center", fontsize=9, color="#1f4e79")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Time window")
    ax.set_ylabel("% of flagged flights with ≥ 1 acceptable alternative")
    ax.set_title("(c) Operational feasibility of avoidance rule", pad=8)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_ylim(0, 110)
    ax.grid(alpha=0.25, axis="y")

    plt.tight_layout()
    fig.savefig(OUT / "demand_shift.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "demand_shift.pdf", bbox_inches="tight")
    fig.savefig(OUT / "demand_shift.eps", bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"Loading model {MODEL_PATH.name} ...")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feats = bundle["features"]

    if PRED_CACHE.exists():
        print(f"Loading 2021 predictions from {PRED_CACHE.name} ...")
        db = pd.read_parquet(PRED_CACHE)
        print(f"  {len(db):,} flights in {time.time()-t0:.1f}s")
    else:
        db = build_predictions_cache(model, feats)

    threshold_log = db["pred_log"].quantile(0.90)
    print(f"\nGlobal 90th-pct threshold: pred_log = {threshold_log:.4f}")
    print(f"  → {(db['pred_log'] >= threshold_log).sum():,} flights flagged "
          f"({(db['pred_log'] >= threshold_log).mean()*100:.1f}%)")

    # For each customer, get top-10% flagged subset, attach airport codes,
    # then run demand-shift analysis.
    all_per_flight = []
    pct_with_all = {}
    for cust in CUSTOMERS:
        cust_log = cust.loader()
        merged = cust_log.merge(
            db[["flight_id", "origin_airport", "destination_airport",
                "first_waypoint_time", "pred_log"]],
            on="flight_id", how="inner",
        ).sort_values("pred_log", ascending=False).reset_index(drop=True)
        n_flag = max(1, int(round(len(merged) * 0.10)))
        flagged = merged.head(n_flag)
        print(f"\n[{cust.name}] flagged top-10% = {len(flagged)} of {len(merged)}")

        per_flight, pct_with = analyze_customer(
            cust.name, flagged, db, threshold_log,
        )
        per_flight["customer"] = cust.name
        all_per_flight.append(per_flight)
        pct_with_all[cust.name] = pct_with

        print(f"  same-route alternatives — pct with ≥1 acceptable:")
        for label, _ in WINDOWS:
            print(f"    {label}: {pct_with[label]:.1f}%")
        print(f"  median acceptable alternatives within ±24h: "
              f"{per_flight['n_±24h'].median():.0f}")
        print(f"  median time delta to nearest acceptable: "
              f"{per_flight['min_time_delta_hours'].median():.1f}h")

    per_flight_all = pd.concat(all_per_flight, ignore_index=True)
    per_flight_all.to_csv(OUT / "demand_shift_per_flight.csv", index=False)
    pd.DataFrame(pct_with_all).to_csv(OUT / "demand_shift_pct_with_alt.csv")

    plot_demand_shift(per_flight_all, pct_with_all)
    print(f"\nWrote demand_shift.{{png,pdf}} + summary CSVs in "
          f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
