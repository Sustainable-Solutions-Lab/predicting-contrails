"""
Fig 4 — Apply the canonical model to a customer flight log.

Customer 1: ~900 unique flights, all in 2021. Joins to the contrails team's
2021 process-model output (read directly from the per-month parquet shards
in Dropbox, filtered to customer flight_ids). The canonical model predicts
per-flight contrail forcing; we then plot the actual customer footprint
versus a counterfactual where the top-K% of flights (by predicted forcing)
are removed.

Outputs are written under figures/customer_outputs/ which is gitignored —
the customer data itself is sensitive and stays out of the repo.

Customer 2 has trip-level itineraries without flight_ids, so it requires a
fuzzier match to the flight database. Left as TODO.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse feature engineering + paths from the experiment package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from feature_pruning import (  # noqa: E402
    DROPBOX_PARQUETS, LEAN_AC_FEATS, NEEDED_RAW_COLS, OUT as EXP_OUT,
    add_features,
)

CUST_DIR = Path(
    "/Users/stevedavis/Library/CloudStorage/Dropbox/"
    "Papers/Active Prep/WS Corp contrails (w Silas)/Results"
)
CUST1_PATH = CUST_DIR / "Results_Cust1.csv"

OUT = Path(__file__).parent / "customer_outputs"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_PATH = EXP_OUT / "final_model.joblib"

# How aggressive a "remove the top K%" rule to display
THRESHOLDS_PCT = [1, 5, 10, 20]


def load_customer1():
    df = pd.read_csv(CUST1_PATH)
    keep_cols = {
        "flight_id": "flight_id",
        "FROM AIRPORT CODE": "origin",
        "TO AIRPORT CODE": "destination",
        "Energy forcing [GJ]": "energy_forcing_GJ",
        "CO2 equivalence [tons] - entire flight and contrails only": "co2_eq_tons",
        "Aircraft Type [ICAO]": "aircraft_type",
        "Total distance [km]": "distance_km",
    }
    df = df[list(keep_cols.keys())].rename(columns=keep_cols)
    df = df.dropna(subset=["flight_id"]).drop_duplicates("flight_id").reset_index(drop=True)
    return df


def fetch_features_for_flights(flight_ids: pd.Series) -> pd.DataFrame:
    """Stream the 2021 per-month parquets, keep only rows whose flight_id
    appears in the customer set, return engineered LEAN+AC features."""
    files = sorted(DROPBOX_PARQUETS.glob("features_2021*_gdf.pq"))
    print(f"  scanning {len(files)} 2021 shards ...")
    wanted = set(flight_ids.astype(str))
    pieces = []
    cols = NEEDED_RAW_COLS + ["flight_id"]
    for i, f in enumerate(files, 1):
        df = pd.read_parquet(f, columns=cols)
        df = df[df["flight_id"].isin(wanted)]
        if len(df):
            pieces.append(df)
        if i % 10 == 0:
            print(f"    [{i}/{len(files)}] kept {sum(len(p) for p in pieces):,} so far",
                  flush=True)
    raw = pd.concat(pieces, ignore_index=True)
    print(f"  matched {len(raw):,} rows for {raw['flight_id'].nunique():,} flights")
    raw = add_features(raw).dropna(subset=LEAN_AC_FEATS)
    raw["aircraft_type_icao"] = raw["aircraft_type_icao"].astype("category")
    raw = raw.drop_duplicates("flight_id")
    return raw


def main():
    t0 = time.time()
    print("Loading model ...")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feats = bundle["features"]
    print(f"  {len(feats)} features, params={bundle['params']}")

    print("\nLoading Customer 1 ...")
    cust = load_customer1()
    print(f"  {len(cust):,} unique flights, {len(cust['origin'].unique()):,} origins, "
          f"{cust['energy_forcing_GJ'].sum():.1f} GJ total energy forcing")

    print("\nFetching features for Cust1 flights from 2021 shards ...")
    feats_df = fetch_features_for_flights(cust["flight_id"])
    print(f"  feature engineering done in {time.time()-t0:.0f}s")

    # Make sure aircraft_type_icao categories include those seen in training
    train_categories = bundle.get("aircraft_type_categories")
    if train_categories is not None:
        feats_df["aircraft_type_icao"] = pd.Categorical(
            feats_df["aircraft_type_icao"], categories=train_categories,
        )

    # Predict (signed-log target)
    yhat_log = model.predict(feats_df[feats])
    feats_df["pred_log"] = yhat_log
    # Convert log → CO2-eq tons / km, sign-preserving
    feats_df["pred_co2_km"] = np.where(
        yhat_log > 0, np.exp(yhat_log),
        np.where(yhat_log < 0, -np.exp(-yhat_log), 0.0),
    )
    feats_df["pred_co2_tons"] = (
        feats_df["pred_co2_km"] * feats_df["total_flight_distance_km"]
    )

    merged = cust.merge(
        feats_df[["flight_id", "pred_log", "pred_co2_km",
                  "pred_co2_tons", "contrail_CO2_km"]],
        on="flight_id", how="inner",
    )
    print(f"\n  {len(merged):,} of {len(cust):,} customer flights matched & predicted")

    # ── Compute counterfactuals ──────────────────────────────────────────
    # Sort customer flights by predicted forcing, descending. The actual
    # customer footprint comes from the contrails team's per-flight
    # `co2_eq_tons` (so the truth is consistent with what they reported).
    merged_sorted = merged.sort_values("pred_log", ascending=False).reset_index(drop=True)
    n = len(merged_sorted)
    actual_total = merged_sorted["co2_eq_tons"].sum()

    rows = []
    for k_pct in THRESHOLDS_PCT:
        n_remove = max(1, int(round(n * k_pct / 100)))
        # "Remove" the top n_remove (model-predicted worst) and pretend they
        # didn't fly. This is the *upper bound* on avoidance benefit.
        kept = merged_sorted.iloc[n_remove:]
        kept_total = kept["co2_eq_tons"].sum()
        # For comparison: best-possible (oracle) avoidance using true forcing
        oracle = merged_sorted.sort_values("co2_eq_tons", ascending=False).iloc[n_remove:]
        oracle_total = oracle["co2_eq_tons"].sum()
        rows.append(dict(
            threshold_pct=k_pct, n_removed=n_remove,
            actual_tons=actual_total,
            after_model_tons=kept_total,
            after_oracle_tons=oracle_total,
            model_reduction_pct=100 * (actual_total - kept_total) / actual_total,
            oracle_reduction_pct=100 * (actual_total - oracle_total) / actual_total,
        ))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "fig4_cust1_summary.csv", index=False)
    print("\nReduction summary:")
    print(summary.to_string(index=False))

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # (a) Lorenz-style cumulative forcing curve
    ax = axes[0]
    # Drop flights with missing co2_eq_tons (sums on these would poison the cumsum)
    lorenz = merged_sorted.dropna(subset=["co2_eq_tons"]).reset_index(drop=True)
    pred_order = lorenz["co2_eq_tons"].to_numpy()  # already sorted by predicted forcing
    n = len(pred_order)
    pos_total = np.maximum(pred_order, 0).sum()
    perfect_sorted = np.sort(pred_order)[::-1]
    cum_perfect = np.cumsum(np.maximum(perfect_sorted, 0)) / pos_total * 100
    cum_pred = np.cumsum(np.maximum(pred_order, 0)) / pos_total * 100
    pct_x = np.linspace(100 / n, 100, n)
    ax.plot(pct_x, cum_perfect, lw=2.4, color="#1f4e79", label="Perfect knowledge")
    ax.plot(pct_x, cum_pred, lw=2.4, color="#d1495b", label="LEAN+AC predictions")
    ax.plot([0, 100], [0, 100], lw=1.2, color="#888888", ls="--",
            label="Random ordering")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Customer flights ranked by predicted forcing (%)")
    ax.set_ylabel("Positive contrail forcing captured (%)")
    ax.set_title(f"(a) Customer 1 — concentration of contrail forcing", pad=8)
    # Annotate model and oracle capture at top-5% and top-10%
    for k_pct, color in [(5, "#d1495b"), (10, "#d1495b")]:
        idx = int(round(k_pct / 100 * n)) - 1
        ax.scatter([k_pct], [cum_pred[idx]], color=color, s=22, zorder=5)
        ax.annotate(f"model: {cum_pred[idx]:.0f}%",
                    xy=(k_pct, cum_pred[idx]), xytext=(k_pct + 1, cum_pred[idx] - 14),
                    fontsize=9, color=color)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(alpha=0.25)

    # (b) Reduction at each threshold — model vs oracle
    ax = axes[1]
    x = np.arange(len(THRESHOLDS_PCT))
    width = 0.36
    ax.bar(x - width/2, summary["model_reduction_pct"], width=width,
           color="#d1495b", label="LEAN+AC model")
    ax.bar(x + width/2, summary["oracle_reduction_pct"], width=width,
           color="#1f4e79", label="Oracle (perfect knowledge)")
    for xi, v in zip(x - width/2, summary["model_reduction_pct"]):
        ax.text(xi, v + 1, f"{v:.0f}%", ha="center", fontsize=9)
    for xi, v in zip(x + width/2, summary["oracle_reduction_pct"]):
        ax.text(xi, v + 1, f"{v:.0f}%", ha="center", fontsize=9, color="#1f4e79")
    ax.set_xticks(x)
    ax.set_xticklabels([f"top {k}%" for k in THRESHOLDS_PCT])
    ax.set_xlabel("Flights removed (ranked by predicted forcing)")
    ax.set_ylabel("Customer 1 contrail-eq forcing reduction (%)")
    ax.set_title("(b) Avoidance scenarios", pad=8)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_ylim(0, max(summary["oracle_reduction_pct"].max() * 1.2, 105))
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle(
        f"Fig 4 — Applying LEAN+AC to Customer 1's flight log "
        f"({len(merged_sorted)} unique flights, 2021, {actual_total:.0f} tCO₂e total)",
        fontsize=11, y=1.02,
    )
    fig.savefig(OUT / "fig4_cust1.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig4_cust1.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote fig4_cust1.{{png,pdf,csv}}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
