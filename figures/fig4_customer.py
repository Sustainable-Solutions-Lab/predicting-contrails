"""
Fig 4 — Apply the canonical model to two corporate flight logs.

Both customers' flight_ids match the contrails team's ID format
(YYMMDD-NNNNN-CCCN), so for each we:
  1. Load customer-specific columns (flight_id, ground-truth forcing).
  2. Inner-join to the 2021 per-month process-model parquets in Dropbox
     to get LEAN+AC features.
  3. Predict per-flight signed-log forcing with the canonical model.
  4. Sort customer flights by predicted forcing and report the
     reduction at top-1%, 5%, 10%, 20% removal.

Outputs go under figures/customer_outputs/ which is gitignored — the
customer data itself stays out of the repo.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from feature_pruning import (  # noqa: E402
    DROPBOX_PARQUETS, LEAN_AC_FEATS, NEEDED_RAW_COLS, OUT as EXP_OUT,
    add_features,
)

CUST_DIR = Path(
    "/Users/stevedavis/Library/CloudStorage/Dropbox/"
    "Papers/Active Prep/WS Corp contrails (w Silas)/Results"
)

OUT = Path(__file__).parent / "customer_outputs"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_PATH = EXP_OUT / "final_model.joblib"

THRESHOLDS_PCT = [1, 5, 10, 20]


@dataclass
class Customer:
    name: str
    label: str
    loader: Callable[[], pd.DataFrame]   # → df with cols [flight_id, co2_eq_tons]


def load_cust1() -> pd.DataFrame:
    df = pd.read_csv(CUST_DIR / "Results_Cust1.csv")
    df = df[[
        "flight_id",
        "CO2 equivalence [tons] - entire flight and contrails only",
    ]].rename(columns={
        "CO2 equivalence [tons] - entire flight and contrails only": "co2_eq_tons",
    })
    return (df.dropna(subset=["flight_id"])
              .drop_duplicates("flight_id")
              .reset_index(drop=True))


def load_cust2() -> pd.DataFrame:
    df = pd.read_excel(
        CUST_DIR / "Results_Cust2.xlsx",
        sheet_name="Contrail-resolving",
        engine="openpyxl",
        header=0,
    )
    df = df[["flight_id", "CO2 equivalence [tons total, Contrails only]"]].rename(
        columns={"CO2 equivalence [tons total, Contrails only]": "co2_eq_tons"},
    )
    return (df.dropna(subset=["flight_id"])
              .drop_duplicates("flight_id")
              .reset_index(drop=True))


CUSTOMERS = [
    Customer("cust1", "Customer 1", load_cust1),
    Customer("cust2", "Customer 2", load_cust2),
]


def fetch_features(flight_ids: pd.Series) -> pd.DataFrame:
    """Stream 2021 monthly parquets, keep only flight_ids in `flight_ids`,
    then run feature engineering + return LEAN+AC features."""
    files = sorted(DROPBOX_PARQUETS.glob("features_2021*_gdf.pq"))
    wanted = set(flight_ids.astype(str))
    pieces = []
    cols = NEEDED_RAW_COLS + ["flight_id"]
    for f in files:
        df = pd.read_parquet(f, columns=cols)
        df = df[df["flight_id"].isin(wanted)]
        if len(df):
            pieces.append(df)
    raw = pd.concat(pieces, ignore_index=True)
    raw = add_features(raw).dropna(subset=LEAN_AC_FEATS)
    raw["aircraft_type_icao"] = raw["aircraft_type_icao"].astype("category")
    return raw.drop_duplicates("flight_id")


def avoidance_summary(merged_sorted: pd.DataFrame) -> pd.DataFrame:
    """merged_sorted is sorted by predicted forcing desc; co2_eq_tons holds
    the customer's actual ground-truth forcing per flight."""
    n = len(merged_sorted)
    actual_total = merged_sorted["co2_eq_tons"].sum()
    rows = []
    for k_pct in THRESHOLDS_PCT:
        n_remove = max(1, int(round(n * k_pct / 100)))
        kept_model = merged_sorted.iloc[n_remove:]
        oracle = merged_sorted.sort_values(
            "co2_eq_tons", ascending=False,
        ).iloc[n_remove:]
        rows.append(dict(
            threshold_pct=k_pct,
            n_removed=n_remove,
            actual_tons=actual_total,
            after_model_tons=kept_model["co2_eq_tons"].sum(),
            after_oracle_tons=oracle["co2_eq_tons"].sum(),
            model_reduction_pct=100 * (
                actual_total - kept_model["co2_eq_tons"].sum()
            ) / actual_total,
            oracle_reduction_pct=100 * (
                actual_total - oracle["co2_eq_tons"].sum()
            ) / actual_total,
        ))
    return pd.DataFrame(rows)


def process_customer(
    cust: Customer, model, feats: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"\n=== {cust.label} ===")
    df = cust.loader()
    print(f"  {len(df):,} unique customer flights")

    print("  fetching features from 2021 shards ...")
    feats_df = fetch_features(df["flight_id"])
    print(f"  matched {len(feats_df):,} flights")

    yhat_log = model.predict(feats_df[feats])
    feats_df["pred_log"] = yhat_log

    merged = df.merge(
        feats_df[["flight_id", "pred_log"]], on="flight_id", how="inner",
    )
    merged_sorted = (
        merged.sort_values("pred_log", ascending=False).reset_index(drop=True)
    )
    summary = avoidance_summary(merged_sorted)
    summary.insert(0, "customer", cust.name)
    print(summary.to_string(index=False))
    return merged_sorted, summary


# ──────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────

def lorenz_panel(ax, merged_sorted: pd.DataFrame, label: str):
    lorenz = merged_sorted.dropna(subset=["co2_eq_tons"]).reset_index(drop=True)
    pred_order = lorenz["co2_eq_tons"].to_numpy()
    n = len(pred_order)
    pos_total = np.maximum(pred_order, 0).sum()
    perfect_sorted = np.sort(pred_order)[::-1]
    cum_perfect = np.cumsum(np.maximum(perfect_sorted, 0)) / pos_total * 100
    cum_pred = np.cumsum(np.maximum(pred_order, 0)) / pos_total * 100
    pct_x = np.linspace(100 / n, 100, n)

    ax.plot(pct_x, cum_perfect, lw=2.4, color="#1f4e79", label="Perfect knowledge")
    ax.plot(pct_x, cum_pred, lw=2.4, color="#d1495b", label="LEAN+AC predictions")
    ax.plot([0, 30], [0, 30], lw=1.2, color="#888888", ls="--",
            label="Random ordering")

    for k_pct in [5, 10]:
        idx = int(round(k_pct / 100 * n)) - 1
        ax.scatter([k_pct], [cum_pred[idx]], color="#d1495b", s=22, zorder=5)
        ax.annotate(f"{cum_pred[idx]:.0f}% at top {k_pct}%",
                    xy=(k_pct, cum_pred[idx]),
                    xytext=(k_pct + 1, cum_pred[idx] - 12),
                    fontsize=9, color="#d1495b")

    ax.set_xlim(0, 30)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Customer flights ranked by predicted forcing (%)")
    ax.set_ylabel("Positive contrail forcing captured (%)")
    ax.set_title(f"{label}", pad=8)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(alpha=0.25)


def bar_panel(ax, summary: pd.DataFrame, label: str):
    x = np.arange(len(summary))
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
    ax.set_xticklabels([f"top {k}%" for k in summary["threshold_pct"]])
    ax.set_xlabel("Flights removed (ranked by predicted forcing)")
    ax.set_ylabel("Contrail-eq forcing reduction (%)")
    ax.set_title(label, pad=8)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_ylim(0, max(summary["oracle_reduction_pct"].max() * 1.18, 105))
    ax.grid(alpha=0.25, axis="y")


def main():
    t0 = time.time()
    print(f"Loading model {MODEL_PATH.name} ...")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feats = bundle["features"]

    results = {}
    summaries = []
    for cust in CUSTOMERS:
        merged, summary = process_customer(cust, model, feats)
        results[cust.name] = (merged, summary)
        summaries.append(summary)

    pd.concat(summaries).to_csv(OUT / "fig4_summary.csv", index=False)

    # ── 2 × 2 figure: Lorenz on top row, bar charts on bottom ─────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, cust in enumerate(CUSTOMERS):
        merged, summary = results[cust.name]
        n_unique = len(merged)
        actual_total = merged["co2_eq_tons"].sum()
        # Top: Lorenz curve
        lorenz_panel(
            axes[0, col], merged,
            f"({chr(ord('a') + 2 * col)}) {cust.label} concentration "
            f"(n={n_unique}, {actual_total:.0f} tCO₂e)",
        )
        # Bottom: avoidance bars
        bar_panel(
            axes[1, col], summary,
            f"({chr(ord('a') + 2 * col + 1)}) {cust.label} avoidance scenarios",
        )

    fig.suptitle(
        "Fig 4 — Applying LEAN+AC to two corporate flight logs (2021)",
        fontsize=12, y=1.00,
    )
    plt.tight_layout()
    fig.savefig(OUT / "fig4_combined.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig4_combined.pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote fig4_combined.{{png,pdf}}, fig4_summary.csv "
          f"in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
