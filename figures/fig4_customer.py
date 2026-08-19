"""
Fig 4 — Apply the canonical model to two corporate flight logs, with
realistic alternative-flight replacement.

For each customer's predicted top-K% flights, instead of pretending the
flight didn't happen (the prior version's upper-bound assumption), we
identify the **nearest below-threshold same-route alternative in the
2021 corpus** and substitute its ACTUAL contrail forcing into the
customer total. The reduction is then the customer's real footprint
under a realistic shift-rather-than-skip rule.

Both the model and oracle bars use the same replacement procedure;
the only difference is which K% they identify (model uses predicted
forcing; oracle uses true forcing).

Layout: 2 × 2 panels — rows = customer; cols = (avoidance bars, time
delta to nearest alternative).

Reads from the 2021-wide predictions cache built by fig5_demand_shift.py
(experiments/outputs/_2021_predictions.parquet). Run that script first
if the cache is missing.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import joblib
import matplotlib.pyplot as plt

# TrueType (Type 42) fonts in EPS/PDF — Illustrator hangs/garbles on
# matplotlib's default Type 3 glyph programs
plt.rcParams.update({"ps.fonttype": 42, "pdf.fonttype": 42})
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from feature_pruning import DROPBOX_PLOTS, OUT as EXP_OUT  # noqa: E402

CUST_DIR = Path(
    "/Users/stevedavis/Library/CloudStorage/Dropbox/"
    "Papers/Active Prep/Contrails/WS Corp contrails (w Silas)/Results"
)

OUT = DROPBOX_PLOTS

PRED_CACHE = EXP_OUT / "_2021_predictions.parquet"
MODEL_PATH = EXP_OUT / "final_model.joblib"

# Threshold for "acceptable alternative": below the global predicted-forcing
# 90th percentile (= our top-10% rule).
THRESHOLD_QUANTILE = 0.90

# Cap for the time-delta histogram (longer alternatives become unrealistic
# bookings)
TIME_DELTA_CAP_DAYS = 7

THRESHOLDS_PCT = [1, 5, 10, 20]


@dataclass
class Customer:
    name: str
    label: str
    loader: Callable[[], pd.DataFrame]


def load_cust1() -> pd.DataFrame:
    df = pd.read_csv(CUST_DIR / "Results_Cust1.csv")
    df = df[[
        "flight_id",
        "CO2 equivalence [tons] - entire flight and contrails only",
    ]].rename(columns={
        "CO2 equivalence [tons] - entire flight and contrails only": "co2_eq_tons",
    })
    return (df.dropna(subset=["flight_id"])
              .drop_duplicates("flight_id").reset_index(drop=True))


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
              .drop_duplicates("flight_id").reset_index(drop=True))


CUSTOMERS = [
    Customer("cust1", "Customer 1", load_cust1),
    Customer("cust2", "Customer 2", load_cust2),
]


# ──────────────────────────────────────────────────────────────────────────
# Replacement procedure
# ──────────────────────────────────────────────────────────────────────────

def find_nearest_alternative(
    flight_row, db_acc_by_route: dict[tuple, pd.DataFrame],
) -> tuple[float, float]:
    """For one flight (with origin_airport, destination_airport,
    first_waypoint_time), look up acceptable same-route alternatives and
    return (replacement_co2_eq_tons, time_delta_hours). If none exist,
    return (NaN, NaN) — caller handles fallback (keep original flight)."""
    key = (flight_row.origin_airport, flight_row.destination_airport)
    cands = db_acc_by_route.get(key)
    if cands is None or len(cands) == 0:
        return (np.nan, np.nan)
    deltas = (
        cands["first_waypoint_time"] - flight_row.first_waypoint_time
    ).dt.total_seconds() / 3600
    deltas_abs = deltas.abs()
    idx = deltas_abs.idxmin()
    alt = cands.loc[idx]
    # Convert to tons: contrail_CO2_km is in kg/km, distance in km, /1000 → tons
    alt_tons = alt["contrail_CO2_km"] * alt["total_flight_distance_km"] / 1000.0
    return float(alt_tons), float(deltas_abs.loc[idx])


def replacement_summary(
    customer_with_pred: pd.DataFrame,    # customer flights + pred_log + airport+time
    db_acc_by_route: dict[tuple, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each threshold, identify model-flagged and oracle-flagged top-K%.
    For each flagged flight find nearest-alternative replacement; total
    becomes (sum of non-flagged actual) + (sum of replacement actual or
    fall back to original if no alternative exists).

    Returns (summary, per_flagged_flight) — the second is the model-flagged
    set with its time deltas, used for the histogram panel.
    """
    actual_total = customer_with_pred["co2_eq_tons"].sum()
    n = len(customer_with_pred)

    by_pred = customer_with_pred.sort_values("pred_log", ascending=False).reset_index(drop=True)
    by_truth = customer_with_pred.sort_values("co2_eq_tons", ascending=False).reset_index(drop=True)

    rows = []
    flagged_records = []  # for time-delta histogram, top-10% only

    for k_pct in THRESHOLDS_PCT:
        n_flag = max(1, int(round(n * k_pct / 100)))

        def total_with_replacement(flagged: pd.DataFrame, kept: pd.DataFrame, *, record=False):
            kept_total = kept["co2_eq_tons"].sum()
            replaced_total = 0.0
            n_replaced = 0
            n_kept_original = 0
            for r in flagged.itertuples():
                alt_tons, delta_h = find_nearest_alternative(r, db_acc_by_route)
                if np.isnan(alt_tons):
                    # No same-route alternative exists in the corpus → keep
                    # the original flight in the total.
                    replaced_total += float(r.co2_eq_tons) if not pd.isna(r.co2_eq_tons) else 0.0
                    n_kept_original += 1
                    if record:
                        flagged_records.append(dict(
                            flight_id=r.flight_id, threshold_pct=k_pct,
                            replacement_tons=np.nan, time_delta_hours=np.nan,
                            kept_original=True,
                        ))
                else:
                    replaced_total += alt_tons
                    n_replaced += 1
                    if record:
                        flagged_records.append(dict(
                            flight_id=r.flight_id, threshold_pct=k_pct,
                            replacement_tons=alt_tons, time_delta_hours=delta_h,
                            kept_original=False,
                        ))
            return kept_total + replaced_total, n_replaced, n_kept_original

        flagged_model = by_pred.head(n_flag)
        kept_model = by_pred.iloc[n_flag:]
        new_total_model, n_repl_m, n_keep_m = total_with_replacement(
            flagged_model, kept_model, record=(k_pct == 10),
        )

        flagged_oracle = by_truth.head(n_flag)
        kept_oracle = by_truth.iloc[n_flag:]
        new_total_oracle, n_repl_o, n_keep_o = total_with_replacement(
            flagged_oracle, kept_oracle,
        )

        rows.append(dict(
            threshold_pct=k_pct,
            n_flagged=n_flag,
            actual_tons=actual_total,
            after_model_tons=new_total_model,
            after_oracle_tons=new_total_oracle,
            model_reduction_pct=100 * (actual_total - new_total_model) / actual_total,
            oracle_reduction_pct=100 * (actual_total - new_total_oracle) / actual_total,
            n_model_replaced=n_repl_m,
            n_model_kept_original=n_keep_m,
            n_oracle_replaced=n_repl_o,
            n_oracle_kept_original=n_keep_o,
        ))

    return pd.DataFrame(rows), pd.DataFrame(flagged_records)


# ──────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────

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
    ax.set_xlabel("Flights replaced (ranked by predicted forcing)")
    ax.set_ylabel("Customer contrail-eq forcing reduction (%)")
    ax.set_title(label, pad=8)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    upper = max(summary["oracle_reduction_pct"].max() * 1.18, 105)
    ax.set_ylim(min(summary[["model_reduction_pct", "oracle_reduction_pct"]].min().min() - 5, 0), upper)
    ax.axhline(0, lw=0.6, color="#444")
    ax.grid(alpha=0.25, axis="y")


def time_delta_panel(ax, flagged: pd.DataFrame, label: str):
    """Histogram of |time delta| to nearest below-threshold same-route
    alternative for the top-10% model-flagged flights. Capped at 7 days."""
    sub = flagged[flagged["threshold_pct"] == 10].copy()
    n_total = len(sub)
    n_no_alt = sub["kept_original"].sum()
    cap_h = TIME_DELTA_CAP_DAYS * 24

    delta = sub["time_delta_hours"].dropna()
    delta_capped = np.minimum(delta, cap_h)

    bins = np.linspace(0, cap_h, 25)
    ax.hist(delta_capped, bins=bins, color="#d1495b", alpha=0.85,
            edgecolor="white", linewidth=0.6)

    # Show median
    median_h = delta.median() if len(delta) else np.nan
    if not np.isnan(median_h):
        ax.axvline(median_h, color="#444", lw=1.2, ls="--",
                   label=f"median {median_h:.1f} h")
    if n_no_alt > 0:
        # Sits BELOW the "median" legend entry, which occupies the
        # upper-right corner — anchoring both at 0.95 made them collide.
        ax.text(
            0.98, 0.82,
            f"+{n_no_alt} flights with no\nsame-route alternative",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#777",
        )

    # Add tick marks for human-readable durations
    ax.set_xticks([0, 24, 48, 72, 96, 120, 144, 168])
    ax.set_xticklabels(["0", "1d", "2d", "3d", "4d", "5d", "6d", "7d"])
    ax.set_xlim(0, cap_h)
    ax.set_xlabel("|Time delta| to nearest acceptable same-route alternative")
    ax.set_ylabel("Number of flagged flights")
    ax.set_title(label, pad=8)
    if not np.isnan(median_h):
        ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(alpha=0.25, axis="y")


def main():
    t0 = time.time()
    print(f"Loading model {MODEL_PATH.name} ...")
    bundle = joblib.load(MODEL_PATH)

    if not PRED_CACHE.exists():
        raise SystemExit(
            f"Predictions cache {PRED_CACHE.name} not found — run "
            "fig5_demand_shift.py first to build it."
        )

    print(f"Loading 2021 predictions {PRED_CACHE.name} ...")
    db = pd.read_parquet(PRED_CACHE)
    threshold_log = db["pred_log"].quantile(THRESHOLD_QUANTILE)
    print(f"  {len(db):,} predictions; "
          f"top-{int((1-THRESHOLD_QUANTILE)*100)}% threshold pred_log = {threshold_log:.4f}")

    # Pre-build the route-keyed acceptable-alternatives index once
    print("Building same-route acceptable-alternative index ...")
    db_acceptable = db[db["pred_log"] < threshold_log][[
        "origin_airport", "destination_airport", "first_waypoint_time",
        "contrail_CO2_km", "total_flight_distance_km", "pred_log",
    ]].copy()
    db_acc_by_route = {
        key: g.sort_values("first_waypoint_time")
        for key, g in db_acceptable.groupby(
            ["origin_airport", "destination_airport"]
        )
    }
    print(f"  {len(db_acceptable):,} acceptable alternatives across "
          f"{len(db_acc_by_route):,} routes")

    summaries = []
    flagged_all = []
    n_unique_by_cust = {}
    for cust in CUSTOMERS:
        print(f"\n=== {cust.label} ===")
        cust_log = cust.loader()
        merged = cust_log.merge(
            db[["flight_id", "origin_airport", "destination_airport",
                "first_waypoint_time", "pred_log"]],
            on="flight_id", how="inner",
        )
        # Drop NaN ground-truth forcing entries (a few per customer)
        merged = merged.dropna(subset=["co2_eq_tons", "first_waypoint_time"])
        n_unique_by_cust[cust.name] = len(merged)
        print(f"  {len(merged):,} customer flights matched & predicted")

        summary, flagged = replacement_summary(merged, db_acc_by_route)
        summary.insert(0, "customer", cust.name)
        flagged.insert(0, "customer_label", cust.label)
        flagged.insert(0, "customer", cust.name)
        summaries.append(summary)
        flagged_all.append(flagged)
        print(summary.to_string(index=False))

    summary_all = pd.concat(summaries, ignore_index=True)
    summary_all.to_csv(OUT / "fig4_summary.csv", index=False)
    flagged_concat = pd.concat(flagged_all, ignore_index=True)
    flagged_concat.to_csv(OUT / "fig4_flagged.csv", index=False)

    # ── 2 × 2 figure: bars left, time-delta right; rows = customer ────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for row, cust in enumerate(CUSTOMERS):
        cust_summary = summary_all[summary_all["customer"] == cust.name]
        cust_flagged = flagged_concat[flagged_concat["customer"] == cust.name]
        n_unique = n_unique_by_cust[cust.name]
        actual_total = cust_summary["actual_tons"].iloc[0]
        bar_panel(
            axes[row, 0], cust_summary,
            f"({chr(ord('a') + 2 * row)}) {cust.label} avoidance "
            f"(n={n_unique}, {actual_total:.0f} tCO₂e)",
        )
        time_delta_panel(
            axes[row, 1], cust_flagged,
            f"({chr(ord('a') + 2 * row + 1)}) {cust.label} time delta to alternative",
        )

    plt.tight_layout()
    fig.savefig(OUT / "fig4_combined.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig4_combined.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig4_combined.eps", bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote fig4_combined.{{png,pdf}}, fig4_summary.csv, "
          f"fig4_flagged.csv in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
