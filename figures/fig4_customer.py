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

Layout (merged Figs 4+5): one panel per customer. Leftmost bar is the
customer's original contrail forcing (100%); then, for each avoidance
threshold (worst 1/5/10/20% of flights by predicted or true forcing),
bars show the post-avoidance total under rebooking windows of +/-2, 4,
8 and 16 h (light -> dark). Each bar stacks the forcing of untouched
flights (solid) and of the replacement flights actually flown (hatched
top). A flagged flight with no acceptable same-route alternative inside
the window keeps its original forcing. Because the nearest acceptable
alternative minimizes |dt|, one nearest-lookup per flight answers every
window: replaced iff nearest |dt| <= W.

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

THRESHOLDS_PCT = [1, 5, 10, 20]
WINDOWS_H = [2, 4, 8, 16]          # rebooking windows (+/- hours)


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


def window_grid(merged: pd.DataFrame, db_acc_by_route: dict) -> pd.DataFrame:
    """Post-avoidance totals for every (ranker, threshold, window).

    One nearest-alternative lookup per flagged flight serves all
    windows: the nearest candidate minimizes |dt|, so it is within
    +/-W iff |dt| <= W, and no candidate is otherwise.
    """
    n = len(merged)
    total = merged["co2_eq_tons"].sum()
    rows = []
    for ranker, keycol in [("model", "pred_log"), ("oracle", "co2_eq_tons")]:
        ordered = merged.sort_values(keycol, ascending=False).reset_index(drop=True)
        n_max = max(1, int(round(n * max(THRESHOLDS_PCT) / 100)))
        top = ordered.head(n_max)
        alt = [find_nearest_alternative(r, db_acc_by_route)
               for r in top.itertuples()]
        alt_tons = np.array([a for a, _ in alt])
        delta_h = np.array([d for _, d in alt])
        orig_tons = top["co2_eq_tons"].to_numpy()

        for k in THRESHOLDS_PCT:
            n_flag = max(1, int(round(n * k / 100)))
            at, dh, ot = alt_tons[:n_flag], delta_h[:n_flag], orig_tons[:n_flag]
            kept_total = total - ot.sum()
            for w in WINDOWS_H:
                repl = ~np.isnan(dh) & (dh <= w)
                repl_tons = at[repl].sum()
                base_tons = kept_total + ot[~repl].sum()
                rows.append(dict(
                    ranker=ranker, threshold_pct=k, window_h=w,
                    n_flagged=n_flag, n_replaced=int(repl.sum()),
                    base_tons=base_tons, replacement_tons=repl_tons,
                    total_after=base_tons + repl_tons,
                    pct_remaining=100 * (base_tons + repl_tons) / total,
                ))
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────

ORIG_COLOR = "#e8834f"
MODEL_SHADES = ["#f4b6bd", "#e98f9b", "#d1495b", "#9d3140"]   # ±2 → ±16 h
ORACLE_SHADES = ["#b9d0e4", "#83a9ca", "#3f6f9e", "#1f4e79"]


def merged_panel(ax, grid: pd.DataFrame, total_tons: float, label: str):
    bar_w = 0.115
    group_centers = {k: 1.15 + i * 1.25 for i, k in enumerate(THRESHOLDS_PCT)}

    # Original forcing reference bar
    ax.bar([0], [100], width=0.42, color=ORIG_COLOR, zorder=3)
    ax.text(0, 101.5, "100%", ha="center", fontsize=8)
    ax.axhline(100, color="#999999", ls="--", lw=0.8, zorder=1)

    for k in THRESHOLDS_PCT:
        for ri, (ranker, shades) in enumerate(
                [("model", MODEL_SHADES), ("oracle", ORACLE_SHADES)]):
            cluster = group_centers[k] + (ri - 0.5) * 0.58
            for wi, w in enumerate(WINDOWS_H):
                r = grid[(grid.ranker == ranker) & (grid.threshold_pct == k)
                         & (grid.window_h == w)].iloc[0]
                x = cluster + (wi - 1.5) * bar_w
                base_pct = 100 * r.base_tons / total_tons
                repl_pct = 100 * r.replacement_tons / total_tons
                ax.bar([x], [base_pct], width=bar_w, color=shades[wi], zorder=3)
                ax.bar([x], [repl_pct], width=bar_w, bottom=base_pct,
                       color=shades[wi], alpha=0.45, hatch="//////",
                       edgecolor="white", linewidth=0, zorder=3)
                ax.text(x, base_pct + max(repl_pct, 0) + 1.2,
                        f"{r.pct_remaining:.0f}", ha="center", va="bottom",
                        fontsize=6.2, rotation=90, color="#444444")

    ax.set_xticks([0] + [group_centers[k] for k in THRESHOLDS_PCT])
    ax.set_xticklabels(["Original"] + [f"worst {k}%\navoided"
                                       for k in THRESHOLDS_PCT])
    ax.set_ylim(0, 112)
    ax.set_ylabel("Contrail-equivalent forcing (% of original)")
    ax.set_title(label, pad=8)
    ax.grid(alpha=0.25, axis="y")
    ax.set_axisbelow(True)


def add_legend(fig):
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=ORIG_COLOR, label="Original forcing"),
        Patch(facecolor=MODEL_SHADES[2], label="Model-flagged avoidance"),
        Patch(facecolor=ORACLE_SHADES[2], label="Perfect-foresight avoidance"),
        Patch(facecolor=MODEL_SHADES[1], alpha=0.45, hatch="//////",
              edgecolor="white", label="Forcing of replacement flights"),
    ] + [
        Patch(facecolor=MODEL_SHADES[i], label=f"±{w} h rebooking window")
        for i, w in enumerate(WINDOWS_H)
    ]
    fig.legend(handles=handles, loc="upper center", frameon=False,
               fontsize=8.5, ncol=4, bbox_to_anchor=(0.5, 1.0))


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

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.0))
    grids = []
    for ax, cust, letter in zip(axes, CUSTOMERS, "ab"):
        print(f"\n=== {cust.label} ===")
        cust_log = cust.loader()
        merged = cust_log.merge(
            db[["flight_id", "origin_airport", "destination_airport",
                "first_waypoint_time", "pred_log"]],
            on="flight_id", how="inner",
        ).dropna(subset=["co2_eq_tons", "first_waypoint_time"])
        total = merged["co2_eq_tons"].sum()
        print(f"  {len(merged):,} flights matched, {total:.0f} tCO2e")

        grid = window_grid(merged, db_acc_by_route)
        grid.insert(0, "customer", cust.name)
        grids.append(grid)
        merged_panel(ax, grid, total,
                     f"({letter}) {cust.label} "
                     f"(n={len(merged)}, {total:.0f} tCO₂e)")

    add_legend(fig)
    pd.concat(grids, ignore_index=True).to_csv(
        OUT / "fig4_window_summary.csv", index=False)

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(OUT / "fig4.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig4.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig4.eps", bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote fig4.{{png,pdf,eps}} + fig4_window_summary.csv "
          f"in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
