"""
Fig 4 — Apply the canonical model to two corporate flight logs, with
realistic alternative-flight replacement.

For each customer's predicted top-K% flights, instead of pretending the
flight didn't happen (the prior version's upper-bound assumption), we
identify the best below-threshold same-route alternative in the 2021
corpus within the rebooking window — lowest PREDICTED forcing for the
model actor, lowest ACTUAL for the perfect-foresight actor (who also
declines swaps that would not help) — and substitute its ACTUAL
contrail forcing into the customer total. The reduction is then the customer's real footprint
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

Reads from the 2021-wide predictions cache built by figS1_demand_shift.py
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

def find_candidates(flight_row, db_acc_by_route: dict[tuple, pd.DataFrame]):
    """All acceptable same-route alternatives for one flight: arrays of
    (|time delta| hours, actual forcing in tons, predicted signed-log
    per-km forcing). Empty arrays if the route has no acceptable
    alternatives."""
    key = (flight_row.origin_airport, flight_row.destination_airport)
    cands = db_acc_by_route.get(key)
    if cands is None or len(cands) == 0:
        z = np.empty(0)
        return z, z, z
    delta_h = np.abs(
        (cands["first_waypoint_time"] - flight_row.first_waypoint_time)
        .dt.total_seconds().to_numpy() / 3600.0)
    actual_tons = (cands["contrail_CO2_km"]
                   * cands["total_flight_distance_km"]).to_numpy() / 1000.0
    return delta_h, actual_tons, cands["pred_log"].to_numpy()


def window_grid(merged: pd.DataFrame, db_acc_by_route: dict) -> pd.DataFrame:
    """Post-avoidance totals for every (ranker, threshold, window).

    Replacement choice models a rational actor, not nearest-in-time:
      model  — swap to the alternative with the LOWEST PREDICTED forcing
               within the window (same route -> same distance, so the
               per-km prediction ranks totals exactly); realized forcing
               of that choice counts, so honest backfires remain
               possible when the prediction is wrong.
      oracle — swap to the LOWEST ACTUAL forcing alternative within the
               window, and only if it beats the flagged flight; the
               perfect-foresight bars therefore can never exceed 100%.
    """
    n = len(merged)
    total = merged["co2_eq_tons"].sum()
    rows = []
    for ranker, keycol in [("model", "pred_log"), ("oracle", "actual_per_km")]:
        ordered = merged.sort_values(keycol, ascending=False).reset_index(drop=True)
        n_max = max(1, int(round(n * max(THRESHOLDS_PCT) / 100)))
        top = ordered.head(n_max)
        cands = [find_candidates(r, db_acc_by_route) for r in top.itertuples()]
        orig_tons = top["co2_eq_tons"].to_numpy()

        # replacement_tons[i, wi]: NaN = keep original. Last column is
        # the ANY-TIME potential (dash marker): rebook the same route on
        # an AVERAGE acceptable day. (Picking the single best flight of
        # the whole year would be degenerate — it cherry-picks extreme
        # cooling days and can push totals negative.)
        all_windows = WINDOWS_H + [np.inf]
        repl_tbl = np.full((len(top), len(all_windows)), np.nan)
        for i, (dh, actual, pred) in enumerate(cands):
            if len(dh) == 0:
                continue
            for wi, w in enumerate(WINDOWS_H):
                m = dh <= w
                if not m.any():
                    continue
                if ranker == "model":
                    repl_tbl[i, wi] = actual[m][np.argmin(pred[m])]
                else:
                    best = actual[m].min()
                    if best < orig_tons[i]:
                        repl_tbl[i, wi] = best
            mean_alt = actual.mean()
            if ranker == "model" or mean_alt < orig_tons[i]:
                repl_tbl[i, -1] = mean_alt

        for k in THRESHOLDS_PCT:
            n_flag = max(1, int(round(n * k / 100)))
            ot = orig_tons[:n_flag]
            kept_total = total - ot.sum()
            for wi, w in enumerate(all_windows):
                rt = repl_tbl[:n_flag, wi]
                repl = ~np.isnan(rt)
                repl_tons = rt[repl].sum()
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
            # unlimited-window potential: short dash across the cluster
            ru = grid[(grid.ranker == ranker) & (grid.threshold_pct == k)
                      & (grid.window_h == np.inf)].iloc[0]
            ax.plot([cluster - 2.1 * bar_w, cluster + 2.1 * bar_w],
                    [ru.pct_remaining] * 2, color="#222222", lw=1.8,
                    zorder=6, solid_capstyle="butt")
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
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], color="#222222", lw=1.8,
                          label="Any-time potential (avg. alternative)"))
    fig.legend(handles=handles, loc="upper center", frameon=False,
               fontsize=8.5, ncol=5, bbox_to_anchor=(0.5, 1.0))


# ──────────────────────────────────────────────────────────────────────────
# Contour figure: % reduction over the (threshold, window) plane
# ──────────────────────────────────────────────────────────────────────────

CONTOUR_THRESHOLDS = np.arange(1, 31)                    # % of worst flights
CONTOUR_WINDOWS = np.unique(np.round(np.logspace(0, np.log10(48), 40), 2))

def reduction_surface(merged: pd.DataFrame, db_acc_by_route: dict,
                      ranker: str) -> np.ndarray:
    """R[k, w] = % reduction in total forcing avoiding the worst k% of
    flights (by predicted or actual forcing) with rebooking window ±w.

    Per flagged flight, candidates are sorted by |dt| once; prefix
    running-minima then give the rational choice at EVERY window:
      model  — actual forcing of the lowest-PREDICTED candidate so far
      oracle — lowest-ACTUAL candidate so far, declined if not better.
    """
    n = len(merged)
    total = merged["co2_eq_tons"].sum()
    keycol = "pred_log" if ranker == "model" else "actual_per_km"
    ordered = merged.sort_values(keycol, ascending=False).reset_index(drop=True)
    n_max = max(1, int(round(n * CONTOUR_THRESHOLDS.max() / 100)))
    top = ordered.head(n_max)

    savings = np.zeros((len(top), len(CONTOUR_WINDOWS)))
    for i, r in enumerate(top.itertuples()):
        dh, actual, pred = find_candidates(r, db_acc_by_route)
        if len(dh) == 0:
            continue
        o = np.argsort(dh)
        dhs, act, prd = dh[o], actual[o], pred[o]
        if ranker == "model":
            # actual forcing of the running lowest-predicted candidate
            idx = np.arange(len(prd))
            run_min = np.minimum.accumulate(prd)
            new_best = prd <= run_min          # True where a new minimum set
            arg_prefix = np.maximum.accumulate(np.where(new_best, idx, -1))
            chosen_act = act[arg_prefix]
        else:
            chosen_act = np.minimum.accumulate(act)
        j = np.searchsorted(dhs, CONTOUR_WINDOWS, side="right") - 1
        has = j >= 0
        sv = np.zeros(len(CONTOUR_WINDOWS))
        sv[has] = r.co2_eq_tons - chosen_act[j[has]]
        if ranker == "oracle":
            sv = np.maximum(sv, 0.0)           # oracle declines bad swaps
        savings[i] = sv

    cum = np.cumsum(savings, axis=0)
    R = np.zeros((len(CONTOUR_THRESHOLDS), len(CONTOUR_WINDOWS)))
    for ki, k in enumerate(CONTOUR_THRESHOLDS):
        n_flag = max(1, int(round(n * k / 100)))
        R[ki] = 100 * cum[min(n_flag, len(top)) - 1] / total
    return R


def global_surfaces(db: pd.DataFrame, threshold_log: float) -> dict:
    """Reduction surfaces over ALL flights in the 2021 corpus.

    Replacement is simplified for 22M-flight scale: an avoided flight is
    rebooked at its route's AVERAGE acceptable alternative, provided at
    least one acceptable alternative exists within the window (a single
    nearest-neighbor lookup answers every window). The oracle declines
    unhelpful swaps. Mildly conservative for the model vs. the
    best-in-window rule used for the customer panels.
    """
    g = db.dropna(subset=["first_waypoint_time", "contrail_CO2_km",
                          "total_flight_distance_km", "pred_log"]).copy()
    tons = (g["contrail_CO2_km"] * g["total_flight_distance_km"]).to_numpy() / 1e3
    t_h = g["first_waypoint_time"].astype("int64").to_numpy() / 3.6e12
    total = tons.sum()
    acc_mask = (g["pred_log"] < threshold_log).to_numpy()

    nearest_dt = np.full(len(g), np.inf)
    mean_acc = np.full(len(g), np.nan)
    codes, _ = pd.factorize(g["origin_airport"].astype(str) + ">"
                            + g["destination_airport"].astype(str), sort=False)
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    for grp in np.split(order, bounds):
        acc = grp[acc_mask[grp]]
        if len(acc) == 0:
            continue
        ta = np.sort(t_h[acc])
        mean_acc[grp] = tons[acc].mean()
        tm = t_h[grp]
        if len(ta) == 1:
            nearest_dt[grp] = np.abs(tm - ta[0])
        else:
            j = np.searchsorted(ta, tm).clip(1, len(ta) - 1)
            nearest_dt[grp] = np.minimum(np.abs(tm - ta[j - 1]),
                                         np.abs(tm - ta[j]))

    out = {}
    for ranker, key in [("model", g["pred_log"].to_numpy()),
                        ("oracle", g["contrail_CO2_km"].to_numpy())]:
        rk = np.argsort(key)[::-1]
        n_max = int(round(len(g) * CONTOUR_THRESHOLDS.max() / 100))
        sel = rk[:n_max]
        sv = tons[sel] - mean_acc[sel]
        if ranker == "oracle":
            sv = np.maximum(sv, 0.0)
        sv = np.where(np.isnan(mean_acc[sel]), 0.0, sv)
        dts = nearest_dt[sel]
        R = np.zeros((len(CONTOUR_THRESHOLDS), len(CONTOUR_WINDOWS)))
        for wi, w in enumerate(CONTOUR_WINDOWS):
            cum = np.cumsum(np.where(dts <= w, sv, 0.0))
            for ki, k in enumerate(CONTOUR_THRESHOLDS):
                n_flag = int(round(len(g) * k / 100))
                R[ki, wi] = 100 * cum[n_flag - 1] / total
        out[("All 2021 flights", ranker)] = R
    return out


def contour_figure(surfaces: dict):
    """surfaces: {(cust_label, ranker): R}"""
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.ndimage import gaussian_filter
    # Reversed ramp: red = little/no reduction, blue = deep reduction —
    # the net-zero line then sits in the last blues (net cooling beyond).
    hex_ramp = ["#9e0142", "#d53e4f", "#f46d43", "#fdae61", "#fee08b",
                "#ffffbf", "#e6f598", "#abdda4", "#66c2a5", "#3288bd"]
    cmap = LinearSegmentedColormap.from_list("reduction", hex_ramp, N=256)
    cmap.set_under("#6e0028")

    # Display-only smoothing: the raw surfaces are step functions of
    # individual flights entering the flagged set / window, which reads
    # as plotting artifacts. Bars and CSV stay exact.
    surfaces = {k: gaussian_filter(R, sigma=1.4) for k, R in surfaces.items()}

    vmax = max(R.max() for R in surfaces.values())
    levels = np.arange(0, np.ceil(vmax / 10) * 10 + 10, 10)

    row_labels = list(dict.fromkeys(k[0] for k in surfaces))
    fig, axes = plt.subplots(len(row_labels), 2,
                             figsize=(12.5, 4.4 * len(row_labels)),
                             sharex=True, sharey=True)
    K, W = np.meshgrid(CONTOUR_THRESHOLDS, CONTOUR_WINDOWS, indexing="ij")
    cust_labels = row_labels
    for row, cl in enumerate(cust_labels):
        for col, (ranker, rname) in enumerate(
                [("model", "Model"), ("oracle", "Perfect foresight")]):
            ax = axes[row, col]
            R = surfaces[(cl, ranker)]
            cf = ax.contourf(K, W, R, levels=levels, cmap=cmap, extend="min")
            thin = [l for l in levels if not (R.max() > 100 and l == 100)]
            cl_lines = ax.contour(K, W, R, levels=thin, colors="black",
                                  linewidths=0.6, alpha=0.6)
            ax.clabel(cl_lines, fmt="%.0f%%", fontsize=7.5, colors="black")
            if R.max() > 100:
                # beyond this line replacements are net-COOLING: the
                # customer's residual contrail forcing goes negative
                l100 = ax.contour(K, W, R, levels=[100], colors="black",
                                  linewidths=1.8)
                ax.clabel(l100, fmt="net zero", fontsize=8, colors="black")
            ax.set_yscale("log")
            ax.set_yticks([1, 2, 4, 8, 16, 32, 48])
            ax.set_yticklabels(["1", "2", "4", "8", "16", "32", "48"])
            ax.minorticks_off()
            ax.set_title(f"({chr(ord('a') + 2*row + col)}) {cl} — {rname}",
                         fontsize=11)
            if row == len(cust_labels) - 1:
                ax.set_xlabel("Worst flights avoided (%)")
            if col == 0:
                ax.set_ylabel("Rebooking window (±h)")
    cbar = fig.colorbar(cf, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label("Reduction in contrail-equivalent forcing (%)")

    for ext in ("png", "pdf", "eps"):
        fig.savefig(OUT / f"fig4_contours.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote fig4_contours.{{png,pdf,eps}}")


def main():
    t0 = time.time()
    print(f"Loading model {MODEL_PATH.name} ...")
    bundle = joblib.load(MODEL_PATH)

    if not PRED_CACHE.exists():
        raise SystemExit(
            f"Predictions cache {PRED_CACHE.name} not found — run "
            "figS1_demand_shift.py first to build it."
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
    surfaces = {}
    for ax, cust, letter in zip(axes, CUSTOMERS, "ab"):
        print(f"\n=== {cust.label} ===")
        cust_log = cust.loader()
        merged = cust_log.merge(
            db[["flight_id", "origin_airport", "destination_airport",
                "first_waypoint_time", "pred_log",
                "total_flight_distance_km"]],
            on="flight_id", how="inner",
        ).dropna(subset=["co2_eq_tons", "first_waypoint_time"])
        # actual per-km forcing: the oracle ranks on this so that both
        # columns define "worst" on the same per-km basis as the model
        merged["actual_per_km"] = (merged["co2_eq_tons"]
                                   / merged["total_flight_distance_km"])
        total = merged["co2_eq_tons"].sum()
        print(f"  {len(merged):,} flights matched, {total:.0f} tCO2e")

        grid = window_grid(merged, db_acc_by_route)
        grid.insert(0, "customer", cust.name)
        grids.append(grid)
        for ranker in ("model", "oracle"):
            surfaces[(cust.label, ranker)] = reduction_surface(
                merged, db_acc_by_route, ranker)
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
    print("Global reduction surfaces (all 2021 flights) ...")
    surfaces.update(global_surfaces(db, threshold_log))
    contour_figure(surfaces)
    print(f"\nWrote fig4.{{png,pdf,eps}} + fig4_window_summary.csv "
          f"in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
