"""
Global companion to the Fig 4 contours: % reduction in TOTAL 2021
contrail forcing over the (worst-x% avoided, ±rebooking window) plane,
for ALL ~22M flights in the predictions corpus — not just the two
customer case studies.

Ranking ("worst") is per-km on both panels: predicted signed-log per-km
forcing for the model, actual per-km forcing for perfect foresight.

Replacement rule (simplified vs. the customer figure, for tractability
at 22M flights): an avoided flight is rebooked at its route's AVERAGE
acceptable alternative — provided at least one acceptable alternative
exists within the window (one nearest-neighbor lookup answers every
window). The oracle declines swaps that would not help. This mirrors
the "any-time potential" dash of the bar figure rather than
best-in-window choice, so these surfaces are mildly conservative for
the model and avoid the best-day cherry-picking degeneracy.

Output: Plots/fig4_contours_global.{png,pdf,eps}
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

# TrueType (Type 42) fonts in EPS/PDF — Illustrator hangs/garbles on
# matplotlib's default Type 3 glyph programs
plt.rcParams.update({"ps.fonttype": 42, "pdf.fonttype": 42})
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from feature_pruning import DROPBOX_PLOTS, OUT as EXP_OUT  # noqa: E402

OUT = DROPBOX_PLOTS
PRED_CACHE = EXP_OUT / "_2021_predictions.parquet"
THRESHOLD_QUANTILE = 0.90

THRESHOLDS = np.arange(1, 31)
WINDOWS = np.unique(np.round(np.logspace(0, np.log10(48), 40), 2))


def main():
    t0 = time.time()
    print(f"Loading {PRED_CACHE.name} ...")
    db = pd.read_parquet(PRED_CACHE, columns=[
        "origin_airport", "destination_airport", "first_waypoint_time",
        "pred_log", "contrail_CO2_km", "total_flight_distance_km",
    ]).dropna()
    db["tons"] = db["contrail_CO2_km"] * db["total_flight_distance_km"] / 1000.0
    db["t_h"] = db["first_waypoint_time"].astype("int64") / 3.6e12
    total = db["tons"].sum()
    thr = db["pred_log"].quantile(THRESHOLD_QUANTILE)
    print(f"  {len(db):,} flights, net total {total/1e6:.1f} Mt CO2e-contrail, "
          f"{time.time()-t0:.0f}s")

    # Per route: sorted acceptable departure times + mean acceptable tons.
    # Then for EVERY flight: |dt| to nearest acceptable alternative.
    print("Route index + nearest-acceptable lookups ...")
    nearest_dt = np.full(len(db), np.inf)
    mean_acc = np.full(len(db), np.nan)
    acc_mask = (db["pred_log"] < thr).to_numpy()
    codes, _ = pd.factorize(
        db["origin_airport"].astype(str) + ">" + db["destination_airport"].astype(str),
        sort=False)
    order = np.argsort(codes, kind="stable")
    t_h = db["t_h"].to_numpy()
    tons = db["tons"].to_numpy()
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    for grp in np.split(order, bounds):
        acc = grp[acc_mask[grp]]
        if len(acc) == 0:
            continue
        ta = np.sort(t_h[acc])
        mean_acc[grp] = tons[acc].mean()
        tm = t_h[grp]
        j = np.searchsorted(ta, tm).clip(1, len(ta) - 1) if len(ta) > 1 else None
        if j is None:
            nearest_dt[grp] = np.abs(tm - ta[0])
        else:
            left = np.abs(tm - ta[j - 1])
            right = np.abs(tm - ta[j])
            nearest_dt[grp] = np.minimum(left, right)
    print(f"  done in {time.time()-t0:.0f}s")

    # Surfaces
    surfaces = {}
    for ranker, key in [("model", db["pred_log"].to_numpy()),
                        ("oracle", (db["contrail_CO2_km"]).to_numpy())]:
        rk = np.argsort(key)[::-1]
        n_max = int(round(len(db) * THRESHOLDS.max() / 100))
        sel = rk[:n_max]
        sv = tons[sel] - mean_acc[sel]                 # saving if swapped
        if ranker == "oracle":
            sv = np.maximum(sv, 0.0)                   # decline bad swaps
        sv = np.where(np.isnan(mean_acc[sel]), 0.0, sv)
        dts = nearest_dt[sel]
        R = np.zeros((len(THRESHOLDS), len(WINDOWS)))
        for wi, w in enumerate(WINDOWS):
            s_w = np.where(dts <= w, sv, 0.0)
            cum = np.cumsum(s_w)
            for ki, k in enumerate(THRESHOLDS):
                n_flag = int(round(len(db) * k / 100))
                R[ki, wi] = 100 * cum[n_flag - 1] / total
        surfaces[ranker] = R
        print(f"  {ranker}: top-10% ±16h reduction "
              f"{R[np.searchsorted(THRESHOLDS, 10), np.searchsorted(WINDOWS, 16)]:.1f}%")

    # ── Render (same styling as the customer contours) ────────────────
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.ndimage import gaussian_filter
    hex_ramp = ["#9e0142", "#d53e4f", "#f46d43", "#fdae61", "#fee08b",
                "#ffffbf", "#e6f598", "#abdda4", "#66c2a5", "#3288bd"]
    cmap = LinearSegmentedColormap.from_list("reduction", hex_ramp, N=256)
    cmap.set_under("#6e0028")
    surfaces = {k: gaussian_filter(R, sigma=1.4) for k, R in surfaces.items()}
    vmax = max(R.max() for R in surfaces.values())
    levels = np.arange(0, np.ceil(vmax / 5) * 5 + 5, 5)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    K, W = np.meshgrid(THRESHOLDS, WINDOWS, indexing="ij")
    for ax, (ranker, rname) in zip(axes, [("model", "Model"),
                                          ("oracle", "Perfect foresight")]):
        R = surfaces[ranker]
        cf = ax.contourf(K, W, R, levels=levels, cmap=cmap, extend="min")
        lines = ax.contour(K, W, R, levels=levels, colors="black",
                           linewidths=0.6, alpha=0.6)
        ax.clabel(lines, fmt="%.0f%%", fontsize=7.5, colors="black")
        ax.set_yscale("log")
        ax.set_yticks([1, 2, 4, 8, 16, 32, 48])
        ax.set_yticklabels(["1", "2", "4", "8", "16", "32", "48"])
        ax.minorticks_off()
        ax.set_title(f"({'ab'[ranker == 'oracle']}) All 2021 flights — {rname}",
                     fontsize=11)
        ax.set_xlabel("Worst flights avoided (%)")
    axes[0].set_ylabel("Rebooking window (±h)")
    cbar = fig.colorbar(cf, ax=axes, shrink=0.88, pad=0.02)
    cbar.set_label("Reduction in global contrail-equivalent forcing (%)")

    for ext in ("png", "pdf", "eps"):
        fig.savefig(OUT / f"fig4_contours_global.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote fig4_contours_global.{{png,pdf,eps}} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
