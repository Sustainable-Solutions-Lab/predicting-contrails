#!/usr/bin/env python3
"""Supplementary Figure S3 — independent cross-model validation.

Two claims, two panels. (a) Discrimination: the share of flights that
form a persistent contrail in the independent simulation, by quintile
of our schedule-only prediction — rises monotonically from the bottom
to the top quintile. (b) Ranking among formers: independently simulated
per-km forcing versus our prediction for the flights that DO form
contrails, with the training labels' own correlation quoted as the
ceiling. Zero-EF flights (the 78% majority) appear only as panel (a)'s
denominators, not as a smear of points.
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.size": 9,
})

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "experiments/validation/crossval_results.parquet"
OUT = Path(
    "/Users/stevedavis/Library/CloudStorage/Dropbox/"
    "Papers/Active Prep/Contrails/WS Corp contrails (w Silas)/Plots"
)

df = pd.read_parquet(RES)
df["py_ef_per_m"] = df["py_ef_J"] / (df["dist_km"] * 1e3)
df["formed"] = df["py_ef_J"] != 0
nz = df[df["formed"]]
r_nz = spearmanr(nz["pred_log"], nz["py_ef_per_m"])
r_label_nz = spearmanr(nz["label_kg_km"], nz["py_ef_per_m"])

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

# ── (a) correspondence: predicted quintile vs independent outcome ──
# Literal quintiles of the independent forcing are undefined (78% of
# flights tie at exactly zero), so the outcome axis uses five ordered
# groups that respect the tie structure: net-cooling formers, no
# contrail, and net-warming formers split into terciles.
ax = axes[0]
df["quintile"] = pd.qcut(df["pred_log"], 5, labels=False)
warm_vals = df.loc[df["py_ef_J"] > 0, "py_ef_per_m"]
t1, t2 = warm_vals.quantile([1 / 3, 2 / 3])

def outcome(r):
    if r["py_ef_J"] < 0:
        return 0
    if r["py_ef_J"] == 0:
        return 1
    if r["py_ef_per_m"] <= t1:
        return 2
    if r["py_ef_per_m"] <= t2:
        return 3
    return 4

df["outcome"] = df.apply(outcome, axis=1)
ROWS = ["net-cooling\ncontrail", "no persistent\ncontrail",
        "net-warming:\nlow tercile", "net-warming:\nmid tercile",
        "net-warming:\ntop tercile"]
counts = np.zeros((5, 5), int)
for (q, o), n_ in df.groupby(["quintile", "outcome"]).size().items():
    counts[o, q] = n_
share = counts / counts.sum(axis=0, keepdims=True)   # column-normalized

from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
# formers colored on a crimson ramp; the dominant no-contrail base-rate
# row in neutral gray so it doesn't shout down the signal
cmap_f = LinearSegmentedColormap.from_list("share", ["#ffffff", "#9E0142"])
cmap_n = LinearSegmentedColormap.from_list("none", ["#ffffff", "#8a8a96"])
for o in range(5):
    for q in range(5):
        v = share[o, q]
        cm = cmap_n if o == 1 else cmap_f
        vmax = 1.0 if o == 1 else 0.4
        ax.add_patch(mpatches.Rectangle((q - 0.5, o - 0.5), 1, 1,
                     facecolor=cm(min(v / vmax, 1.0)), edgecolor="0.85", lw=0.5))
        if counts[o, q]:
            ax.text(q, o, f"{100 * v:.0f}%\n({counts[o, q]})",
                    ha="center", va="center", fontsize=6.8,
                    color="white" if (v / vmax) > 0.55 else "#202124")
ax.set_xlim(-0.5, 4.5)
ax.set_ylim(-0.5, 4.5)
ax.set_xticks(range(5))
ax.set_xticklabels(["Q1\n(lowest)", "Q2", "Q3", "Q4", "Q5\n(highest)"], fontsize=8)
ax.set_yticks(range(5))
ax.set_yticklabels(ROWS, fontsize=7.5)
ax.set_xlabel("Quintile of schedule-only predicted forcing")
ax.set_ylabel("Independent-simulation outcome")
ax.set_title("(a) Outcome by predicted quintile", fontsize=10)
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)

# ── (b) ranking among contrail-forming flights ──
# Both axes in the paper's native energy units (GJ per km): the
# prediction inverts from signed-log space and converts to energy with
# the fixed constant used in label construction (values with small
# magnitude clamp to the symlog linear region, invisible at this scale).
J_PER_UNIT = 82.5e-15 * 365 * 24 * 3600 * 5.101e14   # label-construction constant
nz = nz.copy()
nz["py_gj_km"] = (nz["py_ef_J"] / nz["dist_km"]) / 1e9
pred_unit = np.where(nz["pred_log"] >= 0,
                     np.exp(nz["pred_log"]), -np.exp(-nz["pred_log"]))
nz["pred_gj_km"] = pred_unit * J_PER_UNIT / 1e9
ax = axes[1]
season = pd.to_datetime(nz["date"]).dt.month.map(
    {1: "Jan", 4: "Apr", 7: "Jul", 10: "Oct"})
colors = {"Jan": "#3288BD", "Apr": "#66C2A5", "Jul": "#FDAE61", "Oct": "#D53E4F"}
for s, g in nz.groupby(season):
    ax.scatter(g["pred_gj_km"], g["py_gj_km"], s=16, alpha=0.8,
               color=colors.get(s, "#888"), label=s, linewidths=0)
lim = 1000
ax.plot([-lim, lim], [-lim, lim], color="k", lw=0.7, ls="--", alpha=0.5)
ax.annotate("1:1", xy=(120, 45), fontsize=8, alpha=0.6, rotation=45)
ax.set_xscale("symlog", linthresh=1)
ax.set_yscale("symlog", linthresh=1)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.axhline(0, color="k", lw=0.5, alpha=0.3)
ax.axvline(0, color="k", lw=0.5, alpha=0.3)
ax.set_xlabel("Schedule-only predicted forcing\n(GJ per km)")
ax.set_ylabel("Independently simulated forcing\n(GJ per km)")
ax.annotate(
    f"Spearman ρ = {r_nz.statistic:+.2f} (n = {len(nz)})\n"
    f"training labels reach ρ = {r_label_nz.statistic:+.2f} (ceiling)",
    xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("(b) Forcing rank among formers", fontsize=10)

fig.tight_layout()
for ext in ("png", "pdf", "eps"):
    fig.savefig(OUT / f"figS3_pycontrails.{ext}", dpi=200, bbox_inches="tight")
print("figS3 rewritten; Q5 top-tercile share:", f"{100*share[4,4]:.0f}%", "| Q1:", f"{100*share[4,0]:.0f}%")
