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

# ── (a) warming vs cooling formation share by predicted quintile ──
ax = axes[0]
df["quintile"] = pd.qcut(df["pred_log"], 5, labels=False)
warm = df.groupby("quintile").apply(lambda g: (g["py_ef_J"] > 0).mean())
cool = df.groupby("quintile").apply(lambda g: (g["py_ef_J"] < 0).mean())
nwarm = df.groupby("quintile").apply(lambda g: int((g["py_ef_J"] > 0).sum()))
ncool = df.groupby("quintile").apply(lambda g: int((g["py_ef_J"] < 0).sum()))
ax.bar(warm.index, 100 * warm.values, color="#D53E4F", width=0.72,
       label="net-warming contrail")
ax.bar(cool.index, -100 * cool.values, color="#3288BD", width=0.72,
       label="net-cooling contrail")
for q in warm.index:
    if nwarm[q]:
        ax.annotate(str(nwarm[q]), xy=(q, 100 * warm[q] + 1.2),
                    ha="center", fontsize=7.5, alpha=0.85)
    if ncool[q]:
        ax.annotate(str(ncool[q]), xy=(q, -100 * cool[q] - 4.2),
                    ha="center", fontsize=7.5, alpha=0.85)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(5))
ax.set_xticklabels(["Q1\n(lowest)", "Q2", "Q3", "Q4", "Q5\n(highest)"], fontsize=8)
ax.set_xlabel("Quintile of schedule-only predicted forcing")
ax.set_ylabel("Flights forming a persistent contrail (%)\nin independent simulation")
ax.set_ylim(-22, 45)
ax.legend(frameon=False, fontsize=8, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("(a) Formation discrimination", fontsize=10)

# ── (b) ranking among contrail-forming flights ──
# Both axes in the SAME units: the independent EF converts to kg CO2e
# per km with the labels' own AGWP-100 constant, and the prediction
# inverts from signed-log space (values with |kg| < 1 clamp to the
# symlog linear region, invisible at this scale).
AGWP_J_PER_KG = 82.5e-15 * 365 * 24 * 3600 * 5.101e14   # J per kg CO2e
nz = nz.copy()
nz["py_kg_km"] = (nz["py_ef_J"] / nz["dist_km"]) / AGWP_J_PER_KG
nz["pred_kg_km"] = np.where(nz["pred_log"] >= 0,
                            np.exp(nz["pred_log"]), -np.exp(-nz["pred_log"]))
ax = axes[1]
season = pd.to_datetime(nz["date"]).dt.month.map(
    {1: "Jan", 4: "Apr", 7: "Jul", 10: "Oct"})
colors = {"Jan": "#3288BD", "Apr": "#66C2A5", "Jul": "#FDAE61", "Oct": "#D53E4F"}
for s, g in nz.groupby(season):
    ax.scatter(g["pred_kg_km"], g["py_kg_km"], s=16, alpha=0.8,
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
ax.set_xlabel("Schedule-only predicted forcing\n(kg CO₂e per km)")
ax.set_ylabel("Independently simulated forcing\n(kg CO₂e per km)")
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
print(f"figS3 rewritten; warming shares by quintile: {[f'{100*v:.0f}%' for v in warm.values]}; "
      f"cooling: {[f'{100*v:.0f}%' for v in cool.values]}")
