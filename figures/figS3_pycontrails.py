#!/usr/bin/env python3
"""Supplementary Figure S3 — independent cross-model validation.

Scatter of independently simulated per-km contrail forcing (pycontrails
CoCiP + ARCO ERA5 + Poll–Schumann performance) against our schedule-only
prediction, for the stratified 2021 sample produced by
experiments/validation/pycontrails_crossval.py. Spearman rank
correlations annotated; symlog y-axis because EF spans orders of
magnitude and includes zeros/cooling.
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
r_pred = spearmanr(df["pred_log"], df["py_ef_per_m"])
r_label = spearmanr(df["label_kg_km"], df["py_ef_per_m"])
nz = df[df["py_ef_J"] != 0]
r_pred_nz = spearmanr(nz["pred_log"], nz["py_ef_per_m"])
r_label_nz = spearmanr(nz["label_kg_km"], nz["py_ef_per_m"])

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
season = pd.to_datetime(df["date"]).dt.month.map(
    {1: "Jan", 4: "Apr", 7: "Jul", 10: "Oct"})
colors = {"Jan": "#3288BD", "Apr": "#66C2A5", "Jul": "#FDAE61", "Oct": "#D53E4F"}

for ax, xcol, xlabel, r, rnz in (
    (axes[0], "pred_log", "Schedule-only prediction (signed-log per-km forcing)", r_pred, r_pred_nz),
    (axes[1], "label_kg_km", "Training label (kg CO₂e per km)", r_label, r_label_nz),
):
    for s, g in df.groupby(season):
        ax.scatter(g[xcol], g["py_ef_per_m"], s=12, alpha=0.7,
                   color=colors.get(s, "#888"), label=s, linewidths=0)
    ax.set_yscale("symlog", linthresh=1e5)
    if xcol == "label_kg_km":
        ax.set_xscale("symlog", linthresh=1e-2)
    ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    ax.set_xlabel(xlabel)
    ax.annotate(f"Spearman ρ = {r.statistic:+.2f} (all, n = {len(df)})\n"
                f"ρ = {rnz.statistic:+.2f} (contrail-forming, n = {len(nz)})",
                xy=(0.03, 0.96), xycoords="axes fraction", va="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))

axes[0].set_ylabel("pycontrails CoCiP + ERA5\nenergy forcing (J per m flown)")
axes[0].legend(frameon=False, fontsize=8, loc="lower right", title=None)
axes[0].set_title("(a) vs. schedule-only model", fontsize=10)
axes[1].set_title("(b) vs. process-model labels", fontsize=10)
fig.tight_layout()
for ext in ("png", "pdf", "eps"):
    fig.savefig(OUT / f"figS3_pycontrails.{ext}", dpi=200, bbox_inches="tight")
print(f"figS3 written; rho_pred={r_pred.statistic:+.3f} rho_label={r_label.statistic:+.3f} n={len(df)}")
