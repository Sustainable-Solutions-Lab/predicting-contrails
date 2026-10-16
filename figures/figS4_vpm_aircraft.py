#!/usr/bin/env python3
"""Supplementary Figure S4: the aircraft-type contrast depends on
volatile-particle (vPM) activation.

Each synthetic trajectory from the pycontrails cross-check is flown by a
fixed set of aircraft/engine combinations in CoCiP, with and without vPM
activation (experiments/validation/vpm_sensitivity.py). Points are each
combination's total contrail energy forcing relative to a reference
aircraft on the same trajectories (737-800 for narrowbodies on routes
< 4000 km; 777-300ER for widebodies on routes > 3000 km); bars are 90%
bootstrap intervals over trajectories. Without vPM, the lean-burn engines
(LEAP, GEnx) produce a few percent of the reference forcing, the
order-of-magnitude contrast our training labels and Supplementary Fig. 1
inherit. With vPM, the contrast mostly disappears.
"""
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "font.size": 9,
})

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "experiments/validation/vpm_results.parquet"
OUT = Path(
    "/Users/stevedavis/Library/CloudStorage/Dropbox/"
    "Papers/In Review/Contrails/WS Corp contrails (w Silas)/Plots"
)
LEAN = ("LEAP", "GEnx")
CONFIGS = [("base", "Without vPM activation (as in labels)", "#9E0142"),
           ("vpm", "With vPM activation", "#3288BD")]
PANELS = [
    ("a", "Narrowbodies (routes < 4,000 km)", "737-800 (CFM56-7B)",
     lambda d: d < 4000,
     ["737-800 (CFM56-7B)", "A320 (CFM56-5B)", "A321 (V2500)",
      "A320neo (PW1127G)", "A320neo (LEAP-1A)", "A321neo (LEAP-1A)",
      "737 MAX 8 (LEAP-1B)"]),
    ("b", "Widebodies (routes > 3,000 km)", "777-300ER (GE90)",
     lambda d: d > 3000,
     ["777-300ER (GE90)", "747-400 (CF6-80C2)", "A330-300 (CF6-80E)",
      "A350-900 (Trent XWB)", "787-9 (Trent 1000)", "787-9 (GEnx-1B)"]),
]


def ratios(s, ref, labels, cfg, rng, nboot=2000):
    p = (s[s["config"] == cfg]
         .pivot_table(index="flight_id", columns="label", values="ef_J",
                      aggfunc="sum"))
    idx = rng.integers(0, len(p), size=(nboot, len(p)))
    refv = p[ref].to_numpy()
    rows = []
    for lab in labels:
        v = p[lab].to_numpy()
        bs = v[idx].sum(1) / refv[idx].sum(1)
        rows.append({"label": lab, "config": cfg, "n": len(p),
                     "ratio": v.sum() / refv.sum(),
                     "lo": np.percentile(bs, 5), "hi": np.percentile(bs, 95)})
    return rows


def main() -> None:
    r = pd.read_parquet(RES)
    s = r[r["experiment"] == "swap"]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.7))
    table = []
    for ax, (letter, title, ref, keep, labels) in zip(axes, PANELS):
        sub = s[keep(s["dist_km"]) & s["label"].isin(labels)]
        y = np.arange(len(labels))[::-1]
        for k, (cfg, name, col) in enumerate(CONFIGS):
            rows = ratios(sub, ref, labels, cfg, rng)
            table += [dict(row, panel=letter) for row in rows]
            off = 0.14 if k == 0 else -0.14
            for yi, row in zip(y, rows):
                ax.hlines(yi + off, row["lo"], row["hi"], color=col, lw=1.6)
                ax.scatter(row["ratio"], yi + off, s=26, color=col, zorder=3,
                           edgecolor="white", linewidth=1,
                           label=name if yi == y[0] else None)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        for t in ax.get_yticklabels():
            if any(e in t.get_text() for e in LEAN):
                t.set_fontweight("bold")
        ax.axvline(1, color="#999999", lw=0.8)
        ax.set_xlim(0, 2.6)
        ax.set_xlabel(f"Contrail energy forcing relative to {ref.split(' (')[0]}")
        ax.set_title(f"{title}, n = {sub['flight_id'].nunique()}",
                     fontsize=9, loc="left")
        ax.grid(axis="x", color="#dddddd", lw=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(-0.02, 1.06, letter, transform=ax.transAxes, fontsize=11,
                fontweight="bold", ha="right")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(w_pad=2.5)
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf", "eps"):
        fig.savefig(OUT / f"figS4_vpm_aircraft.{ext}", dpi=200,
                    bbox_inches="tight")
    pd.DataFrame(table).to_csv(OUT / "figS4_vpm_aircraft.csv", index=False)
    print(pd.DataFrame(table).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
