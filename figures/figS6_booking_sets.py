"""
SI figure: the booking situation. How well each schedule-only ranker
does when a traveller chooses among the itineraries actually offered for
a route and day (or week) in 2025, judged out of time (model and
climatologies fitted on 2019-23). Data from experiments/booking_eval.py.

(a) Cumulative share of 2025 net forcing, worst-predicted first, for
    every ranker (the Fig. 3a comparison, out of time, with the
    Travel-Impact-Model-style climatology added)
(b) Share of the same-day (and same-week) reduction available with
    perfect foresight that each ranker's pick captures
(c) The same by number of itineraries in the day's choice set
(d) The same by route length

Run:  python figures/figS6_booking_sets.py [--suffix _s1500000_debug]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments"))
sys.path.insert(0, str(Path.home() / ".claude/skills/paper-figures/scripts"))
import sjd_style as sjd  # noqa: E402
from common import OUT as EXP_OUT, REVISION_PLOTS  # noqa: E402

NAMES = {"perfect": "Perfect foresight", "model": "Scheduling model",
         "tim": "TIM-style climatology", "route": "Route climatology",
         "clim": "Latitude-time climatology", "insolation": "Insolation only"}
COLORS = {"perfect": "#1f4e79", "model": "#d1495b", "tim": "#fdae61",
          "route": "#66c2a5", "clim": "#fee08b", "insolation": "#8fc08f"}
ORDER = ["model", "tim", "clim", "route", "insolation"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    sets = pd.read_csv(EXP_OUT / f"booking_sets_2025{a.suffix}.csv")
    by = pd.read_csv(EXP_OUT / f"booking_sets_by_size_2025{a.suffix}.csv")
    curves = pd.read_csv(EXP_OUT / f"booking_curves_ood{a.suffix}.csv")

    with sjd.style():
        fig, axes = sjd.panels(2, 2, width=sjd.DOUBLE_COLUMN * 0.8)
        # (a) cumulative curves, out of time
        ax = axes[0, 0]
        x = curves["share_of_flights_pct"].to_numpy()
        for key in ["perfect"] + ORDER:
            ax.plot(x, curves[key], color=COLORS[key], lw=1.4, label=NAMES[key])
        ax.plot([0, 100], [0, 100], color="#888888", lw=0.8, ls="--")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 120)
        ax.set_xlabel("Share of 2025 flights, worst-predicted first (%)")
        ax.set_ylabel("Cumulative contrail forcing (% of net)")
        ax.set_title("Ranking skill out of time")
        ax.legend(frameon=False, loc="lower right", fontsize=5.5)
        sjd.letter(ax, "a", "top left")

        # (b) share of available reduction captured, day vs week
        ax = axes[0, 1]
        w = 0.36
        for j, key in enumerate(("day", "week")):
            sub = sets[sets.key == key].set_index("ranker")
            vals = [sub.loc[r, "reduction_vs_mean_pct"] for r in ORDER]
            xs = np.arange(len(ORDER)) + (j - 0.5) * w
            bars = ax.bar(xs, vals, width=w, color=[COLORS[r] for r in ORDER],
                          alpha=1.0 if key == "day" else 0.5,
                          edgecolor="black", linewidth=0.4)
            for xi, v in zip(xs, vals):
                ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=5.5)
        ax.set_xticks(np.arange(len(ORDER)))
        ax.set_xticklabels([NAMES[r].replace(" ", "\n", 1) for r in ORDER], fontsize=5.5)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Available reduction captured (%)")
        ax.set_title("Choosing among a day's (or week's) itineraries")
        ax.text(0.98, 0.95, "solid: same day\npale: same week", transform=ax.transAxes,
                ha="right", va="top", fontsize=5.5)
        sjd.letter(ax, "b", "top left")

        # (c) by set size, (d) by haul (same day)
        for ax, dim, title, xlabel, letter in [
                (axes[1, 0], "size", "By number of itineraries that day", "Itineraries in the choice set", "c"),
                (axes[1, 1], "haul", "By route length", "Great-circle distance", "d")]:
            sub = by[(by.key == "day") & (by.dim == dim)]
            levels = list(dict.fromkeys(sub["level"]))
            for r in ORDER:
                s = sub[sub.ranker == r].set_index("level").reindex(levels)
                ax.plot(range(len(levels)), s["reduction_vs_mean_pct"], color=COLORS[r],
                        lw=1.4, marker="o", ms=3, mec="black", mew=0.3, label=NAMES[r])
            ax.set_xticks(range(len(levels)))
            ax.set_xticklabels(levels)
            ax.set_ylim(0, 100)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Available reduction captured (%)")
            ax.set_title(title)
            sjd.letter(ax, letter, "top left")
        for msg in sjd.check(fig):
            print("check:", msg)
        REVISION_PLOTS.mkdir(parents=True, exist_ok=True)
        fig.savefig(REVISION_PLOTS / f"figS6_booking_sets{a.suffix}.png", dpi=250)
        fig.savefig(REVISION_PLOTS / f"figS6_booking_sets{a.suffix}.pdf")
    print("wrote figS6_booking_sets" + a.suffix)


if __name__ == "__main__":
    main()
