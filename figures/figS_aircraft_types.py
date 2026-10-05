#!/usr/bin/env python3
"""Supplementary figure — aircraft-type effects in the scheduling model.

Counterfactual swap: a fixed sample of real flight schedules is
re-predicted as if each were flown by a given aircraft type, holding
route, date and departure time fixed. Widebody types are evaluated on
long-haul schedules (>5000 km), narrowbody types on medium-haul
schedules (500–3000 km). Predictions are converted to expected per-km
energy forcing through the empirical percentile calibration (mean
actual forcing of 2021 flights in each predicted-percentile bin), which
avoids the bias of inverting the signed-log target directly. Points are
means over schedules with 95% bootstrap intervals.

These are the model's learned type-level associations: they combine
engine soot emissions, typical cruise altitude, and the routes each
type tends to fly, not isolated engine effects.
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                            "font.size": 9})

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
from common import DROPBOX_PLOTS, MODEL_FEATS, OUT  # noqa: E402

CALIB = Path("/Users/stevedavis/Library/CloudStorage/Dropbox/Sites/"
             "SustainableSolutions/api/_contrails_assets/calibration.json")
J_PER_KG = 82.5e-15 * 365 * 24 * 3600 * 5.101e14   # label-construction constant
N_SCHED = 100_000
MIN_TRAIN = 50_000   # only types with ample training data

WIDE = {"B788": "787-8", "B789": "787-9", "B78X": "787-10",
        "A359": "A350-900", "A35K": "A350-1000", "A339": "A330-900neo",
        "A332": "A330-200", "A333": "A330-300", "B77W": "777-300ER",
        "B772": "777-200", "B77L": "777-200LR", "B763": "767-300",
        "B744": "747-400", "B748": "747-8", "A388": "A380"}
NARROW = {"A20N": "A320neo", "A21N": "A321neo", "B38M": "737 MAX 8",
          "B39M": "737 MAX 9", "BCS3": "A220-300", "BCS1": "A220-100",
          "A319": "A319", "A320": "A320", "A321": "A321",
          "B737": "737-700", "B738": "737-800", "B739": "737-900",
          "B752": "757-200", "E190": "E190", "E75L": "E175"}


def main():
    model = joblib.load(OUT / "final_model.joblib")["model"]
    calib = json.load(open(CALIB))
    q = np.asarray(calib["pred_log_quantiles"])
    bins = np.asarray(calib["bin_mean_co2e_kg_per_km"])

    df = pd.read_parquet(OUT / "_pool_cache_all.parquet",
                         columns=list(dict.fromkeys(MODEL_FEATS)))
    counts = df["aircraft_type_icao"].value_counts()
    cats = df["aircraft_type_icao"].astype("category").cat.categories
    d = df["total_flight_distance_km"]
    panels = [
        ("Widebody types on long-haul schedules (>5,000 km)", WIDE,
         df[d > 5000].sample(N_SCHED, random_state=0)),
        ("Narrowbody types on medium-haul schedules (500–3,000 km)", NARROW,
         df[(d > 500) & (d <= 3000)].sample(N_SCHED, random_state=0)),
    ]
    del df

    rng = np.random.default_rng(0)
    rows = []
    for title, types, sched in panels:
        for icao, name in types.items():
            if counts.get(icao, 0) < MIN_TRAIN:
                continue
            s = sched.copy()
            s["aircraft_type_icao"] = pd.Categorical([icao] * len(s),
                                                     categories=cats)
            p = model.predict(s[MODEL_FEATS])
            pct = np.interp(p, q, np.linspace(0, 100, len(q)))
            b = np.minimum(pct.astype(int), len(bins) - 1)
            gj = bins[b] * J_PER_KG / 1e9
            boots = [gj[rng.integers(0, len(gj), len(gj))].mean() for _ in range(200)]
            rows.append(dict(panel=title, icao=icao, name=name, mean=gj.mean(),
                             lo=np.percentile(boots, 2.5),
                             hi=np.percentile(boots, 97.5),
                             n_train=int(counts[icao])))
            print(f"{icao:5s} {gj.mean():7.2f} GJ/km", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(DROPBOX_PLOTS / "figS_aircraft_types.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8))
    ink, mark = "#222222", "#2a6f97"
    for ax, (title, _, _) in zip(axes, panels):
        r = res[res.panel == title].sort_values("mean")
        y = np.arange(len(r))
        ax.hlines(y, r["lo"], r["hi"], color=mark, lw=2)
        ax.scatter(r["mean"], y, s=36, color=mark, zorder=3,
                   edgecolor="white", linewidth=1.5)
        ax.set_yticks(y)
        ax.set_yticklabels(r["name"], color=ink)
        ax.set_xlabel("Expected contrail energy forcing (GJ per km)")
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(axis="x", color="#dddddd", lw=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.axvline(0, color="#999999", lw=0.8)
    for ax, letter in zip(axes, "ab"):
        ax.text(-0.02, 1.06, letter, transform=ax.transAxes, fontsize=11,
                weight="bold", ha="right")
    fig.tight_layout()
    for ext in ("png", "pdf", "eps"):
        fig.savefig(DROPBOX_PLOTS / f"figS_aircraft_types.{ext}", dpi=200,
                    bbox_inches="tight")
    print("wrote figS_aircraft_types")


if __name__ == "__main__":
    main()
