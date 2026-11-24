#!/usr/bin/env python3
"""Sensitivity of the route-mean insolation feature to the number of
interpolated waypoints (a referee asked whether 30 points short-change
long flights).

night_score_full_0 averages max(0, sin(solar altitude)) over 30 points
along the great circle at times interpolated from departure to arrival.
Here the same quantity is recomputed with 30 and 300 points for a sample
of 2025 flights and the differences reported by flight distance.

Usage: python experiments/validation/insolation_waypoints.py [--n 300000]
Writes experiments/outputs/insolation_waypoints.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import BASE_YEAR, OUT, RANDOM_SEED, REVISION_PLOTS, load_corpus  # noqa: E402
from schedule_features import sun_sin_alt, waypoints  # noqa: E402

CRUISE_KMH, TAXI_H = 800.0, 0.4


def score(d: pd.DataFrame, nb: int, chunk: int = 50_000) -> np.ndarray:
    t0 = d["first_waypoint_time"].to_numpy("datetime64[s]").astype(np.int64).astype(float)
    dur = (TAXI_H + d["total_flight_distance_km"].to_numpy(float) / CRUISE_KMH) * 3600
    o = [d[c].to_numpy(float) for c in ("OriginLon", "OriginLat", "DestinationLon", "DestinationLat")]
    frac = np.arange(nb) / (nb - 1)
    out = np.empty(len(d))
    for s in range(0, len(d), chunk):
        sl = slice(s, s + chunk)
        lon, lat = waypoints(*(a[sl] for a in o), nb)
        tt = t0[sl, None] + dur[sl, None] * frac
        out[sl] = 100 * (1 - np.maximum(0, sun_sin_alt(lon, lat, tt)).mean(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300_000)
    a = ap.parse_args()
    cols = ["OriginLon", "OriginLat", "DestinationLon", "DestinationLat",
            "first_waypoint_time", "total_flight_distance_km", "night_score_full_0"]
    d = load_corpus([BASE_YEAR], cols).sample(a.n, random_state=RANDOM_SEED)
    d["first_waypoint_time"] = pd.to_datetime(d["first_waypoint_time"]).dt.tz_localize(None)
    s30, s300 = score(d, 30), score(d, 300)
    diff = s300 - s30
    d["abs_diff"] = np.abs(diff)
    d["haul"] = pd.cut(d["total_flight_distance_km"], [0, 500, 1500, 4000, 8000, np.inf],
                       labels=["<500", "500-1500", "1500-4000", "4000-8000", ">8000"])
    rows = []
    for lab, g in [("all", d)] + list(d.groupby("haul", observed=True)):
        rows.append(dict(distance_km=str(lab), n=len(g),
                         mean_abs_diff=g["abs_diff"].mean(),
                         p95_abs_diff=g["abs_diff"].quantile(0.95),
                         max_abs_diff=g["abs_diff"].max()))
    tab = pd.DataFrame(rows)
    # recompute mean score per group without index gymnastics
    d["s30"] = s30
    tab["mean_score_30pts"] = [d["s30"].mean()] + [g["s30"].mean() for _, g in d.groupby("haul", observed=True)]
    from scipy.stats import spearmanr
    tab["spearman_30_vs_300"] = spearmanr(s30, s300).correlation
    tab["frac_rank_change_gt_1pct"] = (np.abs(pd.Series(s30).rank(pct=True).to_numpy()
                                              - pd.Series(s300).rank(pct=True).to_numpy()) > 0.01).mean()
    for p in (OUT, REVISION_PLOTS):
        tab.to_csv(p / "insolation_waypoints.csv", index=False)
    pd.set_option("display.width", 200)
    print(tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
