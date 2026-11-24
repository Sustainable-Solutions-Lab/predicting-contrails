#!/usr/bin/env python3
"""How much do operated departure times differ from scheduled ones, and
does it matter to the model? (A referee asked how we know the
difference is small.)

The corpus records the first ADS-B waypoint time (what the model is
trained on) and, for most flights since 2024, the scheduled departure
time (what a booking shows; populated for 2019-22, empty in the 2023-25
files). For a sample of 2022 flights this script
reports the distribution of the offset and re-scores the flights with
the departure hour taken from the schedule instead, comparing rankings.

Usage: python experiments/validation/schedule_vs_actual.py [--n 2000000] [--year 2022]
Writes experiments/outputs/schedule_vs_actual.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import MODEL_FEATS, OUT, RANDOM_SEED, REVISION_PLOTS, load_corpus  # noqa: E402

CAT = "aircraft_type_icao"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--year", type=int, default=2022)
    a = ap.parse_args()
    b = joblib.load(OUT / "gaia2025_pooled.joblib")
    cols = list(dict.fromkeys(MODEL_FEATS + ["first_waypoint_time",
                                            "scheduled_departure_time_utc",
                                            "contrail_CO2_km"]))
    d = load_corpus([a.year], cols)
    n_all = len(d)
    sched = pd.to_datetime(d["scheduled_departure_time_utc"], errors="coerce", utc=True)
    d["scheduled_departure_time_utc"] = sched.dt.tz_localize(None)
    d = d.dropna(subset=["scheduled_departure_time_utc"])
    share_sched = len(d) / n_all
    d = d.sample(min(a.n, len(d)), random_state=RANDOM_SEED).reset_index(drop=True)
    d[CAT] = pd.Categorical(d[CAT], categories=b["categories"])
    t_act = pd.to_datetime(d["first_waypoint_time"]).dt.tz_localize(None)
    t_sch = d["scheduled_departure_time_utc"]
    off_min = (t_act - t_sch).dt.total_seconds().to_numpy() / 60
    same_hour = (t_act.dt.hour == t_sch.dt.hour).to_numpy()

    yp_act = b["model"].predict(d[MODEL_FEATS])
    d2 = d.copy()
    h = t_sch.dt.hour.to_numpy(float)
    d2["start_hour_sin"] = np.sin(np.pi * h * 2 / 24).astype(np.float32)
    d2["start_hour_cos"] = np.cos(np.pi * h * 2 / 24).astype(np.float32)
    yp_sch = b["model"].predict(d2[MODEL_FEATS])
    k = int(round(0.1 * len(d)))
    top_act = set(np.argsort(yp_act)[::-1][:k])
    top_sch = set(np.argsort(yp_sch)[::-1][:k])
    co2 = d["contrail_CO2_km"].to_numpy(float)
    cap = lambda yp: 100 * co2[np.argsort(yp)[::-1][:k]].sum() / co2.sum()
    row = dict(
        year=a.year, n=len(d), share_flights_with_schedule=share_sched,
        median_offset_min=float(np.median(off_min)),
        p10_offset_min=float(np.percentile(off_min, 10)),
        p90_offset_min=float(np.percentile(off_min, 90)),
        share_within_30min=float(np.mean(np.abs(off_min) <= 30)),
        share_within_60min=float(np.mean(np.abs(off_min) <= 60)),
        share_same_utc_hour=float(same_hour.mean()),
        spearman_pred_actual_vs_scheduled_hour=float(spearmanr(yp_act, yp_sch).correlation),
        top10_overlap=len(top_act & top_sch) / k,
        top10_capture_actual_hour=cap(yp_act),
        top10_capture_scheduled_hour=cap(yp_sch),
    )
    tab = pd.DataFrame([row])
    for p in (OUT, REVISION_PLOTS):
        tab.to_csv(p / "schedule_vs_actual.csv", index=False)
    print(tab.T.to_string())


if __name__ == "__main__":
    main()
