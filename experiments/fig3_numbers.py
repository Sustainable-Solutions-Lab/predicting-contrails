#!/usr/bin/env python3
"""Every number quoted from Fig. 3, from the saved model and corpus.

Flights are ranked by the model's predicted contrail forcing per km, the
comparison a traveler makes among flights to the same destination. The
paper sums the forcing per km of the flights in each ranked share and
divides by the net total, so every flight counts equally regardless of
its length ("per-km" below; Fig. 3). For reference the script also sums
each flight's total forcing ("total" below), which weights long flights
by their length.

Rows: the model's predicted worst / best k% of flights, perfect foresight
(flights ranked by their simulated forcing), and the single-feature
insolation baseline (flights ranked by night_score_full_0), for the
pooled test set and each year. Also R2 in signed-log space.

Usage:
  python experiments/fig3_numbers.py [--sample 5000000]
Writes experiments/outputs/fig3_numbers.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODEL_FEATS, OUT, RANDOM_SEED, TARGET  # noqa: E402
from gaia2025_retrain import YEARS, load, split  # noqa: E402

KS = (0.05, 0.10)


def shares(score: np.ndarray, w: np.ndarray) -> dict:
    """Share of the net total of w in the top and bottom k% by score.
    Stable sort, so tied scores (common for insolation) keep data order,
    exactly as final_model.build_fig3 draws them."""
    order = np.argsort(score, kind="stable")
    net = w.sum()
    out = {}
    for k in KS:
        n = int(round(k * len(w)))
        out[f"worst{int(k * 100)}"] = 100 * w[order[::-1][:n]].sum() / net
        out[f"best{int(k * 100)}"] = 100 * w[order[:n]].sum() / net
    return out


def numbers(te: pd.DataFrame, yp: np.ndarray, label: str) -> list[dict]:
    per_km = te["contrail_CO2_km"].to_numpy(float)
    weights = {"per-km": per_km,
               "total": per_km * te["total_flight_distance_km"].to_numpy(float)}
    rankings = {"model": yp,
                "perfect foresight": per_km,
                "insolation only": te["night_score_full_0"].to_numpy(float)}
    r2 = r2_score(te[TARGET], yp)
    return [dict(test=label, n=len(te), weighting=wn, ranking=rn, r2=r2,
                 **shares(score, w))
            for wn, w in weights.items() for rn, score in rankings.items()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None,
                    help="score a random subset of the test set (faster)")
    a = ap.parse_args()
    b = joblib.load(OUT / "gaia2025_pooled.joblib")
    parts = [split(load(y, b["categories"]), RANDOM_SEED + y)[1] for y in YEARS]
    te = pd.concat(parts, ignore_index=True)
    if a.sample:
        te = te.sample(a.sample, random_state=RANDOM_SEED)
    yp = b["model"].predict(te[MODEL_FEATS])
    rows = numbers(te, yp, "pooled")
    for y in YEARS:
        m = (te["year"] == y).to_numpy()
        rows += numbers(te[m], yp[m], str(y))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "fig3_numbers.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
