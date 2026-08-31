#!/usr/bin/env python3
"""Cross-year generalization stress test for the canonical scheduling model.

Four runs with the canonical features and tuned hyperparameters:
  train 2019 (2/3) -> test 2019 (1/3)   within-year reference
  train 2019 (all) -> test 2021 (all)   cross-year
  train 2021 (2/3) -> test 2021 (1/3)   within-year reference
  train 2021 (all) -> test 2019 (all)   cross-year

The within-year references use a 2:1 split of the single year so the
comparison isolates cross-year transfer rather than train-set size or
pooled-vs-single-year training. Captures are reported in the paper's
net-forcing convention (cumulative actual forcing of the predicted worst
k%, ranked by predicted per-km forcing, as a share of the test set's NET
total) and, for continuity with final_metrics.csv, in the superseded
gross-positive convention.

Writes experiments/outputs/cross_year_metrics.csv.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MODEL_FEATS, OUT, RANDOM_SEED, TARGET  # noqa: E402
from final_model import PARAMS  # noqa: E402

CACHE = OUT / "_pool_cache_all.parquet"


def evaluate(name, model, test):
    yp = model.predict(test[MODEL_FEATS])
    co2 = test["contrail_CO2_km"].to_numpy()
    net = co2.sum()
    pos = co2[co2 > 0].sum()
    order = np.argsort(yp)[::-1]
    cum = np.cumsum(co2[order])
    row = dict(run=name, n_test=len(test),
               r2=r2_score(test[TARGET], yp))
    for k in (0.05, 0.10):
        c = cum[int(round(k * len(test))) - 1]
        row[f"top{int(k*100)}_capture_net"] = 100 * c / net
        row[f"top{int(k*100)}_capture_pos"] = 100 * c / pos
    print(pd.Series(row).to_string(), flush=True)
    return row


def main():
    t0 = time.time()
    cols = list(dict.fromkeys(
        MODEL_FEATS + [TARGET, "contrail_CO2_km", "year"]))
    pool = pd.read_parquet(CACHE, columns=cols)
    pool["aircraft_type_icao"] = pool["aircraft_type_icao"].astype("category")
    years = {y: pool[pool["year"] == y] for y in (2019, 2021)}
    del pool
    print(f"loaded: " + ", ".join(f"{y}: {len(d):,}" for y, d in years.items())
          + f" in {time.time()-t0:.0f}s", flush=True)

    rows = []
    for a, b in ((2019, 2021), (2021, 2019)):
        # within-year reference: 2:1 split of year a
        tr, te = train_test_split(years[a], test_size=1/3,
                                  random_state=RANDOM_SEED)
        t1 = time.time()
        m = XGBRegressor(**PARAMS)
        m.fit(tr[MODEL_FEATS], tr[TARGET])
        print(f"\ntrain {a} (2/3) fit {time.time()-t1:.0f}s", flush=True)
        rows.append(evaluate(f"train{a}_test{a}_holdout", m, te))
        del tr, te, m

        # cross-year: all of year a -> all of year b
        t1 = time.time()
        m = XGBRegressor(**PARAMS)
        m.fit(years[a][MODEL_FEATS], years[a][TARGET])
        print(f"\ntrain {a} (all) fit {time.time()-t1:.0f}s", flush=True)
        rows.append(evaluate(f"train{a}_test{b}", m, years[b]))
        del m

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "cross_year_metrics.csv", index=False)
    print(f"\nwrote cross_year_metrics.csv; total {time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
