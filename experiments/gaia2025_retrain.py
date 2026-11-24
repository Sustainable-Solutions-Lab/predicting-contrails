#!/usr/bin/env python3
"""Retrain the schedule-only model on GAIA 2019–25 (vPM-enabled labels).

Same 14 features and tuned hyperparameters as the published model
(final_model.PARAMS); corpus from gaia2025_corpus.py. Runs:

  oldmodel  the published model (trained on the old 2019+2021 labels)
            scored against the new labels, every year
  pooled    2:1 random split of all years, stratified by year; the
            canonical revised model (saved to outputs/gaia2025_pooled.joblib)
  ood       train 2019–23, test 2024 and 2025 (out of sample in time);
            saved to outputs/gaia2025_ood_2019-23.joblib
  loyo      leave one year out: train on the other six, test the seventh;
            the leave-2021-out model is saved (gaia2025_loyo_2021.joblib)
            for the company analysis of Fig. 4, whose flights are from 2021
  week      flights assigned to train or test by ISO week of the year
            (one third of each year's weeks are test weeks), so that a
            flight's same-day, same-route neighbours are never in the
            other set; "week_buf" additionally drops the two days on
            either side of every test week from the training set
            (a referee's check against memorising high-forcing days)

Captures follow the paper's net-forcing convention (cross_year.evaluate).
`--max-train N` subsamples each training set to N flights (loyo default
40M) to bound memory and run time.

Usage:
  python experiments/gaia2025_retrain.py oldmodel pooled ood loyo week
Writes experiments/outputs/gaia2025_metrics.csv (one row per run x test set)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DROPBOX_PARQUETS, MODEL_FEATS, OUT, RANDOM_SEED, TARGET  # noqa: E402
from cross_year import evaluate  # noqa: E402
from final_model import PARAMS  # noqa: E402

CORPUS = DROPBOX_PARQUETS.parent / "GAIA2019-25/corpus"
YEARS = list(range(2019, 2026))
COLS = list(dict.fromkeys(MODEL_FEATS + [TARGET, "contrail_CO2_km", "year",
                                        "first_waypoint_time"]))
BUFFER_DAYS = 2            # week_buf: days dropped on each side of a test week
SAVE_LOYO = {2021}
METRICS = OUT / "gaia2025_metrics.csv"
CAT = "aircraft_type_icao"


def categories() -> list[str]:
    """One category list for every year, so codes agree across splits."""
    s = set()
    for y in YEARS:
        s |= set(pd.read_parquet(CORPUS / f"{y}.parquet", columns=[CAT])[CAT]
                 .dropna().unique())
    return sorted(s)


def load(year: int, cats: list[str]) -> pd.DataFrame:
    d = pd.read_parquet(CORPUS / f"{year}.parquet", columns=COLS)
    d[CAT] = pd.Categorical(d[CAT], categories=cats)
    d["year"] = d["year"].astype("int16")
    return d


def split(d: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """2:1 random split (stratified by year when applied year by year)."""
    is_test = np.random.default_rng(seed).random(len(d)) < 1 / 3
    return d[~is_test], d[is_test]


def week_split(d: pd.DataFrame, seed: int, buffer_days: int = 0
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blocked split: one third of the ISO weeks of the year, drawn at
    random, are test weeks. With buffer_days, training also excludes
    the days within that many days of a test week (they go to neither
    set), so no training flight is within `buffer_days` of a test flight."""
    day = d["first_waypoint_time"].dt.normalize()
    week = d["first_waypoint_time"].dt.isocalendar().week.to_numpy(np.int64)
    rng = np.random.default_rng(seed)
    test_week = rng.random(54) < 1 / 3          # index = ISO week number
    is_test = test_week[week]
    if buffer_days == 0:
        return d[~is_test], d[is_test]
    # day index within the loaded data; mark days adjacent to test days
    dnum = ((day - day.min()) / pd.Timedelta("1D")).to_numpy(np.int64)
    n_days = dnum.max() + 1
    test_day = np.zeros(n_days, bool)
    test_day[dnum[is_test]] = True
    near = test_day.copy()
    for k in range(1, buffer_days + 1):
        near[k:] |= test_day[:-k]
        near[:-k] |= test_day[k:]
    buffer = near & ~test_day
    is_buffer = buffer[dnum]
    return d[~is_test & ~is_buffer], d[is_test]


def cap(d: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or len(d) <= n:
        return d
    return d.sample(n, random_state=seed)


def fit(train: pd.DataFrame, label: str) -> XGBRegressor:
    t = time.time()
    m = XGBRegressor(**PARAMS)
    m.fit(train[MODEL_FEATS], train[TARGET])
    print(f"[{label}] fit on {len(train):,} flights: {time.time() - t:.0f}s",
          flush=True)
    return m


def record(rows: list, run: str, model, tests: dict[str, pd.DataFrame],
           n_train: int) -> None:
    for name, te in tests.items():
        r = evaluate(f"{run}:{name}", model, te)
        rows.append(dict(r, run=run, test=name, n_train=n_train))
    pd.DataFrame(rows).to_csv(METRICS, index=False)


def run_oldmodel(rows, cats):
    """The old model's categorical codes are the sorted type list of its
    training cache; re-code the new data to that list."""
    b = joblib.load(OUT / "final_model.joblib")
    old_cats = sorted(pd.read_parquet(OUT / "_pool_cache_all.parquet",
                                      columns=[CAT])[CAT].dropna().unique())
    tests = {}
    for y in YEARS:
        d = load(y, cats)
        d[CAT] = pd.Categorical(d[CAT].astype(object), categories=old_cats)
        tests[str(y)] = d
    record(rows, "oldmodel", b["model"], tests, n_train=0)


def run_pooled(rows, cats, max_train):
    trs, tes = [], {}
    for y in YEARS:
        tr, te = split(load(y, cats), RANDOM_SEED + y)
        trs.append(tr)
        tes[str(y)] = te
    train = cap(pd.concat(trs, ignore_index=True), max_train, RANDOM_SEED)
    del trs
    m = fit(train, "pooled")
    joblib.dump({"model": m, "features": MODEL_FEATS, "params": PARAMS,
                 "categories": cats, "n_train": len(train)},
                OUT / "gaia2025_pooled.joblib")
    n = len(train)
    del train
    tes["pooled"] = pd.concat(tes.values(), ignore_index=True)
    record(rows, "pooled", m, tes, n)


def run_ood(rows, cats, max_train):
    train = cap(pd.concat([load(y, cats) for y in YEARS if y <= 2023],
                          ignore_index=True), max_train, RANDOM_SEED)
    m = fit(train, "ood")
    n = len(train)
    del train
    joblib.dump({"model": m, "features": MODEL_FEATS, "params": PARAMS,
                 "categories": cats, "n_train": n, "train_years": [2019, 2023]},
                OUT / "gaia2025_ood_2019-23.joblib")
    record(rows, "ood_2019-23", m, {str(y): load(y, cats) for y in (2024, 2025)}, n)


def run_loyo(rows, cats, max_train, years=YEARS):
    for hold in years:
        per = max_train // (len(YEARS) - 1)
        train = pd.concat([cap(load(y, cats), per, RANDOM_SEED + y)
                           for y in YEARS if y != hold], ignore_index=True)
        m = fit(train, f"loyo-{hold}")
        n = len(train)
        del train
        if hold in SAVE_LOYO:
            joblib.dump({"model": m, "features": MODEL_FEATS, "params": PARAMS,
                         "categories": cats, "n_train": n, "held_out": hold},
                        OUT / f"gaia2025_loyo_{hold}.joblib")
        record(rows, f"loyo_{hold}", m, {str(hold): load(hold, cats)}, n)


def run_week(rows, cats, max_train, buffer_days):
    """Week-blocked split of every year, same training size as `pooled`
    (all non-test flights) unless capped."""
    label = "week_buf" if buffer_days else "week"
    trs, tes = [], {}
    for y in YEARS:
        tr, te = week_split(load(y, cats), RANDOM_SEED + y, buffer_days)
        print(f"  {label} {y}: train {len(tr):,}  test {len(te):,}", flush=True)
        trs.append(tr)
        tes[str(y)] = te
    train = cap(pd.concat(trs, ignore_index=True), max_train, RANDOM_SEED)
    del trs
    m = fit(train, label)
    n = len(train)
    del train
    tes["pooled"] = pd.concat(tes.values(), ignore_index=True)
    record(rows, label, m, tes, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+",
                    choices=["oldmodel", "pooled", "ood", "loyo", "week",
                             "week_buf"])
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--loyo-years", type=int, nargs="+", default=YEARS,
                    help="hold-out years to (re)run for loyo")
    a = ap.parse_args()
    rows = pd.read_csv(METRICS).to_dict("records") if METRICS.exists() else []
    def family(run: str) -> str:
        return run if run in ("week", "week_buf") else run.split("_")[0]
    rows = [r for r in rows if family(r["run"]) not in a.runs
            or (r["run"].startswith("loyo_")
                and int(r["run"][5:]) not in a.loyo_years)]
    cats = categories()
    for r in a.runs:
        t = time.time()
        if r == "oldmodel":
            run_oldmodel(rows, cats)
        elif r == "pooled":
            run_pooled(rows, cats, a.max_train)
        elif r == "ood":
            run_ood(rows, cats, a.max_train)
        elif r == "loyo":
            run_loyo(rows, cats, a.max_train or 40_000_000, a.loyo_years)
        elif r == "week":
            run_week(rows, cats, a.max_train, 0)
        else:
            run_week(rows, cats, a.max_train, BUFFER_DAYS)
        print(f"== {r} done in {(time.time() - t) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
