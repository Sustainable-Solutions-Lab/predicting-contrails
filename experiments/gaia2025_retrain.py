#!/usr/bin/env python3
"""Retrain the schedule-only model on GAIA 2019–25 (vPM-enabled labels).

Same 14 features and tuned hyperparameters as the published model
(final_model.PARAMS); corpus from gaia2025_corpus.py. Runs:

  oldmodel  the published model (trained on the old 2019+2021 labels)
            scored against the new labels, every year
  pooled    2:1 random split of all years, stratified by year; the
            canonical revised model (saved to outputs/gaia2025_pooled.joblib)
  ood       train 2019–23, test 2024 and 2025 (out of sample in time)
  loyo      leave one year out: train on the other six, test the seventh

Captures follow the paper's net-forcing convention (cross_year.evaluate).
`--max-train N` subsamples each training set to N flights (loyo default
40M) to bound memory and run time.

Usage:
  python experiments/gaia2025_retrain.py oldmodel pooled ood loyo
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
COLS = list(dict.fromkeys(MODEL_FEATS + [TARGET, "contrail_CO2_km", "year"]))
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
    record(rows, "ood_2019-23", m, {str(y): load(y, cats) for y in (2024, 2025)}, n)


def run_loyo(rows, cats, max_train):
    for hold in YEARS:
        per = max_train // (len(YEARS) - 1)
        train = pd.concat([cap(load(y, cats), per, RANDOM_SEED + y)
                           for y in YEARS if y != hold], ignore_index=True)
        m = fit(train, f"loyo-{hold}")
        n = len(train)
        del train
        record(rows, f"loyo_{hold}", m, {str(hold): load(hold, cats)}, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+",
                    choices=["oldmodel", "pooled", "ood", "loyo"])
    ap.add_argument("--max-train", type=int, default=None)
    a = ap.parse_args()
    rows = pd.read_csv(METRICS).to_dict("records") if METRICS.exists() else []
    rows = [r for r in rows if r["run"].split("_")[0] not in a.runs]
    cats = categories()
    for r in a.runs:
        t = time.time()
        if r == "oldmodel":
            run_oldmodel(rows, cats)
        elif r == "pooled":
            run_pooled(rows, cats, a.max_train)
        elif r == "ood":
            run_ood(rows, cats, a.max_train)
        else:
            run_loyo(rows, cats, a.max_train or 40_000_000)
        print(f"== {r} done in {(time.time() - t) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
