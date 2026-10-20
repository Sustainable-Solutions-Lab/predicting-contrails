#!/usr/bin/env python3
"""Build the model-ready GAIA 2019–25 corpus from the slim day files
written by gaia2025_ingest.py.

Each day: merge the files for that date (a few year folders also hold a
small file for the previous 31 December), drop duplicate flight_ids,
apply the original corpus filter (FILTER below), compute the schedule
features (schedule_features.py) and the model encodings and target
(common.add_features), and write `GAIA2019-25/corpus/days/{yyyymmdd}.parquet`.
`--concat` then writes one file per year to `corpus/{year}.parquet`.

The target uses `ef_sum` (NULL -> 0), not `ef_overlap_sum` (Roger Teoh,
2026-10-19).

Usage:
  python experiments/gaia2025_corpus.py [--years ...] [--workers 8] [--concat]
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DROPBOX_PARQUETS, MODEL_FEATS, TARGET, add_features,
)
from schedule_features import add_schedule_features, airport_table  # noqa: E402

G = DROPBOX_PARQUETS.parent / "GAIA2019-25"
DAYS = G / "corpus" / "days"
YEARS = [str(y) for y in range(2019, 2026)]

# Kept beside the model columns, for label comparisons and engine analyses
EXTRA = ["flight_id", "operator_icao", "origin_airport", "destination_airport",
         "aircraft_type", "engine_name", "engine_uid", "n_seats",
         "scheduled_departure_time_utc", "first_waypoint_time",
         "total_fuel_burn", "ef_sum", "total_persistent_contrail_length_km"]


def airports() -> pd.DataFrame:
    """Coordinates from the old shards, plus airportsdata for codes the
    2019/2021 corpus never saw (new airports, mostly in China and India)."""
    import airportsdata
    ap = airport_table()
    extra = pd.DataFrame([
        {"icao": k, "lon": v["lon"], "lat": v["lat"]}
        for k, v in airportsdata.load("ICAO").items() if k not in set(ap.icao)])
    return pd.concat([ap, extra], ignore_index=True)


def corpus_filter(df: pd.DataFrame) -> pd.DataFrame:
    """FILTER: the old corpus kept every jet flight with both airports
    known (on 2019-08-01 this rule reproduces 98.7% of its keep/drop
    decisions); same-airport flights then fall to add_features' distance
    >= 0.1 km. Incomplete trajectories were kept."""
    def known(s):
        return s.notna() & ~s.isin(["None", ""])
    return df[known(df["origin_airport_icao"])
              & known(df["destination_airport_icao"])]


def build_day(stem: str, paths: list[Path], ap: pd.DataFrame) -> tuple[str, int, int]:
    df = (pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
          .drop_duplicates("flight_id"))
    n_raw = len(df)
    df = corpus_filter(df).rename(columns={
        "origin_airport_icao": "origin_airport",
        "destination_airport_icao": "destination_airport"})
    df = add_schedule_features(df, ap)
    df["total_contrail_energy_forcing"] = df["ef_sum"].fillna(0.0)
    df["aircraft_type_icao"] = df["aircraft_type"]
    df = add_features(df)
    df = df.dropna(subset=[c for c in MODEL_FEATS if c != "aircraft_type_icao"]
                   + [TARGET])
    df["year"] = int(stem[:4])
    out = df[EXTRA + [c for c in MODEL_FEATS if c not in EXTRA]
             + [TARGET, "contrail_CO2_km", "year"]].copy()
    for c in out.select_dtypes("float64").columns:
        if c not in ("ef_sum", "contrail_CO2_km"):
            out[c] = out[c].astype("float32")
    tmp = DAYS / f"{stem}.tmp"
    out.to_parquet(tmp, compression="zstd", index=False)
    tmp.rename(DAYS / f"{stem}.parquet")
    return stem, n_raw, len(out)


_AP = None


def _init():
    global _AP
    _AP = airports()


def _work(stem, paths):
    return build_day(stem, paths, _AP)


def concat(years: list[str]) -> None:
    for y in years:
        files = sorted(DAYS.glob(f"{y}*.parquet"))
        d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        d = d.drop_duplicates("flight_id")
        d.to_parquet(G / "corpus" / f"{y}.parquet", compression="zstd",
                     index=False)
        print(f"{y}: {len(files)} days, {len(d):,} flights", flush=True)


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--years", nargs="+", default=YEARS)
    ap_.add_argument("--workers", type=int, default=8)
    ap_.add_argument("--concat", action="store_true")
    a = ap_.parse_args()
    if a.concat:
        concat(a.years)
        return
    DAYS.mkdir(parents=True, exist_ok=True)
    stems: dict[str, list[Path]] = defaultdict(list)
    for folder in G.glob("20??"):
        for p in folder.glob("*.parquet"):
            stems[p.stem].append(p)
    jobs = {s: p for s, p in stems.items()
            if s[:4] in a.years and not (DAYS / f"{s}.parquet").exists()}
    print(f"{len(jobs)} days to build", flush=True)
    t0, done, n_in, n_out = time.time(), 0, 0, 0
    with ProcessPoolExecutor(a.workers, initializer=_init) as ex:
        futs = [ex.submit(_work, s, p) for s, p in sorted(jobs.items())]
        for f in as_completed(futs):
            try:
                _, i, o = f.result()
            except Exception as e:  # noqa: BLE001
                print(f"FAILED {e!r}", flush=True)
                continue
            done, n_in, n_out = done + 1, n_in + i, n_out + o
            if done % 50 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"{done}/{len(jobs)} days, {n_out:,} of {n_in:,} flights "
                      f"kept ({el / 60:.0f} min, "
                      f"~{el / done * (len(jobs) - done) / 60:.0f} min left)",
                      flush=True)


if __name__ == "__main__":
    main()
