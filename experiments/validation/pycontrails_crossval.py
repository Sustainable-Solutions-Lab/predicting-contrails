#!/usr/bin/env python3
"""Independent cross-model validation of the schedule-only contrail model.

For a stratified sample of 2021 flights (spanning our predicted-forcing
deciles), synthesize schedule-only cruise trajectories and run them
through pycontrails' CoCiP implementation with ERA5 reanalysis from
Google's public ARCO archive — an independent contrail model, met
source, and aircraft-performance model (Poll–Schumann) from the
pipeline that produced our training labels. Report rank correlations:

  spearman(our schedule-only prediction, pycontrails EF)
  spearman(our training label,           pycontrails EF)   [context]
  top-decile capture of pycontrails-ranked EF by our flags

Usage:
  python experiments/validation/pycontrails_crossval.py \
      --dates 2021-01-15 2021-04-15 2021-07-15 2021-10-15 \
      --per-decile 8

Each date costs one ERA5 met window (~36 h global, 8 pressure levels);
flights on that date share it. Results append to
experiments/validation/crossval_results.parquet.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "experiments/outputs/_2021_predictions.parquet"
SHARDS = Path(
    "/Users/stevedavis/Library/CloudStorage/Dropbox/"
    "Papers/Active Prep/Contrails/WS Corp contrails (w Silas)/adjustedEFs"
)
OUT = Path(__file__).resolve().parent / "crossval_results.parquet"

CRUISE_ALT_M_SHORT = 10050.0   # <1500 km
CRUISE_ALT_M_LONG = 11280.0
WAYPOINT_MIN = 4.0


def sample_flights(dates: list[str], per_decile: int, seed: int = 7) -> pd.DataFrame:
    print("Loading predictions cache ...")
    db = pd.read_parquet(CACHE)
    edges = np.quantile(db["pred_log"], np.linspace(0, 1, 11))
    rng = np.random.default_rng(seed)
    rows = []
    for date in dates:
        d0 = pd.Timestamp(date)
        day = db[(db["first_waypoint_time"] >= d0)
                 & (db["first_waypoint_time"] < d0 + pd.Timedelta("1D"))].copy()
        day["decile"] = np.clip(
            np.searchsorted(edges[1:-1], day["pred_log"]), 0, 9)
        for dec, g in day.groupby("decile"):
            take = g.sample(min(per_decile, len(g)), random_state=rng.integers(1 << 31))
            rows.append(take)
        print(f"  {date}: {len(day):,} flights on day")
    samp = pd.concat(rows, ignore_index=True)

    print("Joining aircraft type + duration from shards ...")
    months = sorted({d[:7].replace('-', '') for d in dates})
    metas = []
    for mo in months:
        for f in sorted(SHARDS.glob(f"features_{mo}*_gdf.pq")):
            m = pd.read_parquet(f, columns=[
                "flight_id", "aircraft_type_icao", "flight_duration_h",
                "OriginLon", "OriginLat", "DestinationLon", "DestinationLat",
            ])
            metas.append(m[m["flight_id"].isin(samp["flight_id"])])
    meta = pd.concat(metas, ignore_index=True).drop_duplicates("flight_id")
    out = samp.merge(meta, on="flight_id", how="inner")
    print(f"  sampled {len(samp):,} -> matched {len(out):,}")
    return out


def build_flight(row) -> "Flight":
    from pycontrails import Flight
    from pyproj import Geod
    geod = Geod(ellps="WGS84")
    dur_h = float(row.flight_duration_h)
    n = max(8, int(round(dur_h * 60.0 / WAYPOINT_MIN)))
    lons, lats = zip(*geod.npts(row.OriginLon, row.OriginLat,
                                row.DestinationLon, row.DestinationLat, n))
    t0 = pd.Timestamp(row.first_waypoint_time)
    times = pd.date_range(t0, t0 + pd.Timedelta(hours=dur_h), periods=n)
    alt = CRUISE_ALT_M_SHORT if row.total_flight_distance_km < 1500 else CRUISE_ALT_M_LONG
    df = pd.DataFrame({
        "longitude": lons, "latitude": lats,
        "altitude": np.full(n, alt), "time": times,
    })
    return Flight(df, aircraft_type=row.aircraft_type_icao,
                  flight_id=row.flight_id)


ARCO_AR = ("gs://gcp-public-data-arco-era5/ar/"
           "full_37-1h-0p25deg-chunk-1.zarr-v3")


def open_arco_rad(t0, t1):
    """Cocip's TOA radiation fields, direct from the ARCO ERA5 zarr —
    pycontrails' ERA5ARCO wrapper only exposes pressure-level variables,
    but the store itself carries the accumulated tsr/ttr fields."""
    import xarray as xr
    from pycontrails import MetDataset
    ds = xr.open_zarr(ARCO_AR, chunks={}, storage_options={"token": "anon"})
    ds = ds[["top_net_solar_radiation", "top_net_thermal_radiation"]]
    ds = ds.sel(time=slice(t0, t1))
    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    ds = ds.sortby(["longitude", "latitude"])
    ds = ds.expand_dims(level=[-1.0])
    for v in ds.data_vars:
        ds[v].attrs.setdefault("units", "J m**-2")
    return MetDataset(ds)


def run_date(meta: pd.DataFrame, date: str) -> pd.DataFrame:
    from pycontrails.datalib.ecmwf import ERA5ARCO
    from pycontrails.models.cocip import Cocip
    from pycontrails.models.humidity_scaling import HistogramMatching
    from pycontrails.models.ps_model import PSFlight

    d0 = pd.Timestamp(date)
    day = meta[(meta["first_waypoint_time"] >= d0)
               & (meta["first_waypoint_time"] < d0 + pd.Timedelta("1D"))]
    ps = PSFlight()
    day = day[day["aircraft_type_icao"].isin(ps.aircraft_engine_params)]
    print(f"\n=== {date}: {len(day)} flights (PS-supported) ===")
    if not len(day):
        return pd.DataFrame()

    t0 = d0 - pd.Timedelta("1h")
    t1 = d0 + pd.Timedelta("42h")
    levels = [150, 175, 200, 225, 250, 300, 350, 400]
    print("  opening ERA5 (ARCO) ...")
    era5pl = ERA5ARCO(time=(t0, t1), variables=Cocip.met_variables,
                      pressure_levels=levels)
    met = era5pl.open_metdataset()
    rad = open_arco_rad(t0, t1)

    cocip = Cocip(met=met, rad=rad,
                  aircraft_performance=PSFlight(),
                  humidity_scaling=HistogramMatching(),
                  max_age=np.timedelta64(12, "h"))

    recs = []
    for i, row in enumerate(day.itertuples()):
        try:
            fl = build_flight(row)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = cocip.eval(fl)
            ef = float(np.nansum(res["ef"])) if "ef" in res.data else 0.0
        except Exception as e:  # noqa: BLE001 — per-flight isolation
            print(f"    [{i}] {row.flight_id}: FAILED {type(e).__name__}: {e}")
            continue
        recs.append({
            "flight_id": row.flight_id, "date": date,
            "aircraft": row.aircraft_type_icao,
            "dist_km": row.total_flight_distance_km,
            "pred_log": row.pred_log,
            "label_kg_km": row.contrail_CO2_km,
            "py_ef_J": ef,
        })
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(day)} done")
    return pd.DataFrame(recs)


def report(df: pd.DataFrame) -> None:
    from scipy.stats import spearmanr
    df = df.copy()
    df["py_ef_per_km"] = df["py_ef_J"] / df["dist_km"]
    r1 = spearmanr(df["pred_log"], df["py_ef_per_km"])
    r2 = spearmanr(df["label_kg_km"], df["py_ef_per_km"])
    n = len(df)
    k = max(1, n // 10)
    top_py = set(df.nlargest(k, "py_ef_per_km")["flight_id"])
    top_us = set(df.nlargest(k, "pred_log")["flight_id"])
    print(f"\n──── cross-validation vs pycontrails+ERA5 (n={n}) ────")
    print(f"  spearman(our schedule-only pred, pycontrails EF/km): "
          f"{r1.statistic:+.3f}  (p={r1.pvalue:.1e})")
    print(f"  spearman(training label,        pycontrails EF/km): "
          f"{r2.statistic:+.3f}  (p={r2.pvalue:.1e})")
    print(f"  top-decile overlap (ours vs pycontrails): "
          f"{len(top_py & top_us)}/{k}")
    print(f"  share with any persistent contrail (pycontrails): "
          f"{(df['py_ef_J'] != 0).mean():.0%}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--per-decile", type=int, default=8)
    args = ap.parse_args()

    meta = sample_flights(args.dates, args.per_decile)
    frames = [run_date(meta, d) for d in args.dates]
    res = pd.concat([f for f in frames if len(f)], ignore_index=True)
    if OUT.exists():
        prev = pd.read_parquet(OUT)
        res = pd.concat([prev[~prev["flight_id"].isin(res["flight_id"])], res],
                        ignore_index=True)
    res.to_parquet(OUT)
    print(f"\nSaved {len(res):,} rows -> {OUT.name}")
    report(res)


if __name__ == "__main__":
    main()
