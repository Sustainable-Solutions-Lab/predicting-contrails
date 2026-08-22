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
WAYPOINT_KM = 25.0   # below Cocip's 40 km max segment length


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
    n = max(8, int(round(row.total_flight_distance_km / WAYPOINT_KM)))
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

# ARCO variable name -> pycontrails standard name
MET_RENAME = {
    "temperature": "air_temperature",
    "specific_humidity": "specific_humidity",
    "u_component_of_wind": "eastward_wind",
    "v_component_of_wind": "northward_wind",
    "vertical_velocity": "lagrangian_tendency_of_air_pressure",
    "specific_cloud_ice_water_content": "mass_fraction_of_cloud_ice_in_air",
}


MET_CACHE = Path(__file__).resolve().parent / "met_cache"
MET_CACHE.mkdir(exist_ok=True)


def _arco(t0, t1, variables, tag, levels=None, stride=2, time_step=3):
    """ARCO ERA5, fetched the boring reliable way: sequential per-timestep
    loads with retries (parallel dask+gcsfs wedges on anonymous access),
    3-hourly (CoCiP interpolates in time), cached to disk per window so
    reruns never touch the network. Wrapped as a pycontrails MetDataset."""
    import time as _time

    import xarray as xr
    from pycontrails import MetDataset

    cache = MET_CACHE / f"{tag}{time_step}h_{pd.Timestamp(t0):%Y%m%dT%H}.nc"
    if cache.exists():
        print(f"    {tag}: cached ({cache.name})", flush=True)
        ds = xr.open_dataset(cache).load()
    else:
        ds = xr.open_zarr(ARCO_AR, chunks=None,
                          storage_options={"token": "anon"})
        ds = ds[variables].sel(time=slice(t0, t1))
        ds = ds.isel(time=slice(None, None, time_step))
        if levels is not None:
            ds = ds.sel(level=levels)
        ds = ds.isel(latitude=slice(None, None, stride),
                     longitude=slice(None, None, stride))
        nt = ds.sizes["time"]
        print(f"    {tag}: fetching {nt} steps "
              f"({sum(v.nbytes for v in ds.data_vars.values()) / 1e9:.2f} GB) ...",
              flush=True)
        pieces = []
        for i in range(nt):
            for attempt in range(4):
                try:
                    pieces.append(ds.isel(time=slice(i, i + 1)).load())
                    break
                except Exception as e:  # noqa: BLE001 — network retry
                    print(f"      step {i} attempt {attempt + 1} failed: "
                          f"{type(e).__name__}; retrying", flush=True)
                    _time.sleep(5 * (attempt + 1))
            else:
                raise RuntimeError(f"step {i} failed after retries")
            if (i + 1) % 5 == 0 or i == nt - 1:
                print(f"      {i + 1}/{nt}", flush=True)
        ds = xr.concat(pieces, dim="time")
        ds.to_netcdf(cache)
        print(f"    {tag}: cached -> {cache.name}", flush=True)
    ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))
    ds = ds.sortby(["longitude", "latitude"])
    if "level" not in ds.dims:
        ds = ds.expand_dims(level=[-1.0])
    ds = ds.rename({k: v for k, v in MET_RENAME.items() if k in ds.data_vars})
    for v in ds.data_vars:
        if str(v).startswith("top_net"):
            ds[v].attrs.setdefault("units", "J m**-2")
    # Cocip keys accumulated-radiation handling off these provenance attrs
    ds.attrs.update(provider="ECMWF", dataset="ERA5", product="reanalysis")
    return MetDataset(ds)


def run_date(meta: pd.DataFrame, date: str) -> pd.DataFrame:
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
    levels = [175, 200, 225, 250, 300, 350]
    print("  loading ERA5 window from ARCO (0.5 deg, 6 levels, 3-hourly) ...", flush=True)
    met = _arco(t0, t1, list(MET_RENAME), "met", levels=levels)
    print("  loading radiation ...", flush=True)
    rad = _arco(t0, t1, ["top_net_solar_radiation", "top_net_thermal_radiation"],
                "rad", time_step=1)
    print("  met ready", flush=True)

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
