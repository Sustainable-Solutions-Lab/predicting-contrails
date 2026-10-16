#!/usr/bin/env python3
"""vPM-activation sensitivity of the pycontrails cross-check and of the
aircraft-type contrast.

Neither our training labels nor the pycontrails cross-check include ice
nucleation on volatile particles (vPM activation), which matters most for
low-soot (lean-burn) engines. This script re-runs CoCiP on the same
synthetic trajectories and met as pycontrails_crossval.py under three
microphysics settings:

  base   defaults (as in the paper's validation)
  vpm    vpm_activation=True
  vpmU   vpm_activation=True + Unterstrasser ice-survival fraction

and two experiments:

  actual  each validation flight with its own aircraft type
          -> does rank agreement with our predictions survive vPM?
  swap    each trajectory flown by a fixed set of aircraft/engine
          combinations (narrowbody set on routes < 4000 km, widebody set
          on routes > 3000 km) -> does the lean-burn advantage survive?

The swap set separates engine from airframe where it matters: the A320neo
is flown with both of its engines (PW1100G, rich-quench-lean; LEAP-1A,
lean-burn) and the 787-9 with both of its (GEnx-1B, lean-burn; Trent 1000).

Extra long-haul trajectories are sampled on the same four days so the
widebody swap has enough routes; met is the cached ERA5 from the
validation run, so nothing is downloaded.

Usage:
  python experiments/validation/vpm_sensitivity.py [--extra-long 15] [--workers 4]
Writes experiments/validation/vpm_results.parquet and prints a summary.
"""
from __future__ import annotations

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PRED = REPO / "experiments/outputs/_2021_predictions.parquet"
SHARDS = Path(
    "/Users/stevedavis/Library/CloudStorage/Dropbox/"
    "Papers/In Review/Contrails/WS Corp contrails (w Silas)/adjustedEFs"
)
CROSSVAL = HERE / "crossval_results.parquet"
OUT = HERE / "vpm_results.parquet"
DATES = ["2021-01-15", "2021-04-15", "2021-07-15", "2021-10-15"]

CONFIGS = {
    "base": {},
    "vpm": {"vpm_activation": True},
    "vpmU": {"vpm_activation": True, "unterstrasser_ice_survival_fraction": True},
}

# label -> (ICAO type for performance, engine UID or None for the default)
NARROW = {
    "A320 (CFM56-5B)": ("A320", None),
    "A320neo (PW1127G)": ("A20N", "01P18PW153"),
    "A320neo (LEAP-1A)": ("A20N", "01P20CM128"),
    "A321 (V2500)": ("A321", None),
    "A321neo (LEAP-1A)": ("A21N", None),
    "737-800 (CFM56-7B)": ("B738", None),
    "737 MAX 8 (LEAP-1B)": ("B38M", None),
}
WIDE = {
    "777-300ER (GE90)": ("B77W", None),
    "787-9 (GEnx-1B)": ("B789", "01P17GE212"),
    "787-9 (Trent 1000)": ("B789", "01P19RR111"),
    "A350-900 (Trent XWB)": ("A359", None),
    "A330-300 (CF6-80E)": ("A333", None),
    "747-400 (CF6-80C2)": ("B744", None),
}


def load_trajectories(extra_long: int, seed: int = 11) -> pd.DataFrame:
    cv = pd.read_parquet(CROSSVAL)[["flight_id", "date"]]
    pred = pd.read_parquet(PRED, columns=[
        "flight_id", "first_waypoint_time", "total_flight_distance_km",
        "pred_log", "contrail_CO2_km"])
    pred["date"] = pred["first_waypoint_time"].dt.strftime("%Y-%m-%d")
    base = pred.merge(cv[["flight_id"]], on="flight_id")
    base["source"] = "validation"
    rng = np.random.default_rng(seed)
    extras = []
    for d in DATES:
        pool = pred[(pred["date"] == d)
                    & (pred["total_flight_distance_km"] > 3000)
                    & ~pred["flight_id"].isin(cv["flight_id"])]
        extras.append(pool.sample(min(extra_long, len(pool)),
                                  random_state=int(rng.integers(1 << 31))))
    ext = pd.concat(extras, ignore_index=True)
    ext["source"] = "extra_long"
    traj = pd.concat([base, ext], ignore_index=True)

    metas = []
    for f in sorted(SHARDS.glob("features_2021*_gdf.pq")):
        if not any(f"_{d[:7].replace('-', '')}" in f.name for d in DATES):
            continue
        m = pd.read_parquet(f, columns=[
            "flight_id", "aircraft_type_icao", "flight_duration_h",
            "OriginLon", "OriginLat", "DestinationLon", "DestinationLat"])
        metas.append(m[m["flight_id"].isin(traj["flight_id"])])
    meta = pd.concat(metas, ignore_index=True).drop_duplicates("flight_id")
    return traj.merge(meta, on="flight_id", how="inner")


def run_date(args) -> pd.DataFrame:
    date, day = args
    import sys
    sys.path.insert(0, str(HERE))
    from pycontrails.models.cocip import Cocip
    from pycontrails.models.humidity_scaling import HistogramMatching
    from pycontrails.models.ps_model import PSFlight

    import pycontrails_crossval as pcv

    d0 = pd.Timestamp(date)
    t0, t1 = d0 - pd.Timedelta("1h"), d0 + pd.Timedelta("42h")
    met = pcv._arco(t0, t1, list(pcv.MET_RENAME), "met",
                    levels=[175, 200, 225, 250, 300, 350])
    rad = pcv._arco(t0, t1, ["top_net_solar_radiation",
                             "top_net_thermal_radiation"], "rad", time_step=1)
    models = {k: Cocip(met=met, rad=rad, aircraft_performance=PSFlight(),
                       humidity_scaling=HistogramMatching(),
                       max_age=np.timedelta64(12, "h"), **kw)
              for k, kw in CONFIGS.items()}
    ps_types = set(PSFlight().aircraft_engine_params)

    jobs = []
    for row in day.itertuples():
        if row.source == "validation" and row.aircraft_type_icao in ps_types:
            jobs.append((row, "actual", row.aircraft_type_icao,
                         row.aircraft_type_icao, None))
        d = row.total_flight_distance_km
        sets = ([NARROW] if d < 4000 else []) + ([WIDE] if d > 3000 else [])
        for s in sets:
            for label, (ac, uid) in s.items():
                jobs.append((row, "swap", label, ac, uid))

    recs = []
    for i, (row, exp, label, ac, uid) in enumerate(jobs):
        r2 = row._replace(aircraft_type_icao=ac)
        for cfg, cocip in models.items():
            try:
                fl = pcv.build_flight(r2)
                if uid:
                    fl.attrs["engine_uid"] = uid
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = cocip.eval(fl)
                ef = float(np.nansum(res["ef"])) if "ef" in res.data else 0.0
                nv = (float(np.nanmean(res["nvpm_ei_n"]))
                      if "nvpm_ei_n" in res.data else np.nan)
                cont = cocip.contrail
                n_ice0 = (float(np.nanmedian(
                    cont.loc[cont["age"] == cont["age"].min(), "n_ice_per_m"]))
                    if cont is not None and len(cont) else np.nan)
                ok = True
            except Exception as e:  # noqa: BLE001 — per-run isolation
                ef, nv, n_ice0, ok = np.nan, np.nan, np.nan, False
                print(f"  {date} {row.flight_id} {label} {cfg}: "
                      f"{type(e).__name__}: {str(e)[:80]}", flush=True)
            recs.append({
                "flight_id": row.flight_id, "date": date, "source": row.source,
                "experiment": exp, "label": label, "aircraft": ac,
                "engine_uid": uid, "config": cfg,
                "dist_km": row.total_flight_distance_km,
                "pred_log": row.pred_log, "label_kg_km": row.contrail_CO2_km,
                "ef_J": ef, "nvpm_ei_n": nv, "n_ice_per_m_init": n_ice0,
                "ok": ok})
        if (i + 1) % 25 == 0:
            print(f"  {date}: {i + 1}/{len(jobs)} runs", flush=True)
    return pd.DataFrame(recs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-long", type=int, default=15)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    traj = load_trajectories(a.extra_long)
    print(f"{len(traj)} trajectories "
          f"({(traj.source == 'extra_long').sum()} extra long-haul)", flush=True)
    work = [(d, traj[traj["date"] == d]) for d in DATES]
    with ProcessPoolExecutor(a.workers) as ex:
        frames = list(ex.map(run_date, work))
    res = pd.concat(frames, ignore_index=True)
    res.to_parquet(OUT)
    print(f"Saved {len(res):,} rows -> {OUT.name}")


if __name__ == "__main__":
    main()
