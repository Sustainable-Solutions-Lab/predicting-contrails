"""
Rebuild the web-tool serving assets from the GAIA 2019–25 model
(experiments/gaia2025_retrain.py, pooled run) into webtool/assets/.

Same files as export_assets.py, sourced from the revision corpus:

  model_canonical.ubj  the pooled 2019–25 XGBoost model
  aircraft.json        its category list (order fixes the categorical
                       codes) + the 40 most-flown types in 2025
  calibration.json     percentile grid of predicted signed-log forcing
                       over all 2025 flights + per-percentile-bin mean
                       actual CO2e (kg/km and kg/flight)
  routes.json          aircraft mix per airport pair, 2024–25 (>=5 flights)
  airports_search.json autocomplete list of airports seen in 2024–25
  flightnos.json       flight number -> typical route/time/aircraft, 2025

airports.json, land_union.wkb and pax_by_type.json are unchanged.

Run:  python webtool/export_assets_gaia.py   (≈ 20 min: scores 36.7M flights)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from common import CORPUS, MODEL_FEATS, OUT as EXP_OUT  # noqa: E402

ASSETS = Path(__file__).parent / "assets"
RAW = CORPUS.parent                      # GAIA2019-25/{year}/{yyyymmdd}.parquet
MODEL_PATH = EXP_OUT / "gaia2025_pooled.joblib"
CAL_YEAR = 2025
ROUTE_YEARS = [2024, 2025]
MIN_ROUTE_FLIGHTS = 5          # drops 0.7% of flights, 62% of pairs (34 -> ~13 MB)
CAT = "aircraft_type_icao"


def load_year(year: int, cats: list[str], cols: list[str]) -> pd.DataFrame:
    d = pd.read_parquet(CORPUS / f"{year}.parquet", columns=cols)
    if CAT in d:
        d[CAT] = pd.Categorical(d[CAT], categories=cats)
    return d


def export_model_and_calibration():
    t0 = time.time()
    b = joblib.load(MODEL_PATH)
    model, cats = b["model"], b["categories"]
    assert b["features"] == MODEL_FEATS
    model.get_booster().save_model(str(ASSETS / "model_canonical.ubj"))
    print(f"model: {(ASSETS / 'model_canonical.ubj').stat().st_size / 1e6:.0f} MB")

    cols = list(dict.fromkeys(MODEL_FEATS + ["contrail_CO2_km",
                                             "total_flight_distance_km"]))
    d = load_year(CAL_YEAR, cats, cols)
    print(f"{CAL_YEAR}: {len(d):,} flights loaded ({time.time() - t0:.0f}s)")

    counts = d[CAT].value_counts()
    top = [{"icao": k, "n": int(v)} for k, v in counts.head(40).items()]
    with open(ASSETS / "aircraft.json", "w") as f:
        json.dump({"categories": list(cats), "top_types": top}, f)
    print(f"  {len(cats)} aircraft categories")

    pred = model.predict(d[MODEL_FEATS])
    print(f"  scored ({time.time() - t0:.0f}s)")
    q = np.linspace(0, 1, 1001)
    grid = np.quantile(pred, q)
    pct = np.searchsorted(grid, pred, side="right").clip(1, 1000)
    bin100 = ((pct - 1) // 10).clip(0, 99)
    kg_km = d["contrail_CO2_km"].to_numpy(np.float64)
    kg_flight = kg_km * d["total_flight_distance_km"].to_numpy(np.float64)
    g = pd.DataFrame({"b": bin100, "km": kg_km, "fl": kg_flight}).groupby("b")
    with open(ASSETS / "calibration.json", "w") as f:
        json.dump({
            "n_flights": int(len(d)),
            "source": f"all {CAL_YEAR} flights, GAIA 2019-25 pooled model",
            "pred_log_quantiles": [float(x) for x in grid],
            "bin_mean_co2e_kg_per_km": [float(x) for x in g["km"].mean()],
            "bin_mean_co2e_kg_per_flight": [float(x) for x in g["fl"].mean()],
            "threshold_top10_pred_log": float(np.quantile(pred, 0.90)),
        }, f)
    print(f"  {len(d):,} flights calibrated ({time.time() - t0:.0f}s)")


def export_route_meta():
    """routes.json: {"IATA>IATA": {"n": flights, "ac": {type: share}}},
    and airports_search.json: [iata, city, name, country] for airports
    that appear in the corpus. Corpus codes are ICAO."""
    import airportsdata
    icao_tbl = airportsdata.load()
    cols = ["origin_airport", "destination_airport", CAT]
    counts = pd.concat([
        pd.read_parquet(CORPUS / f"{y}.parquet", columns=cols)
        .groupby(cols, observed=True).size() for y in ROUTE_YEARS
    ]).groupby(level=[0, 1, 2]).sum()

    def to_iata(icao):
        e = icao_tbl.get(icao)
        return e["iata"] if e and e.get("iata") else None

    routes, seen = {}, set()
    for (o, d), grp in counts.groupby(level=[0, 1]):
        oi, di = to_iata(o), to_iata(d)
        if not oi or not di:
            continue
        n = int(grp.sum())
        if n < MIN_ROUTE_FLIGHTS:
            continue
        mix = (grp.droplevel([0, 1]).sort_values(ascending=False)
               .head(8) / n).round(4)
        routes[f"{oi}>{di}"] = {"n": n,
                               "ac": {k: float(v) for k, v in mix.items()}}
        seen.update([o, d])
    search = []
    for icao in sorted(seen):
        e = icao_tbl.get(icao)
        if e and e.get("iata"):
            search.append([e["iata"], e.get("city", ""), e.get("name", ""),
                           e.get("country", "")])
    with open(ASSETS / "routes.json", "w") as f:
        json.dump(routes, f)
    with open(ASSETS / "airports_search.json", "w") as f:
        json.dump(search, f)
    print(f"  {len(routes):,} routes, {len(search):,} searchable airports")


def export_flightnos():
    """flightnos.json: IATA flight number -> [origin, dest, median UTC
    departure minute-of-day, modal aircraft, n], from the 2025 day files
    (the corpus drops flight_number). Kept when the top airport pair has
    >=60% share and >=20 observations."""
    import airportsdata
    icao_tbl = airportsdata.load()
    cols = ["flight_number", "origin_airport_icao", "destination_airport_icao",
            "aircraft_type", "first_waypoint_time"]
    pieces = []
    files = sorted((RAW / str(CAL_YEAR)).glob("*.parquet"))
    for i, f in enumerate(files, 1):
        df = pd.read_parquet(f, columns=cols).dropna(subset=["flight_number"])
        df = df[df.flight_number.astype(str).str.len() >= 3]
        t = pd.to_datetime(df.first_waypoint_time)
        df["mod"] = (t.dt.hour * 60 + t.dt.minute).astype("int16")
        pieces.append(df.drop(columns="first_waypoint_time"))
        if i % 50 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}]", flush=True)
    df = pd.concat(pieces, ignore_index=True)
    for c in ("flight_number", "origin_airport_icao", "destination_airport_icao",
              "aircraft_type"):
        df[c] = df[c].astype("category")

    out = {}
    for fn, grp in df.groupby("flight_number", observed=True):
        n = len(grp)
        if n < 20:
            continue
        route = grp.groupby(["origin_airport_icao", "destination_airport_icao"],
                            observed=True).size().sort_values(ascending=False)
        (o, d), top_n = route.index[0], route.iloc[0]
        if top_n / n < 0.6:
            continue
        sub = grp[(grp.origin_airport_icao == o)
                  & (grp.destination_airport_icao == d)]
        oi = (icao_tbl.get(o) or {}).get("iata")
        di = (icao_tbl.get(d) or {}).get("iata")
        if not oi or not di:
            continue
        ac = sub.aircraft_type.mode()
        out[str(fn).upper().replace(" ", "")] = [
            oi, di, int(sub["mod"].median()),
            str(ac.iloc[0]) if len(ac) else "", int(top_n)]
    with open(ASSETS / "flightnos.json", "w") as f:
        json.dump(out, f)
    print(f"  {len(out):,} flight numbers "
          f"({(ASSETS / 'flightnos.json').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    export_model_and_calibration()
    export_route_meta()
    export_flightnos()
    print("\nAll assets written to", ASSETS)
