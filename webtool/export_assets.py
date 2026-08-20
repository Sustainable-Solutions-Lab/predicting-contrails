"""
Build the serving assets for the contrails web tool into webtool/assets/.

  model_web.ubj        distilled XGBoost model (small enough for a git
                       repo / serverless function); metrics vs canonical
                       written to distill_metrics.json
  model_canonical.ubj  full canonical model export (57 MB — reference)
  calibration.json     percentile grid of predicted signed-log forcing
                       over ALL 2021 flights + per-percentile-bin mean
                       actual contrail CO2e (kg/km and kg/flight), so
                       the tool can report honest expected values
  aircraft.json        training category list (order matters for the
                       categorical encoding) + top types for a dropdown
  airports.json        slim IATA → {lat, lon, tz, name} table
  land_union.wkb       Natural Earth 110m land polygons, unioned

Run:  python webtool/export_assets.py            (≈ minutes; needs the
      pool cache and 2021 predictions cache in experiments/outputs)
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
from feature_pruning import (  # noqa: E402
    DROPBOX_PARQUETS, LEAN_AC_FEATS, OUT as EXP_OUT, RANDOM_SEED, TARGET,
    top_k_metrics,
)

ASSETS = Path(__file__).parent / "assets"
ASSETS.mkdir(exist_ok=True)

POOL_CACHE = EXP_OUT / "_pool_cache_all.parquet"
PRED_CACHE = EXP_OUT / "_2021_predictions.parquet"
MODEL_PATH = EXP_OUT / "final_model.joblib"
NE_ZIP = (Path(__file__).resolve().parent.parent / "archived"
          / "sherlock-snapshot" / "ne_110m_admin_0_countries.zip")

DISTILL_PARAMS = dict(
    max_depth=8,
    learning_rate=0.10,
    min_child_weight=50,
    subsample=0.85,
    colsample_bytree=1.0,
    n_estimators=120,
    n_jobs=-1,
    tree_method="hist",
    enable_categorical=True,
    random_state=RANDOM_SEED,
)
DISTILL_TRAIN_ROWS = 8_000_000


def export_models_and_aircraft():
    from sklearn.model_selection import train_test_split
    from xgboost import XGBRegressor

    t0 = time.time()
    print("Loading canonical model ...")
    bundle = joblib.load(MODEL_PATH)
    canonical = bundle["model"]
    canonical.get_booster().save_model(str(ASSETS / "model_canonical.ubj"))

    print("Loading pool cache ...")
    cols = LEAN_AC_FEATS + [TARGET, "contrail_CO2_km", "year"]
    pool = pd.read_parquet(POOL_CACHE, columns=cols)
    pool["aircraft_type_icao"] = pool["aircraft_type_icao"].astype("category")

    # Aircraft categories — must match training exactly (sorted uniques)
    cats = list(pool["aircraft_type_icao"].cat.categories)
    counts = pool["aircraft_type_icao"].value_counts()
    top = [
        {"icao": k, "n": int(v)}
        for k, v in counts.head(40).items()
    ]
    with open(ASSETS / "aircraft.json", "w") as f:
        json.dump({"categories": cats, "top_types": top}, f)
    print(f"  {len(cats)} aircraft categories "
          f"({time.time()-t0:.0f}s)")

    # Same split as final_model.py so distill metrics are comparable
    train, test = train_test_split(
        pool, test_size=1 / 3, random_state=RANDOM_SEED,
        stratify=pool["year"],
    )
    del pool
    tr = train.sample(min(DISTILL_TRAIN_ROWS, len(train)),
                      random_state=RANDOM_SEED)
    del train

    print(f"Training distilled model on {len(tr):,} rows ...")
    t1 = time.time()
    small = XGBRegressor(**DISTILL_PARAMS)
    small.fit(tr[LEAN_AC_FEATS], tr[TARGET])
    print(f"  fit: {time.time()-t1:.0f}s")
    small.get_booster().save_model(str(ASSETS / "model_web.ubj"))

    # Compare capture metrics on the shared test split
    te = test.sample(min(4_000_000, len(test)), random_state=RANDOM_SEED)
    del test
    rows = {}
    for name, model in [("canonical", canonical), ("distilled", small)]:
        yp = model.predict(te[LEAN_AC_FEATS])
        rec10, cap10 = top_k_metrics(te["contrail_CO2_km"], yp, k=0.10)
        rec5, cap5 = top_k_metrics(te["contrail_CO2_km"], yp, k=0.05)
        rows[name] = dict(top10_capture=float(cap10), top10_recall=float(rec10),
                          top5_capture=float(cap5), top5_recall=float(rec5))
        print(f"  {name:10s} top-10% capture: {cap10:.3f}   top-5%: {cap5:.3f}")
    sizes = {p.name: round(p.stat().st_size / 1e6, 1)
             for p in [ASSETS / "model_canonical.ubj", ASSETS / "model_web.ubj"]}
    with open(ASSETS / "distill_metrics.json", "w") as f:
        json.dump({"params": {k: v for k, v in DISTILL_PARAMS.items()
                              if k != "n_jobs"},
                   "train_rows": len(tr), "test_rows": len(te),
                   "metrics": rows, "model_mb": sizes}, f, indent=2)
    print(f"  sizes: {sizes}")


def export_calibration():
    print("Building percentile calibration from 2021 predictions ...")
    db = pd.read_parquet(
        PRED_CACHE,
        columns=["pred_log", "contrail_CO2_km", "total_flight_distance_km"],
    ).dropna(subset=["pred_log"])
    q = np.linspace(0, 1, 1001)
    grid = np.quantile(db["pred_log"], q)

    # Per-percentile-bin mean ACTUAL forcing, for honest expected values
    pct = np.searchsorted(grid, db["pred_log"], side="right").clip(1, 1000)
    bin100 = ((pct - 1) // 10).clip(0, 99)
    g = db.groupby(bin100)
    mean_kg_km = g["contrail_CO2_km"].mean()
    mean_kg_flight = g.apply(
        lambda x: (x["contrail_CO2_km"] * x["total_flight_distance_km"]).mean()
    )
    with open(ASSETS / "calibration.json", "w") as f:
        json.dump({
            "n_flights": int(len(db)),
            "source": "2021 predictions cache, canonical LEAN+AC model",
            "pred_log_quantiles": [float(x) for x in grid],
            "bin_mean_co2e_kg_per_km": [float(x) for x in mean_kg_km],
            "bin_mean_co2e_kg_per_flight": [float(x) for x in mean_kg_flight],
            "threshold_top10_pred_log": float(np.quantile(db["pred_log"], 0.90)),
        }, f)
    print(f"  {len(db):,} flights calibrated")


def export_airports():
    import airportsdata
    ap = airportsdata.load("IATA")
    slim = {
        iata: {"lat": v["lat"], "lon": v["lon"], "tz": v["tz"],
               "name": v["name"], "country": v["country"]}
        for iata, v in ap.items()
    }
    with open(ASSETS / "airports.json", "w") as f:
        json.dump(slim, f)
    print(f"  {len(slim):,} airports written")


def export_land():
    import geopandas as gpd
    from shapely import wkb as shapely_wkb
    from shapely.ops import unary_union
    world = unary_union(gpd.read_file(NE_ZIP).geometry)
    (ASSETS / "land_union.wkb").write_bytes(shapely_wkb.dumps(world))
    print(f"  land union: {(ASSETS/'land_union.wkb').stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    export_airports()
    export_land()
    export_calibration()
    export_models_and_aircraft()
    print("\nAll assets written to webtool/assets/")


def export_route_meta():
    """Stream all shards -> route-level metadata for the web tool.

    routes.json: {"IATA>IATA": {"n": flights, "ac": {type: share...}}}
      — aircraft mix per airport pair (top 8 types), 2019+2021 pooled.
    airports_search.json: slim autocomplete list for airports that
      actually appear in the corpus: [iata, city, name, country].
    Shard airport codes are ICAO; both files are keyed/expressed in
    IATA to match the tool's inputs.
    """
    import airportsdata
    icao_tbl = airportsdata.load()          # ICAO-keyed
    files = sorted(DROPBOX_PARQUETS.glob("features_*_gdf.pq"))
    pieces = []
    for i, f in enumerate(files, 1):
        try:
            df = pd.read_parquet(f, columns=[
                "origin_airport", "destination_airport", "aircraft_type_icao"])
        except OSError:
            continue
        pieces.append(df.groupby(
            ["origin_airport", "destination_airport", "aircraft_type_icao"],
            observed=True).size())
        if i % 20 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] shards read", flush=True)
    counts = pd.concat(pieces).groupby(level=[0, 1, 2]).sum()

    def to_iata(icao):
        e = icao_tbl.get(icao)
        return e["iata"] if e and e.get("iata") else None

    routes = {}
    airports_seen = set()
    for (o, d), grp in counts.groupby(level=[0, 1]):
        oi, di = to_iata(o), to_iata(d)
        if not oi or not di:
            continue
        n = int(grp.sum())
        mix = (grp.droplevel([0, 1]).sort_values(ascending=False)
               .head(8) / n).round(4)
        routes[f"{oi}>{di}"] = {"n": n,
                               "ac": {k: float(v) for k, v in mix.items()}}
        airports_seen.update([o, d])

    search = []
    for icao in sorted(airports_seen):
        e = icao_tbl.get(icao)
        if e and e.get("iata"):
            search.append([e["iata"], e.get("city", ""), e.get("name", ""),
                           e.get("country", "")])

    with open(ASSETS / "routes.json", "w") as f:
        json.dump(routes, f)
    with open(ASSETS / "airports_search.json", "w") as f:
        json.dump(search, f)
    print(f"  {len(routes):,} routes, {len(search):,} searchable airports")
