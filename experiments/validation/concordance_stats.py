#!/usr/bin/env python3
"""Pattern statistics for the literature-concordance check.

Computes, over all 2021 flights (CoCiP-labeled), the aggregate patterns
that published contrail studies report — forcing concentration across
flights, night/day split, seasonal structure, regional shares — so the
paper can show our label corpus and model reproduce independent
findings (Teoh et al., Stuber et al., satellite climatologies).
"""
from pathlib import Path

import numpy as np
import pandas as pd

SHARDS = Path(
    "/Users/stevedavis/Library/CloudStorage/Dropbox/"
    "Papers/Active Prep/Contrails/WS Corp contrails (w Silas)/adjustedEFs"
)
COLS = ["contrail_CO2_km", "total_flight_distance_km", "night_score_bool_0",
        "night_score_full_0", "Season", "meanlat", "Origin_Region",
        "first_waypoint_time"]

frames = []
for f in sorted(SHARDS.glob("features_2021*_gdf.pq")):
    frames.append(pd.read_parquet(f, columns=COLS))
db = pd.concat(frames, ignore_index=True)
db["total_kg"] = db["contrail_CO2_km"] * db["total_flight_distance_km"]
db = db.dropna(subset=["total_kg"])
net = db["total_kg"].sum()
n = len(db)
print(f"n={n:,} flights, net forcing {net / 1e9:.2f} Mt CO2e-equivalent\n")

print("── Concentration (ranked by per-flight total forcing) ──")
srt = np.sort(db["total_kg"].to_numpy())[::-1]
cum = np.cumsum(srt)
for pct in (1, 2, 5, 10, 20):
    k = int(round(n * pct / 100))
    print(f"  worst {pct:>2}% of flights -> {100 * cum[k - 1] / net:5.1f}% of net forcing")
warm = db["total_kg"] > 0
print(f"  warming flights: {warm.mean():.0%}; cooling: {(db['total_kg'] < 0).mean():.0%}")

print("\n── Night vs day (route-mean darkness > 50%) ──")
night = db["night_score_full_0"] > 50
print(f"  night flights: {night.mean():.0%} of flights, "
      f"{db.loc[night, 'total_kg'].sum() / net:.0%} of net forcing")
print(f"  day flights net contribution: {db.loc[~night, 'total_kg'].sum() / net:+.0%}")

print("\n── Season (NH mid-latitude flights, meanlat > 30) ──")
nh = db[db["meanlat"] > 30]
nh_net = nh["total_kg"].sum()
for s, g in nh.groupby("Season"):
    print(f"  {s:>8}: {len(g) / len(nh):5.1%} of flights, "
          f"{g['total_kg'].sum() / nh_net:6.1%} of NH-midlat net forcing, "
          f"mean {g['total_kg'].mean():7.0f} kg/flight")

print("\n── Regions (by origin) ──")
reg = db.groupby("Origin_Region").agg(
    flights=("total_kg", "size"), net=("total_kg", "sum"),
    km=("total_flight_distance_km", "sum"))
reg["net_share"] = reg["net"] / net
reg["per_km"] = reg["net"] / reg["km"]
print(reg.sort_values("net_share", ascending=False)
      [["flights", "net_share", "per_km"]].to_string(
          formatters={"net_share": "{:.1%}".format, "per_km": "{:.2f}".format}))
