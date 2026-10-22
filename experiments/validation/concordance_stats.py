#!/usr/bin/env python3
"""Pattern statistics for the literature-concordance check.

Computes, over all 2021 flights (CoCiP-labeled), the aggregate patterns
that published contrail studies report — forcing concentration across
flights, night/day split, seasonal structure, regional shares — so the
paper can show our label corpus and model reproduce independent
findings (Teoh et al., Stuber et al., satellite climatologies).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --gaia2025: the revision's base year from the GAIA 2019-25 corpus
# (vPM-enabled labels). The corpus has no Season / meanlat / Origin_Region
# columns, so season comes from the departure month, mean latitude from
# the endpoints, and region from Fig. 1c's boxes.
REVISION = "--gaia2025" in sys.argv

SHARDS = Path(
    "/Users/stevedavis/Library/CloudStorage/Dropbox/"
    "Papers/In Review/Contrails/WS Corp contrails (w Silas)/adjustedEFs"
)
COLS = ["contrail_CO2_km", "total_flight_distance_km", "night_score_bool_0",
        "night_score_full_0", "Season", "meanlat", "Origin_Region",
        "first_waypoint_time"]

if REVISION:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "figures"))
    from common import BASE_YEAR, load_corpus
    from fig1c_matrix import assign_region
    db = load_corpus([BASE_YEAR], [
        "contrail_CO2_km", "total_flight_distance_km", "night_score_full_0",
        "OriginLat", "OriginLon", "DestinationLat", "first_waypoint_time",
        "aircraft_type_icao"])
    m = pd.to_datetime(db["first_waypoint_time"]).dt.month
    db["Season"] = np.select(
        [m.isin([12, 1, 2]), m.isin([3, 4, 5]), m.isin([6, 7, 8])],
        ["Winter", "Spring", "Summer"], "Autumn")
    db["meanlat"] = (db["OriginLat"] + db["DestinationLat"]) / 2
    db["Origin_Region"] = assign_region(db["OriginLat"].to_numpy(),
                                        db["OriginLon"].to_numpy())
    print(f"GAIA 2019-25 corpus, {BASE_YEAR}")
else:
    frames = []
    for f in sorted(SHARDS.glob("features_2021*_gdf.pq")):
        frames.append(pd.read_parquet(f, columns=COLS))
    db = pd.concat(frames, ignore_index=True)
db["total_kg"] = db["contrail_CO2_km"] * db["total_flight_distance_km"]
db = db.dropna(subset=["total_kg"])
net = db["total_kg"].sum()
n = len(db)
print(f"n={n:,} flights, net forcing {net / 1e9:.2f} Mt CO2e-equivalent "
      f"({net * 82.5e-15 * 365 * 24 * 3600 * 5.101e14 / 1e18:.0f} EJ)\n")

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

if REVISION:
    # twin-aisle types (ICAO designators) for the widebody share in the text
    WIDE = {"A306", "A30B", "A310", "A332", "A333", "A337", "A338", "A339",
            "A342", "A343", "A345", "A346", "A359", "A35K", "A388", "B744",
            "B748", "B762", "B763", "B764", "B772", "B773", "B77L", "B77W",
            "B778", "B779", "B788", "B789", "B78X", "DC10", "IL96", "MD11"}
    wide = db["aircraft_type_icao"].astype(str).isin(WIDE)
    print("\n── Widebodies ──")
    print(f"  {wide.mean():.1%} of flights, "
          f"{db.loc[wide, 'total_kg'].sum() / net:.1%} of net forcing")
    long = db["total_flight_distance_km"] >= 2000
    print(f"  long-haul (>=2000 km) flights that are widebodies: "
          f"{wide[long].mean():.0%}")
