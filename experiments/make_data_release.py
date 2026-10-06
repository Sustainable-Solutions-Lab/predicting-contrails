#!/usr/bin/env python3
"""Package the cleaned per-flight training table for the Zenodo deposit.

Splits experiments/outputs/_pool_cache_all.parquet (built by
build_cache.py) into one parquet per year, writes a README describing
every column, and records SHA-256 checksums. The table holds engineered
schedule features and contrail-forcing labels only: no flight numbers,
tail numbers, callsigns or airport codes. The corporate flight logs are
never included.

Usage: python make_data_release.py OUTPUT_DIR
"""
import hashlib
import sys
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

from common import OUT

CACHE = OUT / "_pool_cache_all.parquet"

COLUMNS = {
    "contrail_CO2_km": "Contrail forcing per km: the flight's contrail energy forcing (net of cooling) converted to CO2-equivalent mass with AGWP100 = 8.25e-14 W m-2 yr (kg CO2)-1 (1 kg CO2-e per 1.327 GJ), divided by great-circle distance. Units: kg CO2-e per km. Multiply by 1.327e9 J/kg to recover energy forcing in J per km.",
    "contrail_type": "Regression target despite its name: signed natural log of contrail_CO2_km (ln x for x > 0, -ln(-x) for x < 0, 0 for x = 0).",
    "total_flight_distance_km": "Great-circle flight distance (km).",
    "aircraft_type_icao": "ICAO aircraft type designator (e.g. B77W, A320).",
    "OriginLat": "Origin airport latitude (degrees north).",
    "DestinationLat": "Destination airport latitude (degrees north).",
    "OriginLon_sin, OriginLon_cos": "Origin longitude as sin/cos of 2*pi*lon/360.",
    "DestinationLon_sin, DestinationLon_cos": "Destination longitude as sin/cos of 2*pi*lon/360.",
    "day_sin, day_cos": "Day of year (UTC, first waypoint) as sin/cos of 2*pi*doy/365.",
    "start_hour_sin, start_hour_cos": "Departure hour (UTC, first waypoint) as sin/cos of 2*pi*hour/24.",
    "end_hour_sin, end_hour_cos": "Arrival hour (UTC, last waypoint), cyclic. Not a model feature.",
    "mid_hour_sin, mid_hour_cos": "Hour at flight midpoint (UTC), cyclic. Not a model feature.",
    "time_span_hours": "Flight duration, last minus first waypoint (h). Not a model feature.",
    "crosses_midnight": "1 if the UTC date changes during the flight. Not a model feature.",
    "bearing_sin, bearing_cos": "Initial great-circle bearing, cyclic. Not a model feature.",
    "land_score": "Share of the great-circle route over land (0-1).",
    "night_score_full_0": "Insolation feature: sun-altitude-weighted fraction of the great-circle route in darkness (0-100), from solar geometry at 30 interpolated waypoints. Model feature.",
    "night_score_full_1..3": "Same, for the second to fourth quarters of the route. Not model features.",
    "night_score_bool_0..3": "Unweighted (sun up/down) darkness fraction for the whole route and its quarters. Not model features.",
}

MODEL_FEATS = ("total_flight_distance_km, day_sin, day_cos, start_hour_sin, "
               "start_hour_cos, OriginLat, OriginLon_sin, OriginLon_cos, "
               "DestinationLat, DestinationLon_sin, DestinationLon_cos, "
               "land_score, night_score_full_0, aircraft_type_icao")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(CACHE)
    counts = {}
    for year in (2019, 2021):
        part = table.filter(pc.equal(table["year"], year)).drop(["year"])
        path = out / f"contrail_forcing_features_{year}.parquet"
        pq.write_table(part, path, compression="zstd")
        counts[year] = part.num_rows
        print(f"{path.name}: {part.num_rows:,} rows", flush=True)

    lines = [
        "# Per-flight contrail forcing and schedule features, 2019 and 2021",
        "",
        "Training data for: Davis, S. J., Whiteson, S., Bonnemaizon, X., "
        "Caldeira, K., Teoh, R. & Shapiro, M. Flyers can identify the "
        "most-warming flights when booking (2026).",
        "",
        f"Two Parquet files, one per year: {counts[2019]:,} flights (2019) "
        f"and {counts[2021]:,} flights (2021), {sum(counts.values()):,} in "
        "total. Each row is one commercial flight. Labels are per-flight "
        "contrail energy forcing from the GAIA global aviation emissions "
        "inventory (Teoh et al. 2024, Atmos. Chem. Phys. 24, 6071-6093), "
        "simulated with the Contrail Cirrus Prediction model (CoCiP) on "
        "flown trajectories and ERA5 weather. Rows with any non-finite "
        "engineered feature, and flights shorter than 0.1 km, are removed.",
        "",
        "The files contain no flight numbers, tail numbers, callsigns or "
        "airport codes. Proprietary corporate flight logs used in the "
        "paper's case study are not included.",
        "",
        f"The scheduling model uses 14 columns: {MODEL_FEATS}. Its "
        "regression target is contrail_type. Code: "
        "https://github.com/Sustainable-Solutions-Lab/predicting-contrails",
        "",
        "## Columns",
        "",
    ]
    lines += [f"- `{k}`: {v}" for k, v in COLUMNS.items()]
    lines += ["", "## Checksums (SHA-256)", ""]
    for year in (2019, 2021):
        p = out / f"contrail_forcing_features_{year}.parquet"
        lines.append(f"- `{p.name}`: {sha256(p)}")
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print("wrote README.md")


if __name__ == "__main__":
    main()
