#!/usr/bin/env python3
"""Training labels split by engine within aircraft types that fly two
engine families: A320neo / A321neo (PW1100G vs LEAP-1A) and 787-8/-9/-10
(GEnx-1B vs Trent 1000).

The scheduling model sees aircraft type, not engine, so its type effect is
an average over whichever engines each type flies. This reports, per
engine family, the share of flights forming persistent contrails and the
mean contrail energy forcing per km, both raw and matched on
route x month (cells where both engine families fly), since airlines and
routes differ between engines.

Usage: python experiments/engine_split.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SHARDS = Path(
    "/Users/stevedavis/Library/CloudStorage/Dropbox/"
    "Papers/In Review/Contrails/WS Corp contrails (w Silas)/adjustedEFs"
)
TYPES = ["A20N", "A21N", "B788", "B789", "B78X"]
COLS = ["aircraft_type_icao", "engine_name", "origin_airport",
        "destination_airport", "first_waypoint_time",
        "total_flight_distance_km", "total_persistent_contrail_length_km",
        "total_contrail_energy_forcing"]


def family(name: str) -> str:
    n = str(name).upper()
    for key, fam in [("PW11", "PW1100G"), ("LEAP-1A", "LEAP-1A"),
                     ("GENX", "GEnx-1B"), ("TRENT 1000", "Trent 1000")]:
        if key in n:
            return fam
    return "other"


def main() -> None:
    parts = []
    for f in sorted(SHARDS.glob("features_*_gdf.pq")):
        d = pd.read_parquet(f, columns=COLS)
        parts.append(d[d["aircraft_type_icao"].isin(TYPES)])
    df = pd.concat(parts, ignore_index=True)
    df = df[(df["total_flight_distance_km"] > 0)
            & np.isfinite(df["total_contrail_energy_forcing"])]
    df["fam"] = df["engine_name"].map(family)
    df["formed"] = df["total_persistent_contrail_length_km"] > 0
    df["gj_km"] = df["total_contrail_energy_forcing"] / df["total_flight_distance_km"] / 1e9
    df["month"] = df["first_waypoint_time"].dt.to_period("M")
    df["cell"] = (df["origin_airport"] + "-" + df["destination_airport"]
                  + "-" + df["month"].astype(str))

    groups = {"A320neo": ["A20N"], "A321neo": ["A21N"],
              "787 (all)": ["B788", "B789", "B78X"]}
    rows = []
    for g, types in groups.items():
        sub = df[df["aircraft_type_icao"].isin(types) & (df["fam"] != "other")]
        fams = sub["fam"].value_counts()
        fams = fams[fams > 1000].index.tolist()
        if len(fams) != 2:
            continue
        a, b = fams
        # route x month cells flown by both families: compare within cell,
        # weight each cell by its smaller flight count
        cell = (sub[sub["fam"].isin(fams)]
                .groupby(["cell", "fam"])
                .agg(n=("gj_km", "size"), gj=("gj_km", "mean"),
                     form=("formed", "mean"))
                .unstack("fam").dropna())
        w = np.minimum(cell[("n", a)], cell[("n", b)])
        for fam in fams:
            s = sub[sub["fam"] == fam]
            rows.append({
                "type": g, "engine": fam, "flights": len(s),
                "share_of_type": len(s) / len(sub),
                "formed_raw": s["formed"].mean(),
                "gj_km_raw": s["gj_km"].mean(),
                "matched_cells": len(cell),
                "matched_flights": int(w.sum()),
                "formed_matched": np.average(cell[("form", fam)], weights=w),
                "gj_km_matched": np.average(cell[("gj", fam)], weights=w),
            })
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
    out.to_csv(SHARDS.parent / "Plots/engine_split.csv", index=False)


if __name__ == "__main__":
    main()
