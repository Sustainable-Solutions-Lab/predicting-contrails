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

With --gaia2025 the labels come from the GAIA 2019–25 corpus (vPM-enabled
CoCiP; gaia2025_corpus.py), all years pooled.

Usage: python experiments/engine_split.py [--gaia2025]
"""
from __future__ import annotations

import sys
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


def load_old() -> pd.DataFrame:
    parts = []
    for f in sorted(SHARDS.glob("features_*_gdf.pq")):
        d = pd.read_parquet(f, columns=COLS)
        parts.append(d[d["aircraft_type_icao"].isin(TYPES)])
    return pd.concat(parts, ignore_index=True)


def load_gaia2025() -> pd.DataFrame:
    parts = []
    for f in sorted((SHARDS.parent / "GAIA2019-25/corpus").glob("20??.parquet")):
        d = pd.read_parquet(f, columns=[
            "aircraft_type", "engine_name", "origin_airport",
            "destination_airport", "first_waypoint_time",
            "total_flight_distance_km", "total_persistent_contrail_length_km",
            "ef_sum"])
        d = d[d["aircraft_type"].isin(TYPES)]
        parts.append(d.rename(columns={
            "aircraft_type": "aircraft_type_icao",
            "ef_sum": "total_contrail_energy_forcing"}))
    df = pd.concat(parts, ignore_index=True)
    df["total_contrail_energy_forcing"] = (
        df["total_contrail_energy_forcing"].fillna(0).astype(float))
    df["total_persistent_contrail_length_km"] = (
        df["total_persistent_contrail_length_km"].fillna(0))
    return df


def main() -> None:
    new = "--gaia2025" in sys.argv
    df = load_gaia2025() if new else load_old()
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
    name = "engine_split_gaia2025.csv" if new else "engine_split.csv"
    out.to_csv(SHARDS.parent / "Plots" / name, index=False)


if __name__ == "__main__":
    main()
