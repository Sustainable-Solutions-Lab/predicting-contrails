#!/usr/bin/env python3
"""Schedule-only features for flight tables that lack them (GAIA 2019–25).

The GAIA 2019–25 per-flight files carry contrail labels but not the
engineered features our model uses: airport coordinates, great-circle
distance, bearing, land_score and night_score_full_0. This module rebuilds
them with the definitions of Silas's original pipeline (as reproduced
exactly by webtool/features.py), vectorized so ~270M flights take minutes:

  total_flight_distance_km  WGS84 geodesic, origin -> destination airport
                            (Silas overwrote GAIA's flown distance with this)
  bearing                   initial spherical bearing, degrees in [0, 360)
  land_score                100 * share of the STRAIGHT lon/lat segment over
                            Natural Earth 110m land (one value per route)
  night_score_full_0        100 * (1 - mean(max(0, sin(sun altitude)))) over
                            30 geodesic waypoints, times interpolated linearly
                            from first to last waypoint time

Sun position uses the Astronomical Almanac low-precision formulae (about
0.01 deg) plus Saemundsson refraction at standard pressure, in place of
pyephem; `--validate` checks every feature against the stored 2019/2021
shard values.

Usage:
  python experiments/schedule_features.py --validate [--n 200000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
from common import DROPBOX_PARQUETS  # noqa: E402

NB_WAYPOINTS = 30
AIRPORTS = DROPBOX_PARQUETS.parent / "Analysis/airports_icao.parquet"
_geod = Geod(ellps="WGS84")


# ── airports ────────────────────────────────────────────────────────────────
def airport_table(rebuild: bool = False) -> pd.DataFrame:
    """ICAO code -> lon/lat, harvested from the 2019/2021 shards (the
    coordinates our features were originally computed from)."""
    if AIRPORTS.exists() and not rebuild:
        return pd.read_parquet(AIRPORTS)
    parts = []
    for f in sorted(DROPBOX_PARQUETS.glob("features_*_gdf.pq")):
        d = pd.read_parquet(f, columns=[
            "origin_airport", "OriginLon", "OriginLat",
            "destination_airport", "DestinationLon", "DestinationLat"])
        parts += [
            d[["origin_airport", "OriginLon", "OriginLat"]].set_axis(
                ["icao", "lon", "lat"], axis=1),
            d[["destination_airport", "DestinationLon", "DestinationLat"]]
            .set_axis(["icao", "lon", "lat"], axis=1)]
    ap = (pd.concat(parts).dropna().groupby("icao")[["lon", "lat"]]
          .median().reset_index())
    ap.to_parquet(AIRPORTS)
    return ap


def attach_airports(df: pd.DataFrame, ap: pd.DataFrame) -> pd.DataFrame:
    a = ap.set_index("icao")
    df["OriginLon"] = df["origin_airport"].map(a["lon"])
    df["OriginLat"] = df["origin_airport"].map(a["lat"])
    df["DestinationLon"] = df["destination_airport"].map(a["lon"])
    df["DestinationLat"] = df["destination_airport"].map(a["lat"])
    return df


# ── geometry ────────────────────────────────────────────────────────────────
def distance_bearing(olon, olat, dlon, dlat):
    _, _, dist = _geod.inv(olon, olat, dlon, dlat)
    p1, p2 = np.radians(olat), np.radians(dlat)
    dl = np.radians(dlon - olon)
    x = np.cos(p2) * np.sin(dl)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return dist / 1e3, np.degrees(np.arctan2(x, y)) % 360


def waypoints(olon, olat, dlon, dlat, nb=NB_WAYPOINTS):
    """Same points as Geod.npts(nb): nb interior points equally spaced
    along the geodesic (endpoints excluded). Returns (n, nb) lon, lat."""
    az, _, dist = _geod.inv(olon, olat, dlon, dlat)
    frac = np.arange(1, nb + 1) / (nb + 1)
    n = len(olon)
    lon, lat, _ = _geod.fwd(np.repeat(olon, nb), np.repeat(olat, nb),
                            np.repeat(az, nb), (dist[:, None] * frac).ravel())
    return lon.reshape(n, nb), lat.reshape(n, nb)


# ── sun ─────────────────────────────────────────────────────────────────────
def sun_sin_alt(lon, lat, t_unix):
    """sin(apparent solar altitude) for arrays of lon, lat (deg) and UTC
    unix seconds. Astronomical Almanac low-precision sun + refraction."""
    n = t_unix / 86400.0 + 2440587.5 - 2451545.0          # days since J2000
    L = np.radians((280.460 + 0.9856474 * n) % 360)
    g = np.radians((357.528 + 0.9856003 * n) % 360)
    lam = L + np.radians(1.915) * np.sin(g) + np.radians(0.020) * np.sin(2 * g)
    eps = np.radians(23.439 - 4e-7 * n)
    ra = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))
    dec = np.arcsin(np.sin(eps) * np.sin(lam))
    gmst = np.radians((280.46061837 + 360.98564736629 * n) % 360)
    ha = gmst + np.radians(lon) - ra
    phi = np.radians(lat)
    alt = np.arcsin(np.sin(phi) * np.sin(dec)
                    + np.cos(phi) * np.cos(dec) * np.cos(ha))
    h = np.degrees(alt)
    # Saemundsson (1986), arcmin, 1010 hPa / 15 C as pyephem's defaults
    r = 1.02 / np.tan(np.radians(h + 10.3 / (h + 5.11))) / 60.0
    r *= (1010 / 1010) * (283 / (273 + 15))
    h = np.where(h > -1.0, h + r, h)
    return np.sin(np.radians(h))


def night_score(olon, olat, dlon, dlat, t0, t1, chunk=500_000):
    """night_score_full_0 for arrays; t0, t1 datetime64."""
    out = np.empty(len(olon))
    def unix(t):  # unit-safe (shards store datetime64[us], not [ns])
        t = pd.to_datetime(pd.Series(t)).dt.tz_localize(None)
        return ((t - pd.Timestamp("1970-01-01")) / pd.Timedelta("1s")).to_numpy()
    t0s, t1s = unix(t0), unix(t1)
    frac = np.arange(NB_WAYPOINTS) / (NB_WAYPOINTS - 1)
    for s in range(0, len(olon), chunk):
        sl = slice(s, s + chunk)
        lon, lat = waypoints(olon[sl], olat[sl], dlon[sl], dlat[sl])
        tt = t0s[sl, None] + (t1s[sl] - t0s[sl])[:, None] * frac
        sins = np.maximum(0.0, sun_sin_alt(lon, lat, tt))
        out[sl] = 100.0 * (1.0 - sins.mean(1))
    return out


# ── land ────────────────────────────────────────────────────────────────────
def land_scores(df: pd.DataFrame) -> np.ndarray:
    """One shapely intersection per unique route, broadcast to flights."""
    sys.path.insert(0, str(REPO / "webtool"))
    from features import land_score
    key = ["OriginLon", "OriginLat", "DestinationLon", "DestinationLat"]
    routes = df[key].drop_duplicates()
    routes["land_score"] = [land_score(*r) for r in routes.itertuples(index=False)]
    return df[key].merge(routes, on=key, how="left")["land_score"].to_numpy()


def add_schedule_features(df: pd.DataFrame, ap: pd.DataFrame | None = None
                          ) -> pd.DataFrame:
    """Expects origin_airport, destination_airport (ICAO),
    first_waypoint_time, last_waypoint_time. Adds coordinates, distance,
    bearing, land_score and night_score_full_0."""
    if ap is not None:
        df = attach_airports(df, ap)
    ok = df[["OriginLon", "OriginLat", "DestinationLon", "DestinationLat"]].notna().all(1)
    df = df[ok].copy()
    o = [df[c].to_numpy(float) for c in
         ("OriginLon", "OriginLat", "DestinationLon", "DestinationLat")]
    df["total_flight_distance_km"], df["bearing"] = distance_bearing(*o)
    df["night_score_full_0"] = night_score(*o, df["first_waypoint_time"],
                                           df["last_waypoint_time"])
    df["land_score"] = land_scores(df)
    return df


# ── validation against the stored 2019/2021 features ────────────────────────
def validate(n: int) -> None:
    import time
    cols = ["flight_id", "origin_airport", "destination_airport",
            "OriginLon", "OriginLat", "DestinationLon", "DestinationLat",
            "first_waypoint_time", "last_waypoint_time",
            "total_flight_distance_km", "bearing", "land_score",
            "night_score_full_0"]
    files = sorted(DROPBOX_PARQUETS.glob("features_*_gdf.pq"))
    per = max(1, n // len(files))
    d = pd.concat([pd.read_parquet(f, columns=cols).sample(per, random_state=0)
                   for f in files], ignore_index=True)
    ref = d.copy()
    ap = airport_table()
    new = d.drop(columns=["OriginLon", "OriginLat", "DestinationLon",
                          "DestinationLat", "total_flight_distance_km",
                          "bearing", "land_score", "night_score_full_0"])
    t = time.time()
    new = add_schedule_features(new, ap)
    el = time.time() - t
    m = ref.merge(new, on="flight_id", suffixes=("_old", "_new"))
    print(f"{len(d):,} flights sampled from {len(files)} shards; "
          f"{len(m):,} with airports found ({len(m) / len(d):.2%}); "
          f"{el:.0f}s ({el / len(m) * 1e6:.1f} us/flight)")
    for c in ["OriginLat", "total_flight_distance_km", "bearing",
              "land_score", "night_score_full_0"]:
        diff = (m[f"{c}_new"] - m[f"{c}_old"]).abs()
        if c == "bearing":
            diff = np.minimum(diff, 360 - diff)
        print(f"  {c:26s} mean|diff| {diff.mean():.4f}  p99 {diff.quantile(.99):.4f}"
              f"  max {diff.max():.4f}  corr {np.corrcoef(m[f'{c}_new'], m[f'{c}_old'])[0, 1]:.6f}")


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--validate", action="store_true")
    ap_.add_argument("--n", type=int, default=200_000)
    ap_.add_argument("--rebuild-airports", action="store_true")
    a = ap_.parse_args()
    if a.rebuild_airports:
        print(len(airport_table(rebuild=True)), "airports")
    if a.validate:
        validate(a.n)
