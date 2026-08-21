"""Shared paths, feature definitions, and helpers for the
predicting-contrails experiments.

The model uses 14 features computable from a published flight schedule
alone (MODEL_FEATS): great-circle distance, cyclic day-of-year and
departure-hour encodings, endpoint latitudes and cyclic longitudes,
land-overflight share, route-mean darkness (night_score_full_0), and
aircraft ICAO type (categorical). add_features() computes these — plus
additional engineered encodings retained so that corpus filtering
(finite-value requirements) is reproducible — from the raw shard
columns in NEEDED_RAW_COLS.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DROPBOX_PARQUETS = Path(
    "/Users/stevedavis/Library/CloudStorage/Dropbox/"
    "Papers/Active Prep/Contrails/WS Corp contrails (w Silas)/adjustedEFs"
)
# All raw generated figures go straight here (Steve's hand-polished
# versions live in the sibling Figures/ folder; scripts stay in git).
DROPBOX_PLOTS = Path(
    "/Users/stevedavis/Library/CloudStorage/Dropbox/"
    "Papers/Active Prep/Contrails/WS Corp contrails (w Silas)/Plots"
)
OUT = Path(__file__).parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# How many shards per year. None = use all 47.
SHARDS_PER_YEAR = None
RANDOM_SEED = 42

AGWP_100 = 82.5e-15  # W/m²/kg
S_EARTH = 5.101e14   # m²
SECONDS_PER_YEAR = 365 * 24 * 3600


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["total_flight_distance_km"].astype(float) >= 0.1].copy()

    # Target
    df["contrail_CO2"] = df["total_contrail_energy_forcing"] / (
        AGWP_100 * SECONDS_PER_YEAR * S_EARTH
    )
    df["contrail_CO2_km"] = df["contrail_CO2"] / df["total_flight_distance_km"]

    df["first_waypoint_time"] = pd.to_datetime(df["first_waypoint_time"])
    df["last_waypoint_time"] = pd.to_datetime(df["last_waypoint_time"])

    # Day-of-year cyclic
    doy = df["first_waypoint_time"].dt.dayofyear.astype(float)
    df["day_sin"] = np.sin(np.pi * doy * 2 / 365)
    df["day_cos"] = np.cos(np.pi * doy * 2 / 365)

    # Hour-of-day cyclic at start, end, mid
    for col, src in [
        ("start", df["first_waypoint_time"].dt.hour),
        ("end", df["last_waypoint_time"].dt.hour),
        (
            "mid",
            (
                df["first_waypoint_time"]
                + (df["last_waypoint_time"] - df["first_waypoint_time"]) / 2
            ).dt.hour,
        ),
    ]:
        h = src.astype(float)
        df[f"{col}_hour_sin"] = np.sin(np.pi * h * 2 / 24)
        df[f"{col}_hour_cos"] = np.cos(np.pi * h * 2 / 24)

    df["time_span_hours"] = (
        df["last_waypoint_time"] - df["first_waypoint_time"]
    ).dt.total_seconds() / 3600
    df["crosses_midnight"] = (
        df["last_waypoint_time"].dt.date != df["first_waypoint_time"].dt.date
    ).astype(int)

    # Cyclic bearing & longitude
    df["bearing_sin"] = np.sin(np.pi * df["bearing"].astype(float) * 2 / 360)
    df["bearing_cos"] = np.cos(np.pi * df["bearing"].astype(float) * 2 / 360)
    for c in ["OriginLon", "DestinationLon"]:
        v = df[c].astype(float)
        df[f"{c}_sin"] = np.sin(np.pi * v * 2 / 360)
        df[f"{c}_cos"] = np.cos(np.pi * v * 2 / 360)

    # Signed-log target (matches Silas)
    x = df["contrail_CO2_km"].astype(float)
    df["contrail_type"] = np.where(
        x > 0, np.log(np.where(x > 0, x, 1)),
        np.where(x < 0, -np.log(np.abs(np.where(x < 0, x, -1))), 0.0),
    )
    df = df.replace([np.inf, -np.inf], np.nan)
    return df
# every engineered column add_features() produces — the corpus keeps
# rows where all of these are finite
ALL_ENGINEERED_FEATS = [
    "total_flight_distance_km",
    "day_sin", "day_cos",
    "start_hour_sin", "start_hour_cos",
    "end_hour_sin", "end_hour_cos",
    "mid_hour_sin", "mid_hour_cos",
    "time_span_hours", "crosses_midnight",
    "land_score",
    "bearing_sin", "bearing_cos",
    "OriginLon_sin", "OriginLon_cos",
    "DestinationLon_sin", "DestinationLon_cos",
    "OriginLat", "DestinationLat",
    "night_score_full_0", "night_score_full_1",
    "night_score_full_2", "night_score_full_3",
    "night_score_bool_0", "night_score_bool_1",
    "night_score_bool_2", "night_score_bool_3",
]

MODEL_FEATS = [
    "total_flight_distance_km",
    "day_sin", "day_cos",
    "start_hour_sin", "start_hour_cos",
    "OriginLat", "OriginLon_sin", "OriginLon_cos",
    "DestinationLat", "DestinationLon_sin", "DestinationLon_cos",
    "land_score",
    "night_score_full_0",
    "aircraft_type_icao",
]

CATEGORICAL_FEATS = {"aircraft_type_icao"}

TARGET = "contrail_type"
NEEDED_RAW_COLS = [
    "first_waypoint_time", "last_waypoint_time",
    "total_flight_distance_km", "total_contrail_energy_forcing",
    "bearing", "land_score",
    "OriginLon", "OriginLat", "DestinationLon", "DestinationLat",
    "aircraft_type_icao",
    "night_score_full_0", "night_score_full_1",
    "night_score_full_2", "night_score_full_3",
    "night_score_bool_0", "night_score_bool_1",
    "night_score_bool_2", "night_score_bool_3",
]

FEATURE_OUT_COLS = sorted(set(ALL_ENGINEERED_FEATS + MODEL_FEATS)) + [
    TARGET, "contrail_CO2_km",
]

def load_year(year: int, n_shards: int | None) -> pd.DataFrame:
    files = sorted(DROPBOX_PARQUETS.glob(f"features_{year}*_gdf.pq"))
    if n_shards is None or n_shards >= len(files):
        pick = list(range(len(files)))
    else:
        rng = np.random.default_rng(RANDOM_SEED + year)
        pick = sorted(rng.choice(len(files), size=n_shards, replace=False).tolist())

    print(f"  loading {len(pick)} of {len(files)} shards for {year}")
    pieces = []
    total_raw = 0
    for i in pick:
        f = files[i]
        df = pd.read_parquet(f, columns=NEEDED_RAW_COLS)
        total_raw += len(df)
        # Compute features, downcast, drop everything we don't need.
        df = add_features(df).dropna(subset=ALL_ENGINEERED_FEATS + [TARGET])
        df["aircraft_type_icao"] = df["aircraft_type_icao"].astype("category")
        keep = [c for c in FEATURE_OUT_COLS if c in df.columns]
        df = df[keep]
        for c in df.select_dtypes("float64").columns:
            df[c] = df[c].astype("float32")
        pieces.append(df)
    out = pd.concat(pieces, ignore_index=True)
    # Re-unify the categorical dtype across shards (concat with cat dtypes can
    # create a CategoricalDtype with different categories per shard).
    out["aircraft_type_icao"] = out["aircraft_type_icao"].astype("category")
    print(f"  {year}: {total_raw:,} raw → {len(out):,} clean")
    return out


def top_k_metrics(y_true_co2_km, y_pred_log, k=0.05):
    """y_true_co2_km is per-km contrail CO2-eq. y_pred_log is signed log.
    Returns (top-k recall on warming flights, fraction of total POSITIVE
    forcing concentrated in the predicted top-k)."""
    arr = np.asarray(y_true_co2_km, dtype=float)
    pred = np.asarray(y_pred_log, dtype=float)
    n = arr.size
    n_top = max(1, int(round(n * k)))
    true_top_idx = np.argsort(arr)[::-1][:n_top]
    pred_top_idx = np.argsort(pred)[::-1][:n_top]
    recall = np.intersect1d(true_top_idx, pred_top_idx).size / n_top
    pos_total = np.maximum(arr, 0).sum()
    captured = np.maximum(arr[pred_top_idx], 0).sum()
    capture = captured / pos_total if pos_total > 0 else float("nan")
    return recall, capture
