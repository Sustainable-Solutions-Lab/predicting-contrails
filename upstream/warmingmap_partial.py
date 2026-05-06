import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod
from tqdm import tqdm


# Geod.npts(..., nb) places nb intermediate points along the full geodesic (endpoints not included).
# ~1 sample per MAX_STEP_KM reduces gaps between histogram cells vs. sparse waypoints.
DEFAULT_MAX_STEP_KM = 30.0
DEFAULT_NB_CAP = 800

COLUMNS_NEEDED = [
    "Tempmultiplier",
    "total_flight_distance_km",
    "OriginLon",
    "OriginLat",
    "DestinationLon",
    "DestinationLat",
    "Joulesperpasskm",
]


def great_circle_coords(geod: Geod, dep_lon: float, dep_lat: float, arr_lon: float, arr_lat: float, nb: int) -> np.ndarray:
    return np.array(geod.npts(dep_lon, dep_lat, arr_lon, arr_lat, nb))


def flush_points_to_grid(
    xs: list[np.ndarray],
    ys: list[np.ndarray],
    ws: list[np.ndarray],
    sum_grid: np.ndarray,
    count_grid: np.ndarray,
    lon_bins: int,
    lat_bins: int,
    hist_range: tuple[list[float], list[float]],
) -> None:
    if not xs:
        return

    trajx = np.concatenate(xs)
    trajy = np.concatenate(ys)
    trajw = np.concatenate(ws)

    weighted_hist, _, _ = np.histogram2d(
        trajx,
        trajy,
        bins=[lon_bins, lat_bins],
        range=hist_range,
        weights=trajw,
    )
    count_hist, _, _ = np.histogram2d(
        trajx,
        trajy,
        bins=[lon_bins, lat_bins],
        range=hist_range,
    )

    sum_grid += weighted_hist
    count_grid += count_hist

    xs.clear()
    ys.clear()
    ws.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build partial warming-map histograms from one parquet input.")
    parser.add_argument("--input", required=True, help="Path to one parquet shard.")
    parser.add_argument("--output", required=True, help="Path to output .npz partial file.")
    parser.add_argument("--lon-bins", type=int, default=360)
    parser.add_argument("--lat-bins", type=int, default=140)
    parser.add_argument("--flush-points", type=int, default=1_000_000, help="Flush buffered trajectory points to histograms.")
    parser.add_argument(
        "--max-step-km",
        type=float,
        default=DEFAULT_MAX_STEP_KM,
        help="Target spacing along great circle: nb ≈ ceil(distance_km / this). Smaller = denser, slower.",
    )
    parser.add_argument(
        "--nb-cap",
        type=int,
        default=DEFAULT_NB_CAP,
        help="Upper bound on intermediate points per leg (Geod.npts).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hist_range = ([-180.0, 180.0], [-90.0, 90.0])
    xedges = np.linspace(hist_range[0][0], hist_range[0][1], args.lon_bins + 1)
    yedges = np.linspace(hist_range[1][0], hist_range[1][1], args.lat_bins + 1)
    sum_grid = np.zeros((args.lon_bins, args.lat_bins), dtype=np.float64)
    count_grid = np.zeros((args.lon_bins, args.lat_bins), dtype=np.float64)

    df = pd.read_parquet(input_path, columns=COLUMNS_NEEDED)
    df["Joulesperpasskm"] = df["Joulesperpasskm"] * df["Tempmultiplier"]

    geod = Geod(ellps="WGS84")
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    ws: list[np.ndarray] = []
    buffered_points = 0

    cols = [
        "OriginLon",
        "OriginLat",
        "DestinationLon",
        "DestinationLat",
        "Joulesperpasskm",
        "total_flight_distance_km",
    ]
    for row in tqdm(df[cols].values, desc=f"processing {input_path.name}"):
        dist_km = row[5]
        j_value = row[4]
        nb_pts = max(2, min(args.nb_cap, int(np.ceil(dist_km / args.max_step_km))))
        geo = great_circle_coords(geod, row[0], row[1], row[2], row[3], nb_pts)
        xs.append(geo[:, 0])
        ys.append(geo[:, 1])
        ws.append(np.full(geo.shape[0], j_value))
        buffered_points += geo.shape[0]

        if buffered_points >= args.flush_points:
            flush_points_to_grid(
                xs=xs,
                ys=ys,
                ws=ws,
                sum_grid=sum_grid,
                count_grid=count_grid,
                lon_bins=args.lon_bins,
                lat_bins=args.lat_bins,
                hist_range=hist_range,
            )
            buffered_points = 0

    flush_points_to_grid(
        xs=xs,
        ys=ys,
        ws=ws,
        sum_grid=sum_grid,
        count_grid=count_grid,
        lon_bins=args.lon_bins,
        lat_bins=args.lat_bins,
        hist_range=hist_range,
    )

    np.savez_compressed(
        output_path,
        sum_co2=sum_grid,
        count_flights=count_grid,
        xedges=xedges,
        yedges=yedges,
        source_file=str(input_path),
    )
    print(f"wrote partial: {output_path}")


if __name__ == "__main__":
    main()
