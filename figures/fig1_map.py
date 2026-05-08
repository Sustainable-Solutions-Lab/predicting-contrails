"""
Fig 1b — global map of mean contrail forcing.

Replaces Silas's warmingmap.py with a smoother render. The "dotted" look
in the prior version came from sparse 2D histogram bins (only a few
thousand of 50k cells populated). Fix: apply a 2D Gaussian smoothing
filter to both the sum-of-RF and count-of-flights arrays before dividing,
which fills in adjacent cells with weighted averages. Cells that remain
below the minimum-flight threshold even after smoothing stay masked.

Inputs : the cached pooled feature parquet from the feature-pruning run
         (52M flights, all months 2019+2021).
Output : figures/outputs/fig1_map.{png,pdf}
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from pyproj import Geod
from scipy.ndimage import gaussian_filter

# Re-use paths and feature lists from the experiment.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from feature_pruning import OUT as EXP_OUT  # noqa: E402

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

CACHE = EXP_OUT / "_pool_cache_all.parquet"

# ── Tunables ──────────────────────────────────────────────────────────────
SAMPLE_FLIGHTS = 1_000_000          # subsample for speed; visual is saturated
NB_WAYPOINTS = 30                    # great-circle interpolation per flight
LON_BINS = 360                       # 1° lon
LAT_BINS = 140                       # 1° lat from -60 to 80
SMOOTH_SIGMA = 3.0                   # cells; higher = smoother
MIN_FLIGHTS_PER_CELL = 20            # post-smoothing threshold
COASTLINE_PATH = (
    Path(__file__).resolve().parent.parent
    / "archived" / "sherlock-snapshot" / "ne_110m_admin_0_countries.zip"
)
RANDOM_SEED = 42


def great_circle_xy(lon0, lat0, lon1, lat1, nb=NB_WAYPOINTS):
    geod = Geod(ellps="WGS84")
    pts = geod.npts(lon0, lat0, lon1, lat1, nb)
    return np.asarray(pts)


def main():
    t0 = time.time()
    print(f"Loading cache {CACHE.name} ...")
    df = pd.read_parquet(
        CACHE,
        columns=[
            "OriginLon_sin", "OriginLon_cos", "OriginLat",
            "DestinationLon_sin", "DestinationLon_cos", "DestinationLat",
            "contrail_CO2_km", "total_flight_distance_km",
        ],
    )
    # Recover raw longitude from cyclic encoding:
    # encoder used  sin(pi * lon * 2 / 360) → atan2(sin, cos) * 180/pi
    df["OriginLon"] = np.degrees(
        np.arctan2(df["OriginLon_sin"].astype(np.float64),
                   df["OriginLon_cos"].astype(np.float64))
    )
    df["DestinationLon"] = np.degrees(
        np.arctan2(df["DestinationLon_sin"].astype(np.float64),
                   df["DestinationLon_cos"].astype(np.float64))
    )
    df = df[[
        "OriginLon", "OriginLat", "DestinationLon", "DestinationLat",
        "contrail_CO2_km", "total_flight_distance_km",
    ]]
    print(f"  {len(df):,} rows in {time.time()-t0:.1f}s")

    rng = np.random.default_rng(RANDOM_SEED)
    if SAMPLE_FLIGHTS and len(df) > SAMPLE_FLIGHTS:
        idx = rng.choice(len(df), size=SAMPLE_FLIGHTS, replace=False)
        df = df.iloc[idx].reset_index(drop=True)
    print(f"  sampled to {len(df):,} flights")

    # ── Generate waypoints per flight ─────────────────────────────────────
    print(f"\nGenerating ~{NB_WAYPOINTS} great-circle waypoints per flight ...")
    geod = Geod(ellps="WGS84")
    n = len(df)
    trajx = np.empty(n * NB_WAYPOINTS, dtype=np.float32)
    trajy = np.empty(n * NB_WAYPOINTS, dtype=np.float32)
    traj_w = np.empty(n * NB_WAYPOINTS, dtype=np.float32)

    co2_km = df["contrail_CO2_km"].to_numpy()
    olon = df["OriginLon"].to_numpy()
    olat = df["OriginLat"].to_numpy()
    dlon = df["DestinationLon"].to_numpy()
    dlat = df["DestinationLat"].to_numpy()

    t1 = time.time()
    for i in range(n):
        pts = geod.npts(olon[i], olat[i], dlon[i], dlat[i], NB_WAYPOINTS)
        arr = np.asarray(pts, dtype=np.float32)
        s = i * NB_WAYPOINTS
        trajx[s:s + NB_WAYPOINTS] = arr[:, 0]
        trajy[s:s + NB_WAYPOINTS] = arr[:, 1]
        traj_w[s:s + NB_WAYPOINTS] = co2_km[i]
        if (i + 1) % 100_000 == 0:
            elapsed = time.time() - t1
            eta = elapsed / (i + 1) * (n - i - 1)
            print(f"  {i+1:,}/{n:,}   eta {eta/60:.1f} min", flush=True)
    print(f"  waypoints done in {time.time()-t1:.0f}s")

    # ── 2D histogram with smoothing ───────────────────────────────────────
    print("\nBinning + smoothing ...")
    lon_edges = np.linspace(-180, 180, LON_BINS + 1)
    lat_edges = np.linspace(-60, 80, LAT_BINS + 1)
    sum_rf, _, _ = np.histogram2d(
        trajx, trajy, bins=[lon_edges, lat_edges], weights=traj_w,
    )
    count, _, _ = np.histogram2d(trajx, trajy, bins=[lon_edges, lat_edges])

    sum_s = gaussian_filter(sum_rf, sigma=SMOOTH_SIGMA)
    count_s = gaussian_filter(count, sigma=SMOOTH_SIGMA)
    mean_rf = np.divide(sum_s, count_s, out=np.full_like(sum_s, np.nan),
                        where=count_s > MIN_FLIGHTS_PER_CELL)
    print(f"  cells filled: {np.sum(~np.isnan(mean_rf)):,} of {mean_rf.size:,}")

    # ── Plot ─────────────────────────────────────────────────────────────
    print("\nRendering ...")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    if COASTLINE_PATH.exists():
        gpd.read_file(COASTLINE_PATH).plot(
            ax=ax, color="white", edgecolor="#444444", lw=0.4,
        )
    else:
        print(f"  warning: coastline file not at {COASTLINE_PATH}")

    # Full Spectral palette spanning 0 (deep blue) → wine red. Most
    # cells are weakly positive on average, so a positive-only span
    # uses the full color budget for the warming gradient.
    hex_spectral = [
        "#3288bd", "#66c2a5", "#abdda4", "#e6f598",
        "#ffffbf", "#fee08b", "#fdae61", "#f46d43", "#d53e4f", "#9e0142",
    ]
    cmap = LinearSegmentedColormap.from_list("rf_spectral_r", hex_spectral, N=256)
    cmap.set_bad(alpha=0)

    # vmax = 95th percentile of positive cells (negatives clip to deep blue)
    positive = mean_rf[np.isfinite(mean_rf) & (mean_rf > 0)]
    vmax = np.nanpercentile(positive, 95) if positive.size else 1.0
    norm = colors.Normalize(vmin=0, vmax=vmax, clip=True)

    im = ax.pcolormesh(
        lon_edges, lat_edges, mean_rf.T,
        cmap=cmap, norm=norm, shading="auto",
    )

    cbar = plt.colorbar(im, ax=ax, extend="max", shrink=0.7, pad=0.02)
    cbar.set_label("Mean contrail forcing  (kg CO₂-eq / km)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(
        f"Fig 1b — Mean contrail forcing along flown great-circles "
        f"({SAMPLE_FLIGHTS:,} flights sampled, σ={SMOOTH_SIGMA} cells)"
    )

    fig.savefig(OUT / "fig1_map.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig1_map.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved fig1_map.{{png,pdf}}")
    print(f"\nTotal: {time.time()-t0:.0f}s")
    # Mirror to Dropbox Plots/Figure 1/
    import subprocess
    subprocess.run([sys.executable, str(Path(__file__).parent / "sync_to_dropbox.py")])


if __name__ == "__main__":
    main()
