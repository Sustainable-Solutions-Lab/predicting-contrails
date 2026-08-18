"""
Fig 1b — global map of mean contrail energy forcing per passenger-km.

For each of ~1M sampled commercial flights from 2019 + 2021, we generate
30 great-circle waypoints and accumulate Joulesperpasskm (energy forcing
divided by passenger-km, computed by the contrails team) into a 360x140
lat/lon grid. A 2D Gaussian smoother is applied to both the sum-of-RF
and count-of-flights arrays before dividing, then the result is rendered
with imshow + bicubic interpolation for smooth cell-to-cell transitions.
Country borders are drawn on top of the data as hairlines.

Inputs : the per-month process-model parquets in Dropbox.
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from feature_pruning import DROPBOX_PARQUETS, DROPBOX_PLOTS, OUT as EXP_OUT  # noqa: E402

OUT = DROPBOX_PLOTS

# ── Tunables ──────────────────────────────────────────────────────────────
SAMPLE_FLIGHTS = 800_000             # target sample size
N_FILES = 16                         # random monthly shards to read
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

CACHE_PATH = EXP_OUT / "_fig1_jpkm_sample.parquet"  # gitignored via *.parquet glob


def load_jpkm_sample() -> pd.DataFrame:
    """Read Joulesperpasskm + endpoints from a random subset of monthly
    parquets and persist to a small local cache for fast re-runs."""
    if CACHE_PATH.exists():
        df = pd.read_parquet(CACHE_PATH)
        print(f"Loaded {len(df):,} flights from cache {CACHE_PATH.name}")
        return df

    all_files = sorted(DROPBOX_PARQUETS.glob("features_*_gdf.pq"))
    rng = np.random.default_rng(RANDOM_SEED)
    pick = rng.choice(len(all_files), size=min(N_FILES, len(all_files)), replace=False)
    files = [all_files[i] for i in sorted(pick)]
    print(f"Reading {len(files)} of {len(all_files)} monthly shards "
          f"(target {SAMPLE_FLIGHTS:,} flights) ...", flush=True)

    needed = [
        "OriginLon", "OriginLat",
        "DestinationLon", "DestinationLat",
        "Joulesperpasskm", "total_flight_distance_km",
    ]
    per_file = int(np.ceil(SAMPLE_FLIGHTS / len(files)))
    pieces = []
    n_failed = 0
    for i, f in enumerate(files, 1):
        t = time.time()
        try:
            df = pd.read_parquet(f, columns=needed)
        except OSError as e:
            n_failed += 1
            print(f"  [{i:2d}/{len(files)}] {f.name}: SKIPPED ({e})", flush=True)
            continue
        df = df.dropna(subset=["Joulesperpasskm"])
        if len(df) > per_file:
            df = df.sample(per_file, random_state=RANDOM_SEED + i)
        pieces.append(df)
        print(f"  [{i:2d}/{len(files)}] {f.name}: kept {len(df):,}  "
              f"({time.time()-t:.1f}s)", flush=True)
    if n_failed:
        print(f"  ({n_failed} files skipped due to Dropbox cloud-fetch errors)")

    df = pd.concat(pieces, ignore_index=True)
    if len(df) > SAMPLE_FLIGHTS:
        df = df.sample(SAMPLE_FLIGHTS, random_state=RANDOM_SEED)
    df.to_parquet(CACHE_PATH)
    print(f"Cached {len(df):,} flights to {CACHE_PATH.name}")
    return df


def main():
    t0 = time.time()
    df = load_jpkm_sample()
    print(f"  load done in {time.time()-t0:.0f}s")

    # ── Generate waypoints per flight ─────────────────────────────────────
    print(f"\nGenerating {NB_WAYPOINTS} great-circle waypoints per flight ...")
    geod = Geod(ellps="WGS84")
    n = len(df)
    trajx = np.empty(n * NB_WAYPOINTS, dtype=np.float32)
    trajy = np.empty(n * NB_WAYPOINTS, dtype=np.float32)
    traj_w = np.empty(n * NB_WAYPOINTS, dtype=np.float32)

    jpkm = df["Joulesperpasskm"].to_numpy(dtype=np.float64)
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
        traj_w[s:s + NB_WAYPOINTS] = jpkm[i]
        if (i + 1) % 200_000 == 0:
            elapsed = time.time() - t1
            eta = elapsed / (i + 1) * (n - i - 1)
            print(f"  {i+1:,}/{n:,}   eta {eta/60:.1f} min", flush=True)
    print(f"  waypoints done in {time.time()-t1:.0f}s")

    # ── 2D histogram + smoothing ──────────────────────────────────────────
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

    # Full Spectral_r palette spanning 0 → wine red. Negatives clip to deep blue.
    hex_spectral = [
        "#3288bd", "#66c2a5", "#abdda4", "#e6f598",
        "#ffffbf", "#fee08b", "#fdae61", "#f46d43", "#d53e4f", "#9e0142",
    ]
    cmap = LinearSegmentedColormap.from_list("rf_spectral_r", hex_spectral, N=256)
    # Opaque white for no-data cells: EPS has no transparency, so alpha=0
    # rendered as BLACK when the .eps was placed in Illustrator.
    cmap.set_bad("#ffffff")

    # GJ / passenger-km on a FIXED 0-1.0 scale — identical to the Fig 1c
    # matrix (fig1c_matrix.py VMAX_GJ), so the two panels share one
    # colorbar in the assembled figure.
    mean_rf = mean_rf / 1e9

    norm = colors.Normalize(vmin=0, vmax=1.0, clip=True)

    # Use imshow with bicubic interpolation for smooth cell-to-cell gradient.
    # Mask the NaN regions so the basemap shows through as white.
    masked = np.ma.masked_invalid(mean_rf.T)
    im = ax.imshow(
        masked,
        extent=[lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]],
        origin="lower", cmap=cmap, norm=norm,
        interpolation="bicubic", aspect="auto",
        zorder=1,
    )

    # Country borders ON TOP as hairlines
    if COASTLINE_PATH.exists():
        gpd.read_file(COASTLINE_PATH).boundary.plot(
            ax=ax, color="#222222", lw=0.35, zorder=10,
        )
    else:
        print(f"  warning: coastline file not at {COASTLINE_PATH}")

    cbar = plt.colorbar(im, ax=ax, extend="max", shrink=0.7, pad=0.02)
    cbar.set_label("Mean radiative forcing  (GJ / passenger·km)",
                   fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")

    fig.savefig(OUT / "fig1_map.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig1_map.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig1_map.eps", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved fig1_map.{{png,pdf,eps}}")
    print(f"\nTotal: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
