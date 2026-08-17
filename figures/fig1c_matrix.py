"""
Fig 1c — Region-to-region matrix of mean contrail forcing.

For every origin-region → destination-region pair, the mean per-flight
contrail energy forcing per passenger-km (GJ / passenger-km, same
quantity and color scale as the Fig 1b map), across all 2019+2021
flights in the monthly process-model shards.

Regions are approximate lat/lon boxes chosen to mirror the ten regions
of the polished Figure 1 panel c:

    MENA, E Asia, Oceania, S and SE Asia, S America,
    Sub-Saharan Africa, N America (west), N America (east),
    W Europe, E Europe (incl. Russia)

Cells with fewer than MIN_FLIGHTS flights are left blank. Intra-region
(diagonal) cells get a dashed outline, as in the polished figure.

Inputs : monthly parquets in Dropbox adjustedEFs/ (cached after 1st run)
Output : Plots/fig1c_matrix.{png,pdf,eps} + fig1c_matrix_values.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from feature_pruning import DROPBOX_PARQUETS, DROPBOX_PLOTS, OUT as EXP_OUT  # noqa: E402

OUT = DROPBOX_PLOTS
CACHE_PATH = EXP_OUT / "_fig1c_od_cache.parquet"

MIN_FLIGHTS = 250        # blank cells with fewer flights
VMAX_GJ = 1.0            # color scale top (GJ / passenger-km); extend above

# Same Spectral-style ramp as the Fig 1b map so the two panels can share
# a colorbar in the assembled figure.
HEX_SPECTRAL = [
    "#3288bd", "#66c2a5", "#abdda4", "#e6f598",
    "#ffffbf", "#fee08b", "#fdae61", "#f46d43", "#d53e4f", "#9e0142",
]

# Display order copied from the polished panel: rows top→bottom, columns
# are the reverse, so intra-region cells run bottom-left → top-right.
ROW_ORDER = [
    "MENA", "E Asia", "Oceania", "S and SE Asia", "S America",
    "Sub-Saharan Africa", "N America (west)", "N America (east)",
    "W Europe", "E Europe",
]
COL_ORDER = list(reversed(ROW_ORDER))


def assign_region(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Approximate lat/lon boxes; first matching rule wins."""
    rules = [
        ("Oceania",            (lat < -10) & (lon >= 110) & (lon <= 180)),
        ("S America",          (lon >= -90) & (lon <= -30) & (lat < 12)),
        ("Sub-Saharan Africa", (lon >= -20) & (lon <= 52) & (lat >= -35) & (lat < 15)),
        ("MENA",               ((lon >= -15) & (lon < 40) & (lat >= 15) & (lat < 36))
                             | ((lon >= 40) & (lon < 63) & (lat >= 12) & (lat < 40))),
        ("W Europe",           (lon >= -25) & (lon < 15) & (lat >= 36) & (lat <= 72)),
        ("E Europe",           ((lon >= 15) & (lon < 60) & (lat >= 36) & (lat <= 75))
                             | ((lon >= 60) & (lon <= 180) & (lat >= 45))),
        ("E Asia",             (lon >= 100) & (lon <= 150) & (lat >= 23) & (lat < 45)),
        ("S and SE Asia",      ((lon >= 55) & (lon < 100) & (lat >= 5) & (lat < 40))
                             | ((lon >= 95) & (lon <= 145) & (lat >= -10) & (lat < 23))),
        ("N America (west)",   (lon >= -170) & (lon < -100) & (lat >= 12)),
        ("N America (east)",   (lon >= -100) & (lon < -50) & (lat >= 12)),
    ]
    region = np.full(len(lat), "Other", dtype=object)
    for name, mask in rules:
        region = np.where((region == "Other") & mask & np.isfinite(lat) & np.isfinite(lon),
                          name, region)
    return region


def load_od_table() -> pd.DataFrame:
    if CACHE_PATH.exists():
        df = pd.read_parquet(CACHE_PATH)
        print(f"Loaded {len(df):,} flights from cache {CACHE_PATH.name}")
        return df

    files = sorted(DROPBOX_PARQUETS.glob("features_*_gdf.pq"))
    needed = ["OriginLat", "OriginLon", "DestinationLat", "DestinationLon",
              "Joulesperpasskm"]
    pieces, n_failed = [], 0
    t0 = time.time()
    print(f"Reading {len(files)} monthly shards ...", flush=True)
    for i, f in enumerate(files, 1):
        try:
            df = pd.read_parquet(f, columns=needed)
        except OSError as e:
            n_failed += 1
            print(f"  [{i:2d}/{len(files)}] {f.name}: SKIPPED ({e})", flush=True)
            continue
        pieces.append(df.dropna(subset=["Joulesperpasskm"]))
        if i % 10 == 0 or i == len(files):
            print(f"  [{i:2d}/{len(files)}] {sum(len(p) for p in pieces):,} flights, "
                  f"{time.time()-t0:.0f}s", flush=True)
    if n_failed:
        print(f"  ({n_failed} files skipped due to Dropbox cloud-fetch errors)")

    df = pd.concat(pieces, ignore_index=True)
    df.to_parquet(CACHE_PATH)
    print(f"Cached {len(df):,} flights to {CACHE_PATH.name}")
    return df


def main():
    t0 = time.time()
    df = load_od_table()

    print("Assigning regions ...")
    df["origin_region"] = assign_region(df["OriginLat"].to_numpy(),
                                        df["OriginLon"].to_numpy())
    df["dest_region"] = assign_region(df["DestinationLat"].to_numpy(),
                                      df["DestinationLon"].to_numpy())
    df["gj_per_paxkm"] = df["Joulesperpasskm"] / 1e9

    known = df[(df["origin_region"] != "Other") & (df["dest_region"] != "Other")]
    print(f"  {len(known):,} of {len(df):,} flights have both endpoints "
          f"in a named region")

    g = known.groupby(["origin_region", "dest_region"])["gj_per_paxkm"]
    mean_mat = g.mean().unstack().reindex(index=ROW_ORDER, columns=COL_ORDER)
    count_mat = g.count().unstack().reindex(index=ROW_ORDER, columns=COL_ORDER)
    mean_mat = mean_mat.where(count_mat >= MIN_FLIGHTS)

    # Persist the numbers behind the figure
    tidy = (g.agg(["mean", "count"]).reset_index()
            .rename(columns={"mean": "mean_gj_per_paxkm", "count": "n_flights"}))
    tidy.to_csv(OUT / "fig1c_matrix_values.csv", index=False)

    # ── Render ────────────────────────────────────────────────────────
    cmap = LinearSegmentedColormap.from_list("rf_spectral_r", HEX_SPECTRAL, N=256)
    cmap.set_bad("white")
    norm = colors.Normalize(vmin=0, vmax=VMAX_GJ, clip=True)

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    vals = mean_mat.to_numpy(dtype=float)
    n = len(ROW_ORDER)

    # pcolormesh with white gaps between cells
    mesh = ax.pcolormesh(
        np.ma.masked_invalid(vals[::-1]),   # flip so row 0 renders at top
        cmap=cmap, norm=norm,
        edgecolors="white", linewidth=2.5,
    )

    for r in range(n):
        for c in range(n):
            v = vals[r, c]
            if np.isnan(v):
                continue
            label = "<0.01" if v < 0.005 else f"{v:.2f}"
            ax.text(c + 0.5, (n - 1 - r) + 0.5, label,
                    ha="center", va="center", fontsize=9.5, weight="bold",
                    color="#111111")

    # Dashed outline on intra-region cells
    for r, origin in enumerate(ROW_ORDER):
        c = COL_ORDER.index(origin)
        ax.add_patch(Rectangle((c, n - 1 - r), 1, 1, fill=False,
                               edgecolor="black", linestyle=(0, (2, 2)),
                               linewidth=1.8, zorder=5))

    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_xticklabels(COL_ORDER, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_yticklabels(ROW_ORDER[::-1], fontsize=10)
    ax.set_xlabel("Destination region", fontsize=11)
    ax.set_ylabel("Origin region", fontsize=11)
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    cbar = fig.colorbar(mesh, ax=ax, extend="max", shrink=0.75, pad=0.03)
    cbar.set_label("Mean radiative forcing  (GJ / passenger·km)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout()
    for ext in ("png", "pdf", "eps"):
        fig.savefig(OUT / f"fig1c_matrix.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote fig1c_matrix.{{png,pdf,eps}} to {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
