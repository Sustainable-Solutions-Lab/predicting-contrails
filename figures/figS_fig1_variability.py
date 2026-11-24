"""
SI figure: how variable are the flights behind the Fig. 1 means?

A referee asked for the standard deviation of the populations averaged in
Fig. 1b (map) and Fig. 1c (region matrix). Panel a maps the standard
deviation of per-passenger-km forcing across the flights whose great-circle
tracks cross each 1-degree cell (same waypoints, smoothing and sample as
fig1_map.py); panel b gives the standard deviation for every region pair of
Fig. 1c, with the number of flights behind each cell written in the CSV.

Run:  python figures/figS_fig1_variability.py --gaia2025
Outputs (Plots/Revision (GAIA)): figS_fig1_variability.{png,pdf},
figS_fig1_variability_matrix.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "experiments"))
sys.path.insert(0, str(Path.home() / ".claude/skills/paper-figures/scripts"))
import sjd_style as sjd  # noqa: E402
from common import REVISION_PLOTS  # noqa: E402
from fig1_map import (COASTLINE_PATH, LAT_BINS, LON_BINS,  # noqa: E402
                      MIN_FLIGHTS_PER_CELL, NB_WAYPOINTS, SAMPLE_FLIGHTS,
                      SMOOTH_SIGMA, load_gaia2025)
from fig1c_matrix import COL_ORDER, MIN_FLIGHTS, ROW_ORDER, assign_region  # noqa: E402
from schedule_features import waypoints  # noqa: E402

assert "--gaia2025" in sys.argv, "run with --gaia2025 (revision corpus)"
OUT = REVISION_PLOTS
# sequential ramp (the warm half of Spectral, monotone in lightness): the
# referee objected to a diverging palette for a quantity with no centre
SEQ = ["#ffffd9", "#fee08b", "#fdae61", "#f46d43", "#d53e4f", "#9e0142"]
CMAP = LinearSegmentedColormap.from_list("warm_seq", SEQ, N=256)
CMAP.set_bad("white")
VMAX_SD = 1.0            # GJ / passenger-km


def map_sd(df: pd.DataFrame):
    """Per-cell standard deviation of per-passenger-km forcing over the
    waypoints of sampled flights, smoothed like the Fig. 1b mean."""
    jpkm = df["Joulesperpasskm"].to_numpy(np.float64) / 1e9
    lon, lat = waypoints(df["OriginLon"].to_numpy(np.float64),
                         df["OriginLat"].to_numpy(np.float64),
                         df["DestinationLon"].to_numpy(np.float64),
                         df["DestinationLat"].to_numpy(np.float64), NB_WAYPOINTS)
    w = np.repeat(jpkm, NB_WAYPOINTS)
    lon_edges = np.linspace(-180, 180, LON_BINS + 1)
    lat_edges = np.linspace(-60, 80, LAT_BINS + 1)
    bins = [lon_edges, lat_edges]
    s1, _, _ = np.histogram2d(lon.ravel(), lat.ravel(), bins=bins, weights=w)
    s2, _, _ = np.histogram2d(lon.ravel(), lat.ravel(), bins=bins, weights=w * w)
    n, _, _ = np.histogram2d(lon.ravel(), lat.ravel(), bins=bins)
    s1, s2, n = (gaussian_filter(a, sigma=SMOOTH_SIGMA) for a in (s1, s2, n))
    ok = n > MIN_FLIGHTS_PER_CELL
    mean = np.divide(s1, n, out=np.full_like(s1, np.nan), where=ok)
    var = np.divide(s2, n, out=np.full_like(s1, np.nan), where=ok) - mean ** 2
    sd = np.sqrt(np.clip(var, 0, None))
    return mean, sd, lon_edges, lat_edges


def matrix_sd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["origin_region"] = assign_region(df["OriginLat"].to_numpy(), df["OriginLon"].to_numpy())
    df["dest_region"] = assign_region(df["DestinationLat"].to_numpy(), df["DestinationLon"].to_numpy())
    df["gj"] = df["Joulesperpasskm"] / 1e9
    known = df[(df["origin_region"] != "Other") & (df["dest_region"] != "Other")]
    g = known.groupby(["origin_region", "dest_region"])["gj"]
    t = g.agg(["mean", "std", "count"]).reset_index()
    t["sem"] = t["std"] / np.sqrt(t["count"])
    t["cv"] = t["std"] / t["mean"]
    return t


def main():
    t0 = time.time()
    df_all = load_gaia2025(None)
    print(f"loaded {len(df_all):,} flights ({time.time() - t0:.0f}s)", flush=True)
    sample = df_all.sample(SAMPLE_FLIGHTS, random_state=42).reset_index(drop=True)
    mean, sd, lon_e, lat_e = map_sd(sample)
    print(f"map done ({time.time() - t0:.0f}s)", flush=True)
    tab = matrix_sd(df_all)
    tab.to_csv(OUT / "figS_fig1_variability_matrix.csv", index=False)
    sd_mat = (tab.set_index(["origin_region", "dest_region"])["std"].unstack()
              .reindex(index=ROW_ORDER, columns=COL_ORDER))
    n_mat = (tab.set_index(["origin_region", "dest_region"])["count"].unstack()
             .reindex(index=ROW_ORDER, columns=COL_ORDER))
    sd_mat = sd_mat.where(n_mat >= MIN_FLIGHTS)
    print(f"matrix done ({time.time() - t0:.0f}s)", flush=True)

    with sjd.style():
        W = sjd.DOUBLE_COLUMN
        map_w, map_h = 5.6, 5.6 * 140 / 360
        mat = 2.6
        H = 0.3 + map_h + 0.65 + mat + 1.25
        fig = plt.figure(figsize=(W, H))
        ax_map = fig.add_axes((0.55 / W, (H - 0.3 - map_h) / H, map_w / W, map_h / H))
        ax_mat = fig.add_axes((1.45 / W, 1.25 / H, mat / W, mat / H))
        for ax in (ax_map, ax_mat):
            sjd.plain_ticks(ax)

        norm = colors.Normalize(vmin=0, vmax=VMAX_SD, clip=True)
        im = ax_map.imshow(np.ma.masked_invalid(sd.T),
                           extent=[lon_e[0], lon_e[-1], lat_e[0], lat_e[-1]],
                           origin="lower", cmap=CMAP, norm=norm,
                           interpolation="bicubic", aspect="auto", zorder=1)
        if COASTLINE_PATH.exists():
            gpd.read_file(COASTLINE_PATH).boundary.plot(ax=ax_map, color="#222222",
                                                        lw=0.25, zorder=10)
        ax_map.set_xlim(-180, 180)
        ax_map.set_ylim(-60, 80)
        ax_map.set_xticks([-180, -120, -60, 0, 60, 120, 180])
        ax_map.set_yticks([-60, -30, 0, 30, 60])
        ax_map.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}°"))
        ax_map.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}°"))
        ax_map.set_xlabel("Longitude")
        ax_map.set_ylabel("Latitude")
        ax_map.set_title("Spread of forcing among flights crossing each cell")
        sjd.letter(ax_map, "a", "bottom left")

        vals = sd_mat.to_numpy(float)
        n = len(ROW_ORDER)
        ax_mat.pcolormesh(np.ma.masked_invalid(vals[::-1]), cmap=CMAP, norm=norm,
                          edgecolors="white", linewidth=1.0)
        for r in range(n):
            for c in range(n):
                v = vals[r, c]
                if np.isnan(v):
                    continue
                ax_mat.text(c + 0.5, (n - 1 - r) + 0.5, f"{v:.2f}", ha="center",
                            va="center", fontsize=5.5,
                            color="white" if v / VMAX_SD > 0.75 else "#111111")
        ax_mat.set_xticks(np.arange(n) + 0.5)
        ax_mat.set_xticklabels(COL_ORDER, rotation=45, ha="right")
        ax_mat.set_yticks(np.arange(n) + 0.5)
        ax_mat.set_yticklabels(ROW_ORDER[::-1])
        ax_mat.tick_params(length=0)
        ax_mat.set_xlabel("Destination region")
        ax_mat.set_ylabel("Origin region")
        ax_mat.set_title("Spread of forcing by region pair")
        # an empty strip along the top and left keeps the letter off the cells
        ax_mat.set_xlim(-0.8, n)
        ax_mat.set_ylim(0, n + 0.8)
        sjd.letter(ax_mat, "b", "top left")
        sjd.colorbar(fig, im, ax_mat, "Standard deviation of contrail forcing (GJ/passenger-km)",
                     orientation="vertical", length=0.6, extend="max")
        for msg in sjd.check(fig):
            print("check:", msg)
        fig.savefig(OUT / "figS_fig1_variability.png", dpi=250)
        fig.savefig(OUT / "figS_fig1_variability.pdf")
    print(f"wrote figS_fig1_variability ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
