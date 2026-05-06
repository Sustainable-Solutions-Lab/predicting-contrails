import argparse
from glob import glob
from pathlib import Path

import geopandas as gpd
import matplotlib.pylab as plt
import numpy as np
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import convolve2d

from distance_limited_idw_infill import infill_near_data_idw


def main() -> None:
    parser = argparse.ArgumentParser(description="Reduce partial warming-map histograms and create final map outputs.")
    parser.add_argument("--partials-glob", default="./partials/part_*.npz")
    parser.add_argument("--tag", default="all")
    parser.add_argument("--world-shape", default="ne_110m_admin_0_countries.zip")
    parser.add_argument("--min-count", type=float, default=50)
    parser.add_argument(
        "--infill-n-cells",
        type=float,
        default=0.0,
        help="Distance-limited IDW: max grid-cell radius to fill NaNs (<=0 off). "
        "Sparse tracks widen a lot if this is large; try 1.0–1.5 if needed.",
    )
    parser.add_argument("--infill-k", type=int, default=4, help="IDW: nearest valid cells (smaller = less blur).")
    parser.add_argument("--infill-power", type=float, default=2.0, help="IDW inverse-distance exponent.")
    args = parser.parse_args()

    partial_paths = sorted(glob(args.partials_glob))
    if not partial_paths:
        raise FileNotFoundError(f"No partial files matched: {args.partials_glob}")

    sum_co2 = None
    count_flights = None
    xedges = None
    yedges = None

    for path in partial_paths:
        part = np.load(path)
        if sum_co2 is None:
            sum_co2 = part["sum_co2"].astype(np.float64)
            count_flights = part["count_flights"].astype(np.float64)
            xedges = part["xedges"]
            yedges = part["yedges"]
        else:
            sum_co2 += part["sum_co2"]
            count_flights += part["count_flights"]

    average_rf = np.divide(sum_co2, count_flights, where=count_flights > args.min_count)
    average_rf[count_flights <= args.min_count] = np.nan

    valid_mask = ~np.isnan(average_rf)
    kernel = np.ones((3, 3), dtype=float)
    filled = np.where(valid_mask, average_rf, 0.0)

    local_sum = convolve2d(filled, kernel, mode="same", boundary="symm")
    local_count = convolve2d(valid_mask.astype(float), kernel, mode="same", boundary="symm")
    average_rf_smoothed = np.divide(
        local_sum,
        local_count,
        out=np.full_like(local_sum, np.nan),
        where=local_count > 0,
    )
    average_rf_smoothed[count_flights <= args.min_count] = np.nan
    average_rf = average_rf_smoothed

    if args.infill_n_cells > 0:
        average_rf, _ = infill_near_data_idw(
            average_rf,
            n_cells_max=args.infill_n_cells,
            k=args.infill_k,
            power=args.infill_power,
        )

    gpd.read_file(args.world_shape).plot(color="white", edgecolor="black", lw=0.5)

    hex_colors = [
        "#f46d43",
        "#fdae61",
        "#fee08b",
        "#ffffbf",
        "#e6f598",
        "#abdda4",
        "#66c2a5",
        "#3288bd",
    ]
    custom_cmap = LinearSegmentedColormap.from_list("custom_diverging", hex_colors[::-1], N=256)
    thres_rf = np.nanpercentile(average_rf, 98)
    custom_cmap.set_over("#d53e4f")

    norm = colors.Normalize(vmin=0, vmax=thres_rf)
    im = plt.pcolormesh(
        xedges,
        yedges,
        average_rf.T,
        cmap=custom_cmap,
        norm=norm,
        shading="auto",
    )
    cbar = plt.colorbar(im, extend="max", shrink=0.7)
    cbar.set_label("Mean Radiative Forcing (J/passengerkm)", fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    plt.ylim(-60, 80)

    out_png = Path(f"{args.tag}warmingmap_Jan17.png")
    out_eps = Path(f"{args.tag}warmingmap_Jan17.eps")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_eps, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"wrote outputs: {out_png}, {out_eps}")


if __name__ == "__main__":
    main()
