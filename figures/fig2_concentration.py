"""
Fig 2 — Six-panel Lorenz-style concentration of contrail forcing.

Each panel plots cumulative net radiative forcing vs. cumulative flight
distance for a subset of flights, sorted descending by per-flight
forcing. The curve climbs steeply through high-warming flights, plateaus
through the near-neutral middle, then declines as negative (cooling)
flights pull the total down to the net.

Panels:
  a — All flights, with the curve segmented Warming / Neutral / Cooling
  b — By distance:   <2000 km vs ≥2000 km
  c — By time of day: entirely night / nighttime dep / daytime dep / entirely day
  d — By season:     winter / autumn / spring / summer
  e — By latitude:   high-lat (any travel >40°N) vs low-lat
  f — By origin region: North America / Europe / Russia / Asia

Each curve is annotated at its peak (the rank at which cumulative
forcing equals the eventual net total — i.e., the flights up to that
point account for 100% of the subset's net forcing).

Label placement rules (to avoid the collisions of earlier drafts):
  - series labels sit ABOVE each curve, ~2/3 of the way along the
    plateau (right side of the panel);
  - the peak %-annotation sits below-right of the peak marker, or
    above-right when the curve is too low for text to fit beneath it;
  - only the tallest curve in a panel carries the two-line
    "% of flights / 100% of net forcing" annotation; others get "%".

Inputs : pool cache (experiments/outputs/_pool_cache_all.parquet)
Output : figures/outputs/fig2_concentration.{png,pdf,eps}
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from feature_pruning import DROPBOX_PLOTS, OUT as EXP_OUT  # noqa: E402

OUT = DROPBOX_PLOTS

POOL_CACHE = EXP_OUT / "_pool_cache_all.parquet"
SAMPLE = None               # None = ALL flights, so the cumulative axes show
                            # true dataset totals (a 5M sample understated
                            # x and y by ~10.5x); ~87 billion km total
DIST_UNIT_SCALE = 1e9       # display distance in billions of km
FORC_UNIT_SCALE = 1e9       # display forcing in EJ (1 EJ = 1e9 GJ) — full-
                            # dataset totals in millions-of-GJ forced a
                            # floating "1e6" axis offset in half the panels
DIST_UNIT_LABEL = "billions of km"
FORC_UNIT_LABEL = "EJ"
RANDOM_SEED = 42

# AGWP-100 of contrail RF (Lee 2021), Earth area, year — recovers J from CO₂_km
AGWP_100 = 82.5e-15         # W m⁻² yr / kg
S_EARTH = 5.101e14          # m²
SECONDS_PER_YEAR = 365 * 24 * 3600
KG_TO_J = AGWP_100 * SECONDS_PER_YEAR * S_EARTH       # ≈ 1.327e9 J / kg-CO₂eq

# Aesthetic — colours chosen to match the source Lorenz figure
RED = "#DC3D3D"
ORANGE = "#F49441"
LIGHT_GREEN = "#A5CF80"
TEAL = "#54A89D"
GREY = "#9aa7b3"
COOL_BLUE = "#3D88DC"


# ──────────────────────────────────────────────────────────────────────────
# Derived metadata
# ──────────────────────────────────────────────────────────────────────────

def add_metadata(df: pd.DataFrame) -> pd.DataFrame:
    # Recover raw values from cyclic encodings
    df["OriginLon"] = np.degrees(
        np.arctan2(df["OriginLon_sin"].astype(np.float64),
                   df["OriginLon_cos"].astype(np.float64))
    )
    # day-of-year ∈ (0, 365]
    day_angle = np.arctan2(
        df["day_sin"].astype(np.float64),
        df["day_cos"].astype(np.float64),
    )
    day_angle = np.where(day_angle < 0, day_angle + 2 * np.pi, day_angle)
    df["doy"] = day_angle * 365 / (2 * np.pi)

    # Season — northern-hemisphere convention; cosmetic for the figure
    month = (df["doy"] / 30.5).clip(upper=12).astype(int) + 1
    month = np.clip(month, 1, 12)
    df["season"] = pd.Categorical(
        np.where(np.isin(month, [12, 1, 2]), "Winter",
        np.where(np.isin(month, [3, 4, 5]),  "Spring",
        np.where(np.isin(month, [6, 7, 8]),  "Summer", "Autumn"))),
        categories=["Winter", "Spring", "Summer", "Autumn"],
    )

    # Latitude class — max(|OriginLat|, |DestinationLat|) ≥ 40 → high-lat
    max_abs_lat = np.maximum(
        df["OriginLat"].abs(), df["DestinationLat"].abs(),
    )
    df["lat_class"] = pd.Categorical(
        np.where(max_abs_lat > 40, "high", "low"),
        categories=["high", "low"],
    )

    # Distance bin
    df["dist_class"] = pd.Categorical(
        np.where(df["total_flight_distance_km"] < 2000, "short", "long"),
        categories=["short", "long"],
    )

    # Night class — from night_score_BOOL_0, the plain fraction of
    # great-circle waypoints with the sun below the horizon. (An earlier
    # draft used night_score_FULL_0, which is sun-ELEVATION-weighted:
    # 100*(1-mean(sin(sun_alt))). On that score even an all-daylight
    # flight reads 10-30 unless the sun is near zenith throughout, so
    # "entirely daytime" collapsed to 0.13% of flights.)
    nb = df["night_score_bool_0"].astype(np.float64)

    # Sun up at departure? Vectorized NOAA-style solar position at the
    # origin airport at departure time (approximation, fine for binning).
    h_angle = np.arctan2(df["start_hour_sin"].astype(np.float64),
                         df["start_hour_cos"].astype(np.float64))
    h_angle = np.where(h_angle < 0, h_angle + 2 * np.pi, h_angle)
    utc_hour = h_angle * 24 / (2 * np.pi)
    decl = np.radians(-23.44) * np.cos(2 * np.pi * (df["doy"] + 10) / 365)
    lst = (utc_hour + df["OriginLon"] / 15.0) % 24
    hra = np.radians(15.0 * (lst - 12.0))
    lat = np.radians(df["OriginLat"].astype(np.float64))
    sin_alt = (np.sin(lat) * np.sin(decl)
               + np.cos(lat) * np.cos(decl) * np.cos(hra))
    sun_up_dep = sin_alt > 0

    night_class = np.where(
        nb >= 99.9, "entire_night",
        np.where(nb <= 0.1, "entire_day",
        np.where(sun_up_dep, "day_dep", "night_dep"))
    )
    df["night_class"] = pd.Categorical(
        night_class,
        categories=["entire_night", "night_dep", "day_dep", "entire_day"],
    )

    # Origin region — simple lat/lon box (4 named + "Other")
    olat, olon = df["OriginLat"].values, df["OriginLon"].values
    region = np.full(len(df), "Other", dtype=object)
    region = np.where(
        (olat >= 50) & (olon >= 30) & (olon <= 180), "Russia", region,
    )
    region = np.where(
        (region == "Other") & (olon >= -170) & (olon <= -50) &
        (olat >= 10) & (olat <= 75), "North America", region,
    )
    region = np.where(
        (region == "Other") & (olon >= -15) & (olon <= 45) &
        (olat >= 35) & (olat <= 60), "Europe", region,
    )
    region = np.where(
        (region == "Other") & (olon >= 60) & (olon <= 150) &
        (olat >= -10) & (olat <= 55), "Asia", region,
    )
    df["region"] = pd.Categorical(
        region,
        categories=["North America", "Europe", "Russia", "Asia", "Other"],
    )

    # Per-flight forcing in GJ (recover from contrail_CO2_km)
    forcing_kg = df["contrail_CO2_km"].astype(np.float64) * df["total_flight_distance_km"].astype(np.float64)
    df["forcing_GJ"] = forcing_kg * KG_TO_J / 1e9

    return df


def lorenz_curve(forcing_GJ: np.ndarray, distance_km: np.ndarray):
    """Sort flights descending by per-flight forcing and return the
    cumulative curve plus TWO ranks:

      peak_idx  — argmax of the cumulative curve (≈ all warming flights;
                  used only for label placement along the plateau);
      cross_idx — where the RISING curve first reaches the final net
                  total, i.e. the smallest set of worst flights whose
                  warming equals 100% of the subset's net forcing. This
                  is the number the "% of flights" annotation reports.
                  (An earlier draft wrongly annotated peak_idx — the
                  share of flights with ANY positive forcing — which
                  overstated the concentration figure, e.g. 19.8%
                  instead of 6.4% for all flights.)

    cross_idx is None when the subset's net forcing is <= 0 (e.g.
    entirely-daytime flights), where the metric is undefined.
    """
    order = np.argsort(forcing_GJ)[::-1]
    f_sorted = forcing_GJ[order]
    cum_forc = np.cumsum(f_sorted) / FORC_UNIT_SCALE
    cum_dist = np.cumsum(distance_km[order]) / DIST_UNIT_SCALE
    peak_idx = int(np.argmax(cum_forc))
    n_pos = int((f_sorted > 0).sum())
    net = cum_forc[-1]
    if net <= 0 or n_pos == 0:
        cross_idx = None
        cross_pct = None
    else:
        cross_idx = min(int(np.searchsorted(cum_forc[:n_pos], net, side="left")),
                        n_pos - 1)
        cross_pct = 100.0 * (cross_idx + 1) / len(forcing_GJ)
    return cum_forc, cum_dist, peak_idx, cross_idx, cross_pct


# ──────────────────────────────────────────────────────────────────────────
# Panel renderers
# ──────────────────────────────────────────────────────────────────────────

def annotate_pct(ax, x, y, text, *, ymax, fontsize=8.6):
    """Peak %-annotation: below-right of the marker when there is room,
    above-right when the curve sits too low for text beneath it."""
    if y > 0.20 * ymax:
        ax.annotate(text, xy=(x, y), xytext=(0.5, -0.4),
                    textcoords="offset fontsize",
                    fontsize=fontsize, color="#222", ha="left", va="top")
    else:
        ax.annotate(text, xy=(x, y), xytext=(0.5, 0.6),
                    textcoords="offset fontsize",
                    fontsize=fontsize, color="#222", ha="left", va="bottom")


def panel_curves(ax, df, group_col, group_specs, title, label_overrides=None,
                 annot_dyfrac=0.28):
    """group_specs is a list of (group_value, color, label, linestyle).

    label_overrides: optional {group_value: (xfrac, side)} placing that
    curve's series label at xfrac of its own x-extent, above or below
    the curve — for panels whose plateaus are too close for the default
    above-plateau placement (e.g. panel c after the day/night reclass).
    """
    label_overrides = label_overrides or {}
    curves = []
    for value, color, label, ls in group_specs:
        sub = df[df[group_col] == value]
        if len(sub) == 0:
            continue
        cf, cd, peak, cross, pct = lorenz_curve(
            sub["forcing_GJ"].to_numpy(),
            sub["total_flight_distance_km"].to_numpy(),
        )
        curves.append(dict(cf=cf, cd=cd, peak=peak, cross=cross, pct=pct,
                           color=color, label=label, ls=ls, value=value))

    ymax = max(c["cf"].max() for c in curves)
    xmax = max(c["cd"][-1] for c in curves)
    ax.set_xlim(0, xmax * 1.03)
    ax.set_ylim(0, ymax * 1.18)
    tallest = max(range(len(curves)), key=lambda i: curves[i]["cf"].max())

    for i, c in enumerate(curves):
        cf, cd, peak = c["cf"], c["cd"], c["peak"]
        ax.plot(cd, cf, color=c["color"], lw=2.2, ls=c["ls"],
                label=c["label"], solid_capstyle="round")
        # series label above the plateau, ~2/3 of the way to the end;
        # curves hugging the x-axis get extra lift so the label clears
        # the peak %-annotation (which flips above the marker for them).
        # Anchor at the PLATEAU START (first rank reaching 98.5% of max)
        # rather than argmax — float64 cumsum stalls make argmax land
        # early on some curves, dragging labels into the annotations.
        ps = int(np.searchsorted(cf[:peak + 1], 0.985 * cf.max()))
        if c["value"] in label_overrides:
            xfrac, side = label_overrides[c["value"]]
            lx = xfrac * cd[-1]
        else:
            xfrac, side = None, "above"
            lx = cd[ps] + 0.75 * (cd[-1] - cd[ps])
            lx = min(max(lx, 0.20 * xmax), 0.92 * xmax)
        ly = cf[min(int(np.searchsorted(cd, lx)), len(cf) - 1)]
        lift = 0.055 * ymax if cf[peak] < 0.15 * ymax else 0.03 * ymax
        if side == "below":
            ax.text(lx, max(ly, 0) - lift, c["label"], color=c["color"],
                    fontsize=10.5, weight="bold", ha="center", va="top")
        else:
            ax.text(lx, max(ly, 0) + lift, c["label"], color=c["color"],
                    fontsize=10.5, weight="bold", ha="center", va="bottom")
        # Marker at the CROSSING: the rank where the rising curve first
        # reaches the subset's final net forcing. Skipped when net <= 0
        # (metric undefined, e.g. entirely-daytime flights).
        if c["cross"] is not None:
            cross = c["cross"]
            ax.scatter([cd[cross]], [cf[cross]], facecolor="white",
                       edgecolor=c["color"], s=22, linewidth=1.1, zorder=10)
            if i == tallest:
                # Text sits down-right of the marker with a thin leader
                # line — in the pocket between the tallest curve's
                # rising limb and the runner-up's plateau, the one
                # region that is empty in every panel. (Above-left
                # spills over the y-axis; below-right collides with the
                # runner-up's series label.)
                ax.annotate(f"{c['pct']:.1f}% of flights\n"
                            "100% of net forcing",
                            xy=(cd[cross], cf[cross]),
                            xytext=(cd[cross] + 0.22 * xmax,
                                    cf[cross] - annot_dyfrac * ymax),
                            fontsize=8.6, color="#222",
                            ha="left", va="top",
                            arrowprops=dict(arrowstyle="-", lw=0.7,
                                            color="#999999",
                                            shrinkA=2, shrinkB=3))
            else:
                annotate_pct(ax, cd[cross], cf[cross], f"{c['pct']:.1f}%",
                             ymax=ymax)

    ax.set_title(title, fontsize=12, weight="bold")


def panel_a_all_flights(ax, df):
    """Single curve segmented by warming/neutral/cooling."""
    f_gj = df["forcing_GJ"].to_numpy()
    d_km = df["total_flight_distance_km"].to_numpy()
    order = np.argsort(f_gj)[::-1]
    f_sorted = f_gj[order]
    d_sorted = d_km[order]
    cum_forc = np.cumsum(f_sorted) / FORC_UNIT_SCALE
    cum_dist = np.cumsum(d_sorted) / DIST_UNIT_SCALE
    peak_idx = int(np.argmax(cum_forc))
    # Crossing: smallest set of worst flights whose warming equals the
    # final net total (see lorenz_curve docstring)
    n_pos = int((f_sorted > 0).sum())
    cross_idx = min(int(np.searchsorted(cum_forc[:n_pos], cum_forc[-1],
                                        side="left")), n_pos - 1)
    cross_pct = 100.0 * (cross_idx + 1) / len(f_gj)

    # Split into warming / neutral / cooling segments by sign of per-flight forcing
    threshold = max(1e-6, np.abs(f_sorted).max() * 1e-5)
    neutral_start = int(np.searchsorted(-f_sorted, -threshold))
    cooling_start = len(f_sorted) - int(np.searchsorted(np.sort(f_sorted), -threshold))
    cooling_start = max(cooling_start, neutral_start)

    ymax = cum_forc.max()
    xmax = cum_dist[-1]
    ax.set_xlim(0, xmax * 1.03)
    ax.set_ylim(0, ymax * 1.18)

    ax.plot(cum_dist[:neutral_start], cum_forc[:neutral_start],
            color=RED, lw=2.6, solid_capstyle="round", zorder=4)
    ax.plot(cum_dist[neutral_start:cooling_start],
            cum_forc[neutral_start:cooling_start],
            color=GREY, lw=2.6, solid_capstyle="round", zorder=3)
    ax.plot(cum_dist[cooling_start:], cum_forc[cooling_start:],
            color=COOL_BLUE, lw=2.6, solid_capstyle="round", zorder=4)

    # Dashed horizontal line at net forcing (final value)
    net = cum_forc[-1]
    ax.axhline(net, color="#444", ls="--", lw=0.8, zorder=2)
    ax.annotate("100% of net forcing", xy=(0.40 * xmax, net),
                fontsize=9, color="#222", ha="center", va="top",
                xytext=(0, -0.4), textcoords="offset fontsize")

    # Segment labels above the curve
    ax.text(cum_dist[peak_idx], ymax * 1.04, "Warming", color=RED,
            fontsize=11, weight="bold", ha="center", va="bottom")
    mid = (neutral_start + cooling_start) // 2
    ax.text(cum_dist[mid], cum_forc[mid] + ymax * 0.04, "Neutral",
            color="#6b7785", fontsize=11, weight="bold",
            ha="center", va="bottom")
    ax.text(xmax, cum_forc[cooling_start] + ymax * 0.04, "Cooling",
            color=COOL_BLUE, fontsize=11, weight="bold",
            ha="right", va="bottom")

    # Marker where the rising curve first reaches the net total (dashed
    # line) — these worst flights alone account for 100% of net forcing.
    ax.scatter([cum_dist[cross_idx]], [cum_forc[cross_idx]],
               facecolor="white", edgecolor=RED, s=24, linewidth=1.2,
               zorder=10)
    ax.annotate(f"{cross_pct:.1f}% of flights",
                xy=(cum_dist[cross_idx], cum_forc[cross_idx]),
                xytext=(0.6, 0.3), textcoords="offset fontsize",
                fontsize=9, color="#222", ha="left", va="bottom")

    ax.set_title("All flights", fontsize=12, weight="bold")


# ──────────────────────────────────────────────────────────────────────────
# Build
# ──────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print(f"Loading pool cache {POOL_CACHE.name} ...")
    cols = [
        "OriginLat", "DestinationLat",
        "OriginLon_sin", "OriginLon_cos",
        "day_sin", "day_cos",
        "night_score_bool_0",
        "start_hour_sin", "start_hour_cos",
        "total_flight_distance_km", "contrail_CO2_km",
        "year",
    ]
    df = pd.read_parquet(POOL_CACHE, columns=cols)
    print(f"  {len(df):,} rows, {time.time()-t0:.1f}s")

    if SAMPLE and len(df) > SAMPLE:
        df = df.sample(SAMPLE, random_state=RANDOM_SEED).reset_index(drop=True)
        print(f"  sampled to {len(df):,} flights")

    print("Computing derived metadata ...")
    df = add_metadata(df)
    df = df.dropna(subset=["forcing_GJ", "total_flight_distance_km"])
    print(f"  ready in {time.time()-t0:.0f}s")

    # ── 3 rows × 2 cols ───────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(11, 13))

    panel_a_all_flights(axes[0, 0], df)

    panel_curves(axes[0, 1], df, "dist_class", [
        ("short", RED, "<2000 km", "-"),
        ("long",  RED, "≥2000 km", "--"),
    ], "By distance")

    panel_curves(axes[1, 0], df, "night_class", [
        ("night_dep",    RED,         "Nighttime departure", "-"),
        ("entire_night", ORANGE,      "Entirely nighttime",  "-"),
        ("day_dep",      LIGHT_GREEN, "Daytime departure",   "-"),
        ("entire_day",   TEAL,        "Entirely daytime",    "-"),
    ], "By time of day", label_overrides={
        # Plateaus sit too close for the default above-plateau spots;
        # xfrac > 1 floats the label right of that curve's end, in the
        # empty zone left by the shorter curves.
        "entire_night": (0.65, "above"),
        "day_dep":      (1.45, "below"),
        "night_dep":    (1.30, "below"),
        "entire_day":   (0.62, "above"),
    }, annot_dyfrac=0.20)

    panel_curves(axes[1, 1], df, "season", [
        ("Winter", RED,         "Winter", "-"),
        ("Autumn", ORANGE,      "Autumn", "-"),
        ("Spring", LIGHT_GREEN, "Spring", "-"),
        ("Summer", TEAL,        "Summer", "-"),
    ], "By season")

    panel_curves(axes[2, 0], df, "lat_class", [
        ("high", RED,    "High-latitude (some travel >40°)", "-"),
        ("low",  ORANGE, "Low-latitude (no travel >40°)",    "-"),
    ], "By latitude")

    panel_curves(axes[2, 1], df, "region", [
        ("North America", RED,         "North America", "-"),
        ("Europe",        ORANGE,      "Europe",        "-"),
        ("Russia",        LIGHT_GREEN, "Russia",        "-"),
        ("Asia",          TEAL,        "Asia",          "-"),
    ], "By region of origin")

    # Common axis labels and panel letters
    for r, axrow in enumerate(axes):
        for c, ax in enumerate(axrow):
            ax.set_xlabel(f"Cumulative flight distance\n({DIST_UNIT_LABEL})",
                          fontsize=10)
            ax.set_ylabel(f"Cumulative radiative forcing\n({FORC_UNIT_LABEL})",
                          fontsize=10)
            ax.tick_params(labelsize=9)
            ax.grid(False)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_linewidth(0.8)
            ax.text(0.97, 0.05, chr(ord("a") + r * 2 + c),
                    transform=ax.transAxes, fontsize=14, weight="bold",
                    ha="right", va="bottom")

    plt.tight_layout()
    for ext in ("png", "pdf", "eps"):
        fig.savefig(OUT / f"fig2_concentration.{ext}", dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote fig2_concentration.{{png,pdf,eps}} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
