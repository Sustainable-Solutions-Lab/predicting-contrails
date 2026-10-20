#!/usr/bin/env python3
"""Old (2019/2021 corpus) vs new (GAIA 2019–25, vPM-enabled) contrail
labels, flight by flight.

Joins the old feature shards to the new corpus (gaia2025_corpus.py) on
flight_id and reports, per year: coverage, net / warming / cooling totals,
the share of flights forming persistent contrails, rank agreement, the
Fig. 2 concentration statistics, and the change by engine combustor
(lean-burn LEAP/GEnx vs the rest) and by aircraft type.

Usage: python experiments/gaia2025_label_compare.py
Writes Plots/gaia2025_label_compare*.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DROPBOX_PARQUETS, DROPBOX_PLOTS  # noqa: E402

CORPUS = DROPBOX_PARQUETS.parent / "GAIA2019-25/corpus"
OLD_COLS = ["flight_id", "aircraft_type_icao", "engine_name",
            "total_flight_distance_km", "total_contrail_energy_forcing"]
LEAN = ("LEAP", "GENX")


def load(year: int) -> pd.DataFrame:
    old = pd.concat([pd.read_parquet(f, columns=OLD_COLS) for f in
                     sorted(DROPBOX_PARQUETS.glob(f"features_{year}*_gdf.pq"))])
    old = old[(old["total_flight_distance_km"] >= 0.1)
              & np.isfinite(old["total_contrail_energy_forcing"])]
    new = pd.read_parquet(CORPUS / f"{year}.parquet",
                          columns=["flight_id", "ef_sum", "engine_name"])
    new["ef_new"] = new["ef_sum"].fillna(0).astype(float)
    m = old.merge(new[["flight_id", "ef_new"]], on="flight_id")
    m = m.rename(columns={"total_contrail_energy_forcing": "ef_old",
                          "total_flight_distance_km": "km"})
    m.attrs.update(n_old=len(old), n_new=len(new))
    return m


def concentration(ef: np.ndarray, km: np.ndarray) -> dict:
    """Fig. 2 statistics: sort flights by forcing, worst first; the peak
    is where cumulative forcing first reaches the net total."""
    o = np.argsort(ef)[::-1]
    c = np.cumsum(ef[o])
    net = ef.sum()
    peak = np.argmax(c >= net) + 1
    n = len(ef)
    out = {"peak_share_flights": peak / n,
           "peak_share_km": km[o][:peak].sum() / km.sum()}
    for k in (0.01, 0.05, 0.10):
        out[f"top{int(k * 100)}pct_share_of_net"] = c[int(n * k) - 1] / net
    return out


def summarise(m: pd.DataFrame, year: int) -> dict:
    eo, en = m["ef_old"].to_numpy(), m["ef_new"].to_numpy()
    both = (eo != 0) & (en != 0)
    r = {"year": year, "n_old": m.attrs["n_old"], "n_new": m.attrs["n_new"],
         "n_matched": len(m),
         "net_old_EJ": eo.sum() / 1e18, "net_new_EJ": en.sum() / 1e18,
         "warm_old_EJ": eo[eo > 0].sum() / 1e18,
         "warm_new_EJ": en[en > 0].sum() / 1e18,
         "cool_old_EJ": eo[eo < 0].sum() / 1e18,
         "cool_new_EJ": en[en < 0].sum() / 1e18,
         "formed_old": (eo != 0).mean(), "formed_new": (en != 0).mean(),
         "spearman_all": spearmanr(eo, en).statistic,
         "spearman_both_formed": spearmanr(eo[both], en[both]).statistic,
         "sign_agree_both_formed": (np.sign(eo[both]) == np.sign(en[both])).mean()}
    km = m["km"].to_numpy()
    for tag, ef in (("old", eo), ("new", en)):
        r.update({f"{k}_{tag}": v for k, v in concentration(ef, km).items()})
    return r


def by_group(m: pd.DataFrame, key: pd.Series, year: int) -> pd.DataFrame:
    g = m.assign(key=key).groupby("key").agg(
        flights=("ef_old", "size"), km=("km", "sum"),
        ef_old=("ef_old", "sum"), ef_new=("ef_new", "sum"),
        formed_old=("ef_old", lambda s: (s != 0).mean()),
        formed_new=("ef_new", lambda s: (s != 0).mean()))
    g["gj_km_old"] = g["ef_old"] / g["km"] / 1e9
    g["gj_km_new"] = g["ef_new"] / g["km"] / 1e9
    g["ratio_new_old"] = g["ef_new"] / g["ef_old"]
    return g.assign(year=year).reset_index()


def main() -> None:
    rows, comb, types = [], [], []
    for y in (2019, 2021):
        m = load(y)
        rows.append(summarise(m, y))
        lean = m["engine_name"].str.upper().str.contains("|".join(LEAN), na=False)
        comb.append(by_group(m, lean.map({True: "lean-burn (LEAP, GEnx)",
                                          False: "other"}), y))
        top = m["aircraft_type_icao"].value_counts().index[:25]
        t = by_group(m[m["aircraft_type_icao"].isin(top)],
                     m["aircraft_type_icao"], y)
        types.append(t)
    s = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(s.T.to_string())
    c = pd.concat(comb)
    print(c.round(3).to_string(index=False))
    t = pd.concat(types).sort_values(["year", "flights"], ascending=[True, False])
    print(t.round(3).to_string(index=False))
    s.to_csv(DROPBOX_PLOTS / "gaia2025_label_compare.csv", index=False)
    c.to_csv(DROPBOX_PLOTS / "gaia2025_label_compare_combustor.csv", index=False)
    t.to_csv(DROPBOX_PLOTS / "gaia2025_label_compare_types.csv", index=False)


if __name__ == "__main__":
    main()
