#!/usr/bin/env python3
"""Climatological baselines and a booking-situation evaluation (revision).

Both were asked for by a referee: how does the scheduling model compare
with a climatology of the kind used by Google's Travel Impact Model
(TIM), and can a traveller choosing among the itineraries actually on
offer for a route and day pick the less-warming one?

Rankers (all schedule-only except perfect foresight):
  model       the XGBoost scheduling model
  tim         TIM-style climatology (Platt et al. 2024; TIM technical
              briefs Jan and Nov 2025): mean forcing per km of persistent
              contrail in cells of 10 deg latitude x 3 h local time x
              season, multiplied by the probability of persistent
              contrail formation on a 1-degree map (persistent-contrail
              km / flight km), integrated along 30 great-circle waypoints
              at scheduled times
  clim        the same latitude x local-time x season climatology of
              forcing per flight km, without the formation map (TIM's
              January 2025 design, which used one global probability)
  route       mean forcing per flight km by route x month x UTC departure
              hour (fallbacks: route x month, route, global)
  insolation  route-mean insolation alone (night_score_full_0)
  perfect     the simulated forcing itself

Two settings:
  --setting ood     baselines fitted on 2019-23, model = the 2019-23 model
                    (gaia2025_ood_2019-23.joblib), everything judged on all
                    2025 flights (out of time). The booking-set evaluation
                    runs here, because every 2025 itinerary is available.
  --setting pooled  baselines fitted on the training two thirds of each
                    year (gaia2025_retrain.split), judged on the test third
                    with the pooled model, so a TIM-style curve can be drawn
                    on Fig. 3a. Writes outputs/_clim_pred_test.parquet in
                    the test-set order used by final_model.replot_gaia2025.

Booking sets (ood only): flights with the same origin, destination and
UTC departure date ("day"), or ISO week ("week"), with at least two
flights. Each ranker picks the flight it predicts to be least warming;
we record the actual forcing of that pick against the set's best, mean
and worst, and whether the predicted worst is the actual worst.

Usage:
  python experiments/booking_eval.py --setting ood [--fit-sample 5000000]
  python experiments/booking_eval.py --setting pooled
Outputs (experiments/outputs and Plots/Revision (GAIA)):
  booking_capture_<setting>.csv   worst/best k% capture per ranker
  booking_sets_2025.csv           choice-set metrics (ood)
  booking_sets_by_size_2025.csv   ... by set size and haul length (ood)
  booking_curves_<setting>.csv    cumulative-forcing curves per ranker
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CORPUS, MODEL_FEATS, OUT, RANDOM_SEED, REVISION_PLOTS,  # noqa: E402
                    load_corpus)
from gaia2025_retrain import YEARS, split  # noqa: E402
from schedule_features import waypoints  # noqa: E402

CAT = "aircraft_type_icao"
NB = 30                                 # waypoints, as night_score_full_0
LAT_EDGES = np.arange(-90, 91, 10)      # 18 bands
LT_EDGES = np.arange(0, 25, 3)          # 8 local-time bins
SEASON = np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0])   # month-1 -> DJF..SON
CRUISE_KMH = 800.0                      # for waypoint times (corpus has no arrival)
TAXI_H = 0.4
KS = (0.05, 0.10)

BASE_COLS = ["origin_airport", "destination_airport", "first_waypoint_time",
             "OriginLat", "OriginLon", "DestinationLat", "DestinationLon",
             "contrail_CO2_km", "total_flight_distance_km",
             "total_persistent_contrail_length_km", "night_score_full_0", "year"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── waypoint cells ────────────────────────────────────────────────────────
def cells(d: pd.DataFrame, chunk: int = 400_000):
    """Yield (slice, cell index (n, NB) for the lat x local-time x season
    grid, map index (n, NB) for the 1-degree lon-lat map)."""
    t0 = d["first_waypoint_time"].to_numpy("datetime64[s]").astype(np.int64)
    month = d["first_waypoint_time"].dt.month.to_numpy() - 1
    dist = d["total_flight_distance_km"].to_numpy(np.float64)
    olon, olat = d["OriginLon"].to_numpy(np.float64), d["OriginLat"].to_numpy(np.float64)
    dlon, dlat = d["DestinationLon"].to_numpy(np.float64), d["DestinationLat"].to_numpy(np.float64)
    frac = np.arange(1, NB + 1) / (NB + 1)
    for s in range(0, len(d), chunk):
        sl = slice(s, min(s + chunk, len(d)))
        lon, lat = waypoints(olon[sl], olat[sl], dlon[sl], dlat[sl], NB)
        dur_s = (TAXI_H + dist[sl] / CRUISE_KMH) * 3600
        t = t0[sl, None] + dur_s[:, None] * frac               # UTC seconds
        local_h = ((t / 3600.0) + lon / 15.0) % 24
        ilat = np.clip(np.searchsorted(LAT_EDGES, lat, side="right") - 1, 0, 17)
        ilt = np.clip((local_h // 3).astype(int), 0, 7)
        isea = SEASON[month[sl]][:, None]
        cell = (ilat * 8 + ilt) * 4 + isea                       # 18*8*4 = 576
        mlon = np.clip(np.floor(lon + 180).astype(int), 0, 359)
        mlat = np.clip(np.floor(lat + 90).astype(int), 0, 179)
        yield sl, cell, mlon * 180 + mlat


class Climatology:
    """TIM-style and plain latitude x local-time x season climatologies."""

    def __init__(self):
        self.ef_sum = np.zeros(576)       # forcing (kg CO2e) per cell
        self.pers_km = np.zeros(576)      # persistent-contrail km per cell
        self.fl_km = np.zeros(576)        # flight km per cell
        self.map_pers = np.zeros(360 * 180)
        self.map_km = np.zeros(360 * 180)

    def fit(self, d: pd.DataFrame) -> None:
        dist = d["total_flight_distance_km"].to_numpy(np.float64)
        ef = d["contrail_CO2_km"].to_numpy(np.float64) * dist
        pers = d["total_persistent_contrail_length_km"].fillna(0).to_numpy(np.float64)
        for sl, cell, mp in cells(d):
            w = dist[sl] / NB
            np.add.at(self.ef_sum, cell.ravel(), np.repeat(ef[sl] / NB, NB))
            np.add.at(self.pers_km, cell.ravel(), np.repeat(pers[sl] / NB, NB))
            np.add.at(self.fl_km, cell.ravel(), np.repeat(w, NB))
            np.add.at(self.map_pers, mp.ravel(), np.repeat(pers[sl] / NB, NB))
            np.add.at(self.map_km, mp.ravel(), np.repeat(w, NB))

    def predict(self, d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(tim, clim): predicted forcing per flight km."""
        with np.errstate(invalid="ignore", divide="ignore"):
            efpcm = np.where(self.pers_km > 0, self.ef_sum / self.pers_km, 0.0)
            clim = np.where(self.fl_km > 0, self.ef_sum / self.fl_km, 0.0)
            pmap = np.where(self.map_km > 0, self.map_pers / self.map_km, np.nan)
        # unobserved map cells: global mean formation probability
        pmap = np.where(np.isnan(pmap), self.map_pers.sum() / self.map_km.sum(), pmap)
        tim = np.empty(len(d))
        cl = np.empty(len(d))
        for sl, cell, mp in cells(d):
            tim[sl] = (efpcm[cell] * pmap[mp]).mean(1)
            cl[sl] = clim[cell].mean(1)
        return tim, cl


class RouteClimatology:
    """Mean forcing per km by route x month x UTC hour, with fallbacks."""

    def __init__(self):
        self.sums: dict[str, pd.DataFrame] = {}

    @staticmethod
    def keys(d: pd.DataFrame) -> pd.DataFrame:
        t = d["first_waypoint_time"]
        return pd.DataFrame({
            "route": (d["origin_airport"].astype(str) + ">"
                      + d["destination_airport"].astype(str)).to_numpy(),
            "month": t.dt.month.to_numpy(np.int8),
            "hour": t.dt.hour.to_numpy(np.int8),
            "y": d["contrail_CO2_km"].to_numpy(np.float64)})

    def fit(self, d: pd.DataFrame) -> None:
        k = self.keys(d)
        for name, by in [("rmh", ["route", "month", "hour"]),
                         ("rm", ["route", "month"]), ("r", ["route"])]:
            g = k.groupby(by, observed=True)["y"].agg(["sum", "count"])
            self.sums[name] = (g if name not in self.sums
                               else self.sums[name].add(g, fill_value=0))
        self.sums["g"] = (self.sums.get("g", pd.DataFrame({"sum": [0.0], "count": [0.0]}))
                          + pd.DataFrame({"sum": [k["y"].sum()], "count": [float(len(k))]}))

    def predict(self, d: pd.DataFrame) -> np.ndarray:
        k = self.keys(d)
        out = np.full(len(k), np.nan)
        for name, by in [("rmh", ["route", "month", "hour"]),
                         ("rm", ["route", "month"]), ("r", ["route"])]:
            g = self.sums[name]
            mean = (g["sum"] / g["count"]).rename("m").reset_index()
            m = k[by].merge(mean, on=by, how="left")["m"].to_numpy()
            out = np.where(np.isnan(out), m, out)
        gmean = float(self.sums["g"]["sum"].iloc[0] / self.sums["g"]["count"].iloc[0])
        return np.where(np.isnan(out), gmean, out)


# ── data ──────────────────────────────────────────────────────────────────
def load_year(year: int, cats: list[str], model_feats: bool = True) -> pd.DataFrame:
    cols = list(dict.fromkeys(BASE_COLS + (MODEL_FEATS if model_feats else [])))
    d = load_corpus([year], cols)
    if CAT in d:
        d[CAT] = pd.Categorical(d[CAT], categories=cats)
    d["first_waypoint_time"] = pd.to_datetime(d["first_waypoint_time"]).dt.tz_localize(None)
    return d


# ── metrics ───────────────────────────────────────────────────────────────
def shares(score: np.ndarray, w: np.ndarray) -> dict:
    order = np.argsort(score, kind="stable")
    net = w.sum()
    out = {}
    for k in KS:
        n = int(round(k * len(w)))
        out[f"worst{int(k * 100)}"] = 100 * w[order[::-1][:n]].sum() / net
        out[f"best{int(k * 100)}"] = 100 * w[order[:n]].sum() / net
    return out


def curves(scores: dict[str, np.ndarray], w: np.ndarray, n_pts: int = 1500) -> pd.DataFrame:
    """Cumulative share of net forcing, worst-predicted first, per ranker."""
    net = w.sum()
    n = len(w)
    idx = np.linspace(0, n - 1, n_pts).astype(int)
    out = {"share_of_flights_pct": 100 * (idx + 1) / n}
    for name, s in scores.items():
        order = np.argsort(s, kind="stable")[::-1]
        out[name] = (100 * np.cumsum(w[order]) / net)[idx]
    return pd.DataFrame(out)


def rank_corr(a: np.ndarray, b: np.ndarray, n: int = 2_000_000) -> float:
    from scipy.stats import spearmanr
    if len(a) > n:
        i = np.random.default_rng(RANDOM_SEED).choice(len(a), n, replace=False)
        a, b = a[i], b[i]
    return float(spearmanr(a, b).correlation)


def choice_sets(d: pd.DataFrame, scores: dict[str, np.ndarray], key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Within each (origin, destination, day|week) set of >=2 flights, each
    ranker's pick (its lowest predicted forcing) and worst, against the
    set's actual best, mean and worst. Returns (summary rows, per-set table)."""
    t = d["first_waypoint_time"]
    period = (t.dt.normalize() if key == "day"
              else t.dt.isocalendar().week.astype(int).to_numpy()
              + 100 * t.dt.isocalendar().year.astype(int).to_numpy())
    grp = pd.DataFrame({"o": d["origin_airport"].to_numpy(),
                        "d": d["destination_airport"].to_numpy(),
                        "p": period})
    gid = grp.groupby(["o", "d", "p"], sort=False).ngroup().to_numpy()
    actual = d["contrail_CO2_km"].to_numpy(np.float64)        # same distance within a set
    dist = d["total_flight_distance_km"].to_numpy(np.float64)
    size = np.bincount(gid)
    keep = size[gid] >= 2
    gid, actual, dist = gid[keep], actual[keep], dist[keep]
    gid = pd.factorize(gid)[0]
    n_sets = gid.max() + 1
    size = np.bincount(gid)
    s_sum = np.bincount(gid, weights=actual)
    s_mean = s_sum / size
    order = np.lexsort((actual, gid))
    first = np.r_[0, np.cumsum(size)[:-1]]
    s_min = actual[order[first]]
    s_max = actual[order[first + size - 1]]
    s_dist = dist[order[first]]
    spread = s_max > s_min
    per_set = pd.DataFrame({"n": size, "dist_km": s_dist, "mean": s_mean,
                            "best": s_min, "worst": s_max})
    rows = []
    for name, sc in scores.items():
        sc = sc[keep]
        o = np.lexsort((sc, gid))                      # ascending predicted
        pick = actual[o[first]]                        # predicted least warming
        pworst = actual[o[first + size - 1]]           # predicted most warming
        per_set[f"pick_{name}"] = pick
        per_set[f"pworst_{name}"] = pworst
        rows.append(dict(
            key=key, ranker=name, n_sets=n_sets, n_flights=int(size.sum()),
            # forcing-weighted: kg CO2e over all sets (per-km x km)
            reduction_vs_mean_pct=100 * ((s_mean - pick) * s_dist).sum()
            / ((s_mean - s_min) * s_dist).sum(),
            pick_forcing_over_mean_pct=100 * (pick * s_dist).sum() / (s_mean * s_dist).sum(),
            pworst_forcing_over_mean_pct=100 * (pworst * s_dist).sum() / (s_mean * s_dist).sum(),
            # tie-aware: many flights have zero forcing, so "best" is often
            # shared; "strict" restricts to sets whose flights differ
            p_pick_is_best=float(np.mean(pick <= s_min)),
            p_pworst_is_worst=float(np.mean(pworst >= s_max)),
            p_pick_below_mean=float(np.mean(pick <= s_mean)),
            p_pick_is_best_strict=float(np.mean(pick[spread] <= s_min[spread])),
            p_pworst_is_worst_strict=float(np.mean(pworst[spread] >= s_max[spread])),
            n_sets_strict=int(spread.sum()),
            chance_p_best=float(np.mean(1 / size)),
        ))
    return pd.DataFrame(rows), per_set


def by_size_and_haul(per_set: pd.DataFrame, names: list[str], key: str) -> pd.DataFrame:
    size_bin = pd.cut(per_set["n"], [1, 2, 5, 10, np.inf],
                      labels=["2", "3-5", "6-10", ">10"])
    haul = pd.cut(per_set["dist_km"], [0, 1500, 4000, np.inf],
                  labels=["<1500 km", "1500-4000 km", ">4000 km"])
    rows = []
    for dim, lab in [("size", size_bin), ("haul", haul)]:
        for level, g in per_set.groupby(lab, observed=True):
            w_mean = (g["mean"] * g["dist_km"]).sum()
            w_pot = ((g["mean"] - g["best"]) * g["dist_km"]).sum()
            for name in names:
                pick = g[f"pick_{name}"]
                rows.append(dict(
                    key=key, dim=dim, level=str(level), ranker=name,
                    n_sets=len(g), mean_set_size=g["n"].mean(),
                    reduction_vs_mean_pct=100 * ((g["mean"] - pick) * g["dist_km"]).sum() / w_pot,
                    pick_forcing_over_mean_pct=100 * (pick * g["dist_km"]).sum() / w_mean,
                    p_pick_is_best=float((pick <= g["best"]).mean()),
                    chance_p_best=float((1 / g["n"]).mean())))
    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────
def fit_baselines(train_years: list[int], cats: list[str], fit_sample: int | None,
                  setting: str) -> tuple[Climatology, RouteClimatology]:
    clim, route = Climatology(), RouteClimatology()
    for y in train_years:
        t = time.time()
        d = load_year(y, cats, model_feats=False)
        if setting == "pooled":
            d = split(d, RANDOM_SEED + y)[0]            # training two thirds
        route.fit(d)
        if fit_sample and len(d) > fit_sample:
            d = d.sample(fit_sample, random_state=RANDOM_SEED + y)
        clim.fit(d)
        log(f"fitted baselines on {y}: {len(d):,} flights for the TIM-style "
            f"grid ({time.time() - t:.0f}s)")
        del d
    return clim, route


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["ood", "pooled"], default="ood")
    ap.add_argument("--fit-sample", type=int, default=5_000_000,
                    help="flights per year for the TIM-style grid (route "
                         "climatology always uses every flight)")
    ap.add_argument("--test-sample", type=int, default=None,
                    help="debug: evaluate on a random subset")
    ap.add_argument("--model", default=None, help="override model bundle path")
    ap.add_argument("--fit-years", type=int, nargs="+", default=None,
                    help="debug: fit the baselines on these years only")
    a = ap.parse_args()

    if a.setting == "ood":
        bundle = joblib.load(a.model or OUT / "gaia2025_ood_2019-23.joblib")
        train_years, test_years = [y for y in YEARS if y <= 2023], [2025]
    else:
        bundle = joblib.load(a.model or OUT / "gaia2025_pooled.joblib")
        train_years, test_years = YEARS, YEARS
    cats = bundle["categories"]
    assert bundle["features"] == MODEL_FEATS
    if a.fit_years:
        train_years = a.fit_years

    clim, route = fit_baselines(train_years, cats, a.fit_sample, a.setting)

    parts = []
    for y in test_years:
        d = load_year(y, cats)
        if a.setting == "pooled":
            d = split(d, RANDOM_SEED + y)[1]
        parts.append(d)
    te = pd.concat(parts, ignore_index=True)
    del parts
    if a.test_sample and len(te) > a.test_sample:
        te = te.sample(a.test_sample, random_state=RANDOM_SEED).reset_index(drop=True)
    log(f"test set: {len(te):,} flights")

    t = time.time()
    scores = {"model": bundle["model"].predict(te[MODEL_FEATS]).astype(np.float64)}
    log(f"model scored ({time.time() - t:.0f}s)")
    t = time.time()
    scores["tim"], scores["clim"] = clim.predict(te)
    scores["route"] = route.predict(te)
    log(f"climatologies scored ({time.time() - t:.0f}s)")
    scores["insolation"] = te["night_score_full_0"].to_numpy(np.float64)
    scores["perfect"] = te["contrail_CO2_km"].to_numpy(np.float64)
    actual = scores["perfect"]

    tag = a.setting + (f"_s{a.test_sample}" if a.test_sample else "") \
        + ("_debug" if a.fit_years else "")
    REVISION_PLOTS.mkdir(parents=True, exist_ok=True)
    rows = [dict(setting=a.setting, ranker=n, n=len(te),
                 spearman=rank_corr(s, actual) if n != "perfect" else 1.0,
                 **shares(s, actual)) for n, s in scores.items()]
    cap = pd.DataFrame(rows)
    for p in (OUT, REVISION_PLOTS):
        cap.to_csv(p / f"booking_capture_{tag}.csv", index=False)
        curves(scores, actual).to_csv(p / f"booking_curves_{tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(cap.round(3).to_string(index=False))

    if a.setting == "pooled":
        # climatology scores for the Fig. 3a overlay, in test-set order
        pd.DataFrame({"year": te["year"].to_numpy() if "year" in te else 0,
                      "tim": scores["tim"], "clim": scores["clim"],
                      "route": scores["route"]}).to_parquet(OUT / "_clim_pred_test.parquet")
        log("wrote _clim_pred_test.parquet")
        return

    names = [n for n in scores if n != "perfect"]
    summ, detail = [], []
    for key in ("day", "week"):
        t = time.time()
        s, per_set = choice_sets(te, scores, key)
        summ.append(s)
        detail.append(by_size_and_haul(per_set, names, key))
        log(f"choice sets by {key}: {s['n_sets'].iloc[0]:,} sets, "
            f"{s['n_flights'].iloc[0]:,} flights ({time.time() - t:.0f}s)")
        print(s.round(3).to_string(index=False))
    summ, detail = pd.concat(summ), pd.concat(detail)
    for p in (OUT, REVISION_PLOTS):
        summ.to_csv(p / f"booking_sets_2025{tag[3:]}.csv", index=False)
        detail.to_csv(p / f"booking_sets_by_size_2025{tag[3:]}.csv", index=False)
    log("done")


if __name__ == "__main__":
    main()
