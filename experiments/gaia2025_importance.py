#!/usr/bin/env python3
"""What the GAIA 2019–25 model (gaia2025_retrain.py pooled) relies on.

1. Grouped permutation importance (sin/cos pairs shuffled jointly, as in
   Fig. 3c) on a held-out sample.
2. Aircraft-type counterfactual swap (as Supplementary Fig. 1): a fixed
   sample of held-out schedules is re-predicted as each type, and
   predictions are converted to expected GJ per km through a percentile
   calibration built on the same held-out years. Compared side by side
   with the published model's values (Plots/figS1_aircraft_types.csv).

Usage: python experiments/gaia2025_importance.py
Writes experiments/outputs/gaia2025_permutation_importance_grouped.csv and
Plots/figS1_aircraft_types_gaia2025.{csv,png,pdf}
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                            "font.size": 9})

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "figures"))
from common import DROPBOX_PLOTS, MODEL_FEATS, OUT, RANDOM_SEED, TARGET  # noqa: E402
from gaia2025_retrain import YEARS, load, split  # noqa: E402
from figS1_aircraft_types import NARROW, WIDE, J_PER_KG, MIN_TRAIN  # noqa: E402

GROUPS = {
    "Flight distance": ["total_flight_distance_km"],
    "Day of year": ["day_sin", "day_cos"],
    "Departure hour": ["start_hour_sin", "start_hour_cos"],
    "Origin latitude": ["OriginLat"],
    "Origin longitude": ["OriginLon_sin", "OriginLon_cos"],
    "Destination latitude": ["DestinationLat"],
    "Destination longitude": ["DestinationLon_sin", "DestinationLon_cos"],
    "Over-land fraction": ["land_score"],
    "Route-mean insolation": ["night_score_full_0"],
    "Aircraft type (ICAO)": ["aircraft_type_icao"],
}
N_PERM, N_SCHED, N_CAL = 200_000, 100_000, 3_000_000


def held_out(cats) -> tuple[pd.DataFrame, pd.Series]:
    """The pooled model's test split (same seeds), all years; and the
    number of training flights per type."""
    tes, counts = [], []
    for y in YEARS:
        tr, te = split(load(y, cats), RANDOM_SEED + y)
        counts.append(tr["aircraft_type_icao"].value_counts())
        tes.append(te)
    return pd.concat(tes, ignore_index=True), pd.concat(counts).groupby(level=0).sum()


def grouped_importance(model, te) -> pd.DataFrame:
    s = te.sample(N_PERM, random_state=RANDOM_SEED)
    base = r2_score(s[TARGET], model.predict(s[MODEL_FEATS]))
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for g, cols in GROUPS.items():
        drops = []
        for _ in range(5):
            x = s[MODEL_FEATS].copy()
            idx = rng.permutation(len(x))
            for c in cols:
                x[c] = x[c].to_numpy()[idx] if c != "aircraft_type_icao" \
                    else pd.Categorical.from_codes(
                        x[c].cat.codes.to_numpy()[idx], x[c].cat.categories)
            drops.append(base - r2_score(s[TARGET], model.predict(x)))
        rows.append(dict(group=g, n_features=len(cols),
                         importance_mean=np.mean(drops),
                         importance_std=np.std(drops)))
    out = pd.DataFrame(rows).sort_values("importance_mean")
    out.to_csv(OUT / "gaia2025_permutation_importance_grouped.csv", index=False)
    print(out.round(4).to_string(index=False), flush=True)
    return out


def type_swap(model, te, counts) -> pd.DataFrame:
    cal = te.sample(N_CAL, random_state=1)
    p = model.predict(cal[MODEL_FEATS])
    q = np.quantile(p, np.linspace(0, 1, 101))
    b = np.clip(np.searchsorted(q, p, side="right") - 1, 0, 99)
    bins = pd.Series(cal["contrail_CO2_km"].to_numpy()).groupby(b).mean()
    bins = bins.reindex(range(100)).to_numpy()
    d = te["total_flight_distance_km"]
    panels = [("Widebody types on long-haul schedules (>5,000 km)", WIDE,
               te[d > 5000].sample(N_SCHED, random_state=0)),
              ("Narrowbody types on medium-haul schedules (500–3,000 km)",
               NARROW, te[(d > 500) & (d <= 3000)].sample(N_SCHED, random_state=0))]
    cats = te["aircraft_type_icao"].cat.categories
    rng = np.random.default_rng(0)
    rows = []
    for title, types, sched in panels:
        for icao, name in types.items():
            if counts.get(icao, 0) < MIN_TRAIN:
                continue
            s = sched.copy()
            s["aircraft_type_icao"] = pd.Categorical([icao] * len(s), categories=cats)
            pr = model.predict(s[MODEL_FEATS])
            bb = np.clip(np.searchsorted(q, pr, side="right") - 1, 0, 99)
            gj = bins[bb] * J_PER_KG / 1e9
            boots = [gj[rng.integers(0, len(gj), len(gj))].mean() for _ in range(200)]
            rows.append(dict(panel=title, icao=icao, name=name, mean=gj.mean(),
                             lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5),
                             n_train=int(counts[icao])))
    res = pd.DataFrame(rows)
    old = pd.read_csv(DROPBOX_PLOTS / "figS1_aircraft_types.csv")[["icao", "mean"]]
    res = res.merge(old.rename(columns={"mean": "mean_published"}), on="icao", how="left")
    res.to_csv(DROPBOX_PLOTS / "figS1_aircraft_types_gaia2025.csv", index=False)
    print(res.round(2).to_string(index=False), flush=True)
    return res


def plot(res: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9))
    for ax, (title, r) in zip(axes, res.groupby("panel", sort=False)):
        r = r.sort_values("mean")
        y = np.arange(len(r))
        ax.scatter(r["mean_published"], y + 0.15, s=26, color="#9E0142",
                   edgecolor="white", lw=1, label="Published model (old labels)")
        ax.hlines(y - 0.15, r["lo"], r["hi"], color="#3288BD", lw=1.6)
        ax.scatter(r["mean"], y - 0.15, s=26, color="#3288BD", zorder=3,
                   edgecolor="white", lw=1, label="Retrained, GAIA 2019–25 labels")
        ax.set_yticks(y)
        ax.set_yticklabels(r["name"])
        ax.set_xlabel("Expected contrail energy forcing (GJ per km)")
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(axis="x", color="#dddddd", lw=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(w_pad=2.5)
    for ext in ("png", "pdf"):
        fig.savefig(DROPBOX_PLOTS / f"figS1_aircraft_types_gaia2025.{ext}",
                    dpi=200, bbox_inches="tight")


def main() -> None:
    b = joblib.load(OUT / "gaia2025_pooled.joblib")
    te, counts = held_out(b["categories"])
    grouped_importance(b["model"], te)
    plot(type_swap(b["model"], te, counts))


if __name__ == "__main__":
    main()
