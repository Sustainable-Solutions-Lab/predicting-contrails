"""
Score arbitrary flights with the contrails model — the core the web
tool's API wraps.

    from webtool.score import score_flight
    result = score_flight("SFO", "LHR", "2026-10-12T21:40", "B789")

Returns percentile (vs. all 2021 flights), top-10% flag, and calibrated
expected contrail CO2e. `hour_curve=True` additionally scores the same
route/date at all 24 departure hours — the "when to fly" curve.

CLI smoke test:
    python webtool/score.py SFO LHR 2026-10-12T21:40 B789
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from webtool.features import ASSETS, LEAN_AC_FEATS, build_features  # noqa: E402

_booster = None
_calib = None
_aircraft = None


def _load():
    global _booster, _calib, _aircraft
    if _booster is None:
        _booster = xgb.Booster()
        model = ASSETS / "model_canonical.ubj"     # full accuracy; 57 MB
        if not model.exists():
            model = ASSETS / "model_web.ubj"       # 4 MB distilled fallback
        _booster.load_model(str(model))
        with open(ASSETS / "calibration.json") as f:
            _calib = json.load(f)
        with open(ASSETS / "aircraft.json") as f:
            _aircraft = json.load(f)
    return _booster, _calib, _aircraft


def _predict(feature_rows: list[dict]) -> np.ndarray:
    booster, _, aircraft = _load()
    df = pd.DataFrame(feature_rows)
    df["aircraft_type_icao"] = pd.Categorical(
        df["aircraft_type_icao"], categories=aircraft["categories"],
    )
    dm = xgb.DMatrix(df[LEAN_AC_FEATS], enable_categorical=True)
    return booster.predict(dm)


def _percentile(pred_log: float) -> float:
    _, calib, _ = _load()
    grid = np.asarray(calib["pred_log_quantiles"])
    return float(np.interp(pred_log, grid, np.linspace(0, 100, len(grid))))


def score_flight(origin: str, dest: str, dep_utc_iso: str,
                 aircraft_icao: str, hour_curve: bool = False) -> dict:
    dep = datetime.fromisoformat(dep_utc_iso)
    feats = build_features(origin, dest, dep, aircraft_icao)
    pred = float(_predict([feats])[0])
    pct = _percentile(pred)
    _, calib, _ = _load()
    b = min(99, int(pct))
    kg_km = calib["bin_mean_co2e_kg_per_km"][b]
    result = {
        "origin": origin.upper(), "dest": dest.upper(),
        "dep_utc": dep_utc_iso, "aircraft": aircraft_icao.upper(),
        "distance_km": round(feats["total_flight_distance_km"], 1),
        "night_score": round(feats["night_score_full_0"], 1),
        "land_score": round(feats["land_score"], 1),
        "pred_log": round(pred, 4),
        "percentile": round(pct, 1),
        "flagged_top10": pred >= calib["threshold_top10_pred_log"],
        "expected_co2e_kg_per_km": round(kg_km, 3),
        "expected_co2e_t_per_flight": round(
            kg_km * feats["total_flight_distance_km"] / 1000.0, 2),
    }
    if hour_curve:
        base = dep.replace(hour=0, minute=0)
        rows = [build_features(origin, dest, base + timedelta(hours=h),
                               aircraft_icao) for h in range(24)]
        preds = _predict(rows)
        result["hour_curve"] = [
            {"hour_utc": h, "percentile": round(_percentile(float(p)), 1)}
            for h, p in enumerate(preds)
        ]
    return result


if __name__ == "__main__":
    origin, dest, dep, ac = sys.argv[1:5]
    out = score_flight(origin, dest, dep, ac, hour_curve="--curve" in sys.argv)
    print(json.dumps(out, indent=2))
