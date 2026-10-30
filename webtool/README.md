# webtool — serving layer for the booking-time contrail tool

Everything needed to score an **arbitrary** flight (origin, destination,
departure time, aircraft) with the canonical scheduling model, outside the
training pipeline. The public-facing tool lives in the lab-website repo
(`Sites/SustainableSolutions`): `api/contrails.py` + the
`src/tools/contrails/` React island; this folder is the source of truth
for the science side.

- `features.py` — recomputes all 14 features for arbitrary flights.
  The two engineered features (`night_score_full_0`, `land_score`) were
  reverse-validated against training shards to 0.000 mean error; exact
  definitions in the module docstring.
- `export_assets_gaia.py` — writes `assets/` from the GAIA 2019–25
  model (`experiments/gaia2025_retrain.py`, pooled run; 79 MB): the
  36.7M-flight 2025 percentile calibration, aircraft category list,
  2024–25 route aircraft mixes, 2025 flight-number table. This is the
  model the site serves (since 30 Oct 2026).
- `export_assets.py` — the same assets for the original 2019+2021 model
  (canonical 57 MB + 4 MB distilled fallback), plus the slim airport
  table and land mask, which do not change. Distillation accuracy
  tradeoffs in `assets/distill_metrics.json` (canonical 61.4% top-decile
  capture vs 56.1% distilled @ 4 MB, 58.7% @ 20 MB — we ship canonical).
- `score.py` — scoring core + CLI smoke test:
  `python webtool/score.py SFO LHR 2026-10-12T21:40 B789 --curve`

After re-training the canonical model, re-run `export_assets_gaia.py` and
re-copy `assets/` into the website repo's `api/_contrails_assets/`.
