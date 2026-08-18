# webtool — serving layer for the booking-time contrail tool

Everything needed to score an **arbitrary** flight (origin, destination,
departure time, aircraft) with the canonical LEAN+AC model, outside the
training pipeline. The public-facing tool lives in the lab-website repo
(`Sites/SustainableSolutions`): `api/contrails.py` + the
`src/tools/contrails/` React island; this folder is the source of truth
for the science side.

- `features.py` — recomputes all 14 features for arbitrary flights.
  The two engineered features (`night_score_full_0`, `land_score`) were
  reverse-validated against training shards to 0.000 mean error; exact
  definitions in the module docstring.
- `export_assets.py` — writes `assets/`: model exports (canonical
  57 MB + 4 MB distilled fallback), the 22.3M-flight percentile
  calibration, aircraft category list, slim airport table, land mask.
  Distillation accuracy tradeoffs in `assets/distill_metrics.json`
  (canonical 61.4% top-decile capture vs 56.1% distilled @ 4 MB,
  58.7% @ 20 MB — we ship canonical).
- `score.py` — scoring core + CLI smoke test:
  `python webtool/score.py SFO LHR 2026-10-12T21:40 B789 --curve`

After re-training the canonical model, re-run `export_assets.py` and
re-copy `assets/` into the website repo's `api/_contrails_assets/`.
