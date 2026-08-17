# predicting-contrails

Predicting the radiative forcing of individual flights' contrails from
schedule-level information that a passenger or booking service would know
weeks-to-months in advance.

This is the code repository for the manuscript working-titled
**"Flying smarter to reduce radiative forcing of contrails"**
(Whiteson, Bonnemaizon, Shapiro, Davis; in preparation).

## Motivation

Aviation's non-CO2 climate forcing — dominated by warming contrail cirrus —
is comparable to or larger than its CO2 forcing on a 100-yr horizon, yet it
is unevenly distributed across flights: a small minority of flights produce
a large majority of the warming. Process-based models (e.g.
[CoCiP](https://contrails.org/)) can estimate per-flight contrail energy
forcing post hoc from atmospheric state along the actual trajectory, but
those inputs are not available to consumers at booking time.

This project asks: **using only the information a traveler has when booking
(origin, destination, scheduled time, aircraft type, season, sun geometry
along the great-circle route, …), how well can we predict whether a given
flight will be a high-warming contrail outlier?** If we can flag the worst
~5–10% of flights, climate-conscious travelers and corporate travel
programs could meaningfully reduce contrail forcing without changing where
or whether people fly.

## Targets

Two complementary prediction tasks, both trained on the same per-flight
process-model labels:

1. **Top-decile classifier** — binary "is this flight in the top 5–10% of
   per-km contrail energy forcing?" Intended as a simple consumer rule.
2. **Numerical regression** — predict per-flight (or per-passenger-km)
   contrail energy forcing in CO2-equivalent tonnes, suitable for a
   booking-service display alongside CO2.

## Data

The training labels come from a process-based contrail model run by the
Breakthrough Energy contrails team (see [contrails.org](https://contrails.org/))
on commercial flights from **2019 and 2021** (2020 omitted as
COVID-anomalous). The dataset covers ~2.3M flights with per-flight
contrail energy forcing and many auxiliary variables.

**Data is not committed to this repo.** The current canonical copy lives in
the lab Dropbox under
`Papers/Active Prep/WS Corp contrails (w Silas)/`:

- `adjustedEFs/features_YYYYMMW_gdf.pq` — 94 monthly parquet shards (75
  columns) with raw process-model outputs and Silas's engineered features.
- `Analysis/additionalfeatures.pq` — cleaned, deduplicated 2.29M-flight
  table used for downstream modeling.
- `Analysis/flight collection/YYYYMMDD-summary.pq` — slim daily summaries
  (partial coverage).
- `Results/Results_Cust1*.{csv,xlsx}`, `Results/Results_Cust2.xlsx` —
  Watershed customer flight datasets used for the "applied predictions"
  case study (Fig. 4). **Customer datasets are proprietary and stay out of
  this repo.**

A public mirror of the training data (without customer datasets) will be
deposited on Zenodo at submission.

### Schedule-only feature set

To keep the model usable at booking time, training features should be
restricted to variables knowable from a schedule. Acceptable:

- Origin / destination airport, region, continent, transoceanic fraction
- Scheduled departure local/UTC time → hour, day-of-week, season
- Great-circle bearing, distance, mean / max latitude
- Sun geometry along the great-circle (day/night fraction, dawn/dusk flag)
- Aircraft ICAO type, engine, nominal seat capacity

Out of scope (these depend on flown trajectory or atmospheric state and
must not leak into training):

- `Tempmultiplier`, `maxTempmultiplier`
- `night_score_full_*`, `night_score_bool_*` (these are altitude-bin
  specific and computed along the actual waypoints)
- `mean_aircraft_mass`, `total_fuel_burn`, `n_wypts`, `mean_sdr_/olr_*`
- Anything derived from `total_persistent_contrail_length_km` or contrail
  forcing itself

A clean schedule-only feature list is one of the first things we'll lock
down in this repo (see `src/features/`).

## Repo layout

```
predicting-contrails/
├── README.md
├── .gitignore
├── experiments/                # training + evaluation scripts
│   ├── feature_pruning.py        # FULL vs LEAN vs LEAN+AC sweep
│   ├── hyperparameter_tune.py    # 24-config grid + early stopping
│   ├── final_model.py            # canonical retrain + Fig 3c
│   └── outputs/                  # CSVs, PNGs (cache + joblib gitignored)
├── figures/                    # publication figure scripts (no outputs here —
│   ├── fig1_map.py             #   everything renders straight to Dropbox Plots/)
│   ├── fig1c_matrix.py
│   ├── fig2_concentration.py
│   ├── fig4_customer.py
│   └── fig5_demand_shift.py
├── manuscript/                 # paper draft + .docx builder
│   ├── draft.md
│   ├── build_draft_docx.py
│   └── Working Draft <date> [Contrails - autodraft].docx
└── archived/                   # historical artifacts, kept for traceability
    ├── sherlock-snapshot/        # Silas Whiteson's Sherlock files,
    │                             # imported one-time as the starting point
    └── silas-local/              # Silas's laptop-side originals
                                  # (Flights.py, finalcolumns.py, warmingmap.py)
```

## How the GitHub repo and the lab Dropbox interact

This is a hybrid project: code lives in GitHub for version control and
collaboration, while data, raw figure outputs, the Illustrator-polished
figures, and the manuscript drafts live in the lab Dropbox under
`Papers/Active Prep/Contrails/WS Corp contrails (w Silas)/`.

| Where it lives | What's there | Authoritative for |
|---|---|---|
| **GitHub** (private) | Code, manuscript markdown + .docx builder, the auto-generated .docx, small CSV summary outputs | Reproducibility, code review, change history |
| **Dropbox** `adjustedEFs/` | Per-month process-model parquets (94 shards, ~52M flights) | Training data |
| **Dropbox** `Results/` | Customer 1 and Customer 2 flight logs (sensitive — never in repo) | Customer applications |
| **Dropbox** `Plots/` | **Raw machine-generated figures**, written directly by the scripts in this repo | Latest model outputs (bypasses git for binaries) |
| **Dropbox** `Figures/` | **Illustrator-polished publication versions** maintained by hand | Manuscript-ready figures |
| **Dropbox** `Manuscript/` | Hand-edited Working Drafts + autodraft mirror | Active writing |

In short: **`Plots/` is what the code emits, `Figures/` is what humans
clean up for publication.** Every figure script saves its png/pdf/eps
(plus any per-figure CSV) directly into `Plots/` — there are no output
folders in the repo, and nothing needs syncing.

To regenerate the Illustrator-ready `Figures/` versions, open the
corresponding .ai file in Dropbox and re-import the latest raster from
`Plots/`.

## Status

Inherited from prior work by Silas Whiteson. Where things stand:

- ✅ Process-model dataset assembled and engineered features generated
  (`Flights.py`, `finalcolumns.py`).
- ✅ Initial random-forest model trained (location of training script TBD —
  `warmingmap.py` reads predictions from a `Sherlockdfs/` directory not
  yet pulled into this repo).
- ✅ Draft figures 1–4 produced (Adobe Illustrator + PNG in Dropbox).
- ✅ Manuscript outline through Methods, with placeholder numbers.

### Next up (this repo)

- [ ] Recover/relocate the model-training code from Sherlock and commit it.
- [ ] Lock down the schedule-only feature set and re-train both the
      top-decile classifier and the per-km RF regressor.
- [ ] Reproducible Lorenz / concentration plot from current model
      (Figure 2).
- [ ] Refine Figure 3c (ML-vs-baseline bar chart).
- [ ] Re-render Figure 1 global map with interpolated (not dotted) RF
      shading.
- [ ] Apply final rule to Cust1 / Cust2 datasets → Figure 4.
- [ ] First full draft of the paper (numbers in, no `X%` placeholders).

## Reproducing

```bash
# 1. Build the pooled feature cache and run the lean-vs-full sweep
python experiments/feature_pruning.py
# 2. Hyperparameter grid (loads cache from step 1)
python experiments/hyperparameter_tune.py
# 3. Train the canonical model and produce Fig 3c
python experiments/final_model.py
# 4. Figures (each writes straight to Dropbox/Plots/)
python figures/fig1_map.py
python figures/fig1c_matrix.py          # region-to-region forcing matrix
python figures/fig2_concentration.py
python figures/fig5_demand_shift.py     # builds 2021 predictions cache
python figures/fig4_customer.py         # uses fig5's cache
# 5. Manuscript autodraft (also mirrored to Dropbox/Manuscript/)
python manuscript/build_draft_docx.py
```

The 2.35 GB pooled feature cache and 0.77 GB 2021 predictions cache are
gitignored; both are rebuilt on first run from the Dropbox parquets in
~6 min apiece, and reused across subsequent runs.

## Authors

Silas Whiteson, Xavier Bonnemaizon (LSCE/IPSL), Marc Shapiro
(Breakthrough Energy), Steven J. Davis (Stanford / Sustainable Solutions
Lab). Final author list and order TBD.

## License

TBD.
