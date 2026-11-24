# predicting-contrails

Code for the manuscript

> **Flyers can identify the most-warming flights when booking**
> Steven J. Davis, Silas Whiteson, Xavier Bonnemaizon, Ken Caldeira, Roger Teoh, and Marc Shapiro (2026)

We train a model on per-flight contrail forcing for 52.5 million commercial
flights in 2019 and 2021 that uses only information available in a published
flight schedule: origin and destination, departure date and time, aircraft
type, and great-circle geometry, with no weather data. Although schedule
information cannot anticipate the weather a flight will meet, the 10% of
flights the model flags as most warming account for 68% of total contrail
forcing in held-out data, about 60% of what perfect foresight could achieve.
We apply the model to two corporations' 2021 flight logs and to the full 2021
schedule to estimate how much forcing could be avoided by rebooking onto
same-route alternatives.

The model is live in a free web tool, [Predicting contrails at booking](https://sustainablesolutions.vercel.app/tools/contrails),
and a Chrome extension, [Contrail check for Google Flights](https://chromewebstore.google.com/detail/contrail-check-for-google/jgoiipmnojcdecidalejcadljdcjacgl),
that adds predictions to Google Flights results.

## Terminology

Throughout, *contrail forcing* means contrail energy forcing: the radiative
forcing of a flight's contrails integrated over their lifetime and spatial
extent, in joules. Summed over many flights it is a net quantity that includes
the small negative contribution of contrails that cool, so shares of the total
can exceed 100%. Where CO₂-equivalents appear, energy forcing is converted with
AGWP₁₀₀ = 8.25 × 10⁻¹⁴ W m⁻² yr (kg CO₂)⁻¹ (1 kg CO₂-e per 1.33 GJ).
[`docs/data_dictionary.md`](docs/data_dictionary.md) maps every column name in
the code to the term used in the paper, with definitions and units.

## Data

Training labels are per-flight contrail energy forcing from the GAIA global
aviation emissions inventory (Teoh et al. 2024, *Atmos. Chem. Phys.* 24,
6071–6093), computed with the Contrail Cirrus Prediction model (CoCiP) on
flown trajectories and ERA5 reanalysis weather. The submitted version used
2019 and 2021 with soot-only ice nucleation; the revision uses the GAIA
2019–2025 labels (196 million flights, including 2020) in which contrail ice
nucleates on volatile particles as well as soot (see "Revision" below).

**Data files are not in this repository.** The cleaned per-flight label and
feature tables (excluding the corporate flight logs) will be deposited on
Zenodo with the paper. The two corporate flight logs used in the case study are
proprietary and are not released. Scripts read the label shards from the path
set as `DROPBOX_PARQUETS` in [`experiments/common.py`](experiments/common.py);
point it at your copy of the data.

## Model

A gradient-boosted regression tree model (XGBoost; depth 11, learning rate
0.05, 348 trees, minimum child weight 50, 85% row subsampling) predicts the
signed logarithm of per-km contrail forcing from 14 features:

- flight distance
- aircraft type (ICAO designator, native categorical)
- day of year and UTC departure hour (sine/cosine pairs)
- origin and destination latitude, and longitude (sine/cosine pairs)
- share of the great-circle route over land
- insolation along the great-circle route (sun-weighted night fraction from
  solar geometry at 30 interpolated waypoints)

No feature uses atmospheric data or the flown trajectory. Models are trained
on a 2:1 train/test split of the pooled 2019 and 2021 flights, stratified by
year. Hyperparameters come from a 24-configuration grid search scored by
held-out worst-decile capture.

## Repository layout

```
predicting-contrails/
├── experiments/
│   ├── common.py               shared paths, feature engineering, helpers
│   ├── build_cache.py          build the pooled 2019+2021 feature cache (run first)
│   ├── hyperparameter_tune.py  24-configuration grid search with early stopping
│   ├── final_model.py          train the canonical model; metrics, importances, Fig. 3
│   ├── cross_year.py           train on one year, test on the other
│   ├── validation/             independent re-simulation of 273 flights with pycontrails
│   │                           and concordance with published statistics
│   └── outputs/                small metric CSVs and diagnostic plots
├── figures/                    publication figure scripts
│   ├── fig1_map.py, fig1c_matrix.py   global map and region matrix (Fig. 1)
│   ├── fig2_concentration.py          concentration of forcing across flights (Fig. 2)
│   ├── demand_shift.py                builds the 2021 predictions cache used by Fig. 4
│   │                                  (and a diagnostic of same-route alternatives)
│   ├── fig4_customer.py               avoidance surfaces (Fig. 4) and bars (Supplementary Fig. 2)
│   ├── figS1_aircraft_types.py        aircraft-type counterfactual (Supplementary Fig. 1)
│   └── figS3_pycontrails.py           independent validation (Supplementary Fig. 3)
├── webtool/                    feature computation and scoring for arbitrary flights,
│                               and the asset export used by the web tool
└── docs/data_dictionary.md     column names, paper terms, definitions, units
```

Figure scripts write PNG, PDF and EPS files to the path set as
`DROPBOX_PLOTS` in `experiments/common.py`.

## Revision (GAIA 2019–2025 labels)

The revised paper retrains the same 14-feature model on the GAIA 2019–2025
per-flight output. Everything is driven from one corpus (one parquet per
year, built by `experiments/gaia2025_corpus.py` from the ingested daily files)
and the scripts below; figure scripts take `--gaia2025` and write to
`Plots/Revision (GAIA)`.

```bash
cd experiments
python gaia2025_ingest.py                      # daily GAIA files -> slim jets-only parquets
python gaia2025_corpus.py                      # schedule features, one parquet per year
python gaia2025_retrain.py oldmodel pooled ood loyo week week_buf
#   pooled    random 2:1 split within each year (the model in Fig. 3; saved)
#   ood       train 2019-23, test 2024 and 2025 (saved; used by booking_eval.py)
#   loyo      leave one year out; the leave-2021-out model is saved for Fig. 4
#   week      one third of each year's ISO weeks held out (no shared days)
#   week_buf  ... with two buffer days on either side of each test week
python gaia2025_importance.py                  # grouped permutation importance (Fig. 3c)
python fig3_numbers.py                         # every number quoted from Fig. 3
python booking_eval.py --setting pooled        # TIM-style climatology for the Fig. 3a overlay
python booking_eval.py --setting ood           # climatologies and booking-set evaluation on 2025
python validation/insolation_waypoints.py      # 30 vs 300 waypoints for the insolation feature
python validation/schedule_vs_actual.py        # scheduled vs operated departure hour
cd ../figures
python fig1_map.py --gaia2025; python fig1c_matrix.py --gaia2025
python fig2_concentration.py --gaia2025
python ../experiments/final_model.py --gaia2025        # Fig. 3 (with the climatology curve)
python fig4_customer.py --gaia2025 --loyo2021          # Fig. 4 and Supplementary Fig. 2, with bootstrap
python figS1_aircraft_types.py --gaia2025
python figS3_pycontrails.py --gaia2025
python figS_fig1_variability.py --gaia2025             # Supplementary Fig. 5
python figS6_booking_sets.py                           # Supplementary Fig. 6
```

Headline numbers of the revision: the predicted worst 10% of flights carry
71% of net contrail forcing on the random split, 67% out of time (2024 and
2025), 62–69% leaving each year out; `experiments/outputs/gaia2025_metrics.csv`
holds every split. The web tool serves the pooled model
(`webtool/export_assets_gaia.py`).

## Reproducing the results (submitted version, 2019 and 2021 labels)

Requires Python 3.11 with XGBoost, scikit-learn, pandas, pyarrow, matplotlib
and SciPy. The independent validation additionally requires pycontrails
(v0.63) and access to the public ERA5 ARCO archive.

```bash
cd experiments
python build_cache.py             # pooled 52.5M-flight feature cache (~6 min, ~2.4 GB)
python hyperparameter_tune.py     # grid search
python final_model.py             # canonical model, metrics, permutation importance, Fig. 3
python cross_year.py              # cross-year generalization (~20 min)
cd ../figures
python demand_shift.py            # builds the 2021 predictions cache used by Fig. 4
python fig1_map.py
python fig1c_matrix.py
python fig2_concentration.py
python fig4_customer.py           # requires the corporate logs (not released)
python figS1_aircraft_types.py
python ../experiments/validation/pycontrails_crossval.py   # slow; caches weather locally
python figS3_pycontrails.py
```

Caches, trained model files and weather downloads are gitignored and rebuilt
on first run.

## Authors

- Steven J. Davis (Stanford University), corresponding author: sjdavis@stanford.edu
- Silas Whiteson (Colorado College)
- Xavier Bonnemaizon (LSCE/IPSL, Université Paris-Saclay; UC Irvine)
- Ken Caldeira (Gates Ventures)
- Roger Teoh (Imperial College London)
- Marc Shapiro (Contrails.org)

## License

Code is released under the [MIT License](LICENSE). The training data deposited on
Zenodo (https://doi.org/10.5281/zenodo.23194382) are licensed separately, under CC BY 4.0.
