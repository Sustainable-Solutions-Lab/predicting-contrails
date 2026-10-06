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

The model is live in a free web tool, [Contrail Check](https://sustainablesolutions.vercel.app/tools/contrails),
and a [Chrome extension](https://chromewebstore.google.com/detail/contrail-check-for-google/jgoiipmnojcdecidalejcadljdcjacgl)
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
flown trajectories and ERA5 reanalysis weather. 2020 is excluded as
COVID-anomalous.

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
├── docs/data_dictionary.md     column names, paper terms, definitions, units
└── manuscript/                 early auto-generated draft (superseded)
```

Figure scripts write PNG, PDF and EPS files to the path set as
`DROPBOX_PLOTS` in `experiments/common.py`.

## Reproducing the results

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
- Marc Shapiro (Breakthrough Energy / Orca Sciences)
