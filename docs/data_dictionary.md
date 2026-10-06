# Data dictionary

Canonical mapping between the column names used in the code and data
files, the plain-language names used in the paper, and each quantity's
definition and units. The manuscript should use only the paper names;
code and released data keep the column names below (many originate in
the Breakthrough Energy contrails-team shards, and the trained model
bundle validates feature names at predict time, so they are not renamed).

Sources: raw label shards (`adjustedEFs/features_{year}*_gdf.pq`),
engineered columns from `experiments/common.py:add_features()`, and the
model-output columns written by `experiments/final_model.py` and
`figures/demand_shift.py`.

## Target and label columns

| Column | Paper name | Definition / units |
|---|---|---|
| `total_contrail_energy_forcing` | contrail energy forcing | Raw label from the contrail process model: net radiative energy forcing integrated over the contrail's lifetime, for the whole flight (J). Negative = net cooling. |
| `contrail_CO2` | CO₂-equivalent contrail forcing | `total_contrail_energy_forcing / (AGWP₁₀₀ · s_yr · A_Earth)` with AGWP₁₀₀ = 8.25 × 10⁻¹⁴ W m⁻² yr (kg CO₂)⁻¹, s_yr = 3.154 × 10⁷ s, A_Earth = 5.101 × 10¹⁴ m². Units: kg CO₂-e per flight. |
| `contrail_CO2_km` | per-km contrail forcing | `contrail_CO2 / total_flight_distance_km` (kg CO₂-e km⁻¹). The quantity all ranking, percentile, and capture statistics are computed on. |
| `contrail_type` | signed-log per-km forcing (regression target) | Despite the name, this is a continuous value, not a category: `ln(x)` for `x > 0`, `−ln(−x)` for `x < 0`, `0` for `x = 0`, where `x = contrail_CO2_km`. Monotone within each sign; see the caveat below. |

**Caveat on the signed-log target.** The transform is monotone
increasing separately on the positive and the negative side, but not
across zero for small magnitudes: for |x| < 1 kg km⁻¹ a mildly
*warming* flight maps below a mildly *cooling* one (e.g. x = +0.5 →
−0.69 while x = −0.5 → +0.69), and x → 0⁺ maps to −∞ rather than to
values near 0. About 30% of nonzero-forcing flights (8% of all flights)
have |x| < 1 and sit in this scrambled band. The top-tail rankings that
drive the paper's headline results are unaffected (large positive x maps
to large targets unambiguously), and the percentile→kg calibration is
monotonicized empirically, but mid-distribution and cooling-side
rankings inherit some label scrambling from this construction. A
monotone alternative for future work is `sign(x)·ln(1+|x|)`.

## Model features (`MODEL_FEATS`, 14 columns)

| Column | Paper name | Definition / units |
|---|---|---|
| `total_flight_distance_km` | flight distance | Great-circle flight distance (km). Also a label denominator, above. |
| `day_sin`, `day_cos` | day of year (cyclic) | `sin/cos(2π · dayofyear / 365)` of the first waypoint time (UTC). |
| `start_hour_sin`, `start_hour_cos` | departure hour (cyclic) | `sin/cos(2π · hour / 24)` of the first waypoint time (UTC, integer hour). |
| `OriginLat` | origin latitude | Degrees, positive north. |
| `OriginLon_sin`, `OriginLon_cos` | origin longitude (cyclic) | `sin/cos(2π · lon / 360)`, longitude in degrees. |
| `DestinationLat` | destination latitude | Degrees, positive north. |
| `DestinationLon_sin`, `DestinationLon_cos` | destination longitude (cyclic) | As origin longitude. |
| `land_score` | over-land fraction | Share of the great-circle track over land (from the label shards). |
| `night_score_full_0` | insolation | Route-mean darkness over the flight, weighted by solar geometry (from the label shards; 0–100). The paper refers to this feature as "insolation": the stored value is the complement of route-averaged normalized insolation (high value = little sunlight along the route). |
| `aircraft_type_icao` | aircraft type | ICAO type designator, categorical (e.g. B77W, A320). |

## Additional engineered columns (corpus filtering and figures only)

Retained so the corpus's finite-value row filter is reproducible
(`ALL_ENGINEERED_FEATS`); not model inputs.

| Column | Definition |
|---|---|
| `end_hour_sin/cos`, `mid_hour_sin/cos` | Cyclic hour of last waypoint and flight midpoint (UTC). |
| `time_span_hours` | Last minus first waypoint time (h). |
| `crosses_midnight` | 1 if last waypoint's UTC date differs from the first's. |
| `bearing_sin/cos` | Cyclic initial great-circle bearing (degrees from the shards). |
| `night_score_full_1..3` | Sun-weighted night fraction of the 2nd–4th flight quarters. |
| `night_score_bool_0..3` | Unweighted (boolean-sun) night fraction per flight quarter. |
| `year` | Label year (2019 or 2021), added at pooling. |

## Model-output columns

| Column | Paper name | Definition |
|---|---|---|
| `pred_log` | predicted forcing | Model prediction in signed-log target space (`_2021_predictions.parquet` and downstream). |
| `percentile` (API) | contrail-warming percentile | Rank of `pred_log` in the pooled 52.5M-flight prediction distribution (0–100). |

## Other identifiers

| Column | Definition |
|---|---|
| `flight_id` | Contrails-team flight identifier (`YYMMDD–NNNN–CCCN`); join key for the corporate logs and validation runs. |
| `first_waypoint_time`, `last_waypoint_time` | UTC timestamps of the first/last track waypoint; departure/arrival proxies. |
| `origin_airport`, `destination_airport` | IATA codes (prediction cache and corporate-log joins). |
