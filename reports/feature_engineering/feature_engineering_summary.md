# JEEVAN-DRISHTI Phase 3: Feature Engineering Summary Report (2023–2025 Scope)

- **Engine Version**: `3.0.0` | **Dataset Version**: `phase3-v1`
- **Temporal Scope**: **Strictly 2023–2025** (`2023-W01` to `2025-W52`, 156 consecutive 7-day reporting weeks).
- **Entrypoint**: `python -m src.features.feature_pipeline`
- **Models Trained**: **None** (strictly feature engineering and validation per Phase 3 instructions).

## 1. Datasets Used & Generated

### Input Processed Datasets Used (2023–2025)
- `data/processed/cleaned_disease_data.csv` (`source_dataset == 'idsp_weekly'`, 570 cleaned outbreak records across 2023–2025, aggregated to **511** distinct `(district_std, disease_std, report_year, report_week)` reporting-period observations).
- `data/processed/cleaned_climate_data.csv` (39,456 daily district-climate observations from `2023-01-01` to `2025-12-31` across 36 districts, plus `data/raw/imdweb_5dgbAskl.nc` for Gadchiroli's adjacent interior land cell `19.75°N, 80.00°E`).

### Output Feature Datasets (`data/features/`)
1. `data/features/disease_climate_weekly_features.csv` (also saved as `disease_climate_weekly_features_2023_2025.csv`): **511 rows × 182 columns** — Every observed 2023–2025 `(district_std, disease_std, report_year, report_week)` outbreak period aligned with disease history, climate, temporal, and geographic features.
2. `data/features/district_week_climate_features.csv` (also saved as `district_week_climate_features_2023_2025.csv`): **5,616 rows × 114 columns** — Complete 36-district × 156-week (2023–2025) weekly climate, temporal, and geographic feature grid.

## 2. Summary of Engineered Features by Category

| Category | Count | Key Engineered Features |
|---|---|---|
| **Identifier** | 4 | `district_std`, `disease_std`, `disease_group`, `is_target_disease` |
| **Geographic** | 8 | `district_id`, `administrative_division`, `is_coastal_district`, `is_vidarbha`, `is_marathwada`, `district_lat`, … (+2 more) |
| **Temporal** | 24 | `report_year`, `report_week`, `period_idx`, `report_period_start`, `report_period_end`, `canonical_week_start`, … (+18 more) |
| **Metadata** | 5 | `report_period_source`, `distinct_outbreak_ids`, `has_duplicate_or_conflict_flag`, `source_record_ids`, `source_outbreak_ids` |
| **Target (Outcome)** | 5 | `target_cases`, `target_cases_unadjusted_sum`, `target_cases_max_single_outbreak`, `target_deaths`, `outbreak_events_in_period` |
| **Disease History** | 49 | `cases_lag_1p`, `cases_lag_2p`, `cases_lag_4p`, `cases_lag_8p`, `cases_lag_52p`, `cases_lag_53p`, … (+43 more) |
| **Demographic (Unsupported)** | 3 | `district_population`, `population_density`, `cases_per_100k_pop` |
| **Geographic / Burden** | 7 | `district_disease_prior_outbreaks_cum`, `district_disease_prior_cases_cum`, `district_disease_prior_deaths_cum`, `district_disease_prior_mean_cases`, `district_disease_prior_max_cases`, `district_all_diseases_prior_cases_cum`, … (+1 more) |
| **Climate** | 73 | `rain_mm_sum_t`, `rain_mm_strict_nearest_sum_t`, `rain_mm_mean_t`, `rain_mm_max_t`, `rainy_days_t`, `heavy_rain_days_t`, … (+67 more) |
| **Climate Metadata** | 4 | `rain_valid_days_in_week`, `rain_spatial_source`, `temp_valid_days_in_week`, `rh_valid_days_in_week` |

## 3. Target Disease Coverage in 2023–2025 (`disease_climate_weekly_features.csv`)

| Target Disease | 2023–2025 Weekly Outbreak Periods | Districts with Outbreaks | With Event Lag-1 (`prev_outbreak_cases_lag_1`) | With Calendar Lag-1w (`cases_lag_1p`) | With Rainfall Features | With Temp Features | With Pune RH Features |
|---|---|---|---|---|---|---|---|
| **Dengue** | 151 | 30 | 121 | 18 | 151 | 151 | 4 |
| **Malaria** | 39 | 12 | 27 | 4 | 39 | 39 | 0 |
| **Chikungunya** | 34 | 14 | 20 | 0 | 34 | 34 | 1 |
| **Acute Diarrhoeal Disease** | 110 | 32 | 78 | 7 | 109 | 109 | 1 |
| **Cholera** | 18 | 11 | 7 | 0 | 18 | 18 | 3 |
| **Leptospirosis** | 9 | 7 | 2 | 0 | 9 | 9 | 0 |
| **Heat-related illness** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## 4. Data Leakage Prevention & Validation Results

All **23 / 23** automated checks in `reports/feature_engineering/leakage_validation_report.csv` passed:

- **Strict Prediction Cutoff (`<= t - 1`)**: All disease lag features (`cases_lag_1p..53p`, `prev_outbreak_cases_lag_1..8`), rolling averages/stds (`cases_roll_mean_2p..8p`, `prev_outbreak_roll_mean_2..8`), absolute/percentage changes (`cases_abs_change_lag1_lag2`, `prev_outbreak_abs_change_1_2`), expanding seasonal baselines (`seasonal_dd_season_mean_cases_prev`, `seasonal_state_disease_season_mean_prev`), and expanding district burdens (`district_disease_prior_cases_cum`) apply `.shift(1)` prior to aggregation so period `t` outcomes are **never** included.
- **Synthetic Future Perturbation Proof**: Adding `+50,000` cases to all outbreak records at or after `2024-W27` (`period_idx >= 78`) produced **0.000000** difference across all 45 disease-history and district-burden input features on all rows `period_idx <= 78`.
- **Climate Perturbation Proof**: Adding `+500 mm` rainfall and `+15 °C` temperature to daily climate on or after `2024-07-01` produced **0.000000** difference across all weekly climate features for weeks ending `< 2024-07-01`.

## 5. Unsupported Features & Scientific Limitations

1. **Population & Population-Normalized Incidence (`district_population`, `population_density`, `cases_per_100k_pop`) — UNSUPPORTED**:
   - No district population or census dataset exists in the project workspace. Per Prompt 3 (*'Use only the existing project datasets. Do not download new datasets or invent missing records'* and *'Do not calculate population-normalized rates without a valid population denominator'*), these three columns are preserved in the schema as `NaN`.
2. **Heat-Related Illness — UNSUPPORTED (0 Records)**:
   - Zero records exist for `Heat-related illness` in the 2023–2025 IDSP dataset. No synthetic records were created.
3. **Mumbai Suburban Climate — UNSUPPORTED (No Coordinates in Workspace)**:
   - `Mumbai Suburban` has 3 outbreak rows in 2023–2025 (`Zika Virus`, `Food Poisoning`, `Acute Diarrhoeal Disease`), but no latitude/longitude coordinates exist for `Mumbai Suburban` in any project file. Its climate features remain `NaN` (`rain_spatial_source = 'none_no_coordinates'`).
4. **Relative Humidity Statewide Coverage — LIMITED TO PUNE (2024-W19 to 2025-W24)**:
   - `humid_tel_hr_cwprs_mh_2021_2025.csv` contains telemetry for a single station in `Pune` from `2024-05-11` to `2025-06-09` (57 reporting weeks). For all other 35 districts and outside that date window, `rh_mean_t` and its lags remain `NaN`.
5. **Gadchiroli Rainfall Grid Boundary Note**:
   - Gadchiroli's reference coordinate (`19.759°N, 80.162°E`) is closest to `(19.75°N, 80.25°E)`, which lies on the masked eastern border of `imdweb_5dgbAskl.nc`. We extracted Gadchiroli's immediate unmasked interior land cell `(19.75°N, 80.00°E, 0.162° away)` from `data/raw/imdweb_5dgbAskl.nc`, tagged `rain_spatial_source = 'adjacent_land_cell_19.75N_80.00E'`, and also retained `rain_mm_strict_nearest_sum_t` (`NaN`) for full transparency.
6. **Episodic Outbreak Reporting vs. Routine Surveillance**:
   - IDSP reports outbreak clusters rather than routine weekly zero-counts. Calendar-week lags (`cases_lag_1p..8p`) are `NaN` whenever the preceding calendar week had no reported outbreak (never zero-filled), while binary indicators (`outbreak_reported_lag_1p..8p`, `outbreak_weeks_count_prev_2p..8p`) and event-sequence lags (`prev_outbreak_cases_lag_1..8`) provide dense history without fabricating zeros.
