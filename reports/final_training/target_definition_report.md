# JEEVAN-DRISHTI Phase 4: Target Definition Report

All target variables across the three modelling datasets are derived strictly from observed IDSP surveillance outcomes and Training-split (`period_idx <= 84`, `2023-W01..2024-W33`) empirical distributions. **Zero arbitrary risk labels or synthetic thresholds were created.**

## 1. Forecasting Targets (`forecasting_dataset.csv`)

| Target Column | Mathematical Definition | Horizon & Cutoff | Valid Rows |
|---|---|---|---|
| `target_cases_t` | Deduplicated outbreak cases reported in `(district_std, disease_std, report_year, report_week)` | Target period `t` predicted from features strictly at `<= t - 1` | 511 / 511 |
| `target_log1p_cases_t` | `log(1 + target_cases_t)` (variance-stabilized count target) | Target period `t` predicted from `<= t - 1` | 511 / 511 |
| `target_next_outbreak_cases` | `target_cases` of the next outbreak event (`e + 1`) for `(district_std, disease_std)` within the same split (masked to `NaN` across split boundaries) | Event `e + 1` (`> t`) | 249 / 511 |
| `target_cases_calendar_t_plus_1w` | `target_cases` at exact calendar week `t + 1` (`NaN` if no outbreak in `t + 1` or across split boundary) | Calendar week `t + 1` | 38 / 511 |

## 2. Outbreak-Risk Classification Targets (`risk_classification_dataset.csv`)

Why arbitrary labels were rejected: fixed case cutoffs (e.g., `cases > 50`) are epidemiologically invalid across heterogeneous diseases (e.g., Cholera vs. Acute Diarrhoeal Disease). Instead, five defensible empirical targets are provided:

| Target Column | Definition | Justification | Valid Rows | Positive Rate (Overall) |
|---|---|---|---|---|
| `target_severe_outbreak_p75` | `1` if `target_cases > train_disease_p75_threshold` else `0` (`NaN` if `< 5` train obs) | Standard WHO/IDSP upper-quartile (75th percentile) disease-specific severity exceedance fitted on `train` only | 481 | 22.0% |
| `target_above_median_outbreak_p50` | `1` if `target_cases > train_disease_p50_threshold` else `0` (`NaN` if `< 5` train obs) | Disease-specific median severity exceedance fitted on `train` only | 481 | 45.9% |
| `target_fatal_outbreak` | `1` if `target_deaths >= 1` else `0` | Directly observed mortality occurrence in IDSP (`reported_deaths >= 1`) | 511 | 28.6% |
| `target_multi_cluster_outbreak` | `1` if `outbreak_events_in_period >= 2` else `0` | Directly observed multiple concurrent outbreak clusters in the same district-week | 511 | 9.0% |
| `target_next_4w_recurrence` | `1` if another outbreak of `(district_std, disease_std)` occurs in `[t+1, t+4]` within split | Empirical 4-week outbreak recurrence / persistence indicator | 476 | 20.2% |

### Training-Split Disease Threshold Table (Fitted Exclusively on `split == 'train'`, `period_idx <= 84`)

| Disease | Train Outbreaks (`n`) | Train Median (`p50`) | Train 75th Percentile (`p75`) | Train 90th Percentile (`p90`) | Train Mean Cases |
|---|---|---|---|---|---|
| Dengue | 134 | 10.0 | 24.0 | 40.0 | 17.96 |
| Acute Diarrhoeal Disease | 67 | 42.0 | 90.0 | 140.0 | 74.76 |
| Malaria | 29 | 12.0 | 37.0 | 64.0 | 47.14 |
| Food Poisoning | 23 | 48.0 | 78.0 | 187.6 | 103.00 |
| Chikungunya | 20 | 10.5 | 18.2 | 29.2 | 19.70 |
| Zika Virus | 15 | 3.0 | 7.0 | 10.4 | 5.33 |
| Cholera | 13 | 18.0 | 50.0 | 77.8 | 31.46 |
| Fever (unspecified) | 11 | 14.0 | 21.5 | 69.0 | 27.27 |
| Hepatitis A | 9 | 22.0 | 42.0 | 67.2 | 33.33 |
| Scrub Typhus | 8 | 10.0 | 12.5 | 16.1 | 10.12 |
| Dengue + Chikungunya | 6 | 13.0 | 20.2 | 29.0 | 16.67 |
| Leptospirosis | 5 | 7.0 | 7.0 | 14.8 | 8.20 |
| Measles | 4 | NaN (<5 obs) | NaN (<5 obs) | NaN (<5 obs) | 9.25 |
| Mumps | 2 | NaN (<5 obs) | NaN (<5 obs) | NaN (<5 obs) | 6.00 |
| Chickenpox | 1 | NaN (<5 obs) | NaN (<5 obs) | NaN (<5 obs) | 18.00 |
| Acute Encephalitis Syndrome | 1 | NaN (<5 obs) | NaN (<5 obs) | NaN (<5 obs) | 21.00 |
| Acute Hepatitis (unspecified) | 1 | NaN (<5 obs) | NaN (<5 obs) | NaN (<5 obs) | 24.00 |
| Hepatitis A + E | 1 | NaN (<5 obs) | NaN (<5 obs) | NaN (<5 obs) | 101.00 |
| Dysentery | 1 | NaN (<5 obs) | NaN (<5 obs) | NaN (<5 obs) | 98.00 |
| Fever of Unknown Cause | 1 | NaN (<5 obs) | NaN (<5 obs) | NaN (<5 obs) | 2.00 |
| Japanese Encephalitis | 1 | NaN (<5 obs) | NaN (<5 obs) | NaN (<5 obs) | 3.00 |
| Measles + Rubella | 1 | NaN (<5 obs) | NaN (<5 obs) | NaN (<5 obs) | 8.00 |
| Pertussis | 1 | NaN (<5 obs) | NaN (<5 obs) | NaN (<5 obs) | 9.00 |
| Typhoid | 1 | NaN (<5 obs) | NaN (<5 obs) | NaN (<5 obs) | 14.00 |

## 3. Anomaly Detection Targets (`anomaly_detection_dataset.csv`)

| Target Column | Formula / Definition | Baseline Used | Valid Rows | Anomaly Rate |
|---|---|---|---|---|
| `target_anomaly_zscore_vs_train_baseline` | `(target_cases_t - baseline_train_ref_mean) / baseline_train_ref_std` | Training-split (`<= 2024-W33`) `(disease_std, season)` mean & std | 481 | Continuous Z-score |
| `target_is_statistical_anomaly_2sigma` | `1` if `target_anomaly_zscore_vs_train_baseline > 2.0` else `0` | Standard WHO/CDC `mean + 2*SD` epidemic anomaly threshold (train-fitted) | 481 | 5.2% |
| `target_is_extreme_anomaly_p90` | `1` if `target_cases_t > train_baseline_disease_p90` else `0` | Training-split 90th percentile (`p90`) per disease | 481 | 9.6% |
| `target_is_expanding_anomaly_2sigma` | `1` if `target_anomaly_zscore_vs_expanding_prev > 2.0` else `0` | Strictly prior (`< t`) online expanding `(disease_std, season)` mean & std | 377 | 9.8% |
