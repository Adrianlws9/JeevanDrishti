# JEEVAN-DRISHTI Phase 4: Chronological Split & Validation Report

All **8 / 8** automated split, alignment, class-balance, and anti-leakage checks passed (`status = PASS`).

## 1. Automated Validation Check Results

| Check Name | Status | Detail |
|---|---|---|
| `chronological_split_non_overlapping` | **PASS** | train=0..84 (356 rows, 69.67%), validation=85..103 (77 rows, 15.07%), test=104..155 (78 rows, 15.26%) |
| `zero_duplicate_district_disease_period_keys` | **PASS** | Duplicate (district_std, disease_std, period_idx) keys: forecasting=0, risk=0, anomaly=0 |
| `target_availability_verified` | **PASS** | target_cases_t=511/511 non-null; target_fatal_outbreak=511/511; target_severe_outbreak_p75=481/511 (>=5 train obs guard); target_is_statistical_anomaly_2sigma=481/511 |
| `class_distribution_both_classes_in_every_split` | **PASS** | Both class 0 and class 1 are present in Train, Validation, and Test for severe_p75, p50, fatal_outbreak, next_4w_recurrence, and 2-sigma/p90 anomaly targets |
| `zero_cross_split_forward_target_leakage` | **PASS** | Forward multi-step targets in Train stay <= period_idx 84 (<=84) and in Validation stay <= 101 (<=103) |
| `no_contemporaneous_weather_leakage_in_forecasting_or_risk` | **PASS** | 0 contemporaneous week-t weather columns in forecasting_dataset.csv or risk_classification_dataset.csv; all climate predictors are strictly lagged (<= t-1) |
| `all_6_active_core_diseases_in_train_val_test` | **PASS** | All 6 active core target diseases (['Acute Diarrhoeal Disease', 'Chikungunya', 'Cholera', 'Dengue', 'Leptospirosis', 'Malaria']) are present in Train, Validation, and Test |
| `missing_features_documented_and_never_fabricated` | **PASS** | Unsupported population columns remain NaN; missing lagged case counts remain NaN (never zero-filled) |

## 2. Class Distribution Across Chronological Splits (`risk_classification_dataset.csv`)

| Target | Split | Valid `N` | Class `0` | Class `1` | Positive Rate (%) |
|---|---|---|---|---|---|
| `target_severe_outbreak_p75` | `train` | 340 | 258 | 82 | 24.12% |
| `target_severe_outbreak_p75` | `validation` | 76 | 61 | 15 | 19.74% |
| `target_severe_outbreak_p75` | `test` | 65 | 56 | 9 | 13.85% |
| `target_above_median_outbreak_p50` | `train` | 340 | 177 | 163 | 47.94% |
| `target_above_median_outbreak_p50` | `validation` | 76 | 48 | 28 | 36.84% |
| `target_above_median_outbreak_p50` | `test` | 65 | 35 | 30 | 46.15% |
| `target_fatal_outbreak` | `train` | 356 | 242 | 114 | 32.02% |
| `target_fatal_outbreak` | `validation` | 77 | 58 | 19 | 24.68% |
| `target_fatal_outbreak` | `test` | 78 | 65 | 13 | 16.67% |
| `target_multi_cluster_outbreak` | `train` | 356 | 317 | 39 | 10.96% |
| `target_multi_cluster_outbreak` | `validation` | 77 | 74 | 3 | 3.9% |
| `target_multi_cluster_outbreak` | `test` | 78 | 74 | 4 | 5.13% |
| `target_next_4w_recurrence` | `train` | 332 | 253 | 79 | 23.8% |
| `target_next_4w_recurrence` | `validation` | 67 | 58 | 9 | 13.43% |
| `target_next_4w_recurrence` | `test` | 77 | 69 | 8 | 10.39% |

## 3. Anomaly Distribution Across Chronological Splits (`anomaly_detection_dataset.csv`)

| Anomaly Target | Split | Valid `N` | Normal (`0`) | Anomaly (`1`) | Anomaly Rate (%) |
|---|---|---|---|---|---|
| `target_is_statistical_anomaly_2sigma` | `train` | 340 | 321 | 19 | 5.59% |
| `target_is_statistical_anomaly_2sigma` | `validation` | 76 | 75 | 1 | 1.32% |
| `target_is_statistical_anomaly_2sigma` | `test` | 65 | 60 | 5 | 7.69% |
| `target_is_extreme_anomaly_p90` | `train` | 340 | 304 | 36 | 10.59% |
| `target_is_extreme_anomaly_p90` | `validation` | 76 | 73 | 3 | 3.95% |
| `target_is_extreme_anomaly_p90` | `test` | 65 | 58 | 7 | 10.77% |
| `target_is_expanding_anomaly_2sigma` | `train` | 248 | 223 | 25 | 10.08% |
| `target_is_expanding_anomaly_2sigma` | `validation` | 67 | 61 | 6 | 8.96% |
| `target_is_expanding_anomaly_2sigma` | `test` | 62 | 56 | 6 | 9.68% |
