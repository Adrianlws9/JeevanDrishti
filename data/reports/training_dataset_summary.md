# JEEVAN-DRISHTI Phase 4: Final Training Dataset Summary Report (2023–2025 Scope)

- **Dataset Version**: `phase4-v1`
- **Source Feature Dataset**: `data/features/disease_climate_weekly_features.csv` (`phase3-v1`, 511 weekly outbreak-reporting period rows across 2023–2025).
- **Models Trained**: **None** (strictly dataset preparation, target engineering, chronological splitting, and readiness verification).

## 1. Generated Modelling Datasets (`data/final_training/`)

| Dataset File | Prediction Task | Rows | Columns | Predictor Cutoff | Target Columns |
|---|---|---|---|---|---|
| `data/final_training/forecasting_dataset.csv` | Future Disease Case-Count Forecasting | 511 | 156 | Strictly `<= t - 1` (0 week-`t` weather cols) | `target_cases_t`, `target_log1p_cases_t`, `target_next_outbreak_cases`, `target_cases_calendar_t_plus_1w` |
| `data/final_training/risk_classification_dataset.csv` | Outbreak-Risk Classification (Defensible Empirical Labels) | 511 | 158 | Strictly `<= t - 1` (0 week-`t` weather cols) | `target_severe_outbreak_p75`, `target_above_median_outbreak_p50`, `target_fatal_outbreak`, `target_multi_cluster_outbreak`, `target_next_4w_recurrence` |
| `data/final_training/anomaly_detection_dataset.csv` | Historical-Baseline Anomaly Detection | 511 | 192 | Historical baselines fitted on `train` (`<= 2024-W33`) or expanding `< t` | `target_is_statistical_anomaly_2sigma`, `target_is_extreme_anomaly_p90`, `target_is_expanding_anomaly_2sigma`, `target_anomaly_zscore_vs_train_baseline` |

## 2. Chronological Split Summary (70% / 15% / 15%)

Splits are aligned strictly to weekly reporting boundaries (`period_idx`) so that no reporting week is ever split across partitions and the **entire year 2025 is preserved as the untouched final Test period**:

| Split | Reporting Weeks | Calendar Window | `period_idx` Range | Rows | Share (%) | Distinct Districts | Distinct Diseases |
|---|---|---|---|---|---|---|---|
| **Train** | `2023-W01 .. 2024-W33` | `2023-01-02 .. 2024-08-18` | `0 .. 84` | **356** | **69.67%** | 34 | 24 |
| **Validation** | `2024-W34 .. 2024-W52` | `2024-08-19 .. 2024-12-29` | `85 .. 103` | **77** | **15.07%** | 27 | 11 |
| **Test** | `2025-W01 .. 2025-W52` | `2025-01-01 .. 2025-12-28` | `104 .. 155` | **78** | **15.26%** | 24 | 17 |

## 3. Disease & Task Modelling Readiness Summary (`modelling_readiness_report.csv`)

| Disease | Core Target? | Total (2023–25) | Train | Val | Test | Districts | Forecasting Readiness | Risk Classification Readiness | Anomaly Detection Readiness |
|---|---|---|---|---|---|---|---|---|---|
| **Dengue** | Yes | 151 | 134 | 10 | 7 | 30 | `READY_STANDALONE_AND_POOLED` | `READY_STANDALONE_AND_POOLED` | `READY_STANDALONE_AND_POOLED` |
| **Acute Diarrhoeal Disease** | Yes | 110 | 67 | 23 | 20 | 32 | `READY_STANDALONE_AND_POOLED` | `READY_STANDALONE_AND_POOLED` | `READY_STANDALONE_AND_POOLED` |
| **Malaria** | Yes | 39 | 29 | 4 | 6 | 12 | `READY_STANDALONE_AND_POOLED` | `READY_STANDALONE_AND_POOLED` | `READY_STANDALONE_AND_POOLED` |
| **Chikungunya** | Yes | 34 | 20 | 9 | 5 | 14 | `READY_STANDALONE_AND_POOLED` | `READY_STANDALONE_AND_POOLED` | `READY_STANDALONE_AND_POOLED` |
| **Cholera** | Yes | 18 | 13 | 4 | 1 | 11 | `READY_POOLED_OR_SIMPLE_BASELINE` | `READY_POOLED_OR_EMPIRICAL_THRESHOLD` | `READY_STATISTICAL_BASELINE` |
| **Leptospirosis** | Yes | 9 | 5 | 3 | 1 | 7 | `READY_POOLED_OR_SIMPLE_BASELINE` | `READY_POOLED_OR_EMPIRICAL_THRESHOLD` | `READY_STATISTICAL_BASELINE` |
| **Heat-related illness** | Yes | 0 | 0 | 0 | 0 | 0 | `NOT_READY_NO_DATA` | `NOT_READY_NO_DATA` | `NOT_READY_NO_DATA` |
| **Food Poisoning** | No | 50 | 23 | 10 | 17 | 23 | `READY_STANDALONE_AND_POOLED` | `READY_STANDALONE_AND_POOLED` | `READY_STANDALONE_AND_POOLED` |
| **Hepatitis A** | No | 24 | 9 | 10 | 5 | 8 | `READY_POOLED_OR_SIMPLE_BASELINE` | `READY_POOLED_OR_EMPIRICAL_THRESHOLD` | `READY_STATISTICAL_BASELINE` |
| **Zika Virus** | No | 17 | 15 | 2 | 0 | 7 | `READY_POOLED_OR_SIMPLE_BASELINE` | `READY_POOLED_OR_EMPIRICAL_THRESHOLD` | `READY_STATISTICAL_BASELINE` |
| **Scrub Typhus** | No | 9 | 8 | 1 | 0 | 5 | `LIMITED_SAMPLE_POOLED_ONLY` | `LIMITED_SAMPLE_MORTALITY_OR_POOLED_ONLY` | `LIMITED_SAMPLE_RULE_BASED_ONLY` |
| **Measles** | No | 9 | 4 | 0 | 5 | 5 | `LIMITED_SAMPLE_POOLED_ONLY` | `LIMITED_SAMPLE_MORTALITY_OR_POOLED_ONLY` | `LIMITED_SAMPLE_RULE_BASED_ONLY` |

## 4. Key Readiness Conclusions for Phase 5 Model Training

1. **Ready for Standalone & Pooled Models**: **Dengue** (151 periods: 134 Train, 10 Val, 7 Test across 30 districts), **Acute Diarrhoeal Disease** (110 periods: 67 Train, 23 Val, 20 Test across 32 districts), and **Food Poisoning** (50 periods: 23 Train, 10 Val, 17 Test across 23 districts).
2. **Ready for Pooled Multi-Disease Models & Statistical/Empirical Baselines**: **Malaria** (39 periods: 29 Train, 4 Val, 6 Test), **Chikungunya** (34 periods: 20 Train, 9 Val, 5 Test), **Hepatitis A** (24 periods: 9 Train, 10 Val, 5 Test), **Cholera** (18 periods: 13 Train, 4 Val, 1 Test), and **Leptospirosis** (9 periods: 5 Train, 3 Val, 1 Test).
3. **Not Ready (No Data in Workspace)**: **Heat-related illness** (0 records in 2023–2025 IDSP dataset).
