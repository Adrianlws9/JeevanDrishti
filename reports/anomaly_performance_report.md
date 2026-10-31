# JEEVAN-DRISHTI Phase 7: Disease Anomaly Detection Performance & Validation Report

> **IMPORTANT EPIDEMIOLOGICAL NOTE**: An anomaly is an elevated signal requiring investigation, not proof of an outbreak.

- **Generated At**: `2026-10-31T15:03:15`
- **Model Version**: `phase7-v1.0.0`
- **Source Dataset**: `data/final_training/anomaly_detection_dataset.csv` (`511` weekly outbreak observations, 2023–2025)

## 1. Implemented Anomaly Detection Methods

| Method | Mathematical Formulation & Threshold | Training Cutoff |
|---|---|---|
| `Seasonal_Baseline_Deviation` | $Z_{\text{seasonal}} = (x_t - \mu_{\text{train}}(d, s)) / \sigma_{\text{train}}(d, s) > 2.0$ or off-season outbreak $> 1.5\times$ median | `split == 'train'` (`2023-W01..2024-W33`) |
| `Rolling_Statistical_Detector` | $Z_{\text{rolling}} = (x_t - \mu_{\text{roll,4p}}) / (\sigma_{\text{roll,4p}} \cdot \phi_{\text{gap}} \cdot \phi_{\text{coverage}}) > 2.0$ | Strictly prior ($< t$) expanding & 4-period rolling window |
| `Robust_MAD_ZScore` | $Z_{\text{robust}} = \max\big(0.6745(x_t - M)/\text{MAD},\; 0.6745(\ln(1+x_t) - M_{\ln})/\text{MAD}_{\ln}\big) > 2.5$ | `split == 'train'` (`2023-W01..2024-W33`) |
| `Isolation_Forest_Multivariate` | `IsolationForest(n_estimators=300, contamination=0.08, random_state=42)` over 14 epidemiological, mortality, climate & coverage features | `split == 'train'` (`2023-W01..2024-W33`) |
| `Composite_Consensus_Engine` | Weighted ensemble score $\ge 0.55$ OR $\ge 2$ methods flagged OR mortality anomaly (`deaths >= 2` or elevated CFR) | Calibrated on `train` + `validation`; `test` (`2025`) untouched |

## 2. Historical Validation, False-Alert Frequency & Detection Delay

**Why Precision & Recall Are Not Invented**: In accordance with strict epidemiological validation standards, verified external ground-truth anomaly labels do not exist in the IDSP line-list. Rather than fabricating precision/recall against an arbitrary label, we report:
1. **Alert Frequency (%)**: Share of outbreak-reporting weeks flagged by each detector.
2. **Estimated Benign False-Alert Frequency (%)**: Share of benign low-burden weeks (`cases <= seasonal_median`, `0 deaths`, single cluster) erroneously flagged, plus share of total alerts that are benign.
3. **Multi-Method Concordance (%)**: Share of flagged anomalies independently corroborated by $\ge 2$ detection methods.
4. **Historical Extreme-Event Concordance (%)**: Share of historical extreme benchmark events (`> train p90`, `> 2009-2022 p95`, `deaths >= 2`, or `>= 2` concurrent clusters) captured.
5. **Empirical Detection Delay (Days)**: Measured directly from verified `outbreak_start_date`, `last_reported_date`, and `report_period_end` timestamps (`327` records with exact dates).

| Method | Split | N | Flagged | Alert Freq (%) | Mean Score | False-Alert Rate in Benign (%) | Benign Share of Alerts (%) | Multi-Method Concordance (%) | Extreme-Event Capture (%) | Median Reporting Delay (Days) | Median Weekly Pub Lag (Days) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `Seasonal_Baseline_Deviation` | `train` | 356 | 20 | **5.6%** | 0.1827 | 0.0% | 0.0% | **100.0%** | **25.0%** | 1.0d | 4.0d |
| `Seasonal_Baseline_Deviation` | `validation` | 77 | 4 | **5.2%** | 0.1518 | 0.0% | 0.0% | **75.0%** | **12.5%** | 1.0d | 5.0d |
| `Seasonal_Baseline_Deviation` | `test` | 78 | 6 | **7.7%** | 0.1777 | 0.0% | 0.0% | **83.3%** | **36.4%** | 2.0d | 8.0d |
| `Seasonal_Baseline_Deviation` | `all_2023_2025` | 511 | 30 | **5.9%** | 0.1773 | 0.0% | 0.0% | **93.3%** | **25.3%** | 1.0d | 5.0d |
| `Rolling_Statistical_Detector` | `train` | 356 | 34 | **9.6%** | 0.2190 | 0.9% | 2.9% | **73.5%** | **29.4%** | 0.5d | 4.0d |
| `Rolling_Statistical_Detector` | `validation` | 77 | 7 | **9.1%** | 0.1815 | 0.0% | 0.0% | **57.1%** | **25.0%** | 1.0d | 4.5d |
| `Rolling_Statistical_Detector` | `test` | 78 | 8 | **10.3%** | 0.2341 | 0.0% | 0.0% | **87.5%** | **18.2%** | 0.0d | 4.5d |
| `Rolling_Statistical_Detector` | `all_2023_2025` | 511 | 49 | **9.6%** | 0.2157 | 0.6% | 2.0% | **73.5%** | **27.6%** | 0.0d | 4.0d |
| `Robust_MAD_ZScore` | `train` | 356 | 52 | **14.6%** | 0.2581 | 0.0% | 0.0% | **57.7%** | **50.0%** | 1.0d | 5.2d |
| `Robust_MAD_ZScore` | `validation` | 77 | 8 | **10.4%** | 0.2030 | 0.0% | 0.0% | **50.0%** | **37.5%** | 1.0d | 3.5d |
| `Robust_MAD_ZScore` | `test` | 78 | 11 | **14.1%** | 0.2441 | 0.0% | 0.0% | **81.8%** | **63.6%** | 1.5d | 8.0d |
| `Robust_MAD_ZScore` | `all_2023_2025` | 511 | 71 | **13.9%** | 0.2477 | 0.0% | 0.0% | **60.6%** | **50.6%** | 1.0d | 5.5d |
| `Isolation_Forest_Multivariate` | `train` | 356 | 29 | **8.2%** | 0.3609 | 1.8% | 6.9% | **58.6%** | **27.9%** | 0.3d | 5.2d |
| `Isolation_Forest_Multivariate` | `validation` | 77 | 8 | **10.4%** | 0.3974 | 0.0% | 0.0% | **50.0%** | **25.0%** | 1.0d | 5.0d |
| `Isolation_Forest_Multivariate` | `test` | 78 | 19 | **24.4%** | 0.4416 | 9.1% | 15.8% | **57.9%** | **54.5%** | 2.0d | 7.0d |
| `Isolation_Forest_Multivariate` | `all_2023_2025` | 511 | 56 | **11.0%** | 0.3788 | 2.8% | 8.9% | **57.1%** | **31.0%** | 1.0d | 5.2d |
| `Composite_Consensus_Engine` | `train` | 356 | 111 | **31.2%** | 0.3990 | 0.0% | 0.0% | **96.4%** | **60.3%** | 1.0d | 5.0d |
| `Composite_Consensus_Engine` | `validation` | 77 | 21 | **27.3%** | 0.3510 | 0.0% | 0.0% | **85.7%** | **75.0%** | 1.0d | 5.0d |
| `Composite_Consensus_Engine` | `test` | 78 | 22 | **28.2%** | 0.3346 | 0.0% | 0.0% | **95.5%** | **90.9%** | 1.2d | 6.0d |
| `Composite_Consensus_Engine` | `all_2023_2025` | 511 | 154 | **30.1%** | 0.3820 | 0.0% | 0.0% | **94.8%** | **65.5%** | 1.0d | 5.0d |

## 3. Summary of Detected Epidemiological Signal Types (`2023–2025`)

- **Sudden Disease Case Increases (`SUDDEN_CASE_INCREASE`)**: **87** signals.
- **Unusual District-Level Patterns (`UNUSUAL_DISTRICT_PATTERN`)**: **82** signals.
- **Unexpected Seasonal Deviations (`UNEXPECTED_SEASONAL_DEVIATION`)**: **74** signals.
- **Unusual Mortality Patterns (`UNUSUAL_MORTALITY_PATTERN`)**: **102** signals.

## 4. Accounting for Reporting Delays, Seasonality, Missing Data & Coverage Changes

- **Empirical Reporting Delay (`outbreak_start_date -> last_reported_date`, $n = 306$)**: Median **`1.0 days`**, Mean **`1.94 days`**, 90th Percentile **`4.0 days`**.
- **Weekly Bulletin Publication Lag (`outbreak_start_date -> report_period_end`)**: Median **`5.0 days`**, Mean **`7.69 days`**, 90th Percentile **`12.0 days`**.
- **Reporting Coverage Adjustments**: Active IDSP reporting coverage was `39/52` weeks (`75.0%`) in 2023, `52/52` weeks (`100.0%`) in 2024, and `30/52` weeks (`57.7%`) in 2025. Rolling variance denominators are scaled by $(\text{coverage})^{-1/2}$ and silence-gap inflation $(1 + 0.01\max(0, \Delta t - 4))$ to reduce false alerts after silent reporting gaps.
- **Missing Data Handling**: The `6` rows with missing `deaths` are preserved as `NaN` (`0` fabricated mortality alerts); the `16` rare diseases with `< 5` training observations use pooled `disease_group` baselines and are explicitly tagged in metadata.

## 5. Top 20 Highest-Scoring Flagged Anomalies Across Maharashtra (`2023–2025`)

| Rank | Split | Period | District | Disease | Cases | Deaths | Score | Tier | Signal Types | Triggering Features |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `train` | `2024-W06` | Nanded | Acute Diarrhoeal Disease | 1000 | 0 | **0.9063** | `HIGH_PRIORITY_INVESTIGATION_SIGNAL` | `SUDDEN_CASE_INCREASE;UNUSUAL_DISTRICT_PATTERN;UNEXPECTED_SEASONAL_DEVIATION` | cases=1000 vs rolling_baseline=36.0 (rolling_z=+59.27, velocity=27.78x) | district_ratio=3.63x prior mean, clusters_in_week=1 | seasonal_z=+2.79, robust_mad_z=+18.92 vs Winter median=46.5 |
| 2 | `validation` | `2024-W34` | Chhatrapati Sambhajinagar | Food Poisoning | 181 | 0 | **0.8931** | `HIGH_PRIORITY_INVESTIGATION_SIGNAL` | `SUDDEN_CASE_INCREASE;UNUSUAL_DISTRICT_PATTERN;UNEXPECTED_SEASONAL_DEVIATION` | cases=181 vs rolling_baseline=48.6 (rolling_z=+4.14, velocity=3.72x) | district_ratio=3.72x prior mean, clusters_in_week=1 | seasonal_z=+4.14, robust_mad_z=+6.79 vs Southwest_Monsoon median=40.0 |
| 3 | `train` | `2024-W06` | Parbhani | Food Poisoning | 983 | 0 | **0.8904** | `HIGH_PRIORITY_INVESTIGATION_SIGNAL` | `SUDDEN_CASE_INCREASE;UNUSUAL_DISTRICT_PATTERN;UNEXPECTED_SEASONAL_DEVIATION` | cases=983 vs rolling_baseline=54.0 (rolling_z=+20.37, velocity=18.20x) | district_ratio=1.90x prior mean, clusters_in_week=3 | seasonal_z=+2.43, robust_mad_z=+16.96 vs Winter median=78.0 |
| 4 | `test` | `2025-W38` | Raigad | Malaria | 98 | 0 | **0.8881** | `HIGH_PRIORITY_INVESTIGATION_SIGNAL` | `SUDDEN_CASE_INCREASE;UNUSUAL_DISTRICT_PATTERN;UNEXPECTED_SEASONAL_DEVIATION` | cases=98 vs rolling_baseline=3.5 (rolling_z=+2.83, velocity=28.00x) | district_ratio=28.00x prior mean, clusters_in_week=1 | seasonal_z=+3.91, robust_mad_z=+6.45 vs Southwest_Monsoon median=12.0 |
| 5 | `train` | `2024-W32` | Palghar | Acute Diarrhoeal Disease | 249 | 1 | **0.8865** | `HIGH_PRIORITY_INVESTIGATION_SIGNAL` | `SUDDEN_CASE_INCREASE;UNEXPECTED_SEASONAL_DEVIATION` | cases=249 vs rolling_baseline=15.0 (rolling_z=+5.54, velocity=16.60x) | seasonal_z=+3.51, robust_mad_z=+6.33 vs Southwest_Monsoon median=42.5 |
| 6 | `train` | `2023-W25` | Kolhapur | Dengue | 113 | 0 | **0.8855** | `HIGH_PRIORITY_INVESTIGATION_SIGNAL` | `SUDDEN_CASE_INCREASE;UNUSUAL_DISTRICT_PATTERN;UNEXPECTED_SEASONAL_DEVIATION` | cases=113 vs rolling_baseline=29.0 (rolling_z=+3.41, velocity=3.90x) | district_ratio=2.61x prior mean, clusters_in_week=2 | seasonal_z=+4.37, robust_mad_z=+11.52 vs Southwest_Monsoon median=10.5 |
| 7 | `test` | `2025-W41` | Kolhapur | Malaria | 30 | 0 | **0.8802** | `HIGH_PRIORITY_INVESTIGATION_SIGNAL` | `SUDDEN_CASE_INCREASE;UNUSUAL_DISTRICT_PATTERN;UNEXPECTED_SEASONAL_DEVIATION` | cases=30 vs rolling_baseline=7.0 (rolling_z=+4.03, velocity=4.29x) | district_ratio=3.75x prior mean, clusters_in_week=1 | seasonal_z=+3.86, robust_mad_z=+5.40 vs Post-Monsoon median=6.0 | rain_lag1w_z=+3503.16 |
| 8 | `train` | `2024-W26` | Raigad | Dengue | 86 | 1 | **0.8763** | `HIGH_PRIORITY_INVESTIGATION_SIGNAL` | `SUDDEN_CASE_INCREASE;UNUSUAL_DISTRICT_PATTERN;UNEXPECTED_SEASONAL_DEVIATION` | cases=86 vs rolling_baseline=33.0 (rolling_z=+3.99, velocity=2.61x) | district_ratio=5.01x prior mean, clusters_in_week=2 | seasonal_z=+3.11, robust_mad_z=+8.49 vs Southwest_Monsoon median=10.5 |
| 9 | `train` | `2024-W15` | Chandrapur | Food Poisoning | 258 | 1 | **0.8733** | `HIGH_PRIORITY_INVESTIGATION_SIGNAL` | `SUDDEN_CASE_INCREASE;UNUSUAL_DISTRICT_PATTERN;UNEXPECTED_SEASONAL_DEVIATION` | cases=258 vs rolling_baseline=36.0 (rolling_z=+15.30, velocity=7.17x) | district_ratio=4.61x prior mean, clusters_in_week=1 | seasonal_z=+2.79, robust_mad_z=+10.51 vs Pre-Monsoon_Summer median=32.0 |
| 10 | `train` | `2024-W12` | Amravati | Dengue | 64 | 0 | **0.8585** | `HIGH_PRIORITY_INVESTIGATION_SIGNAL` | `SUDDEN_CASE_INCREASE;UNUSUAL_DISTRICT_PATTERN;UNEXPECTED_SEASONAL_DEVIATION` | cases=64 vs rolling_baseline=13.7 (rolling_z=+3.92, velocity=4.68x) | district_ratio=2.44x prior mean, clusters_in_week=1 | seasonal_z=+3.27, robust_mad_z=+7.15 vs Pre-Monsoon_Summer median=11.0 |
| 11 | `train` | `2023-W33` | Thane | Leptospirosis | 1 | 1 | **0.8500** | `HIGH_PRIORITY_INVESTIGATION_SIGNAL` | `UNUSUAL_MORTALITY_PATTERN` | reported_deaths=1 (CFR=100.0%) |
| 12 | `train` | `2023-W37` | Raigad | Malaria | 4 | 1 | **0.8500** | `HIGH_PRIORITY_INVESTIGATION_SIGNAL` | `UNUSUAL_MORTALITY_PATTERN` | reported_deaths=1 (CFR=25.0%) |
| 13 | `train` | `2023-W37` | Beed | Fever (unspecified) | 5 | 1 | **0.8500** | `HIGH_PRIORITY_INVESTIGATION_SIGNAL` | `UNUSUAL_MORTALITY_PATTERN` | reported_deaths=1 (CFR=20.0%) |
| 14 | `train` | `2023-W37` | Beed | Dengue | 5 | 1 | **0.8500** | `HIGH_PRIORITY_INVESTIGATION_SIGNAL` | `UNUSUAL_MORTALITY_PATTERN` | reported_deaths=1 (CFR=20.0%) |
| 15 | `train` | `2023-W38` | Thane | Dengue | 12 | 2 | **0.8500** | `HIGH_PRIORITY_INVESTIGATION_SIGNAL` | `UNUSUAL_DISTRICT_PATTERN;UNUSUAL_MORTALITY_PATTERN` | district_ratio=1.26x prior mean, clusters_in_week=2 | reported_deaths=2 (CFR=16.7%) |
| 16 | `train` | `2023-W43` | Amravati | Dengue | 5 | 1 | **0.8500** | `HIGH_PRIORITY_INVESTIGATION_SIGNAL` | `UNUSUAL_MORTALITY_PATTERN` | reported_deaths=1 (CFR=20.0%) |
| 17 | `train` | `2023-W39` | Chhatrapati Sambhajinagar | Dengue | 3 | 1 | **0.8500** | `HIGH_PRIORITY_INVESTIGATION_SIGNAL` | `UNUSUAL_MORTALITY_PATTERN` | reported_deaths=1 (CFR=33.3%) |
| 18 | `train` | `2023-W01` | Thane | Dengue | 3 | 1 | **0.8500** | `HIGH_PRIORITY_INVESTIGATION_SIGNAL` | `UNUSUAL_MORTALITY_PATTERN` | reported_deaths=1 (CFR=33.3%) |
| 19 | `train` | `2023-W05` | Ahmednagar | Dengue | 2 | 1 | **0.8500** | `HIGH_PRIORITY_INVESTIGATION_SIGNAL` | `UNUSUAL_MORTALITY_PATTERN` | reported_deaths=1 (CFR=50.0%) |
| 20 | `train` | `2023-W22` | Gadchiroli | Malaria | 46 | 3 | **0.8500** | `HIGH_PRIORITY_INVESTIGATION_SIGNAL` | `SUDDEN_CASE_INCREASE;UNUSUAL_DISTRICT_PATTERN;UNEXPECTED_SEASONAL_DEVIATION;UNUSUAL_MORTALITY_PATTERN` | cases=46 vs rolling_baseline=19.8 (rolling_z=+1.13, velocity=2.32x) | district_ratio=0.56x prior mean, clusters_in_week=3 | seasonal_z=+1.31, robust_mad_z=+2.55 vs Southwest_Monsoon median=12.0 | reported_deaths=3 (CFR=6.5%) |
