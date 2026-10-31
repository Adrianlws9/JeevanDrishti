# JEEVAN-DRISHTI Phase 12: Disease Severity & Mortality Intelligence Report

> **Engine Version**: `phase12-v1.0.0` | **Audit Date**: `2026-10-31`
> **Mandatory Privacy & Clinical Safety Notice**: AGGREGATE PUBLIC-HEALTH SURVEILLANCE ONLY — NO INDIVIDUAL PATIENT DATA OR PROGNOSIS. All mortality and case-fatality indicators are derived strictly from aggregate district-level IDSP outbreak cluster reports (2023–2025). Clinical severity counts, inpatient admissions, complications, and recovery counts do not exist in the workspace and are explicitly NaN. This engine never processes personally identifiable information (PII) and must never be used for individual clinical diagnosis, prognosis, or treatment decisions.

---

## 1. Outcome & Severity Data Availability Assessment

The existing **JEEVAN-DRISHTI** workspace was audited across all **7 required outcome and severity categories**:

| Outcome Category Key | Label | Available | Coverage (`2023–2025`) | Non-Null Rows | Missing (`NaN`) Rows | Exact Zero (`0`) Rows | Positive (`>0`) Rows | Engine Handling Mode |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| `disease_related_deaths` | Disease-Related Deaths (Aggregate Outbreak Mortality) | `True` | 34 / 36 active districts (2 districts have 0 outbreaks in 2023-2025) | 511 | 0 | 365 | 146 | `VERIFIED_AGGREGATE_MORTALITY_TREND_AND_FORECAST_EVALUATION` |
| `hospital_admissions` | Hospital Admissions (Inpatient / IPD Counts) | `False` | 0 / 36 districts (0.0%) | 0 | 511 | 0 | 0 | `EXPLICIT_NAN_UNAVAILABLE` |
| `severe_disease_cases` | Severe / Critical Disease Case Counts (Clinical Grading) | `False` | 0 / 36 districts (0.0%) | 0 | 511 | 0 | 0 | `EXPLICIT_NAN_UNAVAILABLE` |
| `complications` | Clinical Complications (e.g., Shock, Organ Failure, Encephalopathy) | `False` | 0 / 36 districts (0.0%) | 0 | 511 | 0 | 0 | `EXPLICIT_NAN_UNAVAILABLE` |
| `recovery_outcomes` | Patient Recovery / Discharge Counts | `False` | 0 / 36 districts (0.0%) | 0 | 511 | 0 | 0 | `EXPLICIT_NAN_UNAVAILABLE` |
| `case_fatality_information` | Outbreak-Cluster Case Fatality Ratio (OC-CFR %) | `True` | 34 / 36 active districts | 511 | 0 | 365 | 146 | `COMPATIBLE_NUMERATOR_DENOMINATOR_WITH_WILSON_CI_AND_EB_SHRINKAGE` |
| `district_wise_mortality_records` | District-Wise Aggregate Mortality Records (Maharashtra 36 Districts) | `True` | 34 / 36 active districts (Mumbai City & Ratnagiri = Insufficient Data) | 511 | 0 | 365 | 146 | `VERIFIED_DISTRICT_AGGREGATION_WITH_INSUFFICIENT_DATA_FLAGS` |

### Missing (`NaN`) vs. Exact Zero (`0`) Mortality Audit Across Workspace Stages

| Dataset Stage | Source Dataset | Duplicate Status | Total Records | Verified Cases | Non-Null Deaths | Missing (`NaN`) Deaths (`%`) | Exact Zero (`0`) Deaths | Positive (`>0`) Deaths | Verified Deaths Sum | Handling Rule |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `cleaned_disease_data.csv` | `historical_epiclim` | `conflict` | 29 | 972 | 9 | `20 (68.97%)` | 4 | 5 | **6** | Excluded as duplicate/conflict |
| `cleaned_disease_data.csv` | `historical_epiclim` | `possible_duplicate` | 22 | 702 | 6 | `16 (72.73%)` | 4 | 2 | **2** | Excluded as duplicate/conflict |
| `cleaned_disease_data.csv` | `historical_epiclim` | `unique` | 1143 | 51707 | 329 | `814 (71.22%)` | 64 | 265 | **332** | Historical 2009-2022 reference only (71.2% missing deaths left as NaN; never pooled with 2023-2025) |
| `cleaned_disease_data.csv` | `idsp_weekly` | `conflict` | 5 | 135 | 5 | `0 (0.00%)` | 2 | 3 | **5** | Excluded as duplicate/conflict |
| `cleaned_disease_data.csv` | `idsp_weekly` | `possible_duplicate_distinct_id` | 13 | 49 | 13 | `0 (0.00%)` | 9 | 4 | **4** | Excluded as duplicate/conflict |
| `cleaned_disease_data.csv` | `idsp_weekly` | `unique` | 552 | 19019 | 552 | `0 (0.00%)` | 394 | 158 | **190** | Included in deduplicated 2023-2025 analysis |
| `disease_climate_weekly_features.csv (Canonical Phase 12 Input)` | `idsp_weekly_deduplicated_2023_2025` | `unique_district_disease_week` | 511 | 19172 | 511 | `0 (0.00%)` | 365 | 146 | **199** | Primary 2023-2025 analytical dataset: 100% non-null deaths (365 exact zeros, 146 positive, 199 total deaths) |

---

## 2. Disease-Wise Mortality & Outbreak-Cluster Case Fatality Ratio (`OC-CFR`) Analysis (`2023–2025`)

- **Methodological Definition**: Because IDSP records outbreak clusters rather than all asymptomatic/mild community infections, the ratio $\text{OC-CFR} = \frac{\text{Verified Outbreak Deaths}}{\text{Verified Outbreak Cases}} \times 100\%$ measures **Outbreak-Cluster Case Fatality Ratio**. It is computed **strictly where numerator and denominator come from the exact same deduplicated `2023–2025` outbreak records** (`verified_cases > 0` and `verified_deaths` non-null).
- **Small-Denominator Safeguard**: Exact **Wilson score 95% confidence intervals** and **Empirical Bayes shrinkage** ($M = 50$ pseudocounts toward the statewide outbreak CFR prior of `1.038%`) are computed alongside crude ratios so that tiny-denominator diseases ($N < 30$, such as `Rabies (human)` with $2/2$ deaths or `Japanese Encephalitis` with $1/3$ deaths) are never misinterpreted as stable population fatality rates.

### All 15 Diseases with $\ge 1$ Verified Outbreak Death in `2023–2025` (Plus Top Zero-Death High-Volume Diseases)

| Disease | Disease Group | Outbreak Weeks | Positive-Death Weeks | Verified Cases | Verified Deaths | Crude OC-CFR (`%`) | Wilson 95% CI (`%`) | Empirical Bayes Shrunk CFR (`%`) | Denominator Stability Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Dengue** | target | 151 | 70 (46.4%) | 2748 | **82** | `2.98%` | `[2.41%, 3.69%]` | `2.95%` | `HIGH_CONFIDENCE_DENOMINATOR_N_GE_100` |
| **Malaria** | target | 39 | 27 (69.2%) | 1584 | **40** | `2.53%` | `[1.86%, 3.42%]` | `2.48%` | `HIGH_CONFIDENCE_DENOMINATOR_N_GE_100` |
| **Acute Encephalitis Syndrome** | other | 3 | 2 (66.7%) | 67 | **26** | `38.81%` | `[28.05%, 50.78%]` | `22.67%` | `MODERATE_DENOMINATOR_N_30_TO_99` |
| **Acute Diarrhoeal Disease** | target | 110 | 11 (10.0%) | 7030 | **12** | `0.17%` | `[0.10%, 0.30%]` | `0.18%` | `HIGH_CONFIDENCE_DENOMINATOR_N_GE_100` |
| **Fever (unspecified)** | other | 14 | 8 (57.1%) | 341 | **9** | `2.64%` | `[1.39%, 4.94%]` | `2.43%` | `HIGH_CONFIDENCE_DENOMINATOR_N_GE_100` |
| **Scrub Typhus** | other | 9 | 7 (77.8%) | 82 | **7** | `8.54%` | `[4.20%, 16.59%]` | `5.70%` | `MODERATE_DENOMINATOR_N_30_TO_99` |
| **Leptospirosis** | target | 9 | 6 (66.7%) | 67 | **6** | `8.96%` | `[4.17%, 18.19%]` | `5.57%` | `MODERATE_DENOMINATOR_N_30_TO_99` |
| **Food Poisoning** | other | 50 | 4 (8.0%) | 4794 | **4** | `0.08%` | `[0.03%, 0.21%]` | `0.09%` | `HIGH_CONFIDENCE_DENOMINATOR_N_GE_100` |
| **Cholera** | target | 18 | 2 (11.1%) | 486 | **4** | `0.82%` | `[0.32%, 2.10%]` | `0.84%` | `HIGH_CONFIDENCE_DENOMINATOR_N_GE_100` |
| **Hepatitis A** | other | 24 | 3 (12.5%) | 618 | **3** | `0.49%` | `[0.17%, 1.42%]` | `0.53%` | `HIGH_CONFIDENCE_DENOMINATOR_N_GE_100` |
| **Rabies (human)** | other | 2 | 2 (100.0%) | 2 | **2** | `100.00%` | `[34.24%, 100.00%]` | `4.84%` | `SMALL_DENOMINATOR_UNSTABLE_RATIO_N_LT_30` |
| **Dengue + Chikungunya** | combined | 6 | 1 (16.7%) | 100 | **1** | `1.00%` | `[0.18%, 5.45%]` | `1.01%` | `HIGH_CONFIDENCE_DENOMINATOR_N_GE_100` |
| **Dengue + Malaria** | combined | 1 | 1 (100.0%) | 64 | **1** | `1.56%` | `[0.28%, 8.33%]` | `1.33%` | `MODERATE_DENOMINATOR_N_30_TO_99` |
| **Japanese Encephalitis** | other | 1 | 1 (100.0%) | 3 | **1** | `33.33%` | `[6.15%, 79.23%]` | `2.87%` | `SMALL_DENOMINATOR_UNSTABLE_RATIO_N_LT_30` |
| **Fever of Unknown Cause** | other | 1 | 1 (100.0%) | 2 | **1** | `50.00%` | `[9.45%, 90.55%]` | `2.92%` | `SMALL_DENOMINATOR_UNSTABLE_RATIO_N_LT_30` |
| **Chikungunya** | target | 34 | 0 (0.0%) | 567 | **0** | `0.00%` | `[0.00%, 0.67%]` | `0.08%` | `HIGH_CONFIDENCE_DENOMINATOR_N_GE_100` |
| **Measles** | other | 9 | 0 (0.0%) | 174 | **0** | `0.00%` | `[0.00%, 2.16%]` | `0.23%` | `HIGH_CONFIDENCE_DENOMINATOR_N_GE_100` |
| **Hepatitis A + E** | combined | 1 | 0 (0.0%) | 101 | **0** | `0.00%` | `[0.00%, 3.66%]` | `0.34%` | `HIGH_CONFIDENCE_DENOMINATOR_N_GE_100` |
| **Dysentery** | other | 1 | 0 (0.0%) | 98 | **0** | `0.00%` | `[0.00%, 3.77%]` | `0.35%` | `MODERATE_DENOMINATOR_N_30_TO_99` |

---

## 3. District-Wise Mortality & Severity Burden Comparison (`36` Official Maharashtra Districts)

- **Active Surveillance Coverage**: `34 / 36` districts have verified `2023–2025` outbreak mortality records (`22` districts recorded $\ge 1$ death; `12` active districts recorded `0` deaths across all reported outbreaks; `2` districts — `Mumbai City` and `Ratnagiri` — had `0` reported outbreaks in `2023–2025` and are explicitly marked `Insufficient Data`).

### Top 15 Maharashtra Districts by Total Verified Outbreak Deaths (`2023–2025`)

| Rank | District ID | District | Division | Outbreak Weeks | Positive-Death Weeks | Verified Cases | Verified Deaths | Crude OC-CFR (`%`) | Wilson 95% CI (`%`) | EB Shrunk CFR (`%`) | Dominant Fatal Disease |
| :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | `MH_D11` | **Gadchiroli** | Nagpur_Vidarbha_East | 27 | 19 | 1432 | **33** | `2.30%` | `[1.65%, 3.22%]` | `2.26%` | Malaria (27) |
| 2 | `MH_D20` | **Nagpur** | Nagpur_Vidarbha_East | 18 | 9 | 775 | **33** | `4.26%` | `[3.05%, 5.92%]` | `4.06%` | Acute Encephalitis Syndrome (25) |
| 3 | `MH_D33` | **Thane** | Konkan | 34 | 21 | 588 | **24** | `4.08%` | `[2.76%, 6.00%]` | `3.84%` | Dengue (13) |
| 4 | `MH_D27` | **Raigad** | Konkan | 24 | 14 | 527 | **14** | `2.66%` | `[1.59%, 4.41%]` | `2.52%` | Dengue (8) |
| 5 | `MH_D04` | **Beed** | Marathwada | 11 | 8 | 87 | **11** | `12.64%` | `[7.21%, 21.24%]` | `8.41%` | Dengue (10) |
| 6 | `MH_D03` | **Amravati** | Amravati_Vidarbha_West | 24 | 7 | 954 | **8** | `0.84%` | `[0.43%, 1.65%]` | `0.85%` | Acute Diarrhoeal Disease (3) |
| 7 | `MH_D08` | **Chhatrapati Sambhajinagar** | Marathwada | 14 | 7 | 602 | **8** | `1.33%` | `[0.67%, 2.60%]` | `1.31%` | Fever (unspecified) (4) |
| 8 | `MH_D07` | **Chandrapur** | Nagpur_Vidarbha_East | 12 | 8 | 490 | **8** | `1.63%` | `[0.83%, 3.19%]` | `1.58%` | Malaria (4) |
| 9 | `MH_D09` | **Dharashiv** | Marathwada | 25 | 6 | 551 | **6** | `1.09%` | `[0.50%, 2.36%]` | `1.08%` | Dengue (4) |
| 10 | `MH_D24` | **Palghar** | Konkan | 13 | 5 | 421 | **5** | `1.19%` | `[0.51%, 2.75%]` | `1.17%` | Dengue (3) |
| 11 | `MH_D23` | **Nashik** | Nashik | 10 | 4 | 362 | **5** | `1.38%` | `[0.59%, 3.19%]` | `1.34%` | Dengue (4) |
| 12 | `MH_D25` | **Parbhani** | Marathwada | 12 | 3 | 1309 | **4** | `0.31%` | `[0.12%, 0.78%]` | `0.33%` | Dengue (4) |
| 13 | `MH_D14` | **Jalgaon** | Nashik | 13 | 2 | 492 | **4** | `0.81%` | `[0.32%, 2.07%]` | `0.83%` | Cholera (3) |
| 14 | `MH_D34` | **Wardha** | Nagpur_Vidarbha_East | 9 | 4 | 177 | **4** | `2.26%` | `[0.88%, 5.67%]` | `1.99%` | Dengue (2) |
| 15 | `MH_D15` | **Jalna** | Marathwada | 6 | 3 | 102 | **4** | `3.92%` | `[1.54%, 9.65%]` | `2.97%` | Dengue (4) |

---

## 4. Temporal (`2023–2025`) & Seasonal Mortality Trends

### A. Annual Statewide Mortality Progression (`ALL_DISEASES`)

| Reporting Period | Active Outbreak Weeks | Weeks with $\ge 1$ Death | Verified Cases | Verified Deaths | Pooled OC-CFR (`%`) | Wilson 95% CI (`%`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **2023_ANNUAL** | 175 | 78 | 4476 | **89** | `1.99%` | `[1.62%, 2.44%]` |
| **2024_ANNUAL** | 258 | 55 | 11291 | **65** | `0.58%` | `[0.45%, 0.73%]` |
| **2025_ANNUAL** | 78 | 13 | 3405 | **45** | `1.32%` | `[0.99%, 1.76%]` |

### B. Seasonal Mortality Distribution (Kruskal-Wallis $H = 21.275$, $p = 9.2e-05$)

| Season | Outbreak Weeks | Positive-Death Weeks (`%`) | Verified Cases | Verified Deaths | Share of Statewide Deaths (`%`) | Pooled OC-CFR (`%`) | Mean Deaths / Outbreak Week |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Southwest_Monsoon** | 238 | 78 (32.8%) | 6717 | **124** | `62.3%` | `1.85%` | `0.521` |
| **Post-Monsoon** | 111 | 43 (38.7%) | 3205 | **47** | `23.6%` | `1.47%` | `0.423` |
| **Pre-Monsoon_Summer** | 97 | 14 (14.4%) | 3734 | **15** | `7.5%` | `0.40%` | `0.155` |
| **Winter** | 65 | 11 (16.9%) | 5516 | **13** | `6.5%` | `0.24%` | `0.200` |

---

## 5. Mortality Time-Series Forecasting Sufficiency & Chronological Evaluation (Section 4)

### A. Data Sufficiency Across Analytical Grains

| Analytical Grain | Active Time Points | Positive-Death Time Points | Zero-Death (`%`) | Total Verified Deaths | Forecasting Supported | Recommended Analytical Mode |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `district_x_disease_x_week` | 511 | 146 | `71.43%` | 199 | `False` | `DESCRIPTIVE_TREND_AND_MATCHED_OC_CFR_ONLY` |
| `district_x_week_pooled` | 468 | 141 | `69.20%` | 199 | `False` | `DESCRIPTIVE_DISTRICT_COMPARISON_AND_EMPIRICAL_BAYES_CFR` |
| `statewide_weekly_other_individual_diseases` | 33 | 26 | `21.21%` | 40 | `False` | `DESCRIPTIVE_TREND_AND_SEASONAL_COMPARISON_ONLY` |
| `statewide_weekly_dengue` | 66 | 36 | `45.45%` | 82 | `True` | `CHRONOLOGICAL_FORECAST_EVALUATION_AGAINST_SIMPLE_BASELINES` |
| `statewide_weekly_all_diseases` | 121 | 71 | `41.32%` | 199 | `True` | `CHRONOLOGICAL_FORECAST_EVALUATION_AGAINST_SIMPLE_BASELINES` |

### B. Chronological Forecasting Validation (`70%` Train / `15%` Validation / `15%` Test — Strictly Lagged $t-1..t-4$ Predictors)

| Target Series | Model Name | Split | Weeks ($N$) | Observed Deaths | Predicted Deaths | MAE | RMSE | 80% PI Coverage (`%`) | MAE Skill vs. Last-Value Naive (`%`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `statewide_weekly_all_diseases` | **Zero_Mortality_Baseline** | `validation` | 18 | 10 | 0.0 | **0.5556** | 1.0000 | `100.0%` | `+41.18%` |
| `statewide_weekly_all_diseases` | **Zero_Mortality_Baseline** | `test` | 17 | 39 | 0.0 | **2.2941** | 6.5034 | `82.3%` | `+45.83%` |
| `statewide_weekly_all_diseases` | **Last_Value_Naive_Baseline** | `validation` | 18 | 10 | 11.0 | **0.9444** | 1.3540 | `100.0%` | `+0.00%` |
| `statewide_weekly_all_diseases` | **Last_Value_Naive_Baseline** | `test` | 17 | 39 | 39.0 | **4.2353** | 9.1201 | `70.6%` | `+0.00%` |
| `statewide_weekly_all_diseases` | **Rolling_4Week_Mean_Baseline** | `validation` | 18 | 10 | 12.0 | **0.7500** | 0.9391 | `94.4%` | `+20.59%` |
| `statewide_weekly_all_diseases` | **Rolling_4Week_Mean_Baseline** | `test` | 17 | 39 | 39.8 | **4.0735** | 7.0916 | `64.7%` | `+3.82%` |
| `statewide_weekly_all_diseases` | **Seasonal_Quarter_Naive_Baseline** | `validation` | 18 | 10 | 20.1 | **0.9689** | 1.1977 | `83.3%` | `-2.59%` |
| `statewide_weekly_all_diseases` | **Seasonal_Quarter_Naive_Baseline** | `test` | 17 | 39 | 35.5 | **2.8636** | 5.9171 | `52.9%` | `+32.39%` |
| `statewide_weekly_all_diseases` | **Poisson_Autoregressive_GLM** | `validation` | 18 | 10 | 19.2 | **0.8414** | 0.9739 | `94.4%` | `+10.91%` |
| `statewide_weekly_all_diseases` | **Poisson_Autoregressive_GLM** | `test` | 17 | 39 | 48.4 | **4.0703** | 7.7681 | `52.9%` | `+3.90%` |
| `statewide_weekly_all_diseases` | **RandomForest_Lagged_Count_Regressor** | `validation` | 18 | 10 | 18.8 | **0.8326** | 1.0038 | `94.4%` | `+11.84%` |
| `statewide_weekly_all_diseases` | **RandomForest_Lagged_Count_Regressor** | `test` | 17 | 39 | 35.9 | **3.1842** | 5.9781 | `47.1%` | `+24.82%` |
| `statewide_weekly_dengue` | **Zero_Mortality_Baseline** | `validation` | 9 | 8 | 0.0 | **0.8889** | 1.4142 | `88.9%` | `+57.89%` |
| `statewide_weekly_dengue` | **Zero_Mortality_Baseline** | `test` | 10 | 6 | 0.0 | **0.6000** | 1.3416 | `90.0%` | `+50.00%` |
| `statewide_weekly_dengue` | **Last_Value_Naive_Baseline** | `validation` | 9 | 8 | 11.0 | **2.1111** | 2.2361 | `100.0%` | `+0.00%` |
| `statewide_weekly_dengue` | **Last_Value_Naive_Baseline** | `test` | 10 | 6 | 6.0 | **1.2000** | 1.8974 | `80.0%` | `+0.00%` |
| `statewide_weekly_dengue` | **Rolling_4Week_Mean_Baseline** | `validation` | 9 | 8 | 9.8 | **1.0278** | 1.1456 | `100.0%` | `+51.32%` |
| `statewide_weekly_dengue` | **Rolling_4Week_Mean_Baseline** | `test` | 10 | 6 | 6.8 | **0.9250** | 1.3532 | `90.0%` | `+22.92%` |
| `statewide_weekly_dengue` | **Seasonal_Quarter_Naive_Baseline** | `validation` | 9 | 8 | 17.3 | **1.3077** | 1.5098 | `100.0%` | `+38.06%` |
| `statewide_weekly_dengue` | **Seasonal_Quarter_Naive_Baseline** | `test` | 10 | 6 | 24.4 | **2.2533** | 2.4523 | `30.0%` | `-87.78%` |
| `statewide_weekly_dengue` | **Poisson_Autoregressive_GLM** | `validation` | 9 | 8 | 12.3 | **1.0128** | 1.0846 | `100.0%` | `+52.02%` |
| `statewide_weekly_dengue` | **Poisson_Autoregressive_GLM** | `test` | 10 | 6 | 12.2 | **1.2051** | 1.4295 | `80.0%` | `-0.43%` |
| `statewide_weekly_dengue` | **RandomForest_Lagged_Count_Regressor** | `validation` | 9 | 8 | 16.7 | **1.5007** | 1.7464 | `66.7%` | `+28.91%` |
| `statewide_weekly_dengue` | **RandomForest_Lagged_Count_Regressor** | `test` | 10 | 6 | 15.8 | **1.4114** | 1.7513 | `50.0%` | `-17.62%` |

---

## 6. Privacy, Clinical Safety & Limitations Summary

1. **Aggregate Data Only**: Every record represents an aggregate district-week or statewide-week IDSP outbreak summary. Zero personally identifiable patient attributes exist in or are emitted by this engine.
2. **No Individual Clinical Prognosis**: Outbreak-Cluster CFR (`OC-CFR %`) reflects historical fatality among reported outbreak clusters and must **never** be interpreted as an individual patient's probability of death or used to guide clinical triage/treatment.
3. **Explicit Unavailable Outcome Preservation**: Hospital admissions, clinical severity counts (`severity_count`), complications, and recovery counts are `100%` absent from the workspace and remain explicitly `NaN` across all outputs.
