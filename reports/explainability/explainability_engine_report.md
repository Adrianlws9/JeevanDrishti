# JEEVAN-DRISHTI Phase 13: Centralized Explainable AI (XAI) Engine Report

> **Engine Version**: `phase13-v1.0.0` | **Audit Date**: `2026-10-31`
> **Mandatory Scientific Disclaimer**: FEATURE CONTRIBUTION IS NOT CAUSATION. High feature importance, positive SHAP values, or regression coefficients indicate that the model utilized a statistical pattern in the training data to form its prediction or risk score; they do NOT prove that a climate or epidemiological variable biologically or environmentally caused an outbreak.

---

## 1. Existing Model Inventory & Explainability Suitability Matrix

Every model artifact across Phases 5–9 was inspected to match each estimator with its mathematically valid explainability method (never forcing SHAP onto statistical baselines or spatial Z-scores):

| Phase | Domain | Model Name | Version | Features | Primary | SHAP Suitable | Explainability Methods Applied |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Phase 5** | Outbreak Risk Classification | `HistGradientBoosting_Classifier` | `phase5-v1.0.0` | 223 | `True` | `False` | Permutation Feature Importance (Validation/Test Split) + Calibrated Probability Sensitivity |
| **Phase 5** | Outbreak Risk Classification | `XGBoost_Classifier` | `phase5-v1.0.0` | 223 | `False` | `True` | Exact TreeSHAP (XGBoost Booster pred_contribs=True) + Gain Importance |
| **Phase 5** | Outbreak Risk Classification | `Random_Forest_Classifier` | `phase5-v1.0.0` | 223 | `False` | `False` | Tree Impurity (Gini) Feature Importance |
| **Phase 5** | Outbreak Risk Classification | `Logistic_Regression_Baseline` | `phase5-v1.0.0` | 223 | `False` | `False` | Standardized Linear Coefficients (β_j), Odds Ratios (exp(β_j)), and Exact Linear Log-Odds Contributions (β_j * x_ij) |
| **Phase 6** | Time-Series Disease Forecasting (7d, 14d, 30d) | `HistGradientBoosting + XGBoost + RandomForest + Ridge (7d/14d/30d)` | `phase6-v1.0.0` | 224 | `True` | `True` | Exact TreeSHAP on XGBoost_Lagged_Regressor + Permutation Importance on HistGradientBoosting_Regressor + Ridge Standardized Coefficients + Conformal Residual & Domain Decomposition |
| **Phase 7** | Disease Anomaly Detection | `Seasonal Z + Rolling Z + Robust MAD Z + Isolation Forest Consensus` | `phase7-v1.0.0` | 14 | `True` | `False` | Explicit Baseline Deviation Evidence (Observed vs. Seasonal/Rolling/MAD Thresholds) + Isolation Forest Feature Perturbation Sensitivity (14 features) |
| **Phase 8** | Climate–Disease Intelligence | `Spearman/BH-FDR + Lagged CCF (0..8w) + RF/XGBoost TreeSHAP & PDP` | `phase8-v1.0.0` | 17 | `True` | `True` | Exact TreeSHAP + 1D Partial Dependence Curves (PDP) + Nonparametric Spearman & BH-FDR |
| **Phase 9** | Geospatial Hotspot Detection | `Global Moran's I & Local Getis-Ord Gi* (Queen Contiguity, 999 Permutations)` | `phase9-v1.0.0` | 4 | `True` | `False` | Spatial Lag & Local Gi* Z-score / Permutation P-Value Evidence Decomposition |

---

## 2. Global Explanations Across Supported Models

### A. Phase 5 Outbreak Risk Classifier (`phase5-v1.0.0`) — Top 10 Global Features

| Rank | Feature Name | Domain | XGBoost Mean `|TreeSHAP|` | Primary HGB Permutation `ΔPR-AUC` | RF Gini | LogReg Standardized $\beta_j$ | Direction of Influence |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| 1 | `district_lat` | `disease_and_district_context` | `0.1333` | `0.0074` | `0.0331` | `+0.0463` | `POSITIVE_HIGHER_VALUE_INCREASES_OUTPUT` |
| 2 | `temp_diurnal_range_lag_4w` | `rainfall_and_temperature_climate` | `0.0462` | `0.0253` | `0.0181` | `-0.1101` | `NEGATIVE_HIGHER_VALUE_DECREASES_OUTPUT` |
| 3 | `seasonal_state_disease_season_mean_prev` | `seasonality_and_calendar` | `0.0904` | `0.0032` | `0.0192` | `-0.1803` | `NEGATIVE_HIGHER_VALUE_DECREASES_OUTPUT` |
| 4 | `rain_mm_yoy_abs_diff_lag_1w` | `rainfall_and_temperature_climate` | `0.0368` | `0.0129` | `0.0151` | `+0.2956` | `POSITIVE_HIGHER_VALUE_INCREASES_OUTPUT` |
| 5 | `district_disease_prior_deaths_cum` | `historical_disease_lags_and_baselines` | `0.0749` | `0.0038` | `0.0114` | `-0.1646` | `NEGATIVE_HIGHER_VALUE_DECREASES_OUTPUT` |
| 6 | `prev_outbreak_roll_mean_2` | `historical_disease_lags_and_baselines` | `0.0676` | `0.0012` | `0.0190` | `+0.0639` | `POSITIVE_HIGHER_VALUE_INCREASES_OUTPUT` |
| 7 | `administrative_division_Marathwada` | `spatial_neighbor_and_geography` | `0.0699` | `0.0000` | `0.0139` | `-0.1137` | `NEGATIVE_HIGHER_VALUE_DECREASES_OUTPUT` |
| 8 | `tmin_c_mean_lag_8w` | `historical_disease_lags_and_baselines` | `0.0469` | `0.0000` | `0.0321` | `+0.0624` | `NEGATIVE_HIGHER_VALUE_DECREASES_OUTPUT` |
| 9 | `rain_mm_sum_lag_8w` | `rainfall_and_temperature_climate` | `0.0406` | `0.0025` | `0.0161` | `-0.1807` | `NEGATIVE_HIGHER_VALUE_DECREASES_OUTPUT` |
| 10 | `temp_diurnal_range_lag_1w` | `rainfall_and_temperature_climate` | `0.0417` | `0.0000` | `0.0231` | `+0.1529` | `POSITIVE_HIGHER_VALUE_INCREASES_OUTPUT` |

### B. Phase 6 Multi-Horizon Disease Forecasting (`phase6-v1.0.0`) — Domain Attribution Share (`%` of Mean `|TreeSHAP|`)

| Feature Domain | `7d` (`1w` Ahead) Share (`%`) | `14d` (`2w` Ahead) Share (`%`) | `30d` (`4w` Ahead) Share (`%`) |
| :--- | :---: | :---: | :---: |
| `disease_and_district_context` | `33.50%` | `36.15%` | `38.68%` |
| `historical_disease_lags_and_baselines` | `29.38%` | `27.60%` | `26.63%` |
| `rainfall_and_temperature_climate` | `17.73%` | `16.27%` | `12.59%` |
| `recent_case_growth_velocity` | `0.18%` | `0.07%` | `0.00%` |
| `seasonality_and_calendar` | `16.53%` | `17.67%` | `19.66%` |
| `spatial_neighbor_and_geography` | `2.68%` | `2.24%` | `2.44%` |

---

## 3. Local Risk, Forecast & Anomaly Explanations

### A. Top 10 Highest Calibrated Outbreak Risk Predictions (`local_risk_explanations.csv`)

| Prediction ID | District | Disease | Period | Calibrated Risk (`Tier`) | Historical Disease SHAP | Case Growth SHAP | Climate SHAP | Seasonal SHAP | Missing Inputs |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `RISK_MH_D03_2024-W11_207` | **Amravati** | Malaria | `2024-W11` | **0.308** (`High`) | `+0.313` | `+0.000` | `+0.135` | `+0.098` | `6` |
| `RISK_MH_D02_2025-W07_445` | **Akola** | Typhoid | `2025-W07` | **0.308** (`UNVALIDATED_LIMITED_SAMPLE`) | `+0.544` | `+0.000` | `+0.200` | `+0.104` | `20` |
| `RISK_MH_D03_2025-W07_446` | **Amravati** | Fever (unspecified) | `2025-W07` | **0.305** (`High`) | `+0.544` | `+0.000` | `+0.172` | `+0.096` | `20` |
| `RISK_MH_D14_2025-W09_455` | **Jalgaon** | Jaundice | `2025-W09` | **0.302** (`UNVALIDATED_LIMITED_SAMPLE`) | `+0.530` | `+0.000` | `+0.202` | `+0.090` | `20` |
| `RISK_MH_D01_2023-W09_012` | **Ahmednagar** | Fever (unspecified) | `2023-W09` | **0.295** (`High`) | `+0.585` | `+0.000` | `+0.237` | `+0.108` | `30` |
| `RISK_MH_D05_2025-W11_464` | **Bhandara** | Fever with Rash | `2025-W11` | **0.294** (`UNVALIDATED_LIMITED_SAMPLE`) | `+0.544` | `+0.000` | `+0.200` | `+0.104` | `20` |
| `RISK_MH_D01_2024-W16_232` | **Ahmednagar** | Scrub Typhus | `2024-W16` | **0.288** (`High`) | `+0.277` | `+0.000` | `+0.185` | `+0.108` | `20` |
| `RISK_MH_D14_2025-W04_439` | **Jalgaon** | Measles | `2025-W04` | **0.288** (`UNVALIDATED_LIMITED_SAMPLE`) | `+0.556` | `+0.000` | `+0.069` | `+0.045` | `18` |
| `RISK_MH_D02_2024-W13_220` | **Akola** | Hepatitis A | `2024-W13` | **0.287** (`High`) | `+0.373` | `+0.000` | `+0.120` | `+0.039` | `18` |
| `RISK_MH_D22_2024-W08_196` | **Nandurbar** | Acute Diarrhoeal Disease | `2024-W08` | **0.286** (`High`) | `+0.552` | `+0.000` | `+0.245` | `-0.294` | `18` |

### B. Top 10 Flagged Anomaly Evidence Explanations (`local_anomaly_explanations.csv`)

| Anomaly ID | District | Disease | Period | Observed Cases | Composite Score (`Tier`) | Seasonal $Z$ | Rolling $Z$ | Robust MAD $Z$ | Isolation Forest |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `ANOM_MH_D21_2024-W06_190` | **Nanded** | Acute Diarrhoeal Disease | `2024-W06` | **1000** | `0.906` (`HIGH_PRIORITY_INVESTIGATION_SIGNAL`) | `+2.79` | `+59.27` | `+18.92` | `0.750` |
| `ANOM_MH_D08_2024-W34_357` | **Chhatrapati Sambhajinagar** | Food Poisoning | `2024-W34` | **181** | `0.893` (`HIGH_PRIORITY_INVESTIGATION_SIGNAL`) | `+4.14` | `+4.14` | `+6.79` | `0.579` |
| `ANOM_MH_D25_2024-W06_191` | **Parbhani** | Food Poisoning | `2024-W06` | **983** | `0.890` (`HIGH_PRIORITY_INVESTIGATION_SIGNAL`) | `+2.43` | `+20.37` | `+16.96` | `0.760` |
| `ANOM_MH_D27_2025-W38_504` | **Raigad** | Malaria | `2025-W38` | **98** | `0.888` (`HIGH_PRIORITY_INVESTIGATION_SIGNAL`) | `+3.91` | `+2.83` | `+6.45` | `0.734` |
| `ANOM_MH_D24_2024-W32_351` | **Palghar** | Acute Diarrhoeal Disease | `2024-W32` | **249** | `0.886` (`HIGH_PRIORITY_INVESTIGATION_SIGNAL`) | `+3.51` | `+5.54` | `+6.33` | `0.565` |
| `ANOM_MH_D16_2023-W25_027` | **Kolhapur** | Dengue | `2023-W25` | **113** | `0.885` (`HIGH_PRIORITY_INVESTIGATION_SIGNAL`) | `+4.37` | `+3.41` | `+11.52` | `0.581` |
| `ANOM_MH_D16_2025-W41_508` | **Kolhapur** | Malaria | `2025-W41` | **30** | `0.880` (`HIGH_PRIORITY_INVESTIGATION_SIGNAL`) | `+3.86` | `+4.03` | `+5.40` | `0.577` |
| `ANOM_MH_D27_2024-W26_301` | **Raigad** | Dengue | `2024-W26` | **86** | `0.876` (`HIGH_PRIORITY_INVESTIGATION_SIGNAL`) | `+3.11` | `+3.99` | `+8.49` | `0.602` |
| `ANOM_MH_D07_2024-W15_223` | **Chandrapur** | Food Poisoning | `2024-W15` | **258** | `0.873` (`HIGH_PRIORITY_INVESTIGATION_SIGNAL`) | `+2.79` | `+15.30` | `+10.51` | `0.586` |
| `ANOM_MH_D03_2024-W12_217` | **Amravati** | Dengue | `2024-W12` | **64** | `0.859` (`HIGH_PRIORITY_INVESTIGATION_SIGNAL`) | `+3.27` | `+3.92` | `+7.15` | `0.500` |

---

## 4. Frontend Integration: `"Why was this district flagged?"` Panel (`36` Districts)

| District ID | District | Flag Priority Tier | Peak Risk Prob (`Category`) | Flagged Anomaly Weeks | Getis-Ord $G_i^*$ $Z$-Score (`Class`) | `7d` Forecast Mean Cases |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| `MH_D03` | **Amravati** | `MULTI_ENGINE_HIGH_PRIORITY_FLAG` | `0.308 (High)` | 8 | `+1.44 (Not a Statistically Significant Hotspot (p >= 0.05) (Cholera))` | 36.5 |
| `MH_D02` | **Akola** | `ELEVATED_SURVEILLANCE_FLAG` | `0.308 (UNVALIDATED_LIMITED_SAMPLE)` | 1 | `+1.44 (Not a Statistically Significant Hotspot (p >= 0.05) (Cholera))` | 28.6 |
| `MH_D14` | **Jalgaon** | `ELEVATED_SURVEILLANCE_FLAG` | `0.302 (UNVALIDATED_LIMITED_SAMPLE)` | 2 | `+2.18 (Statistically Significant Hotspot (HOTSPOT_95_CONFIDENCE / LISA=NOT_SIGNIFICANT, p<0.05) (Measles))` | 26.5 |
| `MH_D01` | **Ahmednagar** | `MULTI_ENGINE_HIGH_PRIORITY_FLAG` | `0.295 (High)` | 3 | `+2.25 (Statistically Significant Hotspot (HOTSPOT_95_CONFIDENCE / LISA=NOT_SIGNIFICANT, p<0.05) (Zika Virus))` | 26.8 |
| `MH_D05` | **Bhandara** | `ELEVATED_SURVEILLANCE_FLAG` | `0.294 (UNVALIDATED_LIMITED_SAMPLE)` | 2 | `+2.83 (Not a Statistically Significant Hotspot (p >= 0.05) (Scrub Typhus))` | 36.9 |
| `MH_D22` | **Nandurbar** | `ELEVATED_SURVEILLANCE_FLAG` | `0.286 (High)` | 0 | `+2.47 (Not a Statistically Significant Hotspot (p >= 0.05) (Measles))` | 44.4 |
| `MH_D29` | **Sangli** | `MULTI_ENGINE_HIGH_PRIORITY_FLAG` | `0.286 (High)` | 4 | `+2.16 (Statistically Significant Hotspot (HOTSPOT_95_CONFIDENCE / LISA=NOT_SIGNIFICANT, p<0.05) (Hepatitis A))` | 30.0 |
| `MH_D20` | **Nagpur** | `MULTI_ENGINE_HIGH_PRIORITY_FLAG` | `0.286 (High)` | 9 | `+2.83 (Statistically Significant Hotspot (99% Gi* + LISA High-High, p<0.01) (Scrub Typhus))` | 25.8 |
| `MH_D10` | **Dhule** | `ELEVATED_SURVEILLANCE_FLAG` | `0.285 (High)` | 1 | `+2.47 (Not a Statistically Significant Hotspot (p >= 0.05) (Measles))` | 38.1 |
| `MH_D21` | **Nanded** | `ELEVATED_SURVEILLANCE_FLAG` | `0.284 (High)` | 2 | `+1.26 (Not a Statistically Significant Hotspot (p >= 0.05) (Food Poisoning))` | 29.8 |
| `MH_D25` | **Parbhani** | `ELEVATED_SURVEILLANCE_FLAG` | `0.282 (High)` | 2 | `+0.89 (Not a Statistically Significant Hotspot (p >= 0.05) (Food Poisoning))` | 37.2 |
| `MH_D06` | **Buldhana** | `MULTI_ENGINE_HIGH_PRIORITY_FLAG` | `0.281 (High)` | 3 | `+2.42 (Not a Statistically Significant Hotspot (p >= 0.05) (Cholera))` | 45.1 |
| `MH_D30` | **Satara** | `MULTI_ENGINE_HIGH_PRIORITY_FLAG` | `0.280 (High)` | 3 | `+2.40 (Statistically Significant Hotspot (HOTSPOT_95_CONFIDENCE / LISA=NOT_SIGNIFICANT, p<0.05) (Chikungunya))` | 18.5 |
| `MH_D12` | **Gondia** | `ELEVATED_SURVEILLANCE_FLAG` | `0.276 (Moderate)` | 2 | `+2.48 (Statistically Significant Hotspot (HOTSPOT_95_CONFIDENCE / LISA=HH_HOTSPOT, p<0.05) (Malaria))` | 30.4 |
| `MH_D31` | **Sindhudurg** | `ELEVATED_SURVEILLANCE_FLAG` | `0.275 (Moderate)` | 3 | `+2.48 (Not a Statistically Significant Hotspot (p >= 0.05) (Hepatitis A))` | 25.3 |
| `MH_D16` | **Kolhapur** | `ELEVATED_SURVEILLANCE_FLAG` | `0.274 (Moderate)` | 9 | `+2.48 (Statistically Significant Hotspot (HOTSPOT_95_CONFIDENCE / LISA=NOT_SIGNIFICANT, p<0.05) (Hepatitis A))` | 27.2 |
| `MH_D34` | **Wardha** | `ELEVATED_SURVEILLANCE_FLAG` | `0.274 (Moderate)` | 1 | `+2.34 (Statistically Significant Hotspot (HOTSPOT_95_CONFIDENCE / LISA=NOT_SIGNIFICANT, p<0.05) (Scrub Typhus))` | 28.8 |
| `MH_D26` | **Pune** | `ELEVATED_SURVEILLANCE_FLAG` | `0.272 (Moderate)` | 4 | `+2.21 (Not a Statistically Significant Hotspot (p >= 0.05) (Leptospirosis))` | 14.5 |

---

## 5. Generated Datasets, Figures & Interactive Artifacts

- **Model Explainability Inventory**: `data/explanations/model_explainability_inventory.csv`
- **Global Feature Explanations**: `data/explanations/global_feature_explanations.csv` & `global_model_explanations_summary.json`
- **Local Risk Explanations (`511` rows)**: `data/explanations/local_risk_explanations.csv`
- **Local Forecast Explanations (`7d`, `14d`, `30d`)**: `data/explanations/local_forecast_explanations.csv`
- **Local Anomaly Explanations (`511` rows, `116` flagged)**: `data/explanations/local_anomaly_explanations.csv`
- **Frontend Panel (`"Why was this district flagged?"` across `36` districts)**: `data/explanations/why_district_flagged_panels.csv` & `why_district_flagged_panels.json`
- **Interactive HTML Analyst Explorer**: `reports/explainability/interactive_why_district_flagged_explorer.html`
- **Static PNG Figures (`5` charts)**: `reports/explainability/figures/01_*.png` through `05_*.png`
