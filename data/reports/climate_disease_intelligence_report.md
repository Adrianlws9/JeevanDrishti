# JEEVAN-DRISHTI Phase 8: Climate–Disease Intelligence Report (Maharashtra, 2023–2025)

> **SCIENTIFIC INTERPRETATION MANDATE**: Observed correlation, regression association, or ML/SHAP feature importance reflects statistical co-movement and predictive utility only; it does NOT establish causation.

- **Generated At**: `2026-10-31T15:45:34`
- **Engine Version**: `phase8-v1.0.0`
- **Data Period**: `2023-W01 to 2025-W52` (`2023-01-01 to 2025-12-31`, `121` active IDSP reporting weeks; `2009–2022` historical EpiClim file has zero date overlap with IMD/CWPRS climate files).
- **Native Reporting Frequency**: Daily IMD/CWPRS climate aggregated to **Weekly** (`W01..W52`) and **Monthly** (`36 months`) to match native weekly IDSP outbreak surveillance.

## 1. Available Climate & Disease Data Coverage Audit

| Variable | Source File | Native Grain | Date Coverage | Spatial Coverage | Usable Scope |
|---|---|---|---|---|---|
| **Rainfall (`rain_mm`)** | `imdweb_5dgbAskl.nc` (0.25° grid) | Daily | `2023-01-01..2025-12-31` | **35 / 36 districts** (`Gadchiroli` via 3×3 neighborhood fallback; `Mumbai Suburban` has no coordinates) | Statewide weekly & monthly analysis |
| **Min & Max Temp (`tmin_c`, `tmax_c`, derived `tavg_c`, diurnal range)** | `imdweb_Oe9H04Bg.nc`, `imdweb_tXexQbB6.nc` (1.0° grid) | Daily | `2023-01-01..2025-12-31` | **35 / 36 districts** (`Mumbai Suburban` missing coordinates) | Statewide weekly & monthly analysis |
| **Relative Humidity (`rh_mean`)** | `humid_tel_hr_cwprs_mh_2021_2025.csv` | Sub-hourly | `2024-05-11..2025-06-09` (395 valid days) | **1 / 36 districts (`Pune` ONLY)** | Pune single-district sub-analysis only |
| **Disease Outbreaks (`2023–2025`)** | `maharashtra_outbreaks_2023_2025.csv` | Weekly (`W01..W52`) | `2023-W01..2025-W52` (`121` active weeks) | **34 / 36 districts** (`511` deduplicated outbreak weeks; `508` climate-aligned; `Mumbai City` & `Ratnagiri` have `0` outbreaks) | 11 diseases with $\ge 8$ records; 17 rare/unsupported diseases |

## 2. Statewide Weekly Lagged Climate–Disease Associations (`121` Active IDSP Weeks)

| Series | Climate Variable | Lag (Weeks) | N | Pearson $r$ (log1p) | Spearman $\rho$ | $p$-value | FDR $q$-value | Statistical Association Verdict |
|---|---|---|---|---|---|---|---|---|
| `Statewide_Vector_Borne_Weekly_Cases` | `tmin_c_mean_lag_4w` | 4w | 118 | +0.522 | **+0.502** | 0.0000 | 0.0000 | Statistically significant after FDR correction (Spearman rho=+0.502, p=0.0000, q=0.0000) |
| `Statewide_Dengue_Weekly_Cases` | `tmin_c_mean_lag_4w` | 4w | 118 | +0.488 | **+0.490** | 0.0000 | 0.0000 | Statistically significant after FDR correction (Spearman rho=+0.490, p=0.0000, q=0.0000) |
| `Statewide_Vector_Borne_Weekly_Cases` | `rain_mm_sum_t` | 0w | 121 | +0.334 | **+0.475** | 0.0000 | 0.0000 | Statistically significant after FDR correction (Spearman rho=+0.475, p=0.0000, q=0.0000) |
| `Statewide_Vector_Borne_Weekly_Cases` | `temp_diurnal_range_mean_t` | 0w | 121 | -0.461 | **-0.475** | 0.0000 | 0.0000 | Statistically significant after FDR correction (Spearman rho=-0.475, p=0.0000, q=0.0000) |
| `Statewide_Vector_Borne_Weekly_Cases` | `tmin_c_mean_lag_2w` | 2w | 119 | +0.469 | **+0.457** | 0.0000 | 0.0000 | Statistically significant after FDR correction (Spearman rho=+0.457, p=0.0000, q=0.0000) |
| `Statewide_Vector_Borne_Weekly_Cases` | `temp_diurnal_range_lag_1w` | 1w | 120 | -0.450 | **-0.448** | 0.0000 | 0.0000 | Statistically significant after FDR correction (Spearman rho=-0.448, p=0.0000, q=0.0000) |
| `Statewide_Vector_Borne_Weekly_Cases` | `rain_mm_accum_prev_4w` | 4w | 118 | +0.306 | **+0.447** | 0.0000 | 0.0000 | Statistically significant after FDR correction (Spearman rho=+0.447, p=0.0000, q=0.0000) |
| `Statewide_Dengue_Weekly_Cases` | `tmin_c_mean_lag_2w` | 2w | 119 | +0.462 | **+0.447** | 0.0000 | 0.0000 | Statistically significant after FDR correction (Spearman rho=+0.447, p=0.0000, q=0.0000) |
| `Statewide_Vector_Borne_Weekly_Cases` | `rain_mm_accum_prev_2w` | 2w | 119 | +0.294 | **+0.429** | 0.0000 | 0.0000 | Statistically significant after FDR correction (Spearman rho=+0.429, p=0.0000, q=0.0000) |
| `Statewide_Vector_Borne_Weekly_Cases` | `tmin_c_mean_lag_8w` | 8w | 115 | +0.437 | **+0.425** | 0.0000 | 0.0000 | Statistically significant after FDR correction (Spearman rho=+0.425, p=0.0000, q=0.0000) |
| `Statewide_Vector_Borne_Weekly_Cases` | `rain_mm_accum_prev_8w` | 8w | 115 | +0.317 | **+0.424** | 0.0000 | 0.0000 | Statistically significant after FDR correction (Spearman rho=+0.424, p=0.0000, q=0.0000) |
| `Statewide_Vector_Borne_Weekly_Cases` | `rain_mm_sum_lag_1w` | 1w | 120 | +0.328 | **+0.416** | 0.0000 | 0.0000 | Statistically significant after FDR correction (Spearman rho=+0.416, p=0.0000, q=0.0000) |
| `Statewide_Vector_Borne_Weekly_Cases` | `rain_mm_sum_lag_4w` | 4w | 118 | +0.235 | **+0.410** | 0.0000 | 0.0000 | Statistically significant after FDR correction (Spearman rho=+0.410, p=0.0000, q=0.0000) |
| `Statewide_Vector_Borne_Weekly_Cases` | `temp_diurnal_range_lag_2w` | 2w | 119 | -0.391 | **-0.410** | 0.0000 | 0.0000 | Statistically significant after FDR correction (Spearman rho=-0.410, p=0.0000, q=0.0000) |
| `Statewide_Vector_Borne_Weekly_Cases` | `rain_mm_sum_lag_3w` | 3w | 118 | +0.264 | **+0.403** | 0.0000 | 0.0002 | Statistically significant after FDR correction (Spearman rho=+0.403, p=0.0000, q=0.0002) |
| `Statewide_Dengue_Weekly_Cases` | `rain_mm_sum_t` | 0w | 121 | +0.315 | **+0.402** | 0.0000 | 0.0000 | Statistically significant after FDR correction (Spearman rho=+0.402, p=0.0000, q=0.0000) |
| `Statewide_Dengue_Weekly_Cases` | `tavg_c_mean_lag_4w` | 4w | 118 | +0.430 | **+0.399** | 0.0000 | 0.0002 | Statistically significant after FDR correction (Spearman rho=+0.399, p=0.0000, q=0.0002) |
| `Statewide_Dengue_Weekly_Cases` | `temp_diurnal_range_mean_t` | 0w | 121 | -0.382 | **-0.397** | 0.0000 | 0.0002 | Statistically significant after FDR correction (Spearman rho=-0.397, p=0.0000, q=0.0002) |

## 3. Disease-Wise Climate Correlation Summary & Four-Tier Scientific Interpretation

| Disease | Aligned Outbreaks ($N$) | Districts | Dominant Season | Best Rainfall Lag ($\rho$, $p$) | Best Temp Lag ($\rho$, $p$) | Status | Observed Relationship | Statistical Association | Predictive Usefulness |
|---|---|---|---|---|---|---|---|---|---|
| **Dengue** | 151 | 30 | `Southwest_Monsoon` | `rain_mm_sum_lag_2w` (+0.111, p=0.176) | `tavg_c_mean_t` (+0.160, p=0.050) | `SUPPORTED_STATISTICAL_ANALYSIS` | Observed weak-to-moderate positive empirical co-movement (rho=+0.160, n=151) between tavg_c_mean_t and Dengue. | Nominally significant before FDR (Spearman rho=+0.160, p=0.0495, q=0.2017) | Useful non-linear/lagged covariate when combined with seasonal baselines in tree/ensemble models. |
| **Acute Diarrhoeal Disease** | 109 | 31 | `Southwest_Monsoon` | `rain_mm_sum_lag_3w` (-0.129, p=0.182) | `tavg_c_mean_t` (-0.171, p=0.075) | `SUPPORTED_STATISTICAL_ANALYSIS` | Observed weak-to-moderate negative empirical co-movement (rho=-0.171, n=109) between tavg_c_mean_t and Acute Diarrhoeal Disease. | Not statistically significant at alpha=0.05 (Spearman rho=-0.171, p=0.0753, q=0.2772) | Useful non-linear/lagged covariate when combined with seasonal baselines in tree/ensemble models. |
| **Food Poisoning** | 49 | 22 | `Pre-Monsoon_Summer` | `rain_mm_accum_prev_8w` (-0.311, p=0.038) | `tavg_c_mean_lag_8w` (-0.472, p=0.001) | `SUPPORTED_STATISTICAL_ANALYSIS` | Observed moderate-to-strong negative empirical co-movement (rho=-0.472, n=49) between tavg_c_mean_lag_8w and Food Poisoning. | Statistically significant after FDR correction (Spearman rho=-0.472, p=0.0011, q=0.0104) | Useful non-linear/lagged covariate when combined with seasonal baselines in tree/ensemble models. |
| **Malaria** | 39 | 12 | `Southwest_Monsoon` | `rain_mm_sum_lag_8w` (-0.224, p=0.171) | `tmin_c_mean_lag_8w` (-0.252, p=0.122) | `SUPPORTED_STATISTICAL_ANALYSIS` | Observed weak-to-moderate negative empirical co-movement (rho=-0.252, n=39) between tmin_c_mean_lag_8w and Malaria. | Not statistically significant at alpha=0.05 (Spearman rho=-0.252, p=0.1220, q=0.3663) | Useful non-linear/lagged covariate when combined with seasonal baselines in tree/ensemble models. |
| **Chikungunya** | 34 | 14 | `Southwest_Monsoon` | `rain_mm_sum_lag_4w` (+0.270, p=0.123) | `temp_diurnal_range_lag_4w` (-0.319, p=0.066) | `SUPPORTED_STATISTICAL_ANALYSIS` | Observed moderate-to-strong negative empirical co-movement (rho=-0.319, n=34) between temp_diurnal_range_lag_4w and Chikungunya. | Not statistically significant at alpha=0.05 (Spearman rho=-0.319, p=0.0656, q=0.2580) | Useful non-linear/lagged covariate when combined with seasonal baselines in tree/ensemble models. |
| **Hepatitis A** | 24 | 8 | `Southwest_Monsoon` | `rain_mm_sum_lag_3w` (-0.344, p=0.100) | `tavg_c_mean_lag_8w` (-0.488, p=0.015) | `SUPPORTED_STATISTICAL_ANALYSIS` | Observed moderate-to-strong negative empirical co-movement (rho=-0.488, n=24) between tavg_c_mean_lag_8w and Hepatitis A. | Nominally significant before FDR (Spearman rho=-0.488, p=0.0155, q=0.0891) | Useful non-linear/lagged covariate when combined with seasonal baselines in tree/ensemble models. |
| **Cholera** | 18 | 11 | `Southwest_Monsoon` | `rain_mm_sum_lag_1w` (+0.243, p=0.332) | `temp_diurnal_range_lag_8w` (+0.415, p=0.087) | `SUPPORTED_STATISTICAL_ANALYSIS` | Observed moderate-to-strong positive empirical co-movement (rho=+0.415, n=18) between temp_diurnal_range_lag_8w and Cholera. | Not statistically significant at alpha=0.05 (Spearman rho=+0.415, p=0.0871, q=0.3018) | Useful non-linear/lagged covariate when combined with seasonal baselines in tree/ensemble models. |
| **Zika Virus** | 16 | 6 | `Southwest_Monsoon` | `rain_mm_accum_prev_8w` (+0.428, p=0.098) | `temp_diurnal_range_mean_t` (-0.766, p=0.001) | `SUPPORTED_STATISTICAL_ANALYSIS` | Observed moderate-to-strong negative empirical co-movement (rho=-0.766, n=16) between temp_diurnal_range_mean_t and Zika Virus. | Statistically significant after FDR correction (Spearman rho=-0.766, p=0.0005, q=0.0055) | Useful non-linear/lagged covariate when combined with seasonal baselines in tree/ensemble models. |
| **Fever (unspecified)** | 14 | 11 | `Southwest_Monsoon` | `rain_mm_sum_lag_8w` (-0.317, p=0.270) | `tavg_c_mean_lag_2w` (-0.648, p=0.012) | `SUPPORTED_STATISTICAL_ANALYSIS` | Observed moderate-to-strong negative empirical co-movement (rho=-0.648, n=14) between tavg_c_mean_lag_2w and Fever (unspecified). | Nominally significant before FDR (Spearman rho=-0.648, p=0.0121, q=0.0755) | Useful non-linear/lagged covariate when combined with seasonal baselines in tree/ensemble models. |
| **Leptospirosis** | 9 | 7 | `Post-Monsoon` | `rain_mm_accum_prev_8w` (-0.724, p=0.028) | `temp_diurnal_range_lag_4w` (+0.655, p=0.055) | `SUPPORTED_STATISTICAL_ANALYSIS` | Observed moderate-to-strong negative empirical co-movement (rho=-0.724, n=9) between rain_mm_accum_prev_8w and Leptospirosis. | Nominally significant before FDR (Spearman rho=-0.724, p=0.0276, q=0.4124) | Useful non-linear/lagged covariate when combined with seasonal baselines in tree/ensemble models. |
| **Scrub Typhus** | 9 | 5 | `Southwest_Monsoon` | `rain_mm_sum_lag_8w` (-0.613, p=0.079) | `temp_diurnal_range_lag_8w` (+0.628, p=0.070) | `SUPPORTED_STATISTICAL_ANALYSIS` | Observed moderate-to-strong positive empirical co-movement (rho=+0.628, n=9) between temp_diurnal_range_lag_8w and Scrub Typhus. | Not statistically significant at alpha=0.05 (Spearman rho=+0.628, p=0.0704, q=0.5297) | Useful non-linear/lagged covariate when combined with seasonal baselines in tree/ensemble models. |
| **Measles** | 9 | 5 | `Winter` | `rain_mm_sum_t` (+0.550, p=0.125) | `tavg_c_mean_lag_1w` (+0.586, p=0.097) | `SUPPORTED_STATISTICAL_ANALYSIS` | Observed moderate-to-strong positive empirical co-movement (rho=+0.586, n=9) between tavg_c_mean_lag_1w and Measles. | Not statistically significant at alpha=0.05 (Spearman rho=+0.586, p=0.0974, q=0.5489) | Useful non-linear/lagged covariate when combined with seasonal baselines in tree/ensemble models. |
| **Dengue + Chikungunya** | 6 | 3 | `Pre-Monsoon_Summer` | N/A (<8 obs) | N/A (<8 obs) | `INSUFFICIENT_SAMPLE_LT8_OBS` | Insufficient paired observations (n=6) for Dengue + Chikungunya and climate covariates. | UNSUPPORTED_INSUFFICIENT_SAMPLE | Not usable as a standalone predictor due to sparse sample size. |
| **Acute Encephalitis Syndrome** | 3 | 2 | `Southwest_Monsoon` | N/A (<8 obs) | N/A (<8 obs) | `INSUFFICIENT_SAMPLE_LT8_OBS` | Insufficient paired observations (n=3) for Acute Encephalitis Syndrome and climate covariates. | UNSUPPORTED_INSUFFICIENT_SAMPLE | Not usable as a standalone predictor due to sparse sample size. |
| **Rabies (human)** | 2 | 1 | `Post-Monsoon` | N/A (<8 obs) | N/A (<8 obs) | `INSUFFICIENT_SAMPLE_LT8_OBS` | Insufficient paired observations (n=2) for Rabies (human) and climate covariates. | UNSUPPORTED_INSUFFICIENT_SAMPLE | Not usable as a standalone predictor due to sparse sample size. |
| **Typhoid** | 2 | 2 | `Southwest_Monsoon` | N/A (<8 obs) | N/A (<8 obs) | `INSUFFICIENT_SAMPLE_LT8_OBS` | Insufficient paired observations (n=2) for Typhoid and climate covariates. | UNSUPPORTED_INSUFFICIENT_SAMPLE | Not usable as a standalone predictor due to sparse sample size. |
| **Mumps** | 2 | 2 | `Post-Monsoon` | N/A (<8 obs) | N/A (<8 obs) | `INSUFFICIENT_SAMPLE_LT8_OBS` | Insufficient paired observations (n=2) for Mumps and climate covariates. | UNSUPPORTED_INSUFFICIENT_SAMPLE | Not usable as a standalone predictor due to sparse sample size. |
| **Chickenpox** | 2 | 2 | `Pre-Monsoon_Summer` | N/A (<8 obs) | N/A (<8 obs) | `INSUFFICIENT_SAMPLE_LT8_OBS` | Insufficient paired observations (n=2) for Chickenpox and climate covariates. | UNSUPPORTED_INSUFFICIENT_SAMPLE | Not usable as a standalone predictor due to sparse sample size. |
| **Acute Hepatitis (unspecified)** | 1 | 1 | `Pre-Monsoon_Summer` | N/A (<8 obs) | N/A (<8 obs) | `INSUFFICIENT_SAMPLE_LT8_OBS` | Insufficient paired observations (n=1) for Acute Hepatitis (unspecified) and climate covariates. | UNSUPPORTED_INSUFFICIENT_SAMPLE | Not usable as a standalone predictor due to sparse sample size. |
| **Dysentery** | 1 | 1 | `Southwest_Monsoon` | N/A (<8 obs) | N/A (<8 obs) | `INSUFFICIENT_SAMPLE_LT8_OBS` | Insufficient paired observations (n=1) for Dysentery and climate covariates. | UNSUPPORTED_INSUFFICIENT_SAMPLE | Not usable as a standalone predictor due to sparse sample size. |
| **Dengue + Malaria** | 1 | 1 | `Southwest_Monsoon` | N/A (<8 obs) | N/A (<8 obs) | `INSUFFICIENT_SAMPLE_LT8_OBS` | Insufficient paired observations (n=1) for Dengue + Malaria and climate covariates. | UNSUPPORTED_INSUFFICIENT_SAMPLE | Not usable as a standalone predictor due to sparse sample size. |
| **Fever with Rash** | 1 | 1 | `Pre-Monsoon_Summer` | N/A (<8 obs) | N/A (<8 obs) | `INSUFFICIENT_SAMPLE_LT8_OBS` | Insufficient paired observations (n=1) for Fever with Rash and climate covariates. | UNSUPPORTED_INSUFFICIENT_SAMPLE | Not usable as a standalone predictor due to sparse sample size. |
| **Fever of Unknown Cause** | 1 | 1 | `Post-Monsoon` | N/A (<8 obs) | N/A (<8 obs) | `INSUFFICIENT_SAMPLE_LT8_OBS` | Insufficient paired observations (n=1) for Fever of Unknown Cause and climate covariates. | UNSUPPORTED_INSUFFICIENT_SAMPLE | Not usable as a standalone predictor due to sparse sample size. |
| **Hepatitis A + E** | 1 | 1 | `Winter` | N/A (<8 obs) | N/A (<8 obs) | `INSUFFICIENT_SAMPLE_LT8_OBS` | Insufficient paired observations (n=1) for Hepatitis A + E and climate covariates. | UNSUPPORTED_INSUFFICIENT_SAMPLE | Not usable as a standalone predictor due to sparse sample size. |
| **Japanese Encephalitis** | 1 | 1 | `Southwest_Monsoon` | N/A (<8 obs) | N/A (<8 obs) | `INSUFFICIENT_SAMPLE_LT8_OBS` | Insufficient paired observations (n=1) for Japanese Encephalitis and climate covariates. | UNSUPPORTED_INSUFFICIENT_SAMPLE | Not usable as a standalone predictor due to sparse sample size. |
| **Jaundice** | 1 | 1 | `Winter` | N/A (<8 obs) | N/A (<8 obs) | `INSUFFICIENT_SAMPLE_LT8_OBS` | Insufficient paired observations (n=1) for Jaundice and climate covariates. | UNSUPPORTED_INSUFFICIENT_SAMPLE | Not usable as a standalone predictor due to sparse sample size. |
| **Measles + Rubella** | 1 | 1 | `Winter` | N/A (<8 obs) | N/A (<8 obs) | `INSUFFICIENT_SAMPLE_LT8_OBS` | Insufficient paired observations (n=1) for Measles + Rubella and climate covariates. | UNSUPPORTED_INSUFFICIENT_SAMPLE | Not usable as a standalone predictor due to sparse sample size. |
| **Pertussis** | 1 | 1 | `Southwest_Monsoon` | N/A (<8 obs) | N/A (<8 obs) | `INSUFFICIENT_SAMPLE_LT8_OBS` | Insufficient paired observations (n=1) for Pertussis and climate covariates. | UNSUPPORTED_INSUFFICIENT_SAMPLE | Not usable as a standalone predictor due to sparse sample size. |
| **Heat-related illness** | 0 | 0 | `N/A` | N/A (<8 obs) | N/A (<8 obs) | `UNSUPPORTED_ZERO_RECORDS` | Insufficient paired observations (n=0) for Heat-related illness and climate covariates. | UNSUPPORTED_INSUFFICIENT_SAMPLE | Not usable as a standalone predictor due to sparse sample size. |

## 4.Pune Single-Station Relative Humidity Sub-Analysis (`May 2024 – Jun 2025`)

| Grain | Series | RH Variable | Lag (Weeks) | N | Spearman $\rho$ | $p$-value | Interpretation |
|---|---|---|---|---|---|---|---|
| `pune_only_weekly_rh_subpanel` | `Pune_All_Diseases_Weekly_Cases` | `rh_mean_t` | 0w | 53 | **+0.379** | 0.0052 | Statistically significant after FDR correction (Spearman rho=+0.379, p=0.0052, q=0.0397) |
| `pune_only_outbreak_severity` | `Pune_Active_Outbreaks` | `rh_mean_t` | 0w | 20 | **+0.008** | 0.9721 | Not statistically significant at alpha=0.05 (Spearman rho=+0.008, p=0.9721, q=0.9816) |
| `pune_only_weekly_rh_subpanel` | `Pune_All_Diseases_Weekly_Cases` | `rh_mean_lag_1w` | 1w | 52 | **+0.436** | 0.0012 | Statistically significant after FDR correction (Spearman rho=+0.436, p=0.0012, q=0.0118) |
| `pune_only_outbreak_severity` | `Pune_Active_Outbreaks` | `rh_mean_lag_1w` | 1w | 20 | **+0.134** | 0.5743 | Not statistically significant at alpha=0.05 (Spearman rho=+0.134, p=0.5743, q=0.8005) |
| `pune_only_weekly_rh_subpanel` | `Pune_All_Diseases_Weekly_Cases` | `rh_mean_lag_2w` | 2w | 51 | **+0.366** | 0.0083 | Nominally significant before FDR (Spearman rho=+0.366, p=0.0083, q=0.0558) |
| `pune_only_outbreak_severity` | `Pune_Active_Outbreaks` | `rh_mean_lag_2w` | 2w | 20 | **-0.111** | 0.6417 | Not statistically significant at alpha=0.05 (Spearman rho=-0.111, p=0.6417, q=0.8483) |
| `pune_only_weekly_rh_subpanel` | `Pune_All_Diseases_Weekly_Cases` | `rh_mean_lag_4w` | 4w | 49 | **+0.322** | 0.0243 | Nominally significant before FDR (Spearman rho=+0.322, p=0.0243, q=0.1242) |
| `pune_only_outbreak_severity` | `Pune_Active_Outbreaks` | `rh_mean_lag_4w` | 4w | 18 | **-0.352** | 0.1518 | Not statistically significant at alpha=0.05 (Spearman rho=-0.352, p=0.1518, q=0.4202) |
| `pune_only_weekly_rh_subpanel` | `Pune_All_Diseases_Weekly_Cases` | `rh_mean_roll_prev_4w` | 4w | 52 | **+0.397** | 0.0036 | Statistically significant after FDR correction (Spearman rho=+0.397, p=0.0036, q=0.0288) |
| `pune_only_outbreak_severity` | `Pune_Active_Outbreaks` | `rh_mean_roll_prev_4w` | 4w | 20 | **-0.204** | 0.3877 | Not statistically significant at alpha=0.05 (Spearman rho=-0.204, p=0.3877, q=0.6705) |

## 5. Seasonal Regime Comparisons (`Kruskal-Wallis` Non-Parametric Test)

| Season | Active Weeks | Mean Weekly Rain (mm) | Mean Weekly Tavg (°C) | Mean Diurnal Range (°C) | Mean Weekly Outbreaks | Mean Vector-Borne Cases | Mean Water/Food-Borne Cases | Kruskal-Wallis Vector $H$ ($p$) | Kruskal-Wallis Water $H$ ($p$) |
|---|---|---|---|---|---|---|---|---|---|
| `Post-Monsoon` | 27 | 7.00 | 24.87 | 12.88 | 4.67 | **39.33** | **77.78** | 29.04 (0.0000) | 2.33 (0.5065) |
| `Pre-Monsoon_Summer` | 27 | 8.45 | 29.89 | 14.68 | 3.74 | **36.11** | **94.56** | 29.04 (0.0000) | 2.33 (0.5065) |
| `Southwest_Monsoon` | 43 | 70.49 | 27.28 | 8.04 | 6.26 | **57.44** | **90.74** | 29.04 (0.0000) | 2.33 (0.5065) |
| `Winter` | 24 | 0.26 | 23.65 | 15.94 | 2.96 | **30.08** | **195.21** | 29.04 (0.0000) | 2.33 (0.5065) |

## 6. Confounder-Adjusted Regression Analysis (Unadjusted vs. Seasonality & Geography Controlled)

| Regression Model | N | Climate Predictor | Unadjusted $\beta$ ($p$) | Season-Adjusted $\beta$ ($p$) | Unadj $R^2$ -> Adj $R^2$ | Survives Seasonal Control? | Scientific Interpretation |
|---|---|---|---|---|---|---|---|
| `Statewide_Vector_Borne_Weekly_Log1p_Cases` | 118 | `rain_mm_accum_prev_4w` | +0.8524 (0.9384) | **-6.4648** (0.6023) | 0.011 -> 0.035 | `False` | Standardized slope beta=-6.4648 (p=0.6023) after controlling for week_sin,week_cos. Association attenuates after controlling for seasonality/geography, indicating seasonal confounding. |
| `Statewide_Vector_Borne_Weekly_Log1p_Cases` | 118 | `tmin_c_mean_lag_1w` | +2.4214 (0.7952) | **-9.3533** (0.5149) | 0.011 -> 0.035 | `False` | Standardized slope beta=-9.3533 (p=0.5149) after controlling for week_sin,week_cos. Association attenuates after controlling for seasonality/geography, indicating seasonal confounding. |
| `Statewide_Vector_Borne_Weekly_Log1p_Cases` | 118 | `temp_diurnal_range_lag_1w` | -5.8681 (0.6354) | **+9.7129** (0.5326) | 0.011 -> 0.035 | `False` | Standardized slope beta=+9.7129 (p=0.5326) after controlling for week_sin,week_cos. Association attenuates after controlling for seasonality/geography, indicating seasonal confounding. |
| `Statewide_Water_Borne_Weekly_Cases` | 120 | `rain_mm_sum_lag_1w` | +11.4352 (0.4410) | **+7.5076** (0.7275) | 0.009 -> 0.027 | `False` | Standardized slope beta=+7.5076 (p=0.7275) after controlling for week_sin,week_cos. Association attenuates after controlling for seasonality/geography, indicating seasonal confounding. |
| `Statewide_Water_Borne_Weekly_Cases` | 120 | `tmax_c_mean_lag_1w` | -8.3252 (0.7050) | **+5.3716** (0.8799) | 0.009 -> 0.027 | `False` | Standardized slope beta=+5.3716 (p=0.8799) after controlling for week_sin,week_cos. Association attenuates after controlling for seasonality/geography, indicating seasonal confounding. |
| `Statewide_Water_Borne_Weekly_Cases` | 120 | `temp_diurnal_range_lag_1w` | +29.1483 (0.3075) | **-41.0447** (0.2615) | 0.009 -> 0.027 | `False` | Standardized slope beta=-41.0447 (p=0.2615) after controlling for week_sin,week_cos. Association attenuates after controlling for seasonality/geography, indicating seasonal confounding. |
| `All_Outbreaks_Severity_Log1p_Cases` | 502 | `rain_mm_accum_prev_4w` | +0.0146 (0.8448) | **+0.0186** (0.8056) | 0.025 -> 0.041 | `False` | Standardized slope beta=+0.0186 (p=0.8056) after controlling for week_sin,week_cos,district_lat,district_lon. Association attenuates after controlling for seasonality/geography, indicating seasonal confounding. |
| `All_Outbreaks_Severity_Log1p_Cases` | 502 | `tmin_c_mean_lag_1w` | -0.0623 (0.1263) | **-0.0780** (0.2895) | 0.025 -> 0.041 | `False` | Standardized slope beta=-0.0780 (p=0.2895) after controlling for week_sin,week_cos,district_lat,district_lon. Association attenuates after controlling for seasonality/geography, indicating seasonal confounding. |
| `All_Outbreaks_Severity_Log1p_Cases` | 502 | `tmax_c_mean_lag_1w` | +0.0624 (0.0861) | **+0.0019** (0.9651) | 0.025 -> 0.041 | `False` | Standardized slope beta=+0.0019 (p=0.9651) after controlling for week_sin,week_cos,district_lat,district_lon. Association attenuates after controlling for seasonality/geography, indicating seasonal confounding. |
| `All_Outbreaks_Severity_Log1p_Cases` | 502 | `temp_diurnal_range_lag_1w` | +0.1042 (0.0043) | **+0.0661** (0.3129) | 0.025 -> 0.041 | `False` | Standardized slope beta=+0.0661 (p=0.3129) after controlling for week_sin,week_cos,district_lat,district_lon. Association attenuates after controlling for seasonality/geography, indicating seasonal confounding. |
| `Dengue_Outbreak_Severity_Log1p_Cases` | 150 | `rain_mm_accum_prev_4w` | -0.0489 (0.5258) | **-0.0699** (0.3689) | 0.009 -> 0.064 | `False` | Standardized slope beta=-0.0699 (p=0.3689) after controlling for week_sin,week_cos,district_lat. Association attenuates after controlling for seasonality/geography, indicating seasonal confounding. |
| `Dengue_Outbreak_Severity_Log1p_Cases` | 150 | `tmin_c_mean_lag_1w` | +0.0721 (0.2703) | **-0.0528** (0.6152) | 0.009 -> 0.064 | `False` | Standardized slope beta=-0.0528 (p=0.6152) after controlling for week_sin,week_cos,district_lat. Association attenuates after controlling for seasonality/geography, indicating seasonal confounding. |
| `Dengue_Outbreak_Severity_Log1p_Cases` | 150 | `temp_diurnal_range_lag_1w` | -0.0194 (0.8112) | **+0.0528** (0.6085) | 0.009 -> 0.064 | `False` | Standardized slope beta=+0.0528 (p=0.6085) after controlling for week_sin,week_cos,district_lat. Association attenuates after controlling for seasonality/geography, indicating seasonal confounding. |
| `Acute Diarrhoeal Disease_Outbreak_Severity_Log1p_Cases` | 108 | `rain_mm_accum_prev_4w` | +0.1222 (0.5598) | **+0.1174** (0.6049) | 0.030 -> 0.057 | `False` | Standardized slope beta=+0.1174 (p=0.6049) after controlling for week_sin,week_cos,district_lat. Association attenuates after controlling for seasonality/geography, indicating seasonal confounding. |
| `Acute Diarrhoeal Disease_Outbreak_Severity_Log1p_Cases` | 108 | `tmin_c_mean_lag_1w` | -0.0756 (0.5704) | **-0.2117** (0.2516) | 0.030 -> 0.057 | `False` | Standardized slope beta=-0.2117 (p=0.2516) after controlling for week_sin,week_cos,district_lat. Association attenuates after controlling for seasonality/geography, indicating seasonal confounding. |
| `Acute Diarrhoeal Disease_Outbreak_Severity_Log1p_Cases` | 108 | `temp_diurnal_range_lag_1w` | +0.1203 (0.4172) | **+0.2779** (0.1422) | 0.030 -> 0.057 | `False` | Standardized slope beta=+0.2779 (p=0.1422) after controlling for week_sin,week_cos,district_lat. Association attenuates after controlling for seasonality/geography, indicating seasonal confounding. |
| `Malaria_Outbreak_Severity_Log1p_Cases` | 39 | `rain_mm_accum_prev_4w` | +0.2227 (0.5582) | **+0.1468** (0.7929) | 0.103 -> 0.144 | `False` | Standardized slope beta=+0.1468 (p=0.7928) after controlling for week_sin,week_cos,district_lat. Association attenuates after controlling for seasonality/geography, indicating seasonal confounding. |
| `Malaria_Outbreak_Severity_Log1p_Cases` | 39 | `tmin_c_mean_lag_1w` | -0.0619 (0.7973) | **-0.4410** (0.4159) | 0.103 -> 0.144 | `False` | Standardized slope beta=-0.4410 (p=0.4159) after controlling for week_sin,week_cos,district_lat. Association attenuates after controlling for seasonality/geography, indicating seasonal confounding. |
| `Malaria_Outbreak_Severity_Log1p_Cases` | 39 | `temp_diurnal_range_lag_1w` | +0.4909 (0.1091) | **+0.3601** (0.5479) | 0.103 -> 0.144 | `False` | Standardized slope beta=+0.3601 (p=0.5479) after controlling for week_sin,week_cos,district_lat. Association attenuates after controlling for seasonality/geography, indicating seasonal confounding. |
| `Food Poisoning_Outbreak_Severity_Log1p_Cases` | 48 | `rain_mm_accum_prev_4w` | -0.1628 (0.3964) | **-0.2025** (0.4026) | 0.078 -> 0.138 | `False` | Standardized slope beta=-0.2025 (p=0.4026) after controlling for week_sin,week_cos,district_lat. Association attenuates after controlling for seasonality/geography, indicating seasonal confounding. |
| `Food Poisoning_Outbreak_Severity_Log1p_Cases` | 48 | `tmin_c_mean_lag_1w` | -0.0975 (0.6218) | **+0.2494** (0.4639) | 0.078 -> 0.138 | `False` | Standardized slope beta=+0.2494 (p=0.4639) after controlling for week_sin,week_cos,district_lat. Association attenuates after controlling for seasonality/geography, indicating seasonal confounding. |
| `Food Poisoning_Outbreak_Severity_Log1p_Cases` | 48 | `temp_diurnal_range_lag_1w` | +0.0773 (0.7667) | **+0.3505** (0.2869) | 0.078 -> 0.138 | `False` | Standardized slope beta=+0.3505 (p=0.2869) after controlling for week_sin,week_cos,district_lat. Association attenuates after controlling for seasonality/geography, indicating seasonal confounding. |

## 7. Advanced ML Feature Importance & Exact TreeSHAP Summary (`Pooled Outbreak Severity`)

| Rank | Feature Name | Role | RF Impurity | RF Permutation | XGBoost Gain | Mean `|TreeSHAP|` | TreeSHAP Share | Direction Corr | Consensus Score |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `district_lat` | `Seasonal_or_Geographic_Confounder_Control` | 0.1422 | 0.0467 | 0.0561 | **0.1596** | 17.3% | +0.484 | **0.1129** |
| 2 | `tmin_c_mean_lag_8w` | `Climate_Variable` | 0.1722 | 0.0000 | 0.0781 | **0.0850** | 9.2% | -0.666 | **0.0902** |
| 3 | `tmin_c_mean_t` | `Climate_Variable` | 0.0322 | 0.2931 | 0.0460 | **0.0252** | 2.7% | +0.452 | **0.0738** |
| 4 | `tmax_c_mean_t` | `Climate_Variable` | 0.0306 | 0.2415 | 0.0334 | **0.0425** | 4.6% | +0.576 | **0.0685** |
| 5 | `tmin_c_mean_lag_4w` | `Climate_Variable` | 0.0232 | 0.1682 | 0.0315 | **0.0139** | 1.5% | +0.457 | **0.0446** |
| 6 | `tmin_c_mean_lag_1w` | `Climate_Variable` | 0.0598 | 0.0000 | 0.0461 | **0.0476** | 5.2% | -0.553 | **0.0439** |
| 7 | `tmax_c_mean_lag_1w` | `Climate_Variable` | 0.0237 | 0.1268 | 0.0336 | **0.0239** | 2.6% | +0.566 | **0.0429** |
| 8 | `rain_mm_accum_prev_8w` | `Climate_Variable` | 0.0437 | 0.0000 | 0.0379 | **0.0541** | 5.9% | -0.769 | **0.0407** |
| 9 | `tavg_c_mean_lag_4w` | `Climate_Variable` | 0.0456 | 0.0000 | 0.0543 | **0.0330** | 3.6% | -0.439 | **0.0380** |
| 10 | `temp_diurnal_range_lag_4w` | `Climate_Variable` | 0.0215 | 0.0458 | 0.0331 | **0.0397** | 4.3% | -0.704 | **0.0362** |
| 11 | `tmax_c_mean_lag_8w` | `Climate_Variable` | 0.0405 | 0.0000 | 0.0408 | **0.0378** | 4.1% | -0.044 | **0.0347** |
| 12 | `rain_mm_sum_lag_8w` | `Climate_Variable` | 0.0321 | 0.0000 | 0.0388 | **0.0382** | 4.2% | -0.582 | **0.0326** |
| 13 | `district_lon` | `Seasonal_or_Geographic_Confounder_Control` | 0.0462 | 0.0000 | 0.0354 | **0.0325** | 3.5% | +0.479 | **0.0322** |
| 14 | `week_sin` | `Seasonal_or_Geographic_Confounder_Control` | 0.0173 | 0.0000 | 0.0577 | **0.0285** | 3.1% | +0.920 | **0.0316** |
| 15 | `rain_mm_accum_prev_4w` | `Climate_Variable` | 0.0321 | 0.0000 | 0.0344 | **0.0383** | 4.2% | -0.613 | **0.0313** |

## 8. District-Wise Climate–Disease Comparison (Top 20 Districts by Outbreak Weeks)

| District | Division | Outbreak Weeks | Total Cases | Dominant Disease | Mean Weekly Rain (mm) | Mean Weekly Tavg (°C) | Panel Rain-4w $\rho$ ($p$) | Panel Tavg-1w $\rho$ ($p$) | Status |
|---|---|---|---|---|---|---|---|---|---|
| **Kolhapur** | Pune | 42 | 2335 | Hepatitis A | 18.1 | 26.4 | -0.087 (0.347) | +0.135 (0.143) | `SUPPORTED_DISTRICT_OUTBREAK_AND_PANEL_ANALYSIS` |
| **Thane** | Konkan | 34 | 588 | Dengue | 49.1 | 27.1 | +0.181 (0.050) | +0.015 (0.870) | `SUPPORTED_DISTRICT_OUTBREAK_AND_PANEL_ANALYSIS` |
| **Pune** | Pune | 33 | 587 | Dengue | 20.6 | 26.3 | +0.178 (0.054) | +0.132 (0.152) | `SUPPORTED_DISTRICT_OUTBREAK_AND_PANEL_ANALYSIS` |
| **Gadchiroli** | Nagpur_Vidarbha_East | 27 | 1432 | Malaria | 32.8 | 27.1 | +0.167 (0.070) | +0.103 (0.262) | `SUPPORTED_DISTRICT_OUTBREAK_AND_PANEL_ANALYSIS` |
| **Nanded** | Marathwada | 25 | 1518 | Dengue | 25.2 | 27.2 | -0.002 (0.985) | +0.133 (0.146) | `SUPPORTED_DISTRICT_OUTBREAK_AND_PANEL_ANALYSIS` |
| **Dharashiv** | Marathwada | 25 | 551 | Dengue | 21.8 | 26.8 | -0.036 (0.696) | +0.085 (0.356) | `SUPPORTED_DISTRICT_OUTBREAK_AND_PANEL_ANALYSIS` |
| **Raigad** | Konkan | 24 | 527 | Dengue | 81.4 | 26.3 | +0.085 (0.358) | +0.059 (0.524) | `SUPPORTED_DISTRICT_OUTBREAK_AND_PANEL_ANALYSIS` |
| **Amravati** | Amravati_Vidarbha_West | 24 | 954 | Acute Diarrhoeal Disease | 17.9 | 26.8 | +0.186 (0.043) | -0.129 (0.159) | `SUPPORTED_DISTRICT_OUTBREAK_AND_PANEL_ANALYSIS` |
| **Satara** | Pune | 19 | 470 | Dengue | 7.9 | 25.9 | +0.132 (0.154) | +0.149 (0.105) | `SUPPORTED_DISTRICT_OUTBREAK_AND_PANEL_ANALYSIS` |
| **Nagpur** | Nagpur_Vidarbha_East | 18 | 775 | Acute Diarrhoeal Disease | 27.9 | 25.7 | +0.078 (0.401) | +0.093 (0.311) | `SUPPORTED_DISTRICT_OUTBREAK_AND_PANEL_ANALYSIS` |
| **Yavatmal** | Amravati_Vidarbha_West | 16 | 604 | Acute Diarrhoeal Disease | 23.2 | 27.3 | +0.207 (0.025) | +0.259 (0.004) | `SUPPORTED_DISTRICT_OUTBREAK_AND_PANEL_ANALYSIS` |
| **Bhandara** | Nagpur_Vidarbha_East | 15 | 387 | Acute Diarrhoeal Disease | 27.6 | 25.7 | +0.073 (0.430) | +0.023 (0.804) | `SUPPORTED_DISTRICT_OUTBREAK_AND_PANEL_ANALYSIS` |
| **Chhatrapati Sambhajinagar** | Marathwada | 14 | 602 | Dengue | 14.1 | 26.3 | +0.176 (0.057) | -0.029 (0.752) | `SUPPORTED_DISTRICT_OUTBREAK_AND_PANEL_ANALYSIS` |
| **Sangli** | Pune | 14 | 743 | Food Poisoning | 9.8 | 25.9 | -0.092 (0.321) | -0.080 (0.387) | `SUPPORTED_DISTRICT_OUTBREAK_AND_PANEL_ANALYSIS` |
| **Akola** | Amravati_Vidarbha_West | 14 | 665 | Dengue | 17.5 | 26.8 | +0.021 (0.825) | +0.194 (0.034) | `SUPPORTED_DISTRICT_OUTBREAK_AND_PANEL_ANALYSIS` |
| **Ahmednagar** | Nashik | 13 | 663 | Acute Diarrhoeal Disease | 14.7 | 26.0 | -0.049 (0.601) | -0.007 (0.939) | `SUPPORTED_DISTRICT_OUTBREAK_AND_PANEL_ANALYSIS` |
| **Palghar** | Konkan | 13 | 421 | Dengue | 67.7 | 27.1 | +0.068 (0.467) | -0.103 (0.261) | `SUPPORTED_DISTRICT_OUTBREAK_AND_PANEL_ANALYSIS` |
| **Washim** | Amravati_Vidarbha_West | 13 | 271 | Dengue | 20.9 | 26.8 | +0.053 (0.566) | +0.098 (0.288) | `SUPPORTED_DISTRICT_OUTBREAK_AND_PANEL_ANALYSIS` |
| **Jalgaon** | Nashik | 13 | 492 | Dengue | 20.2 | 26.6 | +0.097 (0.299) | +0.082 (0.374) | `SUPPORTED_DISTRICT_OUTBREAK_AND_PANEL_ANALYSIS` |
| **Parbhani** | Marathwada | 12 | 1309 | Chikungunya | 21.4 | 27.1 | +0.157 (0.090) | +0.030 (0.748) | `SUPPORTED_DISTRICT_OUTBREAK_AND_PANEL_ANALYSIS` |

## 9. Generated Visualizations (`reports/climate_intelligence/figures/`)

- `reports/climate_intelligence/figures/01_rainfall_vs_disease_trends.png`
- `reports/climate_intelligence/figures/02_temperature_vs_disease_trends.png`
- `reports/climate_intelligence/figures/03_humidity_vs_disease_trends_pune.png`
- `reports/climate_intelligence/figures/04_lagged_climate_disease_relationships.png`
- `reports/climate_intelligence/figures/05_district_climate_disease_comparison.png`
- `reports/climate_intelligence/figures/06_climate_disease_correlation_heatmap.png`
- `reports/climate_intelligence/figures/07_shap_and_partial_dependence_climate.png`

## 10. Unsupported Analyses & Methodological Limitations

1. **Statewide Relative Humidity Analysis Cannot Be Performed**: `humid_tel_hr_cwprs_mh_2021_2025.csv` contains measurements for a single station (`CWPRS Khadakvasla, Pune`) from `2024-05-11` to `2025-06-09`. Extrapolating Pune's relative humidity to coastal Konkan or semi-arid Vidarbha/Marathwada would be scientifically invalid.
2. **2009–2022 Historical Records Cannot Be Aligned with 2023–2025 IMD Gridded Climate**: `maharashtra_epiclim_filtered.csv` ends on `2022-06-15`, whereas all three IMD NetCDF files begin on `2023-01-01`.
3. **Mumbai Suburban Climate Alignment Unavailable**: `Mumbai Suburban` (`3` outbreak records in 2023–2025) has no latitude/longitude coordinates in the project lookup table and is excluded from gridded climate joins.
4. **Zero-Record Districts & Diseases**: `Mumbai City` and `Ratnagiri` have `0` reported outbreaks in the 2023–2025 IDSP dataset; `Heat-related illness` has `0` records across all disease files; 17 rare diseases have `< 8` climate-aligned records and cannot support standalone parametric lag inference.
5. **Non-Causal Interpretation Boundary**: Observed correlation, regression association, or ML/SHAP feature importance reflects statistical co-movement and predictive utility only; it does NOT establish causation.
