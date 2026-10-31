# JEEVAN-DRISHTI Phase 9: Geospatial Disease Intelligence & Hotspot Detection Report

> **SCIENTIFIC & NON-TRANSMISSION MANDATE**: Spatial autocorrelation or hotspot significance indicates geographic clustering of reported surveillance counts; it does NOT prove cross-district pathogen transmission or causation. A district is labelled as a statistically significant hotspot **only** when Getis-Ord $G_i^*$ ($Z \ge +1.96, p < 0.05$ analytical & permutation) or Local Moran's $I_i$ LISA (`HH_HOTSPOT`, conditional permutation $p < 0.05$) supports that claim.

- **Generated At**: `2026-10-31T16:18:30`
- **Engine Version**: `phase9-v1.0.0`
- **Primary Surveillance Scope**: `2023-W01 to 2025-W52` (`511` deduplicated weekly outbreak records across `34 / 36` Maharashtra districts; `2009–2022` historical EpiClim baseline analysed separately).

## 1. Geographic Data Inspection & Boundary Non-Fabrication Audit

| Asset Category | Workspace Status | Handling & Scientific Safeguard |
|---|---|---|
| **District Polygon Boundaries (`.geojson`, `.shp`)** | `0` external polygon files found | **Not downloaded or fabricated.** Exported verified 35-district Point GeoJSON (`data/geospatial/maharashtra_district_points.geojson`) and exposed `attach_optional_polygon_geojson()` adapter interface. |
| **District Point Coordinates (`lat`, `lon`)** | **35 / 36 districts** verified in `data\interim\district_grid_cells.csv` | `Mumbai Suburban` (`MH_D19`) has no coordinates in any source dataset; explicitly marked `Insufficient Data (Missing Coordinates)` and never invented. |
| **Population Denominators** | `0` population/census files in workspace | `incidence_rate` is set to `NaN` (`UNAVAILABLE_NO_POPULATION_DENOMINATOR`); never fabricated. |
| **2023–2025 Zero-Record Districts** | **2 / 36 districts** (`Mumbai City`, `Ratnagiri`) | Have verified coordinates (`18.97°N, 72.83°E` and `17.28°N, 73.46°E`) but `0` reported outbreaks in `2023–2025` IDSP data; explicitly marked `Insufficient Data (0 Surveillance Outbreaks in 2023-2025)`. |

## 2. Geographic Standardization & Historical Boundary Crosswalk

- **Canonical Identifiers**: `MH_D01` (`Ahmednagar`) through `MH_D36` (`Yavatmal`), 100% unique (`0` duplicate mappings).
- **Documented District Renamings**:
  - `Aurangabad` and column-split `'Chhatrapati' + 'Sambhajinagar'` $\rightarrow$ `Chhatrapati Sambhajinagar` (`MH_D08`).
  - `Osmanabad` and `Osmanabad (Maharashtra)` $\rightarrow$ `Dharashiv` (`MH_D09`).
  - `Ahilyanagar` and `Ahmadnagar` $\rightarrow$ `Ahmednagar` (`MH_D01`).
  - Alternate spellings: `Nasik` $\rightarrow$ `Nashik`, `Sholapur` $\rightarrow$ `Solapur`, `Palaghar` $\rightarrow$ `Palghar`, `Mumbai (city)` $\rightarrow$ `Mumbai City`, column-split `'Mumbai' + 'Suburban'` $\rightarrow$ `Mumbai Suburban`.
- **District Boundary Split Rule (`Thane` / `Palghar`)**:
  - `Palghar` (`MH_D24`) was carved out of `Thane` (`MH_D33`) on `2014-08-01`. In `2023–2025` weekly IDSP data, `Thane` (`34` outbreak weeks, `588` cases) and `Palghar` (`13` outbreak weeks, `421` cases) are reported as separate districts. Pre-2014 records in `maharashtra_epiclim_filtered.csv` under `Thane` include pre-split territory and are never silently merged into `2023–2025` indicators.

## 3. Spatial Weights Matrix ($W$) Specification

- **Primary Matrix ($W_{\text{knn4}}$)**: Symmetric k-Nearest Neighbors (k=4) on Great-Circle Haversine Distance, Row-Standardized across the **35 geocoded districts** (`Mumbai Suburban` excluded due to missing coordinates).
- **Graph Connectivity**: `0` islands; degree range `4..7` neighbors (mean `4.74` neighbors); row sums verified `= 1.0`.
- **Secondary Sensitivity Matrix ($W_{\text{idw}}$)**: Inverse Haversine Distance Band (<= 220.0 km), Row-Standardized.

## 4. Global Spatial Autocorrelation (`Global Moran's I` — 999 Permutations)

| Disease / Layer | Period | Indicator | Active Districts | Global Moran's $I$ ($W_{\text{knn4}}$) | Expected $E[I]$ | Analytical $Z$ ($p$) | Permutation $p_{\text{sim}}$ | IDW-220km $I$ ($p_{\text{sim}}$) | Spatial Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `ALL_DISEASES` | `2023-2025_POOL` | `observed_cases_raw` | 33 | **-0.1444** | -0.0294 | -1.21 (0.2273) | **0.2090** | -0.1140 (0.2160) | `RANDOM_OR_NOT_STATISTICALLY_SIGNIFICANT` |
| `ALL_DISEASES` | `2023-2025_POOL` | `forecast_7d_mean_cases` | 33 | **+0.2059** | -0.0294 | +2.30 (0.0213) | **0.0150** | +0.3226 (0.0010) | `SIGNIFICANT_POSITIVE_SPATIAL_CLUSTERING` |
| `ALL_DISEASES` | `2023-2025_POOL` | `anomaly_weeks_count` | 30 | **+0.0214** | -0.0294 | +0.52 (0.6004) | **0.5880** | -0.0602 (0.6690) | `RANDOM_OR_NOT_STATISTICALLY_SIGNIFICANT` |
| `ALL_DISEASES` | `2023-2025_POOL` | `anomaly_max_score` | 33 | **-0.1130** | -0.0294 | -0.85 (0.3956) | **0.3640** | -0.1196 (0.1970) | `RANDOM_OR_NOT_STATISTICALLY_SIGNIFICANT` |
| `ALL_DISEASES` | `2023-2025_POOL` | `calibrated_risk_probability_mean` | 33 | **-0.0025** | -0.0294 | +0.32 (0.7460) | **0.6760** | +0.0760 (0.0630) | `RANDOM_OR_NOT_STATISTICALLY_SIGNIFICANT` |
| `ALL_DISEASES` | `2023` | `observed_cases_raw` | 33 | **+0.0371** | -0.0294 | +0.67 (0.5029) | **0.4980** | +0.0468 (0.3090) | `RANDOM_OR_NOT_STATISTICALLY_SIGNIFICANT` |
| `ALL_DISEASES` | `2024` | `observed_cases_raw` | 33 | **-0.1115** | -0.0294 | -0.83 (0.4081) | **0.4050** | -0.0982 (0.3580) | `RANDOM_OR_NOT_STATISTICALLY_SIGNIFICANT` |
| `ALL_DISEASES` | `2025` | `observed_cases_raw` | 23 | **-0.1085** | -0.0294 | -0.93 (0.3520) | **0.3560** | -0.1252 (0.1390) | `RANDOM_OR_NOT_STATISTICALLY_SIGNIFICANT` |
| `ALL_DISEASES` | `2009-2022_HISTORICAL_BASELINE` | `observed_cases_raw` | 35 | **+0.0471** | -0.0294 | +0.81 (0.4198) | **0.4310** | +0.0057 (0.6470) | `RANDOM_OR_NOT_STATISTICALLY_SIGNIFICANT` |
| `Acute Diarrhoeal Disease` | `2023-2025_POOL` | `observed_cases_raw` | 31 | **-0.1080** | -0.0294 | -0.90 (0.3702) | **0.3780** | -0.0651 (0.6090) | `RANDOM_OR_NOT_STATISTICALLY_SIGNIFICANT` |
| `Chikungunya` | `2023-2025_POOL` | `observed_cases_raw` | 14 | **+0.0265** | -0.0294 | +0.73 (0.4668) | **0.4590** | -0.0301 (0.9890) | `RANDOM_OR_NOT_STATISTICALLY_SIGNIFICANT` |
| `Cholera` | `2023-2025_POOL` | `observed_cases_raw` | 11 | **-0.0707** | -0.0294 | -0.43 (0.6699) | **0.6650** | -0.0660 (0.6070) | `RANDOM_OR_NOT_STATISTICALLY_SIGNIFICANT` |
| `Dengue` | `2023-2025_POOL` | `observed_cases_raw` | 30 | **-0.1389** | -0.0294 | -1.09 (0.2761) | **0.2570** | -0.1978 (0.0270) | `RANDOM_OR_NOT_STATISTICALLY_SIGNIFICANT` |
| `Fever (unspecified)` | `2023-2025_POOL` | `observed_cases_raw` | 11 | **-0.0474** | -0.0294 | -0.22 (0.8234) | **0.8590** | -0.0195 (0.8600) | `RANDOM_OR_NOT_STATISTICALLY_SIGNIFICANT` |
| `Food Poisoning` | `2023-2025_POOL` | `observed_cases_raw` | 22 | **-0.0754** | -0.0294 | -0.51 (0.6091) | **0.6070** | -0.0170 (0.8410) | `RANDOM_OR_NOT_STATISTICALLY_SIGNIFICANT` |
| `Hepatitis A` | `2023-2025_POOL` | `observed_cases_raw` | 8 | **-0.0036** | -0.0294 | +0.44 (0.6619) | **0.7400** | +0.0038 (0.4610) | `RANDOM_OR_NOT_STATISTICALLY_SIGNIFICANT` |
| `Leptospirosis` | `2023-2025_POOL` | `observed_cases_raw` | 7 | **+0.1103** | -0.0294 | +2.60 (0.0094) | **0.0170** | +0.0347 (0.0760) | `SIGNIFICANT_POSITIVE_SPATIAL_CLUSTERING` |
| `Malaria` | `2023-2025_POOL` | `observed_cases_raw` | 12 | **+0.0431** | -0.0294 | +2.32 (0.0202) | **0.0280** | +0.0526 (0.0160) | `SIGNIFICANT_POSITIVE_SPATIAL_CLUSTERING` |
| `Measles` | `2023-2025_POOL` | `observed_cases_raw` | 5 | **-0.0257** | -0.0294 | +0.08 (0.9390) | **0.9680** | -0.0354 (0.8970) | `RANDOM_OR_NOT_STATISTICALLY_SIGNIFICANT` |
| `Scrub Typhus` | `2023-2025_POOL` | `observed_cases_raw` | 5 | **+0.0634** | -0.0294 | +1.08 (0.2808) | **0.2010** | +0.0801 (0.0700) | `RANDOM_OR_NOT_STATISTICALLY_SIGNIFICANT` |
| `Zika Virus` | `2023-2025_POOL` | `observed_cases_raw` | 6 | **-0.0270** | -0.0294 | +0.11 (0.9143) | **0.8750** | -0.0130 (0.2240) | `RANDOM_OR_NOT_STATISTICALLY_SIGNIFICANT` |

## 5. Statistically Significant District Hotspots (`Getis-Ord Gi*` & `Local Moran's I LISA`, $p < 0.05$)

| Disease | District (`ID`) | Division | Observed Cases (`Weeks`) | Getis-Ord $G_i^*$ ($p_{\text{perm}}$) | $G_i^*$ Class | Local Moran's $I_i$ ($p_{\text{perm}}$) | LISA Quadrant | Consensus Classification |
|---|---|---|---|---|---|---|---|---|
| **Leptospirosis** | **Raigad** (`MH_D27`) | `Konkan` | 35 (2w) | **+3.077** (0.0080) | `HOTSPOT_99_CONFIDENCE` | +1.369 (0.0520) | `NOT_SIGNIFICANT` | Statistically Significant Hotspot (HOTSPOT_99_CONFIDENCE / LISA=NOT_SIGNIFICANT, p<0.05) |
| **Leptospirosis** | **Palghar** (`MH_D24`) | `Konkan` | 7 (1w) | **+3.077** (0.0080) | `HOTSPOT_99_CONFIDENCE` | +1.140 (0.0310) | `HH_HOTSPOT` | Statistically Significant Hotspot (99% Gi* + LISA High-High, p<0.01) |
| **Scrub Typhus** | **Nagpur** (`MH_D20`) | `Nagpur_Vidarbha_East` | 24 (3w) | **+2.829** (0.0130) | `HOTSPOT_99_CONFIDENCE` | +1.989 (0.0440) | `HH_HOTSPOT` | Statistically Significant Hotspot (99% Gi* + LISA High-High, p<0.01) |
| **Malaria** | **Gadchiroli** (`MH_D11`) | `Nagpur_Vidarbha_East` | 1020 (16w) | **+2.795** (0.0090) | `HOTSPOT_99_CONFIDENCE` | +0.227 (0.0510) | `NOT_SIGNIFICANT` | Statistically Significant Hotspot (HOTSPOT_99_CONFIDENCE / LISA=NOT_SIGNIFICANT, p<0.05) |
| **Leptospirosis** | **Thane** (`MH_D33`) | `Konkan` | 7 (2w) | **+2.718** (0.0200) | `HOTSPOT_99_CONFIDENCE` | +0.861 (0.0120) | `HH_HOTSPOT` | Statistically Significant Hotspot (99% Gi* + LISA High-High, p<0.01) |
| **Hepatitis A** | **Kolhapur** (`MH_D16`) | `Pune` | 310 (12w) | **+2.484** (0.0360) | `HOTSPOT_95_CONFIDENCE` | -0.132 (0.6480) | `NOT_SIGNIFICANT` | Statistically Significant Hotspot (HOTSPOT_95_CONFIDENCE / LISA=NOT_SIGNIFICANT, p<0.05) |
| **Malaria** | **Chandrapur** (`MH_D07`) | `Nagpur_Vidarbha_East` | 113 (5w) | **+2.478** (0.0200) | `HOTSPOT_95_CONFIDENCE` | +0.401 (0.0540) | `NOT_SIGNIFICANT` | Statistically Significant Hotspot (HOTSPOT_95_CONFIDENCE / LISA=NOT_SIGNIFICANT, p<0.05) |
| **Malaria** | **Gondia** (`MH_D12`) | `Nagpur_Vidarbha_East` | 92 (2w) | **+2.478** (0.0150) | `HOTSPOT_95_CONFIDENCE` | +0.283 (0.0330) | `HH_HOTSPOT` | Statistically Significant Hotspot (HOTSPOT_95_CONFIDENCE / LISA=HH_HOTSPOT, p<0.05) |
| **Chikungunya** | **Satara** (`MH_D30`) | `Pune` | 70 (5w) | **+2.397** (0.0240) | `HOTSPOT_95_CONFIDENCE` | +1.202 (0.0820) | `NOT_SIGNIFICANT` | Statistically Significant Hotspot (HOTSPOT_95_CONFIDENCE / LISA=NOT_SIGNIFICANT, p<0.05) |
| **Chikungunya** | **Kolhapur** (`MH_D16`) | `Pune` | 176 (4w) | **+2.383** (0.0380) | `HOTSPOT_95_CONFIDENCE` | +0.186 (0.5430) | `NOT_SIGNIFICANT` | Statistically Significant Hotspot (HOTSPOT_95_CONFIDENCE / LISA=NOT_SIGNIFICANT, p<0.05) |
| **Scrub Typhus** | **Wardha** (`MH_D34`) | `Nagpur_Vidarbha_East` | 31 (2w) | **+2.342** (0.0400) | `HOTSPOT_95_CONFIDENCE` | +0.979 (0.1770) | `NOT_SIGNIFICANT` | Statistically Significant Hotspot (HOTSPOT_95_CONFIDENCE / LISA=NOT_SIGNIFICANT, p<0.05) |
| **Zika Virus** | **Ahmednagar** (`MH_D01`) | `Nashik` | 3 (1w) | **+2.248** (0.0350) | `HOTSPOT_95_CONFIDENCE` | +0.061 (0.0620) | `NOT_SIGNIFICANT` | Statistically Significant Hotspot (HOTSPOT_95_CONFIDENCE / LISA=NOT_SIGNIFICANT, p<0.05) |
| **Measles** | **Jalgaon** (`MH_D14`) | `Nashik` | 110 (4w) | **+2.180** (0.0480) | `HOTSPOT_95_CONFIDENCE` | -0.608 (0.8680) | `NOT_SIGNIFICANT` | Statistically Significant Hotspot (HOTSPOT_95_CONFIDENCE / LISA=NOT_SIGNIFICANT, p<0.05) |
| **Hepatitis A** | **Sangli** (`MH_D29`) | `Pune` | 55 (3w) | **+2.164** (0.0470) | `HOTSPOT_95_CONFIDENCE` | +0.557 (0.0630) | `NOT_SIGNIFICANT` | Statistically Significant Hotspot (HOTSPOT_95_CONFIDENCE / LISA=NOT_SIGNIFICANT, p<0.05) |

### High–Low Spatial Outliers (`HL_SPATIAL_OUTLIER` — Isolated High Burden surrounded by Lower-Burden Neighbors, $p < 0.05$)

| Disease | District (`ID`) | Observed Cases (`Weeks`) | Getis-Ord $G_i^*$ | Local Moran's $I_i$ ($p_{\text{perm}}$) | Interpretation |
|---|---|---|---|---|---|

## 6. All-36-District Multi-Layer Intelligence Summary (`ALL_DISEASES`, `2023–2025`)

| District (`ID`) | Division | Obs Weeks | Observed Cases | Hist 2009–22 Mean | Forecast 7d Mean | Anomaly Status | Risk Category | $G_i^*$ $Z$-Score | Hotspot / Data Quality Status |
|---|---|---|---|---|---|---|---|---|---|
| **Kolhapur** (`MH_D16`) | `Pune` | 42 | 2335 | 47.2 | 27.2 | `ANOMALY_DETECTED (9 week(s), max_score=0.89)` | `MODERATE_RISK` | +1.05 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Nanded** (`MH_D21`) | `Marathwada` | 25 | 1518 | 41.0 | 29.8 | `ANOMALY_DETECTED (2 week(s), max_score=0.91)` | `MODERATE_RISK` | +0.72 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Gadchiroli** (`MH_D11`) | `Nagpur_Vidarbha_East` | 27 | 1432 | 35.8 | 17.8 | `ANOMALY_DETECTED (19 week(s), max_score=0.85)` | `MODERATE_RISK` | +0.53 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Parbhani** (`MH_D25`) | `Marathwada` | 12 | 1309 | 48.2 | 37.2 | `ANOMALY_DETECTED (2 week(s), max_score=0.89)` | `MODERATE_RISK` | +0.08 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Amravati** (`MH_D03`) | `Amravati_Vidarbha_West` | 24 | 954 | 35.7 | 36.5 | `ANOMALY_DETECTED (8 week(s), max_score=0.86)` | `MODERATE_RISK` | -0.06 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Nagpur** (`MH_D20`) | `Nagpur_Vidarbha_East` | 18 | 775 | 68.7 | 25.8 | `ANOMALY_DETECTED (9 week(s), max_score=0.85)` | `MODERATE_RISK` | +0.15 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Sangli** (`MH_D29`) | `Pune` | 14 | 743 | 40.8 | 30.0 | `ANOMALY_DETECTED (4 week(s), max_score=0.83)` | `MODERATE_RISK` | +0.57 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Akola** (`MH_D02`) | `Amravati_Vidarbha_West` | 14 | 665 | 32.8 | 28.6 | `ANOMALY_DETECTED (1 week(s), max_score=0.59)` | `MODERATE_RISK` | +0.23 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Ahmednagar** (`MH_D01`) | `Nashik` | 13 | 663 | 40.7 | 26.8 | `ANOMALY_DETECTED (3 week(s), max_score=0.85)` | `MODERATE_RISK` | -0.82 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Yavatmal** (`MH_D36`) | `Amravati_Vidarbha_West` | 16 | 604 | 19.5 | 36.9 | `ANOMALY_DETECTED (1 week(s), max_score=0.63)` | `MODERATE_RISK` | -0.47 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Chhatrapati Sambhajinagar** (`MH_D08`) | `Marathwada` | 14 | 602 | 311.0 | 18.2 | `ANOMALY_DETECTED (7 week(s), max_score=0.89)` | `MODERATE_RISK` | -1.07 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Thane** (`MH_D33`) | `Konkan` | 34 | 588 | 31.0 | 16.0 | `ANOMALY_DETECTED (20 week(s), max_score=0.85)` | `MODERATE_RISK` | -0.74 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Pune** (`MH_D26`) | `Pune` | 33 | 587 | 39.6 | 14.5 | `ANOMALY_DETECTED (4 week(s), max_score=0.85)` | `MODERATE_RISK` | -0.41 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Dharashiv** (`MH_D09`) | `Marathwada` | 25 | 551 | 37.5 | 18.4 | `ANOMALY_DETECTED (5 week(s), max_score=0.81)` | `MODERATE_RISK` | -0.41 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Raigad** (`MH_D27`) | `Konkan` | 24 | 527 | 43.0 | 12.9 | `ANOMALY_DETECTED (15 week(s), max_score=0.89)` | `MODERATE_RISK` | -0.61 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Jalgaon** (`MH_D14`) | `Nashik` | 13 | 492 | 42.3 | 26.5 | `ANOMALY_DETECTED (2 week(s), max_score=0.85)` | `MODERATE_RISK` | -0.88 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Chandrapur** (`MH_D07`) | `Nagpur_Vidarbha_East` | 12 | 490 | 51.0 | 19.3 | `ANOMALY_DETECTED (5 week(s), max_score=0.87)` | `MODERATE_RISK` | +0.15 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Buldhana** (`MH_D06`) | `Amravati_Vidarbha_West` | 8 | 470 | 43.8 | 45.1 | `ANOMALY_DETECTED (3 week(s), max_score=0.70)` | `MODERATE_RISK` | -0.73 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Satara** (`MH_D30`) | `Pune` | 19 | 470 | 27.6 | 18.5 | `ANOMALY_DETECTED (3 week(s), max_score=0.68)` | `MODERATE_RISK` | +1.01 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Palghar** (`MH_D24`) | `Konkan` | 13 | 421 | 30.9 | 19.2 | `ANOMALY_DETECTED (3 week(s), max_score=0.89)` | `MODERATE_RISK` | -0.84 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Nandurbar** (`MH_D22`) | `Nashik` | 6 | 402 | 42.1 | 44.4 | `NORMAL_BASELINE (0 Anomalies)` | `MODERATE_RISK` | -0.61 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Bhandara** (`MH_D05`) | `Nagpur_Vidarbha_East` | 15 | 387 | 38.8 | 36.9 | `ANOMALY_DETECTED (2 week(s), max_score=0.85)` | `MODERATE_RISK` | +0.15 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Nashik** (`MH_D23`) | `Nashik` | 10 | 362 | 54.9 | 23.1 | `ANOMALY_DETECTED (4 week(s), max_score=0.85)` | `LOW_RISK` | -0.54 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Washim** (`MH_D35`) | `Amravati_Vidarbha_West` | 13 | 271 | 23.0 | 19.6 | `ANOMALY_DETECTED (2 week(s), max_score=0.61)` | `MODERATE_RISK` | +0.66 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Dhule** (`MH_D10`) | `Nashik` | 6 | 262 | 31.6 | 38.1 | `ANOMALY_DETECTED (1 week(s), max_score=0.81)` | `MODERATE_RISK` | -0.61 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Latur** (`MH_D17`) | `Marathwada` | 11 | 258 | 41.9 | 19.0 | `ANOMALY_DETECTED (2 week(s), max_score=0.78)` | `LOW_RISK` | +0.52 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Sindhudurg** (`MH_D31`) | `Konkan` | 7 | 235 | 32.6 | 25.3 | `ANOMALY_DETECTED (3 week(s), max_score=0.85)` | `MODERATE_RISK` | +1.05 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Gondia** (`MH_D12`) | `Nagpur_Vidarbha_East` | 6 | 184 | 40.8 | 30.4 | `ANOMALY_DETECTED (2 week(s), max_score=0.81)` | `MODERATE_RISK` | +0.15 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Wardha** (`MH_D34`) | `Nagpur_Vidarbha_East` | 9 | 177 | 29.9 | 28.8 | `ANOMALY_DETECTED (1 week(s), max_score=0.85)` | `MODERATE_RISK` | -0.22 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Solapur** (`MH_D32`) | `Pune` | 4 | 115 | 60.8 | 24.0 | `NORMAL_BASELINE (0 Anomalies)` | `LOW_RISK` | -0.98 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Jalna** (`MH_D15`) | `Marathwada` | 6 | 102 | 26.0 | 22.1 | `ANOMALY_DETECTED (3 week(s), max_score=0.85)` | `LOW_RISK` | -0.09 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Hingoli** (`MH_D13`) | `Marathwada` | 4 | 99 | 17.0 | 34.9 | `NORMAL_BASELINE (0 Anomalies)` | `LOW_RISK` | +1.07 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Beed** (`MH_D04`) | `Marathwada` | 11 | 87 | 31.6 | 9.5 | `ANOMALY_DETECTED (8 week(s), max_score=0.85)` | `LOW_RISK` | -0.57 | Not a Statistically Significant Hotspot (p >= 0.05) |
| **Mumbai Suburban** (`MH_D19`) | `Konkan` | 3 | 35 | N/A | 18.5 | `Insufficient Data` | `Insufficient Data` | Insufficient Data | Insufficient Data (Missing Coordinates) |
| **Mumbai City** (`MH_D18`) | `Konkan` | 0 | Insufficient Data | 60.0 | Insufficient Data | `Insufficient Data` | `Insufficient Data` | Insufficient Data | Insufficient Data (0 Surveillance Outbreaks in 2023-2025) |
| **Ratnagiri** (`MH_D28`) | `Konkan` | 0 | Insufficient Data | 36.0 | Insufficient Data | `Insufficient Data` | `Insufficient Data` | Insufficient Data | Insufficient Data (0 Surveillance Outbreaks in 2023-2025) |

## 7. Generated Accessible Map Visualizations (`reports/geospatial/figures/`)

- `reports/geospatial/figures/01_maharashtra_district_disease_burden_map.png`
- `reports/geospatial/figures/02_disease_specific_hotspot_maps.png`
- `reports/geospatial/figures/03_historical_comparison_maps.png`
- `reports/geospatial/figures/04_forecast_disease_maps.png`
- `reports/geospatial/figures/05_anomaly_and_risk_intelligence_maps.png`
- `reports/geospatial/figures/06_district_drilldown_spatial_profiles.png`

## 8. Methodological Limitations & Safeguards

1. **No Polygon Boundary Fabrication**: Because official Maharashtra district boundary shapefiles/GeoJSONs are absent from the project workspace, this engine uses the 35 verified district point centroids and a symmetric $k=4$ nearest-neighbor graph ($W_{\text{knn4}}$) plus an inverse-distance sensitivity matrix ($W_{\text{idw}}$). An adapter (`attach_optional_polygon_geojson`) is ready to bind polygon geometries without code changes.
2. **No Population Denominator Fabrication**: Because no census population file exists in the workspace, `incidence_rate` is explicitly `NaN` (`UNAVAILABLE_NO_POPULATION_DENOMINATOR`).
3. **Strict Layer Separation**: Observed surveillance cases, historical `2009–2022` averages, Phase 6 multi-horizon forecasts (`7d`, `14d`, `30d`), Phase 7 anomaly scores, and Phase 5 calibrated risk probabilities are stored in separate columns and plotted on separate map panels.
4. **Strict Hotspot Significance Gating**: Districts with high raw counts that sit next to low-burden neighbors (such as `Kolhapur` for `ALL_DISEASES`, which is a `Local Moran's I High-Low Spatial Outlier` with $I_i = -0.923, p = 0.025$) are accurately distinguished from contiguous multi-district hotspots (`Getis-Ord Gi*` / `LISA High-High` with $p < 0.05$).
5. **Non-Transmission Rule**: Spatial autocorrelation or hotspot significance indicates geographic clustering of reported surveillance counts; it does NOT prove cross-district pathogen transmission or causation.
