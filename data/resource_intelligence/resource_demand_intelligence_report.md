# JEEVAN-DRISHTI Phase 11: Healthcare Resource Demand Intelligence Report

> **Engine Version**: `phase11-v1.0.0` | **Audit Date**: `2026-10-31`
> **Mandatory Scientific Disclaimer**: NO EMPIRICAL HEALTHCARE RESOURCE DATASET EXISTS IN THE WORKSPACE. Actual hospital admissions, available/occupied beds, ICU capacity, testing volume, medicine stock, and facility counts are UNAVAILABLE (NaN) and are NEVER invented. All resource demand projections produced by the scenario calculator are HYPOTHETICAL WHAT-IF CALCULATIONS based solely on visible, user-editable assumptions.

---

## 1. Executive Summary & Workspace Data Availability Audit

Before implementing any resource demand analysis or forecasting, the entire **JEEVAN-DRISHTI** workspace (`data/raw/`, `data/processed/`, `data/features/`, `data/final_training/`, `data/geospatial/`, `data/scenarios/`, and `data/reports/dataset_inventory.csv`) was audited across all **9 required healthcare resource categories**:

| Variable Key | Variable Label | Domain Category | Present in Workspace | Coverage | Missingness (%) | Engine Handling Mode |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| `hospital_admissions` | Hospital Admissions (Inpatient / IPD) | Clinical / Hospital Utilization | `False` | 0 / 36 districts (0.0%) | `100.00%` | `EXPLICIT_NAN_PLUS_HYPOTHETICAL_SCENARIO_CALCULATOR_FALLBACK` |
| `available_beds` | Available General / Isolation Beds | Hospital Bed Capacity | `False` | 0 / 36 districts (0.0%) | `100.00%` | `EXPLICIT_NAN_PLUS_HYPOTHETICAL_SCENARIO_CALCULATOR_FALLBACK` |
| `occupied_beds` | Occupied Hospital Beds | Hospital Bed Utilization | `False` | 0 / 36 districts (0.0%) | `100.00%` | `EXPLICIT_NAN_PLUS_HYPOTHETICAL_SCENARIO_CALCULATOR_FALLBACK` |
| `icu_capacity` | ICU Beds & Ventilator Capacity | Critical Care Capacity | `False` | 0 / 36 districts (0.0%) | `100.00%` | `EXPLICIT_NAN_PLUS_HYPOTHETICAL_SCENARIO_CALCULATOR_FALLBACK` |
| `testing_volume` | Diagnostic Laboratory Testing Volume (ELISA/RT-PCR/RDT) | Laboratory Surveillance | `False` | 0 / 36 districts (0.0%) | `100.00%` | `EXPLICIT_NAN_PLUS_HYPOTHETICAL_SCENARIO_CALCULATOR_FALLBACK` |
| `medicine_consumption` | Medicine & IV Fluid Consumption | Pharmacy / Supply Chain | `False` | 0 / 36 districts (0.0%) | `100.00%` | `EXPLICIT_NAN_PLUS_HYPOTHETICAL_SCENARIO_CALCULATOR_FALLBACK` |
| `medicine_stock` | Medicine, ORS & IV Fluid Stock Levels | Pharmacy / Supply Chain | `False` | 0 / 36 districts (0.0%) | `100.00%` | `EXPLICIT_NAN_PLUS_HYPOTHETICAL_SCENARIO_CALCULATOR_FALLBACK` |
| `healthcare_facilities` | Healthcare Facilities Count (PHC/CHC/DH/Medical Colleges) | Health Infrastructure | `False` | 0 / 36 districts (0.0%) | `100.00%` | `EXPLICIT_NAN_PLUS_HYPOTHETICAL_SCENARIO_CALCULATOR_FALLBACK` |
| `district_health_infrastructure` | District Health Infrastructure & Staffing | Health Infrastructure | `False` | 0 / 36 districts (0.0%) | `100.00%` | `EXPLICIT_NAN_PLUS_HYPOTHETICAL_SCENARIO_CALCULATOR_FALLBACK` |
| `observed_outbreak_cases` | IDSP Reported Weekly Outbreak Cases (Upstream Surveillance Driver) | Epidemiological Surveillance (Not Hospital Admissions) | `True` | 34 / 36 districts (511 outbreak weeks) | `0.00%` | `UPSTREAM_EPIDEMIOLOGICAL_CONTEXT_ONLY` |
| `observed_outbreak_deaths` | IDSP Reported Outbreak Deaths (Upstream Severity Driver) | Epidemiological Surveillance | `True` | 34 / 36 districts (244 non-null death records) | `52.25%` | `UPSTREAM_EPIDEMIOLOGICAL_CONTEXT_ONLY` |
| `phase6_case_forecasts` | Phase 6 Multi-Horizon Disease Case Forecasts (7d, 14d, 30d) | Upstream Case Forecast (Not Resource Forecast) | `True` | 34 / 36 districts | `0.00%` | `UPSTREAM_EPIDEMIOLOGICAL_CONTEXT_ONLY` |

### Key Findings from the Data Availability Assessment
1. **Zero Empirical Healthcare Resource Datasets Exist in the Workspace**: None of the project files contain hospital admissions (`IPD`), available beds, occupied beds, ICU/ventilator capacity, diagnostic testing volume, medicine consumption/stock, or health facility inventories (`0 / 36` districts, `100%` missingness).
2. **Strict Non-Fabrication Enforcement**: In compliance with **Prompt 11 Sections 2, 4, and 5**, this engine **does not invent hospital capacities, bed counts, or medicine stock values**, and **does not convert disease cases into empirical hospital admissions using hidden or arbitrary percentages**.
3. **Why Empirical ML Resource Forecasting Is Blocked**: Because historical resource target time series (`hospital_admissions_t`, `occupied_beds_t`, `testing_volume_t`) are `100%` absent (`NaN`), training an ETS, SARIMA, or XGBoost model on synthetic targets derived from multiplying disease cases by a constant would be circular and scientifically invalid. Empirical resource forecasting is therefore explicitly marked `UNSUPPORTED_NO_HISTORICAL_RESOURCE_DATA`.
4. **Activation of the Transparent Hypothetical Scenario Calculator Fallback (Section 5)**: To enable public-health contingency planning without fabricating data, the engine implements a deterministic, user-editable **Hypothetical Scenario Calculator** (`src/engines/resource_intelligence/scenario_calculator.py`) where every assumption is explicitly visible, editable, and stamped with `HYPOTHETICAL_SCENARIO_NOT_ACTUAL_CAPACITY_OR_FORECAST`.

---

## 2. Verified Upstream Epidemiological Demand Drivers (`2023–2025`)

While hospital utilization records are absent, the project contains verified district-level epidemiological surveillance caseloads (`2023–2025` IDSP outbreaks across `34 / 36` active districts, plus Phase 6 multi-horizon case forecasts). These upstream epidemiological metrics are preserved strictly as optional baseline anchors for the hypothetical scenario calculator:

| Rank | District ID | District Name | Division | Outbreak Weeks (`23–25`) | Total Cases (`23–25`) | Mean Cases / Active Wk | Peak Weekly Cases (`23–25`) | Phase 6 `7d` Forecast Mean | Dominant Disease (`23–25`) | Empirical Hospital Beds / Admissions |
| :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |
| 1 | `MH_D21` | **Nanded** | Marathwada | 25 | 1518 | 60.72 | **1000** | 29.80 | Acute Diarrhoeal Disease | `NaN (Insufficient Data)` |
| 2 | `MH_D25` | **Parbhani** | Marathwada | 12 | 1309 | 109.08 | **983** | 37.19 | Food Poisoning | `NaN (Insufficient Data)` |
| 3 | `MH_D16` | **Kolhapur** | Pune | 42 | 2335 | 55.60 | **651** | 27.16 | Food Poisoning | `NaN (Insufficient Data)` |
| 4 | `MH_D11` | **Gadchiroli** | Nagpur_Vidarbha_East | 27 | 1432 | 53.04 | **576** | 17.79 | Malaria | `NaN (Insufficient Data)` |
| 5 | `MH_D07` | **Chandrapur** | Nagpur_Vidarbha_East | 12 | 490 | 40.83 | **283** | 19.34 | Food Poisoning | `NaN (Insufficient Data)` |
| 6 | `MH_D08` | **Chhatrapati Sambhajinagar** | Marathwada | 14 | 602 | 43.00 | **255** | 18.16 | Food Poisoning | `NaN (Insufficient Data)` |
| 7 | `MH_D24` | **Palghar** | Konkan | 13 | 421 | 32.38 | **249** | 19.20 | Acute Diarrhoeal Disease | `NaN (Insufficient Data)` |
| 8 | `MH_D29` | **Sangli** | Pune | 14 | 743 | 53.07 | **246** | 30.01 | Food Poisoning | `NaN (Insufficient Data)` |
| 9 | `MH_D06` | **Buldhana** | Amravati_Vidarbha_West | 8 | 470 | 58.75 | **208** | 45.14 | Food Poisoning | `NaN (Insufficient Data)` |
| 10 | `MH_D22` | **Nandurbar** | Nashik | 6 | 402 | 67.00 | **193** | 44.41 | Acute Diarrhoeal Disease | `NaN (Insufficient Data)` |

- **Insufficient Surveillance Anchor Districts (`2 / 36`)**: `Mumbai City` (`MH_D18`) and `Ratnagiri` (`MH_D28`) have `0` reported outbreak events in `2023–2025`. For these districts, surveillance-anchored scenarios automatically return `Insufficient Data`, while users can still evaluate hypothetical demand using `case_load_anchor_mode='user_entered_cases'`.

---

## 3. Hypothetical Scenario Calculator Methodology (Section 5 Fallback)

Given a baseline weekly case count $C_{\text{base}}$ (either user-entered or anchored to observed/forecasted surveillance cases) and visible, user-editable parameters:
- Hypothetical case surge percentage $\Delta_{\%} \in [-90\%, +1000\%]$
- Assumed inpatient admission proportion $\alpha_{\text{adm}} \in [0, 1]$
- Assumed ICU share of admissions $\alpha_{\text{icu}} \in [0, 1]$
- Assumed average length of stay $L_{\text{stay}} \in [1, 30]$ days
- Assumed diagnostic tests per case $\tau_{\text{test}}$, IV fluid units per admission $\nu_{\text{iv}}$, and treatment courses per case $\mu_{\text{rx}}$
- Optional user-assumed reference general beds $B_{\text{gen}}^{\text{assumed}}$ and ICU beds $B_{\text{icu}}^{\text{assumed}}$

the calculator computes:
$$C_{\text{proj}} = C_{\text{base}} \times \left(1 + \frac{\Delta_{\%}}{100}\right), \quad A_{\text{wk}} = C_{\text{proj}} \times \alpha_{\text{adm}}, \quad D_{\text{bed-days}} = A_{\text{wk}} \times L_{\text{stay}}$$
$$B_{\text{gen, req}} = \frac{A_{\text{wk}} \cdot (1 - \alpha_{\text{icu}}) \cdot L_{\text{stay}}}{7}, \quad B_{\text{icu, req}} = \frac{A_{\text{wk}} \cdot \alpha_{\text{icu}} \cdot L_{\text{stay}}}{7}$$
$$U_{\text{gen}} = \frac{B_{\text{gen, req}}}{B_{\text{gen}}^{\text{assumed}}}, \quad U_{\text{icu}} = \frac{B_{\text{icu, req}}}{B_{\text{icu}}^{\text{assumed}}}$$

### Benchmark Hypothetical Scenarios Evaluated Across All 36 Districts

| Scenario ID | Anchor Mode | Surge (`%`) | Assumed Adm (`%`) | Assumed ICU (`%`) | Assumed LOS (`d`) | Statewide Hypothetical Weekly Cases | Statewide Hypothetical Weekly Admissions | Statewide Concurrent General Beds | Statewide Concurrent ICU Beds | Statewide Diagnostic Tests | Districts `>100%` of Assumed Ward |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `RES_SCEN_01_FORECAST_BASELINE_MODERATE` | `phase6_forecast_7d_cases` | `+0%` | `15%` | `10%` | `4.5d` | 879.6 | 131.9 | **76.4** | **8.5** | 2639 | `0 / 34` |
| `RES_SCEN_02_MONSOON_SURGE_PLUS_50PCT` | `phase6_forecast_7d_cases` | `+50%` | `22%` | `12%` | `5.0d` | 1319.5 | 290.3 | **182.5** | **24.9** | 5278 | `0 / 34` |
| `RES_SCEN_03_HISTORICAL_PEAK_STRESS_TEST` | `observed_peak_weekly_cases` | `+25%` | `20%` | `10%` | `4.0d` | 8732.5 | 1746.5 | **898.2** | **99.8** | 30564 | `4 / 34` |
| `RES_SCEN_04_ACUTE_DIARRHOEAL_REHYDRATION_FOCUS` | `observed_mean_weekly_cases` | `+40%` | `18%` | `5%` | `2.5d` | 1761.4 | 317.0 | **107.6** | **5.7** | 3523 | `0 / 34` |

### Top 10 Districts Under `RES_SCEN_03_HISTORICAL_PEAK_STRESS_TEST` (Hypothetical Peak Weekly Load +25% vs. User-Assumed 50-Bed Isolation Ward)

| District ID | District Name | Peak Weekly Base Cases | Hypothetical Projected Cases (`+25%`) | Hypothetical Weekly Admissions (`20%`) | Concurrent General Beds (`LOS=4d`) | Concurrent ICU Beds | Utilization of Assumed 50-Bed Ward | Hypothetical Shortage Indicator |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `MH_D21` | **Nanded** | 1000 | 1250.0 | 250.0 | **128.57** | 14.29 | `257.1%` | `HYPOTHETICAL_CAPACITY_DEFICIT_EXCEEDS_100PCT_ASSUMED_BEDS` |
| `MH_D25` | **Parbhani** | 983 | 1228.8 | 245.8 | **126.39** | 14.04 | `252.8%` | `HYPOTHETICAL_CAPACITY_DEFICIT_EXCEEDS_100PCT_ASSUMED_BEDS` |
| `MH_D16` | **Kolhapur** | 651 | 813.8 | 162.8 | **83.70** | 9.30 | `167.4%` | `HYPOTHETICAL_CAPACITY_DEFICIT_EXCEEDS_100PCT_ASSUMED_BEDS` |
| `MH_D11` | **Gadchiroli** | 576 | 720.0 | 144.0 | **74.06** | 8.23 | `148.1%` | `HYPOTHETICAL_CAPACITY_DEFICIT_EXCEEDS_100PCT_ASSUMED_BEDS` |
| `MH_D07` | **Chandrapur** | 283 | 353.8 | 70.8 | **36.39** | 4.04 | `72.8%` | `HYPOTHETICAL_MODERATE_LOAD_50_TO_80PCT_ASSUMED_BEDS` |
| `MH_D08` | **Chhatrapati Sambhajinagar** | 255 | 318.8 | 63.8 | **32.79** | 3.64 | `65.6%` | `HYPOTHETICAL_MODERATE_LOAD_50_TO_80PCT_ASSUMED_BEDS` |
| `MH_D24` | **Palghar** | 249 | 311.2 | 62.2 | **32.01** | 3.56 | `64.0%` | `HYPOTHETICAL_MODERATE_LOAD_50_TO_80PCT_ASSUMED_BEDS` |
| `MH_D29` | **Sangli** | 246 | 307.5 | 61.5 | **31.63** | 3.51 | `63.3%` | `HYPOTHETICAL_MODERATE_LOAD_50_TO_80PCT_ASSUMED_BEDS` |
| `MH_D06` | **Buldhana** | 208 | 260.0 | 52.0 | **26.74** | 2.97 | `53.5%` | `HYPOTHETICAL_MODERATE_LOAD_50_TO_80PCT_ASSUMED_BEDS` |
| `MH_D22` | **Nandurbar** | 193 | 241.2 | 48.2 | **24.81** | 2.76 | `49.6%` | `HYPOTHETICAL_WITHIN_ASSUMED_CAPACITY_LT50PCT` |

---

## 4. Generated Datasets, Figures & Frontend Integration Artifacts

- **Data Completeness Audit**: `data/resource_intelligence/resource_data_completeness_audit.csv`
- **District Resource & Upstream Summary (`36` districts)**: `data/resource_intelligence/district_resource_summary.csv`
- **Historical Weekly Epidemiological Demand Trends (`2023–2025`)**: `data/resource_intelligence/historical_epidemiological_demand_trends.csv`
- **Benchmark Hypothetical Scenarios Catalog**: `data/resource_intelligence/hypothetical_resource_scenarios_catalog.csv`
- **District-Level Scenario Evaluations (`4 × 36 = 144` rows)**: `data/resource_intelligence/hypothetical_district_scenario_evaluations.csv`
- **Frontend Integration Payload (`6` sections)**: `data/resource_intelligence/frontend_integration_payload.json`
- **Interactive HTML/JS Scenario Calculator**: `reports/resource_intelligence/interactive_resource_demand_calculator.html`
- **Static PNG Figures (`4` charts)**: `reports/resource_intelligence/figures/01_*.png` through `04_*.png`
