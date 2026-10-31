# JEEVAN-DRISHTI: Dataset Inspection & ML Feasibility Analysis Report
**A Climate-Sensitive Public Health Early Warning and Decision Support System for Maharashtra**  
*Audit Version 2.0 (Post Contemporary 2023–2025 Outbreak Dataset Verification)*

---

## Executive Summary & Core Feasibility Verdict

> **Feasibility Verdict: YES, THIS WORKS!**  
> The contemporary outbreak dataset (`maharashtra_weekly_outbreak_records_2023_2025.csv` / `maharashtra_outbreaks_2023_2025.csv`) **completely resolves the primary temporal blocker** identified during the initial audit.  
> 
> - **Climate Data Window:** 2023-01-01 to 2025-12-31 (1,096 consecutive daily gridded readings).  
> - **New Disease Data Window:** 2023 to 2025 (Weeks 1 to 52 across 2023, 2024, and 2025).  
> - **Temporal Overlap:** **100% of the 575 contemporary outbreak records overlap directly** with the daily IMD gridded rainfall, $T_{\min}$, and $T_{\max}$ NetCDF datasets, as well as the 2024–2025 CWPRS Pune station humidity records.

---

## 1. Summary of All Audited Datasets

| Dataset Identifier | Primary Variable(s) | Units | Spatial Coverage / Resolution | Temporal Span & Frequency | Total Records / Shape | Integrity & Feasibility Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`maharashtra_weekly_outbreak_records_2023_2025.csv`** *(Contemporary IDSP)* | Outbreak alerts, Cases, Deaths, Dates, PDF sources | Counts, Calendar Dates | Maharashtra (34 districts represented, incl. Mumbai Suburban) | **2023-W01 to 2025-W52** (Weekly IDSP alerts) | **575 records** (19,242 cases, 201 deaths) | **Active & Ready**: 100% overlap with IMD climate series. |
| **`maharashtra_epiclim_filtered.csv`** *(Historical IDSP)* | Outbreak alerts, Cases, Deaths, LAI, Temp_C, preci | Counts, mm, °C | Maharashtra (35 districts; Mumbai Suburban missing) | **2009-07-23 to 2022-06-15** (Event-based) | **1,195 records** (53,575 cases, 340 deaths) | **Historical Baseline**: Useful for long-term climatology & seasonal priors. |
| **`imdweb_5dgbAskl.nc`** *(IMD Gridded Rainfall)* | Daily gridded precipitation (`/rain/rf`) | mm / day | Maharashtra bbox: 72.5°E–81.0°E, 15.5°N–22.25°N ($0.25^\circ \times 0.25^\circ$, 35x28 grid) | **2023-01-01 to 2025-12-31** (Daily) | **1,096 days** (1,074,080 data points) | **Active & Ready**: Zero missing days; covers all 36 districts. |
| **`imdweb_Oe9H04Bg.nc`** *(IMD Gridded Tmin)* | Daily minimum temperature (`/tmin/temperature`) | °C | Maharashtra bbox: 72.5°E–81.0°E, 15.5°N–22.25°N ($1.0^\circ \times 1.0^\circ$, 9x7 grid) | **2023-01-01 to 2025-12-31** (Daily) | **1,096 days** (69,048 data points) | **Active & Ready**: Zero missing days; coarse grid covers all districts. |
| **`imdweb_tXexQbB6.nc`** *(IMD Gridded Tmax)* | Daily maximum temperature (`/tmax/temperature`) | °C | Maharashtra bbox: 72.5°E–81.0°E, 15.5°N–22.25°N ($1.0^\circ \times 1.0^\circ$, 9x7 grid) | **2023-01-01 to 2025-12-31** (Daily) | **1,096 days** (69,048 data points) | **Active & Ready**: Zero missing days; coarse grid covers all districts. |
| **`humid_tel_hr_cwprs_mh_2021_2025.csv`** *(CWPRS Pune RH)* | Relative Humidity, Battery, Solar Radiation, Barometric P | % RH, V, W/m², hPa | Single Point: Pune (`CWPRS campus`, 18.4469°N, 73.7850°E) | **2024-05-11 to 2025-06-09** (Sub-hourly / Hourly) | **87,990 rows** (Single weather station) | **Station-Specific**: Validates Pune microclimate; cannot represent coastal/Vidarbha districts. |

---

## 2. Temporal & Spatial Alignment Feasibility

### 2.1 Temporal Overlap Matrix

- **Historical CSV + IMD NetCDFs:** 0 days overlap.
- **Contemporary CSV + IMD NetCDFs:** **3 Full Years (2023, 2024, 2025) complete overlap**.
- **Contemporary CSV + CWPRS Humidity:** Direct overlap for the monsoon and post-monsoon periods of 2024 and early 2025.

---

### 2.2 District Representation Across 36 Maharashtra Districts

A total of **36 official districts** must be covered under JEEVAN-DRISHTI:

1. **All 36 Districts** have continuous daily gridded coverage from IMD rainfall ($0.25^\circ \times 0.25^\circ$) and temperatures ($1.0^\circ \times 1.0^\circ$).
2. **Historical vs. Contemporary Disease Representation:**
   - In the historical (2009–2022) dataset, `Mumbai Suburban` had **0 outbreak records** (only `Mumbai City` was logged).
   - In the contemporary (2023–2025) dataset, `Mumbai Suburban` has **3 outbreak alerts** (including Zika Virus and Food Poisoning).
   - Across both datasets combined, **all 36 districts have documented outbreak surveillance data**.
   - `Ratnagiri` has 13 historical alerts, but 0 alerts in 2023–2025. Per our core guidelines: **Absence of an outbreak alert in IDSP is NOT assumed to be zero incidence; it indicates no localized cluster breached the epidemic threshold during that reporting week.**

---

## 3. Disease Coverage & Climate Modeling Suitability

| Disease Category | Key Diseases | 2009–2022 Records | 2023–2025 Records | Combined Records | Cases (Combined) | Deaths (Combined) | Primary Climate Drivers & Lag Dynamics |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Vector-Borne Diseases (VBD)** | Dengue, Chikungunya, Malaria, Zika Virus, JE | **543** | **282** | **825** | 25,600+ | 85 | **Rainfall accumulation (2–6 week lag)** creating breeding pools; **Mean Temp (24–32°C)** optimizing extrinsic incubation period (EIP); high humidity sustaining vector longevity. |
| **Water-Borne & Enteric Diseases** | Acute Diarrheal Disease (ADD), Cholera, Food Poisoning, Typhoid, Viral Hepatitis A & E | **587** | **206** | **793** | 43,800+ | 260 | **Flash rainfall / flooding (0–2 week lag)** washing sewage into drinking supplies; **Extreme heat** accelerating bacterial replication in unprotected water. |
| **Zoonotic Infections** | Leptospirosis, Scrub Typhus, Rabies | **45** | **20** | **65** | 1,400+ | 150 | **Monsoon waterlogging & soil contact (1–3 week lag)**; rodent urine runoff in urban and peri-urban water accumulations. |
| **Airborne & Viral Syndromes** | Measles, Mumps, Chickenpox, AES, Fever of Unknown Origin | **20** | **67** | **87** | 1,900+ | 46 | Seasonal temperature inversion, winter/spring humidity drops, and indoor crowding. |

---

## 4. Data Hygiene & Parsing Corrections Applied

During the ingestion of `maharashtra_weekly_outbreak_records_2023_2025.csv`, our pipeline identified and resolved minor CSV delimiter shifts:
1. **Unquoted Compound District Names:** In 5 rows, compound names like `Mumbai,Suburban` and `Chhatrapati,Sambhajinagar` contained raw unquoted commas, causing fields to shift into the disease column (e.g. `Suburban Zika Virus`, `Sambhajinagar Food Poisoning`). These were cleanly corrected and mapped to `Mumbai Suburban` and `Chhatrapati Sambhajinagar`.
2. **District Name Harmonization:**
   - `Aurangabad` -> `Chhatrapati Sambhajinagar` (Official updated name)
   - `Osmanabad` -> `Dharashiv` (Official updated name)
   - `Palaghar` -> `Palghar` (Typographical correction)
   - `Mumbai` -> `Mumbai City` vs `Mumbai Suburban` (Disambiguated via outbreak ID codes `MH/MMB` and `MH/MMS`)

---

## 5. Feasibility Verdict & ML Path Forward

| Requirement | Audit Status | Recommendation / Action for ML Phase |
| :--- | :--- | :--- |
| **Temporal Overlap** | **RESOLVED (100% Match)** | Use the 2023–2025 contemporary records aligned with daily gridded IMD rainfall and temperature series. |
| **Spatial Coverage** | **RESOLVED (All 36 Districts)** | IMD gridded NetCDF files cover the entire state bbox. Distribute climate metrics to districts via polygon centroid / spatial intersection. |
| **Disease Target Selection** | **RESOLVED** | Focus early warning ML classifiers and lag regressors on **Dengue**, **Chikungunya**, **Malaria**, and **Acute Diarrheal Disease (ADD)**. |
| **Handling Sparse / Zero Reports** | **STRICT PROTOCOL** | Adhere strictly to the negative: Do NOT fill unobserved weeks with 0 cases. Formulate early warning as a **Hazard Level / Outbreak Exceedance Probability** classification task or **Zero-Inflated / Hurdle Poisson model** rather than continuous linear regression. |
| **Humidity Feature Strategy** | **CRITICAL ARCHITECTURAL DECISION** | Because CWPRS humidity is strictly a single Pune station, do NOT apply it statewide. Instead: (a) Use CWPRS humidity for local Pune district validation, and (b) Derive state-wide vapor pressure deficit (VPD) or relative humidity proxy from gridded $T_{\min}$ and $T_{\max}$ using the standard Magnus-Tetens / FAO-56 Penman-Monteith dewpoint formulation. |

---

## Deliverables Generated in Project Workspace:
- [`dataset_audit_report.md`](file:///C:/Users/Janhvi%20Suryavanshi/.gemini/antigravity/scratch/JEEVAN-DRISHTI/dataset_audit_report.md)
- [`district_data_availability.csv`](file:///C:/Users/Janhvi%20Suryavanshi/.gemini/antigravity/scratch/JEEVAN-DRISHTI/district_data_availability.csv) (Updated with historical + contemporary breakdowns for all 36 districts)
- [`disease_data_availability.csv`](file:///C:/Users/Janhvi%20Suryavanshi/.gemini/antigravity/scratch/JEEVAN-DRISHTI/disease_data_availability.csv) (Updated with 40+ disease categories and climate driver links)
- [`maharashtra_outbreaks_2023_2025.csv`](file:///C:/Users/Janhvi%20Suryavanshi/.gemini/antigravity/scratch/JEEVAN-DRISHTI/maharashtra_outbreaks_2023_2025.csv) (Sanitized, quote-corrected contemporary dataset)
