# JEEVAN-DRISHTI Phase 2: Preprocessing Summary

Pipeline `jeevan_pipeline` v2.0.0, dataset version `phase2-v1`. Run: `python -m jeevan_pipeline.run_pipeline`. All inputs are the SHA-256-verified copies in `data/raw/`; nothing in `data/raw/` was changed. No models were trained.

## 1. Outputs

| File | Content |
|---|---|
| `data/processed/cleaned_disease_data.csv` | 1,764 disease records (1,194 historical + 570 IDSP), original source columns preserved, flags added |
| `data/processed/cleaned_climate_data.csv` | 39,456 district-day rows (36 districts × 1096 days) |
| `data/reports/district_mapping.csv` | raw to canonical district names, coordinates and grid cells used |
| `data/reports/rejected_records.csv` | 1,886 rows removed from the cleaned tables, each with a reason and the original row as JSON |
| `data/reports/preprocessing_log.csv` | one row per transformation step |
| `data/reports/missing_data_report.csv` | missing values per column with meaning and treatment |
| `data/interim/*` | humidity daily series, district grid cells, frequency validation, district/date compatibility, duplicate groups for review |

## 2. Record reconciliation

| Source | Raw rows | Cleaned | Rejected |
|---|---|---|---|
| historical_epiclim | 1,195 | 1,194 | 1 |
| idsp_weekly | 575 | 570 | 5 |

Rejection reasons: historical_epiclim / exact_duplicate: 1; humidity_cwprs / rh_above_100: 1,875; humidity_cwprs / rh_zero_sensor_failure: 5; idsp_weekly / exact_duplicate: 5.

Rejected rows are exact duplicates, and humidity readings above 100 % or equal to 0 %. Disease rows rejected for invalid values: 0.

## 3. Standardization

- **Dates:** ISO `yyyy-mm-dd`. IDSP ordinal dates (`2nd January 2023`) were parsed; nothing was guessed. Report periods missing in the source for 26 rows were derived from the report-week convention, which was verified against every full 7-day week in the source (week-1 start per year: {2023: '2023-01-02', 2024: '2024-01-01', 2025: '2024-12-30'}). Those rows carry `report_period_source=derived_week_convention`.
- **Districts:** all names map to the 36 official districts ({'alias': 98, 'exact': 1661, 'exact+structural_repair': 5}). The 5 split names in the raw upload ('Mumbai'+'Suburban …', 'Chhatrapati'+'Sambhajinagar …') were repaired, with the source values kept in `district_source`/`disease_source`.
- **Diseases:** 52 raw labels became 28 standard labels. Combined labels stay separate categories (`Dengue + Chikungunya`, `Dengue + Malaria`). Single-disease counts below are never merged with them.

| Target disease | Historical | IDSP 2023–25 |
|---|---|---|
| Dengue | 378 | 176 |
| Malaria | 114 | 49 |
| Chikungunya | 147 | 34 |
| Acute Diarrhoeal Disease | 470 | 114 |
| Cholera | 58 | 20 |
| Leptospirosis | 0 | 9 |
| Heat-related illness | 0 | 0 |

Combined-label records: 24. Unmapped labels: 0 rows.

## 4. Duplicates and conflicts

Exact duplicates were removed to `rejected_records.csv`, with a pointer to the kept record. Group statuses among the kept records: {'unique': 1695, 'conflict': 34, 'possible_duplicate': 22, 'possible_duplicate_distinct_id': 13}. `conflict` means the group has differing cases or deaths. `possible_duplicate*` means the same key with identical values or a different outbreak ID. All of these rows are **kept and flagged**. The groups are listed in `data/interim/disease_duplicate_groups_for_review.csv`. A human decision is needed before they are collapsed.

## 5. Missing values (never zero-filled)

- Deaths missing: historical 850/1194, IDSP 0/570. They stay NaN (flag `deaths_missing_not_zero`).
- `outbreak_start_date` missing in 187 IDSP rows. The report week is used for joins.
- No record was inserted for weeks, districts or diseases without reports, because absence of an outbreak report is not zero cases.

## 6. Numerical validation

- Cases ≤ 0 or non-integer: 0 kept. Deaths > cases or negative: 0 kept (see rejected table for any removed).
- Historical embedded `Temp_C` > 50 °C: 16 rows flagged (not removed). These are April–May values that look like land-surface rather than air temperature.
- Very high case counts above the 99.5th percentile are flagged for review only: 9 rows.

## 7. Climate validation

- IMD rainfall, Tmin and Tmax: 2023-01-01 to 2025-12-31, 1096 days, no non-daily steps. Out-of-range values set to NaN: rain 0, tmin 0, tmax 0. Tmin > Tmax days at district cells: 0.
- Grid values that are NaN in the source (ocean/outside-India mask): rain 386,888, tmin 32,880, tmax 32,880.
- Districts with coordinates: 35/36. **Mumbai Suburban has no coordinates in any project dataset, so its climate is NaN**. Coordinates were not invented.
- Districts whose nearest IMD cell is all-NaN for at least one variable: ['Gadchiroli', 'Mumbai Suburban']. Their values stay NaN. Using a neighbouring cell would be a methodological decision that I have not taken.
- Humidity: 87,990 raw readings, 1,880 invalid (>100 or 0) excluded from aggregates; 371/395 calendar days have data; 45 days below 50 % coverage; Pune station only.

## 8. Reporting frequency

| Dataset | Check | Observed | Status |
|---|---|---|---|
| idsp_weekly | weeks_present_2023 | 39 | ATTENTION |
| idsp_weekly | weeks_present_2024 | 52 | OK |
| idsp_weekly | weeks_present_2025 | 30 | ATTENTION |
| idsp_weekly | one_period_per_year_week | 0 (year,week) with >1 period | OK |
| idsp_weekly | period_length_days | min=4 max=6 | ATTENTION |
| idsp_weekly | report_period_source | {'source': 544, 'derived_week_convention': 26} | ATTENTION |
| historical_epiclim | reporting_frequency | event-based; median gap 2 d, max 57 d | INFO |
| imd_rain | daily_continuity | 1096 days, shape=(1096, 30, 31) | OK |
| imd_tmin | daily_continuity | 1096 days, shape=(1096, 9, 9) | OK |
| imd_tmax | daily_continuity | 1096 days, shape=(1096, 9, 9) | OK |
| humidity_cwprs | native_frequency | {5.0: 87321, 65.0: 536, 125.0: 50, 185.0: 11, 245.0: 7, 365.0: 5} | ATTENTION |
| humidity_cwprs | spatial_coverage | 1 station (Pune) | ATTENTION |
| cross | disease(weekly)_vs_climate(daily) | weekly vs daily | AGGREGATE_CLIMATE |
| cross | historical_vs_climate | 0 days | INCOMPATIBLE |

Details are in `data/interim/reporting_frequency_validation.csv`.

**Attention: weeks without rows.** 2023: 11 weeks have a PDF in `sevadataset.zip` but no rows, 2 weeks have no PDF; 2024: 0 weeks have a PDF in `sevadataset.zip` but no rows, 0 weeks have no PDF; 2025: 22 weeks have a PDF in `sevadataset.zip` but no rows, 0 weeks have no PDF. It is not known whether those PDFs reported no outbreaks or the PDF-to-CSV step missed content (PDFs were not parsed in Phase 1 or 2). These weeks are left empty, never zero-filled, and the 2025 series should not be treated as complete until this is checked.

## 9. District and date compatibility

- IDSP records joinable to IMD climate (district has a valid cell and the report week lies in 2023–25): **567/570**.
- Historical records joinable to climate: **0/1194**. There is no date overlap, so they only carry their embedded covariates.
- IDSP records joinable to humidity (Pune and inside the station window): 30.

| District | Issue |
|---|---|
| Gadchiroli | nearest rain grid cell is all-NaN |
| Mumbai City | no 2023-25 records |
| Mumbai Suburban | no coordinates -> no climate extraction; no historical records |
| Ratnagiri | no 2023-25 records |

## 10. Readiness for feature engineering

| Dataset | Status | Notes |
|---|---|---|
| `cleaned_disease_data` IDSP 2023–25 | **READY (with caveats)** | district-week outbreak counts; flagged duplicate groups (18 rows, 5 conflicts) need a rule; missing weeks in 2023 and 2025 are not zero; outbreak alerts, not incidence |
| `cleaned_disease_data` historical 2009–22 | **READY for disease-only features** | seasonality and baselines; **cannot** be linked to the climate files (no overlap); no Leptospirosis or heat illness |
| `cleaned_climate_data` rainfall | **READY** except ['Gadchiroli', 'Mumbai Suburban'] | daily mm at nearest 0.25° cell |
| `cleaned_climate_data` Tmin/Tmax/derived Tavg | **NEEDS ATTENTION** for ['Mumbai Suburban'] | 1° grid is coarse (~110 km); a cell-selection decision is needed for districts with NaN cells |
| `cleaned_climate_data` relative humidity | **ATTENTION: Pune only** | cannot be a statewide feature; needs a proxy or exclusion |
| Heat-related illness | **NO DATA** | zero records in any disease file; cannot be modelled |

## 11. Decisions needed before Phase 3

1. Rule for the flagged duplicate/conflict groups (keep all, keep max, or manual review).
2. Handling of districts whose nearest temperature cell is NaN (nearest valid cell, or exclude).
3. Whether combined labels (Dengue + Chikungunya) stay separate or are counted under both diseases.
4. Mumbai Suburban coordinates: not available in project data. Supply a source, or treat it with Mumbai City.
5. Humidity strategy outside Pune.
