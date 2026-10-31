# JEEVAN-DRISHTI Phase 1: Dataset Inspection Summary

Every number below was measured from the files by `phase1_inspect.py`. No source file was modified, cleaned, imputed or used to train a model. The detail is in the sibling CSVs and in `../../model_dataset_registry.json`.

## 1. Datasets found
No Parquet, JSON or Excel dataset belongs to the project. The source files sit **outside the project folder**, mostly in `C:\Users\Janhvi Suryavanshi\Downloads`.

| File | Role | Rows | Date range | Frequency |
|---|---|---|---|---|
| `Downloads\maharashtra_epiclim_filtered.csv` | Disease, historical (EpiClim-derived IDSP outbreaks) | 1,195 × 14 | 2009-07-23 → 2022-06-15 | Event-based |
| `…\.user_uploaded\media_1793432423584_ffb33b26.csv` | Disease, 2023–25, raw upload | 575 × 15 | 2023 W01 → 2025 W52 | Weekly report |
| `JEEVAN-DRISHTI\maharashtra_outbreaks_2023_2025.csv` | Same 575 rows, derived copy (quoting fix only) | 575 × 15 | same | Weekly report |
| `Downloads\imdweb_5dgbAskl.nc` | IMD rainfall (mm/day), 0.25°, 30×31 grid | 1,096 days | 2023-01-01 → 2025-12-31 | Daily |
| `Downloads\imdweb_Oe9H04Bg.nc` | IMD Tmin (°C), 1.0°, 9×9 grid | 1,096 days | same | Daily |
| `Downloads\imdweb_tXexQbB6.nc` | IMD Tmax (°C), 1.0°, 9×9 grid | 1,096 days | same | Daily |
| `Downloads\humid_tel_hr_cwprs_mh_2021_2025.csv` | Relative humidity, one Pune station | 87,990 × 21 | 2024-05-11 → 2025-06-09 | 5-minute, with gaps |
| `Downloads\sevadataset.zip` | 155 IDSP weekly PDFs, the raw source of the 2023–25 CSV | 158 entries | n/a | Not parsed |
| `JEEVAN-DRISHTI\sample_reports\week1_2023.pdf` | Sample PDF | n/a | n/a | Not parsed |

The two `*_availability.csv` files and `dataset_audit_report.md` were generated earlier by this project. They are not source data.

**Provenance.** The IMD, CWPRS and EpiClim origins are inferred from filenames and structure. The files carry no formal metadata. The 2023–25 rows cite `source_pdf` and `page` columns, which is the real record-level provenance. I could not verify the origin of the historical file's `preci`, `LAI` and `Temp` columns.

## 2. Quality findings
- **Historical CSV:**
  - 1 exact duplicate row.
  - 27 repeats of (date, district, disease).
  - 16 `Temp_C` values outside 0–50 °C.
  - 6 district spelling variants.
  - No cases or deaths anomalies.
- **2023–25 CSV:**
  - 7 duplicate `outbreak_id` values.
  - 13 repeated (year, week, district, disease, cases) keys.
  - 26 rows are missing the report period.
  - 192 rows are missing `outbreak_start_date`.
  - 44 raw disease labels, some with broken tokens such as "Acute Gastroenterit is".
  - 5 rows in the raw upload have unquoted multi-word district names that shift columns.
  - Only 39 of 52 weeks are present for 2023 and 30 of 52 for 2025. 2024 is complete.
- **Humidity:**
  - The filename says 2021–2025, but the data starts 2024-05-11.
  - 1,875 values fall outside 0–100.
  - 6,017 values equal exactly 0 or 100.
  - Data exists on 371 of 395 days.
  - There is one station only.
- **IMD grids:**
  - There are no negative rainfall values and no cells where Tmin exceeds Tmax.
  - The time axis is continuous.
  - 38% of rainfall cells and about 37% of temperature cells are NaN (ocean or outside India).

## 3. Disease availability (outbreak and cluster reports, **not** incidence)
| Disease | 2009–22 records | 2023–25 records |
|---|---|---|
| Dengue | 392 (33 districts) | 187 (30) |
| Malaria | 114 (19) | 51 (12) |
| Chikungunya | 161 (25) | 41 (14) |
| Acute diarrhoeal disease | 471 (35) | 114 (32) |
| Cholera | 58 (22) | 20 (11) |
| Leptospirosis | **0** | 9 (7) |
| Heat-related illness | **0** | **0** |

Heat-related illness is absent from both disease files. Leptospirosis exists only for 2023–25, and only 9 records. Dengue/chikungunya combined labels and the "Acute Gastroenteritis" label need a mapping decision. The disease counts above use regex matching, so mixed labels are counted under each matching disease.

An absent week or district is **not** zero cases. These are outbreak alerts, so zero-filling would be invention.

## 4. Climate variables
| Variable | Status |
|---|---|
| Rainfall | Available. Daily, 0.25° grid, 2023–25. |
| Tmin and Tmax | Available. Daily, coarse 1° grid, 2023–25. |
| Average temperature | **No dataset.** It could be derived later as (Tmin+Tmax)/2. |
| Relative humidity | **One Pune station only**, 2024-05 → 2025-06. Not statewide. |
| Embedded `preci`, `LAI`, `Temp`, `Temp_C` | In the historical CSV only, at event rows. Source and units are undocumented. |

## 5. Districts (36 official)
- The historical file covers 35 districts and lacks **Mumbai Suburban**.
- The 2023–25 file covers 34 districts and lacks **Mumbai City** and **Ratnagiri**.
- Across the two files all 36 districts appear, but only 33 appear in both. No single usable time series covers all 36.
- Name variants found:

  | Raw name | Canonical name |
  |---|---|
  | Aurangabad | Chhatrapati Sambhajinagar |
  | Osmanabad / Osmanabad (Maharashtra) | Dharashiv |
  | Nasik | Nashik |
  | Sholapur | Solapur |
  | Palaghar | Palghar |
  | Mumbai (city) | Mumbai City |

- Historical boundary differences:
  - Palghar was carved from Thane in 2014, so earlier Palghar events sit under Thane.
  - Ahmednagar was renamed Ahilyanagar in 2025, but neither file uses the new name.
  - Mumbai handling differs between the two files.
- No names were left unmapped. The aliases come from my own mapping table, so please confirm them.

## 6. Integration feasibility
| Pair | Overlap | Verdict |
|---|---|---|
| Historical disease × IMD or CWPRS | **0 days** | Impossible. |
| 2023–25 disease × IMD rain, Tmin, Tmax | 1,092 days (report periods 2023-01-02 → 2025-12-28) | Feasible after aggregating the daily grid to district-week. A district-to-grid mapping is still needed and does not exist yet. |
| 2023–25 disease × CWPRS humidity | 395 days | **Pune only.** |
| Historical × 2023–25 disease | none; gap from 2022-06 to 2023 | Only with caution, because the schemas and reporting practice differ. |
| IMD rain × IMD Tmin/Tmax | full | Needs regridding (0.25° vs 1°). |

Integration is on district + report week. Daily and sub-hourly climate data must be aggregated, and weekly outbreak reports cannot be disaggregated.

## 7. Corrections to the earlier `dataset_audit_report.md`
That document should not be relied on. It disagrees with the measured values in these places:
- It says the IMD grids are 35×28 and 9×7. Measured: 30×31 for rainfall and 9×9 for Tmin and Tmax.
- It says "all 36 districts" have usable coverage and "100% overlap". Measured: 34 districts in 2023–25, and 2023 and 2025 have missing weeks.
- It says "zero missing days" for the IMD files. Measured: the time axis is complete, but 38% of rainfall cells and about 37% of temperature cells are NaN. The NaN cells are spatial, not temporal.
- It reports other counts for the historical file's disease groups (for example 543 vector-borne records). I did not reproduce those figures.
- It issues a "YES, THIS WORKS!" feasibility verdict and recommends models. That was premature and outside Phase 1.

## 8. Open items for approval
1. The historical file lives only in Downloads. Should Phase 2 copy it into the project, or keep the current paths?
2. Is the district alias table above correct, especially the two Mumbai districts and the pre-2014 Palghar/Thane handling?
3. How should combined labels be treated: dengue/chikungunya, "Suspected Dengue", "Acute Gastroenteritis" and "Malaria (PV)"?
4. Heat-related illness has no data. Should it be dropped from scope, or do you have another file for it?
5. How should the 192 rows without an outbreak start date, and the report weeks missing in 2023 and 2025, be handled?

**Awaiting approval before Phase 2.**
