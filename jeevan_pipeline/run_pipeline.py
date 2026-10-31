"""Run the full Phase 2 pipeline:  python -m jeevan_pipeline.run_pipeline

Reads only data/raw/*, writes data/interim, data/processed, data/reports and updates model_dataset_registry.json.
"""
import hashlib
import json
from datetime import datetime
import numpy as np
import pandas as pd
from . import config as C
from .tlog import TransformLog
from .disease import build_disease, rel
from .climate import build_climate
from .validate import frequency_validation, annotate_and_check_compatibility, missing_report


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_raw_unchanged(log):
    man = pd.read_csv(C.RAW_MANIFEST)
    bad = []
    for _, r in man.iterrows():
        from pathlib import Path
        if sha256(Path(r.raw_copy)) != r.sha256_copy:
            bad.append(r.raw_copy)
    log.add("raw", "integrity_check", "SHA-256 of data/raw files compared with Phase-1 manifest", len(man), len(man), len(bad),
            "all raw files unchanged" if not bad else f"MODIFIED: {bad}")
    if bad:
        raise RuntimeError(f"raw files changed: {bad}")


def main(verbose=False):
    for d in (C.INTERIM, C.PROCESSED, C.REPORTS):
        d.mkdir(parents=True, exist_ok=True)
    log = TransformLog()
    verify_raw_unchanged(log)

    disease, rejected, dmap, dstats = build_disease(log)
    climate, rh_rej, cell, rh_daily, cstats = build_climate(disease, log)
    cstats["anchors_text"] = dstats["anchors"]
    freq = frequency_validation(disease, cstats, dstats["anchors"])
    disease, compat = annotate_and_check_compatibility(disease, climate, cell, cstats)
    log.add("all", "compatibility_check", "district + date compatibility between disease and climate tables", len(disease), len(disease),
            int((~disease.climate_joinable).sum()), f"records joinable to climate: {int(disease.climate_joinable.sum())}/{len(disease)}")

    rejected_all = pd.concat([rejected, rh_rej], ignore_index=True)
    disease["pipeline_version"] = C.PIPELINE_VERSION
    climate["pipeline_version"] = C.PIPELINE_VERSION

    # ---- district mapping with coordinates / grid info
    cell_keep = cell.rename(columns={"district_std": "district_std"})
    dmap = dmap.merge(cell_keep, on="district_std", how="left")
    dmap["boundary_note"] = dmap.district_std.map({
        "Thane": "pre-2014-08-01 records include the area later split off as Palghar",
        "Palghar": "district created 2014-08-01; earlier events (if any) are not comparable",
        "Chhatrapati Sambhajinagar": "renamed from Aurangabad (name change only)",
        "Dharashiv": "renamed from Osmanabad (name change only)",
        "Ahmednagar": "renamed Ahilyanagar in 2025 by the state; neither source file uses the new name",
        "Mumbai City": "historical file labels it 'Mumbai (city)'; absent from 2023-25 file",
        "Mumbai Suburban": "no historical records; no coordinates in project data -> no climate extraction"}).fillna("")

    # ---- write processed / interim
    disease.to_csv(C.PROCESSED / "cleaned_disease_data.csv", index=False, date_format="%Y-%m-%d")
    climate.to_csv(C.PROCESSED / "cleaned_climate_data.csv", index=False, date_format="%Y-%m-%d")
    rh_daily.to_csv(C.INTERIM / "humidity_pune_daily.csv", index=False, date_format="%Y-%m-%d")
    cell.to_csv(C.INTERIM / "district_grid_cells.csv", index=False)
    freq.to_csv(C.INTERIM / "reporting_frequency_validation.csv", index=False)
    compat.to_csv(C.INTERIM / "district_date_compatibility.csv", index=False)
    disease[disease.duplicate_status != "unique"].sort_values(["dup_group_id", "record_id"]).to_csv(
        C.INTERIM / "disease_duplicate_groups_for_review.csv", index=False, date_format="%Y-%m-%d")
    dmap.to_csv(C.REPORTS / "district_mapping.csv", index=False)
    rejected_all.to_csv(C.REPORTS / "rejected_records.csv", index=False)

    raw_h = pd.read_csv(C.HIST_RAW)
    raw_i = pd.read_csv(C.IDSP_RAW, encoding="utf-8-sig", dtype=str)
    raw_r = pd.read_csv(C.RH_RAW)
    mr = missing_report({"cleaned_disease_data": disease, "cleaned_climate_data": climate, "raw:maharashtra_epiclim_filtered": raw_h,
                         "raw:idsp_weekly_original_upload": raw_i, "raw:humid_tel_hr_cwprs": raw_r})
    mr.to_csv(C.REPORTS / "missing_data_report.csv", index=False)

    log.add("all", "write_outputs", "wrote cleaned tables and reports", None, None, None,
            f"cleaned_disease={len(disease)} rows; cleaned_climate={len(climate)} rows; rejected={len(rejected_all)} rows")
    summary = build_summary(disease, climate, rejected_all, dstats, cstats, freq, compat, cell, mr)
    (C.REPORTS / "preprocessing_summary.md").write_text(summary, encoding="utf-8")
    log.add("all", "write_summary", "preprocessing_summary.md written", None, None, None, "")
    log.frame().to_csv(C.REPORTS / "preprocessing_log.csv", index=False)
    update_registry(disease, climate, dstats, cstats)
    if verbose:
        print(summary[:3000])


def update_registry(disease, climate, dstats, cstats):
    reg = json.loads(C.REGISTRY.read_text(encoding="utf-8"))
    outs = [
        ("cleaned_disease_data", C.PROCESSED / "cleaned_disease_data.csv", disease, "disease",
         dict(date_start=str(disease.join_date_start.min().date()), date_end=str(disease.join_date_end.max().date()),
              derived_from=[rel(C.HIST_RAW), rel(C.IDSP_RAW)])),
        ("cleaned_climate_data", C.PROCESSED / "cleaned_climate_data.csv", climate, "climate",
         dict(date_start=cstats["date_min"], date_end=cstats["date_max"],
              derived_from=[rel(C.RAIN_NC), rel(C.TMIN_NC), rel(C.TMAX_NC), rel(C.RH_RAW)])),
    ]
    reg["processed_datasets"] = [dict(name=n, path=str(p), relative_path=rel(p), version=C.DATASET_VERSION, pipeline_version=C.PIPELINE_VERSION,
                                      rows=len(df), columns=df.shape[1], sha256=sha256(p), kind=k, **extra) for n, p, df, k, extra in outs]
    reg["processed_datasets"] += [dict(name=n, path=str(C.REPORTS / n), relative_path=rel(C.REPORTS / n), version=C.DATASET_VERSION, kind="report")
                                  for n in ["district_mapping.csv", "rejected_records.csv", "preprocessing_log.csv", "missing_data_report.csv", "preprocessing_summary.md"]]
    reg["pipeline"] = dict(package="jeevan_pipeline", entrypoint="python -m jeevan_pipeline.run_pipeline", tests="python -m unittest discover -s tests -v",
                           version=C.PIPELINE_VERSION, dataset_version=C.DATASET_VERSION, last_run=datetime.now().isoformat(timespec="seconds"))
    reg["climate_file_metadata"] = cstats["nc_attrs"]
    reg.setdefault("decisions", {})["phase2"] = dict(
        district_climate_points="district coordinates from the historical file (one fixed point per district); nearest IMD cell; no gap filling",
        humidity="valid RH in (0,100] only; exclusive to Pune; never applied to other districts",
        average_temperature="provided as tavg_derived_c = (tmin+tmax)/2, explicitly derived",
        missing_deaths="kept as NaN, never converted to zero")
    C.REGISTRY.write_text(json.dumps(reg, indent=2, default=str), encoding="utf-8")


def build_summary(disease, climate, rej, ds, cs, freq, compat, cell, mr):
    L = []
    w = L.append
    h = disease[disease.source_dataset == "historical_epiclim"]
    i = disease[disease.source_dataset == "idsp_weekly"]
    w("# JEEVAN-DRISHTI Phase 2: Preprocessing Summary\n")
    w(f"Pipeline `jeevan_pipeline` v{C.PIPELINE_VERSION}, dataset version `{C.DATASET_VERSION}`. Run: `python -m jeevan_pipeline.run_pipeline`. "
      "All inputs are the SHA-256-verified copies in `data/raw/`; nothing in `data/raw/` was changed. No models were trained.\n")
    w("## 1. Outputs\n")
    w("| File | Content |\n|---|---|")
    w(f"| `data/processed/cleaned_disease_data.csv` | {len(disease):,} disease records ({len(h):,} historical + {len(i):,} IDSP), original source columns preserved, flags added |")
    w(f"| `data/processed/cleaned_climate_data.csv` | {len(climate):,} district-day rows (36 districts × {cs['n_dates']} days) |")
    w("| `data/reports/district_mapping.csv` | raw to canonical district names, coordinates and grid cells used |")
    w(f"| `data/reports/rejected_records.csv` | {len(rej):,} rows removed from the cleaned tables, each with a reason and the original row as JSON |")
    w("| `data/reports/preprocessing_log.csv` | one row per transformation step |")
    w("| `data/reports/missing_data_report.csv` | missing values per column with meaning and treatment |")
    w("| `data/interim/*` | humidity daily series, district grid cells, frequency validation, district/date compatibility, duplicate groups for review |\n")
    w("## 2. Record reconciliation\n")
    w("| Source | Raw rows | Cleaned | Rejected |\n|---|---|---|---|")
    for k, n in ds["raw_rows"].items():
        w(f"| {k} | {n:,} | {int((disease.source_dataset == k).sum()):,} | {int((rej.source_dataset == k).sum()):,} |")
    r = rej.groupby(["source_dataset", "reject_reason"]).size()
    w("\nRejection reasons: " + ("; ".join(f"{a} / {b}: {n:,}" for (a, b), n in r.items()) or "none") + ".\n")
    w("Rejected rows are exact duplicates, and humidity readings above 100 % or equal to 0 %. Disease rows rejected for invalid values: "
      f"{int(rej[rej.source_dataset != 'humidity_cwprs'].reject_reason.isin(['invalid_cases', 'invalid_deaths', 'unmapped_district', 'no_usable_date']).sum())}.\n")
    w("## 3. Standardization\n")
    w("- **Dates:** ISO `yyyy-mm-dd`. IDSP ordinal dates (`2nd January 2023`) were parsed; nothing was guessed. "
      f"Report periods missing in the source for {int((i.report_period_source == 'derived_week_convention').sum())} rows were derived from the report-week "
      "convention, which was verified against every full 7-day week in the source (week-1 start per year: "
      f"{ds['anchors']}). Those rows carry `report_period_source=derived_week_convention`.")
    mt = disease.groupby("district_mapping_type").size().to_dict()
    w(f"- **Districts:** all names map to the 36 official districts ({mt}). The 5 split names in the raw upload ('Mumbai'+'Suburban …', 'Chhatrapati'+'Sambhajinagar …') were repaired, "
      "with the source values kept in `district_source`/`disease_source`.")
    w(f"- **Diseases:** {disease.disease_raw.nunique()} raw labels became {disease.disease_std.nunique()} standard labels. Combined labels stay separate categories "
      "(`Dengue + Chikungunya`, `Dengue + Malaria`). Single-disease counts below are never merged with them.")
    tg = disease[disease.is_target_disease].groupby(["disease_std", "source_dataset"]).size().unstack(fill_value=0)
    w("\n| Target disease | Historical | IDSP 2023–25 |\n|---|---|---|")
    for d in C.TARGET_DISEASES:
        w(f"| {d} | {int(tg.loc[d].get('historical_epiclim', 0)) if d in tg.index else 0} | {int(tg.loc[d].get('idsp_weekly', 0)) if d in tg.index else 0} |")
    w(f"\nCombined-label records: {int((disease.disease_group == 'combined').sum())}. Unmapped labels: {int((~disease.disease_mapped).sum())} rows.\n")
    w("## 4. Duplicates and conflicts\n")
    w(f"Exact duplicates were removed to `rejected_records.csv`, with a pointer to the kept record. Group statuses among the kept records: {ds['dup_status']}. "
      "`conflict` means the group has differing cases or deaths. `possible_duplicate*` means the same key with identical values or a different outbreak ID. "
      "All of these rows are **kept and flagged**. The groups are listed in `data/interim/disease_duplicate_groups_for_review.csv`. "
      "A human decision is needed before they are collapsed.\n")
    w("## 5. Missing values (never zero-filled)\n")
    w(f"- Deaths missing: historical {int(h.deaths.isna().sum())}/{len(h)}, IDSP {int(i.deaths.isna().sum())}/{len(i)}. They stay NaN (flag `deaths_missing_not_zero`).")
    w(f"- `outbreak_start_date` missing in {int(i.event_date.isna().sum())} IDSP rows. The report week is used for joins.")
    w("- No record was inserted for weeks, districts or diseases without reports, because absence of an outbreak report is not zero cases.\n")
    w("## 6. Numerical validation\n")
    w(f"- Cases ≤ 0 or non-integer: 0 kept. Deaths > cases or negative: 0 kept (see rejected table for any removed).")
    w(f"- Historical embedded `Temp_C` > 50 °C: {int(h['flags'].str.contains('embedded_temp_c_gt_50').sum())} rows flagged (not removed). "
      "These are April–May values that look like land-surface rather than air temperature.")
    w(f"- Very high case counts above the 99.5th percentile are flagged for review only: {int(disease['flags'].str.contains('cases_above_p99_5').sum())} rows.\n")
    w("## 7. Climate validation\n")
    rh = cs["rh"]
    w(f"- IMD rainfall, Tmin and Tmax: {cs['date_min']} to {cs['date_max']}, {cs['n_dates']} days, no non-daily steps. "
      f"Out-of-range values set to NaN: rain {cs['rain']['n_bad']}, tmin {cs['tmin']['n_bad']}, tmax {cs['tmax']['n_bad']}. "
      f"Tmin > Tmax days at district cells: {int(climate['flags'].str.contains('tmin_gt_tmax').sum())}.")
    w(f"- Grid values that are NaN in the source (ocean/outside-India mask): rain {cs['rain']['n_nan']:,}, tmin {cs['tmin']['n_nan']:,}, tmax {cs['tmax']['n_nan']:,}.")
    w(f"- Districts with coordinates: {cs['pts']}/36. **Mumbai Suburban has no coordinates in any project dataset, so its climate is NaN**. Coordinates were not invented.")
    w(f"- Districts whose nearest IMD cell is all-NaN for at least one variable: {cs['nan_districts'] or 'none'}. "
      "Their values stay NaN. Using a neighbouring cell would be a methodological decision that I have not taken.")
    w(f"- Humidity: {rh['n_raw']:,} raw readings, {rh['n_invalid']:,} invalid (>100 or 0) excluded from aggregates; "
      f"{rh['days_with_data']}/{rh['span_days']} calendar days have data; {rh['days_low_coverage']} days below 50 % coverage; Pune station only.\n")
    w("## 8. Reporting frequency\n")
    w("| Dataset | Check | Observed | Status |\n|---|---|---|---|")
    for _, x in freq.iterrows():
        w(f"| {x.dataset} | {x.check} | {str(x.observed)[:90]} | {x.status} |")
    w("\nDetails are in `data/interim/reporting_frequency_validation.csv`.\n")
    from .validate import _pdf_weeks
    pw = _pdf_weeks()
    if pw:
        parts = []
        for y in sorted(i.report_year.dropna().unique()):
            y = int(y)
            miss = set(range(1, 53)) - set(i[i.report_year == y].report_week.dropna().astype(int))
            parts.append(f"{y}: {len(miss & pw.get(y, set()))} weeks have a PDF in `sevadataset.zip` but no rows, {len(miss - pw.get(y, set()))} weeks have no PDF")
        w("**Attention: weeks without rows.** " + "; ".join(parts) + ". It is not known whether those PDFs reported no outbreaks or the PDF-to-CSV step missed content "
          "(PDFs were not parsed in Phase 1 or 2). These weeks are left empty, never zero-filled, and the 2025 series should not be treated as complete until this is checked.\n")
    w("## 9. District and date compatibility\n")
    jo = i.climate_joinable.sum()
    w(f"- IDSP records joinable to IMD climate (district has a valid cell and the report week lies in 2023–25): **{int(jo)}/{len(i)}**.")
    w(f"- Historical records joinable to climate: **{int(h.climate_joinable.sum())}/{len(h)}**. There is no date overlap, so they only carry their embedded covariates.")
    w(f"- IDSP records joinable to humidity (Pune and inside the station window): {int(i.climate_rh_available.sum())}.")
    prob = compat[compat.issues != ""][["district", "issues"]]
    w("\n| District | Issue |\n|---|---|")
    for _, x in prob.iterrows():
        w(f"| {x.district} | {x.issues} |")
    w("\n## 10. Readiness for feature engineering\n")
    w("| Dataset | Status | Notes |\n|---|---|---|")
    conf = int((i.duplicate_status == 'conflict').sum())
    w(f"| `cleaned_disease_data` IDSP 2023–25 | **READY (with caveats)** | district-week outbreak counts; flagged duplicate groups ({int((i.duplicate_status != 'unique').sum())} rows, {conf} conflicts) need a rule; "
      "missing weeks in 2023 and 2025 are not zero; outbreak alerts, not incidence |")
    w("| `cleaned_disease_data` historical 2009–22 | **READY for disease-only features** | seasonality and baselines; **cannot** be linked to the climate files (no overlap); no Leptospirosis or heat illness |")
    nanr = [d for d in cell.district_std if cell.set_index('district_std').loc[d, 'rain_valid_days'] == 0]
    nant = [d for d in cell.district_std if cell.set_index('district_std').loc[d, 'tmin_valid_days'] == 0 or cell.set_index('district_std').loc[d, 'tmax_valid_days'] == 0]
    w(f"| `cleaned_climate_data` rainfall | **READY** except {nanr} | daily mm at nearest 0.25° cell |")
    w(f"| `cleaned_climate_data` Tmin/Tmax/derived Tavg | **NEEDS ATTENTION** for {nant} | 1° grid is coarse (~110 km); a cell-selection decision is needed for districts with NaN cells |")
    w("| `cleaned_climate_data` relative humidity | **ATTENTION: Pune only** | cannot be a statewide feature; needs a proxy or exclusion |")
    w("| Heat-related illness | **NO DATA** | zero records in any disease file; cannot be modelled |")
    w("\n## 11. Decisions needed before Phase 3\n")
    w("1. Rule for the flagged duplicate/conflict groups (keep all, keep max, or manual review).")
    w("2. Handling of districts whose nearest temperature cell is NaN (nearest valid cell, or exclude).")
    w("3. Whether combined labels (Dengue + Chikungunya) stay separate or are counted under both diseases.")
    w("4. Mumbai Suburban coordinates: not available in project data. Supply a source, or treat it with Mumbai City.")
    w("5. Humidity strategy outside Pune.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()


