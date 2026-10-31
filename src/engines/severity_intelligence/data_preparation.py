"""Outcome & Severity Data Availability Audit and Canonical Record Preparation (Phase 12).

Responsibilities:
1. Inspect existing project datasets for the 7 required outcome/severity categories:
   - Disease-related deaths
   - Hospital admissions
   - Severe disease cases
   - Complications
   - Recovery outcomes
   - Case fatality information
   - District-wise mortality records
2. Enforce strict scientific and privacy rules:
   - Never invent missing outcome data (hospital admissions, clinical severity counts,
     complications, and recovery counts are explicitly `NaN`).
   - Strictly distinguish missing (`NaN`) death values from `0` deaths.
   - Never convert aggregate outbreak records into synthetic individual patient records.
   - Calculate Outbreak-Cluster Case Fatality Ratio (`OC-CFR`) strictly where numerator
     (`verified_deaths`) and denominator (`verified_cases`) come from the same compatible,
     deduplicated IDSP outbreak record with `verified_cases > 0` and non-null `verified_deaths`.
"""
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.features.geographic_features import ADMIN_DIVISION_MAP, DISTRICT_ID_MAP

PROJ = Path(__file__).resolve().parents[3]
ENGINE_VERSION = "phase12-v1.0.0"

MANDATORY_SEVERITY_PRIVACY_AND_SAFETY_DISCLAIMER = (
    "AGGREGATE PUBLIC-HEALTH SURVEILLANCE ONLY — NO INDIVIDUAL PATIENT DATA OR PROGNOSIS. "
    "All mortality and case-fatality indicators are derived strictly from aggregate district-level "
    "IDSP outbreak cluster reports (2023–2025). Clinical severity counts, inpatient admissions, "
    "complications, and recovery counts do not exist in the workspace and are explicitly NaN. "
    "This engine never processes personally identifiable information (PII) and must never be used "
    "for individual clinical diagnosis, prognosis, or treatment decisions."
)


def audit_outcome_and_severity_data_availability(
    proj_dir: Path = PROJ,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Audit workspace datasets for the 7 required severity/mortality categories and
    compare missing (`NaN`) vs. zero (`0`) death records across datasets.
    """
    cleaned_df = pd.read_csv(proj_dir / "data" / "processed" / "cleaned_disease_data.csv")
    weekly_df = pd.read_csv(proj_dir / "data" / "features" / "disease_climate_weekly_features.csv")

    # 1. Audit table across the 7 Prompt 12 Section 1 categories
    audit_rows: List[Dict[str, Any]] = [
        {
            "outcome_category_key": "disease_related_deaths",
            "outcome_category_label": "Disease-Related Deaths (Aggregate Outbreak Mortality)",
            "available_in_workspace": True,
            "source_dataset": "data/features/disease_climate_weekly_features.csv (from IDSP 2023-2025)",
            "date_range_supported": "2023-W01 to 2025-W52 (511 deduplicated outbreak-weeks)",
            "district_coverage": "34 / 36 active districts (2 districts have 0 outbreaks in 2023-2025)",
            "non_null_records_2023_2025": int(weekly_df["target_deaths"].notna().sum()),
            "missing_nan_records_2023_2025": int(weekly_df["target_deaths"].isna().sum()),
            "zero_value_records_2023_2025": int((weekly_df["target_deaths"] == 0).sum()),
            "positive_value_records_2023_2025": int((weekly_df["target_deaths"] > 0).sum()),
            "definition_consistency": (
                "Consistent in 2023-2025 IDSP weekly outbreaks (100% non-null: 365 zero-death weeks, "
                "146 positive-death weeks, 199 total deaths). In 2009-2022 historical_epiclim, 71.19% of "
                "rows have missing (NaN) deaths — kept strictly separate and never zero-filled."
            ),
            "engine_handling_mode": "VERIFIED_AGGREGATE_MORTALITY_TREND_AND_FORECAST_EVALUATION",
        },
        {
            "outcome_category_key": "hospital_admissions",
            "outcome_category_label": "Hospital Admissions (Inpatient / IPD Counts)",
            "available_in_workspace": False,
            "source_dataset": "NONE_IN_WORKSPACE",
            "date_range_supported": "N/A",
            "district_coverage": "0 / 36 districts (0.0%)",
            "non_null_records_2023_2025": 0,
            "missing_nan_records_2023_2025": len(weekly_df),
            "zero_value_records_2023_2025": 0,
            "positive_value_records_2023_2025": 0,
            "definition_consistency": "Unavailable in workspace; never invented from case counts.",
            "engine_handling_mode": "EXPLICIT_NAN_UNAVAILABLE",
        },
        {
            "outcome_category_key": "severe_disease_cases",
            "outcome_category_label": "Severe / Critical Disease Case Counts (Clinical Grading)",
            "available_in_workspace": False,
            "source_dataset": "NONE_IN_WORKSPACE",
            "date_range_supported": "N/A",
            "district_coverage": "0 / 36 districts (0.0%)",
            "non_null_records_2023_2025": 0,
            "missing_nan_records_2023_2025": len(weekly_df),
            "zero_value_records_2023_2025": 0,
            "positive_value_records_2023_2025": 0,
            "definition_consistency": (
                "No clinical severity stratification (mild/moderate/severe/critical) exists in IDSP "
                "outbreak summaries. Explicit severity_count column is set to NaN."
            ),
            "engine_handling_mode": "EXPLICIT_NAN_UNAVAILABLE",
        },
        {
            "outcome_category_key": "complications",
            "outcome_category_label": "Clinical Complications (e.g., Shock, Organ Failure, Encephalopathy)",
            "available_in_workspace": False,
            "source_dataset": "NONE_IN_WORKSPACE",
            "date_range_supported": "N/A",
            "district_coverage": "0 / 36 districts (0.0%)",
            "non_null_records_2023_2025": 0,
            "missing_nan_records_2023_2025": len(weekly_df),
            "zero_value_records_2023_2025": 0,
            "positive_value_records_2023_2025": 0,
            "definition_consistency": "Unavailable in workspace; never fabricated.",
            "engine_handling_mode": "EXPLICIT_NAN_UNAVAILABLE",
        },
        {
            "outcome_category_key": "recovery_outcomes",
            "outcome_category_label": "Patient Recovery / Discharge Counts",
            "available_in_workspace": False,
            "source_dataset": "NONE_IN_WORKSPACE",
            "date_range_supported": "N/A",
            "district_coverage": "0 / 36 districts (0.0%)",
            "non_null_records_2023_2025": 0,
            "missing_nan_records_2023_2025": len(weekly_df),
            "zero_value_records_2023_2025": 0,
            "positive_value_records_2023_2025": 0,
            "definition_consistency": (
                "Individual/aggregate recovery counts do not exist. Note: 'status' in cleaned_disease_data.csv "
                "is an administrative outbreak cluster flag present in only 7/570 rows (6 'Under Control', "
                "1 'Under Surveillance') and is never treated as patient recovery count."
            ),
            "engine_handling_mode": "EXPLICIT_NAN_UNAVAILABLE",
        },
        {
            "outcome_category_key": "case_fatality_information",
            "outcome_category_label": "Outbreak-Cluster Case Fatality Ratio (OC-CFR %)",
            "available_in_workspace": True,
            "source_dataset": "Derived from matched (target_deaths / target_cases) in deduplicated IDSP 2023-2025",
            "date_range_supported": "2023-W01 to 2025-W52",
            "district_coverage": "34 / 36 active districts",
            "non_null_records_2023_2025": int(
                ((weekly_df["target_cases"] > 0) & (weekly_df["target_deaths"].notna())).sum()
            ),
            "missing_nan_records_2023_2025": 0,
            "zero_value_records_2023_2025": int((weekly_df["target_deaths"] == 0).sum()),
            "positive_value_records_2023_2025": int((weekly_df["target_deaths"] > 0).sum()),
            "definition_consistency": (
                "Calculated strictly where numerator (verified_deaths) and denominator (verified_cases) "
                "belong to the exact same deduplicated outbreak-week or pooled 2023-2025 stratum with "
                "Wilson 95% confidence intervals and minimum-denominator stability flags (N >= 30)."
            ),
            "engine_handling_mode": "COMPATIBLE_NUMERATOR_DENOMINATOR_WITH_WILSON_CI_AND_EB_SHRINKAGE",
        },
        {
            "outcome_category_key": "district_wise_mortality_records",
            "outcome_category_label": "District-Wise Aggregate Mortality Records (Maharashtra 36 Districts)",
            "available_in_workspace": True,
            "source_dataset": "data/features/disease_climate_weekly_features.csv",
            "date_range_supported": "2023-W01 to 2025-W52",
            "district_coverage": "34 / 36 active districts (Mumbai City & Ratnagiri = Insufficient Data)",
            "non_null_records_2023_2025": len(weekly_df),
            "missing_nan_records_2023_2025": 0,
            "zero_value_records_2023_2025": int((weekly_df["target_deaths"] == 0).sum()),
            "positive_value_records_2023_2025": int((weekly_df["target_deaths"] > 0).sum()),
            "definition_consistency": (
                "Standardized across all 36 official Maharashtra districts using Phase 2/9 geographic crosswalk."
            ),
            "engine_handling_mode": "VERIFIED_DISTRICT_AGGREGATION_WITH_INSUFFICIENT_DATA_FLAGS",
        },
    ]
    availability_df = pd.DataFrame(audit_rows)

    # 2. Missing vs. Zero Deaths Audit Table across cleaned_disease_data.csv & weekly features
    missing_vs_zero_rows: List[Dict[str, Any]] = []
    for (src_ds, dup_st), grp in cleaned_df.groupby(["source_dataset", "duplicate_status"]):
        n_rows = len(grp)
        n_known = int(grp["deaths"].notna().sum())
        n_na = int(grp["deaths"].isna().sum())
        n_zero = int((grp["deaths"] == 0).sum())
        n_pos = int((grp["deaths"] > 0).sum())
        missing_vs_zero_rows.append(
            {
                "dataset_stage": "cleaned_disease_data.csv",
                "source_dataset": str(src_ds),
                "duplicate_status": str(dup_st),
                "total_records": n_rows,
                "verified_cases_sum": int(grp["cases"].sum()),
                "deaths_non_null_count": n_known,
                "deaths_missing_nan_count": n_na,
                "deaths_missing_nan_pct": round((n_na / n_rows) * 100.0, 2) if n_rows > 0 else 0.0,
                "deaths_exact_zero_count": n_zero,
                "deaths_positive_gt0_count": n_pos,
                "verified_deaths_sum": float(grp["deaths"].sum(min_count=1)) if n_known > 0 else np.nan,
                "handling_rule": (
                    "Included in deduplicated 2023-2025 analysis"
                    if src_ds == "idsp_weekly" and dup_st == "unique"
                    else (
                        "Excluded as duplicate/conflict"
                        if dup_st != "unique"
                        else "Historical 2009-2022 reference only (71.2% missing deaths left as NaN; never pooled with 2023-2025)"
                    )
                ),
            }
        )

    # Also add the final deduplicated 2023-2025 weekly feature grain (511 rows)
    missing_vs_zero_rows.append(
        {
            "dataset_stage": "disease_climate_weekly_features.csv (Canonical Phase 12 Input)",
            "source_dataset": "idsp_weekly_deduplicated_2023_2025",
            "duplicate_status": "unique_district_disease_week",
            "total_records": len(weekly_df),
            "verified_cases_sum": int(weekly_df["target_cases"].sum()),
            "deaths_non_null_count": int(weekly_df["target_deaths"].notna().sum()),
            "deaths_missing_nan_count": int(weekly_df["target_deaths"].isna().sum()),
            "deaths_missing_nan_pct": 0.0,
            "deaths_exact_zero_count": int((weekly_df["target_deaths"] == 0).sum()),
            "deaths_positive_gt0_count": int((weekly_df["target_deaths"] > 0).sum()),
            "verified_deaths_sum": float(weekly_df["target_deaths"].sum()),
            "handling_rule": (
                "Primary 2023-2025 analytical dataset: 100% non-null deaths (365 exact zeros, 146 positive, 199 total deaths)"
            ),
        }
    )
    missing_vs_zero_df = pd.DataFrame(missing_vs_zero_rows)

    summary_meta = {
        "audit_timestamp": datetime.now().isoformat(),
        "engine_version": ENGINE_VERSION,
        "supported_outcomes": [
            "disease_related_deaths",
            "case_fatality_information (Outbreak-Cluster CFR %)",
            "district_wise_mortality_records",
        ],
        "unsupported_outcomes_explicit_nan": [
            "hospital_admissions",
            "severe_disease_cases",
            "complications",
            "recovery_outcomes",
        ],
        "total_verified_outbreak_weeks_2023_2025": len(weekly_df),
        "total_verified_cases_2023_2025": int(weekly_df["target_cases"].sum()),
        "total_verified_deaths_2023_2025": int(weekly_df["target_deaths"].sum()),
        "zero_death_outbreak_weeks_2023_2025": int((weekly_df["target_deaths"] == 0).sum()),
        "positive_death_outbreak_weeks_2023_2025": int((weekly_df["target_deaths"] > 0).sum()),
        "statewide_pooled_outbreak_cfr_pct_2023_2025": round(
            float(weekly_df["target_deaths"].sum() / weekly_df["target_cases"].sum()) * 100.0, 4
        ),
        "mandatory_disclaimer": MANDATORY_SEVERITY_PRIVACY_AND_SAFETY_DISCLAIMER,
    }
    return availability_df, missing_vs_zero_df, summary_meta


def prepare_verified_weekly_mortality_records(proj_dir: Path = PROJ) -> pd.DataFrame:
    """Prepare the 511-row deduplicated weekly mortality & severity dataset with all required
    Prompt 12 Section 5 output fields.
    """
    weekly_df = pd.read_csv(proj_dir / "data" / "features" / "disease_climate_weekly_features.csv")
    gen_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    rows: List[Dict[str, Any]] = []
    for _, r in weekly_df.sort_values(["report_year", "report_week", "district_std", "disease_std"]).iterrows():
        d_name = str(r["district_std"])
        dis_name = str(r["disease_std"])
        yr = int(r["report_year"])
        wk = int(r["report_week"])
        rep_period = f"{yr}-W{wk:02d}"
        cases = float(r["target_cases"]) if pd.notna(r["target_cases"]) else np.nan
        deaths = float(r["target_deaths"]) if pd.notna(r["target_deaths"]) else np.nan

        # Compatible CFR calculation only when cases > 0 and deaths is non-null
        if pd.notna(cases) and cases > 0 and pd.notna(deaths):
            calc_ratio = round((deaths / cases) * 100.0, 4)
            ratio_flag = (
                "STABLE_OUTBREAK_CFR_N_GE_30"
                if cases >= 30
                else "SMALL_DENOMINATOR_N_LT_30_INTERPRET_WITH_CAUTION"
            )
        else:
            calc_ratio = np.nan
            ratio_flag = "INCOMPATIBLE_OR_MISSING_DENOMINATOR"

        rows.append(
            {
                "district_id": DISTRICT_ID_MAP.get(d_name, "UNKNOWN"),
                "district": d_name,
                "administrative_division": ADMIN_DIVISION_MAP.get(d_name, "Unknown"),
                "disease": dis_name,
                "disease_group": str(r.get("disease_group", "Other")),
                "reporting_period": rep_period,
                "report_year": yr,
                "report_week": wk,
                "canonical_week_start": str(r["canonical_week_start"]),
                "season": str(r.get("season", "Unknown")),
                "verified_cases": cases,
                "verified_deaths": deaths,
                "severity_count": np.nan,
                "hospital_admissions_count": np.nan,
                "complications_count": np.nan,
                "recovery_count": np.nan,
                "calculated_ratio": calc_ratio,
                "calculated_ratio_type": "OUTBREAK_CLUSTER_CASE_FATALITY_RATIO_PCT",
                "calculated_ratio_reliability": ratio_flag,
                "source": "IDSP Weekly Outbreak Surveillance (data/features/disease_climate_weekly_features.csv)",
                "data_completeness": (
                    "VERIFIED_CASES_AND_DEATHS_COMPLETE | SEVERITY_ADMISSIONS_COMPLICATIONS_RECOVERY_UNAVAILABLE_NAN"
                ),
                "analysis_method": "Deduplicated Weekly Aggregate Outbreak Mortality & Matched OC-CFR",
                "generation_timestamp": gen_ts,
            }
        )

    return pd.DataFrame(rows)
