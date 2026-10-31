"""End-to-End Pipeline for Phase 12 Disease Severity & Mortality Intelligence Engine."""
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.engines.severity_intelligence.data_preparation import (
    ENGINE_VERSION,
    MANDATORY_SEVERITY_PRIVACY_AND_SAFETY_DISCLAIMER,
    audit_outcome_and_severity_data_availability,
    prepare_verified_weekly_mortality_records,
)
from src.engines.severity_intelligence.mortality_analysis import (
    compute_disease_mortality_summary,
    compute_district_mortality_summary,
    compute_temporal_and_seasonal_mortality_trends,
)
from src.engines.severity_intelligence.mortality_forecasting import (
    evaluate_chronological_mortality_forecasting,
)
from src.engines.severity_intelligence.visualizations import generate_all_severity_visualizations

PROJ = Path(__file__).resolve().parents[3]


def _clean_nan_for_json(obj: Any) -> Any:
    """Recursively convert NaN/inf to None for strict JSON serialization."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _clean_nan_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nan_for_json(v) for v in obj]
    return obj


def write_severity_mortality_report(
    availability_df: pd.DataFrame,
    missing_vs_zero_df: pd.DataFrame,
    audit_meta: Dict[str, Any],
    disease_summary_df: pd.DataFrame,
    district_summary_df: pd.DataFrame,
    annual_df: pd.DataFrame,
    seasonal_df: pd.DataFrame,
    temporal_stats: Dict[str, Any],
    sufficiency_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    report_path: Path,
) -> Path:
    """Write the comprehensive Phase 12 Markdown technical report."""
    lines: List[str] = [
        "# JEEVAN-DRISHTI Phase 12: Disease Severity & Mortality Intelligence Report",
        "",
        f"> **Engine Version**: `{ENGINE_VERSION}` | **Audit Date**: `{datetime.now().strftime('%Y-%m-%d')}`",
        f"> **Mandatory Privacy & Clinical Safety Notice**: {MANDATORY_SEVERITY_PRIVACY_AND_SAFETY_DISCLAIMER}",
        "",
        "---",
        "",
        "## 1. Outcome & Severity Data Availability Assessment",
        "",
        "The existing **JEEVAN-DRISHTI** workspace was audited across all **7 required outcome and severity categories**:",
        "",
        "| Outcome Category Key | Label | Available | Coverage (`2023–2025`) | Non-Null Rows | Missing (`NaN`) Rows | Exact Zero (`0`) Rows | Positive (`>0`) Rows | Engine Handling Mode |",
        "| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :--- |",
    ]
    for _, r in availability_df.iterrows():
        lines.append(
            f"| `{r['outcome_category_key']}` | {r['outcome_category_label']} | `{bool(r['available_in_workspace'])}` | "
            f"{r['district_coverage']} | {int(r['non_null_records_2023_2025'])} | {int(r['missing_nan_records_2023_2025'])} | "
            f"{int(r['zero_value_records_2023_2025'])} | {int(r['positive_value_records_2023_2025'])} | `{r['engine_handling_mode']}` |"
        )

    lines.extend(
        [
            "",
            "### Missing (`NaN`) vs. Exact Zero (`0`) Mortality Audit Across Workspace Stages",
            "",
            "| Dataset Stage | Source Dataset | Duplicate Status | Total Records | Verified Cases | Non-Null Deaths | Missing (`NaN`) Deaths (`%`) | Exact Zero (`0`) Deaths | Positive (`>0`) Deaths | Verified Deaths Sum | Handling Rule |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]
    )
    for _, r in missing_vs_zero_df.iterrows():
        d_sum = f"{int(r['verified_deaths_sum'])}" if pd.notna(r["verified_deaths_sum"]) else "NaN"
        lines.append(
            f"| `{r['dataset_stage']}` | `{r['source_dataset']}` | `{r['duplicate_status']}` | "
            f"{int(r['total_records'])} | {int(r['verified_cases_sum'])} | {int(r['deaths_non_null_count'])} | "
            f"`{int(r['deaths_missing_nan_count'])} ({r['deaths_missing_nan_pct']:.2f}%)` | {int(r['deaths_exact_zero_count'])} | "
            f"{int(r['deaths_positive_gt0_count'])} | **{d_sum}** | {r['handling_rule']} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Disease-Wise Mortality & Outbreak-Cluster Case Fatality Ratio (`OC-CFR`) Analysis (`2023–2025`)",
            "",
            "- **Methodological Definition**: Because IDSP records outbreak clusters rather than all asymptomatic/mild community infections, the ratio $\\text{OC-CFR} = \\frac{\\text{Verified Outbreak Deaths}}{\\text{Verified Outbreak Cases}} \\times 100\\%$ measures **Outbreak-Cluster Case Fatality Ratio**. It is computed **strictly where numerator and denominator come from the exact same deduplicated `2023–2025` outbreak records** (`verified_cases > 0` and `verified_deaths` non-null).",
            "- **Small-Denominator Safeguard**: Exact **Wilson score 95% confidence intervals** and **Empirical Bayes shrinkage** ($M = 50$ pseudocounts toward the statewide outbreak CFR prior of `1.038%`) are computed alongside crude ratios so that tiny-denominator diseases ($N < 30$, such as `Rabies (human)` with $2/2$ deaths or `Japanese Encephalitis` with $1/3$ deaths) are never misinterpreted as stable population fatality rates.",
            "",
            "### All 15 Diseases with $\\ge 1$ Verified Outbreak Death in `2023–2025` (Plus Top Zero-Death High-Volume Diseases)",
            "",
            "| Disease | Disease Group | Outbreak Weeks | Positive-Death Weeks | Verified Cases | Verified Deaths | Crude OC-CFR (`%`) | Wilson 95% CI (`%`) | Empirical Bayes Shrunk CFR (`%`) | Denominator Stability Status |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]
    )
    for _, r in disease_summary_df.head(19).iterrows():
        lines.append(
            f"| **{r['disease']}** | {r['disease_group']} | {int(r['outbreak_weeks_count'])} | "
            f"{int(r['positive_death_outbreak_weeks'])} ({r['positive_death_weeks_pct']:.1f}%) | "
            f"{int(r['verified_cases'])} | **{int(r['verified_deaths'])}** | `{r['calculated_ratio']:.2f}%` | "
            f"`[{r['wilson_95ci_lower_pct']:.2f}%, {r['wilson_95ci_upper_pct']:.2f}%]` | "
            f"`{r['empirical_bayes_shrunk_cfr_pct']:.2f}%` | `{r['denominator_stability_status']}` |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. District-Wise Mortality & Severity Burden Comparison (`36` Official Maharashtra Districts)",
            "",
            "- **Active Surveillance Coverage**: `34 / 36` districts have verified `2023–2025` outbreak mortality records (`22` districts recorded $\\ge 1$ death; `12` active districts recorded `0` deaths across all reported outbreaks; `2` districts — `Mumbai City` and `Ratnagiri` — had `0` reported outbreaks in `2023–2025` and are explicitly marked `Insufficient Data`).",
            "",
            "### Top 15 Maharashtra Districts by Total Verified Outbreak Deaths (`2023–2025`)",
            "",
            "| Rank | District ID | District | Division | Outbreak Weeks | Positive-Death Weeks | Verified Cases | Verified Deaths | Crude OC-CFR (`%`) | Wilson 95% CI (`%`) | EB Shrunk CFR (`%`) | Dominant Fatal Disease |",
            "| :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]
    )
    top15_dist = district_summary_df[district_summary_df["verified_deaths"].notna()].head(15)
    for idx, (_, r) in enumerate(top15_dist.iterrows(), start=1):
        lines.append(
            f"| {idx} | `{r['district_id']}` | **{r['district']}** | {r['administrative_division']} | "
            f"{int(r['outbreak_weeks_count'])} | {int(r['positive_death_outbreak_weeks'])} | "
            f"{int(r['verified_cases'])} | **{int(r['verified_deaths'])}** | `{r['calculated_ratio']:.2f}%` | "
            f"`[{r['wilson_95ci_lower_pct']:.2f}%, {r['wilson_95ci_upper_pct']:.2f}%]` | "
            f"`{r['empirical_bayes_shrunk_cfr_pct']:.2f}%` | {r['dominant_fatal_disease']} ({int(r['dominant_fatal_disease_deaths'])}) |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Temporal (`2023–2025`) & Seasonal Mortality Trends",
            "",
            "### A. Annual Statewide Mortality Progression (`ALL_DISEASES`)",
            "",
            "| Reporting Period | Active Outbreak Weeks | Weeks with $\\ge 1$ Death | Verified Cases | Verified Deaths | Pooled OC-CFR (`%`) | Wilson 95% CI (`%`) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for _, r in annual_df[annual_df["disease"] == "ALL_DISEASES"].iterrows():
        lines.append(
            f"| **{r['reporting_period']}** | {int(r['outbreak_weeks_count'])} | {int(r['positive_death_weeks'])} | "
            f"{int(r['verified_cases'])} | **{int(r['verified_deaths'])}** | `{r['calculated_ratio']:.2f}%` | "
            f"`[{r['wilson_95ci_lower_pct']:.2f}%, {r['wilson_95ci_upper_pct']:.2f}%]` |"
        )

    lines.extend(
        [
            "",
            f"### B. Seasonal Mortality Distribution (Kruskal-Wallis $H = {temporal_stats.get('kruskal_wallis_seasonal_deaths_stat')}$, $p = {temporal_stats.get('kruskal_wallis_seasonal_deaths_pvalue')}$)",
            "",
            "| Season | Outbreak Weeks | Positive-Death Weeks (`%`) | Verified Cases | Verified Deaths | Share of Statewide Deaths (`%`) | Pooled OC-CFR (`%`) | Mean Deaths / Outbreak Week |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for _, r in seasonal_df.iterrows():
        lines.append(
            f"| **{r['season']}** | {int(r['outbreak_weeks_count'])} | "
            f"{int(r['positive_death_weeks'])} ({r['positive_death_weeks_pct']:.1f}%) | "
            f"{int(r['verified_cases'])} | **{int(r['verified_deaths'])}** | `{r['share_of_statewide_deaths_pct']:.1f}%` | "
            f"`{r['calculated_ratio']:.2f}%` | `{r['mean_deaths_per_outbreak_week']:.3f}` |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 5. Mortality Time-Series Forecasting Sufficiency & Chronological Evaluation (Section 4)",
            "",
            "### A. Data Sufficiency Across Analytical Grains",
            "",
            "| Analytical Grain | Active Time Points | Positive-Death Time Points | Zero-Death (`%`) | Total Verified Deaths | Forecasting Supported | Recommended Analytical Mode |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]
    )
    for _, r in sufficiency_df.iterrows():
        lines.append(
            f"| `{r['analytical_grain']}` | {int(r['active_time_points'])} | {int(r['positive_death_time_points'])} | "
            f"`{r['zero_death_pct']:.2f}%` | {int(r['total_verified_deaths'])} | `{bool(r['forecasting_supported'])}` | "
            f"`{r['recommended_mode']}` |"
        )

    lines.extend(
        [
            "",
            "### B. Chronological Forecasting Validation (`70%` Train / `15%` Validation / `15%` Test — Strictly Lagged $t-1..t-4$ Predictors)",
            "",
            "| Target Series | Model Name | Split | Weeks ($N$) | Observed Deaths | Predicted Deaths | MAE | RMSE | 80% PI Coverage (`%`) | MAE Skill vs. Last-Value Naive (`%`) |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for _, r in metrics_df.iterrows():
        lines.append(
            f"| `{r['target_series']}` | **{r['model_name']}** | `{r['evaluation_split']}` | "
            f"{int(r['n_weeks'])} | {int(r['observed_deaths_sum'])} | {r['predicted_deaths_sum']:.1f} | "
            f"**{r['mae']:.4f}** | {r['rmse']:.4f} | `{r['pi_80_coverage_pct']:.1f}%` | "
            f"`{r['mae_skill_vs_last_value_naive_pct']:+.2f}%` |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 6. Privacy, Clinical Safety & Limitations Summary",
            "",
            "1. **Aggregate Data Only**: Every record represents an aggregate district-week or statewide-week IDSP outbreak summary. Zero personally identifiable patient attributes exist in or are emitted by this engine.",
            "2. **No Individual Clinical Prognosis**: Outbreak-Cluster CFR (`OC-CFR %`) reflects historical fatality among reported outbreak clusters and must **never** be interpreted as an individual patient's probability of death or used to guide clinical triage/treatment.",
            "3. **Explicit Unavailable Outcome Preservation**: Hospital admissions, clinical severity counts (`severity_count`), complications, and recovery counts are `100%` absent from the workspace and remain explicitly `NaN` across all outputs.",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def update_model_dataset_registry(
    proj_dir: Path,
    audit_meta: Dict[str, Any],
    metrics_df: pd.DataFrame,
) -> Path:
    """Register Phase 12 Disease Severity & Mortality Intelligence Engine in model_dataset_registry.json."""
    reg_path = proj_dir / "model_dataset_registry.json"
    if reg_path.exists():
        reg = json.loads(reg_path.read_text(encoding="utf-8"))
    else:
        reg = {}

    reg["phase12_severity_mortality_intelligence_engine"] = {
        "engine_name": "Disease Severity and Mortality Intelligence Engine",
        "version": ENGINE_VERSION,
        "updated_at": datetime.now().isoformat(),
        "supported_outcomes": audit_meta["supported_outcomes"],
        "unsupported_outcomes_explicit_nan": audit_meta["unsupported_outcomes_explicit_nan"],
        "total_verified_outbreak_weeks_2023_2025": audit_meta["total_verified_outbreak_weeks_2023_2025"],
        "total_verified_cases_2023_2025": audit_meta["total_verified_cases_2023_2025"],
        "total_verified_deaths_2023_2025": audit_meta["total_verified_deaths_2023_2025"],
        "statewide_pooled_outbreak_cfr_pct_2023_2025": audit_meta["statewide_pooled_outbreak_cfr_pct_2023_2025"],
        "chronological_forecast_models_evaluated": int(metrics_df["model_name"].nunique()),
        "mandatory_disclaimer": MANDATORY_SEVERITY_PRIVACY_AND_SAFETY_DISCLAIMER,
        "artifacts": {
            "outcome_availability_audit_csv": "data/severity_intelligence/outcome_data_availability_audit.csv",
            "missing_vs_zero_audit_csv": "data/severity_intelligence/deaths_missing_vs_zero_audit.csv",
            "verified_weekly_mortality_records_csv": "data/severity_intelligence/verified_weekly_mortality_records.csv",
            "disease_mortality_summary_csv": "data/severity_intelligence/disease_mortality_summary.csv",
            "district_mortality_summary_csv": "data/severity_intelligence/district_mortality_summary.csv",
            "annual_mortality_trends_csv": "data/severity_intelligence/annual_mortality_trends.csv",
            "seasonal_mortality_trends_csv": "data/severity_intelligence/seasonal_mortality_trends.csv",
            "weekly_statewide_mortality_series_csv": "data/severity_intelligence/weekly_statewide_mortality_series.csv",
            "forecast_sufficiency_audit_csv": "data/severity_intelligence/mortality_forecasting_sufficiency_audit.csv",
            "forecast_model_comparison_csv": "data/severity_intelligence/mortality_forecast_model_comparison.csv",
            "forecast_actual_vs_predicted_csv": "data/severity_intelligence/mortality_forecast_actual_vs_predicted.csv",
            "technical_report_md": "reports/severity_intelligence/severity_mortality_intelligence_report.md",
        },
    }
    reg_path.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    return reg_path


def run_severity_intelligence_pipeline(proj_dir: Path = PROJ) -> Dict[str, Any]:
    """Execute the complete Phase 12 Disease Severity & Mortality Intelligence pipeline."""
    data_dir = proj_dir / "data" / "severity_intelligence"
    reports_dir = proj_dir / "reports" / "severity_intelligence"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Audit outcome/severity availability and missing vs zero deaths
    availability_df, missing_vs_zero_df, audit_meta = audit_outcome_and_severity_data_availability(proj_dir)

    # 2. Prepare canonical 511-row deduplicated weekly mortality & severity dataset
    weekly_records_df = prepare_verified_weekly_mortality_records(proj_dir)

    # 3. Compute disease-wise, district-wise, and temporal/seasonal mortality analyses
    disease_summary_df = compute_disease_mortality_summary(weekly_records_df)
    district_summary_df = compute_district_mortality_summary(weekly_records_df)
    annual_df, seasonal_df, weekly_series_df, temporal_stats = (
        compute_temporal_and_seasonal_mortality_trends(weekly_records_df)
    )

    # 4. Evaluate chronological mortality forecasting where supported
    sufficiency_df, metrics_df, preds_df = evaluate_chronological_mortality_forecasting(
        weekly_records_df, weekly_series_df
    )

    # 5. Save all CSVs and frontend/API summary JSON to data/ and reports/
    csv_outputs = {
        "outcome_data_availability_audit.csv": availability_df,
        "deaths_missing_vs_zero_audit.csv": missing_vs_zero_df,
        "verified_weekly_mortality_records.csv": weekly_records_df,
        "disease_mortality_summary.csv": disease_summary_df,
        "district_mortality_summary.csv": district_summary_df,
        "annual_mortality_trends.csv": annual_df,
        "seasonal_mortality_trends.csv": seasonal_df,
        "weekly_statewide_mortality_series.csv": weekly_series_df,
        "mortality_forecasting_sufficiency_audit.csv": sufficiency_df,
        "mortality_forecast_model_comparison.csv": metrics_df,
        "mortality_forecast_actual_vs_predicted.csv": preds_df,
    }
    for fname, df in csv_outputs.items():
        p_dat = data_dir / fname
        p_rep = reports_dir / fname
        df.to_csv(p_dat, index=False)
        shutil.copy2(p_dat, p_rep)

    summary_payload = {
        "engine_version": ENGINE_VERSION,
        "generated_at": datetime.now().isoformat(),
        "mandatory_disclaimer": MANDATORY_SEVERITY_PRIVACY_AND_SAFETY_DISCLAIMER,
        "audit_summary": audit_meta,
        "temporal_stats": temporal_stats,
        "disease_mortality_summary": _clean_nan_for_json(disease_summary_df.to_dict(orient="records")),
        "district_mortality_summary": _clean_nan_for_json(district_summary_df.to_dict(orient="records")),
        "seasonal_mortality_summary": _clean_nan_for_json(seasonal_df.to_dict(orient="records")),
        "forecasting_sufficiency": _clean_nan_for_json(sufficiency_df.to_dict(orient="records")),
        "forecast_model_metrics": _clean_nan_for_json(metrics_df.to_dict(orient="records")),
    }
    json_dat = data_dir / "severity_mortality_summary_payload.json"
    json_rep = reports_dir / "severity_mortality_summary_payload.json"
    json_dat.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    shutil.copy2(json_dat, json_rep)

    # 6. Generate all 5 static PNG figures
    figs = generate_all_severity_visualizations(
        availability_df=availability_df,
        missing_vs_zero_df=missing_vs_zero_df,
        disease_summary_df=disease_summary_df,
        district_summary_df=district_summary_df,
        weekly_series_df=weekly_series_df,
        seasonal_df=seasonal_df,
        temporal_stats=temporal_stats,
        metrics_df=metrics_df,
        preds_df=preds_df,
        reports_dir=reports_dir,
        data_dir=data_dir,
    )

    # 7. Write Markdown report and update model_dataset_registry.json
    report_md = write_severity_mortality_report(
        availability_df=availability_df,
        missing_vs_zero_df=missing_vs_zero_df,
        audit_meta=audit_meta,
        disease_summary_df=disease_summary_df,
        district_summary_df=district_summary_df,
        annual_df=annual_df,
        seasonal_df=seasonal_df,
        temporal_stats=temporal_stats,
        sufficiency_df=sufficiency_df,
        metrics_df=metrics_df,
        report_path=reports_dir / "severity_mortality_intelligence_report.md",
    )
    shutil.copy2(report_md, data_dir / "severity_mortality_intelligence_report.md")
    update_model_dataset_registry(proj_dir, audit_meta, metrics_df)

    return {
        "audit_meta": audit_meta,
        "diseases_count": len(disease_summary_df),
        "districts_count": len(district_summary_df),
        "weekly_records_count": len(weekly_records_df),
        "figures": figs,
        "report_path": str(report_md),
    }


if __name__ == "__main__":
    res = run_severity_intelligence_pipeline()
    print(
        f"[Phase 12 Pipeline Complete] Verified weekly outbreak records: {res['weekly_records_count']} | "
        f"Total verified deaths (2023-2025): {res['audit_meta']['total_verified_deaths_2023_2025']} | "
        f"Diseases analyzed: {res['diseases_count']} | Districts summarized: {res['districts_count']}"
    )
