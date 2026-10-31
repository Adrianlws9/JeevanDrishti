"""End-to-End Pipeline for Phase 11 Healthcare Resource Demand Intelligence Engine."""
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.engines.resource_intelligence.data_availability import (
    MANDATORY_RESOURCE_DISCLAIMER,
    audit_resource_data_availability,
    build_district_resource_and_epidemiological_summary,
)
from src.engines.resource_intelligence.scenario_calculator import (
    ENGINE_VERSION,
    build_default_resource_scenarios,
    evaluate_scenarios_across_districts,
)
from src.engines.resource_intelligence.visualizations import generate_all_resource_visualizations

PROJ = Path(__file__).resolve().parents[3]


def _clean_nan_for_json(obj: Any) -> Any:
    """Recursively convert NaN/inf to None so json.dumps produces strict JSON."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _clean_nan_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nan_for_json(v) for v in obj]
    return obj


def build_frontend_integration_payload(
    completeness_df: pd.DataFrame,
    audit_meta: Dict[str, Any],
    district_summary_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Prepare structured frontend payload covering all 6 required sections in Prompt 11 Section 7."""
    # 1. Resource overview
    active_epi_districts = int((district_summary_df["observed_2023_2025_outbreak_weeks"] > 0).sum())
    zero_epi_districts = int((district_summary_df["observed_2023_2025_outbreak_weeks"] == 0).sum())
    resource_overview = {
        "engine_version": ENGINE_VERSION,
        "generated_at": datetime.now().isoformat(),
        "mandatory_disclaimer": MANDATORY_RESOURCE_DISCLAIMER,
        "empirical_resource_datasets_in_workspace": 0,
        "empirical_resource_variables_available": 0,
        "total_maharashtra_districts": len(district_summary_df),
        "districts_with_upstream_epidemiological_anchors": active_epi_districts,
        "districts_with_insufficient_surveillance_anchors": zero_epi_districts,
        "operating_mode": "EXPLICIT_INSUFFICIENT_EMPIRICAL_DATA_PLUS_HYPOTHETICAL_SCENARIO_CALCULATOR",
    }

    # 2. Bed availability (strictly non-fabricated; explicit Insufficient Data per district)
    bed_availability = {
        "status": "INSUFFICIENT_DATA_NO_BED_INVENTORY_IN_WORKSPACE",
        "explanation": (
            "No dataset in the workspace contains actual hospital available beds, occupied beds, "
            "or ICU capacity. All empirical bed columns are explicitly null (NaN)."
        ),
        "districts": _clean_nan_for_json(
            district_summary_df[
                [
                    "district_id",
                    "district_name",
                    "administrative_division",
                    "empirical_available_beds",
                    "empirical_occupied_beds",
                    "empirical_bed_occupancy_ratio",
                    "empirical_icu_capacity",
                    "empirical_resource_shortage_status",
                ]
            ].to_dict(orient="records")
        ),
    }

    # 3. Demand forecast (explains why empirical resource ML forecasting is blocked + provides upstream case forecast context)
    demand_forecast = {
        "empirical_resource_ml_forecast_status": "BLOCKED_NO_HISTORICAL_RESOURCE_TARGETS",
        "scientific_rationale": audit_meta["why_empirical_forecasting_is_blocked"],
        "upstream_epidemiological_case_forecasts_by_district": _clean_nan_for_json(
            district_summary_df[
                [
                    "district_id",
                    "district_name",
                    "observed_2023_2025_mean_cases_per_outbreak_week",
                    "observed_2023_2025_peak_weekly_cases",
                    "upstream_phase6_forecast_7d_mean_cases",
                    "upstream_phase6_forecast_14d_mean_cases",
                    "upstream_phase6_forecast_30d_mean_cases",
                    "upstream_phase5_risk_category",
                    "upstream_phase7_anomaly_weeks_count",
                ]
            ].to_dict(orient="records")
        ),
    }

    # 4. Capacity comparison (Strictly hypothetical comparison against user-assumed capacities)
    scen01_eval = eval_df[eval_df["scenario_id"] == "RES_SCEN_01_FORECAST_BASELINE_MODERATE"]
    capacity_comparison = {
        "comparison_nature": "HYPOTHETICAL_SCENARIO_ONLY_AGAINST_USER_ASSUMED_REFERENCE_CAPACITY",
        "warning": (
            "Assumed bed capacities (e.g., 40 general beds, 6 ICU beds) are user-editable reference "
            "inputs for hypothetical stress-testing, NOT actual district hospital inventories."
        ),
        "reference_scenario_id": "RES_SCEN_01_FORECAST_BASELINE_MODERATE",
        "district_hypothetical_comparisons": _clean_nan_for_json(
            scen01_eval[
                [
                    "district_id",
                    "district_name",
                    "anchor_baseline_cases",
                    "hypothetical_projected_weekly_cases",
                    "hypothetical_concurrent_general_beds_required",
                    "user_assumed_available_general_beds",
                    "hypothetical_general_bed_utilization_ratio",
                    "hypothetical_general_bed_gap",
                    "hypothetical_concurrent_icu_beds_required",
                    "user_assumed_available_icu_beds",
                    "hypothetical_icu_bed_utilization_ratio",
                    "hypothetical_icu_bed_gap",
                    "hypothetical_shortage_indicator",
                    "output_nature",
                ]
            ].to_dict(orient="records")
        ),
    }

    # 5. Scenario planning (Editable parameter schema + 4 benchmark scenarios)
    scenario_planning = {
        "output_nature": "HYPOTHETICAL_SCENARIO_NOT_ACTUAL_CAPACITY_OR_FORECAST",
        "editable_parameters_schema": [
            {
                "name": "case_load_anchor_mode",
                "type": "enum",
                "options": [
                    "phase6_forecast_7d_cases",
                    "observed_peak_weekly_cases",
                    "observed_mean_weekly_cases",
                    "user_entered_cases",
                ],
                "default": "phase6_forecast_7d_cases",
                "description": "Source for baseline weekly disease case count before applying hypothetical surge.",
            },
            {
                "name": "hypothetical_case_increase_pct",
                "type": "float",
                "min": -90.0,
                "max": 1000.0,
                "default": 25.0,
                "unit": "%",
                "description": "Hypothetical percentage change in weekly cases.",
            },
            {
                "name": "assumed_admission_proportion",
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.15,
                "unit": "proportion [0..1]",
                "description": "Assumed fraction of cases requiring inpatient hospital admission.",
            },
            {
                "name": "assumed_icu_proportion_of_admissions",
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.10,
                "unit": "proportion [0..1]",
                "description": "Assumed fraction of admitted patients requiring ICU care.",
            },
            {
                "name": "assumed_avg_length_of_stay_days",
                "type": "float",
                "min": 1.0,
                "max": 30.0,
                "default": 4.5,
                "unit": "days",
                "description": "Assumed average inpatient length of stay in days.",
            },
            {
                "name": "assumed_tests_per_case",
                "type": "float",
                "min": 0.0,
                "max": 25.0,
                "default": 3.0,
                "unit": "tests/case",
                "description": "Assumed diagnostic tests per reported case (including suspect screening).",
            },
            {
                "name": "assumed_available_general_beds",
                "type": "float",
                "min": 1.0,
                "max": 50000.0,
                "default": 40.0,
                "unit": "beds (hypothetical)",
                "description": "User-assumed available general/isolation beds for hypothetical strain ratio.",
            },
            {
                "name": "assumed_available_icu_beds",
                "type": "float",
                "min": 1.0,
                "max": 10000.0,
                "default": 6.0,
                "unit": "ICU beds (hypothetical)",
                "description": "User-assumed available ICU beds for hypothetical strain ratio.",
            },
        ],
        "benchmark_scenarios_catalog": _clean_nan_for_json(catalog_df.to_dict(orient="records")),
    }

    # 6. Insufficient data indicators
    insufficient_data_indicators = {
        "missing_empirical_resource_categories": _clean_nan_for_json(
            completeness_df[~completeness_df["present_in_workspace"]].to_dict(orient="records")
        ),
        "districts_missing_surveillance_case_anchors_2023_2025": [
            {
                "district_id": str(r["district_id"]),
                "district_name": str(r["district_name"]),
                "reason": "0 reported IDSP outbreaks in 2023-2025; requires user_entered_cases mode in scenario calculator.",
            }
            for _, r in district_summary_df[
                district_summary_df["observed_2023_2025_outbreak_weeks"] == 0
            ].iterrows()
        ],
    }

    return {
        "resource_overview": resource_overview,
        "bed_availability": bed_availability,
        "demand_forecast": demand_forecast,
        "capacity_comparison": capacity_comparison,
        "scenario_planning": scenario_planning,
        "insufficient_data_indicators": insufficient_data_indicators,
    }


def write_resource_intelligence_report(
    completeness_df: pd.DataFrame,
    audit_meta: Dict[str, Any],
    district_summary_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    report_path: Path,
) -> Path:
    """Write comprehensive Markdown technical report for Phase 11."""
    top_epi = (
        district_summary_df[district_summary_df["observed_2023_2025_peak_weekly_cases"].notna()]
        .sort_values("observed_2023_2025_peak_weekly_cases", ascending=False)
        .head(10)
    )
    s3_top = (
        eval_df[
            (eval_df["scenario_id"] == "RES_SCEN_03_HISTORICAL_PEAK_STRESS_TEST")
            & (eval_df["hypothetical_general_bed_utilization_ratio"].notna())
        ]
        .sort_values("hypothetical_general_bed_utilization_ratio", ascending=False)
        .head(10)
    )

    lines: List[str] = [
        "# JEEVAN-DRISHTI Phase 11: Healthcare Resource Demand Intelligence Report",
        "",
        f"> **Engine Version**: `{ENGINE_VERSION}` | **Audit Date**: `{datetime.now().strftime('%Y-%m-%d')}`",
        f"> **Mandatory Scientific Disclaimer**: {MANDATORY_RESOURCE_DISCLAIMER}",
        "",
        "---",
        "",
        "## 1. Executive Summary & Workspace Data Availability Audit",
        "",
        "Before implementing any resource demand analysis or forecasting, the entire **JEEVAN-DRISHTI** workspace (`data/raw/`, `data/processed/`, `data/features/`, `data/final_training/`, `data/geospatial/`, `data/scenarios/`, and `data/reports/dataset_inventory.csv`) was audited across all **9 required healthcare resource categories**:",
        "",
        "| Variable Key | Variable Label | Domain Category | Present in Workspace | Coverage | Missingness (%) | Engine Handling Mode |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :--- |",
    ]
    for _, r in completeness_df.iterrows():
        lines.append(
            f"| `{r['variable_key']}` | {r['variable_label']} | {r['domain_category']} | "
            f"`{bool(r['present_in_workspace'])}` | {r['district_coverage']} | `{r['missingness_pct']:.2f}%` | `{r['engine_handling_mode']}` |"
        )

    lines.extend(
        [
            "",
            "### Key Findings from the Data Availability Assessment",
            "1. **Zero Empirical Healthcare Resource Datasets Exist in the Workspace**: None of the project files contain hospital admissions (`IPD`), available beds, occupied beds, ICU/ventilator capacity, diagnostic testing volume, medicine consumption/stock, or health facility inventories (`0 / 36` districts, `100%` missingness).",
            "2. **Strict Non-Fabrication Enforcement**: In compliance with **Prompt 11 Sections 2, 4, and 5**, this engine **does not invent hospital capacities, bed counts, or medicine stock values**, and **does not convert disease cases into empirical hospital admissions using hidden or arbitrary percentages**.",
            "3. **Why Empirical ML Resource Forecasting Is Blocked**: Because historical resource target time series (`hospital_admissions_t`, `occupied_beds_t`, `testing_volume_t`) are `100%` absent (`NaN`), training an ETS, SARIMA, or XGBoost model on synthetic targets derived from multiplying disease cases by a constant would be circular and scientifically invalid. Empirical resource forecasting is therefore explicitly marked `UNSUPPORTED_NO_HISTORICAL_RESOURCE_DATA`.",
            "4. **Activation of the Transparent Hypothetical Scenario Calculator Fallback (Section 5)**: To enable public-health contingency planning without fabricating data, the engine implements a deterministic, user-editable **Hypothetical Scenario Calculator** (`src/engines/resource_intelligence/scenario_calculator.py`) where every assumption is explicitly visible, editable, and stamped with `HYPOTHETICAL_SCENARIO_NOT_ACTUAL_CAPACITY_OR_FORECAST`.",
            "",
            "---",
            "",
            "## 2. Verified Upstream Epidemiological Demand Drivers (`2023–2025`)",
            "",
            "While hospital utilization records are absent, the project contains verified district-level epidemiological surveillance caseloads (`2023–2025` IDSP outbreaks across `34 / 36` active districts, plus Phase 6 multi-horizon case forecasts). These upstream epidemiological metrics are preserved strictly as optional baseline anchors for the hypothetical scenario calculator:",
            "",
            "| Rank | District ID | District Name | Division | Outbreak Weeks (`23–25`) | Total Cases (`23–25`) | Mean Cases / Active Wk | Peak Weekly Cases (`23–25`) | Phase 6 `7d` Forecast Mean | Dominant Disease (`23–25`) | Empirical Hospital Beds / Admissions |",
            "| :---: | :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: |",
        ]
    )
    for idx, (_, r) in enumerate(top_epi.iterrows(), start=1):
        lines.append(
            f"| {idx} | `{r['district_id']}` | **{r['district_name']}** | {r['administrative_division']} | "
            f"{int(r['observed_2023_2025_outbreak_weeks'])} | {int(r['observed_2023_2025_total_outbreak_cases'])} | "
            f"{r['observed_2023_2025_mean_cases_per_outbreak_week']:.2f} | **{r['observed_2023_2025_peak_weekly_cases']:.0f}** | "
            f"{r['upstream_phase6_forecast_7d_mean_cases']:.2f} | {r['dominant_outbreak_disease_2023_2025']} | `NaN (Insufficient Data)` |"
        )

    lines.extend(
        [
            "",
            "- **Insufficient Surveillance Anchor Districts (`2 / 36`)**: `Mumbai City` (`MH_D18`) and `Ratnagiri` (`MH_D28`) have `0` reported outbreak events in `2023–2025`. For these districts, surveillance-anchored scenarios automatically return `Insufficient Data`, while users can still evaluate hypothetical demand using `case_load_anchor_mode='user_entered_cases'`.",
            "",
            "---",
            "",
            "## 3. Hypothetical Scenario Calculator Methodology (Section 5 Fallback)",
            "",
            "Given a baseline weekly case count $C_{\\text{base}}$ (either user-entered or anchored to observed/forecasted surveillance cases) and visible, user-editable parameters:",
            "- Hypothetical case surge percentage $\\Delta_{\\%} \\in [-90\\%, +1000\\%]$",
            "- Assumed inpatient admission proportion $\\alpha_{\\text{adm}} \\in [0, 1]$",
            "- Assumed ICU share of admissions $\\alpha_{\\text{icu}} \\in [0, 1]$",
            "- Assumed average length of stay $L_{\\text{stay}} \\in [1, 30]$ days",
            "- Assumed diagnostic tests per case $\\tau_{\\text{test}}$, IV fluid units per admission $\\nu_{\\text{iv}}$, and treatment courses per case $\\mu_{\\text{rx}}$",
            "- Optional user-assumed reference general beds $B_{\\text{gen}}^{\\text{assumed}}$ and ICU beds $B_{\\text{icu}}^{\\text{assumed}}$",
            "",
            "the calculator computes:",
            "$$C_{\\text{proj}} = C_{\\text{base}} \\times \\left(1 + \\frac{\\Delta_{\\%}}{100}\\right), \\quad A_{\\text{wk}} = C_{\\text{proj}} \\times \\alpha_{\\text{adm}}, \\quad D_{\\text{bed-days}} = A_{\\text{wk}} \\times L_{\\text{stay}}$$",
            "$$B_{\\text{gen, req}} = \\frac{A_{\\text{wk}} \\cdot (1 - \\alpha_{\\text{icu}}) \\cdot L_{\\text{stay}}}{7}, \\quad B_{\\text{icu, req}} = \\frac{A_{\\text{wk}} \\cdot \\alpha_{\\text{icu}} \\cdot L_{\\text{stay}}}{7}$$",
            "$$U_{\\text{gen}} = \\frac{B_{\\text{gen, req}}}{B_{\\text{gen}}^{\\text{assumed}}}, \\quad U_{\\text{icu}} = \\frac{B_{\\text{icu, req}}}{B_{\\text{icu}}^{\\text{assumed}}}$$",
            "",
            "### Benchmark Hypothetical Scenarios Evaluated Across All 36 Districts",
            "",
            "| Scenario ID | Anchor Mode | Surge (`%`) | Assumed Adm (`%`) | Assumed ICU (`%`) | Assumed LOS (`d`) | Statewide Hypothetical Weekly Cases | Statewide Hypothetical Weekly Admissions | Statewide Concurrent General Beds | Statewide Concurrent ICU Beds | Statewide Diagnostic Tests | Districts `>100%` of Assumed Ward |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for _, r in catalog_df.iterrows():
        lines.append(
            f"| `{r['scenario_id']}` | `{r['case_load_anchor_mode']}` | `+{r['hypothetical_case_increase_pct']:.0f}%` | "
            f"`{r['assumed_admission_proportion']*100:.0f}%` | `{r['assumed_icu_proportion_of_admissions']*100:.0f}%` | "
            f"`{r['assumed_avg_length_of_stay_days']:.1f}d` | {r['statewide_hypothetical_weekly_cases']:.1f} | "
            f"{r['statewide_hypothetical_weekly_admissions']:.1f} | **{r['statewide_hypothetical_general_beds_required']:.1f}** | "
            f"**{r['statewide_hypothetical_icu_beds_required']:.1f}** | {r['statewide_hypothetical_diagnostic_tests_required']:.0f} | "
            f"`{int(r['districts_exceeding_100pct_assumed_capacity'])} / {int(r['evaluated_districts_with_anchor'])}` |"
        )

    lines.extend(
        [
            "",
            "### Top 10 Districts Under `RES_SCEN_03_HISTORICAL_PEAK_STRESS_TEST` (Hypothetical Peak Weekly Load +25% vs. User-Assumed 50-Bed Isolation Ward)",
            "",
            "| District ID | District Name | Peak Weekly Base Cases | Hypothetical Projected Cases (`+25%`) | Hypothetical Weekly Admissions (`20%`) | Concurrent General Beds (`LOS=4d`) | Concurrent ICU Beds | Utilization of Assumed 50-Bed Ward | Hypothetical Shortage Indicator |",
            "| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]
    )
    for _, r in s3_top.iterrows():
        lines.append(
            f"| `{r['district_id']}` | **{r['district_name']}** | {r['anchor_baseline_cases']:.0f} | "
            f"{r['hypothetical_projected_weekly_cases']:.1f} | {r['hypothetical_weekly_admissions']:.1f} | "
            f"**{r['hypothetical_concurrent_general_beds_required']:.2f}** | {r['hypothetical_concurrent_icu_beds_required']:.2f} | "
            f"`{r['hypothetical_general_bed_utilization_ratio']*100:.1f}%` | `{r['hypothetical_shortage_indicator']}` |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Generated Datasets, Figures & Frontend Integration Artifacts",
            "",
            "- **Data Completeness Audit**: `data/resource_intelligence/resource_data_completeness_audit.csv`",
            "- **District Resource & Upstream Summary (`36` districts)**: `data/resource_intelligence/district_resource_summary.csv`",
            "- **Historical Weekly Epidemiological Demand Trends (`2023–2025`)**: `data/resource_intelligence/historical_epidemiological_demand_trends.csv`",
            "- **Benchmark Hypothetical Scenarios Catalog**: `data/resource_intelligence/hypothetical_resource_scenarios_catalog.csv`",
            "- **District-Level Scenario Evaluations (`4 × 36 = 144` rows)**: `data/resource_intelligence/hypothetical_district_scenario_evaluations.csv`",
            "- **Frontend Integration Payload (`6` sections)**: `data/resource_intelligence/frontend_integration_payload.json`",
            "- **Interactive HTML/JS Scenario Calculator**: `reports/resource_intelligence/interactive_resource_demand_calculator.html`",
            "- **Static PNG Figures (`4` charts)**: `reports/resource_intelligence/figures/01_*.png` through `04_*.png`",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def update_model_dataset_registry(
    proj_dir: Path,
    audit_meta: Dict[str, Any],
    catalog_df: pd.DataFrame,
) -> Path:
    """Register Phase 11 Healthcare Resource Demand Intelligence Engine in model_dataset_registry.json."""
    reg_path = proj_dir / "model_dataset_registry.json"
    if reg_path.exists():
        reg = json.loads(reg_path.read_text(encoding="utf-8"))
    else:
        reg = {}

    reg["phase11_healthcare_resource_demand_engine"] = {
        "engine_name": "Healthcare Resource Demand Intelligence & Hypothetical Scenario Calculator Engine",
        "version": ENGINE_VERSION,
        "updated_at": datetime.now().isoformat(),
        "empirical_resource_datasets_in_workspace": audit_meta["empirical_resource_datasets_found"],
        "empirical_resource_forecasting_supported": audit_meta["empirical_resource_forecasting_supported"],
        "operating_mode": "EXPLICIT_NAN_PLUS_HYPOTHETICAL_SCENARIO_CALCULATOR_FALLBACK",
        "mandatory_disclaimer": MANDATORY_RESOURCE_DISCLAIMER,
        "benchmark_scenarios_evaluated": len(catalog_df),
        "artifacts": {
            "completeness_audit_csv": "data/resource_intelligence/resource_data_completeness_audit.csv",
            "district_resource_summary_csv": "data/resource_intelligence/district_resource_summary.csv",
            "historical_epidemiological_trends_csv": "data/resource_intelligence/historical_epidemiological_demand_trends.csv",
            "scenarios_catalog_csv": "data/resource_intelligence/hypothetical_resource_scenarios_catalog.csv",
            "district_scenario_evaluations_csv": "data/resource_intelligence/hypothetical_district_scenario_evaluations.csv",
            "frontend_integration_payload_json": "data/resource_intelligence/frontend_integration_payload.json",
            "interactive_calculator_html": "reports/resource_intelligence/interactive_resource_demand_calculator.html",
            "technical_report_md": "reports/resource_intelligence/resource_demand_intelligence_report.md",
        },
    }
    reg_path.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    return reg_path


def run_resource_intelligence_pipeline(proj_dir: Path = PROJ) -> Dict[str, Any]:
    """Execute the complete Phase 11 Healthcare Resource Demand Intelligence pipeline."""
    data_dir = proj_dir / "data" / "resource_intelligence"
    reports_dir = proj_dir / "reports" / "resource_intelligence"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Audit workspace data availability
    completeness_df, audit_meta = audit_resource_data_availability(proj_dir)

    # 2. Build 36-district summary (explicit NaN for empirical resources + verified upstream caseloads)
    district_summary_df, trend_df = build_district_resource_and_epidemiological_summary(proj_dir)

    # 3. Evaluate benchmark hypothetical scenarios across all 36 districts
    scenarios = build_default_resource_scenarios()
    catalog_df, eval_df = evaluate_scenarios_across_districts(district_summary_df, scenarios)

    # 4. Build frontend integration payload
    frontend_payload = build_frontend_integration_payload(
        completeness_df=completeness_df,
        audit_meta=audit_meta,
        district_summary_df=district_summary_df,
        catalog_df=catalog_df,
        eval_df=eval_df,
    )

    # 5. Save CSVs and JSON to data/resource_intelligence/ and mirror to reports/resource_intelligence/
    csv_files = {
        "resource_data_completeness_audit.csv": completeness_df,
        "district_resource_summary.csv": district_summary_df,
        "historical_epidemiological_demand_trends.csv": trend_df,
        "hypothetical_resource_scenarios_catalog.csv": catalog_df,
        "hypothetical_district_scenario_evaluations.csv": eval_df,
    }
    for fname, df in csv_files.items():
        p_dat = data_dir / fname
        p_rep = reports_dir / fname
        df.to_csv(p_dat, index=False)
        shutil.copy2(p_dat, p_rep)

    payload_dat = data_dir / "frontend_integration_payload.json"
    payload_rep = reports_dir / "frontend_integration_payload.json"
    payload_dat.write_text(json.dumps(frontend_payload, indent=2), encoding="utf-8")
    shutil.copy2(payload_dat, payload_rep)

    # 6. Generate static PNG figures and interactive HTML calculator
    viz_info = generate_all_resource_visualizations(
        completeness_df=completeness_df,
        district_summary_df=district_summary_df,
        catalog_df=catalog_df,
        eval_df=eval_df,
        reports_dir=reports_dir,
        data_dir=data_dir,
    )

    # 7. Write Markdown report and update model_dataset_registry.json
    report_md = write_resource_intelligence_report(
        completeness_df=completeness_df,
        audit_meta=audit_meta,
        district_summary_df=district_summary_df,
        catalog_df=catalog_df,
        eval_df=eval_df,
        report_path=reports_dir / "resource_demand_intelligence_report.md",
    )
    shutil.copy2(report_md, data_dir / "resource_demand_intelligence_report.md")
    update_model_dataset_registry(proj_dir, audit_meta, catalog_df)

    return {
        "audit_meta": audit_meta,
        "districts_count": len(district_summary_df),
        "scenarios_count": len(catalog_df),
        "evaluations_count": len(eval_df),
        "visualizations": viz_info,
        "report_path": str(report_md),
    }


if __name__ == "__main__":
    res = run_resource_intelligence_pipeline()
    print(
        f"[Phase 11 Pipeline Complete] Empirical resource datasets: "
        f"{res['audit_meta']['empirical_resource_datasets_found']} | "
        f"Districts summarized: {res['districts_count']} | "
        f"Hypothetical scenarios: {res['scenarios_count']} ({res['evaluations_count']} district rows)"
    )
