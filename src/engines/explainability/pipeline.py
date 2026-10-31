"""End-to-End Pipeline for Phase 13 Centralized Explainable AI (XAI) Engine."""
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List

import pandas as pd

from src.engines.explainability.frontend_panel import build_why_district_flagged_panels
from src.engines.explainability.global_explainer import compute_all_global_explanations
from src.engines.explainability.local_explainer import (
    compute_local_anomaly_explanations,
    compute_local_forecast_explanations,
    compute_local_risk_explanations,
)
from src.engines.explainability.model_inspector import (
    ENGINE_VERSION,
    MANDATORY_XAI_NON_CAUSAL_DISCLAIMER,
    PROJ,
    inspect_existing_models_and_artifacts,
)
from src.engines.explainability.visualizations import generate_all_explainability_visualizations


def write_explainability_report(
    inventory_df: pd.DataFrame,
    global_df: pd.DataFrame,
    global_summary: Dict[str, Any],
    local_risk_df: pd.DataFrame,
    local_fc_df: pd.DataFrame,
    local_anom_df: pd.DataFrame,
    panels_df: pd.DataFrame,
    report_path: Path,
) -> Path:
    """Write the comprehensive Phase 13 Markdown technical report."""
    lines: List[str] = [
        "# JEEVAN-DRISHTI Phase 13: Centralized Explainable AI (XAI) Engine Report",
        "",
        f"> **Engine Version**: `{ENGINE_VERSION}` | **Audit Date**: `{datetime.now().strftime('%Y-%m-%d')}`",
        f"> **Mandatory Scientific Disclaimer**: {MANDATORY_XAI_NON_CAUSAL_DISCLAIMER}",
        "",
        "---",
        "",
        "## 1. Existing Model Inventory & Explainability Suitability Matrix",
        "",
        "Every model artifact across Phases 5–9 was inspected to match each estimator with its mathematically valid explainability method (never forcing SHAP onto statistical baselines or spatial Z-scores):",
        "",
        "| Phase | Domain | Model Name | Version | Features | Primary | SHAP Suitable | Explainability Methods Applied |",
        "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
    ]
    for _, r in inventory_df.iterrows():
        lines.append(
            f"| **{r['phase']}** | {r['engine_domain']} | `{r['model_name']}` | `{r['model_version']}` | "
            f"{int(r['n_transformed_features'])} | `{bool(r['is_primary_production_model'])}` | "
            f"`{bool(r['shap_suitable'])}` | {r['explainability_methods_applied']} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Global Explanations Across Supported Models",
            "",
            "### A. Phase 5 Outbreak Risk Classifier (`phase5-v1.0.0`) — Top 10 Global Features",
            "",
            "| Rank | Feature Name | Domain | XGBoost Mean `|TreeSHAP|` | Primary HGB Permutation `ΔPR-AUC` | RF Gini | LogReg Standardized $\\beta_j$ | Direction of Influence |",
            "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :--- |",
        ]
    )
    for r in global_summary["phase5_risk_classification"]["top_10_features"]:
        lines.append(
            f"| {int(r['global_importance_rank'])} | `{r['feature_name']}` | `{r['feature_domain']}` | "
            f"`{r['xgb_mean_abs_treeshap']:.4f}` | `{r['hgb_primary_permutation_importance_pr_auc']:.4f}` | "
            f"`{r['rf_gini_importance']:.4f}` | `{r['logreg_standardized_coef_beta']:+.4f}` | `{r['xgb_treeshap_direction']}` |"
        )

    lines.extend(
        [
            "",
            "### B. Phase 6 Multi-Horizon Disease Forecasting (`phase6-v1.0.0`) — Domain Attribution Share (`%` of Mean `|TreeSHAP|`)",
            "",
            "| Feature Domain | `7d` (`1w` Ahead) Share (`%`) | `14d` (`2w` Ahead) Share (`%`) | `30d` (`4w` Ahead) Share (`%`) |",
            "| :--- | :---: | :---: | :---: |",
        ]
    )
    h_dict = global_summary["phase6_disease_forecasting"]["horizons"]
    all_doms = sorted(
        set(h_dict["7d"]["domain_treeshap_share_pct"].keys())
        | set(h_dict["14d"]["domain_treeshap_share_pct"].keys())
        | set(h_dict["30d"]["domain_treeshap_share_pct"].keys())
    )
    for dm in all_doms:
        s7 = h_dict["7d"]["domain_treeshap_share_pct"].get(dm, 0.0)
        s14 = h_dict["14d"]["domain_treeshap_share_pct"].get(dm, 0.0)
        s30 = h_dict["30d"]["domain_treeshap_share_pct"].get(dm, 0.0)
        lines.append(f"| `{dm}` | `{s7:.2f}%` | `{s14:.2f}%` | `{s30:.2f}%` |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Local Risk, Forecast & Anomaly Explanations",
            "",
            "### A. Top 10 Highest Calibrated Outbreak Risk Predictions (`local_risk_explanations.csv`)",
            "",
            "| Prediction ID | District | Disease | Period | Calibrated Risk (`Tier`) | Historical Disease SHAP | Case Growth SHAP | Climate SHAP | Seasonal SHAP | Missing Inputs |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    top_risk = local_risk_df.sort_values("model_estimated_risk_prob", ascending=False).head(10)
    for _, r in top_risk.iterrows():
        lines.append(
            f"| `{r['prediction_or_anomaly_id']}` | **{r['district']}** | {r['disease']} | `{r['reporting_period']}` | "
            f"**{r['model_estimated_risk_prob']:.3f}** (`{r['risk_category']}`) | "
            f"`{r['historical_disease_contribution']:+.3f}` | `{r['case_growth_contribution']:+.3f}` | "
            f"`{r['climate_contribution']:+.3f}` | `{r['seasonal_contribution']:+.3f}` | "
            f"`{int(r['missing_features_count'])}` |"
        )

    lines.extend(
        [
            "",
            "### B. Top 10 Flagged Anomaly Evidence Explanations (`local_anomaly_explanations.csv`)",
            "",
            "| Anomaly ID | District | Disease | Period | Observed Cases | Composite Score (`Tier`) | Seasonal $Z$ | Rolling $Z$ | Robust MAD $Z$ | Isolation Forest |",
            "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    top_anom = (
        local_anom_df[local_anom_df["is_anomaly"] == True]
        .sort_values("composite_anomaly_score", ascending=False)
        .head(10)
    )
    for _, r in top_anom.iterrows():
        lines.append(
            f"| `{r['prediction_or_anomaly_id']}` | **{r['district']}** | {r['disease']} | `{r['reporting_period']}` | "
            f"**{int(r['observed_value'])}** | `{r['composite_anomaly_score']:.3f}` (`{r['anomaly_severity_tier']}`) | "
            f"`{r['deviation_seasonal_z']:+.2f}` | `{r['deviation_rolling_z']:+.2f}` | "
            f"`{r['deviation_robust_mad_z']:+.2f}` | `{r['isolation_forest_score']:.3f}` |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Frontend Integration: `\"Why was this district flagged?\"` Panel (`36` Districts)",
            "",
            "| District ID | District | Flag Priority Tier | Peak Risk Prob (`Category`) | Flagged Anomaly Weeks | Getis-Ord $G_i^*$ $Z$-Score (`Class`) | `7d` Forecast Mean Cases |",
            "| :---: | :--- | :--- | :---: | :---: | :---: | :---: |",
        ]
    )
    for _, r in panels_df.head(18).iterrows():
        pr = f"{r['peak_risk_probability']:.3f} ({r['peak_risk_category']})" if pd.notna(r["peak_risk_probability"]) else "Insuff. Data"
        gz = f"{r['gi_star_zscore']:+.2f} ({r['geospatial_hotspot_class']})" if pd.notna(r["gi_star_zscore"]) else "Insuff. Data"
        f7 = f"{r['forecast_7d_mean_cases']:.1f}" if pd.notna(r["forecast_7d_mean_cases"]) else "Insuff. Data"
        lines.append(
            f"| `{r['district_id']}` | **{r['district']}** | `{r['flag_priority_tier']}` | "
            f"`{pr}` | {int(r['flagged_anomaly_weeks_count'])} | `{gz}` | {f7} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 5. Generated Datasets, Figures & Interactive Artifacts",
            "",
            "- **Model Explainability Inventory**: `data/explanations/model_explainability_inventory.csv`",
            "- **Global Feature Explanations**: `data/explanations/global_feature_explanations.csv` & `global_model_explanations_summary.json`",
            "- **Local Risk Explanations (`511` rows)**: `data/explanations/local_risk_explanations.csv`",
            "- **Local Forecast Explanations (`7d`, `14d`, `30d`)**: `data/explanations/local_forecast_explanations.csv`",
            "- **Local Anomaly Explanations (`511` rows, `116` flagged)**: `data/explanations/local_anomaly_explanations.csv`",
            "- **Frontend Panel (`\"Why was this district flagged?\"` across `36` districts)**: `data/explanations/why_district_flagged_panels.csv` & `why_district_flagged_panels.json`",
            "- **Interactive HTML Analyst Explorer**: `reports/explainability/interactive_why_district_flagged_explorer.html`",
            "- **Static PNG Figures (`5` charts)**: `reports/explainability/figures/01_*.png` through `05_*.png`",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def update_model_dataset_registry(
    proj_dir: Path,
    inventory_df: pd.DataFrame,
    local_risk_df: pd.DataFrame,
    local_fc_df: pd.DataFrame,
    local_anom_df: pd.DataFrame,
    panels_df: pd.DataFrame,
) -> Path:
    """Register Phase 13 Centralized Explainable AI Engine in model_dataset_registry.json."""
    reg_path = proj_dir / "model_dataset_registry.json"
    if reg_path.exists():
        reg = json.loads(reg_path.read_text(encoding="utf-8"))
    else:
        reg = {}

    reg["phase13_explainable_ai_engine"] = {
        "engine_name": "Centralized Explainable AI (XAI) Engine",
        "version": ENGINE_VERSION,
        "updated_at": datetime.now().isoformat(),
        "models_inspected_count": len(inventory_df),
        "local_risk_explanations_count": len(local_risk_df),
        "local_forecast_explanations_count": len(local_fc_df),
        "local_anomaly_explanations_count": len(local_anom_df),
        "district_flagged_panels_count": len(panels_df),
        "mandatory_disclaimer": MANDATORY_XAI_NON_CAUSAL_DISCLAIMER,
        "artifacts": {
            "model_inventory_csv": "data/explanations/model_explainability_inventory.csv",
            "global_explanations_csv": "data/explanations/global_feature_explanations.csv",
            "global_summary_json": "data/explanations/global_model_explanations_summary.json",
            "local_risk_explanations_csv": "data/explanations/local_risk_explanations.csv",
            "local_forecast_explanations_csv": "data/explanations/local_forecast_explanations.csv",
            "local_anomaly_explanations_csv": "data/explanations/local_anomaly_explanations.csv",
            "why_district_flagged_panels_csv": "data/explanations/why_district_flagged_panels.csv",
            "why_district_flagged_panels_json": "data/explanations/why_district_flagged_panels.json",
            "interactive_explorer_html": "reports/explainability/interactive_why_district_flagged_explorer.html",
            "technical_report_md": "reports/explainability/explainability_engine_report.md",
        },
    }
    reg_path.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    return reg_path


def run_explainability_pipeline(proj_dir: Path = PROJ) -> Dict[str, Any]:
    """Execute the complete Phase 13 Explainable AI pipeline."""
    data_dir = proj_dir / "data" / "explanations"
    reports_dir = proj_dir / "reports" / "explainability"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Inspect existing models
    inventory_df, bundles_dict = inspect_existing_models_and_artifacts(proj_dir)

    # 2. Compute global explanations
    global_df, global_summary = compute_all_global_explanations(proj_dir, bundles_dict)

    # 3. Compute local explanations for Risk, Forecasts, and Anomalies
    local_risk_df = compute_local_risk_explanations(proj_dir, bundles_dict)
    local_fc_df = compute_local_forecast_explanations(proj_dir, bundles_dict)
    local_anom_df = compute_local_anomaly_explanations(proj_dir, bundles_dict)

    # 4. Build "Why was this district flagged?" frontend panels (36 districts)
    panels_df, panel_payload = build_why_district_flagged_panels(
        proj_dir=proj_dir,
        local_risk_df=local_risk_df,
        local_fc_df=local_fc_df,
        local_anom_df=local_anom_df,
    )

    # 5. Save CSVs and JSONs to data/explanations/ and mirror to reports/explainability/
    csv_map = {
        "model_explainability_inventory.csv": inventory_df,
        "global_feature_explanations.csv": global_df,
        "local_risk_explanations.csv": local_risk_df,
        "local_forecast_explanations.csv": local_fc_df,
        "local_anomaly_explanations.csv": local_anom_df,
        "why_district_flagged_panels.csv": panels_df,
    }
    for fname, df in csv_map.items():
        p_dat = data_dir / fname
        p_rep = reports_dir / fname
        df.to_csv(p_dat, index=False)
        shutil.copy2(p_dat, p_rep)

    for jname, jobj in [
        ("global_model_explanations_summary.json", global_summary),
        ("why_district_flagged_panels.json", panel_payload),
    ]:
        p_dat = data_dir / jname
        p_rep = reports_dir / jname
        p_dat.write_text(json.dumps(jobj, indent=2), encoding="utf-8")
        shutil.copy2(p_dat, p_rep)

    # 6. Generate 5 static PNG figures + interactive HTML explorer
    viz_info = generate_all_explainability_visualizations(
        global_df=global_df,
        global_summary=global_summary,
        local_risk_df=local_risk_df,
        local_anom_df=local_anom_df,
        panels_df=panels_df,
        panel_payload=panel_payload,
        reports_dir=reports_dir,
        data_dir=data_dir,
    )

    # 7. Write Markdown report and update model_dataset_registry.json
    report_md = write_explainability_report(
        inventory_df=inventory_df,
        global_df=global_df,
        global_summary=global_summary,
        local_risk_df=local_risk_df,
        local_fc_df=local_fc_df,
        local_anom_df=local_anom_df,
        panels_df=panels_df,
        report_path=reports_dir / "explainability_engine_report.md",
    )
    shutil.copy2(report_md, data_dir / "explainability_engine_report.md")
    update_model_dataset_registry(
        proj_dir, inventory_df, local_risk_df, local_fc_df, local_anom_df, panels_df
    )

    return {
        "models_inspected": len(inventory_df),
        "global_rows": len(global_df),
        "local_risk_rows": len(local_risk_df),
        "local_forecast_rows": len(local_fc_df),
        "local_anomaly_rows": len(local_anom_df),
        "district_panels": len(panels_df),
        "visualizations": viz_info,
        "report_path": str(report_md),
    }


if __name__ == "__main__":
    res = run_explainability_pipeline()
    print(
        f"[Phase 13 Pipeline Complete] Models inspected: {res['models_inspected']} | "
        f"Global rows: {res['global_rows']} | Local risk: {res['local_risk_rows']} | "
        f"Local forecasts: {res['local_forecast_rows']} | Local anomalies: {res['local_anomaly_rows']} | "
        f"District panels: {res['district_panels']}"
    )
