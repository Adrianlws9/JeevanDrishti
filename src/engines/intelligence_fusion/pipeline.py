"""Master Pipeline for Central Predictive Intelligence Fusion Engine (Phase 15).

Executes:
1. Upstream Engine Inspection across all 9 implemented engines (Section 1)
2. Data Alignment, Schema Standardization & Deduplication (Sections 2 & 3)
3. Transparent Rule-Based Fusion across:
   - 513 Weekly Event Records (`511` active outbreak weeks + `2` zero-surveillance districts)
   - 36 Official Maharashtra District Pooled Views (`2023-2025_POOL`)
4. Engine Disagreement Extraction & Traceable `"Why this result?"` Generation (Sections 5, 6 & 7)
5. Static Visualizations (5 PNG figures) + Standalone Interactive HTML Explorer (Section 8)
6. Comprehensive Technical & Governance Report (`reports/fusion/predictive_intelligence_fusion_report.md`).
"""
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.engines.intelligence_fusion.alignment_validator import (
    align_weekly_and_district_engine_outputs,
)
from src.engines.intelligence_fusion.engine_inspector import (
    FUSION_ENGINE_VERSION,
    MANDATORY_FUSION_GOVERNANCE_DISCLAIMER,
    PROJ,
    inspect_and_load_upstream_engines,
)
from src.engines.intelligence_fusion.fusion_engine import (
    fuse_dataframe_records,
)
from src.engines.intelligence_fusion.visualizations import (
    generate_fusion_visualizations,
)


def run_intelligence_fusion_pipeline(proj_dir: Path = PROJ) -> Dict[str, Any]:
    """Run the end-to-end Phase 15 Predictive Intelligence Fusion Engine pipeline."""
    data_dir = proj_dir / "data" / "fused_intelligence"
    report_dir = proj_dir / "reports" / "fusion"
    fig_dir = report_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Inspect and load all 9 upstream engines
    audit_df, loaded_bundle = inspect_and_load_upstream_engines(proj_dir)
    audit_csv = data_dir / "upstream_engines_inspection_audit.csv"
    audit_df.to_csv(audit_csv, index=False)

    # 2. Align and validate weekly & district-pooled outputs
    aligned_weekly_df, aligned_pool_df, align_meta = align_weekly_and_district_engine_outputs(
        loaded_bundle
    )

    # 3. Run rule-based fusion across weekly records (513) and district-pool records (36)
    weekly_nested, weekly_fused_df = fuse_dataframe_records(aligned_weekly_df)
    pool_nested, pool_fused_df = fuse_dataframe_records(aligned_pool_df)

    # Extract explicit engine disagreement records (Section 6)
    disagreements_df = weekly_fused_df[weekly_fused_df["disagreement_flag"] == True].copy()

    # Save tabular CSVs and nested JSON payloads in `data/fused_intelligence/`
    wk_csv = data_dir / "fused_weekly_district_disease_intelligence.csv"
    pool_csv = data_dir / "fused_district_pooled_intelligence_36.csv"
    disagree_csv = data_dir / "engine_disagreement_review_queue.csv"
    json_payload_path = data_dir / "fused_intelligence_common_schema_payload.json"

    weekly_fused_df.to_csv(wk_csv, index=False)
    pool_fused_df.to_csv(pool_csv, index=False)
    disagreements_df.to_csv(disagree_csv, index=False)

    full_payload = {
        "fusion_engine_version": FUSION_ENGINE_VERSION,
        "generated_timestamp": datetime.now().isoformat(),
        "governance_disclaimer": MANDATORY_FUSION_GOVERNANCE_DISCLAIMER,
        "upstream_engines_audit": audit_df.to_dict(orient="records"),
        "alignment_and_validation_metadata": align_meta,
        "weekly_signal_status_counts": (
            weekly_fused_df["overall_signal_status"].value_counts().to_dict()
        ),
        "district_pool_signal_status_counts": (
            pool_fused_df["overall_signal_status"].value_counts().to_dict()
        ),
        "district_pooled_intelligence_36": pool_nested,
        "weekly_fused_sample_top50": weekly_nested[:50],
    }
    json_payload_path.write_text(json.dumps(full_payload, indent=2), encoding="utf-8")

    # 4. Generate visualizations & interactive HTML explorer
    html_path = report_dir / "interactive_predictive_intelligence_fusion_explorer.html"
    fig_paths = generate_fusion_visualizations(
        audit_df=audit_df,
        weekly_fused_df=weekly_fused_df,
        district_pool_fused_df=pool_fused_df,
        geo_pool_df=loaded_bundle["geo_df"][
            loaded_bundle["geo_df"]["reporting_period"] == "2023-2025_POOL"
        ],
        fig_dir=fig_dir,
        html_path=html_path,
    )

    # 5. Write comprehensive Markdown report (`reports/fusion/predictive_intelligence_fusion_report.md`)
    report_md = report_dir / "predictive_intelligence_fusion_report.md"
    _write_fusion_markdown_report(
        report_md=report_md,
        audit_df=audit_df,
        align_meta=align_meta,
        weekly_fused_df=weekly_fused_df,
        pool_fused_df=pool_fused_df,
        disagreements_df=disagreements_df,
        fig_paths=fig_paths,
    )

    result_meta = {
        "fusion_engine_version": FUSION_ENGINE_VERSION,
        "upstream_engines_integrated": int(audit_df["implemented_and_verified"].sum()),
        "weekly_fused_records": len(weekly_fused_df),
        "district_pooled_records": len(pool_fused_df),
        "disagreement_records_preserved": len(disagreements_df),
        "weekly_status_counts": weekly_fused_df["overall_signal_status"].value_counts().to_dict(),
        "pool_status_counts": pool_fused_df["overall_signal_status"].value_counts().to_dict(),
        "artifacts": {
            "upstream_audit_csv": str(audit_csv),
            "weekly_fused_csv": str(wk_csv),
            "district_pool_csv": str(pool_csv),
            "disagreement_queue_csv": str(disagree_csv),
            "common_schema_json": str(json_payload_path),
            "interactive_html": str(html_path),
            "markdown_report": str(report_md),
            **fig_paths,
        },
    }
    print(
        f"[Phase 15 Fusion Pipeline Complete] Engines integrated: {result_meta['upstream_engines_integrated']} | "
        f"Weekly fused records: {len(weekly_fused_df)} | "
        f"District pooled records: {len(pool_fused_df)} | "
        f"Disagreements preserved: {len(disagreements_df)}"
    )
    return result_meta


def _write_fusion_markdown_report(
    report_md: Path,
    audit_df: pd.DataFrame,
    align_meta: Dict[str, Any],
    weekly_fused_df: pd.DataFrame,
    pool_fused_df: pd.DataFrame,
    disagreements_df: pd.DataFrame,
    fig_paths: Dict[str, str],
) -> None:
    """Write the comprehensive Phase 15 Predictive Intelligence Fusion Engine report."""
    wk_status = weekly_fused_df["overall_signal_status"].value_counts().to_dict()
    wk_prio = weekly_fused_df["recommended_review_priority"].value_counts().to_dict()
    pool_status = pool_fused_df["overall_signal_status"].value_counts().to_dict()

    lines: List[str] = [
        "# JEEVAN-DRISHTI — Phase 15: Central Predictive Intelligence Fusion Engine Report",
        "",
        f"**Engine Version**: `{FUSION_ENGINE_VERSION}`  ",
        f"**Generated Timestamp**: `{datetime.now().isoformat()}`  ",
        "**Fusion Architecture**: `Transparent Tiered Evidence-Based Rule Hierarchy with Explicit Disagreement Preservation`",
        "",
        "> [!CAUTION]",
        f"> **GOVERNANCE SAFEGUARD**: {MANDATORY_FUSION_GOVERNANCE_DISCLAIMER}",
        "",
        "---",
        "",
        "## 1. Upstream Analytical Engine Inspection (Section 1)",
        "",
        "All **9 upstream analytical engines** (Phases 5–13) were inspected and verified before integration:",
        "",
        "| Engine ID | Engine Name | Version | Verified Rows/Panels | Temporal Grain | Reliability Tier | Role in Fusion Engine |",
        "| :--- | :--- | :--- | ---: | :--- | :--- | :--- |",
    ]

    for _, r in audit_df.iterrows():
        lines.append(
            f"| `{r['engine_id']}` | {r['engine_name']} | `{r['model_version']}` | "
            f"`{r['artifact_row_or_panel_count']}` | `{r['temporal_grain']}` | "
            f"`{r['reliability_tier']}` | {r['fusion_role']} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Common Output Schema, Alignment & Validation Gate (Sections 2 & 3)",
            "",
            "1. **Standardized Common Output Schema**: Every fused record contains `district_id`, `district_name`, `disease`, `reporting_period`, `source_observation_date`, `risk_classification_output`, `forecast_output`, `anomaly_output`, `climate_indicators`, `geospatial_indicators`, `resource_indicators`, `severity_indicators`, `model_versions`, `data_quality_status`, and `generated_timestamp`.",
            "2. **Strict Temporal Grain Separation**:",
            "   - **Weekly Event Grain (`513` records)**: Matches `(district_id, disease, reporting_period=YYYY-Www)` across all `511` deduplicated `2023–2025` outbreak weeks plus `2` explicit `Insufficient data` records for the zero-surveillance districts (`Mumbai City`, `Ratnagiri`).",
            "   - **36-District Pooled Grain (`36` records)**: Combines `2023-2025_POOL` district summaries across all `36` official Maharashtra districts.",
            "   - **Cross-Sectional Context Tagging**: Pooled indicators (Phase 9 Getis-Ord $G_i^*$, Phase 10 graph degree, Phase 11 resource audit) are explicitly tagged as `2023-2025_POOL_CROSS_SECTIONAL_CONTEXT` when attached to weekly records so incompatible temporal periods are never silently conflated.",
            "3. **Validation Checks Enforced**:",
            "   - District alias standardization (`Aurangabad -> Chhatrapati Sambhajinagar`, `Osmanabad -> Dharashiv`) and rejection of non-Maharashtra districts.",
            "   - Incompatible reporting period rejection (`check_reporting_period_compatibility`).",
            "   - Model version & timestamp freshness verification (`evaluate_model_and_data_freshness`).",
            "   - Pre-fusion deduplication across all upstream tables.",
            "",
            "---",
            "",
            "## 3. Transparent Rule-Based Fusion Architecture & Reliability Hierarchy (Sections 4 & 5)",
            "",
            "Because a calibrated probability (`0.28`), a case count forecast (`45.2`), a composite anomaly score (`0.85`), and a spatial $G_i^*$ $z$-score (`+2.41`) represent **incomparable mathematical quantities**, this engine **never averages model scores**. Instead, it evaluates a 3-tier evidence hierarchy:",
            "",
            "- **Tier 1 — Primary Empirical Weekly Pillars (`4` Pillars)**:",
            "  1. **Risk Classification Pillar (`R_flag`)**: Elevated if `calibrated_risk_probability >= 0.265` (F1-optimal threshold) and disease tier is not `UNVALIDATED_LIMITED_SAMPLE` ($n < 5$ training weeks).",
            "  2. **Time-Series Forecast Pillar (`F_flag`)**: Elevated if `forecast_7d_cases >= 1.15 * observed_cases` (with $\\ge 15$ cases), or `forecast_14d_cases >= 1.10 * forecast_7d_cases` ($\\ge 20$ cases), or `forecast_7d_cases >= 45.0`.",
            "  3. **Anomaly Detection Pillar (`A_flag`)**: Elevated if `is_anomaly == True` or `composite_anomaly_score >= 0.55`.",
            "  4. **Verified Severity / Mortality Pillar (`M_flag`)**: Elevated if `verified_outbreak_deaths >= 1`.",
            "- **Tier 2 — Contextual Environmental & Spatial Pillars (`2` Pillars)**:",
            "  5. **Climate Conduciveness Context (`C_context`)**: Supportive if `risk_shap_climate_log_odds >= +0.05` or `prev_4w_rainfall_accum_mm >= 50.0` (correlational context only).",
            "  6. **Geospatial Hotspot Context (`G_context`)**: Supportive if Getis-Ord $G_i^*$ is statistically significant ($p < 0.05$).",
            "- **Tier 3 — Capacity & Connectivity Caveats**:",
            "  7. **Resource Demand (`Res_status`)**: Explicitly reports `Insufficient Data (No Hospital/Resource Dataset in Workspace)` (`NaN` beds/ICU) and points to the Phase 11/14 What-If Calculators.",
            "",
            "### Canonical Signal Status Distribution (`513` Weekly Records vs. `36` Pooled District Records)",
            "",
            "| Canonical Signal Status (Section 5) | Weekly Records (`N = 513`) | Pooled District Records (`N = 36`) | Rule Definition |",
            "| :--- | ---: | ---: | :--- |",
            f"| `Multiple indicators aligned` | `{wk_status.get('Multiple indicators aligned', 0)}` | `{pool_status.get('Multiple indicators aligned', 0)}` | $\\ge 2$ Tier-1 primary pillars simultaneously elevated |",
            f"| `Conflicting indicators` | `{wk_status.get('Conflicting indicators', 0)}` | `{pool_status.get('Conflicting indicators', 0)}` | 1 primary pillar strongly elevated while other primary pillars contradict it |",
            f"| `Elevated signal` | `{wk_status.get('Elevated signal', 0)}` | `{pool_status.get('Elevated signal', 0)}` | 1 primary or spatial hotspot indicator elevated without sharp contradiction |",
            f"| `No unusual signal detected` | `{wk_status.get('No unusual signal detected', 0)}` | `{pool_status.get('No unusual signal detected', 0)}` | All 4 primary pillars within normal/baseline bounds |",
            f"| `Insufficient data` | `{wk_status.get('Insufficient data', 0)}` | `{pool_status.get('Insufficient data', 0)}` | Zero surveillance records in `2023–2025` (`Mumbai City`, `Ratnagiri`) or $<2$ primary pillars available |",
            "",
            "### Documented Epidemiological Review Priority Breakdown (`513` Weekly Records)",
            "",
            "| Review Priority Tier | Weekly Count | Assignment Rule |",
            "| :--- | ---: | :--- |",
            f"| `PRIORITY_1_IMMEDIATE_EPIDEMIOLOGICAL_REVIEW` | `{wk_prio.get('PRIORITY_1_IMMEDIATE_EPIDEMIOLOGICAL_REVIEW', 0)}` | `Multiple indicators aligned` + (`verified_deaths >= 1` or $\\ge 3$ aligned pillars or `HIGH_PRIORITY` anomaly) |",
            f"| `PRIORITY_2_TARGETED_ANALYST_REVIEW` | `{wk_prio.get('PRIORITY_2_TARGETED_ANALYST_REVIEW', 0)}` | `Conflicting indicators` (analyst adjudication required) or `Multiple indicators aligned` (0 deaths) or `Elevated signal` with contextual support |",
            f"| `PRIORITY_3_ROUTINE_ENHANCED_MONITORING` | `{wk_prio.get('PRIORITY_3_ROUTINE_ENHANCED_MONITORING', 0)}` | Single `Elevated signal` with 0 deaths |",
            f"| `PRIORITY_4_STANDARD_SURVEILLANCE` | `{wk_prio.get('PRIORITY_4_STANDARD_SURVEILLANCE', 0)}` | `No unusual signal detected` |",
            f"| `PRIORITY_DATA_GAP_AUDIT` | `{wk_prio.get('PRIORITY_DATA_GAP_AUDIT', 0)}` | `Insufficient data` (`Mumbai City`, `Ratnagiri`) |",
            "",
            "---",
            "",
            "## 4. Engine Disagreement Handling & Sample Conflict Queue (Section 6)",
            "",
            f"Across the `513` weekly records, **`{len(disagreements_df)}` records** exhibited explicit inter-engine disagreements (`ANOMALY_VS_PREDICTIVE_DISAGREEMENT`, `PREDICTIVE_VS_ANOMALY_DISAGREEMENT`, or `FORECAST_VS_RISK_DISAGREEMENT`). Every disagreeing record is preserved intact in `data/fused_intelligence/engine_disagreement_review_queue.csv`:",
            "",
            "| District | Disease | Period | Status | Disagreement Type | Risk Prob | 7d Forecast | Anomaly Score | Deaths | Preserved Conflict Summary |",
            "| :--- | :--- | :---: | :--- | :--- | ---: | ---: | ---: | ---: | :--- |",
        ]
    )

    for _, r in disagreements_df.head(10).iterrows():
        c_short = str(r["conflicting_evidence"])[:135] + "..."
        lines.append(
            f"| {r['district_name']} | {r['disease']} | `{r['reporting_period']}` | "
            f"`{r['overall_signal_status']}` | `{r['disagreement_type']}` | "
            f"`{r['risk_calibrated_probability']:.3f}` | `{r['forecast_7d_cases']:.1f}` | "
            f"`{r['anomaly_composite_score']:.3f}` | `{int(r['verified_outbreak_deaths'] or 0)}` | {c_short} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 5. Generated Visual & Interactive Artifacts (Section 8)",
            "",
            f"- **Figure 1 (Upstream Engines & Reliability Hierarchy)**: `{fig_paths['engines_hierarchy_png']}`",
            f"- **Figure 2 (Weekly Signal Status & Review Priority Distribution)**: `{fig_paths['signal_status_distribution_png']}`",
            f"- **Figure 3 (Maharashtra 36-District Fused Intelligence Map & Matrix)**: `{fig_paths['district_map_matrix_png']}`",
            f"- **Figure 4 (Engine Disagreement & Conflict Quadrants)**: `{fig_paths['disagreement_analysis_png']}`",
            f"- **Figure 5 (Traceable 'Why This Result?' Evidence Cards)**: `{fig_paths['why_this_result_cards_png']}`",
            f"- **Interactive HTML Unified Fusion Explorer**: `{fig_paths['interactive_fusion_html']}`",
            "",
            "---",
            "",
            "## 6. Uncertainty, Limitations & Governance Checklist (Section 10)",
            "",
            "- **No Automated Outbreak Declarations**: Every record enforces `automated_outbreak_declaration = False`.",
            "- **Zero Blind Score Averaging**: Probabilities, case counts, anomaly Z-scores, and spatial $G_i^*$ statistics are never averaged.",
            "- **Preserved Disagreements**: Conflicting engine signals are explicitly surfaced in `conflicting_evidence` and routed to `PRIORITY_2_TARGETED_ANALYST_REVIEW`.",
            "- **Documented Data Gaps**: Hospital bed capacities, ICU inventories, and clinical severity counts remain explicitly `NaN`/`None` across all fused records.",
        ]
    )

    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_intelligence_fusion_pipeline()
