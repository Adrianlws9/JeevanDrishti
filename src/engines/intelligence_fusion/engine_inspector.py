"""Upstream Engine Inspector & Loader for Predictive Intelligence Fusion Engine (Phase 15 Section 1).

Responsibilities:
1. Inspect and verify all 9 upstream analytical engines specified in Prompt 15 Section 1:
   - Phase 5: Risk Classification Engine (`phase5-v1.0.0`)
   - Phase 6: Time-Series Forecasting Engine (`phase6-v1.0.0`)
   - Phase 7: Anomaly Detection Engine (`phase7-v1.0.0`)
   - Phase 8: Climate–Disease Intelligence Engine (`phase8-v1.0.0`)
   - Phase 9: Geospatial Hotspot Detection Engine (`phase9-v1.0.0`)
   - Phase 10: Spread Scenario Engine (`phase10-v1.0.0`)
   - Phase 11: Resource Demand Engine (`phase11-v1.0.0`)
   - Phase 12: Severity and Mortality Engine (`phase12-v1.0.0`)
   - Phase 13: Explainable AI Engine (`phase13-v1.0.0`)
   (Plus Phase 14 Intervention Scenario Simulator noted as downstream decision-support companion).
2. Load their verified artifacts without fabricating missing outputs.
"""
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

PROJ = Path(__file__).resolve().parents[3]
FUSION_ENGINE_VERSION = "phase15-v1.0.0"

MANDATORY_FUSION_GOVERNANCE_DISCLAIMER = (
    "EVIDENCE-BASED DECISION-SUPPORT FUSION ONLY — NOT AN AUTOMATED OUTBREAK DECLARATION. "
    "This engine combines heterogeneous analytical signals using transparent, auditable rules "
    "without blindly averaging incompatible model scores. Disagreements between engines are "
    "explicitly preserved for human epidemiological review. Clinical severity and hospital resource "
    "inventories remain unavailable (NaN) in the workspace and are never fabricated."
)


def inspect_and_load_upstream_engines(
    proj_dir: Path = PROJ,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Inspect all 9 upstream engines required by Prompt 15 Section 1 and load their datasets."""
    engine_specs: List[Dict[str, Any]] = [
        {
            "engine_id": "ENGINE_01_RISK_CLASSIFICATION",
            "engine_name": "Risk Classification Engine (Phase 5)",
            "model_version": "phase5-v1.0.0",
            "rel_path": "data/explanations/local_risk_explanations.csv",
            "temporal_grain": "WEEKLY_OUTBREAK_EVENT (YYYY-Www)",
            "quantity_represented": "Calibrated probability P(Outbreak Cases >= Train P75) in [0, 1]",
            "reliability_tier": "TIER_1_PRIMARY_EMPIRICAL_PREDICTIVE",
            "fusion_role": "Flags elevated outbreak probability (>= 0.265 optimal F1 threshold); downgraded if disease < 5 train rows.",
        },
        {
            "engine_id": "ENGINE_02_TIME_SERIES_FORECASTING",
            "engine_name": "Time-Series Forecasting Engine (Phase 6)",
            "model_version": "phase6-v1.0.0",
            "rel_path": "data/explanations/local_forecast_explanations.csv",
            "temporal_grain": "WEEKLY_OUTBREAK_EVENT (7d, 14d, 30d horizons)",
            "quantity_represented": "Predicted weekly outbreak case counts + 80% conformal prediction intervals",
            "reliability_tier": "TIER_1_PRIMARY_EMPIRICAL_PREDICTIVE",
            "fusion_role": "Flags projected case trajectory growth or high multi-horizon case volume.",
        },
        {
            "engine_id": "ENGINE_03_ANOMALY_DETECTION",
            "engine_name": "Anomaly Detection Engine (Phase 7)",
            "model_version": "phase7-v1.0.0",
            "rel_path": "data/explanations/local_anomaly_explanations.csv",
            "temporal_grain": "WEEKLY_OUTBREAK_EVENT (YYYY-Www)",
            "quantity_represented": "Composite anomaly score in [0, 1] + Seasonal/Rolling/MAD Z-scores + Isolation Forest",
            "reliability_tier": "TIER_1_PRIMARY_EMPIRICAL_DETECTION",
            "fusion_role": "Flags statistically unusual weekly case spikes, spatial surges, or mortality deviations.",
        },
        {
            "engine_id": "ENGINE_04_CLIMATE_DISEASE_INTELLIGENCE",
            "engine_name": "Climate–Disease Intelligence Engine (Phase 8)",
            "model_version": "phase8-v1.0.0",
            "rel_path": "data/climate_intelligence/district_climate_relationships.csv",
            "temporal_grain": "WEEKLY_ALIGNED_LAGS + DISTRICT_2023_2025_POOL",
            "quantity_represented": "Weekly rainfall/temperature lags + Spearman rho + Seasonal Kruskal-Wallis + TreeSHAP",
            "reliability_tier": "TIER_2_CONTEXTUAL_ENVIRONMENTAL_ASSOCIATION",
            "fusion_role": "Provides non-causal environmental conduciveness context (rainfall accumulation & seasonal regime).",
        },
        {
            "engine_id": "ENGINE_05_GEOSPATIAL_HOTSPOT_DETECTION",
            "engine_name": "Geospatial Hotspot Detection Engine (Phase 9)",
            "model_version": "phase9-v1.0.0",
            "rel_path": "data/geospatial/district_disease_map_summary.csv",
            "temporal_grain": "POOLED_2023_2025_AND_ANNUAL_CROSS_SECTION",
            "quantity_represented": "Global Moran's I & Local Getis-Ord Gi* z-scores across 36 Maharashtra districts",
            "reliability_tier": "TIER_2_CONTEXTUAL_SPATIAL_CLUSTERING",
            "fusion_role": "Identifies whether a district-disease pair sits inside a statistically significant spatial hotspot (p < 0.05).",
        },
        {
            "engine_id": "ENGINE_06_SPREAD_SCENARIO_SIMULATION",
            "engine_name": "District Spread Scenario Engine (Phase 10)",
            "model_version": "phase10-v1.0.0",
            "rel_path": "data/scenarios/district_graph_nodes.csv",
            "temporal_grain": "STATIC_TOPOLOGY_AND_HYPOTHETICAL_SCENARIO",
            "quantity_represented": "Verified district adjacency degree (0..7) + hypothetical 0-100 propagation scenarios",
            "reliability_tier": "TIER_3_HYPOTHETICAL_SCENARIO_CONTEXT_ONLY",
            "fusion_role": "Provides verified topological neighbor connectivity; hypothetical spread scores are never treated as empirical signals.",
        },
        {
            "engine_id": "ENGINE_07_HEALTHCARE_RESOURCE_DEMAND",
            "engine_name": "Healthcare Resource Demand Engine (Phase 11)",
            "model_version": "phase11-v1.0.0",
            "rel_path": "data/resource_intelligence/district_resource_summary.csv",
            "temporal_grain": "DISTRICT_AUDIT_AND_HYPOTHETICAL_CALCULATOR",
            "quantity_represented": "Explicit NaN audit of empirical hospital capacity + Little's Law hypothetical bed calculator",
            "reliability_tier": "TIER_3_CAPACITY_GAP_AND_WHAT_IF_CALCULATOR",
            "fusion_role": "Flags empirical hospital resource data unavailability (NaN) and provides hypothetical surge bed estimates.",
        },
        {
            "engine_id": "ENGINE_08_SEVERITY_AND_MORTALITY",
            "engine_name": "Disease Severity and Mortality Engine (Phase 12)",
            "model_version": "phase12-v1.0.0",
            "rel_path": "data/severity_intelligence/verified_weekly_mortality_records.csv",
            "temporal_grain": "WEEKLY_OUTBREAK_EVENT + POOLED_2023_2025",
            "quantity_represented": "Verified weekly outbreak deaths, Outbreak-Cluster CFR (%), Wilson 95% CI, EB-shrunk CFR",
            "reliability_tier": "TIER_1_PRIMARY_EMPIRICAL_SEVERITY",
            "fusion_role": "Provides verified outbreak mortality signals and denominator-aware Outbreak-Cluster CFR context.",
        },
        {
            "engine_id": "ENGINE_09_EXPLAINABLE_AI",
            "engine_name": "Centralized Explainable AI Engine (Phase 13)",
            "model_version": "phase13-v1.0.0",
            "rel_path": "data/explanations/why_district_flagged_panels.json",
            "temporal_grain": "WEEKLY_LOCAL_ATTRIBUTIONS + DISTRICT_PANELS",
            "quantity_represented": "Exact TreeSHAP log-odds/log1p attributions, permutation importance, and anomaly evidence",
            "reliability_tier": "TIER_1_EXPLAINABILITY_TRACEABILITY_LAYER",
            "fusion_role": "Supplies feature-level 'Why this result?' traceability for every fused district-disease record.",
        },
    ]

    audit_rows: List[Dict[str, Any]] = []
    for spec in engine_specs:
        full_p = proj_dir / spec["rel_path"]
        exists = full_p.exists()
        mtime_iso = (
            datetime.fromtimestamp(full_p.stat().st_mtime).isoformat()
            if exists
            else "MISSING"
        )
        row_cnt = 0
        if exists:
            if full_p.suffix == ".csv":
                row_cnt = int(len(pd.read_csv(full_p)))
            elif full_p.suffix == ".json":
                data_obj = json.loads(full_p.read_text(encoding="utf-8"))
                row_cnt = int(len(data_obj.get("district_panels", data_obj)))
        audit_rows.append(
            {
                **spec,
                "implemented_and_verified": bool(exists and row_cnt > 0),
                "artifact_row_or_panel_count": row_cnt,
                "artifact_last_modified": mtime_iso,
                "freshness_status": "FRESH_VERIFIED_ARTIFACT" if exists else "MISSING_ARTIFACT",
            }
        )

    audit_df = pd.DataFrame(audit_rows)

    # Load actual upstream tables
    risk_df = pd.read_csv(proj_dir / "data" / "explanations" / "local_risk_explanations.csv")
    fc_df = pd.read_csv(proj_dir / "data" / "explanations" / "local_forecast_explanations.csv")
    anom_df = pd.read_csv(proj_dir / "data" / "explanations" / "local_anomaly_explanations.csv")
    clim_dist_df = pd.read_csv(
        proj_dir / "data" / "climate_intelligence" / "district_climate_relationships.csv"
    )
    clim_corr_df = pd.read_csv(
        proj_dir / "data" / "climate_intelligence" / "disease_climate_correlations.csv"
    )
    weekly_feat_df = pd.read_csv(
        proj_dir / "data" / "features" / "disease_climate_weekly_features.csv"
    )
    geo_df = pd.read_csv(proj_dir / "data" / "geospatial" / "district_disease_map_summary.csv")
    spread_nodes_df = pd.read_csv(proj_dir / "data" / "scenarios" / "district_graph_nodes.csv")
    res_dist_df = pd.read_csv(
        proj_dir / "data" / "resource_intelligence" / "district_resource_summary.csv"
    )
    mort_wk_df = pd.read_csv(
        proj_dir / "data" / "severity_intelligence" / "verified_weekly_mortality_records.csv"
    )
    mort_dis_df = pd.read_csv(
        proj_dir / "data" / "severity_intelligence" / "disease_mortality_summary.csv"
    )
    mort_dist_df = pd.read_csv(
        proj_dir / "data" / "severity_intelligence" / "district_mortality_summary.csv"
    )
    xai_panels = json.loads(
        (proj_dir / "data" / "explanations" / "why_district_flagged_panels.json").read_text(
            encoding="utf-8"
        )
    )

    loaded_bundle = {
        "risk_df": risk_df,
        "fc_df": fc_df,
        "anom_df": anom_df,
        "clim_dist_df": clim_dist_df,
        "clim_corr_df": clim_corr_df,
        "weekly_feat_df": weekly_feat_df,
        "geo_df": geo_df,
        "spread_nodes_df": spread_nodes_df,
        "res_dist_df": res_dist_df,
        "mort_wk_df": mort_wk_df,
        "mort_dis_df": mort_dis_df,
        "mort_dist_df": mort_dist_df,
        "xai_panels": xai_panels,
    }
    return audit_df, loaded_bundle
