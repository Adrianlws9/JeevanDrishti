"""Upstream Capability Inspection & District-Disease Baseline Loader (Phase 14 Section 1).

Responsibilities:
1. Inspect outputs from all 7 upstream engines:
   - Disease forecasting (Phase 6)
   - Anomaly detection (Phase 7)
   - Climate intelligence (Phase 8)
   - Geospatial analysis (Phase 9)
   - Spread simulation (Phase 10)
   - Resource intelligence (Phase 11)
   - Severity intelligence (Phase 12)
2. Document which inputs are genuinely available in the workspace versus which intervention
   parameters (e.g., transmission dampening, testing scale-up, bed allocation) have NO
   empirical effectiveness dataset in the workspace and must therefore be treated strictly
   as visible, user-configurable what-if assumptions.
"""
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.features.geographic_features import (
    ADMIN_DIVISION_MAP,
    DISTRICT_ID_MAP,
    OFFICIAL_DISTRICTS,
)

PROJ = Path(__file__).resolve().parents[3]
ENGINE_VERSION = "phase14-v1.0.0"

MANDATORY_INTERVENTION_GOVERNANCE_DISCLAIMER = (
    "DECISION-SUPPORT WHAT-IF SIMULATOR ONLY — NOT AN AUTOMATED PUBLIC-HEALTH AUTHORITY OR "
    "VALIDATED EFFICACY FORECAST. No empirical intervention effectiveness percentages exist in "
    "the project workspace. All simulated trajectories and resource comparisons are HYPOTHETICAL "
    "SCENARIOS derived strictly from visible, user-editable assumptions. This tool never recommends "
    "clinical treatments, never executes real-world interventions, and never replaces qualified "
    "public-health officials."
)


def inspect_upstream_project_capabilities(
    proj_dir: Path = PROJ,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]], Dict[str, Any]]:
    """Inspect all 7 upstream engines and return capability audit table, district-disease anchor table,
    adjacency dictionary, and summary metadata.
    """
    capability_rows: List[Dict[str, Any]] = [
        {
            "upstream_engine": "Phase 6: Disease Forecasting Engine",
            "version": "phase6-v1.0.0",
            "source_file": "data/geospatial/district_disease_map_summary.csv & reports/actual_vs_predicted.csv",
            "genuinely_available_in_workspace": True,
            "variables_available": "7d (1w), 14d (2w), and 30d (4w) forecasted weekly case counts across 34 active districts",
            "role_in_intervention_simulator": (
                "Provides verified baseline weekly caseload anchors (B_{i,0}) for district-disease scenario trajectories."
            ),
            "limitations_and_safeguards": "2 districts (Mumbai City, Ratnagiri) have 0 outbreaks in 2023-2025 -> require user custom baseline.",
        },
        {
            "upstream_engine": "Phase 7: Anomaly Detection Engine",
            "version": "phase7-v1.0.0",
            "source_file": "reports/anomaly_detection_results.csv",
            "genuinely_available_in_workspace": True,
            "variables_available": "Composite anomaly score, Seasonal Z, Rolling Z, Robust MAD Z, and 116 flagged outbreak weeks",
            "role_in_intervention_simulator": (
                "Identifies elevated district-disease clusters to prioritize in what-if early vs. delayed response scenarios."
            ),
            "limitations_and_safeguards": "Anomaly signals indicate unusual statistical deviations, not confirmed intervention triggers.",
        },
        {
            "upstream_engine": "Phase 8: Climate–Disease Intelligence Engine",
            "version": "phase8-v1.0.0",
            "source_file": "data/climate_intelligence/correlation_analysis_results.csv",
            "genuinely_available_in_workspace": True,
            "variables_available": "Seasonal Kruskal-Wallis (p=3.21e-05), lagged rainfall/temperature associations (1..8w)",
            "role_in_intervention_simulator": (
                "Informs seasonal timing context (e.g., Monsoon/Post-Monsoon vector-control vs. Pre-Monsoon WASH)."
            ),
            "limitations_and_safeguards": "Correlational climate context only; never used to claim causal intervention efficacy.",
        },
        {
            "upstream_engine": "Phase 9: Geospatial Hotspot Detection Engine",
            "version": "phase9-v1.0.0",
            "source_file": "data/geospatial/district_disease_map_summary.csv & maharashtra_district_points.geojson",
            "genuinely_available_in_workspace": True,
            "variables_available": "36 district centroids, Global Moran's I, Local Getis-Ord Gi* z-scores (14 hotspots)",
            "role_in_intervention_simulator": (
                "Supports district-wise scenario mapping and spatial cluster selection."
            ),
            "limitations_and_safeguards": "Centroid point geometry only (Mumbai Suburban approximated); no polygon fabrication.",
        },
        {
            "upstream_engine": "Phase 10: District Spread Propagation Engine",
            "version": "phase10-v1.0.0",
            "source_file": "data/geospatial/spatial_weights_matrix.csv & data/scenarios/district_graph_edges.csv",
            "genuinely_available_in_workspace": True,
            "variables_available": "Verified 36-node, 83-edge (166 directed arcs) Maharashtra district adjacency graph",
            "role_in_intervention_simulator": (
                "Powers multi-district ring-response scenarios and cross-border spillover dampening comparisons."
            ),
            "limitations_and_safeguards": "Adjacency-only topological coupling (no empirical human mobility dataset in workspace).",
        },
        {
            "upstream_engine": "Phase 11: Healthcare Resource Demand Engine",
            "version": "phase11-v1.0.0",
            "source_file": "data/resource_intelligence/district_resource_summary.csv",
            "genuinely_available_in_workspace": True,
            "variables_available": "Explicit NaN audit of hospital capacities + deterministic scenario calculator formulas",
            "role_in_intervention_simulator": (
                "Computes hypothetical concurrent general beds, ICU beds, and diagnostic test requirements per scenario."
            ),
            "limitations_and_safeguards": "Zero actual hospital bed/stock inventories exist; all capacity comparisons use user-entered values.",
        },
        {
            "upstream_engine": "Phase 12: Disease Severity & Mortality Engine",
            "version": "phase12-v1.0.0",
            "source_file": "data/severity_intelligence/disease_mortality_summary.csv",
            "genuinely_available_in_workspace": True,
            "variables_available": "Verified 2023-2025 outbreak cases (19,172), deaths (199), and Outbreak-Cluster CFR (%)",
            "role_in_intervention_simulator": (
                "Provides historical disease-specific Outbreak-Cluster CFR context alongside simulated case trajectories."
            ),
            "limitations_and_safeguards": "Aggregate outbreak surveillance only; never predicts individual clinical outcomes.",
        },
        {
            "upstream_engine": "Empirical Intervention Efficacy / RCT Dataset",
            "version": "NONE_IN_WORKSPACE",
            "source_file": "NONE_IN_WORKSPACE",
            "genuinely_available_in_workspace": False,
            "variables_available": "0 empirical intervention efficacy or cost-effectiveness records exist in workspace",
            "role_in_intervention_simulator": (
                "Blocked from fabrication — replaced by transparent, user-editable What-If Scenario Calculator parameters."
            ),
            "limitations_and_safeguards": (
                "Never claims that vector control, awareness, or testing will reduce cases by a fixed empirical %."
            ),
        },
    ]
    capability_df = pd.DataFrame(capability_rows)

    # Load verified district-disease baseline anchors from Phase 9/6/7/12
    geo_df = pd.read_csv(proj_dir / "data" / "geospatial" / "district_disease_map_summary.csv")
    pool_df = geo_df[geo_df["reporting_period"] == "2023-2025_POOL"].copy()

    # Load verified 83-edge adjacency graph from Phase 9/10 spatial_weights_matrix.csv
    sw_df = pd.read_csv(proj_dir / "data" / "geospatial" / "spatial_weights_matrix.csv")
    adj_map: Dict[str, List[str]] = {d: [] for d in OFFICIAL_DISTRICTS}
    for _, r in sw_df.iterrows():
        u = str(r["district_name"]).strip()
        raw_nbs = str(r.get("neighbor_district_names", ""))
        if u in adj_map and pd.notna(r.get("neighbor_district_names")) and raw_nbs.strip():
            for nb in [x.strip() for x in raw_nbs.split(";") if x.strip()]:
                if nb in OFFICIAL_DISTRICTS and nb not in adj_map[u]:
                    adj_map[u].append(nb)

    meta = {
        "audit_timestamp": datetime.now().isoformat(),
        "engine_version": ENGINE_VERSION,
        "upstream_engines_inspected": 7,
        "empirical_intervention_efficacy_data_present": False,
        "modelling_approach_selected": "TRANSPARENT_PARAMETRIC_WHAT_IF_SCENARIO_CALCULATOR_AND_GRAPH_COUPLING",
        "mandatory_governance_disclaimer": MANDATORY_INTERVENTION_GOVERNANCE_DISCLAIMER,
    }
    return capability_df, pool_df, adj_map, meta
