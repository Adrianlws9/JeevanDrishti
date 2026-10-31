"""Reusable FastAPI-Ready District Spread Scenario Simulation Service (Phase 10).

Provides:
1. Inspection of the 36-district graph topology & adjacency edges.
2. Retrieval of pre-computed benchmark scenarios (`SCEN_01`..`SCEN_06`).
3. On-demand deterministic execution of custom what-if scenarios with full parameter
   validation and before/after intervention comparison.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.engines.spread_simulation.district_graph import (
    MANDATORY_SCENARIO_DISCLAIMER,
    build_district_graph,
)
from src.engines.spread_simulation.scenario_simulator import (
    ScenarioParameters,
    run_scenario_simulation,
)

PROJ = Path(__file__).resolve().parents[3]


class DistrictSpreadScenarioService:
    """FastAPI-compatible service wrapper for Phase 10 District Spread & Scenario Engine."""

    def __init__(self, proj_dir: Optional[Path] = None):
        self.proj_dir = Path(proj_dir) if proj_dir else PROJ
        self.data_dir = self.proj_dir / "data" / "scenarios"
        self.graph = build_district_graph(self.proj_dir)
        self.catalog_df = pd.read_csv(self.data_dir / "scenario_summary_catalog.csv")
        self.step_records_df = pd.read_csv(self.data_dir / "scenario_simulation_step_records.csv")
        with open(self.data_dir / "scenario_runs_metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def get_graph_topology(self) -> Dict[str, Any]:
        """Return the 36-district nodes and 166 directed adjacency arcs."""
        return {
            "status": "OK",
            "total_nodes": len(self.graph.nodes_df),
            "geocoded_nodes": int(self.graph.nodes_df["has_verified_coordinates"].sum()),
            "isolated_nodes": ["Mumbai Suburban"],
            "undirected_edges": len(self.graph.edges_df) // 2,
            "directed_arcs": len(self.graph.edges_df),
            "nodes": json.loads(self.graph.nodes_df.to_json(orient="records")),
            "edges": json.loads(self.graph.edges_df.to_json(orient="records")),
            "non_prediction_disclaimer": MANDATORY_SCENARIO_DISCLAIMER,
        }

    def list_benchmark_scenarios(self) -> List[Dict[str, Any]]:
        """Return summaries of all pre-configured benchmark scenarios."""
        return self.metadata.get("scenarios", [])

    def simulate_custom_scenario(
        self,
        initial_affected_districts: Dict[str, float],
        scenario_duration_steps: int = 10,
        hypothetical_propagation_factor: float = 0.35,
        within_district_persistence_factor: float = 0.80,
        neighbour_influence_decay_km: float = 120.0,
        edge_weighting_mode: str = "distance_decayed",
        cross_division_attenuation: float = 0.65,
        intervention_start_step: int = 5,
        intervention_response_coverage: float = 0.40,
        activation_threshold: float = 5.0,
        scenario_id: str = "CUSTOM_SCENARIO",
        scenario_name: str = "User-Configured What-If Scenario",
    ) -> Dict[str, Any]:
        """Run a custom deterministic scenario and return both the intervention run and
        an unmitigated baseline comparison."""
        params = ScenarioParameters(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            scenario_description="Custom hypothetical scenario run via DistrictSpreadScenarioService",
            initial_affected_districts=dict(initial_affected_districts),
            scenario_duration_steps=scenario_duration_steps,
            hypothetical_propagation_factor=hypothetical_propagation_factor,
            within_district_persistence_factor=within_district_persistence_factor,
            neighbour_influence_decay_km=neighbour_influence_decay_km,
            edge_weighting_mode=edge_weighting_mode,
            cross_division_attenuation=cross_division_attenuation,
            intervention_start_step=intervention_start_step,
            intervention_response_coverage=intervention_response_coverage,
            activation_threshold=activation_threshold,
        )
        res = run_scenario_simulation(self.graph, params)

        # Also run unmitigated baseline (0% coverage) for before-and-after comparison
        baseline_params = ScenarioParameters(
            scenario_id=f"{scenario_id}_UNMITIGATED_REF",
            scenario_name=f"{scenario_name} (Unmitigated Reference)",
            scenario_description="Unmitigated reference run (0% response coverage)",
            initial_affected_districts=dict(initial_affected_districts),
            scenario_duration_steps=scenario_duration_steps,
            hypothetical_propagation_factor=hypothetical_propagation_factor,
            within_district_persistence_factor=within_district_persistence_factor,
            neighbour_influence_decay_km=neighbour_influence_decay_km,
            edge_weighting_mode=edge_weighting_mode,
            cross_division_attenuation=cross_division_attenuation,
            intervention_start_step=scenario_duration_steps,
            intervention_response_coverage=0.0,
            activation_threshold=activation_threshold,
        )
        base_res = run_scenario_simulation(self.graph, baseline_params)

        return {
            "status": "OK",
            "summary": res["summary"],
            "unmitigated_reference_summary": base_res["summary"],
            "final_step_signal_reduction_pct": round(
                100.0
                * (
                    1.0
                    - res["summary"]["final_step_signal_sum"]
                    / max(base_res["summary"]["final_step_signal_sum"], 1e-6)
                ),
                2,
            ),
            "step_records": json.loads(res["step_records_df"].to_json(orient="records")),
            "non_prediction_disclaimer": MANDATORY_SCENARIO_DISCLAIMER,
        }
