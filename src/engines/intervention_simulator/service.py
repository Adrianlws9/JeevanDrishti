"""Backend Query & Simulation Service for Public-Health Intervention Scenario Simulator (Phase 14).

Provides `PublicHealthInterventionSimulatorService` for:
1. Querying upstream capability audit status
2. Listing pre-computed benchmark scenarios & `Baseline vs. Scenario A vs. Scenario B` comparisons
3. Running custom user-configured intervention scenarios on the fly
4. Running custom head-to-head `Baseline vs. Scenario A vs. Scenario B` comparisons
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from jeevan_pipeline.standardize import standardize_disease, standardize_district
from src.engines.intervention_simulator.capability_inspector import (
    ENGINE_VERSION,
    MANDATORY_INTERVENTION_GOVERNANCE_DISCLAIMER,
    PROJ,
    inspect_upstream_project_capabilities,
)
from src.engines.intervention_simulator.intervention_calculator import (
    SUPPORTED_INTERVENTION_TYPES,
    InterventionScenarioConfig,
    compare_baseline_vs_scenarios_a_and_b,
    simulate_intervention_scenario,
)
from src.features.geographic_features import OFFICIAL_DISTRICTS


class PublicHealthInterventionSimulatorService:
    """High-level query and on-demand simulation service for Phase 14 Intervention Simulator."""

    def __init__(self, proj_dir: Path = PROJ) -> None:
        self.proj_dir = proj_dir
        self.data_dir = proj_dir / "data" / "intervention_scenarios"
        self.capability_df, self.pool_df, self.adj_map, self.cap_meta = (
            inspect_upstream_project_capabilities(proj_dir)
        )

    def _normalize_district_list(self, districts: List[str]) -> List[str]:
        normed: List[str] = []
        for raw_d in districts:
            std_d, _ = standardize_district(str(raw_d).strip())
            candidate = std_d if std_d in OFFICIAL_DISTRICTS else str(raw_d).strip()
            if candidate not in OFFICIAL_DISTRICTS:
                raise ValueError(
                    f"Unrecognized Maharashtra district '{raw_d}' (resolved: '{candidate}')."
                )
            if candidate not in normed:
                normed.append(candidate)
        return normed

    def _normalize_disease_name(self, disease: str) -> str:
        raw = str(disease).strip()
        if raw.upper() == "ALL_DISEASES":
            return "ALL_DISEASES"
        std_dis, _, is_mapped = standardize_disease(raw)
        if is_mapped and std_dis:
            return std_dis
        return raw

    def get_capability_inspection(self) -> Dict[str, Any]:
        """Return upstream capability inspection table and governance metadata."""
        return {
            "engine_version": ENGINE_VERSION,
            "governance_disclaimer": MANDATORY_INTERVENTION_GOVERNANCE_DISCLAIMER,
            "supported_intervention_types": list(SUPPORTED_INTERVENTION_TYPES),
            "metadata": self.cap_meta,
            "upstream_capabilities": self.capability_df.to_dict(orient="records"),
        }

    def get_benchmark_scenarios_catalog(self) -> Dict[str, Any]:
        """Return pre-computed benchmark scenarios and comparisons from disk."""
        payload_path = self.data_dir / "intervention_scenarios_frontend_payload.json"
        if not payload_path.exists():
            raise FileNotFoundError(
                f"Frontend payload not found at {payload_path}. Run pipeline first."
            )
        return json.loads(payload_path.read_text(encoding="utf-8"))

    def run_custom_intervention_scenario(
        self,
        intervention_type: str = "ADDITIONAL_VECTOR_CONTROL_ACTIVITY",
        selected_disease: str = "Dengue",
        selected_districts: Optional[List[str]] = None,
        include_adjacent_ring_districts: bool = False,
        scenario_duration_weeks: int = 8,
        intervention_start_week: int = 2,
        intervention_ramp_up_weeks: int = 2,
        baseline_weekly_growth_multiplier: float = 1.08,
        baseline_cross_district_coupling_beta: float = 0.06,
        custom_initial_weekly_cases_per_district: Optional[float] = None,
        user_assumed_transmission_dampening_factor: float = 0.25,
        user_assumed_cross_district_spillover_dampening: float = 0.20,
        user_assumed_active_testing_detection_multiplier: float = 1.15,
        user_assumed_admission_proportion: float = 0.20,
        user_assumed_icu_proportion_of_admissions: float = 0.15,
        user_assumed_avg_length_of_stay_days: float = 5.0,
        user_assumed_tests_per_case: float = 10.0,
        hypothetical_allocated_general_beds_per_district: Optional[float] = 12.0,
        hypothetical_allocated_icu_beds_per_district: Optional[float] = 2.0,
        scenario_id: str = "CUSTOM_USER_SCENARIO",
        scenario_name: str = "Custom Analyst What-If Scenario",
        analyst_notes: str = "",
    ) -> Dict[str, Any]:
        """Run a custom user-defined intervention scenario and return Section 5 outputs + weekly trajectory."""
        dists = self._normalize_district_list(selected_districts or ["Pune"])
        dis = self._normalize_disease_name(selected_disease)

        cfg = InterventionScenarioConfig(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            intervention_type=intervention_type,
            selected_disease=dis,
            selected_districts=dists,
            include_adjacent_ring_districts=include_adjacent_ring_districts,
            scenario_duration_weeks=scenario_duration_weeks,
            intervention_start_week=intervention_start_week,
            intervention_ramp_up_weeks=intervention_ramp_up_weeks,
            baseline_weekly_growth_multiplier=baseline_weekly_growth_multiplier,
            baseline_cross_district_coupling_beta=baseline_cross_district_coupling_beta,
            custom_initial_weekly_cases_per_district=custom_initial_weekly_cases_per_district,
            user_assumed_transmission_dampening_factor=user_assumed_transmission_dampening_factor,
            user_assumed_cross_district_spillover_dampening=user_assumed_cross_district_spillover_dampening,
            user_assumed_active_testing_detection_multiplier=user_assumed_active_testing_detection_multiplier,
            user_assumed_admission_proportion=user_assumed_admission_proportion,
            user_assumed_icu_proportion_of_admissions=user_assumed_icu_proportion_of_admissions,
            user_assumed_avg_length_of_stay_days=user_assumed_avg_length_of_stay_days,
            user_assumed_tests_per_case=user_assumed_tests_per_case,
            hypothetical_allocated_general_beds_per_district=hypothetical_allocated_general_beds_per_district,
            hypothetical_allocated_icu_beds_per_district=hypothetical_allocated_icu_beds_per_district,
            analyst_notes=analyst_notes,
        )
        summary_out, traj_df = simulate_intervention_scenario(cfg, self.pool_df, self.adj_map)
        return {
            "summary": summary_out,
            "weekly_trajectory": json.loads(traj_df.to_json(orient="records")),
        }

    def compare_custom_scenarios_a_and_b(
        self,
        selected_disease: str,
        selected_districts: List[str],
        scenario_a_params: Dict[str, Any],
        scenario_b_params: Dict[str, Any],
        scenario_duration_weeks: int = 8,
        baseline_weekly_growth_multiplier: float = 1.10,
        comparison_id: str = "CUSTOM_COMPARISON_A_VS_B",
        comparison_title: str = "Custom Analyst Comparison: Baseline vs. Scenario A vs. Scenario B",
    ) -> Dict[str, Any]:
        """Compare `Baseline` vs. `Scenario A` vs. `Scenario B` on the same district set and disease."""
        dists = self._normalize_district_list(selected_districts)
        dis = self._normalize_disease_name(selected_disease)

        base_cfg = InterventionScenarioConfig(
            scenario_id="CUSTOM_BASELINE",
            scenario_name="Baseline (No New Intervention)",
            intervention_type="BASELINE_NO_ADDITIONAL_INTERVENTION",
            selected_disease=dis,
            selected_districts=dists,
            scenario_duration_weeks=scenario_duration_weeks,
            baseline_weekly_growth_multiplier=baseline_weekly_growth_multiplier,
            user_assumed_transmission_dampening_factor=0.0,
            user_assumed_cross_district_spillover_dampening=0.0,
            user_assumed_active_testing_detection_multiplier=1.0,
        )

        a_cfg = InterventionScenarioConfig(
            scenario_id=scenario_a_params.get("scenario_id", "CUSTOM_SCENARIO_A"),
            scenario_name=scenario_a_params.get("scenario_name", "Intervention Scenario A"),
            intervention_type=scenario_a_params.get(
                "intervention_type", "ADDITIONAL_VECTOR_CONTROL_ACTIVITY"
            ),
            selected_disease=dis,
            selected_districts=dists,
            include_adjacent_ring_districts=bool(
                scenario_a_params.get("include_adjacent_ring_districts", False)
            ),
            scenario_duration_weeks=scenario_duration_weeks,
            intervention_start_week=int(scenario_a_params.get("intervention_start_week", 2)),
            intervention_ramp_up_weeks=int(scenario_a_params.get("intervention_ramp_up_weeks", 2)),
            baseline_weekly_growth_multiplier=baseline_weekly_growth_multiplier,
            user_assumed_transmission_dampening_factor=float(
                scenario_a_params.get("user_assumed_transmission_dampening_factor", 0.25)
            ),
            user_assumed_cross_district_spillover_dampening=float(
                scenario_a_params.get("user_assumed_cross_district_spillover_dampening", 0.20)
            ),
            user_assumed_active_testing_detection_multiplier=float(
                scenario_a_params.get("user_assumed_active_testing_detection_multiplier", 1.15)
            ),
            hypothetical_allocated_general_beds_per_district=scenario_a_params.get(
                "hypothetical_allocated_general_beds_per_district", 12.0
            ),
            hypothetical_allocated_icu_beds_per_district=scenario_a_params.get(
                "hypothetical_allocated_icu_beds_per_district", 2.0
            ),
        )

        b_cfg = InterventionScenarioConfig(
            scenario_id=scenario_b_params.get("scenario_id", "CUSTOM_SCENARIO_B"),
            scenario_name=scenario_b_params.get("scenario_name", "Intervention Scenario B"),
            intervention_type=scenario_b_params.get(
                "intervention_type", "EARLY_INTERVENTION_VS_DELAYED_INTERVENTION"
            ),
            selected_disease=dis,
            selected_districts=dists,
            include_adjacent_ring_districts=bool(
                scenario_b_params.get("include_adjacent_ring_districts", False)
            ),
            scenario_duration_weeks=scenario_duration_weeks,
            intervention_start_week=int(scenario_b_params.get("intervention_start_week", 5)),
            intervention_ramp_up_weeks=int(scenario_b_params.get("intervention_ramp_up_weeks", 2)),
            baseline_weekly_growth_multiplier=baseline_weekly_growth_multiplier,
            user_assumed_transmission_dampening_factor=float(
                scenario_b_params.get("user_assumed_transmission_dampening_factor", 0.25)
            ),
            user_assumed_cross_district_spillover_dampening=float(
                scenario_b_params.get("user_assumed_cross_district_spillover_dampening", 0.20)
            ),
            user_assumed_active_testing_detection_multiplier=float(
                scenario_b_params.get("user_assumed_active_testing_detection_multiplier", 1.15)
            ),
            hypothetical_allocated_general_beds_per_district=scenario_b_params.get(
                "hypothetical_allocated_general_beds_per_district", 12.0
            ),
            hypothetical_allocated_icu_beds_per_district=scenario_b_params.get(
                "hypothetical_allocated_icu_beds_per_district", 2.0
            ),
        )

        cmp_sum, wk_df, dist_df = compare_baseline_vs_scenarios_a_and_b(
            comparison_id=comparison_id,
            comparison_title=comparison_title,
            baseline_cfg=base_cfg,
            scenario_a_cfg=a_cfg,
            scenario_b_cfg=b_cfg,
            pool_df=self.pool_df,
            adj_map=self.adj_map,
        )
        return {
            "comparison_summary": cmp_sum,
            "weekly_side_by_side": json.loads(wk_df.to_json(orient="records")),
            "district_side_by_side": json.loads(dist_df.to_json(orient="records")),
        }
