"""Unit & Governance Tests for Phase 14 Public-Health Intervention Scenario Simulator."""
import unittest
from pathlib import Path

import pandas as pd

from src.engines.intervention_simulator import (
    ENGINE_VERSION,
    MANDATORY_INTERVENTION_GOVERNANCE_DISCLAIMER,
    SUPPORTED_INTERVENTION_TYPES,
    InterventionScenarioConfig,
    PublicHealthInterventionSimulatorService,
    build_benchmark_intervention_scenarios,
    compare_baseline_vs_scenarios_a_and_b,
    inspect_upstream_project_capabilities,
    simulate_intervention_scenario,
)

PROJ = Path(__file__).resolve().parents[1]


class TestPublicHealthInterventionSimulator(unittest.TestCase):
    """Comprehensive verification suite for Phase 14 Intervention Scenario Simulator."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cap_df, cls.pool_df, cls.adj_map, cls.cap_meta = (
            inspect_upstream_project_capabilities(PROJ)
        )
        cls.service = PublicHealthInterventionSimulatorService(PROJ)

    def test_01_upstream_capability_inspection_honesty(self) -> None:
        """Section 1: Verify 7 upstream engines are available and RCT efficacy data is flagged absent."""
        self.assertEqual(len(self.cap_df), 8)
        avail_count = int(self.cap_df["genuinely_available_in_workspace"].sum())
        self.assertEqual(avail_count, 7)
        self.assertFalse(self.cap_meta["empirical_intervention_efficacy_data_present"])

        rct_row = self.cap_df[~self.cap_df["genuinely_available_in_workspace"]].iloc[0]
        self.assertIn("Empirical Intervention Efficacy", rct_row["upstream_engine"])
        self.assertEqual(len(self.adj_map), 36)
        self.assertGreater(len(self.adj_map["Pune"]), 0)

    def test_02_all_seven_scenario_types_supported(self) -> None:
        """Section 2: Verify all 7 required scenario types (+ baseline) are configured and supported."""
        required_types = {
            "INCREASED_SURVEILLANCE_AND_TESTING",
            "ADDITIONAL_VECTOR_CONTROL_ACTIVITY",
            "PUBLIC_AWARENESS_CAMPAIGN",
            "INCREASED_HEALTHCARE_RESOURCE_ALLOCATION",
            "EARLY_INTERVENTION_VS_DELAYED_INTERVENTION",
            "DISTRICT_FOCUSED_RESPONSE",
            "MULTI_DISTRICT_RESPONSE",
        }
        self.assertTrue(required_types.issubset(set(SUPPORTED_INTERVENTION_TYPES)))
        benchmarks, suites = build_benchmark_intervention_scenarios()
        self.assertEqual(len(benchmarks), 8)
        self.assertEqual(len(suites), 3)
        configured_types = {b.intervention_type for b in benchmarks}
        self.assertTrue(required_types.issubset(configured_types))

    def test_03_required_scenario_output_fields_and_separation(self) -> None:
        """Sections 3 & 5: Verify every required output field and strict separation from observed data."""
        cfg = InterventionScenarioConfig(
            scenario_id="TEST_SCEN_DENGUE_PUNE",
            scenario_name="Test Vector Control Pune",
            intervention_type="ADDITIONAL_VECTOR_CONTROL_ACTIVITY",
            selected_disease="Dengue",
            selected_districts=["Pune", "Satara"],
            include_adjacent_ring_districts=True,
            scenario_duration_weeks=6,
            intervention_start_week=2,
            intervention_ramp_up_weeks=2,
            user_assumed_transmission_dampening_factor=0.25,
            hypothetical_allocated_general_beds_per_district=10.0,
            hypothetical_allocated_icu_beds_per_district=2.0,
        )
        summary_out, traj_df = simulate_intervention_scenario(cfg, self.pool_df, self.adj_map)

        required_keys = {
            "scenario_id",
            "selected_districts",
            "selected_intervention",
            "input_assumptions",
            "baseline_comparison",
            "simulated_indicators",
            "expected_resource_requirements",
            "model_or_calculation_method",
            "generation_timestamp",
            "limitations",
        }
        self.assertTrue(required_keys.issubset(set(summary_out.keys())))
        self.assertGreater(
            summary_out["selected_districts"]["simulated_districts_count"], 2
        )
        self.assertLess(
            summary_out["simulated_indicators"]["hypothetical_cumulative_underlying_cases"],
            summary_out["baseline_comparison"]["unmitigated_baseline_cumulative_cases"],
        )
        self.assertTrue(
            (
                traj_df["record_nature_label"]
                == "HYPOTHETICAL_SIMULATED_TRAJECTORY_NEVER_MIX_WITH_OBSERVED"
            ).all()
        )

    def test_04_early_vs_delayed_intervention_monotonicity(self) -> None:
        """Sections 2 & 6: Verify early intervention (Week 2) yields lower cumulative burden than delayed (Week 5)."""
        early_cfg = InterventionScenarioConfig(
            scenario_id="TEST_EARLY_W2",
            scenario_name="Early Start Week 2",
            intervention_type="EARLY_INTERVENTION_VS_DELAYED_INTERVENTION",
            selected_disease="Dengue",
            selected_districts=["Pune", "Nashik"],
            scenario_duration_weeks=8,
            intervention_start_week=2,
            user_assumed_transmission_dampening_factor=0.30,
        )
        delayed_cfg = InterventionScenarioConfig(
            scenario_id="TEST_DELAYED_W5",
            scenario_name="Delayed Start Week 5",
            intervention_type="EARLY_INTERVENTION_VS_DELAYED_INTERVENTION",
            selected_disease="Dengue",
            selected_districts=["Pune", "Nashik"],
            scenario_duration_weeks=8,
            intervention_start_week=5,
            user_assumed_transmission_dampening_factor=0.30,
        )
        s_early, _ = simulate_intervention_scenario(early_cfg, self.pool_df, self.adj_map)
        s_delayed, _ = simulate_intervention_scenario(delayed_cfg, self.pool_df, self.adj_map)

        self.assertLess(
            s_early["simulated_indicators"]["hypothetical_cumulative_underlying_cases"],
            s_delayed["simulated_indicators"]["hypothetical_cumulative_underlying_cases"],
        )

    def test_05_service_custom_simulation_and_comparison(self) -> None:
        """Sections 4 & 6: Verify service layer handles district aliases and Scenario A vs. B comparisons."""
        res = self.service.run_custom_intervention_scenario(
            intervention_type="PUBLIC_AWARENESS_CAMPAIGN",
            selected_disease="Acute Diarrhoeal Disease",
            selected_districts=["Aurangabad", "Osmanabad"],  # historical names -> standardized
            scenario_duration_weeks=6,
            intervention_start_week=2,
            user_assumed_transmission_dampening_factor=0.20,
        )
        focal = res["summary"]["selected_districts"]["focal_districts"]
        self.assertEqual(focal, ["Chhatrapati Sambhajinagar", "Dharashiv"])

        cmp_res = self.service.compare_custom_scenarios_a_and_b(
            selected_disease="Dengue",
            selected_districts=["Pune", "Thane"],
            scenario_a_params={
                "scenario_id": "A_EARLY",
                "intervention_start_week": 2,
                "user_assumed_transmission_dampening_factor": 0.25,
            },
            scenario_b_params={
                "scenario_id": "B_DELAYED",
                "intervention_start_week": 5,
                "user_assumed_transmission_dampening_factor": 0.25,
            },
            scenario_duration_weeks=8,
        )
        h = cmp_res["comparison_summary"]["head_to_head_metrics"]
        self.assertLess(h["scenario_a_cumulative_cases"], h["scenario_b_cumulative_cases"])
        self.assertLess(h["scenario_b_cumulative_cases"], h["baseline_cumulative_cases"])

    def test_06_generated_files_and_visual_artifacts_exist(self) -> None:
        """Section 7 & 9: Verify all CSVs, JSON payload, 5 PNGs, HTML simulator, and report exist."""
        data_dir = PROJ / "data" / "intervention_scenarios"
        rep_dir = PROJ / "reports" / "intervention_simulator"
        fig_dir = rep_dir / "figures"

        expected_files = [
            data_dir / "upstream_capability_inspection.csv",
            data_dir / "intervention_scenario_weekly_trajectories.csv",
            data_dir / "intervention_scenario_summary_catalog.csv",
            data_dir / "baseline_vs_scenario_a_vs_scenario_b_weekly.csv",
            data_dir / "baseline_vs_scenario_a_vs_scenario_b_districts.csv",
            data_dir / "intervention_scenarios_frontend_payload.json",
            rep_dir / "interactive_intervention_scenario_simulator.html",
            rep_dir / "intervention_scenario_simulator_report.md",
            fig_dir / "01_project_capability_and_intervention_support_audit.png",
            fig_dir / "02_baseline_vs_scenario_a_vs_scenario_b_timelines.png",
            fig_dir / "03_district_wise_scenario_comparison_map.png",
            fig_dir / "04_intervention_timing_and_assumption_sensitivity_grid.png",
            fig_dir / "05_hypothetical_resource_allocation_and_bed_strain_comparison.png",
        ]
        for p in expected_files:
            self.assertTrue(p.exists(), f"Missing artifact: {p}")
            self.assertGreater(p.stat().st_size, 500, f"Artifact unexpectedly small: {p}")


if __name__ == "__main__":
    unittest.main()
