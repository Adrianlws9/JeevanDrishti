"""Unit Tests for Phase 11 Healthcare Resource Demand Intelligence Engine."""
from pathlib import Path
import unittest

import pandas as pd

from src.engines.resource_intelligence.data_availability import (
    MANDATORY_RESOURCE_DISCLAIMER,
    RESOURCE_CATEGORIES_SPEC,
    audit_resource_data_availability,
    build_district_resource_and_epidemiological_summary,
)
from src.engines.resource_intelligence.scenario_calculator import (
    ResourceScenarioAssumptions,
    build_default_resource_scenarios,
    compute_single_resource_scenario_row,
    evaluate_scenarios_across_districts,
)
from src.engines.resource_intelligence.service import HealthcareResourceDemandService

PROJ = Path(__file__).resolve().parents[1]


class TestResourceIntelligenceEngine(unittest.TestCase):
    """Verify scientific safeguards, non-fabrication of hospital capacities, and hypothetical scenario calculator."""

    def test_01_audit_confirms_zero_empirical_resource_datasets(self) -> None:
        completeness_df, meta = audit_resource_data_availability(PROJ)
        self.assertEqual(meta["empirical_resource_datasets_found"], 0)
        self.assertEqual(meta["empirical_resource_variables_available"], 0)
        self.assertFalse(meta["empirical_resource_forecasting_supported"])

        emp_rows = completeness_df.iloc[: len(RESOURCE_CATEGORIES_SPEC)]
        self.assertTrue((emp_rows["present_in_workspace"] == False).all())
        self.assertTrue((emp_rows["missingness_pct"] == 100.0).all())
        self.assertTrue((emp_rows["empirical_forecasting_supported"] == False).all())

    def test_02_district_summary_never_invents_hospital_capacities(self) -> None:
        dist_df, trend_df = build_district_resource_and_epidemiological_summary(PROJ)
        self.assertEqual(len(dist_df), 36)

        empirical_cols = [
            "empirical_hospital_admissions",
            "empirical_available_beds",
            "empirical_occupied_beds",
            "empirical_bed_occupancy_ratio",
            "empirical_icu_capacity",
            "empirical_testing_volume",
            "empirical_medicine_consumption",
            "empirical_medicine_stock",
            "empirical_healthcare_facilities_count",
        ]
        for col in empirical_cols:
            self.assertTrue(
                dist_df[col].isna().all(),
                f"Column {col} must be strictly NaN (never fabricated).",
            )

        # Verify 34 active districts have upstream epidemiological anchors and 2 (Mumbai City, Ratnagiri) are NaN
        self.assertEqual(int(dist_df["observed_2023_2025_peak_weekly_cases"].notna().sum()), 34)
        self.assertEqual(int((dist_df["observed_2023_2025_outbreak_weeks"] == 0).sum()), 2)

        # Verify trend_df keeps empirical columns strictly NaN
        self.assertTrue(trend_df["empirical_hospital_admissions"].isna().all())
        self.assertTrue(trend_df["empirical_occupied_beds"].isna().all())

    def test_03_scenario_calculator_validation_and_math(self) -> None:
        # Invalid admission proportion > 1.0 must raise ValueError
        with self.assertRaises(ValueError):
            ResourceScenarioAssumptions(
                scenario_id="BAD",
                scenario_name="Bad",
                scenario_description="Bad",
                assumed_admission_proportion=1.5,
            ).validate()

        # Invalid zero/negative bed capacity must raise ValueError
        with self.assertRaises(ValueError):
            ResourceScenarioAssumptions(
                scenario_id="BAD_CAP",
                scenario_name="Bad Cap",
                scenario_description="Bad Cap",
                assumed_available_general_beds=0.0,
            ).validate()

        # Deterministic check: 100 base cases, +40% increase -> 140 cases
        # 20% admission -> 28 admissions
        # 10% ICU share, 3.5-day stay -> total bed-days = 98.0
        # Concurrent general beds = (28 * 0.90 * 3.5) / 7 = 12.6
        # Concurrent ICU beds = (28 * 0.10 * 3.5) / 7 = 1.4
        sc = ResourceScenarioAssumptions(
            scenario_id="TEST_SCEN",
            scenario_name="Test Scenario",
            scenario_description="Exact math check",
            case_load_anchor_mode="user_entered_cases",
            user_entered_baseline_cases=100.0,
            hypothetical_case_increase_pct=40.0,
            assumed_admission_proportion=0.20,
            assumed_icu_proportion_of_admissions=0.10,
            assumed_avg_length_of_stay_days=3.5,
            assumed_tests_per_case=2.5,
            assumed_iv_fluid_units_per_admission=5.0,
            assumed_treatment_courses_per_case=1.0,
            assumed_available_general_beds=10.0,
            assumed_available_icu_beds=2.0,
        )
        out = compute_single_resource_scenario_row(
            district_name="Pune",
            district_id="MH_D26",
            division="Pune",
            baseline_cases=100.0,
            assumptions=sc,
            calc_timestamp="2026-10-31T16:40:00",
        )
        self.assertAlmostEqual(out["hypothetical_projected_weekly_cases"], 140.0)
        self.assertAlmostEqual(out["hypothetical_weekly_admissions"], 28.0)
        self.assertAlmostEqual(out["hypothetical_total_bed_days_required"], 98.0)
        self.assertAlmostEqual(out["hypothetical_concurrent_general_beds_required"], 12.6)
        self.assertAlmostEqual(out["hypothetical_concurrent_icu_beds_required"], 1.4)
        self.assertAlmostEqual(out["hypothetical_general_bed_utilization_ratio"], 1.26)
        self.assertEqual(
            out["hypothetical_shortage_indicator"],
            "HYPOTHETICAL_CAPACITY_DEFICIT_EXCEEDS_100PCT_ASSUMED_BEDS",
        )
        self.assertEqual(
            out["output_nature"],
            "HYPOTHETICAL_SCENARIO_NOT_ACTUAL_CAPACITY_OR_FORECAST",
        )

    def test_04_generated_artifacts_and_service_layer(self) -> None:
        data_dir = PROJ / "data" / "resource_intelligence"
        rep_dir = PROJ / "reports" / "resource_intelligence"

        expected_files = [
            data_dir / "resource_data_completeness_audit.csv",
            data_dir / "district_resource_summary.csv",
            data_dir / "historical_epidemiological_demand_trends.csv",
            data_dir / "hypothetical_resource_scenarios_catalog.csv",
            data_dir / "hypothetical_district_scenario_evaluations.csv",
            data_dir / "frontend_integration_payload.json",
            rep_dir / "resource_demand_intelligence_report.md",
            rep_dir / "interactive_resource_demand_calculator.html",
            rep_dir / "figures" / "01_resource_data_availability_and_completeness_audit.png",
            rep_dir / "figures" / "02_upstream_epidemiological_demand_drivers_by_district.png",
            rep_dir / "figures" / "03_hypothetical_resource_scenario_comparison.png",
            rep_dir / "figures" / "04_hypothetical_capacity_strain_sensitivity_matrix.png",
        ]
        for fp in expected_files:
            self.assertTrue(fp.exists(), f"Missing expected artifact: {fp}")

        svc = HealthcareResourceDemandService(PROJ)
        payload = svc.get_frontend_integration_payload()
        for key in [
            "resource_overview",
            "bed_availability",
            "demand_forecast",
            "capacity_comparison",
            "scenario_planning",
            "insufficient_data_indicators",
        ]:
            self.assertIn(key, payload)

        # Verify district status for Pune vs Mumbai City
        pune_status = svc.get_district_resource_status("Pune")
        self.assertIsNone(pune_status["empirical_available_beds"])
        self.assertIsNotNone(pune_status["observed_2023_2025_peak_weekly_cases"])

        mumbai_status = svc.get_district_resource_status("Mumbai City")
        self.assertIsNone(mumbai_status["observed_2023_2025_peak_weekly_cases"])

        # Custom scenario for Mumbai City using user_entered_cases works cleanly
        custom_res = svc.run_custom_hypothetical_scenario(
            district_name="Mumbai City",
            case_load_anchor_mode="user_entered_cases",
            user_entered_baseline_cases=80.0,
            hypothetical_case_increase_pct=25.0,
            assumed_admission_proportion=0.15,
        )
        self.assertAlmostEqual(custom_res["hypothetical_projected_weekly_cases"], 100.0)
        self.assertEqual(
            custom_res["output_nature"],
            "HYPOTHETICAL_SCENARIO_NOT_ACTUAL_CAPACITY_OR_FORECAST",
        )
        self.assertEqual(custom_res["mandatory_disclaimer"], MANDATORY_RESOURCE_DISCLAIMER)


if __name__ == "__main__":
    unittest.main()
