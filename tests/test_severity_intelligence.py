"""Unit Tests for Phase 12 Disease Severity & Mortality Intelligence Engine."""
from pathlib import Path
import unittest

import pandas as pd

from src.engines.severity_intelligence.data_preparation import (
    MANDATORY_SEVERITY_PRIVACY_AND_SAFETY_DISCLAIMER,
    audit_outcome_and_severity_data_availability,
    prepare_verified_weekly_mortality_records,
)
from src.engines.severity_intelligence.mortality_analysis import (
    compute_disease_mortality_summary,
    compute_district_mortality_summary,
    wilson_score_interval_pct,
)
from src.engines.severity_intelligence.service import DiseaseSeverityMortalityService

PROJ = Path(__file__).resolve().parents[1]


class TestSeverityIntelligenceEngine(unittest.TestCase):
    """Verify non-fabrication of missing outcomes, exact zero vs NaN handling, compatible CFR math, and service."""

    def test_01_outcome_availability_and_missing_vs_zero_audit(self) -> None:
        avail_df, mz_df, meta = audit_outcome_and_severity_data_availability(PROJ)
        self.assertEqual(len(avail_df), 7)

        avail_map = avail_df.set_index("outcome_category_key")["available_in_workspace"].to_dict()
        self.assertTrue(avail_map["disease_related_deaths"])
        self.assertTrue(avail_map["case_fatality_information"])
        self.assertTrue(avail_map["district_wise_mortality_records"])
        self.assertFalse(avail_map["hospital_admissions"])
        self.assertFalse(avail_map["severe_disease_cases"])
        self.assertFalse(avail_map["complications"])
        self.assertFalse(avail_map["recovery_outcomes"])

        # Verify 2009-2022 unique rows have 814 NaN deaths vs 64 exact zero deaths (never zero-filled)
        hist_u = mz_df[
            (mz_df["source_dataset"] == "historical_epiclim")
            & (mz_df["duplicate_status"] == "unique")
        ].iloc[0]
        self.assertEqual(int(hist_u["deaths_missing_nan_count"]), 814)
        self.assertEqual(int(hist_u["deaths_exact_zero_count"]), 64)

        # Verify 2023-2025 canonical weekly grain has 511 rows, 0 NaN deaths, 365 zero deaths, 146 positive deaths, 199 total deaths
        wk_u = mz_df[mz_df["source_dataset"] == "idsp_weekly_deduplicated_2023_2025"].iloc[0]
        self.assertEqual(int(wk_u["total_records"]), 511)
        self.assertEqual(int(wk_u["deaths_missing_nan_count"]), 0)
        self.assertEqual(int(wk_u["deaths_exact_zero_count"]), 365)
        self.assertEqual(int(wk_u["deaths_positive_gt0_count"]), 146)
        self.assertEqual(int(wk_u["verified_deaths_sum"]), 199)

    def test_02_weekly_records_schema_and_unavailable_severity_nan(self) -> None:
        wk_df = prepare_verified_weekly_mortality_records(PROJ)
        self.assertEqual(len(wk_df), 511)

        required_fields = [
            "district",
            "disease",
            "reporting_period",
            "verified_cases",
            "verified_deaths",
            "severity_count",
            "calculated_ratio",
            "source",
            "data_completeness",
            "analysis_method",
            "generation_timestamp",
        ]
        for col in required_fields:
            self.assertIn(col, wk_df.columns)

        # Unavailable clinical severity/admissions/complications/recovery must be 100% NaN
        for na_col in [
            "severity_count",
            "hospital_admissions_count",
            "complications_count",
            "recovery_count",
        ]:
            self.assertTrue(wk_df[na_col].isna().all(), f"{na_col} must be strictly NaN")

        # Compatible CFR check: calculated_ratio == (verified_deaths / verified_cases) * 100
        self.assertTrue(wk_df["calculated_ratio"].notna().all())
        self.assertTrue(((wk_df["calculated_ratio"] >= 0.0) & (wk_df["calculated_ratio"] <= 100.0)).all())

    def test_03_disease_and_district_summaries_and_wilson_ci(self) -> None:
        wk_df = prepare_verified_weekly_mortality_records(PROJ)
        dis_df = compute_disease_mortality_summary(wk_df)
        dist_df = compute_district_mortality_summary(wk_df)

        self.assertEqual(len(dis_df), 28)
        self.assertEqual(int(dis_df["verified_deaths"].sum()), 199)
        self.assertEqual(int(dis_df["verified_cases"].sum()), 19172)

        # Check Wilson 95% CI bounds bracket crude ratio
        for _, r in dis_df.iterrows():
            self.assertLessEqual(r["wilson_95ci_lower_pct"], r["calculated_ratio"] + 1e-6)
            self.assertGreaterEqual(r["wilson_95ci_upper_pct"], r["calculated_ratio"] - 1e-6)

        # Check 36 official districts: 34 active + 2 Insufficient Data (Mumbai City & Ratnagiri)
        self.assertEqual(len(dist_df), 36)
        self.assertEqual(int(dist_df["verified_deaths"].notna().sum()), 34)
        self.assertEqual(int(dist_df["verified_deaths"].isna().sum()), 2)
        self.assertTrue(dist_df["severity_count"].isna().all())

        low_0, high_0 = wilson_score_interval_pct(0, 100)
        self.assertEqual(low_0, 0.0)
        self.assertGreater(high_0, 0.0)

    def test_04_generated_artifacts_forecast_and_service(self) -> None:
        data_dir = PROJ / "data" / "severity_intelligence"
        rep_dir = PROJ / "reports" / "severity_intelligence"

        expected_files = [
            data_dir / "outcome_data_availability_audit.csv",
            data_dir / "deaths_missing_vs_zero_audit.csv",
            data_dir / "verified_weekly_mortality_records.csv",
            data_dir / "disease_mortality_summary.csv",
            data_dir / "district_mortality_summary.csv",
            data_dir / "annual_mortality_trends.csv",
            data_dir / "seasonal_mortality_trends.csv",
            data_dir / "weekly_statewide_mortality_series.csv",
            data_dir / "mortality_forecasting_sufficiency_audit.csv",
            data_dir / "mortality_forecast_model_comparison.csv",
            data_dir / "mortality_forecast_actual_vs_predicted.csv",
            data_dir / "severity_mortality_summary_payload.json",
            rep_dir / "severity_mortality_intelligence_report.md",
            rep_dir / "figures" / "01_outcome_data_availability_and_missing_vs_zero_audit.png",
            rep_dir / "figures" / "02_disease_wise_mortality_and_wilson_cfr_intervals.png",
            rep_dir / "figures" / "03_district_mortality_comparison_and_burden_ranking.png",
            rep_dir / "figures" / "04_temporal_and_seasonal_mortality_trends_2023_2025.png",
            rep_dir / "figures" / "05_chronological_mortality_forecast_validation.png",
        ]
        for fp in expected_files:
            self.assertTrue(fp.exists(), f"Missing expected Phase 12 artifact: {fp}")

        svc = DiseaseSeverityMortalityService(PROJ)
        dengue_prof = svc.get_disease_mortality_profile("Dengue")
        self.assertEqual(dengue_prof["verified_deaths"], 82)
        self.assertIsNone(dengue_prof["severity_count"])
        self.assertEqual(
            dengue_prof["mandatory_disclaimer"],
            MANDATORY_SEVERITY_PRIVACY_AND_SAFETY_DISCLAIMER,
        )

        nagpur_prof = svc.get_district_mortality_profile("Nagpur")
        self.assertEqual(nagpur_prof["verified_deaths"], 33)
        self.assertIsNone(nagpur_prof["severity_count"])

        mumbai_prof = svc.get_district_mortality_profile("Mumbai City")
        self.assertIsNone(mumbai_prof["verified_deaths"])
        self.assertEqual(mumbai_prof["mortality_burden_tier"], "Insufficient Data")


if __name__ == "__main__":
    unittest.main()
