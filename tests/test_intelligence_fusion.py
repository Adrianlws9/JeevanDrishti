"""Unit & Governance Tests for Phase 15 Predictive Intelligence Fusion Engine (Prompt 15 Section 9).

Tests all 7 explicit test cases required by Prompt 15 Section 9 plus full artifact verification:
1. Missing engine outputs
2. Conflicting engine results
3. Different reporting periods
4. Unsupported districts
5. Stale model outputs
6. Duplicate records
7. Incomplete data
8. Service layer & generated files verification
"""
from datetime import datetime, timedelta
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.engines.intelligence_fusion import (
    CANONICAL_SIGNAL_STATUSES,
    FUSION_ENGINE_VERSION,
    PredictiveIntelligenceFusionService,
    check_reporting_period_compatibility,
    deduplicate_engine_dataframe,
    evaluate_model_and_data_freshness,
    fuse_single_aligned_record,
    inspect_and_load_upstream_engines,
    validate_district_and_disease,
)

PROJ = Path(__file__).resolve().parents[1]


class TestPredictiveIntelligenceFusionEngine(unittest.TestCase):
    """Comprehensive verification suite for Phase 15 Central Predictive Intelligence Fusion Engine."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.audit_df, cls.loaded_bundle = inspect_and_load_upstream_engines(PROJ)
        cls.service = PredictiveIntelligenceFusionService(PROJ)

    def test_01_missing_engine_outputs_and_insufficient_data(self) -> None:
        """Section 9.1: Verify missing primary engine outputs produce 'Insufficient data' without fake values."""
        payload = {
            "district_name": "Mumbai City",
            "disease": "Dengue",
            "reporting_period": "2024-W25",
            "model_estimated_risk_prob": np.nan,
            "forecast_7d_cases": np.nan,
            "composite_anomaly_score": np.nan,
            "verified_cases": np.nan,
            "verified_deaths": np.nan,
        }
        rec = fuse_single_aligned_record(payload)
        self.assertEqual(rec["overall_signal_status"], "Insufficient data")
        self.assertEqual(rec["recommended_review_priority"], "PRIORITY_DATA_GAP_AUDIT")
        self.assertFalse(rec["automated_outbreak_declaration"])
        self.assertFalse(rec["risk_classification_output"]["available"])
        self.assertFalse(rec["forecast_output"]["available"])
        self.assertFalse(rec["anomaly_output"]["available"])
        self.assertFalse(rec["severity_indicators"]["available"])

        # Also verify Mumbai City and Ratnagiri in the 36-district pooled intelligence
        mc_pool = self.service.get_district_pooled_intelligence("Mumbai City")
        rt_pool = self.service.get_district_pooled_intelligence("Ratnagiri")
        self.assertIn(mc_pool["overall_signal_status"], CANONICAL_SIGNAL_STATUSES)
        self.assertIn(rt_pool["overall_signal_status"], CANONICAL_SIGNAL_STATUSES)

    def test_02_conflicting_engine_results_preserved(self) -> None:
        """Section 9.2 & Section 6: Verify conflicting engine results preserve both outputs and flag disagreement."""
        # Case A: Anomaly detector flags high score (0.82), while Risk (0.18) and Forecaster (5.0 vs 12.0 obs) do not
        conflict_payload = {
            "district_name": "Pune",
            "disease": "Dengue",
            "reporting_period": "2024-W30",
            "verified_cases": 12.0,
            "verified_deaths": 0.0,
            "model_estimated_risk_prob": 0.18,
            "risk_category": "Low",
            "forecast_7d_cases": 6.5,
            "is_anomaly": True,
            "composite_anomaly_score": 0.82,
            "anomaly_severity_tier": "ELEVATED_ANOMALY_SIGNAL",
            "signal_types": "SEASONAL_SPIKE",
        }
        rec = fuse_single_aligned_record(conflict_payload)
        self.assertEqual(rec["overall_signal_status"], "Conflicting indicators")
        self.assertTrue(rec["disagreement_flag"])
        self.assertEqual(rec["disagreement_type"], "ANOMALY_VS_PREDICTIVE_DISAGREEMENT")
        self.assertEqual(rec["recommended_review_priority"], "PRIORITY_2_TARGETED_ANALYST_REVIEW")
        # Both outputs preserved intact
        self.assertAlmostEqual(
            rec["risk_classification_output"]["calibrated_risk_probability"], 0.18, places=3
        )
        self.assertAlmostEqual(
            rec["anomaly_output"]["composite_anomaly_score"], 0.82, places=3
        )
        self.assertGreater(len(rec["conflicting_evidence"]), 0)

        # Case B: Multiple indicators aligned (Risk + Forecast + Anomaly + Mortality all elevated)
        aligned_payload = {
            "district_name": "Thane",
            "disease": "Dengue",
            "reporting_period": "2024-W35",
            "verified_cases": 42.0,
            "verified_deaths": 2.0,
            "model_estimated_risk_prob": 0.34,
            "risk_category": "High",
            "forecast_7d_cases": 58.0,
            "is_anomaly": True,
            "composite_anomaly_score": 0.88,
            "anomaly_severity_tier": "HIGH_PRIORITY_INVESTIGATION_SIGNAL",
        }
        rec_aligned = fuse_single_aligned_record(aligned_payload)
        self.assertEqual(rec_aligned["overall_signal_status"], "Multiple indicators aligned")
        self.assertEqual(
            rec_aligned["recommended_review_priority"],
            "PRIORITY_1_IMMEDIATE_EPIDEMIOLOGICAL_REVIEW",
        )
        self.assertFalse(rec_aligned["automated_outbreak_declaration"])

    def test_03_different_reporting_periods_blocked(self) -> None:
        """Section 9.3 & Section 3: Verify incompatible reporting periods are rejected and never silently merged."""
        ok_exact, _ = check_reporting_period_compatibility("2024-W10", "2024-W10")
        self.assertTrue(ok_exact)

        ok_mismatch, reason = check_reporting_period_compatibility("2024-W10", "2024-W14")
        self.assertFalse(ok_mismatch)
        self.assertIn("INCOMPATIBLE_REPORTING_PERIODS_BLOCKED", reason)

        # Pooled context allowed only when explicitly flagged as cross-sectional context
        ok_pool_blocked, _ = check_reporting_period_compatibility(
            "2024-W10", "2023-2025_POOL", allow_pooled_context_attachment=False
        )
        self.assertFalse(ok_pool_blocked)
        ok_pool_allowed, mode = check_reporting_period_compatibility(
            "2024-W10", "2023-2025_POOL", allow_pooled_context_attachment=True
        )
        self.assertTrue(ok_pool_allowed)
        self.assertEqual(mode, "CROSS_SECTIONAL_POOL_CONTEXT_ONLY")

        # Verify service blocks mismatched weekly engine periods
        with self.assertRaises(ValueError):
            self.service.fuse_custom_aligned_payload(
                {
                    "district_name": "Pune",
                    "disease": "Dengue",
                    "reporting_period": "2024-W10",
                    "engine_reporting_periods": {
                        "risk_classification": "2024-W10",
                        "anomaly_detection": "2024-W12",  # mismatched week!
                    },
                }
            )

    def test_04_unsupported_districts_rejected_and_aliases_resolved(self) -> None:
        """Section 9.4: Verify unsupported districts raise ValueError while historical Maharashtra aliases resolve."""
        with self.assertRaises(ValueError):
            validate_district_and_disease("Bengaluru Urban", "Dengue")
        with self.assertRaises(ValueError):
            validate_district_and_disease("New Delhi", "Malaria")

        dist_id, std_dist, std_dis = validate_district_and_disease("Aurangabad", "Dengue")
        self.assertEqual(std_dist, "Chhatrapati Sambhajinagar")
        self.assertEqual(dist_id, "MH_D08")
        self.assertEqual(std_dis, "Dengue")

        dist_id2, std_dist2, _ = validate_district_and_disease("Osmanabad", "Cholera")
        self.assertEqual(std_dist2, "Dharashiv")
        self.assertEqual(dist_id2, "MH_D09")

    def test_05_stale_model_outputs_flagged(self) -> None:
        """Section 9.5: Verify stale model versions or timestamps trigger STALE_MODEL_OR_DATA_WARNING."""
        now_dt = datetime(2026, 10, 31, 12, 0, 0)
        old_ts = (now_dt - timedelta(days=90)).isoformat()

        stale_payload = {
            "district_name": "Nagpur",
            "disease": "Malaria",
            "reporting_period": "2024-W20",
            "verified_cases": 25.0,
            "verified_deaths": 0.0,
            "model_estimated_risk_prob": 0.22,
            "risk_category": "Moderate",
            "risk_model_version": "phase5-v0.1.0-deprecated",
            "forecast_7d_cases": 22.0,
            "is_anomaly": False,
            "composite_anomaly_score": 0.21,
            "artifact_timestamp_iso": old_ts,
        }
        rec = fuse_single_aligned_record(
            stale_payload, reference_now=now_dt, max_staleness_days=30
        )
        self.assertEqual(rec["model_freshness_status"], "STALE_MODEL_OR_DATA_WARNING")
        self.assertEqual(
            rec["data_quality_status"], "STALE_MODEL_WARNING_CHECK_VERSIONS"
        )
        self.assertTrue(
            any("STALE" in lim for lim in rec["uncertainty_and_limitations"])
        )

    def test_06_duplicate_records_deduplicated(self) -> None:
        """Section 9.6: Verify duplicate engine records on (district, disease, reporting_period) are deduplicated."""
        dup_df = pd.DataFrame(
            [
                {"district": "Pune", "disease": "Dengue", "reporting_period": "2024-W10", "val": 10},
                {"district": "Pune", "disease": "Dengue", "reporting_period": "2024-W10", "val": 10},
                {"district": "Nashik", "disease": "Dengue", "reporting_period": "2024-W10", "val": 5},
            ]
        )
        cleaned, dup_cnt = deduplicate_engine_dataframe(
            dup_df, ["district", "disease", "reporting_period"], "test_engine"
        )
        self.assertEqual(dup_cnt, 1)
        self.assertEqual(len(cleaned), 2)

    def test_07_incomplete_data_and_common_schema_traceability(self) -> None:
        """Section 9.7 & Sections 2, 7: Verify incomplete data (2/4 primary pillars, missing lags) and traceability."""
        partial_payload = {
            "district_name": "Kolhapur",
            "disease": "Hepatitis A",
            "reporting_period": "2024-W18",
            "verified_cases": 35.0,
            "verified_deaths": 0.0,
            "model_estimated_risk_prob": 0.29,
            "risk_category": "High",
            "risk_record_id": "RISK_MH_D16_2024-W18_999",
            "forecast_7d_cases": np.nan,  # Missing forecast output
            "is_anomaly": np.nan,         # Missing anomaly output
            "composite_anomaly_score": np.nan,
            "missing_features_count": 38,
        }
        rec = fuse_single_aligned_record(partial_payload)
        self.assertEqual(rec["primary_pillars_available_count"], 2)
        self.assertEqual(rec["overall_signal_status"], "Elevated signal")
        self.assertIn("RISK_MH_D16_2024-W18_999", rec["why_this_result_explanation"])
        self.assertTrue(
            any("38 historical lag/climate features" in u for u in rec["uncertainty_and_limitations"])
        )
        self.assertIsNone(rec["resource_indicators"]["empirical_available_beds"])
        self.assertIsNone(rec["severity_indicators"]["clinical_hospital_admissions_count"])

    def test_08_generated_artifacts_and_service_queries(self) -> None:
        """Section 8 & 10: Verify all 9 upstream engines are integrated and all output files exist."""
        self.assertEqual(len(self.audit_df), 9)
        self.assertTrue(self.audit_df["implemented_and_verified"].all())

        # Query service
        pune_dengue = self.service.query_weekly_fused_intelligence(
            district="Pune", disease="Dengue"
        )
        self.assertGreater(len(pune_dengue), 0)

        conflicts = self.service.query_weekly_fused_intelligence(disagreement_only=True)
        self.assertGreater(len(conflicts), 0)

        data_dir = PROJ / "data" / "fused_intelligence"
        rep_dir = PROJ / "reports" / "fusion"
        fig_dir = rep_dir / "figures"

        expected_files = [
            data_dir / "upstream_engines_inspection_audit.csv",
            data_dir / "fused_weekly_district_disease_intelligence.csv",
            data_dir / "fused_district_pooled_intelligence_36.csv",
            data_dir / "engine_disagreement_review_queue.csv",
            data_dir / "fused_intelligence_common_schema_payload.json",
            rep_dir / "interactive_predictive_intelligence_fusion_explorer.html",
            rep_dir / "predictive_intelligence_fusion_report.md",
            fig_dir / "01_upstream_engines_and_reliability_hierarchy_audit.png",
            fig_dir / "02_weekly_fused_signal_status_and_priority_distribution.png",
            fig_dir / "03_maharashtra_36_district_fused_intelligence_map_and_matrix.png",
            fig_dir / "04_engine_disagreement_and_conflict_analysis.png",
            fig_dir / "05_traceable_why_this_result_decomposition_cards.png",
        ]
        for p in expected_files:
            self.assertTrue(p.exists(), f"Missing artifact: {p}")
            self.assertGreater(p.stat().st_size, 500, f"Artifact unexpectedly small: {p}")


if __name__ == "__main__":
    unittest.main()
