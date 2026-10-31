"""
Unit Tests for WHO EWARS-Inspired Threshold Detection Engine
===========================================================
JEEVAN-DRISHTI Public Health Surveillance Platform
"""

import json
import unittest
from pathlib import Path

import pandas as pd

from src.anomaly_detection.ewars_thresholds import (
    EWARS_GOVERNANCE_DISCLAIMER,
    EWARS_MODULE_VERSION,
    RARE_HIGH_CONSEQUENCE_DISEASES,
    WHOEWARSThresholdEngine,
)
from src.anomaly_detection.service import DiseaseAnomalyDetectionService

PROJ = Path(__file__).resolve().parents[1]


class TestWHOEWARSThresholdEngine(unittest.TestCase):
    """Test suite validating WHO EWARS-inspired threshold detection and governance."""

    def setUp(self) -> None:
        self.proj = PROJ
        self.thresh_registry_path = self.proj / "models" / "registry" / "threshold_registry.json"
        self.backtest_csv_path = self.proj / "data" / "reports" / "ewars_threshold_evaluations.csv"
        self.validation_report_path = self.proj / "reports" / "threshold_evaluation" / "ewars_threshold_validation_report.md"

        # Load small sample of surveillance data
        data_path = self.proj / "data" / "reports" / "anomaly_detection_results.csv"
        self.surveillance_df = pd.read_csv(data_path)
        self.train_df = self.surveillance_df[self.surveillance_df["split"] == "train"].copy()

        self.engine = WHOEWARSThresholdEngine()
        self.engine.calibrate(self.train_df)

    def test_ewars_engine_calibration_and_rules(self):
        """Verify calibration creates genuine baselines without fabricating values."""
        self.assertTrue(self.engine._is_calibrated)
        self.assertGreaterEqual(len(self.engine.disease_fixed_thresholds), 10)
        self.assertGreaterEqual(len(self.engine.seasonal_baselines), 20)

        # Check high-consequence diseases have strict low fixed thresholds
        self.assertEqual(self.engine.disease_fixed_thresholds.get("Cholera"), 5)
        self.assertEqual(self.engine.disease_fixed_thresholds.get("Rabies (human)"), 1)
        self.assertEqual(self.engine.disease_fixed_thresholds.get("Japanese Encephalitis"), 1)

    def test_fixed_threshold_detection(self):
        """Test fixed threshold breach for epidemic-prone conditions."""
        alert = self.engine.evaluate_observation(
            district="Pune",
            disease="Cholera",
            reporting_period="2025-W30",
            observed_cases=6.0,  # Threshold is 5
        )
        self.assertEqual(alert.alert_status, "ALERT_CONFIRMED_INVESTIGATION_REQUIRED")
        self.assertIn("Fixed_Disease_Threshold", alert.triggered_methods)
        self.assertTrue(alert.human_verification_required)

    def test_moving_average_threshold_detection(self):
        """Test moving average threshold breach with low prior baseline."""
        # Prior history of 4 weeks with low cases
        prior_cases = [5.0, 6.0, 5.0, 7.0]  # mean ~5.75, std ~0.95 -> th ~ 7.65
        alert = self.engine.evaluate_observation(
            district="Thane",
            disease="Dengue",
            reporting_period="2025-W25",
            observed_cases=18.0,  # Sudden spike well above rolling 4w threshold
            prior_cases_history=prior_cases,
        )
        self.assertIn("Moving_Average_Threshold", alert.triggered_methods)

    def test_consecutive_increase_detection(self):
        """Test detection of accelerating epidemic curve via strictly increasing cases."""
        prior_cases = [10.0, 15.0]  # Week t-2: 10, Week t-1: 15
        alert = self.engine.evaluate_observation(
            district="Nagpur",
            disease="Dengue",
            reporting_period="2025-W32",
            observed_cases=22.0,  # Week t: 22 -> 10 < 15 < 22 (strictly increasing)
            prior_cases_history=prior_cases,
        )
        self.assertTrue(alert.is_consecutive_increase)
        self.assertEqual(alert.consecutive_increase_count, 3)
        self.assertIn("Consecutive_Increase_Surge", alert.triggered_methods)

    def test_lagged_alert_persistence(self):
        """Test persistence flag when prior week was also elevated."""
        alert = self.engine.evaluate_observation(
            district="Nashik",
            disease="Dengue",
            reporting_period="2025-W35",
            observed_cases=35.0,
            prior_cases_history=[15.0, 20.0, 28.0],
            lagged_alerts_history=[True],  # Week t-1 had alert
        )
        self.assertTrue(alert.lagged_persistence)
        self.assertIn("Lagged_Alert_Persistence", alert.triggered_methods)

    def test_threshold_registry_10_field_schema(self):
        """Assert threshold_registry.json adheres strictly to the 10-field schema."""
        self.assertTrue(self.thresh_registry_path.is_file(), "threshold_registry.json must exist")

        with open(self.thresh_registry_path, "r", encoding="utf-8") as f:
            registry = json.load(f)

        thresholds = registry.get("thresholds", [])
        self.assertGreaterEqual(len(thresholds), 15)

        required_10_fields = [
            "threshold_id",
            "disease",
            "district",
            "threshold_methodology",
            "historical_calibration_period",
            "threshold_configuration",
            "validation_results",
            "version",
            "last_calibration_date",
            "approval_status",
        ]

        for entry in thresholds:
            for f_name in required_10_fields:
                self.assertIn(
                    f_name,
                    entry,
                    f"Threshold entry {entry.get('threshold_id')} missing field {f_name}",
                )

    def test_governance_disclaimers(self):
        """Verify governance rules: WHO-inspired, no official certification, verification mandatory."""
        alert = self.engine.evaluate_observation(
            district="Pune",
            disease="Dengue",
            reporting_period="2025-W30",
            observed_cases=30.0,
        )
        self.assertIn("WHO EWARS-Inspired", alert.governance_disclaimer)
        self.assertIn("Not officially endorsed or certified", alert.governance_disclaimer)
        self.assertTrue(alert.human_verification_required)

    def test_anomaly_service_integration(self):
        """Verify DiseaseAnomalyDetectionService integrates EWARS as additional evidence without breaking APIs."""
        service = DiseaseAnomalyDetectionService()

        # Regular evaluation must succeed and include ewars_threshold_results
        res = service.evaluate_observation(
            district="Pune",
            disease="Dengue",
            reporting_period="2025-W32",
            observed_cases=55.0,  # Elevated
            observed_deaths=1.0,
        )
        self.assertIn("ewars_threshold_results", res)
        ewars_res = res["ewars_threshold_results"]
        self.assertIn("alert_status", ewars_res)
        self.assertIn("threshold", ewars_res)
        self.assertIn("deviation", ewars_res)

        # Dedicated EWARS method
        ewars_direct = service.evaluate_ewars_thresholds(
            district="Pune",
            disease="Cholera",
            reporting_period="2025-W32",
            observed_cases=8.0,
        )
        self.assertEqual(ewars_direct["alert_status"], "ALERT_CONFIRMED_INVESTIGATION_REQUIRED")

    def test_artifacts_and_reports_exist(self):
        """Ensure all backtesting CSVs, reports, and registry files exist on disk."""
        self.assertTrue(self.thresh_registry_path.is_file())
        self.assertGreater(self.thresh_registry_path.stat().st_size, 1000)

        self.assertTrue(self.backtest_csv_path.is_file())
        self.assertGreater(self.backtest_csv_path.stat().st_size, 1000)

        self.assertTrue(self.validation_report_path.is_file())
        self.assertGreater(self.validation_report_path.stat().st_size, 2000)


if __name__ == "__main__":
    unittest.main()
