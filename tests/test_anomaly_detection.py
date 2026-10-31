"""Automated validation tests for Phase 7: Disease Anomaly Detection Engine."""
import json
from pathlib import Path
import unittest
import joblib
import numpy as np
import pandas as pd

from src.anomaly_detection import ANOMALY_DISCLAIMER
from src.anomaly_detection.service import (
    DiseaseAnomalyDetectionService,
    detect_district_disease_anomaly,
)

PROJ = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJ / "models" / "anomaly_detection"
REPORTS_DIR = PROJ / "reports"
DATA_REPORTS_DIR = PROJ / "data" / "reports"
REGISTRY_PATH = PROJ / "model_dataset_registry.json"


class TestPhase7AnomalyDetectionEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = joblib.load(MODELS_DIR / "anomaly_detector.pkl")
        cls.preprocessor = joblib.load(MODELS_DIR / "preprocessing_pipeline.pkl")
        with open(MODELS_DIR / "anomaly_model_metadata.json", "r", encoding="utf-8") as f:
            cls.metadata = json.load(f)
        cls.results_df = pd.read_csv(REPORTS_DIR / "anomaly_detection_results.csv")
        cls.service = DiseaseAnomalyDetectionService()

    def test_01_required_files_exist(self):
        """Verify all Phase 7 artifacts and reports exist."""
        for p in [
            MODELS_DIR / "anomaly_detector.pkl",
            MODELS_DIR / "preprocessing_pipeline.pkl",
            MODELS_DIR / "anomaly_model_metadata.json",
            REPORTS_DIR / "anomaly_detection_results.csv",
            REPORTS_DIR / "anomaly_performance_report.md",
            DATA_REPORTS_DIR / "anomaly_detection_results.csv",
            DATA_REPORTS_DIR / "anomaly_performance_report.md",
        ]:
            self.assertTrue(p.exists(), f"Missing required Phase 7 file: {p}")
            self.assertGreater(p.stat().st_size, 0, f"Empty deliverable file: {p}")

    def test_02_results_schema_and_disclaimer(self):
        """Verify every anomaly record contains required fields and the mandatory disclaimer."""
        self.assertEqual(len(self.results_df), 511)
        for req_col in [
            "district",
            "disease",
            "reporting_period",
            "anomaly_score",
            "historical_baseline",
            "triggering_features",
            "detection_timestamp",
            "method",
            "model_version",
            "interpretation_note",
        ]:
            self.assertIn(req_col, self.results_df.columns)

        self.assertTrue((self.results_df["interpretation_note"] == ANOMALY_DISCLAIMER).all())
        self.assertTrue(((self.results_df["anomaly_score"] >= 0.0) & (self.results_df["anomaly_score"] <= 1.0)).all())

    def test_03_all_four_methods_and_signal_categories_implemented(self):
        """Verify Seasonal, Rolling, Robust MAD Z-score, and Isolation Forest are evaluated without fake precision/recall."""
        eval_summary = self.metadata["method_evaluation_summary"]
        methods_evaluated = {r["method"] for r in eval_summary}
        for req_m in [
            "Seasonal_Baseline_Deviation",
            "Rolling_Statistical_Detector",
            "Robust_MAD_ZScore",
            "Isolation_Forest_Multivariate",
            "Composite_Consensus_Engine",
        ]:
            self.assertIn(req_m, methods_evaluated)

        # Verify precision and recall are NOT invented in evaluation records
        for r in eval_summary:
            self.assertNotIn("precision", r)
            self.assertNotIn("recall", r)
            self.assertIn("false_alert_rate_among_benign_pct", r)
            self.assertIn("median_reporting_delay_days", r)

        # Verify all 4 signal types are detected across the dataset
        all_signals = ";".join(self.results_df["signal_types"].dropna().tolist())
        for sig in [
            "SUDDEN_CASE_INCREASE",
            "UNUSUAL_DISTRICT_PATTERN",
            "UNEXPECTED_SEASONAL_DEVIATION",
            "UNUSUAL_MORTALITY_PATTERN",
        ]:
            self.assertIn(sig, all_signals)

    def test_04_service_supported_and_unsupported_queries(self):
        """Verify reusable anomaly detection service works on live and historical queries."""
        res_high = detect_district_disease_anomaly(
            district="Pune",
            disease="Dengue",
            reporting_period="2025-W30",
            observed_cases=180.0,
            observed_deaths=3.0,
            season="Monsoon",
        )
        self.assertTrue(res_high["is_supported"])
        self.assertTrue(res_high["is_anomaly"])
        self.assertIn("UNUSUAL_MORTALITY_PATTERN", res_high["signal_types"])
        self.assertEqual(res_high["interpretation_note"], ANOMALY_DISCLAIMER)

        res_norm = detect_district_disease_anomaly(
            district="Pune",
            disease="Dengue",
            reporting_period="2025-W30",
            observed_cases=10.0,
            observed_deaths=0.0,
            season="Monsoon",
        )
        self.assertTrue(res_norm["is_supported"])
        self.assertFalse(res_norm["is_anomaly"])

        res_un = detect_district_disease_anomaly(
            district="Pune",
            disease="Heat-related illness",
            reporting_period="2025-W20",
            observed_cases=15.0,
        )
        self.assertFalse(res_un["is_supported"])

    def test_05_registry_updated_with_phase7_metadata(self):
        """Verify model_dataset_registry.json contains Phase 7 anomaly detection section."""
        reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        self.assertIn("phase7_anomaly_detection_engine", reg)
        self.assertEqual(reg["phase7_anomaly_detection_engine"]["disclaimer"], ANOMALY_DISCLAIMER)


if __name__ == "__main__":
    unittest.main()
