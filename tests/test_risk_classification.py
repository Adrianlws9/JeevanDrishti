"""Automated validation tests for Phase 5: Outbreak Risk Classification Engine."""
import json
from pathlib import Path
import unittest
import joblib
import numpy as np
import pandas as pd

from src.risk_classification.predictor import OutbreakRiskPredictor, predict_district_disease_risk
from src.risk_classification.train_risk_models import (
    EXPLICIT_EXCLUDE_COLS,
    PRIMARY_TARGET,
    select_training_features,
)

PROJ = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJ / "models" / "risk_classification"
REPORTS_DIR = PROJ / "reports"
DATA_REPORTS_DIR = PROJ / "data" / "reports"
DATASET_PATH = PROJ / "data" / "final_training" / "risk_classification_dataset.csv"
REGISTRY_PATH = PROJ / "model_dataset_registry.json"


class TestPhase5RiskClassificationEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(DATASET_PATH)
        cls.model_bundle = joblib.load(MODELS_DIR / "trained_model.pkl")
        cls.preprocessor = joblib.load(MODELS_DIR / "preprocessing_pipeline.pkl")
        with open(MODELS_DIR / "feature_schema.json", "r", encoding="utf-8") as f:
            cls.schema = json.load(f)
        with open(MODELS_DIR / "model_metadata.json", "r", encoding="utf-8") as f:
            cls.metadata = json.load(f)
        cls.perf_df = pd.read_csv(REPORTS_DIR / "risk_model_performance.csv")
        cls.predictor = OutbreakRiskPredictor()

    def test_01_required_files_exist(self):
        """Verify all Phase 5 required model artifacts and evaluation reports exist."""
        for p in [
            MODELS_DIR / "trained_model.pkl",
            MODELS_DIR / "preprocessing_pipeline.pkl",
            MODELS_DIR / "feature_schema.json",
            MODELS_DIR / "model_metadata.json",
            REPORTS_DIR / "risk_model_performance.csv",
            REPORTS_DIR / "risk_model_evaluation.md",
            REPORTS_DIR / "risk_model_card.md",
            DATA_REPORTS_DIR / "risk_model_performance.csv",
            DATA_REPORTS_DIR / "risk_model_evaluation.md",
            DATA_REPORTS_DIR / "risk_model_card.md",
        ]:
            self.assertTrue(p.exists(), f"Missing required Phase 5 deliverable: {p}")
            self.assertGreater(p.stat().st_size, 0, f"Empty deliverable file: {p}")

    def test_02_zero_target_or_future_leakage_in_feature_schema(self):
        """Ensure no target variable, threshold reference, or monotonic year/period counter is in X."""
        input_cols = set(self.schema["categorical_features"] + self.schema["numerical_features"])
        for forbidden in EXPLICIT_EXCLUDE_COLS:
            self.assertNotIn(forbidden, input_cols, f"Forbidden column leaked into input features: {forbidden}")
        for col in input_cols:
            self.assertFalse(col.startswith("target_"), f"Target column leaked: {col}")
            self.assertFalse(col.startswith("train_disease_"), f"Threshold column leaked: {col}")

    def test_03_preprocessing_fitted_strictly_on_train_split(self):
        """Verify preprocessing pipeline median imputer matches the training split medians."""
        selected_cols, cat_cols, num_cols, _ = select_training_features(self.df)
        prim_tr = self.df[(self.df["split"] == "train") & (self.df[PRIMARY_TARGET].notna())]
        expected_medians = prim_tr[num_cols].median().to_numpy()
        num_imputer = self.preprocessor.named_transformers_["numerical"].named_steps["imputer"]
        actual_medians = num_imputer.statistics_
        np.testing.assert_allclose(actual_medians, expected_medians, rtol=1e-6, atol=1e-6)

    def test_04_all_required_algorithms_evaluated_and_compared_to_baseline(self):
        """Verify Logistic Regression baseline, Random Forest, XGBoost, and calibrated ensemble are evaluated."""
        prim_perf = self.perf_df[self.perf_df["target_variable"] == PRIMARY_TARGET]
        models_present = set(prim_perf["model_name"].unique())
        for req_model in [
            "Logistic_Regression_Baseline",
            "Random_Forest_Classifier",
            "XGBoost_Classifier",
            "HistGradientBoosting_Classifier",
            "SoftVoting_Ensemble_Calibrated",
        ]:
            self.assertIn(req_model, models_present)

        for req_metric in [
            "accuracy", "precision", "recall", "f1_score", "pr_auc", "roc_auc",
            "brier_score_calibrated", "ece_calibrated", "tn", "fp", "fn", "tp",
            "delta_pr_auc_vs_baseline",
        ]:
            self.assertIn(req_metric, prim_perf.columns)

    def test_05_predictor_supported_and_unsupported_queries(self):
        """Verify reusable predictor returns all required fields and flags unsupported diseases/districts."""
        res_valid = predict_district_disease_risk("Pune", "Dengue", "2024-W35")
        self.assertTrue(res_valid["is_supported"])
        for key in [
            "district", "disease", "reporting_period", "model_estimated_risk",
            "risk_category", "calibrated_probability", "important_contributing_features",
            "model_version", "source_data_timestamp",
        ]:
            self.assertIn(key, res_valid)
        self.assertIn(res_valid["risk_category"], {"Low", "Moderate", "High"})
        self.assertGreaterEqual(res_valid["calibrated_probability"], 0.0)
        self.assertLessEqual(res_valid["calibrated_probability"], 1.0)
        self.assertGreater(len(res_valid["important_contributing_features"]), 0)

        # Unsupported zero-record disease
        res_heat = predict_district_disease_risk("Pune", "Heat-related illness", "2025-W20")
        self.assertFalse(res_heat["is_supported"])
        self.assertEqual(res_heat["risk_category"], "UNSUPPORTED")

        # Unsupported zero-record district
        res_mc = predict_district_disease_risk("Mumbai City", "Dengue", "2025-W20")
        self.assertFalse(res_mc["is_supported"])
        self.assertEqual(res_mc["risk_category"], "UNSUPPORTED")

        # Unsupported <5 training-obs rare disease for p75 severity
        res_measles = predict_district_disease_risk("Mumbai Suburban", "Measles", "2023-W06")
        self.assertFalse(res_measles["is_supported"])
        self.assertEqual(res_measles["risk_category"], "UNSUPPORTED_INSUFFICIENT_TRAIN_LABELS")

    def test_06_registry_updated_with_phase5_metadata(self):
        """Verify model_dataset_registry.json contains Phase 5 training and evaluation details."""
        reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        self.assertIn("phase5_risk_classification_engine", reg)
        p5 = reg["phase5_risk_classification_engine"]
        self.assertEqual(p5["dataset_used"]["total_rows"], 511)
        self.assertEqual(p5["dataset_used"]["primary_target_valid_rows"], 481)
        self.assertGreater(len(p5["held_out_test_metrics_primary_target"]), 0)


if __name__ == "__main__":
    unittest.main()
