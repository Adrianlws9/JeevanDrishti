"""Automated validation tests for Phase 6: Time-Series Disease Forecasting Engine."""
import json
from pathlib import Path
import unittest
import joblib
import numpy as np
import pandas as pd

from src.disease_forecasting.service import DiseaseForecastingService, forecast_district_disease_cases
from src.disease_forecasting.train_forecasting_models import (
    BASE_EXCLUDE_COLS,
    LAG1_DEPENDENT_COLS,
    LAG2_LAG3_DEPENDENT_COLS,
    select_horizon_features,
)

PROJ = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJ / "models" / "disease_forecasting"
REPORTS_DIR = PROJ / "reports"
DATA_REPORTS_DIR = PROJ / "data" / "reports"
DATASET_PATH = PROJ / "data" / "final_training" / "forecasting_dataset.csv"
REGISTRY_PATH = PROJ / "model_dataset_registry.json"


class TestPhase6DiseaseForecastingEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(DATASET_PATH)
        cls.forecaster = joblib.load(MODELS_DIR / "forecasting_model.pkl")
        cls.preprocessor = joblib.load(MODELS_DIR / "preprocessing_pipeline.pkl")
        with open(MODELS_DIR / "forecast_metadata.json", "r", encoding="utf-8") as f:
            cls.metadata = json.load(f)
        cls.perf_df = pd.read_csv(REPORTS_DIR / "forecasting_performance.csv")
        cls.avp_df = pd.read_csv(REPORTS_DIR / "actual_vs_predicted.csv")
        cls.service = DiseaseForecastingService()

    def test_01_required_files_exist(self):
        """Verify all Phase 6 model artifacts and reports exist."""
        for p in [
            MODELS_DIR / "forecasting_model.pkl",
            MODELS_DIR / "preprocessing_pipeline.pkl",
            MODELS_DIR / "forecast_metadata.json",
            REPORTS_DIR / "forecasting_performance.csv",
            REPORTS_DIR / "forecasting_evaluation.md",
            REPORTS_DIR / "actual_vs_predicted.csv",
            DATA_REPORTS_DIR / "forecasting_performance.csv",
            DATA_REPORTS_DIR / "forecasting_evaluation.md",
            DATA_REPORTS_DIR / "actual_vs_predicted.csv",
        ]:
            self.assertTrue(p.exists(), f"Missing required Phase 6 deliverable: {p}")
            self.assertGreater(p.stat().st_size, 0, f"Empty deliverable file: {p}")

    def test_02_horizon_feature_cutoffs_prevent_leakage(self):
        """Verify 14d (h=2w) excludes lag-1 features and 30d (h=4w) excludes lag-1..3 features."""
        cols_7d, _, _ = select_horizon_features(self.df, "7d")
        cols_14d, _, _ = select_horizon_features(self.df, "14d")
        cols_30d, _, _ = select_horizon_features(self.df, "30d")

        for c in BASE_EXCLUDE_COLS:
            self.assertNotIn(c, cols_7d)
            self.assertNotIn(c, cols_14d)
            self.assertNotIn(c, cols_30d)

        for c in LAG1_DEPENDENT_COLS:
            self.assertNotIn(c, cols_14d, f"Lag-1 feature {c} leaked into 14d (2-week) horizon!")
            self.assertNotIn(c, cols_30d, f"Lag-1 feature {c} leaked into 30d (4-week) horizon!")

        for c in LAG2_LAG3_DEPENDENT_COLS:
            self.assertNotIn(c, cols_30d, f"Lag-2/3 feature {c} leaked into 30d (4-week) horizon!")

    def test_03_all_horizons_models_and_rolling_origin_evaluated(self):
        """Verify 7d, 14d, and 30d horizons, required models, and rolling-origin backtesting exist."""
        self.assertEqual(set(self.perf_df["horizon"].unique()), {"7d", "14d", "30d"})
        self.assertEqual(
            set(self.perf_df["split"].unique()),
            {"train", "validation", "test", "rolling_origin_backtest_4fold"},
        )
        for req_model in [
            "LastValue_Baseline",
            "SeasonalNaive_Baseline",
            "Statsmodels_ETS",
            "Statsmodels_SARIMA",
            "XGBoost_Lagged_Regressor",
            "HistGradientBoosting_Regressor",
            "RandomForest_Lagged_Regressor",
        ]:
            self.assertIn(req_model, set(self.perf_df["model_name"].unique()))

        for col in [
            "mae", "rmse", "pi_coverage_80", "pi_coverage_90",
            "trend_direction_accuracy", "delta_mae_vs_last_value", "delta_mae_vs_seasonal_naive",
        ]:
            self.assertIn(col, self.perf_df.columns)

    def test_04_actual_vs_predicted_not_labelled_as_observed(self):
        """Ensure forecasts are explicitly labelled MODEL_FORECAST_VS_OBSERVED and intervals are valid."""
        self.assertEqual(len(self.avp_df), 511 * 3)
        self.assertTrue((self.avp_df["record_type"] == "MODEL_FORECAST_VS_OBSERVED").all())
        self.assertTrue((self.avp_df["prediction_lower_80"] <= self.avp_df["predicted_cases"]).all())
        self.assertTrue((self.avp_df["predicted_cases"] <= self.avp_df["prediction_upper_80"]).all())
        self.assertTrue((self.avp_df["prediction_lower_90"] <= self.avp_df["prediction_lower_80"]).all())
        self.assertTrue((self.avp_df["prediction_upper_80"] <= self.avp_df["prediction_upper_90"]).all())

    def test_05_forecasting_service_supported_and_unsupported(self):
        """Verify reusable forecasting service returns multi-horizon forecasts and flags unsupported inputs."""
        res = forecast_district_disease_cases("Pune", "Dengue", "2024-W35")
        self.assertTrue(res["is_supported"])
        self.assertEqual(res["value_type"], "MODEL_FORECAST_NOT_OBSERVED")
        for h in ("7d", "14d", "30d"):
            self.assertIn(h, res["horizons"])
            h_obj = res["horizons"][h]
            self.assertGreaterEqual(h_obj["predicted_cases"], 0.0)
            self.assertEqual(len(h_obj["prediction_interval_80"]), 2)
            self.assertEqual(len(h_obj["prediction_interval_90"]), 2)
            self.assertLessEqual(h_obj["prediction_interval_80"][0], h_obj["predicted_cases"])
            self.assertGreaterEqual(h_obj["prediction_interval_80"][1], h_obj["predicted_cases"])

        res_un = forecast_district_disease_cases("Pune", "Heat-related illness", "2025-W20")
        self.assertFalse(res_un["is_supported"])
        self.assertEqual(res_un["value_type"], "UNSUPPORTED_NO_FORECAST")

        res_dist = forecast_district_disease_cases("Mumbai City", "Dengue", "2025-W20")
        self.assertFalse(res_dist["is_supported"])

    def test_06_registry_updated_with_phase6_metadata(self):
        """Verify model_dataset_registry.json contains Phase 6 forecasting section."""
        reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        self.assertIn("phase6_disease_forecasting_engine", reg)
        p6 = reg["phase6_disease_forecasting_engine"]
        self.assertEqual(p6["dataset_used"]["total_rows"], 511)
        self.assertEqual(set(p6["selected_model_by_horizon"].keys()), {"7d", "14d", "30d"})


if __name__ == "__main__":
    unittest.main()
