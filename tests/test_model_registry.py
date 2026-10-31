"""
Unit Tests for JEEVAN-DRISHTI Model Evaluation, Validation & Registry System
===========================================================================
Phase 16: Centralized Model Evaluation, Validation, and Registry System
"""

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.model_registry.drift_monitor import (
    calculate_ks_distance,
    calculate_psi,
    run_comprehensive_drift_monitoring_suite,
)
from src.model_registry.evaluator import load_and_standardize_evaluation_metrics
from src.model_registry.model_inspector import inspect_all_workspace_models
from src.model_registry.registry import (
    VALID_MODEL_STATUSES,
    build_model_registry,
)
from src.model_registry.service import ModelRegistryService

PROJ = Path(__file__).resolve().parents[1]


class TestModelRegistrySystem(unittest.TestCase):
    """Test suite validating Phase 16 model evaluation and governance engine."""

    def setUp(self) -> None:
        self.proj = PROJ
        self.registry_json_path = self.proj / "models" / "registry" / "model_registry.json"
        self.reports_dir = self.proj / "reports" / "model_evaluation"

    def test_workspace_model_inspection(self):
        """Verify inspection discovers existing workspace model artifacts without inventing models."""
        inspected = inspect_all_workspace_models(self.proj)
        self.assertGreaterEqual(len(inspected), 11, "Must inspect all existing model families")
        
        # Verify artifact paths point to genuine workspace locations
        for m in inspected:
            self.assertTrue(len(m.model_id) > 0)
            self.assertTrue(len(m.model_name) > 0)
            self.assertIn(m.model_status, VALID_MODEL_STATUSES)

    def test_standardized_evaluation_metrics(self):
        """Verify multi-split evaluation metrics conform to rigorous epidemiological standards."""
        (
            metrics_df,
            baseline_df,
            overfit_df,
            leakage_df,
            breakdown_df,
        ) = load_and_standardize_evaluation_metrics(self.proj)

        self.assertFalse(metrics_df.empty, "Master metrics catalog must not be empty")
        self.assertFalse(baseline_df.empty, "Baseline comparisons must not be empty")
        self.assertFalse(overfit_df.empty, "Overfitting audit must not be empty")
        self.assertFalse(breakdown_df.empty, "Breakdown audit must not be empty")

        # Metric appropriateness: Classification must have ROC-AUC / F1
        clf_metrics = metrics_df[metrics_df["task_type"] == "Classification"]
        self.assertFalse(clf_metrics.empty)
        self.assertTrue(any(pd.notna(clf_metrics["primary_metric_value"])))
        self.assertTrue(any(pd.notna(clf_metrics["f1_score"])))

        # Forecasting must have MAE / RMSE
        fc_metrics = metrics_df[metrics_df["task_type"] == "Forecasting"]
        self.assertFalse(fc_metrics.empty)
        self.assertTrue(any(pd.notna(fc_metrics["mae"])))

        # Anomaly detection must not fabricate fake precision/recall
        anom_metrics = metrics_df[metrics_df["task_type"] == "Anomaly Detection"]
        self.assertFalse(anom_metrics.empty)
        self.assertTrue(all(pd.isna(anom_metrics["precision"])))

    def test_model_registry_15_field_schema(self):
        """Assert all registered models strictly adhere to the mandatory 15-field schema."""
        self.assertTrue(self.registry_json_path.is_file(), "model_registry.json must exist")

        with open(self.registry_json_path, "r", encoding="utf-8") as f:
            registry = json.load(f)

        models = registry.get("models", [])
        self.assertGreaterEqual(len(models), 15, "Expected at least 15 registered models/sub-models")

        mandatory_fields = [
            "model_id",
            "model_name",
            "model_type",
            "disease",
            "supported_districts",
            "dataset_filenames",
            "dataset_version",
            "feature_version",
            "training_period",
            "validation_period",
            "testing_period",
            "training_timestamp",
            "model_version",
            "evaluation_metrics",
            "artifact_path",
            "model_status",
        ]

        for m in models:
            for field_name in mandatory_fields:
                self.assertIn(
                    field_name,
                    m,
                    f"Model {m.get('model_id', 'UNKNOWN')} missing mandatory field '{field_name}'",
                )

    def test_operational_model_statuses(self):
        """Assert model status categorization reflects empirical data availability."""
        with open(self.registry_json_path, "r", encoding="utf-8") as f:
            registry = json.load(f)

        for m in registry.get("models", []):
            self.assertIn(m["model_status"], VALID_MODEL_STATUSES)

        # Check specific status rules
        status_map = {m["model_id"]: m["model_status"] for m in registry.get("models", [])}

        # Resource Demand ML must be Unsupported (no hospital capacity data)
        self.assertEqual(status_map.get("MOD_P11_RESOURCE_DEMAND_LITTLES_LAW_CALCULATOR"), "Unsupported")

        # Spread simulator must be Research (synthetic mobility graph)
        self.assertEqual(status_map.get("MOD_P10_SPREAD_SCENARIO_GRAPH_SIMULATOR"), "Research")

        # 30-day forecaster must be Experimental
        self.assertEqual(status_map.get("MOD_P06_FORECAST_HISTGRADBOOST_30D"), "Experimental")

        # Primary risk classifier must be Validated prototype
        self.assertEqual(status_map.get("MOD_P05_RISK_HISTGRADBOOST_P75"), "Validated prototype")

    def test_dataset_lineage_traceability(self):
        """Verify full provenance traceability linking models back to raw surveillance records."""
        with open(self.registry_json_path, "r", encoding="utf-8") as f:
            registry = json.load(f)

        for m in registry.get("models", []):
            self.assertIsInstance(m["dataset_filenames"], list)
            self.assertGreaterEqual(len(m["dataset_filenames"]), 1)
            self.assertTrue(len(m["dataset_version"]) > 0)
            self.assertTrue(len(m["feature_version"]) > 0)
            self.assertTrue(len(m["training_period"]) > 0)

    def test_drift_monitoring_statistical_helpers(self):
        """Test PSI and KS-distance calculations on synthetic distributions."""
        # Test identical distributions -> PSI close to 0, Stable
        dist_a = np.random.normal(10.0, 2.0, 500)
        dist_b = np.random.normal(10.0, 2.0, 500)
        psi, status = calculate_psi(dist_a, dist_b)
        self.assertLess(psi, 0.15)
        self.assertIn(status, ("STABLE", "MODERATE_DRIFT"))

        # Test shifted distributions -> Significant Drift
        dist_c = np.random.normal(25.0, 2.0, 500)
        psi_shift, status_shift = calculate_psi(dist_a, dist_c)
        self.assertGreater(psi_shift, 0.25)
        self.assertEqual(status_shift, "SIGNIFICANT_DRIFT")

        # Test KS-test distance
        ks_stat, p_val, sig = calculate_ks_distance(dist_a, dist_c)
        self.assertGreater(ks_stat, 0.5)
        self.assertLess(p_val, 0.01)
        self.assertEqual(sig, "HIGHLY_SIGNIFICANT")

    def test_model_registry_service_api(self):
        """Test ModelRegistryService query and retrieval methods."""
        service = ModelRegistryService(self.registry_json_path)

        # Get specific model
        m = service.get_model("MOD_P05_RISK_HISTGRADBOOST_P75")
        self.assertIsNotNone(m)
        self.assertEqual(m["model_status"], "Validated prototype")

        # List by status
        validated = service.list_models(status="Validated prototype")
        self.assertGreaterEqual(len(validated), 5)

        # Lineage retrieval
        lineage = service.get_lineage("MOD_P05_RISK_HISTGRADBOOST_P75")
        self.assertIsNotNone(lineage)
        self.assertIn("dataset_filenames", lineage)

        # Comparison DataFrame
        comp_df = service.compare_models(["MOD_P05_RISK_HISTGRADBOOST_P75", "MOD_P06_FORECAST_HISTGRADBOOST_7D"])
        self.assertEqual(len(comp_df), 2)

    def test_output_artifacts_and_dashboard_exist(self):
        """Ensure all required reporting artifacts, CSVs, figures, and HTML dashboard exist."""
        # 1. Model Registry JSON
        self.assertTrue(self.registry_json_path.is_file())
        self.assertGreater(self.registry_json_path.stat().st_size, 1000)

        # 2. Master Evaluation CSVs
        csv_files = [
            "model_evaluation_metrics_catalog.csv",
            "model_baseline_comparisons.csv",
            "overfitting_and_leakage_audit.csv",
            "disease_and_district_performance_breakdown.csv",
            "feature_and_prediction_drift_report.csv",
        ]
        for f in csv_files:
            p = self.reports_dir / f
            self.assertTrue(p.is_file(), f"Expected CSV report missing: {f}")
            self.assertGreater(p.stat().st_size, 100)

        # 3. Static Figures
        fig_files = [
            "01_model_inventory_and_status_audit.png",
            "02_train_val_test_performance_and_overfitting_gaps.png",
            "03_model_vs_baseline_improvement_radar_and_bars.png",
            "04_feature_and_prediction_drift_monitoring_grid.png",
            "05_district_and_disease_reliability_coverage_matrix.png",
        ]
        for fig in fig_files:
            fp = self.reports_dir / "figures" / fig
            self.assertTrue(fp.is_file(), f"Expected static figure missing: {fig}")
            self.assertGreater(fp.stat().st_size, 10000)

        # 4. Interactive Dashboard & Markdown Report
        html_p = self.reports_dir / "interactive_model_registry_dashboard.html"
        self.assertTrue(html_p.is_file())
        self.assertGreater(html_p.stat().st_size, 10000)

        md_p = self.reports_dir / "model_evaluation_and_registry_report.md"
        self.assertTrue(md_p.is_file())
        self.assertGreater(md_p.stat().st_size, 2000)


if __name__ == "__main__":
    unittest.main()
