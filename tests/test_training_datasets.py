"""Automated unit & integration tests for JEEVAN-DRISHTI Phase 4 Final Training Datasets.

Run:
    python -m unittest discover -s tests -v
"""
import json
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training_datasets.build_training_datasets import (  # noqa: E402
    run_phase4_pipeline,
    FINAL_TRAINING_DIR,
    REPORTS_FT_DIR,
    DATA_REPORTS_DIR,
    REGISTRY_PATH,
    CONTEMPORANEOUS_CLIMATE_COLS_T,
)


class TestPhase4TrainingDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = run_phase4_pipeline(verbose=False)
        cls.fc = cls.res["forecasting"]
        cls.rc = cls.res["risk_classification"]
        cls.ad = cls.res["anomaly_detection"]
        cls.val = cls.res["validation"]
        cls.manifests = cls.res["manifests"]

    def test_all_phase4_validation_checks_pass(self):
        self.assertTrue(self.val["passed"].all(), self.val[~self.val["passed"]].to_string())
        self.assertTrue((self.val["status"] == "PASS").all())

    def test_chronological_splits_70_15_15(self):
        for df in (self.fc, self.rc, self.ad):
            vc = df["split"].value_counts()
            self.assertEqual(int(vc["train"]), 356)
            self.assertEqual(int(vc["validation"]), 77)
            self.assertEqual(int(vc["test"]), 78)
            self.assertEqual(int(df.loc[df["split"] == "train", "period_idx"].max()), 84)
            self.assertEqual(int(df.loc[df["split"] == "validation", "period_idx"].min()), 85)
            self.assertEqual(int(df.loc[df["split"] == "validation", "period_idx"].max()), 103)
            self.assertEqual(int(df.loc[df["split"] == "test", "period_idx"].min()), 104)

    def test_no_contemporaneous_weather_in_forecasting_or_risk(self):
        self.assertEqual(len(set(self.fc.columns) & set(CONTEMPORANEOUS_CLIMATE_COLS_T)), 0)
        self.assertEqual(len(set(self.rc.columns) & set(CONTEMPORANEOUS_CLIMATE_COLS_T)), 0)

    def test_thresholds_fitted_solely_on_training_split(self):
        tr_fc = self.fc[self.fc["split"] == "train"]
        tr_rc = self.rc[self.rc["split"] == "train"]
        for dis, grp in tr_fc.groupby("disease_std"):
            rc_row = tr_rc[tr_rc["disease_std"] == dis].iloc[0]
            if len(grp) >= 5:
                self.assertAlmostEqual(float(grp["target_cases_t"].median()), float(rc_row["train_disease_p50_threshold"]), places=6)
                self.assertAlmostEqual(float(grp["target_cases_t"].quantile(0.75)), float(rc_row["train_disease_p75_threshold"]), places=6)
            else:
                self.assertTrue(pd.isna(rc_row["train_disease_p75_threshold"]))

    def test_required_files_and_manifests_exist(self):
        for f in (
            FINAL_TRAINING_DIR / "forecasting_dataset.csv",
            FINAL_TRAINING_DIR / "risk_classification_dataset.csv",
            FINAL_TRAINING_DIR / "anomaly_detection_dataset.csv",
            FINAL_TRAINING_DIR / "training_manifest.json",
            FINAL_TRAINING_DIR / "validation_manifest.json",
            FINAL_TRAINING_DIR / "testing_manifest.json",
            REPORTS_FT_DIR / "training_dataset_summary.md",
            REPORTS_FT_DIR / "target_definition_report.md",
            REPORTS_FT_DIR / "split_validation_report.md",
            REPORTS_FT_DIR / "modelling_readiness_report.csv",
            DATA_REPORTS_DIR / "training_dataset_summary.md",
            DATA_REPORTS_DIR / "target_definition_report.md",
            DATA_REPORTS_DIR / "split_validation_report.md",
            DATA_REPORTS_DIR / "modelling_readiness_report.csv",
        ):
            self.assertTrue(f.exists(), f"Missing required Phase 4 file: {f}")

        reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        self.assertIn("final_training_datasets", reg)
        self.assertIn("chronological_splits", reg)
        self.assertEqual(len(reg["final_training_datasets"]), 3)


if __name__ == "__main__":
    unittest.main()
