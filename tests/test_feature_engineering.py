"""Automated unit & integration tests for the JEEVAN-DRISHTI Phase 3 Feature Engineering Engine (2023-2025 scope).

Run:
    python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.feature_pipeline import (  # noqa: E402
    run_feature_pipeline,
    build_feature_dictionary,
    FEATURES_DIR,
    REPORTS_FE_DIR,
)
from src.features.temporal_features import build_weekly_calendar  # noqa: E402


class TestFeatureEngineering2023To2025(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = run_feature_pipeline(verbose=False)
        cls.wj = cls.res["weekly_joined"]
        cls.cw = cls.res["climate_weekly"]
        cls.val = cls.res["validation"]
        cls.fdict = cls.res["dictionary"]

    def test_strict_2023_2025_scope(self):
        self.assertEqual(sorted(self.wj["report_year"].unique().tolist()), [2023, 2024, 2025])
        self.assertEqual(sorted(self.cw["report_year"].unique().tolist()), [2023, 2024, 2025])
        self.assertTrue((self.wj["period_idx"].between(0, 155)).all())
        self.assertEqual(len(self.cw), 36 * 156)

    def test_all_automated_validation_checks_pass(self):
        self.assertTrue(self.val["passed"].all(), self.val[~self.val["passed"]].to_string())
        self.assertTrue((self.val["status"] == "PASS").all())

    def test_feature_dictionary_covers_every_column(self):
        all_cols = set(self.wj.columns) | set(self.cw.columns)
        dict_cols = set(self.fdict["feature_name"])
        self.assertEqual(all_cols - dict_cols, set())
        self.assertEqual(dict_cols - all_cols, set())

    def test_no_zero_filling_of_missing_disease_lags(self):
        for lag in (1, 2, 4, 8, 52):
            col = f"cases_lag_{lag}p"
            self.assertTrue(self.wj[col].isna().any(), f"{col} should have NaNs where prior week had no outbreak")
            self.assertTrue((self.wj[col].dropna() > 0).all(), f"{col} must never be zero-filled")

    def test_cyclical_and_seasonal_temporal_features(self):
        sin2_cos2_wk = self.wj["week_sin"] ** 2 + self.wj["week_cos"] ** 2
        sin2_cos2_mo = self.wj["month_sin"] ** 2 + self.wj["month_cos"] ** 2
        self.assertTrue(np.allclose(sin2_cos2_wk, 1.0, atol=1e-4))
        self.assertTrue(np.allclose(sin2_cos2_mo, 1.0, atol=1e-4))
        seasons_sum = (
            self.wj["is_winter"]
            + self.wj["is_pre_monsoon_summer"]
            + self.wj["is_monsoon"]
            + self.wj["is_post_monsoon"]
        )
        self.assertTrue((seasons_sum == 1).all())
        self.assertTrue((self.wj["temporal_grain"] == "weekly").all())

    def test_population_denominator_not_fabricated(self):
        for col in ("district_population", "population_density", "cases_per_100k_pop"):
            self.assertTrue(self.wj[col].isna().all(), f"{col} must remain NaN without a valid denominator file")

    def test_climate_gadchiroli_and_mumbai_suburban_rules(self):
        gad = self.cw[self.cw["district_std"] == "Gadchiroli"]
        self.assertEqual(int(gad["rain_mm_sum_t"].notna().sum()), 156)
        self.assertTrue((gad["rain_spatial_source"] == "adjacent_land_cell_19.75N_80.00E").all())
        self.assertTrue(gad["rain_mm_strict_nearest_sum_t"].isna().all())

        ms = self.cw[self.cw["district_std"] == "Mumbai Suburban"]
        self.assertTrue(ms["rain_mm_sum_t"].isna().all())
        self.assertTrue(ms["tavg_c_mean_t"].isna().all())
        self.assertTrue((ms["rain_spatial_source"] == "none_no_coordinates").all())

    def test_required_files_exist(self):
        for f in (
            FEATURES_DIR / "disease_climate_weekly_features.csv",
            FEATURES_DIR / "disease_climate_weekly_features_2023_2025.csv",
            FEATURES_DIR / "district_week_climate_features.csv",
            FEATURES_DIR / "district_week_climate_features_2023_2025.csv",
            REPORTS_FE_DIR / "feature_dictionary.csv",
            REPORTS_FE_DIR / "feature_dictionary.md",
            REPORTS_FE_DIR / "feature_generation_log.csv",
            REPORTS_FE_DIR / "missing_feature_report.csv",
            REPORTS_FE_DIR / "leakage_validation_report.csv",
            REPORTS_FE_DIR / "leakage_validation_report.md",
            REPORTS_FE_DIR / "feature_coverage_report.csv",
            REPORTS_FE_DIR / "feature_engineering_summary.md",
        ):
            self.assertTrue(f.exists(), f"Missing output file: {f}")


if __name__ == "__main__":
    unittest.main()
