"""Automated validation tests for Phase 8: Climate-Disease Intelligence Engine."""
import json
from pathlib import Path
import unittest
import pandas as pd

from src.engines.climate_intelligence import CAUSALITY_DISCLAIMER
from src.engines.climate_intelligence.service import ClimateDiseaseIntelligenceService

PROJ = Path(__file__).resolve().parents[1]
DATA_CI_DIR = PROJ / "data" / "climate_intelligence"
REPORTS_CI_DIR = PROJ / "reports" / "climate_intelligence"
REGISTRY_PATH = PROJ / "model_dataset_registry.json"


class TestPhase8ClimateDiseaseIntelligence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = ClimateDiseaseIntelligenceService()
        cls.dis_corr = pd.read_csv(DATA_CI_DIR / "disease_climate_correlations.csv")
        cls.dist_rel = pd.read_csv(DATA_CI_DIR / "district_climate_relationships.csv")
        cls.lag_df = pd.read_csv(DATA_CI_DIR / "lag_analysis_results.csv")
        cls.reg_df = pd.read_csv(DATA_CI_DIR / "regression_analysis_results.csv")
        cls.imp_df = pd.read_csv(DATA_CI_DIR / "climate_feature_importance_and_shap.csv")
        cls.pdp_df = pd.read_csv(DATA_CI_DIR / "partial_dependence_profiles.csv")

    def test_01_required_files_and_visualizations_exist(self):
        """Verify all Phase 8 data tables, reports, and 7 visualization PNGs exist."""
        for p in [
            DATA_CI_DIR / "aligned_weekly_climate_disease_outbreaks.csv",
            DATA_CI_DIR / "aligned_district_week_surveillance_panel.csv",
            DATA_CI_DIR / "aligned_monthly_climate_disease_panel.csv",
            DATA_CI_DIR / "disease_climate_correlations.csv",
            DATA_CI_DIR / "district_climate_relationships.csv",
            DATA_CI_DIR / "lag_analysis_results.csv",
            DATA_CI_DIR / "seasonal_comparison_results.csv",
            DATA_CI_DIR / "regression_analysis_results.csv",
            DATA_CI_DIR / "climate_feature_importance_and_shap.csv",
            DATA_CI_DIR / "partial_dependence_profiles.csv",
            DATA_CI_DIR / "data_coverage_and_alignment_audit.json",
            REPORTS_CI_DIR / "climate_disease_intelligence_report.md",
            REPORTS_CI_DIR / "figures" / "01_rainfall_vs_disease_trends.png",
            REPORTS_CI_DIR / "figures" / "02_temperature_vs_disease_trends.png",
            REPORTS_CI_DIR / "figures" / "03_humidity_vs_disease_trends_pune.png",
            REPORTS_CI_DIR / "figures" / "04_lagged_climate_disease_relationships.png",
            REPORTS_CI_DIR / "figures" / "05_district_climate_disease_comparison.png",
            REPORTS_CI_DIR / "figures" / "06_climate_disease_correlation_heatmap.png",
            REPORTS_CI_DIR / "figures" / "07_shap_and_partial_dependence_climate.png",
        ]:
            self.assertTrue(p.exists(), f"Missing required Phase 8 file: {p}")
            self.assertGreater(p.stat().st_size, 0, f"Empty Phase 8 file: {p}")

    def test_02_four_tier_non_causal_interpretation_columns_present(self):
        """Ensure every finding distinguishes observed relationship, statistical association, predictive utility, and non-causation."""
        for df in (self.dis_corr, self.dist_rel, self.lag_df):
            for col in (
                "observed_relationship",
                "statistical_association",
                "predictive_usefulness",
                "unsupported_causal_interpretation_warning",
            ):
                self.assertIn(col, df.columns)
            self.assertTrue(
                (df["unsupported_causal_interpretation_warning"] == CAUSALITY_DISCLAIMER).all()
            )

    def test_03_humidity_strictly_restricted_to_pune(self):
        """Verify relative humidity correlations are computed only for Pune (not fabricated statewide)."""
        rh_rows = self.lag_df[self.lag_df["climate_group"] == "Relative_Humidity_Pune_Only"]
        self.assertGreater(len(rh_rows), 0)
        self.assertTrue(rh_rows["disease_or_series"].str.startswith("Pune_").all())

    def test_04_rf_xgb_treeshap_and_pdp_computed(self):
        """Verify RF importance, XGBoost importance, exact TreeSHAP, and PDP curves are present."""
        for col in (
            "rf_impurity_importance",
            "rf_permutation_importance",
            "xgb_gain_importance",
            "mean_abs_treeshap",
            "treeshap_share",
            "shap_feature_direction_corr",
        ):
            self.assertIn(col, self.imp_df.columns)
        self.assertGreater(len(self.pdp_df), 0)

    def test_05_service_queries_and_registry(self):
        """Verify ClimateDiseaseIntelligenceService and model_dataset_registry.json."""
        dengue_prof = self.service.get_disease_climate_profile("Dengue")
        self.assertTrue(dengue_prof["is_supported"])
        self.assertEqual(dengue_prof["causality_disclaimer"], CAUSALITY_DISCLAIMER)

        heat_prof = self.service.get_disease_climate_profile("Heat-related illness")
        self.assertFalse(heat_prof["is_supported"])

        pune_prof = self.service.get_district_climate_profile("Pune")
        self.assertTrue(pune_prof["is_supported"])

        reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
        self.assertIn("phase8_climate_disease_intelligence_engine", reg)


if __name__ == "__main__":
    unittest.main()
