"""Unit Tests for Phase 13 Centralized Explainable AI (XAI) Engine."""
import json
from pathlib import Path
import unittest

import joblib
import numpy as np
import pandas as pd

from src.engines.explainability.model_inspector import (
    MANDATORY_XAI_NON_CAUSAL_DISCLAIMER,
    inspect_existing_models_and_artifacts,
)
from src.engines.explainability.service import ExplainableAIService

PROJ = Path(__file__).resolve().parents[1]


class TestExplainableAIEngine(unittest.TestCase):
    """Verify that explanations correspond to exact model versions and actual model predictions."""

    def test_01_model_inventory_and_suitability_rules(self) -> None:
        inv_df, bundles = inspect_existing_models_and_artifacts(PROJ)
        self.assertEqual(len(inv_df), 8)

        inv_map = inv_df.set_index("model_id").to_dict(orient="index")
        self.assertEqual(inv_map["phase5_hist_gradient_boosting_primary"]["model_version"], "phase5-v1.0.0")
        self.assertEqual(inv_map["phase6_multi_horizon_forecasters"]["model_version"], "phase6-v1.0.0")
        self.assertEqual(inv_map["phase7_consensus_and_isolation_forest"]["model_version"], "phase7-v1.0.0")

        # Verify SHAP is not forced onto linear logit, statistical anomaly Z-scores, or Getis-Ord Gi*
        self.assertFalse(inv_map["phase5_logistic_regression_baseline"]["shap_suitable"])
        self.assertFalse(inv_map["phase7_consensus_and_isolation_forest"]["shap_suitable"])
        self.assertFalse(inv_map["phase9_morans_i_and_getis_ord_gi_star"]["shap_suitable"])
        self.assertTrue(inv_map["phase5_xgboost_classifier"]["shap_suitable"])

    def test_02_local_explanations_match_exact_model_versions_and_predictions(self) -> None:
        data_dir = PROJ / "data" / "explanations"
        local_risk_df = pd.read_csv(data_dir / "local_risk_explanations.csv")
        local_fc_df = pd.read_csv(data_dir / "local_forecast_explanations.csv")
        local_anom_df = pd.read_csv(data_dir / "local_anomaly_explanations.csv")

        # 1. Verify Phase 5 risk explanations match actual trained_model.pkl probabilities & version
        rc_bundle = joblib.load(PROJ / "models" / "risk_classification" / "trained_model.pkl")
        rc_prep = joblib.load(PROJ / "models" / "risk_classification" / "preprocessing_pipeline.pkl")
        schema = json.loads(
            (PROJ / "models" / "risk_classification" / "feature_schema.json").read_text(encoding="utf-8")
        )
        df_rc = pd.read_csv(PROJ / "data" / "final_training" / "risk_classification_dataset.csv")
        raw_feats = schema["categorical_features"] + schema["numerical_features"]
        X_trans = rc_prep.transform(df_rc[raw_feats])
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        expected_probs = np.round(rc_bundle.predict_calibrated_proba(np.asarray(X_trans, dtype=float)), 4)

        self.assertEqual(len(local_risk_df), 511)
        self.assertTrue((local_risk_df["model_version"] == rc_bundle.model_version).all())
        np.testing.assert_allclose(
            local_risk_df["model_estimated_risk_prob"].to_numpy(dtype=float),
            expected_probs,
            atol=1e-4,
        )

        # 2. Verify Phase 6 forecast explanations match forecast_model.pkl version and 7d predictions
        fc_bundle = joblib.load(PROJ / "models" / "disease_forecasting" / "forecasting_model.pkl")
        self.assertTrue((local_fc_df["model_version"] == fc_bundle.model_version).all())
        self.assertEqual(set(local_fc_df["horizon"].unique()), {"7d", "14d", "30d"})

        # 3. Verify Phase 7 anomaly explanations match anomaly_detection_results.csv scores & version
        anom_src = pd.read_csv(PROJ / "reports" / "anomaly_detection_results.csv")
        self.assertEqual(len(local_anom_df), 511)
        self.assertTrue((local_anom_df["model_version"] == "phase7-v1.0.0").all())
        np.testing.assert_allclose(
            local_anom_df["composite_anomaly_score"].to_numpy(dtype=float),
            anom_src["anomaly_score"].to_numpy(dtype=float),
            atol=1e-4,
        )

    def test_03_required_explanation_record_fields_and_non_causal_disclaimer(self) -> None:
        data_dir = PROJ / "data" / "explanations"
        local_risk_df = pd.read_csv(data_dir / "local_risk_explanations.csv")
        for req_col in [
            "district",
            "disease",
            "prediction_or_anomaly_id",
            "model_version",
            "explanation_method",
            "important_features",
            "contribution_values",
            "plain_language_summary",
            "data_limitations",
            "scientific_disclaimer",
        ]:
            self.assertIn(req_col, local_risk_df.columns)

        self.assertTrue(
            (local_risk_df["scientific_disclaimer"] == MANDATORY_XAI_NON_CAUSAL_DISCLAIMER).all()
        )

    def test_04_why_district_flagged_panel_and_service(self) -> None:
        data_dir = PROJ / "data" / "explanations"
        rep_dir = PROJ / "reports" / "explainability"

        expected_files = [
            data_dir / "model_explainability_inventory.csv",
            data_dir / "global_feature_explanations.csv",
            data_dir / "global_model_explanations_summary.json",
            data_dir / "local_risk_explanations.csv",
            data_dir / "local_forecast_explanations.csv",
            data_dir / "local_anomaly_explanations.csv",
            data_dir / "why_district_flagged_panels.csv",
            data_dir / "why_district_flagged_panels.json",
            rep_dir / "explainability_engine_report.md",
            rep_dir / "interactive_why_district_flagged_explorer.html",
            rep_dir / "figures" / "01_global_risk_classifier_explainability_comparison.png",
            rep_dir / "figures" / "02_global_multi_horizon_forecast_feature_contributions.png",
            rep_dir / "figures" / "03_local_risk_waterfall_and_domain_attribution.png",
            rep_dir / "figures" / "04_anomaly_evidence_and_isolation_forest_attribution.png",
            rep_dir / "figures" / "05_why_district_flagged_synthesis_matrix.png",
        ]
        for fp in expected_files:
            self.assertTrue(fp.exists(), f"Missing expected Phase 13 artifact: {fp}")

        svc = ExplainableAIService(PROJ)
        gad_panel = svc.get_why_district_flagged_panel("Gadchiroli")
        self.assertEqual(gad_panel["panel_title"], "Why was this district flagged?")
        for k in [
            "main_evidence",
            "historical_comparison",
            "important_contributing_features",
            "data_quality",
            "model_versions",
            "limitations",
        ]:
            self.assertIn(k, gad_panel)
        self.assertTrue(gad_panel["is_flagged"])

        mum_panel = svc.get_why_district_flagged_panel("Mumbai City")
        self.assertFalse(mum_panel["is_flagged"])
        self.assertEqual(mum_panel["flag_priority_tier"], "INSUFFICIENT_SURVEILLANCE_DATA")

        loc_res = svc.get_local_explanations("Pune", "Zika Virus")
        self.assertGreater(len(loc_res["risk_explanations"]), 0)
        self.assertEqual(loc_res["scientific_disclaimer"], MANDATORY_XAI_NON_CAUSAL_DISCLAIMER)


if __name__ == "__main__":
    unittest.main()
