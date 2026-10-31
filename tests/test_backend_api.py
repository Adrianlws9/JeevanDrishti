"""
Unit and Integration Tests for JEEVAN-DRISHTI FastAPI Backend
=============================================================
Phase 17: FastAPI Backend & Analytical Engine Integration
"""

import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from backend.config import settings
from backend.database.connection import get_db_connection, init_db, seed_db
from backend.main import app

PROJ = Path(__file__).resolve().parents[1]


class TestBackendAPI(unittest.TestCase):
    """Integration test suite validating all versioned REST API endpoints."""

    @classmethod
    def setUpClass(cls) -> None:
        init_db()
        seed_db()
        cls.client = TestClient(app)
        cls.auth_header = {"X-Reviewer-Key": settings.REVIEWER_API_KEY}

    def test_01_root_and_overview(self):
        """Test root health check and platform overview endpoints."""
        res_root = self.client.get("/")
        self.assertEqual(res_root.status_code, 200)
        self.assertEqual(res_root.json()["status"], "HEALTHY")

        res_ov = self.client.get("/api/v1/overview")
        self.assertEqual(res_ov.status_code, 200)
        data = res_ov.json()
        self.assertEqual(data["total_active_districts"], 34)
        self.assertIn("Mumbai City", data["unsupported_districts"])
        self.assertEqual(data["metadata"]["prediction_type"], "OBSERVED")

    def test_02_districts_endpoints(self):
        """Test listing and individual district detail endpoints."""
        # List all
        res = self.client.get("/api/v1/districts")
        self.assertEqual(res.status_code, 200)
        districts = res.json()["districts"]
        self.assertEqual(len(districts), 36)

        # Active district (Pune)
        res_pune = self.client.get("/api/v1/districts/Pune")
        self.assertEqual(res_pune.status_code, 200)
        pune_data = res_pune.json()
        self.assertTrue(pune_data["is_supported"])
        self.assertGreater(pune_data["historical_outbreak_weeks"], 0)

        # Documented unsupported district (Mumbai City) -> is_supported=False
        res_mum = self.client.get("/api/v1/districts/Mumbai%20City")
        self.assertEqual(res_mum.status_code, 200)
        mum_data = res_mum.json()
        self.assertFalse(mum_data["is_supported"])
        self.assertIn("Zero historical IDSP outbreak records", mum_data["unsupported_reason"])

        # Non-existent district -> 404
        res_none = self.client.get("/api/v1/districts/Atlantis")
        self.assertEqual(res_none.status_code, 404)

    def test_03_diseases_endpoints(self):
        """Test disease catalog and historical trend endpoints."""
        res = self.client.get("/api/v1/diseases")
        self.assertEqual(res.status_code, 200)
        diseases = res.json()["diseases"]
        self.assertGreaterEqual(len(diseases), 20)

        # Disease trends (Dengue)
        res_dengue = self.client.get("/api/v1/diseases/Dengue/trends")
        self.assertEqual(res_dengue.status_code, 200)
        d_data = res_dengue.json()
        self.assertEqual(d_data["disease_name"], "Dengue")
        self.assertGreater(d_data["total_historical_weeks"], 0)

        # Non-existent disease -> 404
        res_fake = self.client.get("/api/v1/diseases/FakeDisease123/trends")
        self.assertEqual(res_fake.status_code, 404)

    def test_04_risk_intelligence_endpoints(self):
        """Test statewide risk overview and district-specific risk assessment."""
        # Statewide overview
        res_risk = self.client.get("/api/v1/risk?disease=Dengue")
        self.assertEqual(res_risk.status_code, 200)
        r_data = res_risk.json()
        self.assertEqual(len(r_data["district_risks"]), 34)
        self.assertEqual(r_data["metadata"]["prediction_type"], "RISK_CLASSIFICATION")

        # District detail (Pune)
        res_pune = self.client.get("/api/v1/risk/Pune?disease=Dengue")
        self.assertEqual(res_pune.status_code, 200)
        self.assertTrue(res_pune.json()["is_supported"])

        # Unsupported district (Ratnagiri)
        res_rat = self.client.get("/api/v1/risk/Ratnagiri?disease=Dengue")
        self.assertEqual(res_rat.status_code, 200)
        self.assertFalse(res_rat.json()["is_supported"])

    def test_05_forecasting_endpoints(self):
        """Test case forecasting overview and district multi-horizon projections."""
        # Statewide 7-day forecast
        res_fc = self.client.get("/api/v1/forecast?disease=Dengue&horizon_days=7")
        self.assertEqual(res_fc.status_code, 200)
        fc_data = res_fc.json()
        self.assertEqual(fc_data["forecast_horizon_days"], 7)
        self.assertEqual(fc_data["metadata"]["prediction_type"], "FORECAST")

        # District multi-horizon forecast (Pune)
        res_pune = self.client.get("/api/v1/forecast/Pune?disease=Dengue")
        self.assertEqual(res_pune.status_code, 200)
        pune_fc = res_pune.json()
        self.assertTrue(pune_fc["is_supported"])
        self.assertEqual(len(pune_fc["horizons"]), 3)  # 7d, 14d, 30d

        # Unsupported district
        res_mum = self.client.get("/api/v1/forecast/Mumbai%20City?disease=Dengue")
        self.assertEqual(res_mum.status_code, 200)
        self.assertFalse(res_mum.json()["is_supported"])

    def test_06_anomalies_endpoints(self):
        """Test anomaly listing, pagination, and detail diagnostics."""
        res_anom = self.client.get("/api/v1/anomalies?limit=10")
        self.assertEqual(res_anom.status_code, 200)
        anomalies = res_anom.json()["anomalies"]
        self.assertGreater(len(anomalies), 0)

        # Anomaly detail
        anom_id = anomalies[0]["anomaly_id"]
        res_det = self.client.get(f"/api/v1/anomalies/{anom_id}")
        self.assertEqual(res_det.status_code, 200)
        det_data = res_det.json()
        self.assertIn("historical_baseline", det_data)
        self.assertEqual(det_data["metadata"]["prediction_type"], "ANOMALY_SIGNAL")

    def test_07_climate_and_hotspots(self):
        """Test climate indicators and geospatial hotspot clusters."""
        # Climate
        res_clim = self.client.get("/api/v1/climate")
        self.assertEqual(res_clim.status_code, 200)
        c_data = res_clim.json()
        self.assertGreater(len(c_data["district_climate_indicators"]), 0)

        # Hotspots
        res_hot = self.client.get("/api/v1/hotspots?disease=Dengue")
        self.assertEqual(res_hot.status_code, 200)
        h_data = res_hot.json()
        self.assertTrue(h_data["spatial_clustering_significant"])

    def test_08_resource_intelligence_with_disclaimer(self):
        """Test resource demand estimates, verifying unsupported categories disclaimer."""
        res_res = self.client.get("/api/v1/resources?disease=Dengue")
        self.assertEqual(res_res.status_code, 200)
        r_data = res_res.json()
        self.assertGreater(r_data["total_estimated_bed_days"], 0)
        self.assertIn("Empirical Hospital Bed Occupancy", r_data["unsupported_categories"][0])
        self.assertEqual(r_data["metadata"]["prediction_type"], "SIMULATION")

    def test_09_explainability_and_models(self):
        """Test prediction SHAP explanation and model registry catalog."""
        # Explanation
        res_xai = self.client.get("/api/v1/explanations/PRED_PUNE_DENGUE_2025_W32")
        self.assertEqual(res_xai.status_code, 200)
        x_data = res_xai.json()
        self.assertGreater(len(x_data["top_contributing_features"]), 0)

        # Model Registry
        res_mods = self.client.get("/api/v1/models")
        self.assertEqual(res_mods.status_code, 200)
        m_data = res_mods.json()
        self.assertGreaterEqual(m_data["total_models"], 15)

        # Model Metrics
        res_met = self.client.get("/api/v1/models/metrics")
        self.assertEqual(res_met.status_code, 200)

    def test_10_data_quality_report(self):
        """Test data quality, missingness audit, and lineage integrity."""
        res = self.client.get("/api/v1/data-quality")
        self.assertEqual(res.status_code, 200)
        dq = res.json()
        self.assertEqual(dq["total_verified_outbreak_weeks"], 511)
        self.assertTrue(dq["data_leakage_safeguards_active"])

    def test_11_alert_review_actions_and_security(self):
        """Test alert review endpoint security, authentication, and audit persistence."""
        # Get an alert
        alerts_res = self.client.get("/api/v1/alerts")
        alerts = alerts_res.json()["alerts"]
        self.assertGreater(len(alerts), 0)
        target_alert = alerts[0]["alert_id"]

        # 1. Unauthenticated review -> 401 Unauthorized
        res_unauth = self.client.post(
            f"/api/v1/alerts/{target_alert}/review",
            json={
                "reviewer_id": "dr_patil",
                "action_taken": "CONFIRMED_OUTBREAK",
                "comments": "Confirmed via blood cultures.",
            },
        )
        self.assertEqual(res_unauth.status_code, 401)

        # 2. Authenticated review with valid X-Reviewer-Key -> 200 OK
        res_auth = self.client.post(
            f"/api/v1/alerts/{target_alert}/review",
            headers=self.auth_header,
            json={
                "reviewer_id": "dr_patil_epidemiologist",
                "action_taken": "CONFIRMED_OUTBREAK",
                "comments": "Confirmed via district civil hospital field visit.",
            },
        )
        self.assertEqual(res_auth.status_code, 200)
        self.assertEqual(res_auth.json()["status"], "RECORDED")

        # 3. Verify audit persistence in review_actions table
        with get_db_connection() as conn:
            row = conn.execute("SELECT * FROM review_actions WHERE alert_id = ? ORDER BY id DESC LIMIT 1", (target_alert,)).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["action_taken"], "CONFIRMED_OUTBREAK")
            self.assertEqual(row["reviewer_id"], "dr_patil_epidemiologist")

    def test_12_scenario_simulation(self):
        """Test scenario simulation run and detail retrieval."""
        run_res = self.client.post(
            "/api/v1/scenarios/run",
            json={
                "scenario_type": "VECTOR_CONTROL_INTENSIFICATION",
                "target_district": "Pune",
                "disease": "Dengue",
                "intervention_start_week": 1,
                "duration_weeks": 4,
                "assumed_efficacy_tier": "MODERATE",
            },
        )
        self.assertEqual(run_res.status_code, 200)
        scen_id = run_res.json()["scenario_id"]

        # Retrieve detail
        det_res = self.client.get(f"/api/v1/scenarios/{scen_id}")
        self.assertEqual(det_res.status_code, 200)
        self.assertEqual(det_res.json()["status"], "COMPLETED")

    def test_13_ewars_endpoints(self):
        """Test WHO EWARS-inspired threshold endpoints."""
        # Threshold catalog
        res_th = self.client.get("/api/v1/ewars/thresholds")
        self.assertEqual(res_th.status_code, 200)
        self.assertGreaterEqual(res_th.json()["total_rules"], 15)

        # Live evaluation
        res_eval = self.client.post(
            "/api/v1/ewars/evaluate",
            json={
                "district": "Pune",
                "disease": "Cholera",
                "reporting_period": "2025-W32",
                "observed_cases": 8.0,
            },
        )
        self.assertEqual(res_eval.status_code, 200)
        self.assertEqual(res_eval.json()["evaluation"]["alert_status"], "ALERT_CONFIRMED_INVESTIGATION_REQUIRED")


if __name__ == "__main__":
    unittest.main()
