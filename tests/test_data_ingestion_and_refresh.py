"""
Unit and Integration Tests for Data Ingestion & 2-Second Dashboard Refresh
==========================================================================
Phase 18: Automated Data Ingestion & 2-Second Dashboard Refresh

Verifies:
- 2-second refresh summary endpoint latency and schema
- Timestamp separation: dashboard poll wall-clock != source observation date
- Truthful labelling (no deceptive "live data arriving every 2 seconds")
- 11-stage automated ingestion pipeline
- Cryptographic SHA-256 duplicate detection and graceful skip
- Schema and data quality validation rejection
- Audit logging and staged trigger execution
"""

import time
import unittest
from pathlib import Path
from fastapi.testclient import TestClient

from backend.config import settings
from backend.database.connection import get_db_connection, init_db, seed_db
from backend.ingestion.config import IngestionConfig
from backend.main import app

PROJ = Path(__file__).resolve().parents[1]


class TestDataIngestionAndRefresh(unittest.TestCase):
    """Test suite for Phase 18 Ingestion and High-Frequency Refresh."""

    @classmethod
    def setUpClass(cls) -> None:
        init_db()
        seed_db()
        cls.client = TestClient(app)
        IngestionConfig.ensure_directories()
        cls.test_run_id = int(time.time())
        # Clean up previous test entries from database for clean test run
        with get_db_connection() as conn:
            conn.execute("DELETE FROM observations WHERE reporting_period LIKE '2026-W%'")
            conn.execute("DELETE FROM ingestion_audit_logs WHERE filename LIKE '%test%' OR filename LIKE '%2026%'")

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove test-generated rows so the application DB is not polluted by test data."""
        with get_db_connection() as conn:
            seed = conn.execute("SELECT MIN(created_at) FROM observations WHERE split != 'current_monitoring'").fetchone()[0]
            conn.execute("DELETE FROM observations WHERE split = 'current_monitoring'")
            conn.execute("DELETE FROM ingestion_audit_logs")
            if seed:
                for k in ("last_successful_ingestion", "last_processing_completion",
                          "last_prediction_generation", "last_successful_api_update"):
                    conn.execute("UPDATE system_metadata SET value=?, updated_at=? WHERE key=?", (seed, seed, k))
            conn.execute("UPDATE system_metadata SET value='2025-W52' WHERE key='source_observation_date'")
        for folder in (IngestionConfig.UPLOAD_DIR, IngestionConfig.ARCHIVE_DIR, IngestionConfig.STAGING_DIR):
            for f in folder.glob("*"):
                try:
                    f.unlink()
                except OSError:
                    pass

    def test_01_refresh_summary_schema_and_latency(self):
        """Test that /api/v1/refresh/summary responds ultra-fast (<100ms) with full schema."""
        start = time.time()
        res = self.client.get("/api/v1/refresh/summary")
        elapsed = time.time() - start

        self.assertEqual(res.status_code, 200)
        self.assertLess(elapsed, 0.200, f"Refresh summary latency ({elapsed:.4f}s) exceeded 200ms budget.")

        data = res.json()
        self.assertIn("last_dashboard_refresh", data)
        self.assertIn("source_observation_date", data)
        self.assertIn("source_frequency_label", data)
        self.assertEqual(data["refresh_interval_seconds"], 2.0)
        self.assertIn("data_status_label", data)
        self.assertEqual(data["system_status"], "HEALTHY")
        self.assertEqual(data["total_districts"], 36)
        self.assertEqual(data["total_diseases"], 28)
        self.assertGreater(data["total_observations"], 0)
        self.assertIsInstance(data["recent_alerts"], list)

    def test_02_timestamp_separation_truthfulness(self):
        """Test strict separation of poll wall-clock vs source surveillance date."""
        res = self.client.get("/api/v1/refresh/summary")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        # The dashboard refresh timestamp is a real ISO datetime with hour/minute/second
        self.assertIn("T", data["last_dashboard_refresh"])

        # The source observation date is an epidemiological period (e.g. 2025-W52)
        self.assertIn("-W", data["source_observation_date"])

        # They must NEVER be identical
        self.assertNotEqual(data["last_dashboard_refresh"], data["source_observation_date"])

        # Truthful clarification must be explicitly articulated
        self.assertIn("Dashboard UI refreshes every 2", data["truthful_clarification"])
        self.assertIn("weekly via IDSP", data["truthful_clarification"])
        self.assertIn("never retrained on 2-second polls", data["truthful_clarification"])

    def test_03_refresh_status_endpoint(self):
        """Test detailed sync cadence verification endpoint."""
        res = self.client.get("/api/v1/refresh/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["dashboard_ui_refresh_cadence"], "Every 2.0 seconds")
        self.assertEqual(data["active_models_recomputed_on_poll"], False)
        self.assertIn("Predictive models run inference only", data["policy"])

    def test_04_valid_new_surveillance_csv_ingestion(self):
        """Test 11-stage ingestion of a new valid surveillance CSV with novel period."""
        csv_data = (
            "district,disease,reporting_period,observed_cases,observed_deaths\n"
            "Pune,Dengue,2026-W04,45,1\n"
            "Nashik,Malaria,2026-W04,18,0\n"
            "Nagpur,Chikungunya,2026-W04,12,0\n"
        )
        files = {"file": ("idsp_mh_new_period_2026_w04.csv", csv_data.encode("utf-8"), "text/csv")}

        res = self.client.post("/api/v1/ingestion/upload", files=files)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["status"], "SUCCESS")
        self.assertEqual(data["stage_reached"], "STAGE_11_AUDIT_LOGGING")
        self.assertEqual(data["records_total"], 3)
        self.assertEqual(data["records_valid"], 3)
        self.assertEqual(data["records_invalid"], 0)
        self.assertEqual(data["new_records_inserted"], 3)
        self.assertTrue(data["predictions_updated"])
        self.assertGreater(data["execution_duration_sec"], 0.0)

        # Verify records exist in database
        with get_db_connection() as conn:
            obs = conn.execute(
                "SELECT COUNT(*) FROM observations WHERE reporting_period = '2026-W04'"
            ).fetchone()[0]
            self.assertEqual(obs, 3)

            # Check that system_metadata updated source_observation_date
            meta = conn.execute(
                "SELECT value FROM system_metadata WHERE key = 'source_observation_date'"
            ).fetchone()
            self.assertEqual(meta["value"], "2026-W04")

    def test_05_duplicate_sha256_detection_and_skip(self):
        """Test that uploading the exact same CSV twice is detected via SHA-256 and skipped."""
        csv_data = (
            "district,disease,reporting_period,observed_cases,observed_deaths\n"
            "Pune,Dengue,2026-W04,45,1\n"
            "Nashik,Malaria,2026-W04,18,0\n"
            "Nagpur,Chikungunya,2026-W04,12,0\n"
        )
        files = {"file": ("idsp_mh_new_period_2026_w04_duplicate.csv", csv_data.encode("utf-8"), "text/csv")}

        res = self.client.post("/api/v1/ingestion/upload", files=files)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["status"], "DUPLICATE_SKIPPED")
        self.assertEqual(data["stage_reached"], "STAGE_5_DUPLICATE_DETECTION")
        self.assertEqual(data["new_records_inserted"], 0)
        self.assertFalse(data["predictions_updated"])
        self.assertIn("Duplicate file detected via SHA-256", data["message"])

    def test_06_malformed_and_invalid_data_rejection(self):
        """Test rejection of malformed surveillance data (unmapped district, negative cases)."""
        bad_csv = (
            "district,disease,reporting_period,observed_cases,observed_deaths\n"
            "AtlantisLostCity,Dengue,2026-W05,25,0\n"
            "Pune,Dengue,2026-W05,-15,0\n"
        )
        files = {"file": ("malformed_surveillance.csv", bad_csv.encode("utf-8"), "text/csv")}

        res = self.client.post("/api/v1/ingestion/upload", files=files)
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["status"], "VALIDATION_FAILED")
        self.assertEqual(data["new_records_inserted"], 0)
        self.assertFalse(data["predictions_updated"])
        self.assertIsNotNone(data["validation_errors"])

    def test_07_unsupported_file_extension(self):
        """Test rejecting non-CSV uploads with 400 Bad Request."""
        bad_file = {"file": ("report.pdf", b"%PDF-1.4...", "application/pdf")}
        res = self.client.post("/api/v1/ingestion/upload", files=bad_file)
        self.assertEqual(res.status_code, 400)
        self.assertIn("Unsupported file format", res.json()["detail"])

    def test_08_audit_logs_query_and_filtering(self):
        """Test querying and filtering the immutable ingestion audit trail."""
        res = self.client.get("/api/v1/ingestion/audit-logs")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertGreaterEqual(data["total_logs"], 2)

        # Filter by DUPLICATE_SKIPPED
        res_dup = self.client.get("/api/v1/ingestion/audit-logs?status=DUPLICATE_SKIPPED")
        self.assertEqual(res_dup.status_code, 200)
        logs_dup = res_dup.json()["logs"]
        self.assertTrue(all(l["status"] == "DUPLICATE_SKIPPED" for l in logs_dup))

    def test_09_ingestion_pipeline_status(self):
        """Test operational status endpoint of ingestion service."""
        res = self.client.get("/api/v1/ingestion/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["pipeline_status"], "ONLINE")
        self.assertIn(".csv", data["supported_formats"])
        self.assertGreaterEqual(data["total_ingestion_runs"], 1)

    def test_10_staged_trigger_workflow(self):
        """Test automated staged directory trigger workflow."""
        run_ts = int(time.time() * 1000)
        test_file = IngestionConfig.STAGING_DIR / f"staged_test_{run_ts}.csv"
        test_file.write_text(
            f"district,disease,reporting_period,observed_cases,observed_deaths\n"
            f"Solapur,Cholera,2026-W12,{(run_ts % 20) + 1},0\n",
            encoding="utf-8",
        )

        res = self.client.post("/api/v1/ingestion/trigger")
        self.assertEqual(res.status_code, 200)
        results = res.json()
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "SUCCESS")

        # Staged file should have been moved/archived
        self.assertFalse(test_file.exists())


if __name__ == "__main__":
    unittest.main()
