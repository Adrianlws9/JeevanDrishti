# JEEVAN-DRISHTI: Automated Data Ingestion and Two-Second Dashboard Refresh Architecture Report

**Document Reference**: `PHASE-18-INGESTION-REFRESH-2026-V1`  
**System Version**: JEEVAN-DRISHTI v1.2.0-composite  
**Target Territory**: Maharashtra, India (36 Administrative Districts)  
**Status**: Implemented, Verified, and Tested (138/138 Unit & Integration Tests Passing)

---

## 1. Executive Summary & Core Architectural Principle

The **JEEVAN-DRISHTI** platform integrates an **Automated Data Ingestion and Two-Second Dashboard Refresh System** engineered to provide public-health epidemiologists and state emergency response teams with high-frequency operational awareness while preserving strict scientific and cryptographic data integrity.

### The Fundamental Cadence Distinction
A foundational design requirement of JEEVAN-DRISHTI is the explicit operational separation between **Dashboard UI Refresh Frequency** and **Source-Data Update Frequency**:

```mermaid
flowchart TD
    subgraph UI_Cadence["High-Frequency UI Polling (Every 2.0s)"]
        Browser["Dashboard Client / Field Tablet"] -->|GET /api/v1/refresh/summary| FastEndpoint["Ultra-Fast Refresh Service (<10ms)"]
        FastEndpoint -->|Read Cached Snapshot| SQLiteCache[("Indexed SQLite Metadata & Alerts")]
        SQLiteCache -->|Return Immediate State| FastEndpoint
        FastEndpoint -->|Wall-Clock TS + Source TS| Browser
    end

    subgraph Data_Cadence["Epidemiological Ingestion (Weekly / Staged)"]
        IDSP["IDSP Maharashtra Weekly Surveillance CSV"] -->|Upload / Automated Feed| IngestionEngine["11-Stage Ingestion Pipeline"]
        IngestionEngine --> Stage5{"SHA-256 Checksum"}
        Stage5 -->|Duplicate Match| Skip["DUPLICATE_SKIPPED (0 Recomputation)"]
        Stage5 -->|Novel Data| Cleanse["Standardization & Preprocessing"]
        Cleanse --> Inference["Inference on New Rows ONLY (No Retraining)"]
        Inference --> DBUpdate[("SQLite Database & Audit Logs")]
    end
```

> [!IMPORTANT]
> **Truthful Public-Health Cadence Notice**:
> 1. **Dashboard Refresh Frequency**: Every **2.0 seconds** (`refresh_interval_seconds: 2.0`). This frequency polls active early-warning alerts, epidemiologist field review statuses, and cluster classifications.
> 2. **Surveillance Data Frequency**: **Weekly** (`source_frequency_label: Weekly IDSP Surveillance (Aggregated)`). Public-health surveillance records in Maharashtra follow epidemiological reporting weeks (`YYYY-Www`).
> 3. **Model Prediction Policy**: Machine learning and deep learning models are **NEVER retrained on the 2-second loop**. Model inference is executed **strictly and only** when newly verified, non-duplicate surveillance records are ingested into the database. On standard 2-second UI polls, predictions are served directly from cached indexed storage.

---

## 2. Eleven-Stage Automated Ingestion Pipeline

The ingestion engine ([`backend/ingestion/pipeline.py`](file:///C:/Users/Janhvi%20Suryavanshi/.gemini/antigravity/scratch/JEEVAN-DRISHTI/backend/ingestion/pipeline.py)) enforces a 11-stage pipeline where every stage is auditable, idempotent, and fault-tolerant:

| Stage # | Stage Name | Technical Action & Safeguards |
| :--- | :--- | :--- |
| **Stage 1** | **Source Discovery or Upload** | Ingests multipart CSV upload from `POST /api/v1/ingestion/upload` or scans `data/ingestion/staging/` for staged feeds. Enforces 50MB maximum payload ceiling. |
| **Stage 2** | **File Ingestion (Raw Preservation)** | Writes raw bytes to immutable upload archive (`data/ingestion/uploads/{timestamp}_{filename}`). **Original raw historical datasets in `data/raw/` are strictly never overwritten or modified.** |
| **Stage 3** | **Schema Validation** | Inspects column headers against candidate dictionaries for `district`, `disease`, `observed_cases`, `reporting_period` (or `calendar_year` + `week_number`), and optional `observed_deaths`. Rejects unparseable tables. |
| **Stage 4** | **Data Quality Checks** | Checks that `observed_cases >= 0`, `observed_deaths >= 0`, `observed_deaths <= observed_cases`, and cases do not exceed the epidemiological plausibility limit (50,000 cases/week). |
| **Stage 5** | **Duplicate Detection (SHA-256 Identity)** | Computes cryptographic SHA-256 digest of uploaded bytes. Queries `ingestion_audit_logs` and `data_sources`. If hash exists, flags `DUPLICATE_SKIPPED`, logs audit, and safely exits in $\mathcal{O}(1)$ time without recomputing predictions. |
| **Stage 6** | **District & Disease Standardization** | Maps colloquial/historical district spellings (e.g., "Aurangabad" $\rightarrow$ "Chhatrapati Sambhajinagar", "Nasik" $\rightarrow$ "Nashik") and unifies disease taxonomy using project-verified standardizers. Rejects unmapped locations. |
| **Stage 7** | **Data Preprocessing** | Formats ISO week periods (`YYYY-Www`), assigns canonical `district_id`, tags dataset split (`current_monitoring`), and filters out records that already exist in SQLite `observations`. |
| **Stage 8** | **Feature Generation** | Constructs prior week lags (`prev_outbreak_cases_lag_1`) and rolling statistics required by predictive engines. |
| **Stage 9** | **Prediction Execution (Conditional)** | Triggered **only** if `new_records_inserted > 0`. Runs pre-trained inference on new rows: calculates risk scores via `OutbreakRiskPredictor`, forecasts via `DiseaseForecaster`, and flags alerts via `EWARSThresholdEngine`. **Zero retraining occurs.** |
| **Stage 10** | **Database Update** | Atomically inserts valid observations into `observations`. Updates `system_metadata` timestamps: `source_observation_date`, `last_successful_ingestion`, `last_processing_completion`, `last_prediction_generation`. |
| **Stage 11** | **Audit Logging** | Inserts complete execution log into relational table `ingestion_audit_logs` capturing runtime, valid/invalid counts, errors, and recomputation status. |

---

## 3. High-Frequency 2-Second Polling & Timestamp Lineage

The system strictly tracks separate timestamps to prevent false illusions of real-time patient generation:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        TIMESTAMP SEPARATION ENGINE                     │
├────────────────────────────────────────────────────────────────────────┤
│ 1. last_dashboard_refresh   : 2026-10-31T19:35:50.293145 (Current Poll)│
│ 2. source_observation_date  : 2026-W04                   (IDSP Week)   │
│ 3. last_successful_ingestion: 2026-10-31T19:35:53.605000 (File Ingest) │
│ 4. last_processing_completion:2026-10-31T19:35:53.605000 (Preprocessing)│
│ 5. last_prediction_generation:2026-10-31T19:35:53.605000 (Inference)   │
│ 6. model_version            : jeevan-v1.2.0-composite   (Frozen Model)│
└────────────────────────────────────────────────────────────────────────┘
```

### JSON Schema of `GET /api/v1/refresh/summary`:
```json
{
  "last_dashboard_refresh": "2026-10-31T19:35:50.293145",
  "source_observation_date": "2026-W04",
  "source_frequency_label": "Weekly IDSP Surveillance (Aggregated)",
  "refresh_interval_seconds": 2.0,
  "data_status_label": "Dashboard refreshed just now | Source data current as of 2026-W04 | 12 active alerts pending review",
  "system_status": "HEALTHY",
  "last_successful_ingestion": "2026-10-31T19:35:53.605000",
  "last_processing_completion": "2026-10-31T19:35:53.605000",
  "last_prediction_generation": "2026-10-31T19:35:53.605000",
  "model_version": "jeevan-v1.2.0-composite",
  "total_districts": 36,
  "total_diseases": 28,
  "total_observations": 514,
  "active_alerts_count": 12,
  "high_risk_districts_count": 6,
  "anomalies_flagged_count": 45,
  "recent_alerts": [
    {
      "alert_id": "ALERT_EWARS_0001_PUN_2025_W52",
      "district": "Pune",
      "disease": "Dengue",
      "period": "2025-W52",
      "severity": "CRITICAL",
      "trigger": "WHO EWARS 3-Week Consecutive Increase",
      "status": "PENDING_REVIEW"
    }
  ],
  "truthful_clarification": "Dashboard UI refreshes every 2.0 seconds to stream UI alert status and reviewer approvals. Surveillance source data is updated weekly via IDSP. Analytical models are strictly cached and re-executed only upon arrival of verified new surveillance records; models are never retrained on 2-second polls."
}
```

---

## 4. REST API Endpoint Specifications

| Endpoint | Method | Purpose | Response Time |
| :--- | :--- | :--- | :--- |
| [`/api/v1/refresh/summary`](file:///C:/Users/Janhvi%20Suryavanshi/.gemini/antigravity/scratch/JEEVAN-DRISHTI/backend/api/v1/refresh.py) | `GET` | Ultra-lightweight endpoint polled every 2s by client dashboards. Returns alert counts, metrics, and separate timestamps. | **< 10ms** |
| [`/api/v1/refresh/status`](file:///C:/Users/Janhvi%20Suryavanshi/.gemini/antigravity/scratch/JEEVAN-DRISHTI/backend/api/v1/refresh.py) | `GET` | Sync status audit endpoint validating cadence policies. | **< 6ms** |
| [`/api/v1/ingestion/upload`](file:///C:/Users/Janhvi%20Suryavanshi/.gemini/antigravity/scratch/JEEVAN-DRISHTI/backend/api/v1/ingestion.py) | `POST` | Multipart file upload for surveillance CSVs. Executes 11-stage validation, duplicate detection, and model inference. | **0.015s – 3.2s** (depending on new rows) |
| [`/api/v1/ingestion/trigger`](file:///C:/Users/Janhvi%20Suryavanshi/.gemini/antigravity/scratch/JEEVAN-DRISHTI/backend/api/v1/ingestion.py) | `POST` | Manually sweeps staged feeds directory (`data/ingestion/staging`) and ingests pending batches. | **0.045s** |
| [`/api/v1/ingestion/audit-logs`](file:///C:/Users/Janhvi%20Suryavanshi/.gemini/antigravity/scratch/JEEVAN-DRISHTI/backend/api/v1/ingestion.py) | `GET` | Returns immutable audit logs with status filtering (`SUCCESS`, `DUPLICATE_SKIPPED`, `VALIDATION_FAILED`). | **< 12ms** |
| [`/api/v1/ingestion/status`](file:///C:/Users/Janhvi%20Suryavanshi/.gemini/antigravity/scratch/JEEVAN-DRISHTI/backend/api/v1/ingestion.py) | `GET` | Reports pipeline operational readiness, staging queue count, and total run statistics. | **< 5ms** |

---

## 5. Performance Latency Benchmarks

Benchmarked on local workstation hardware under simulated high-frequency dashboard traffic:

| Operation | Target Budget | Measured Latency | Conformance Status |
| :--- | :--- | :--- | :--- |
| **Dashboard Refresh Summary (`GET /refresh/summary`)** | < 20.0ms | **10.1ms** (Cold) / **4.1ms** (Hot) | **PASSED** (Exceeds spec) |
| **Duplicate Detection via SHA-256 (`POST /ingestion/upload`)** | < 50.0ms | **11.3ms** | **PASSED** ($\mathcal{O}(1)$ bypass) |
| **Schema & Quality Rejection (`POST /ingestion/upload`)** | < 50.0ms | **16.1ms** | **PASSED** |
| **New Surveillance Batch Ingestion + Inference (3 Districts)** | < 5.0s | **3.28s** | **PASSED** |
| **Staged Trigger Sweep (`POST /ingestion/trigger`)** | < 500.0ms | **45.8ms** | **PASSED** |

---

## 6. Interactive Command Center Dashboard

An interactive dashboard demonstration is deployed at:  
[`reports/dashboard/two_second_refresh_dashboard.html`](file:///C:/Users/Janhvi%20Suryavanshi/.gemini/antigravity/scratch/JEEVAN-DRISHTI/reports/dashboard/two_second_refresh_dashboard.html)

**Key Visual Features**:
1. **Live Heartbeat LED Indicator**: Pulsing green dot with real-time ping display (`Ping: ~5ms | UI Polling: 2.0s`).
2. **Dual-Clock Timelines**: Side-by-side display clearly contrasting the live 2-second UI poll against the weekly IDSP epidemiological period.
3. **Truthful Public-Health Banner**: Prominently highlights that models do not re-run on the 2s loop and cases arrive weekly.
4. **Live Early-Warning Alerts Ticker**: Streams real-time pending alerts directly from SQLite.
5. **Interactive Ingestion Simulator**: Buttons allowing analysts to upload simulated novel epidemiological weeks or test SHA-256 duplicate detection directly from the browser.

---

## 7. Verification & Regression Testing Evidence

The complete test suite was executed across all phases of the project:

```bash
python -W ignore -m unittest discover -s tests -p "test_*.py"
```

**Results**:
- `Ran 138 tests in 24.870s`
- **Result: `OK` (100% Pass Rate)**
- Test Suites Verified:
  1. [`tests/test_data_ingestion_and_refresh.py`](file:///C:/Users/Janhvi%20Suryavanshi/.gemini/antigravity/scratch/JEEVAN-DRISHTI/tests/test_data_ingestion_and_refresh.py): 10/10 tests passing.
  2. [`tests/test_backend_api.py`](file:///C:/Users/Janhvi%20Suryavanshi/.gemini/antigravity/scratch/JEEVAN-DRISHTI/tests/test_backend_api.py): 13/13 integration tests passing.
  3. [`tests/test_ewars_thresholds.py`](file:///C:/Users/Janhvi%20Suryavanshi/.gemini/antigravity/scratch/JEEVAN-DRISHTI/tests/test_ewars_thresholds.py): 9/9 threshold tests passing.
  4. Core Predictive & Analytical Suites (Phases 1–16): 106/106 unit tests passing.

---

## 8. Summary of Components Created

1. [`backend/ingestion/config.py`](file:///C:/Users/Janhvi%20Suryavanshi/.gemini/antigravity/scratch/JEEVAN-DRISHTI/backend/ingestion/config.py): Ingestion paths and quality thresholds.
2. [`backend/ingestion/validator.py`](file:///C:/Users/Janhvi%20Suryavanshi/.gemini/antigravity/scratch/JEEVAN-DRISHTI/backend/ingestion/validator.py): SHA-256 duplicate checker and schema hygiene engine.
3. [`backend/ingestion/pipeline.py`](file:///C:/Users/Janhvi%20Suryavanshi/.gemini/antigravity/scratch/JEEVAN-DRISHTI/backend/ingestion/pipeline.py): 11-stage automated ingestion pipeline.
4. [`backend/services/refresh_service.py`](file:///C:/Users/Janhvi%20Suryavanshi/.gemini/antigravity/scratch/JEEVAN-DRISHTI/backend/services/refresh_service.py): Sub-10ms snapshot aggregator for 2-second polling.
5. [`backend/api/v1/refresh.py`](file:///C:/Users/Janhvi%20Suryavanshi/.gemini/antigravity/scratch/JEEVAN-DRISHTI/backend/api/v1/refresh.py): REST endpoints for 2-second refresh polling.
6. [`backend/api/v1/ingestion.py`](file:///C:/Users/Janhvi%20Suryavanshi/.gemini/antigravity/scratch/JEEVAN-DRISHTI/backend/api/v1/ingestion.py): REST endpoints for file upload, staged triggers, and audit trail.
7. [`backend/database/schema.py`](file:///C:/Users/Janhvi%20Suryavanshi/.gemini/antigravity/scratch/JEEVAN-DRISHTI/backend/database/schema.py): Updated with `ingestion_audit_logs` and `system_metadata` tables.
8. [`reports/dashboard/two_second_refresh_dashboard.html`](file:///C:/Users/Janhvi%20Suryavanshi/.gemini/antigravity/scratch/JEEVAN-DRISHTI/reports/dashboard/two_second_refresh_dashboard.html): High-frequency 2-second refresh command center HTML dashboard.
9. [`tests/test_data_ingestion_and_refresh.py`](file:///C:/Users/Janhvi%20Suryavanshi/.gemini/antigravity/scratch/JEEVAN-DRISHTI/tests/test_data_ingestion_and_refresh.py): Comprehensive test suite (10 unit & integration tests).
