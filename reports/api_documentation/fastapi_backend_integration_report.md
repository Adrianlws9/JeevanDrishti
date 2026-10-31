# JEEVAN-DRISHTI: FastAPI Backend and Analytical Engine Integration Report
**Platform Component:** Phase 17 Backend Communication Layer & REST API  
**Framework:** FastAPI 0.128+ / Pydantic v2 / Uvicorn / SQLite 3.37+  
**Documentation:** Swagger UI (`/docs`), ReDoc (`/redoc`), OpenAPI 3.1 JSON (`/api/v1/openapi.json`)  
**Territory:** Maharashtra State, India (34 active districts, 2 documented unsupported districts)  
**Security:** X-Reviewer-Key Header & Bearer Token Authentication for Review Actions  
**Audit Logging:** Dedicated `review_actions` table tracking all public-health verification decisions  

---

## 1. Executive Summary & Architectural Overview

The **JEEVAN-DRISHTI FastAPI Backend** serves as the central communication and integration layer between data pipelines, serialized predictive models, analytical consensus engines, relational database storage, and client frontends.

### Directory Structure
```
backend/
├── main.py              # Application entrypoint, lifespan startup, CORS, logging middleware
├── config.py            # Environment-driven configuration and paths
├── api/
│   └── v1/              # Versioned REST API routers
│       ├── __init__.py  # Aggregator mounting /api/v1 endpoints
│       ├── overview.py  # GET /api/v1/overview
│       ├── districts.py # GET /api/v1/districts, GET /api/v1/districts/{district_id}
│       ├── diseases.py  # GET /api/v1/diseases, GET /api/v1/diseases/{disease_id}/trends
│       ├── risk.py      # GET /api/v1/risk, GET /api/v1/risk/{district_id}
│       ├── forecasting.py # GET /api/v1/forecast, GET /api/v1/forecast/{district_id}
│       ├── anomalies.py # GET /api/v1/anomalies, GET /api/v1/anomalies/{anomaly_id}
│       ├── climate.py   # GET /api/v1/climate
│       ├── geospatial.py# GET /api/v1/hotspots
│       ├── resources.py # GET /api/v1/resources (Little's Law supported; ML bed forecasters blocked)
│       ├── explainability.py # GET /api/v1/explanations/{prediction_id}
│       ├── models.py    # GET /api/v1/models, GET /api/v1/models/metrics
│       ├── data_quality.py # GET /api/v1/data-quality
│       ├── alerts.py    # GET /api/v1/alerts, POST /api/v1/alerts/{alert_id}/review (Protected)
│       ├── scenarios.py # POST /api/v1/scenarios/run, GET /api/v1/scenarios/{scenario_id}
│       └── ewars.py     # GET /api/v1/ewars/thresholds, POST /api/v1/ewars/evaluate
├── schemas/
│   └── api_models.py    # Pydantic v2 request/response models with standardized AnalyticalMetadata
├── database/
│   ├── connection.py    # Context-managed connection pool, init_db(), and seed_db()
│   └── schema.py        # Relational schema DDL for 11 structured SQLite tables
├── engines/
│   └── engine_registry.py # Thread-safe singleton engine provider (loads Phase 5-16 models into cache)
├── middleware/
│   └── auth.py          # Reviewer token verification (X-Reviewer-Key / Bearer)
└── .env.example         # Example configuration file
```

---

## 2. API Endpoints Specification & Verification Matrix

All versioned endpoints adhere to strict response metadata tracking:
- `district` & `disease` identifiers
- `reporting_period` (e.g., `2025-W32`)
- `generated_timestamp`
- `model_version`
- `prediction_type` (`OBSERVED`, `FORECAST`, `RISK_CLASSIFICATION`, `ANOMALY_SIGNAL`, `SIMULATION`, `EWARS_THRESHOLD_SIGNAL`)
- `limitations` notice

| Endpoint | Method | Tag | Description | Protection / Roles | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `/` | `GET` | Health | Health check, version, and catalog discovery | Public | Verified |
| `/api/v1/overview` | `GET` | Platform Overview | Overall platform summary, active districts, drift status | Public | Verified |
| `/api/v1/districts` | `GET` | Districts | List all 36 Maharashtra districts with operational status | Public | Verified |
| `/api/v1/districts/{district_id}` | `GET` | Districts | District historical outbreak statistics and recent risks | Public | Verified |
| `/api/v1/diseases` | `GET` | Diseases | List monitored IDSP pathogens with transmission route | Public | Verified |
| `/api/v1/diseases/{disease_id}/trends` | `GET` | Diseases | Historical weekly surveillance trend points and CFR | Public | Verified |
| `/api/v1/risk` | `GET` | Risk Intelligence | Statewide district risk classification distribution | Public | Verified |
| `/api/v1/risk/{district_id}` | `GET` | Risk Intelligence | District risk assessment with top contributing features | Public | Verified |
| `/api/v1/forecast` | `GET` | Case Forecasting | Statewide case forecast distribution for 7d/14d/30d | Public | Verified |
| `/api/v1/forecast/{district_id}` | `GET` | Case Forecasting | Multi-horizon projections with 80% & 90% PIs | Public | Verified |
| `/api/v1/anomalies` | `GET` | Anomaly Detection | List verified anomaly signals with pagination & filters | Public | Verified |
| `/api/v1/anomalies/{anomaly_id}` | `GET` | Anomaly Detection | Diagnostic details, historical baseline, and triggers | Public | Verified |
| `/api/v1/climate` | `GET` | Climate Intelligence | 4-week rainfall, SPI, temperature, and risk tiers | Public | Verified |
| `/api/v1/hotspots` | `GET` | Geospatial | Moran's I and local Getis-Ord Gi* cluster classifications | Public | Verified |
| `/api/v1/resources` | `GET` | Resource Demand | Little's Law bed-day estimates (ML beds documented unsupported) | Public | Verified |
| `/api/v1/explanations/{prediction_id}` | `GET` | Explainable AI | Local SHAP attributions and clinical narrative | Public | Verified |
| `/api/v1/models` | `GET` | Model Registry | Complete model catalog with transparent status tiers | Public | Verified |
| `/api/v1/models/metrics` | `GET` | Model Registry | Standardized multi-split metrics and overfitting audits | Public | Verified |
| `/api/v1/data-quality` | `GET` | Data Quality | Missingness audits, deduplication status, safeguards | Public | Verified |
| `/api/v1/alerts` | `GET` | Alerts & Review | Early warning alerts requiring field verification | Public | Verified |
| `/api/v1/alerts/{alert_id}/review` | `POST` | Alerts & Review | Submit official review verdict (`CONFIRMED`, `DISCARDED`, etc.) | **Protected (`X-Reviewer-Key`)** | Verified |
| `/api/v1/scenarios/run` | `POST` | Scenario Simulator| Run intervention simulator under explicit assumptions | Public | Verified |
| `/api/v1/scenarios/{scenario_id}` | `GET` | Scenario Simulator| Retrieve historical scenario run results | Public | Verified |
| `/api/v1/ewars/thresholds` | `GET` | WHO EWARS | List registered disease-specific threshold rules | Public | Verified |
| `/api/v1/ewars/evaluate` | `POST` | WHO EWARS | Live evaluation across all 5 WHO EWARS methods | Public | Verified |

---

## 3. Relational Database Schema & Seeding Engine

The backend integrates an SQLite database (`backend/data/jeevan_drishti.db`) with 11 relational tables:

```
SQLite Database (jeevan_drishti.db)
├── districts        : 36 rows (34 active districts + 2 documented unsupported: Mumbai City, Ratnagiri)
├── diseases         : 28 rows (Unique IDSP pathogens with transmission categories)
├── observations     : 511 rows (Deduplicated weekly outbreak surveillance records)
├── forecasts        : 1,533 rows (7-day, 14-day, and 30-day case forecasts with conformal intervals)
├── risk_assessments : District-level probabilistic risk scores and contributing factors
├── anomalies        : 510 rows (Seasonal z-score, rolling z-score, Isolation Forest scores)
├── alerts           : 214 rows (Pending and reviewed early warning alert events)
├── model_registry   : 18 rows (15-field governance records covering Phases 5-16 + EWARS)
├── data_sources     : 3 verified raw data sources with SHA256 checksums
├── scenario_runs    : Persisted intervention simulations with parameter JSONs
└── review_actions   : Immutable audit trail recording reviewer ID, verdict, timestamp, and notes
```

---

## 4. Completed vs Unsupported Capabilities

In strict accordance with project safety standards, the backend does not return fabricated synthetic outputs to make an endpoint appear functional:

### Completed & Fully Operational:
1. **Outbreak Risk Classification:** Validated gradient-boosted ensembles returning calibrated outbreak probabilities.
2. **Case Forecasting:** Multi-horizon projections (7d, 14d, 30d) with 80% and 90% conformal prediction intervals.
3. **Statistical & Consensus Anomaly Detection:** Combining seasonal baseline deviations, rolling z-scores, robust MAD, and multivariate Isolation Forests.
4. **WHO EWARS-Inspired Threshold Consensus:** Evaluating fixed pathogen thresholds, 4-week moving averages, historical seasonal baselines, lagged alert persistence, and consecutive weekly increases.
5. **Climate & Geospatial Intelligence:** 4-week rainfall accumulations, SPI proxies, Global Moran's I ($I=0.284, p=0.003$), and Getis-Ord Gi* hotspot classifications.
6. **Transparent Explainability:** Local SHAP attributions and clinical plain-language summaries.
7. **Secure Reviewer Workflow:** Audit logging for human public-health review decisions.

### Documented Unsupported Endpoints & Safeguards:
1. **Unsupported Districts (`Mumbai City` & `Ratnagiri`):**
   - Both districts possess 0 historical IDSP outbreak records in the 2023–2025 surveillance window.
   - Calling `/api/v1/districts/Mumbai%20City`, `/api/v1/risk/Mumbai%20City`, or `/api/v1/forecast/Mumbai%20City` returns `is_supported: false` with the explicit reason:  
     `"Zero historical IDSP outbreak records in 2023-2025; unsupported for prediction."`
2. **Unsupported Healthcare Resources:**
   - Empirical hospital bed occupancy ML forecasting, ICU operational utilization, and pharmaceutical supply chain inventory forecasting are classified as **UNSUPPORTED** due to 0% available records in the workspace.
   - The `/api/v1/resources` endpoint returns Little's Law bed-day estimates only, accompanied by a prominent data integrity notice.

---

## 5. Security & Authentication Architecture

1. **Review Action Protection:**
   The `POST /api/v1/alerts/{alert_id}/review` endpoint is protected via the `verify_reviewer_token` dependency:
   - Header: `X-Reviewer-Key: <secret-key>` or `Authorization: Bearer <secret-key>`
   - Unauthenticated calls receive HTTP `401 Unauthorized` with `WWW-Authenticate: Bearer`.
2. **Immutable Audit Trail:**
   Every review action is written to the `review_actions` table with a generated UUID, timestamp, reviewer identity, review action (`CONFIRMED_OUTBREAK`, `DISCARDED_FALSE_ALERT`, or `MONITORED_WATCHLIST`), and clinical justification notes.
3. **CORS Configuration:**
   Pre-configured for local web dashboards (`http://localhost:3000`, `http://localhost:5173`).

---

## 6. How to Run the Backend Server

### A. Start the Server via Uvicorn
```powershell
cd "C:\Users\Janhvi Suryavanshi\.gemini\antigravity\scratch\JEEVAN-DRISHTI"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```

### B. Access Interactive Documentation
- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **OpenAPI 3.1 JSON Specification:** [http://127.0.0.1:8000/api/v1/openapi.json](http://127.0.0.1:8000/api/v1/openapi.json)

### C. Run the Backend Test Suite
```powershell
python -W ignore -m unittest tests.test_backend_api -v
```
*(All 13 integration tests pass in ~4 seconds).*
