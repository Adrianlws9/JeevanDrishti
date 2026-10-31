"""
Database Schema Definition and Migrations for JEEVAN-DRISHTI
============================================================
Phase 17: FastAPI Backend & Analytical Engine Integration

Defines 11 structured relational tables:
1. districts
2. diseases
3. observations
4. forecasts
5. risk_assessments
6. anomalies
7. alerts
8. model_registry
9. data_sources
10. scenario_runs
11. review_actions
"""

SCHEMA_SQL = """
-- 1. Districts Table
CREATE TABLE IF NOT EXISTS districts (
    district_id TEXT PRIMARY KEY,
    district_name TEXT NOT NULL UNIQUE,
    division TEXT NOT NULL,
    latitude REAL,
    longitude REAL,
    is_active INTEGER NOT NULL DEFAULT 1,
    notes TEXT
);

-- 2. Diseases Table
CREATE TABLE IF NOT EXISTS diseases (
    disease_id TEXT PRIMARY KEY,
    disease_name TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL,
    is_priority INTEGER NOT NULL DEFAULT 1,
    transmission_route TEXT
);

-- 3. Surveillance Observations Table
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    district_id TEXT NOT NULL,
    district_name TEXT NOT NULL,
    disease_name TEXT NOT NULL,
    reporting_period TEXT NOT NULL,
    calendar_year INTEGER,
    week_number INTEGER,
    observed_cases REAL NOT NULL,
    observed_deaths REAL DEFAULT 0.0,
    split TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(district_id) REFERENCES districts(district_id)
);
CREATE INDEX IF NOT EXISTS idx_obs_dist_dis ON observations(district_id, disease_name);
CREATE INDEX IF NOT EXISTS idx_obs_period ON observations(reporting_period);

-- 4. Case Forecasts Table
CREATE TABLE IF NOT EXISTS forecasts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    district_id TEXT NOT NULL,
    district_name TEXT NOT NULL,
    disease_name TEXT NOT NULL,
    reporting_period TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    predicted_cases REAL NOT NULL,
    pi_lower_80 REAL,
    pi_upper_80 REAL,
    model_version TEXT NOT NULL,
    prediction_type TEXT NOT NULL DEFAULT 'FORECAST',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_forecast_dist ON forecasts(district_id, horizon_days);

-- 5. Outbreak Risk Assessments Table
CREATE TABLE IF NOT EXISTS risk_assessments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    district_id TEXT NOT NULL,
    district_name TEXT NOT NULL,
    disease_name TEXT NOT NULL,
    reporting_period TEXT NOT NULL,
    risk_tier TEXT NOT NULL,
    risk_score REAL NOT NULL,
    predicted_probability REAL,
    contributing_factors TEXT,
    model_version TEXT NOT NULL,
    prediction_type TEXT NOT NULL DEFAULT 'RISK_CLASSIFICATION',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_risk_dist ON risk_assessments(district_id, reporting_period);

-- 6. Anomalies Table
CREATE TABLE IF NOT EXISTS anomalies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    anomaly_id TEXT UNIQUE NOT NULL,
    district_id TEXT NOT NULL,
    district_name TEXT NOT NULL,
    disease_name TEXT NOT NULL,
    reporting_period TEXT NOT NULL,
    anomaly_score REAL NOT NULL,
    is_anomaly INTEGER NOT NULL,
    severity_tier TEXT NOT NULL,
    signal_types TEXT,
    seasonal_zscore REAL,
    rolling_zscore REAL,
    prediction_type TEXT NOT NULL DEFAULT 'ANOMALY_SIGNAL',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_anom_dist ON anomalies(district_id, reporting_period);

-- 7. Early Warning Alerts Table
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id TEXT UNIQUE NOT NULL,
    district_id TEXT NOT NULL,
    district_name TEXT NOT NULL,
    disease_name TEXT NOT NULL,
    reporting_period TEXT NOT NULL,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    trigger_reason TEXT NOT NULL,
    observed_cases REAL NOT NULL,
    threshold_value REAL,
    alert_status TEXT NOT NULL DEFAULT 'PENDING_REVIEW',
    human_verified INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(alert_status);

-- 8. Model Registry Table
CREATE TABLE IF NOT EXISTS model_registry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT UNIQUE NOT NULL,
    model_name TEXT NOT NULL,
    model_type TEXT NOT NULL,
    disease TEXT NOT NULL,
    supported_districts_count INTEGER NOT NULL,
    operational_status TEXT NOT NULL,
    primary_metric_name TEXT,
    primary_metric_value REAL,
    artifact_path TEXT NOT NULL,
    version TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- 9. Data Sources Table
CREATE TABLE IF NOT EXISTS data_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name TEXT NOT NULL,
    source_file TEXT NOT NULL,
    records_count INTEGER NOT NULL,
    time_span TEXT NOT NULL,
    sha256_checksum TEXT,
    status TEXT NOT NULL
);

-- 10. Scenario Simulation Runs Table
CREATE TABLE IF NOT EXISTS scenario_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario_id TEXT UNIQUE NOT NULL,
    scenario_type TEXT NOT NULL,
    target_district TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    results_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- 11. Review Actions Audit Table
CREATE TABLE IF NOT EXISTS review_actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id TEXT UNIQUE NOT NULL,
    alert_id TEXT NOT NULL,
    reviewer_id TEXT NOT NULL,
    action_taken TEXT NOT NULL,
    comments TEXT,
    review_timestamp TEXT NOT NULL,
    FOREIGN KEY(alert_id) REFERENCES alerts(alert_id)
);

-- 12. Ingestion Audit Logs Table
CREATE TABLE IF NOT EXISTS ingestion_audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ingestion_id TEXT UNIQUE NOT NULL,
    filename TEXT NOT NULL,
    file_hash_sha256 TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'UPLOAD_SURVEILLANCE_CSV',
    records_total INTEGER NOT NULL DEFAULT 0,
    records_valid INTEGER NOT NULL DEFAULT 0,
    records_invalid INTEGER NOT NULL DEFAULT 0,
    records_duplicates INTEGER NOT NULL DEFAULT 0,
    new_records_inserted INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    stage_completed TEXT NOT NULL,
    validation_errors TEXT,
    predictions_recomputed INTEGER NOT NULL DEFAULT 0,
    duration_seconds REAL NOT NULL DEFAULT 0.0,
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_hash ON ingestion_audit_logs(file_hash_sha256);
CREATE INDEX IF NOT EXISTS idx_audit_status ON ingestion_audit_logs(status);

-- 13. System Metadata Table
CREATE TABLE IF NOT EXISTS system_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""
