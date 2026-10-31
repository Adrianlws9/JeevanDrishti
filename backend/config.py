"""
JEEVAN-DRISHTI FastAPI Backend Configuration
===========================================
Phase 17: FastAPI Backend & Analytical Engine Integration
"""

import os
from pathlib import Path
from typing import List

# Paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJ_ROOT = BACKEND_DIR.parent
DATA_DIR = PROJ_ROOT / "data"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
DB_DIR = BACKEND_DIR / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)
SQLITE_DB_PATH = DB_DIR / "jeevan_drishti.db"


class Settings:
    """Application configuration settings."""

    PROJECT_NAME: str = "JEEVAN-DRISHTI Public Health API"
    PROJECT_DESCRIPTION: str = (
        "AI-Powered Disease Surveillance, Outbreak Early Warning, and Public-Health "
        "Intelligence Platform for Maharashtra, India."
    )
    API_VERSION: str = "v1"
    API_V1_PREFIX: str = "/api/v1"

    # Security & Governance
    REVIEWER_API_KEY: str = os.getenv("JEEVAN_REVIEWER_KEY", "jeevan-drishti-reviewer-key-2026")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1")

    # Paths
    PROJ_ROOT: Path = PROJ_ROOT
    DATA_DIR: Path = DATA_DIR
    MODELS_DIR: Path = MODELS_DIR
    REPORTS_DIR: Path = REPORTS_DIR

    # Database
    DATABASE_PATH: Path = SQLITE_DB_PATH
    DATABASE_URL: str = f"sqlite:///{SQLITE_DB_PATH}"

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "*",
    ]

    # Model Artifact Paths
    RISK_MODEL_PATH: Path = MODELS_DIR / "risk_classification" / "trained_model.pkl"
    FORECAST_MODEL_PATH: Path = MODELS_DIR / "disease_forecasting" / "forecasting_model.pkl"
    ANOMALY_MODEL_PATH: Path = MODELS_DIR / "anomaly_detection" / "anomaly_detector.pkl"
    MODEL_REGISTRY_JSON: Path = MODELS_DIR / "registry" / "model_registry.json"
    THRESHOLD_REGISTRY_JSON: Path = MODELS_DIR / "registry" / "threshold_registry.json"
    FUSED_INTELLIGENCE_CSV: Path = DATA_DIR / "fused_intelligence" / "fused_district_pooled_intelligence_36.csv"


settings = Settings()
