"""
Ingestion Configuration & Path Management
=========================================
Phase 18: Automated Data Ingestion & 2-Second Dashboard Refresh
"""

from pathlib import Path
from backend.config import settings


class IngestionConfig:
    """Paths and settings for the automated ingestion pipeline."""

    BASE_INGESTION_DIR: Path = settings.DATA_DIR / "ingestion"
    UPLOAD_DIR: Path = BASE_INGESTION_DIR / "uploads"
    STAGING_DIR: Path = BASE_INGESTION_DIR / "staging"
    QUARANTINE_DIR: Path = BASE_INGESTION_DIR / "quarantine"
    ARCHIVE_DIR: Path = BASE_INGESTION_DIR / "archive"

    # Schema specifications
    REQUIRED_COLUMNS_CANDIDATES = {
        "district": ["district", "district_name", "District", "DistrictName"],
        "disease": ["disease", "disease_name", "Disease", "DiseaseName"],
        "cases": ["observed_cases", "cases", "Cases", "ObservedCases", "num_cases"],
        "period": ["reporting_period", "ReportingPeriod", "period", "epiwk", "week"],
        "deaths": ["observed_deaths", "deaths", "Deaths", "ObservedDeaths", "num_deaths"],
        "year": ["calendar_year", "year", "Year", "report_year"],
        "week": ["week_number", "week", "Week", "report_week"],
    }

    # Quality constraints
    MAX_PLAUSIBLE_WEEKLY_CASES = 50000.0
    MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

    @classmethod
    def ensure_directories(cls):
        """Create necessary ingestion directories if they do not exist."""
        for p in [cls.BASE_INGESTION_DIR, cls.UPLOAD_DIR, cls.STAGING_DIR, cls.QUARANTINE_DIR, cls.ARCHIVE_DIR]:
            p.mkdir(parents=True, exist_ok=True)


# Initialize paths on module import
IngestionConfig.ensure_directories()
