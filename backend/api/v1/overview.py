"""
Overview API Endpoint for JEEVAN-DRISHTI
========================================
GET /api/v1/overview
"""

from fastapi import APIRouter
from backend.config import settings
from backend.database.connection import get_db_connection
from backend.schemas.api_models import AnalyticalMetadata, OverviewResponse, PredictionType

router = APIRouter(prefix="", tags=["Platform Overview"])


@router.get("/overview", response_model=OverviewResponse)
def get_platform_overview() -> OverviewResponse:
    """Return platform-wide disease surveillance summary and operational status."""
    with get_db_connection() as conn:
        active_districts_cnt = conn.execute("SELECT COUNT(*) FROM districts WHERE is_active = 1").fetchone()[0]
        obs_cnt = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        alerts_cnt = conn.execute("SELECT COUNT(*) FROM alerts WHERE alert_status = 'PENDING_REVIEW'").fetchone()[0]
        models_cnt = conn.execute("SELECT COUNT(*) FROM model_registry").fetchone()[0]
        diseases_cnt = conn.execute("SELECT COUNT(*) FROM diseases").fetchone()[0]

    return OverviewResponse(
        platform_name=settings.PROJECT_NAME,
        version=settings.API_VERSION,
        status="OPERATIONAL",
        surveillance_territory="Maharashtra State, India",
        total_active_districts=active_districts_cnt,
        unsupported_districts=["Mumbai City", "Ratnagiri"],
        total_weekly_observations=obs_cnt,
        active_outbreak_alerts_count=alerts_cnt,
        registered_models_count=models_cnt,
        monitored_diseases_count=diseases_cnt,
        system_drift_status="WATCHLIST (Monsoon feature shift monitored)",
        governance_notice=(
            "JEEVAN-DRISHTI is an epidemiological decision-support platform. All forecasts, risk estimates, "
            "and threshold alerts require human public-health verification and do not substitute clinical authority."
        ),
        metadata=AnalyticalMetadata(
            prediction_type=PredictionType.OBSERVED,
            data_quality_status="IDSP_VERIFIED_2023_2025",
            model_version="platform-v1.0.0",
        ),
    )
