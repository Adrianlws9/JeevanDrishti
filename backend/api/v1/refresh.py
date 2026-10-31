"""
2-Second Dashboard Refresh API Endpoints for JEEVAN-DRISHTI
===========================================================
Phase 18: Automated Data Ingestion & 2-Second Dashboard Refresh

GET /api/v1/refresh/summary
GET /api/v1/refresh/status
"""

import time
from fastapi import APIRouter
from backend.schemas.api_models import RefreshSummaryResponse
from backend.services.refresh_service import DashboardRefreshService

router = APIRouter(prefix="", tags=["Dashboard Refresh & Live State"])


@router.get("/refresh/summary", response_model=RefreshSummaryResponse)
def get_dashboard_refresh_summary() -> RefreshSummaryResponse:
    """
    High-frequency 2-second dashboard poll endpoint.
    Returns real-time system state, distinct timestamp lineages, and alert counts in <10ms.
    Does NOT trigger heavy recomputation or model retraining.
    """
    return DashboardRefreshService.get_summary_snapshot()


@router.get("/refresh/status")
def get_refresh_sync_status():
    """
    Detailed metadata and sync cadence verification endpoint.
    Exposes explicit audit of poll frequency vs source data arrival frequency.
    """
    summary = DashboardRefreshService.get_summary_snapshot()
    return {
        "dashboard_ui_refresh_cadence": f"Every {summary.refresh_interval_seconds} seconds",
        "source_data_update_frequency": summary.source_frequency_label,
        "latest_source_observation": summary.source_observation_date,
        "timestamps": {
            "last_dashboard_poll_served": summary.last_dashboard_refresh,
            "last_successful_ingestion": summary.last_successful_ingestion,
            "last_processing_completion": summary.last_processing_completion,
            "last_prediction_generation": summary.last_prediction_generation,
        },
        "model_version": summary.model_version,
        "active_models_recomputed_on_poll": False,
        "policy": "Predictive models run inference only on ingestion of verified new source records; never on UI polls.",
    }
