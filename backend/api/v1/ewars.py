"""
WHO EWARS-Inspired Threshold API Endpoints for JEEVAN-DRISHTI
=============================================================
GET /api/v1/ewars/thresholds
POST /api/v1/ewars/evaluate
"""

import json
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
from backend.config import settings
from backend.engines.engine_registry import get_engine_registry
from backend.schemas.api_models import AnalyticalMetadata, PredictionType
from src.anomaly_detection.ewars_thresholds import EWARS_GOVERNANCE_DISCLAIMER

router = APIRouter(prefix="/ewars", tags=["WHO EWARS-Inspired Early Warning"])


class EWARSEvaluationRequest(BaseModel):
    district: str = Field(..., description="Target Maharashtra district")
    disease: str = Field(..., description="Target pathogen")
    reporting_period: str = Field(..., description="Reporting period (e.g., 2025-W32)")
    observed_cases: float = Field(..., ge=0, description="Reported case volume")
    prior_cases_history: Optional[List[float]] = Field(None, description="Recent weekly cases (past 2-4 weeks)")
    lagged_alerts_history: Optional[List[bool]] = Field(None, description="Prior week alert flags")
    season: Optional[str] = Field(None, description="Epidemiological season")


@router.get("/thresholds")
def list_ewars_threshold_rules(disease: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve registered WHO EWARS-inspired threshold configurations and calibration records."""
    thresh_file = settings.THRESHOLD_REGISTRY_JSON
    if not thresh_file.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Threshold registry not found on disk.",
        )

    with open(thresh_file, "r", encoding="utf-8") as f:
        registry = json.load(f)

    thresholds = registry.get("thresholds", [])
    if disease:
        thresholds = [t for t in thresholds if disease.lower() in t.get("disease", "").lower()]

    return {
        "version": registry.get("version"),
        "governance_notice": registry.get("governance_notice", EWARS_GOVERNANCE_DISCLAIMER),
        "calibration_period": registry.get("calibration_period"),
        "total_rules": len(thresholds),
        "thresholds": thresholds,
    }


@router.post("/evaluate")
def evaluate_ewars_observation(payload: EWARSEvaluationRequest) -> Dict[str, Any]:
    """
    Score live weekly observation against all 5 WHO EWARS-inspired threshold methods:
    1. Fixed disease-specific thresholds
    2. Moving-average threshold (4w)
    3. Historical seasonal threshold
    4. Lagged alert persistence
    5. Consecutive weekly increase detection
    """
    registry = get_engine_registry()
    if not registry.anomaly_service or not registry.anomaly_service.ewars_engine:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="EWARS threshold engine is not available.",
        )

    alert_output = registry.anomaly_service.evaluate_ewars_thresholds(
        district=payload.district,
        disease=payload.disease,
        reporting_period=payload.reporting_period,
        observed_cases=payload.observed_cases,
        prior_cases_history=payload.prior_cases_history,
        lagged_alerts_history=payload.lagged_alerts_history,
        season=payload.season,
    )

    return {
        "evaluation": alert_output,
        "metadata": AnalyticalMetadata(
            district=payload.district,
            disease=payload.disease,
            reporting_period=payload.reporting_period,
            prediction_type=PredictionType.EWARS_THRESHOLD_SIGNAL,
            model_version="ewars-v1.0.0",
            limitations=EWARS_GOVERNANCE_DISCLAIMER,
        ).model_dump(),
    }
