"""
Diseases API Endpoints for JEEVAN-DRISHTI
=========================================
GET /api/v1/diseases
GET /api/v1/diseases/{disease_id}/trends
"""

from fastapi import APIRouter, HTTPException, status
from backend.database.connection import get_db_connection
from backend.schemas.api_models import (
    AnalyticalMetadata,
    DiseaseItem,
    DiseaseListResponse,
    DiseaseTrendPoint,
    DiseaseTrendResponse,
    PredictionType,
)

router = APIRouter(prefix="", tags=["Diseases"])


@router.get("/diseases", response_model=DiseaseListResponse)
def list_diseases() -> DiseaseListResponse:
    """List all monitored infectious diseases under Maharashtra IDSP surveillance."""
    with get_db_connection() as conn:
        rows = conn.execute("SELECT disease_id, disease_name, category, is_priority, transmission_route FROM diseases ORDER BY disease_name ASC").fetchall()

    items = [
        DiseaseItem(
            disease_id=r["disease_id"],
            disease_name=r["disease_name"],
            category=r["category"],
            is_priority=bool(r["is_priority"]),
            transmission_route=r["transmission_route"],
        )
        for r in rows
    ]
    return DiseaseListResponse(total_diseases=len(items), diseases=items)


@router.get("/diseases/{disease_id}/trends", response_model=DiseaseTrendResponse)
def get_disease_trends(disease_id: str) -> DiseaseTrendResponse:
    """
    Retrieve historical weekly surveillance trends, peak volumes, and CFR
    by disease ID (e.g. DIS_001) or disease name (e.g. Dengue).
    """
    with get_db_connection() as conn:
        dis_row = conn.execute(
            "SELECT disease_id, disease_name FROM diseases WHERE disease_id = ? OR UPPER(disease_name) = UPPER(?)",
            (disease_id, disease_id),
        ).fetchone()

        if not dis_row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Disease '{disease_id}' not found in IDSP pathogen catalog.",
            )

        dis_name = dis_row["disease_name"]

        trend_rows = conn.execute(
            """
            SELECT reporting_period, calendar_year, week_number, SUM(observed_cases) as cases, SUM(observed_deaths) as deaths
            FROM observations
            WHERE disease_name = ?
            GROUP BY reporting_period, calendar_year, week_number
            ORDER BY reporting_period ASC
            """,
            (dis_name,),
        ).fetchall()

    if not trend_rows:
        return DiseaseTrendResponse(
            disease_name=dis_name,
            total_historical_weeks=0,
            peak_weekly_cases=0.0,
            historical_cfr_pct=0.0,
            trends=[],
            metadata=AnalyticalMetadata(disease=dis_name, prediction_type=PredictionType.OBSERVED),
        )

    points = [
        DiseaseTrendPoint(
            reporting_period=r["reporting_period"],
            calendar_year=r["calendar_year"] or 2023,
            week_number=r["week_number"] or 1,
            observed_cases=float(r["cases"]),
            observed_deaths=float(r["deaths"]),
        )
        for r in trend_rows
    ]

    total_cases = sum(p.observed_cases for p in points)
    total_deaths = sum(p.observed_deaths for p in points)
    cfr = round((total_deaths / max(total_cases, 1.0)) * 100.0, 3)
    peak = max(p.observed_cases for p in points)

    return DiseaseTrendResponse(
        disease_name=dis_name,
        total_historical_weeks=len(points),
        peak_weekly_cases=float(peak),
        historical_cfr_pct=cfr,
        trends=points,
        metadata=AnalyticalMetadata(
            disease=dis_name,
            prediction_type=PredictionType.OBSERVED,
            data_quality_status="IDSP_VERIFIED",
        ),
    )
