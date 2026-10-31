"""
Districts API Endpoints for JEEVAN-DRISHTI
==========================================
GET /api/v1/districts
GET /api/v1/districts/{district_id}
"""

from fastapi import APIRouter, HTTPException, status
from backend.database.connection import get_db_connection
from backend.schemas.api_models import (
    AnalyticalMetadata,
    DistrictDetailResponse,
    DistrictItem,
    DistrictListResponse,
    PredictionType,
)

router = APIRouter(prefix="", tags=["Districts"])


@router.get("/districts", response_model=DistrictListResponse)
def list_districts(active_only: bool = False) -> DistrictListResponse:
    """List all districts in Maharashtra with division and operational status."""
    query = "SELECT district_id, district_name, division, is_active, notes FROM districts"
    if active_only:
        query += " WHERE is_active = 1"
    query += " ORDER BY district_name ASC"

    with get_db_connection() as conn:
        rows = conn.execute(query).fetchall()

    items = [
        DistrictItem(
            district_id=r["district_id"],
            district_name=r["district_name"],
            division=r["division"],
            is_active=bool(r["is_active"]),
            notes=r["notes"],
        )
        for r in rows
    ]
    active_cnt = sum(1 for it in items if it.is_active)

    return DistrictListResponse(
        total_districts=len(items),
        active_count=active_cnt,
        districts=items,
    )


@router.get("/districts/{district_id}", response_model=DistrictDetailResponse)
def get_district_details(district_id: str) -> DistrictDetailResponse:
    """
    Retrieve comprehensive district surveillance profile by ID (e.g. DIST_PUNE)
    or standardized district name (e.g. Pune).
    """
    with get_db_connection() as conn:
        # Match by district_id or district_name
        query = """
        SELECT district_id, district_name, division, is_active, notes
        FROM districts
        WHERE district_id = ? OR UPPER(district_name) = UPPER(?)
        """
        row = conn.execute(query, (district_id, district_id)).fetchone()
        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"District '{district_id}' not found in Maharashtra surveillance territory.",
            )

        d_item = DistrictItem(
            district_id=row["district_id"],
            district_name=row["district_name"],
            division=row["division"],
            is_active=bool(row["is_active"]),
            notes=row["notes"],
        )

        # Unsupported district safeguard
        if not d_item.is_active:
            return DistrictDetailResponse(
                district=d_item,
                historical_outbreak_weeks=0,
                total_observed_cases=0.0,
                total_observed_deaths=0.0,
                recent_risk_tier="UNSUPPORTED",
                recent_forecast_cases=None,
                is_supported=False,
                unsupported_reason="Zero historical IDSP outbreak records in 2023-2025; unsupported for prediction.",
                metadata=AnalyticalMetadata(
                    district=d_item.district_name,
                    prediction_type=PredictionType.OBSERVED,
                    limitations="District not covered by historical IDSP outbreak reporting.",
                ),
            )

        # Active district statistics
        stats_query = """
        SELECT COUNT(*) as n_weeks, COALESCE(SUM(observed_cases), 0) as tot_cases, COALESCE(SUM(observed_deaths), 0) as tot_deaths
        FROM observations
        WHERE district_id = ?
        """
        st = conn.execute(stats_query, (d_item.district_id,)).fetchone()

        # Recent risk assessment
        risk_row = conn.execute(
            "SELECT risk_tier FROM risk_assessments WHERE district_id = ? ORDER BY id DESC LIMIT 1",
            (d_item.district_id,),
        ).fetchone()

        # Recent forecast
        fc_row = conn.execute(
            "SELECT predicted_cases FROM forecasts WHERE district_id = ? AND horizon_days = 7 ORDER BY id DESC LIMIT 1",
            (d_item.district_id,),
        ).fetchone()

    return DistrictDetailResponse(
        district=d_item,
        historical_outbreak_weeks=st["n_weeks"],
        total_observed_cases=float(st["tot_cases"]),
        total_observed_deaths=float(st["tot_deaths"]),
        recent_risk_tier=risk_row["risk_tier"] if risk_row else "NORMAL_BASELINE",
        recent_forecast_cases=float(fc_row["predicted_cases"]) if fc_row else None,
        is_supported=True,
        unsupported_reason=None,
        metadata=AnalyticalMetadata(
            district=d_item.district_name,
            prediction_type=PredictionType.OBSERVED,
            data_quality_status="IDSP_VERIFIED",
        ),
    )
