"""
Alerts & Human Review API Endpoints for JEEVAN-DRISHTI
======================================================
GET /api/v1/alerts
POST /api/v1/alerts/{alert_id}/review
"""

import uuid
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from backend.database.connection import get_db_connection
from backend.middleware.auth import verify_reviewer_token
from backend.schemas.api_models import (
    AlertItem,
    AlertListResponse,
    AlertReviewRequest,
    AlertReviewResponse,
    AnalyticalMetadata,
    PredictionType,
)

router = APIRouter(prefix="", tags=["Early Warning Alerts & Field Review"])


@router.get("/alerts", response_model=AlertListResponse)
def list_alerts(
    district: Optional[str] = Query(None, description="Filter by district name"),
    disease: Optional[str] = Query(None, description="Filter by disease name"),
    alert_status: Optional[str] = Query(None, description="Filter by review status (PENDING_REVIEW, CONFIRMED_OUTBREAK, etc.)"),
    severity: Optional[str] = Query(None, description="Filter by severity (HIGH, MODERATE)"),
    limit: int = Query(50, ge=1, le=200, description="Pagination limit"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
) -> AlertListResponse:
    """Retrieve early warning alerts requiring human field verification."""
    query = """
    SELECT alert_id, district_id, district_name, disease_name, reporting_period,
           alert_type, severity, trigger_reason, observed_cases, threshold_value,
           alert_status, human_verified, created_at
    FROM alerts
    """
    conditions = []
    params = []

    if district:
        conditions.append("UPPER(district_name) = UPPER(?)")
        params.append(district)
    if disease:
        conditions.append("UPPER(disease_name) = UPPER(?)")
        params.append(disease)
    if alert_status:
        conditions.append("alert_status = ?")
        params.append(alert_status)
    if severity:
        conditions.append("severity = ?")
        params.append(severity)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with get_db_connection() as conn:
        rows = conn.execute(query, params).fetchall()
        pending_cnt = conn.execute("SELECT COUNT(*) FROM alerts WHERE alert_status = 'PENDING_REVIEW'").fetchone()[0]

    items = [
        AlertItem(
            alert_id=r["alert_id"],
            district_name=r["district_name"],
            disease_name=r["disease_name"],
            reporting_period=r["reporting_period"],
            alert_type=r["alert_type"],
            severity=r["severity"],
            trigger_reason=r["trigger_reason"],
            observed_cases=float(r["observed_cases"]),
            threshold_value=float(r["threshold_value"]) if r["threshold_value"] is not None else None,
            alert_status=r["alert_status"],
            human_verified=bool(r["human_verified"]),
            created_at=r["created_at"],
        )
        for r in rows
    ]

    return AlertListResponse(
        total_alerts=len(items),
        pending_review_count=pending_cnt,
        alerts=items,
        metadata=AnalyticalMetadata(
            prediction_type=PredictionType.EWARS_THRESHOLD_SIGNAL,
            model_version="ewars-v1.0.0",
            limitations="Alerts indicate statistical threshold breach; field verification mandatory before treating as confirmed outbreak.",
        ),
    )


@router.post(
    "/alerts/{alert_id}/review",
    response_model=AlertReviewResponse,
    status_code=status.HTTP_200_OK,
)
def review_alert(
    alert_id: str,
    payload: AlertReviewRequest,
    reviewer_identity: str = Depends(verify_reviewer_token),
) -> AlertReviewResponse:
    """
    Submit an official public-health field review action for an early warning alert.
    Protected endpoint requiring valid X-Reviewer-Key or Bearer token.
    Persists reviewer action into review_actions audit log.
    """
    now_str = datetime.now().isoformat()
    review_uuid = f"REV_{uuid.uuid4().hex[:10].upper()}"

    with get_db_connection() as conn:
        # Check alert exists
        alert = conn.execute("SELECT alert_id, district_name, disease_name FROM alerts WHERE alert_id = ?", (alert_id,)).fetchone()
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Alert '{alert_id}' not found.",
            )

        # Update alert status
        conn.execute(
            """
            UPDATE alerts
            SET alert_status = ?, human_verified = 1
            WHERE alert_id = ?
            """,
            (payload.action_taken.value, alert_id),
        )

        # Insert audit trail into review_actions
        conn.execute(
            """
            INSERT INTO review_actions
            (review_id, alert_id, reviewer_id, action_taken, comments, review_timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                review_uuid,
                alert_id,
                payload.reviewer_id,
                payload.action_taken.value,
                payload.comments,
                now_str,
            ),
        )

    return AlertReviewResponse(
        review_id=review_uuid,
        alert_id=alert_id,
        status="RECORDED",
        action_taken=payload.action_taken.value,
        reviewer_id=payload.reviewer_id,
        review_timestamp=now_str,
        message=f"Alert {alert_id} successfully updated to {payload.action_taken.value}. Audit record logged.",
    )
