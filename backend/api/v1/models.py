"""
Model Registry API Endpoints for JEEVAN-DRISHTI
===============================================
GET /api/v1/models
GET /api/v1/models/metrics
"""

import json
from typing import Optional
from fastapi import APIRouter, Query
from backend.config import settings
from backend.database.connection import get_db_connection
from backend.schemas.api_models import (
    ModelMetricsResponse,
    ModelRegistryItem,
    ModelRegistryListResponse,
)

router = APIRouter(prefix="", tags=["Model Registry & Governance"])


@router.get("/models", response_model=ModelRegistryListResponse)
def list_registered_models(
    status: Optional[str] = Query(None, description="Filter by operational status"),
    disease: Optional[str] = Query(None, description="Filter by target disease"),
) -> ModelRegistryListResponse:
    """Retrieve full catalog of registered analytical engines and operational statuses."""
    query = "SELECT model_id, model_name, model_type, disease, supported_districts_count, operational_status, primary_metric_name, primary_metric_value, artifact_path, version FROM model_registry"
    conditions = []
    params = []

    if status:
        conditions.append("UPPER(operational_status) = UPPER(?)")
        params.append(status)
    if disease:
        conditions.append("UPPER(disease) LIKE UPPER(?)")
        params.append(f"%{disease}%")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY id ASC"

    with get_db_connection() as conn:
        rows = conn.execute(query, params).fetchall()

    items = [
        ModelRegistryItem(
            model_id=r["model_id"],
            model_name=r["model_name"],
            model_type=r["model_type"],
            disease=r["disease"],
            supported_districts_count=r["supported_districts_count"],
            operational_status=r["operational_status"],
            primary_metric_name=r["primary_metric_name"],
            primary_metric_value=float(r["primary_metric_value"]) if r["primary_metric_value"] is not None else None,
            artifact_path=r["artifact_path"],
            version=r["version"],
        )
        for r in rows
    ]

    return ModelRegistryListResponse(
        total_models=len(items),
        models=items,
        drift_status="WATCHLIST (Monsoon feature shift monitored)",
        governance_notice=(
            "No model is labeled production-ready without clinical review. "
            "All engines operate under transparent status categories (Validated prototype, Experimental, "
            "Data limited, Research, Unsupported)."
        ),
    )


@router.get("/models/metrics", response_model=ModelMetricsResponse)
def get_model_evaluation_metrics() -> ModelMetricsResponse:
    """Retrieve standardized multi-split evaluation metrics and baseline comparisons."""
    catalog_csv = settings.REPORTS_DIR / "model_evaluation" / "model_evaluation_metrics_catalog.csv"
    baseline_csv = settings.REPORTS_DIR / "model_evaluation" / "model_baseline_comparisons.csv"
    overfit_csv = settings.REPORTS_DIR / "model_evaluation" / "overfitting_and_leakage_audit.csv"

    import pandas as pd
    cat_records = pd.read_csv(catalog_csv).head(20).to_dict(orient="records") if catalog_csv.is_file() else []
    base_records = pd.read_csv(baseline_csv).to_dict(orient="records") if baseline_csv.is_file() else []
    over_records = pd.read_csv(overfit_csv).to_dict(orient="records") if overfit_csv.is_file() else []

    return ModelMetricsResponse(
        total_models_evaluated=len(cat_records),
        metrics_catalog=cat_records,
        baseline_comparisons=base_records,
        overfitting_gaps=over_records,
    )
