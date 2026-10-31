"""
Automated Data Ingestion API Endpoints for JEEVAN-DRISHTI
=========================================================
Phase 18: Automated Data Ingestion & 2-Second Dashboard Refresh

POST /api/v1/ingestion/upload
POST /api/v1/ingestion/trigger
GET  /api/v1/ingestion/audit-logs
GET  /api/v1/ingestion/status
"""

from typing import List, Optional
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from backend.database.connection import get_db_connection
from backend.ingestion.config import IngestionConfig
from backend.ingestion.pipeline import DataIngestionPipeline
from backend.schemas.api_models import AuditLogItem, AuditLogListResponse, IngestionResponse

router = APIRouter(prefix="", tags=["Automated Data Ingestion"])


@router.post("/ingestion/upload", response_model=IngestionResponse)
async def upload_surveillance_dataset(
    file: UploadFile = File(..., description="Surveillance CSV dataset to ingest")
) -> IngestionResponse:
    """
    Execute 11-stage automated ingestion pipeline on uploaded surveillance CSV.
    Validates schema, checks SHA-256 for duplicate avoidance, cleanses records,
    updates database, and conditionally executes model inference for new periods.
    """
    if not file.filename.lower().endswith((".csv", ".txt")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Please upload a valid CSV file.",
        )

    content = await file.read()
    if len(content) > IngestionConfig.MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {IngestionConfig.MAX_FILE_SIZE_BYTES // (1024*1024)}MB.",
        )

    pipeline = DataIngestionPipeline()
    response = pipeline.process_file_content(content=content, filename=file.filename)
    return response


@router.post("/ingestion/trigger", response_model=List[IngestionResponse])
def trigger_staged_ingestion() -> List[IngestionResponse]:
    """
    Scan the staging folder for newly arrived automated data feeds and process them.
    """
    IngestionConfig.ensure_directories()
    staged_files = list(IngestionConfig.STAGING_DIR.glob("*.csv"))

    if not staged_files:
        return []

    pipeline = DataIngestionPipeline()
    results = []
    for fpath in staged_files:
        try:
            with open(fpath, "rb") as f:
                content = f.read()
            res = pipeline.process_file_content(content=content, filename=fpath.name, source_type="STAGED_FEED_CSV")
            results.append(res)
            # Move processed file to archive
            archive_path = IngestionConfig.ARCHIVE_DIR / fpath.name
            fpath.replace(archive_path)
        except Exception as e:
            results.append(
                IngestionResponse(
                    ingestion_id="ERR_STAGED",
                    filename=fpath.name,
                    file_hash_sha256="",
                    status="PROCESSING_FAILED",
                    stage_reached="TRIGGER",
                    records_total=0,
                    records_valid=0,
                    records_invalid=0,
                    records_duplicates=0,
                    new_records_inserted=0,
                    predictions_updated=False,
                    validation_errors=[str(e)],
                    execution_duration_sec=0.0,
                    started_at="",
                    completed_at="",
                    message=f"Failed to process staged file: {str(e)}",
                )
            )

    return results


@router.get("/ingestion/audit-logs", response_model=AuditLogListResponse)
def get_ingestion_audit_logs(
    limit: int = Query(50, ge=1, le=200, description="Max logs to return"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    status: Optional[str] = Query(None, description="Filter by status (SUCCESS, DUPLICATE_SKIPPED, etc.)"),
) -> AuditLogListResponse:
    """
    Query the complete immutable historical audit trail of data ingestion runs.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM ingestion_audit_logs"
        params = []
        if status:
            query += " WHERE status = ?"
            params.append(status.upper())

        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = cursor.execute(query, params).fetchall()

        count_query = "SELECT COUNT(*) FROM ingestion_audit_logs"
        count_params = []
        if status:
            count_query += " WHERE status = ?"
            count_params.append(status.upper())
        total = cursor.execute(count_query, count_params).fetchone()[0]

    log_items = [
        AuditLogItem(
            id=r["id"],
            ingestion_id=r["ingestion_id"],
            filename=r["filename"],
            file_hash_sha256=r["file_hash_sha256"],
            file_size_bytes=r["file_size_bytes"],
            source_type=r["source_type"],
            records_total=r["records_total"],
            records_valid=r["records_valid"],
            records_invalid=r["records_invalid"],
            records_duplicates=r["records_duplicates"],
            new_records_inserted=r["new_records_inserted"],
            status=r["status"],
            stage_completed=r["stage_completed"],
            validation_errors=r["validation_errors"],
            predictions_recomputed=bool(r["predictions_recomputed"]),
            duration_seconds=float(r["duration_seconds"]),
            started_at=r["started_at"],
            completed_at=r["completed_at"],
        )
        for r in rows
    ]

    return AuditLogListResponse(total_logs=total, logs=log_items)


@router.get("/ingestion/status")
def get_ingestion_pipeline_status():
    """
    Return operational readiness, storage health, and staging counts.
    """
    staged_count = len(list(IngestionConfig.STAGING_DIR.glob("*.csv")))
    uploads_count = len(list(IngestionConfig.UPLOAD_DIR.glob("*.csv")))

    with get_db_connection() as conn:
        cursor = conn.cursor()
        total_runs = cursor.execute("SELECT COUNT(*) FROM ingestion_audit_logs").fetchone()[0]
        successful_runs = cursor.execute("SELECT COUNT(*) FROM ingestion_audit_logs WHERE status = 'SUCCESS'").fetchone()[0]
        duplicate_skips = cursor.execute("SELECT COUNT(*) FROM ingestion_audit_logs WHERE status = 'DUPLICATE_SKIPPED'").fetchone()[0]

    return {
        "pipeline_status": "ONLINE",
        "staged_files_pending": staged_count,
        "historical_uploads_archived": uploads_count,
        "total_ingestion_runs": total_runs,
        "successful_ingestions": successful_runs,
        "duplicate_skips": duplicate_skips,
        "supported_formats": [".csv"],
        "max_file_size_mb": IngestionConfig.MAX_FILE_SIZE_BYTES // (1024 * 1024),
    }
