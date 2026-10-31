"""
API v1 Router Aggregator for JEEVAN-DRISHTI
===========================================
Phase 17: FastAPI Backend & Analytical Engine Integration
"""

from fastapi import APIRouter
from backend.api.v1.alerts import router as alerts_router
from backend.api.v1.anomalies import router as anomalies_router
from backend.api.v1.climate import router as climate_router
from backend.api.v1.data_quality import router as data_quality_router
from backend.api.v1.diseases import router as diseases_router
from backend.api.v1.districts import router as districts_router
from backend.api.v1.ewars import router as ewars_router
from backend.api.v1.explainability import router as explainability_router
from backend.api.v1.forecasting import router as forecasting_router
from backend.api.v1.geospatial import router as geospatial_router
from backend.api.v1.models import router as models_router
from backend.api.v1.overview import router as overview_router
from backend.api.v1.resources import router as resources_router
from backend.api.v1.risk import router as risk_router
from backend.api.v1.scenarios import router as scenarios_router
from backend.api.v1.refresh import router as refresh_router
from backend.api.v1.ingestion import router as ingestion_router
from backend.api.v1.ui import router as ui_router

api_v1_router = APIRouter(prefix="/api/v1")

# Mount modular routers
api_v1_router.include_router(overview_router)
api_v1_router.include_router(districts_router)
api_v1_router.include_router(diseases_router)
api_v1_router.include_router(risk_router)
api_v1_router.include_router(forecasting_router)
api_v1_router.include_router(anomalies_router)
api_v1_router.include_router(climate_router)
api_v1_router.include_router(geospatial_router)
api_v1_router.include_router(resources_router)
api_v1_router.include_router(explainability_router)
api_v1_router.include_router(models_router)
api_v1_router.include_router(data_quality_router)
api_v1_router.include_router(alerts_router)
api_v1_router.include_router(scenarios_router)
api_v1_router.include_router(ewars_router)
api_v1_router.include_router(refresh_router)
api_v1_router.include_router(ingestion_router)
api_v1_router.include_router(ui_router)
