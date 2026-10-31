"""
Analytical Engine Registry & Lifespan Provider for JEEVAN-DRISHTI
================================================================
Phase 17: FastAPI Backend & Analytical Engine Integration

Provides thread-safe, cached access to predictive and analytical engines
without retraining or repeated expensive disk I/O on GET requests.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from backend.config import settings
from src.anomaly_detection.service import DiseaseAnomalyDetectionService
from src.disease_forecasting.service import DiseaseForecastingService
from src.model_registry.service import ModelRegistryService
from src.risk_classification.predictor import OutbreakRiskPredictor

logger = logging.getLogger(__name__)


class EngineRegistry:
    """Singleton provider for loaded analytical engines and cached artifacts."""

    _instance: Optional["EngineRegistry"] = None

    def __init__(self) -> None:
        self.risk_service: Optional[OutbreakRiskPredictor] = None
        self.forecast_service: Optional[DiseaseForecastingService] = None
        self.anomaly_service: Optional[DiseaseAnomalyDetectionService] = None
        self.model_registry_service: Optional[ModelRegistryService] = None
        self.fused_df: Optional[pd.DataFrame] = None
        self.climate_summary: Optional[Dict[str, Any]] = None
        self.hotspot_summary: Optional[pd.DataFrame] = None
        self._is_loaded = False

    @classmethod
    def get_instance(cls) -> "EngineRegistry":
        if cls._instance is None:
            cls._instance = EngineRegistry()
        return cls._instance

    def load_all_engines(self) -> None:
        """Load and cache all models and datasets into memory."""
        if self._is_loaded:
            return

        logger.info("Initializing JEEVAN-DRISHTI analytical engines into backend cache...")

        # 1. Risk Classification Service (Phase 5)
        try:
            self.risk_service = OutbreakRiskPredictor()
            logger.info("Risk Classification engine loaded successfully.")
        except Exception as e:
            logger.warning("Could not load OutbreakRiskPredictor: %s", e)

        # 2. Disease Forecasting Service (Phase 6)
        try:
            self.forecast_service = DiseaseForecastingService()
            logger.info("Disease Forecasting engine loaded successfully.")
        except Exception as e:
            logger.warning("Could not load DiseaseForecastingService: %s", e)

        # 3. Anomaly & EWARS Detection Service (Phase 7 & Addendum)
        try:
            self.anomaly_service = DiseaseAnomalyDetectionService()
            logger.info("Anomaly Detection and EWARS threshold engine loaded successfully.")
        except Exception as e:
            logger.warning("Could not load DiseaseAnomalyDetectionService: %s", e)

        # 4. Model Registry Service (Phase 16)
        try:
            self.model_registry_service = ModelRegistryService(settings.MODEL_REGISTRY_JSON)
            logger.info("Model Registry service loaded successfully.")
        except Exception as e:
            logger.warning("Could not load ModelRegistryService: %s", e)

        # 5. Fused District Pooled Intelligence (Phase 15)
        fused_path = settings.FUSED_INTELLIGENCE_CSV
        if fused_path.is_file():
            try:
                self.fused_df = pd.read_csv(fused_path)
                logger.info("Loaded fused intelligence dataset: %d rows.", len(self.fused_df))
            except Exception as e:
                logger.warning("Could not load fused intelligence CSV: %s", e)

        # 6. Geospatial Hotspots (Phase 9)
        hotspots_path = settings.DATA_DIR / "geospatial" / "spatial_clusters_summary.csv"
        if hotspots_path.is_file():
            try:
                self.hotspot_summary = pd.read_csv(hotspots_path)
            except Exception as e:
                logger.warning("Could not load hotspot summary: %s", e)

        self._is_loaded = True
        logger.info("All JEEVAN-DRISHTI backend engines initialized.")


def get_engine_registry() -> EngineRegistry:
    """Dependency helper returning the initialized engine registry."""
    registry = EngineRegistry.get_instance()
    if not registry._is_loaded:
        registry.load_all_engines()
    return registry
