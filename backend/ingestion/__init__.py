"""
Data Ingestion Module for JEEVAN-DRISHTI
========================================
Phase 18: Automated Data Ingestion and 2-Second Dashboard Refresh System
"""

from backend.ingestion.config import IngestionConfig
from backend.ingestion.validator import IngestionValidator
from backend.ingestion.pipeline import DataIngestionPipeline

__all__ = ["IngestionConfig", "IngestionValidator", "DataIngestionPipeline"]
