"""
JEEVAN-DRISHTI Model Evaluation, Validation, and Registry Package
================================================================
Phase 16: Centralized Model Evaluation, Validation, and Registry System
"""

from src.model_registry.drift_monitor import (
    calculate_ks_distance,
    calculate_psi,
    run_comprehensive_drift_monitoring_suite,
)
from src.model_registry.evaluator import load_and_standardize_evaluation_metrics
from src.model_registry.model_inspector import (
    ModelArtifactRecord,
    inspect_all_workspace_models,
)
from src.model_registry.pipeline import run_model_registry_pipeline
from src.model_registry.registry import (
    DatasetLineageRecord,
    build_model_registry,
)
from src.model_registry.service import ModelRegistryService

__all__ = [
    "ModelArtifactRecord",
    "DatasetLineageRecord",
    "inspect_all_workspace_models",
    "load_and_standardize_evaluation_metrics",
    "build_model_registry",
    "calculate_psi",
    "calculate_ks_distance",
    "run_comprehensive_drift_monitoring_suite",
    "run_model_registry_pipeline",
    "ModelRegistryService",
]
