"""Centralized Explainable AI (XAI) Engine for JEEVAN-DRISHTI (Phase 13)."""
from src.engines.explainability.global_explainer import compute_all_global_explanations
from src.engines.explainability.local_explainer import (
    compute_local_anomaly_explanations,
    compute_local_forecast_explanations,
    compute_local_risk_explanations,
)
from src.engines.explainability.model_inspector import (
    MANDATORY_XAI_NON_CAUSAL_DISCLAIMER,
    inspect_existing_models_and_artifacts,
)
from src.engines.explainability.pipeline import run_explainability_pipeline
from src.engines.explainability.service import ExplainableAIService

__all__ = [
    "MANDATORY_XAI_NON_CAUSAL_DISCLAIMER",
    "inspect_existing_models_and_artifacts",
    "compute_all_global_explanations",
    "compute_local_risk_explanations",
    "compute_local_forecast_explanations",
    "compute_local_anomaly_explanations",
    "run_explainability_pipeline",
    "ExplainableAIService",
]
