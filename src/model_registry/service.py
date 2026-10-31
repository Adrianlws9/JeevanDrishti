"""
JEEVAN-DRISHTI Model Registry Service
====================================
Phase 16: Centralized Model Evaluation, Validation, and Registry System

Provides programmatic query and governance APIs for interacting with the
model registry, lineage records, evaluation metrics, and drift monitoring.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)
PROJ = Path(__file__).resolve().parents[2]


class ModelRegistryService:
    """Service API for interacting with the JEEVAN-DRISHTI model registry."""

    def __init__(self, registry_path: Optional[Union[str, Path]] = None) -> None:
        if registry_path is None:
            registry_path = PROJ / "models" / "registry" / "model_registry.json"
        self.registry_path = Path(registry_path)
        self._data: Optional[Dict[str, Any]] = None
        self._models_cache: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Load the registry JSON from disk if available."""
        if not self.registry_path.is_file():
            logger.warning("Model registry not found at %s. Attempting to build or reload.", self.registry_path)
            self._data = {"registered_models": []}
            return

        with open(self.registry_path, "r", encoding="utf-8") as f:
            self._data = json.load(f)

        models_raw = self._data.get("models") or self._data.get("registered_models") or []
        self._models_cache = {
            m["model_id"]: m for m in models_raw if "model_id" in m
        }

    def reload(self) -> None:
        """Reload the registry from disk."""
        self._load()

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full 15-field registry metadata for a specific model ID."""
        return self._models_cache.get(model_id)

    def list_models(
        self,
        status: Optional[str] = None,
        disease: Optional[str] = None,
        model_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List registered models filtered by status, disease, or model type.
        """
        results = list(self._models_cache.values())

        if status:
            results = [m for m in results if m.get("model_status", "").lower() == status.lower()]
        if disease:
            results = [m for m in results if m.get("disease", "").lower() == disease.lower()]
        if model_type:
            results = [m for m in results if model_type.lower() in m.get("model_type", "").lower()]

        return results

    def get_lineage(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get complete dataset and code lineage for a model.
        """
        model = self.get_model(model_id)
        if not model:
            return None

        return {
            "model_id": model_id,
            "model_name": model.get("model_name"),
            "dataset_filenames": model.get("dataset_filenames", []),
            "dataset_version": model.get("dataset_version"),
            "feature_version": model.get("feature_version"),
            "training_period": model.get("training_period"),
            "validation_period": model.get("validation_period"),
            "testing_period": model.get("testing_period"),
            "artifact_path": model.get("artifact_path"),
            "model_version": model.get("model_version"),
            "model_status": model.get("model_status"),
        }

    def get_evaluation_metrics(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get evaluation metrics across train, val, and test splits."""
        model = self.get_model(model_id)
        if not model:
            return None
        return model.get("evaluation_metrics", {})

    def compare_models(self, model_ids: List[str]) -> pd.DataFrame:
        """
        Generate a side-by-side comparison DataFrame of requested models.
        """
        rows = []
        for mid in model_ids:
            m = self.get_model(mid)
            if not m:
                continue
            metrics = m.get("evaluation_metrics", {})
            test_m = metrics.get("held_out_test", {})
            rows.append({
                "model_id": m.get("model_id"),
                "model_name": m.get("model_name"),
                "model_type": m.get("model_type"),
                "model_status": m.get("model_status"),
                "disease": m.get("disease"),
                "test_metrics": str(test_m),
                "artifact_path": m.get("artifact_path"),
            })
        return pd.DataFrame(rows)

    def get_drift_report(self) -> Dict[str, Any]:
        """Load and return the feature and prediction drift report if available."""
        drift_csv = PROJ / "reports" / "model_evaluation" / "feature_and_prediction_drift_report.csv"
        if not drift_csv.is_file():
            return {"status": "NO_DRIFT_REPORT_FOUND"}

        df = pd.read_csv(drift_csv)
        return {
            "status": "LOADED",
            "feature_drift_records": df.to_dict(orient="records"),
            "high_drift_features": df[df["drift_severity"] == "HIGH"]["feature_name"].tolist() if "drift_severity" in df.columns else [],
        }
