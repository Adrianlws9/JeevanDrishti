"""Pickle-safe scikit-learn compatible bundle for JEEVAN-DRISHTI Disease Anomaly Detection."""
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.base import BaseEstimator


class DiseaseAnomalyDetectorBundle(BaseEstimator):
    """Stores training-fitted seasonal baselines, robust MAD baselines, district baselines,
    mortality CFR baselines, Isolation Forest estimator, and feature preprocessor.
    """

    def __init__(
        self,
        isolation_forest: Any = None,
        preprocessor: Any = None,
        iforest_features: Optional[List[str]] = None,
        seasonal_baselines: Optional[Dict[str, Dict[str, float]]] = None,
        disease_baselines: Optional[Dict[str, Dict[str, float]]] = None,
        district_disease_baselines: Optional[Dict[str, Dict[str, float]]] = None,
        historical_2009_2022_p95: Optional[Dict[str, float]] = None,
        score_threshold: float = 0.55,
        high_priority_threshold: float = 0.72,
        supported_diseases: Optional[List[str]] = None,
        limited_diseases: Optional[List[str]] = None,
        supported_districts: Optional[List[str]] = None,
        model_version: str = "phase7-v1.0.0",
    ):
        self.isolation_forest = isolation_forest
        self.preprocessor = preprocessor
        self.iforest_features = iforest_features or []
        self.seasonal_baselines = seasonal_baselines or {}
        self.disease_baselines = disease_baselines or {}
        self.district_disease_baselines = district_disease_baselines or {}
        self.historical_2009_2022_p95 = historical_2009_2022_p95 or {}
        self.score_threshold = score_threshold
        self.high_priority_threshold = high_priority_threshold
        self.supported_diseases = supported_diseases or []
        self.limited_diseases = limited_diseases or []
        self.supported_districts = supported_districts or []
        self.model_version = model_version

    def score_iforest(self, X_trans: np.ndarray) -> np.ndarray:
        """Return normalized [0, 1] Isolation Forest anomaly score (higher = more anomalous)."""
        # decision_function is positive for inliers, negative for outliers
        raw_df = self.isolation_forest.decision_function(X_trans)
        # Logistic squash centered at decision_function = 0.0
        score = 1.0 / (1.0 + np.exp(8.0 * raw_df))
        return np.clip(np.asarray(score, dtype=float), 0.0, 1.0)
