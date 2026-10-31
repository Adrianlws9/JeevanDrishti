"""Pickle-safe scikit-learn compatible model bundle for JEEVAN-DRISHTI Outbreak Risk Classification."""
from typing import Any, Dict, Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class CalibratedOutbreakRiskClassifier(BaseEstimator, ClassifierMixin):
    """Wraps a base or ensemble classifier with a validation-fitted Platt (sigmoid) probability
    calibrator, optimal decision thresholds, and secondary outbreak-risk estimators.

    Exposes standard scikit-learn `.predict_proba(X)` and `.predict(X)` methods while also
    storing calibrated probabilities, risk-tier cutoffs, and individual baseline/tree models
    for inspection and FastAPI inference.
    """

    def __init__(
        self,
        primary_model_name: str = "SoftVoting_Ensemble",
        primary_estimator: Any = None,
        calibrator: Any = None,
        optimal_threshold: float = 0.5,
        moderate_risk_threshold: float = 0.20,
        high_risk_threshold: float = 0.35,
        estimators_dict: Optional[Dict[str, Any]] = None,
        secondary_task_models: Optional[Dict[str, Any]] = None,
        feature_names_out: Optional[list] = None,
        global_feature_importance: Optional[Dict[str, float]] = None,
        model_version: str = "phase5-v1.0.0",
    ):
        self.primary_model_name = primary_model_name
        self.primary_estimator = primary_estimator
        self.calibrator = calibrator
        self.optimal_threshold = optimal_threshold
        self.moderate_risk_threshold = moderate_risk_threshold
        self.high_risk_threshold = high_risk_threshold
        self.estimators_dict = estimators_dict or {}
        self.secondary_task_models = secondary_task_models or {}
        self.feature_names_out = feature_names_out or []
        self.global_feature_importance = global_feature_importance or {}
        self.model_version = model_version
        self.classes_ = np.array([0, 1])

    def predict_raw_proba(self, X: np.ndarray) -> np.ndarray:
        """Return uncalibrated positive-class probabilities from the primary estimator."""
        probs = self.primary_estimator.predict_proba(X)
        if probs.ndim == 2:
            return probs[:, 1]
        return np.asarray(probs, dtype=float)

    def predict_calibrated_proba(self, X: np.ndarray) -> np.ndarray:
        """Return Platt-calibrated positive-class probabilities in [0, 1]."""
        raw_p = self.predict_raw_proba(X)
        if self.calibrator is None:
            return raw_p
        # Logit transform clipped for numerical stability before Platt logistic calibrator
        eps = 1e-6
        p_clip = np.clip(raw_p, eps, 1.0 - eps)
        logit = np.log(p_clip / (1.0 - p_clip)).reshape(-1, 1)
        cal_p = self.calibrator.predict_proba(logit)[:, 1]
        return np.asarray(cal_p, dtype=float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Standard scikit-learn 2-column probability output `[P(y=0), P(y=1)]` (uncalibrated raw score)."""
        p1 = self.predict_raw_proba(X)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Binary prediction using the validation-tuned optimal F1 threshold."""
        p1 = self.predict_raw_proba(X)
        return (p1 >= self.optimal_threshold).astype(int)

    def classify_risk_tier(self, calibrated_prob: float, is_validated_disease: bool = True) -> str:
        """Assign an epidemiological risk category based on calibrated probability and validation status."""
        if not is_validated_disease:
            return "UNVALIDATED_LIMITED_SAMPLE"
        if calibrated_prob >= self.high_risk_threshold:
            return "High"
        if calibrated_prob >= self.moderate_risk_threshold:
            return "Moderate"
        return "Low"
