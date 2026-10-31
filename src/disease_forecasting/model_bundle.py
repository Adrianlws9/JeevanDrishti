"""Pickle-safe scikit-learn compatible bundle for JEEVAN-DRISHTI Time-Series Disease Forecasting."""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class MultiHorizonDiseaseForecaster(BaseEstimator, RegressorMixin):
    """Wraps horizon-specific (7d=1w, 14d=2w, 30d=4w) preprocessing, ML regressors,
    statistical time-series baselines (ETS, SARIMA, SeasonalNaive, LastValue), and
    split-conformal log1p residual quantiles for calibrated 80% and 90% prediction intervals.

    Exposes standard scikit-learn `.predict(X)` (for the default 7d / 1-week horizon in raw cases)
    as well as `.predict_with_intervals(X, horizon='7d')`.
    """

    def __init__(
        self,
        selected_model_by_horizon: Optional[Dict[str, str]] = None,
        horizon_bundles: Optional[Dict[str, Dict[str, Any]]] = None,
        disease_stats_train: Optional[Dict[str, Dict[str, float]]] = None,
        supported_diseases: Optional[List[str]] = None,
        supported_districts: Optional[List[str]] = None,
        model_version: str = "phase6-v1.0.0",
    ):
        self.selected_model_by_horizon = selected_model_by_horizon or {}
        self.horizon_bundles = horizon_bundles or {}
        self.disease_stats_train = disease_stats_train or {}
        self.supported_diseases = supported_diseases or []
        self.supported_districts = supported_districts or []
        self.model_version = model_version

    def predict_log1p(self, X_trans: np.ndarray, horizon: str = "7d", model_name: Optional[str] = None) -> np.ndarray:
        """Predict log(1 + cases) for the specified horizon ('7d', '14d', '30d')."""
        h_bundle = self.horizon_bundles[horizon]
        m_name = model_name or self.selected_model_by_horizon[horizon]
        est = h_bundle["ml_models"][m_name]
        pred_log = est.predict(X_trans)
        return np.clip(np.asarray(pred_log, dtype=float), 0.0, 9.5)

    def predict(self, X_trans: np.ndarray, horizon: str = "7d", model_name: Optional[str] = None) -> np.ndarray:
        """Predict expected case counts in original scale (`expm1(pred_log1p)`)."""
        pred_log = self.predict_log1p(X_trans, horizon=horizon, model_name=model_name)
        return np.maximum(0.0, np.expm1(pred_log))

    def predict_with_intervals(
        self,
        X_trans: np.ndarray,
        horizon: str = "7d",
        model_name: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """Return point forecast plus calibrated 80% and 90% conformal prediction intervals."""
        h_bundle = self.horizon_bundles[horizon]
        m_name = model_name or self.selected_model_by_horizon[horizon]
        pred_log = self.predict_log1p(X_trans, horizon=horizon, model_name=m_name)

        q80 = float(h_bundle["conformal_q80_log1p"].get(m_name, 0.95))
        q90 = float(h_bundle["conformal_q90_log1p"].get(m_name, 1.25))

        pt = np.maximum(0.0, np.expm1(pred_log))
        lo80 = np.maximum(0.0, np.expm1(np.maximum(0.0, pred_log - q80)))
        hi80 = np.expm1(np.minimum(9.5, pred_log + q80))
        lo90 = np.maximum(0.0, np.expm1(np.maximum(0.0, pred_log - q90)))
        hi90 = np.expm1(np.minimum(9.5, pred_log + q90))

        return {
            "point_forecast": np.round(pt, 2),
            "lower_80": np.round(lo80, 2),
            "upper_80": np.round(hi80, 2),
            "lower_90": np.round(lo90, 2),
            "upper_90": np.round(hi90, 2),
        }
