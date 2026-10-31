"""Reusable Time-Series Disease Forecasting Service for JEEVAN-DRISHTI (FastAPI-Ready).

Loads serialized artifacts from `models/disease_forecasting/`:
- `forecasting_model.pkl`
- `preprocessing_pipeline.pkl`
- `forecast_metadata.json`

Generates multi-horizon forecasts (7 days = 1 reporting week, 14 days = 2 reporting weeks,
30 days = 4 reporting weeks) with calibrated 80% and 90% prediction intervals.
Explicitly tags all predictions with `value_type = "MODEL_FORECAST_NOT_OBSERVED"` so
forecasts are never presented as observed surveillance cases.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from . import HORIZON_MAP

PROJ = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = PROJ / "models" / "disease_forecasting"
DEFAULT_DATASET_PATH = PROJ / "data" / "final_training" / "forecasting_dataset.csv"


class DiseaseForecastingService:
    """Reusable multi-horizon disease case-count forecasting service for Maharashtra."""

    def __init__(self, models_dir: Optional[Union[str, Path]] = None):
        self.models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
        self.forecaster = joblib.load(self.models_dir / "forecasting_model.pkl")
        self.default_preprocessor = joblib.load(self.models_dir / "preprocessing_pipeline.pkl")
        with open(self.models_dir / "forecast_metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.supported_diseases = set(self.metadata["supported_diseases_standalone_and_pooled"])
        self.limited_diseases = set(self.metadata["limited_sample_diseases_lt5_train_obs"])
        self.zero_record_diseases = set(self.metadata["unsupported_diseases_zero_records"])
        self.supported_districts = set(self.metadata["supported_districts"])
        self.partial_districts = set(self.metadata["partial_districts_no_climate_coords"])
        self.zero_record_districts = set(self.metadata["unsupported_districts_zero_records"])

    def forecast_dataframe(
        self,
        df: pd.DataFrame,
        horizons: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate multi-horizon forecasts with 80% and 90% uncertainty intervals for a DataFrame."""
        horizons = horizons or ["7d", "14d", "30d"]
        work = df.copy().reset_index(drop=True)

        # Precompute transformed arrays and predictions per horizon
        horizon_preds: Dict[str, Dict[str, np.ndarray]] = {}
        for h in horizons:
            if h not in self.forecaster.horizon_bundles:
                raise ValueError(f"Unsupported horizon '{h}'. Expected one of {list(HORIZON_MAP.keys())}.")
            h_bundle = self.forecaster.horizon_bundles[h]
            req_cols = h_bundle["selected_features"]
            for c in req_cols:
                if c not in work.columns:
                    work[c] = np.nan
            X_trans = h_bundle["preprocessor"].transform(work[req_cols])
            horizon_preds[h] = self.forecaster.predict_with_intervals(X_trans, horizon=h)

        outputs: List[Dict[str, Any]] = []
        for i, row in work.iterrows():
            district = str(row.get("district_std", row.get("district", "Unknown")))
            disease = str(row.get("disease_std", row.get("disease", "Unknown")))
            if "report_year" in row and "report_week" in row and pd.notna(row["report_year"]) and pd.notna(row["report_week"]):
                origin_period = f"{int(row['report_year'])}-W{int(row['report_week']):02d}"
            else:
                origin_period = str(row.get("reporting_period", "2025-W52"))

            # Check support status
            if district in self.zero_record_districts or district not in self.supported_districts:
                outputs.append(
                    {
                        "district": district,
                        "disease": disease,
                        "origin_reporting_period": origin_period,
                        "is_supported": False,
                        "support_status": "UNSUPPORTED_DISTRICT_ZERO_RECORDS_2023_2025",
                        "native_reporting_frequency": "weekly",
                        "value_type": "UNSUPPORTED_NO_FORECAST",
                        "horizons": {},
                        "model_version": self.metadata["model_version"],
                    }
                )
                continue

            if disease in self.zero_record_diseases or (
                disease not in self.supported_diseases and disease not in self.limited_diseases
            ):
                outputs.append(
                    {
                        "district": district,
                        "disease": disease,
                        "origin_reporting_period": origin_period,
                        "is_supported": False,
                        "support_status": "UNSUPPORTED_DISEASE_ZERO_RECORDS_IN_IDSP",
                        "native_reporting_frequency": "weekly",
                        "value_type": "UNSUPPORTED_NO_FORECAST",
                        "horizons": {},
                        "model_version": self.metadata["model_version"],
                    }
                )
                continue

            if disease in self.limited_diseases:
                support_status = "LIMITED_SAMPLE_POOLED_FORECAST_ONLY (<5 train obs)"
            elif district in self.partial_districts:
                support_status = "SUPPORTED_PARTIAL_NO_CLIMATE_COORDS"
            else:
                support_status = "SUPPORTED_VALIDATED"

            h_out: Dict[str, Any] = {}
            for h in horizons:
                h_meta = HORIZON_MAP[h]
                hp = horizon_preds[h]
                h_out[h] = {
                    "horizon_days_requested": h_meta["horizon_days"],
                    "native_future_weeks": h_meta["horizon_weeks"],
                    "horizon_explanation": h_meta["description"],
                    "selected_model": self.forecaster.selected_model_by_horizon[h],
                    "predicted_cases": float(hp["point_forecast"][i]),
                    "prediction_interval_80": [
                        float(hp["lower_80"][i]),
                        float(hp["upper_80"][i]),
                    ],
                    "prediction_interval_90": [
                        float(hp["lower_90"][i]),
                        float(hp["upper_90"][i]),
                    ],
                    "value_type": "MODEL_FORECAST_NOT_OBSERVED",
                }

            outputs.append(
                {
                    "district": district,
                    "disease": disease,
                    "origin_reporting_period": origin_period,
                    "is_supported": True,
                    "support_status": support_status,
                    "native_reporting_frequency": "weekly (7-day IDSP reporting periods)",
                    "value_type": "MODEL_FORECAST_NOT_OBSERVED",
                    "horizons": h_out,
                    "model_version": self.metadata["model_version"],
                }
            )
        return outputs

    def forecast_single(
        self,
        district: str,
        disease: str,
        reporting_period: Optional[str] = None,
        horizons: Optional[List[str]] = None,
        feature_overrides: Optional[Dict[str, Any]] = None,
        dataset_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Look up `(district, disease, reporting_period)` and return multi-horizon forecasts."""
        ds_path = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
        if ds_path.exists():
            df = pd.read_csv(ds_path)
            sub = df[(df["district_std"] == district) & (df["disease_std"] == disease)].copy()
            if reporting_period and not sub.empty:
                yw = (
                    sub["report_year"].astype(int).astype(str)
                    + "-W"
                    + sub["report_week"].astype(int).map(lambda w: f"{w:02d}")
                )
                sub = sub[yw == reporting_period]
            if not sub.empty:
                row_df = sub.sort_values("period_idx").iloc[[-1]].copy()
                if feature_overrides:
                    for k, v in feature_overrides.items():
                        row_df[k] = v
                return self.forecast_dataframe(row_df, horizons=horizons)[0]

        payload: Dict[str, Any] = {
            "district_std": district,
            "disease_std": disease,
            "reporting_period": reporting_period or "2025-W52",
        }
        if feature_overrides:
            payload.update(feature_overrides)
        return self.forecast_dataframe(pd.DataFrame([payload]), horizons=horizons)[0]


def forecast_district_disease_cases(
    district: str,
    disease: str,
    reporting_period: Optional[str] = None,
    horizons: Optional[List[str]] = None,
    feature_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Functional entrypoint for FastAPI route integration."""
    service = DiseaseForecastingService()
    return service.forecast_single(
        district=district,
        disease=disease,
        reporting_period=reporting_period,
        horizons=horizons,
        feature_overrides=feature_overrides,
    )
