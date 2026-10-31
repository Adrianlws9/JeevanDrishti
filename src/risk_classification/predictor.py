"""Reusable Outbreak Risk Classification Predictor for FastAPI Integration.

Loads the serialized artifacts from `models/risk_classification/` (`trained_model.pkl`,
`preprocessing_pipeline.pkl`, `feature_schema.json`, `model_metadata.json`) and returns
structured predictions containing:
- District
- Disease
- Reporting period
- Model-estimated risk
- Risk category where validated
- Calibrated probability where available
- Important contributing features
- Model version
- Source-data timestamp
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = PROJ / "models" / "risk_classification"
DEFAULT_DATASET_PATH = PROJ / "data" / "final_training" / "risk_classification_dataset.csv"


class OutbreakRiskPredictor:
    """FastAPI-ready inference wrapper for the JEEVAN-DRISHTI Outbreak Risk Classification Engine."""

    def __init__(self, models_dir: Optional[Union[str, Path]] = None):
        self.models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
        self.model_bundle = joblib.load(self.models_dir / "trained_model.pkl")
        self.preprocessor = joblib.load(self.models_dir / "preprocessing_pipeline.pkl")
        with open(self.models_dir / "feature_schema.json", "r", encoding="utf-8") as f:
            self.feature_schema = json.load(f)
        with open(self.models_dir / "model_metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.input_features: List[str] = (
            self.feature_schema["categorical_features"] + self.feature_schema["numerical_features"]
        )
        self.categorical_features: List[str] = self.feature_schema["categorical_features"]
        self.numerical_features: List[str] = self.feature_schema["numerical_features"]
        self.transformed_feature_names: List[str] = self.feature_schema["transformed_feature_names"]
        self.supported_diseases = set(self.metadata["supported_diseases_primary_p75"])
        self.rare_diseases = set(self.metadata["unsupported_diseases_rare_lt5_train_obs"])
        self.zero_record_diseases = set(self.metadata["unsupported_diseases_zero_records"])
        self.supported_districts = set(self.metadata["supported_districts_with_climate"])
        self.partial_districts = set(self.metadata["partial_support_districts_no_climate_coords"])
        self.zero_record_districts = set(self.metadata["unsupported_districts_zero_records_2023_2025"])

    def _compute_local_feature_contributions(
        self, X_trans_row: np.ndarray, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Compute local feature contributions for a single transformed feature vector."""
        lr = self.model_bundle.estimators_dict.get("Logistic_Regression_Baseline")
        global_imp = self.model_bundle.global_feature_importance

        contributions = []
        if lr is not None and hasattr(lr, "coef_"):
            coefs = lr.coef_.ravel()
            for idx, fname in enumerate(self.transformed_feature_names):
                val = float(X_trans_row[idx])
                if abs(val) < 1e-6:
                    continue
                # Combine standardized linear log-odds contribution with ensemble global weight
                linear_contrib = float(coefs[idx] * val)
                g_wt = float(global_imp.get(fname, 0.0))
                score = abs(linear_contrib) * (1.0 + 10.0 * g_wt)
                clean_name = (
                    fname.replace("numerical__", "")
                    .replace("categorical__", "")
                )
                contributions.append(
                    {
                        "feature": clean_name,
                        "transformed_value": round(val, 4),
                        "log_odds_contribution": round(linear_contrib, 4),
                        "direction": "increases_risk" if linear_contrib >= 0 else "decreases_risk",
                        "importance_rank_score": round(score, 4),
                    }
                )
        contributions.sort(key=lambda x: x["importance_rank_score"], reverse=True)
        for item in contributions:
            item.pop("importance_rank_score", None)
        return contributions[:top_k]

    def predict_dataframe(
        self, df: pd.DataFrame, top_k_features: int = 5
    ) -> List[Dict[str, Any]]:
        """Generate structured outbreak-risk predictions for a DataFrame of feature rows."""
        work = df.copy()
        for col in self.input_features:
            if col not in work.columns:
                work[col] = np.nan

        X_in = work[self.input_features]
        X_trans = self.preprocessor.transform(X_in)

        raw_probs = self.model_bundle.predict_raw_proba(X_trans)
        cal_probs = self.model_bundle.predict_calibrated_proba(X_trans)

        # Also compute fatal outbreak risk from the secondary task model if available
        fatal_probs = None
        if "target_fatal_outbreak" in self.model_bundle.secondary_task_models:
            f_bundle = self.model_bundle.secondary_task_models["target_fatal_outbreak"]
            X_f = f_bundle["preprocessor"].transform(X_in)
            fatal_probs = f_bundle["estimator"].predict_proba(X_f)[:, 1]

        results: List[Dict[str, Any]] = []
        for i, (_, row) in enumerate(work.iterrows()):
            district = str(row.get("district_std", row.get("district", "Unknown")))
            disease = str(row.get("disease_std", row.get("disease", "Unknown")))

            if "report_year" in row and "report_week" in row and pd.notna(row["report_year"]) and pd.notna(row["report_week"]):
                rep_period = f"{int(row['report_year'])}-W{int(row['report_week']):02d}"
            elif "reporting_period" in row and pd.notna(row["reporting_period"]):
                rep_period = str(row["reporting_period"])
            else:
                rep_period = str(row.get("canonical_week_start", "Unknown"))

            src_ts = str(
                row.get(
                    "report_period_end",
                    row.get("canonical_week_end", self.metadata["source_data_timestamp"]),
                )
            )

            # Check district & disease support status
            if district in self.zero_record_districts:
                results.append(
                    {
                        "district": district,
                        "disease": disease,
                        "reporting_period": rep_period,
                        "is_supported": False,
                        "support_status": "UNSUPPORTED_DISTRICT_ZERO_RECORDS_2023_2025",
                        "model_estimated_risk": None,
                        "risk_category": "UNSUPPORTED",
                        "calibrated_probability": None,
                        "fatal_outbreak_risk_score": None,
                        "important_contributing_features": [],
                        "model_version": self.metadata["model_version"],
                        "source_data_timestamp": src_ts,
                    }
                )
                continue

            if disease in self.zero_record_diseases:
                results.append(
                    {
                        "district": district,
                        "disease": disease,
                        "reporting_period": rep_period,
                        "is_supported": False,
                        "support_status": "UNSUPPORTED_DISEASE_ZERO_RECORDS_IN_IDSP",
                        "model_estimated_risk": None,
                        "risk_category": "UNSUPPORTED",
                        "calibrated_probability": None,
                        "fatal_outbreak_risk_score": None,
                        "important_contributing_features": [],
                        "model_version": self.metadata["model_version"],
                        "source_data_timestamp": src_ts,
                    }
                )
                continue

            if disease in self.rare_diseases or disease not in self.supported_diseases:
                f_score = round(float(fatal_probs[i]), 4) if fatal_probs is not None else None
                results.append(
                    {
                        "district": district,
                        "disease": disease,
                        "reporting_period": rep_period,
                        "is_supported": False,
                        "support_status": "UNSUPPORTED_FOR_P75_SEVERITY_LT5_TRAIN_OBS (FATAL_RISK_ONLY)",
                        "model_estimated_risk": None,
                        "risk_category": "UNSUPPORTED_INSUFFICIENT_TRAIN_LABELS",
                        "calibrated_probability": None,
                        "fatal_outbreak_risk_score": f_score,
                        "important_contributing_features": self._compute_local_feature_contributions(
                            X_trans[i], top_k=top_k_features
                        ),
                        "model_version": self.metadata["model_version"],
                        "source_data_timestamp": src_ts,
                    }
                )
                continue

            raw_p = round(float(raw_probs[i]), 4)
            cal_p = round(float(cal_probs[i]), 4)
            f_score = round(float(fatal_probs[i]), 4) if fatal_probs is not None else None

            is_partial_dist = district in self.partial_districts
            risk_cat = self.model_bundle.classify_risk_tier(cal_p, is_validated_disease=True)
            if is_partial_dist:
                support_status = "SUPPORTED_PARTIAL_NO_CLIMATE_COORDS"
            else:
                support_status = "SUPPORTED_VALIDATED"

            results.append(
                {
                    "district": district,
                    "disease": disease,
                    "reporting_period": rep_period,
                    "is_supported": True,
                    "support_status": support_status,
                    "model_estimated_risk": raw_p,
                    "risk_category": risk_cat,
                    "calibrated_probability": cal_p,
                    "fatal_outbreak_risk_score": f_score,
                    "important_contributing_features": self._compute_local_feature_contributions(
                        X_trans[i], top_k=top_k_features
                    ),
                    "model_version": self.metadata["model_version"],
                    "source_data_timestamp": src_ts,
                }
            )
        return results

    def predict_single(
        self,
        district: str,
        disease: str,
        reporting_period: Optional[str] = None,
        feature_overrides: Optional[Dict[str, Any]] = None,
        dataset_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Look up a `(district, disease, reporting_period)` record in the dataset (or construct
        from `feature_overrides`) and return a FastAPI-compatible prediction dictionary."""
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
                return self.predict_dataframe(row_df)[0]

        # Construct synthetic query row when querying an unsupported or custom combination
        payload: Dict[str, Any] = {
            "district_std": district,
            "disease_std": disease,
            "reporting_period": reporting_period or "2025-W52",
        }
        if feature_overrides:
            payload.update(feature_overrides)
        return self.predict_dataframe(pd.DataFrame([payload]))[0]


def predict_district_disease_risk(
    district: str,
    disease: str,
    reporting_period: Optional[str] = None,
    feature_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience functional entrypoint for FastAPI route handlers."""
    predictor = OutbreakRiskPredictor()
    return predictor.predict_single(
        district=district,
        disease=disease,
        reporting_period=reporting_period,
        feature_overrides=feature_overrides,
    )
