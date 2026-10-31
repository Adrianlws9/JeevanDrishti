"""Reusable Disease Anomaly Detection Service for JEEVAN-DRISHTI (FastAPI-Ready).

Loads serialized artifacts from `models/anomaly_detection/`:
- `anomaly_detector.pkl`
- `preprocessing_pipeline.pkl`
- `anomaly_model_metadata.json`

Returns structured anomaly records containing:
- District
- Disease
- Reporting period
- Anomaly score
- Historical baseline
- Triggering features
- Detection timestamp
- Method and version
- Explicit interpretation note:
  "An anomaly is an elevated signal requiring investigation, not proof of an outbreak."
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from . import ANOMALY_DISCLAIMER

PROJ = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = PROJ / "models" / "anomaly_detection"
DEFAULT_RESULTS_PATH = PROJ / "reports" / "anomaly_detection_results.csv"


class DiseaseAnomalyDetectionService:
    """FastAPI-ready service for district-level disease anomaly detection in Maharashtra."""

    def __init__(self, models_dir: Optional[Union[str, Path]] = None):
        self.models_dir = Path(models_dir) if models_dir else DEFAULT_MODELS_DIR
        self.bundle = joblib.load(self.models_dir / "anomaly_detector.pkl")
        self.preprocessor = joblib.load(self.models_dir / "preprocessing_pipeline.pkl")
        with open(self.models_dir / "anomaly_model_metadata.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.supported_diseases = set(self.metadata["supported_diseases_full_baseline"])
        self.limited_diseases = set(self.metadata["limited_diseases_group_baseline_fallback"])
        self.zero_record_diseases = set(self.metadata["unsupported_diseases_zero_records"])
        self.supported_districts = set(self.metadata["supported_districts"])
        self.zero_record_districts = set(self.metadata["unsupported_districts_zero_records"])

        # Integrated WHO EWARS-inspired early warning threshold engine
        from .ewars_thresholds import WHOEWARSThresholdEngine
        self.ewars_engine = WHOEWARSThresholdEngine()
        if DEFAULT_RESULTS_PATH.is_file():
            try:
                _df_hist = pd.read_csv(DEFAULT_RESULTS_PATH)
                _train_sub = _df_hist[_df_hist["split"] == "train"] if "split" in _df_hist.columns else _df_hist
                self.ewars_engine.calibrate(_train_sub)
            except Exception as e:
                pass

    def evaluate_observation(
        self,
        district: str,
        disease: str,
        reporting_period: str,
        observed_cases: float,
        observed_deaths: Optional[float] = None,
        season: str = "Monsoon",
        rolling_prior_mean: Optional[float] = None,
        rolling_prior_std: Optional[float] = None,
        outbreak_clusters_in_period: int = 1,
        periods_since_prev_outbreak: float = 4.0,
        climate_rain_lag1w_zscore: float = 0.0,
        climate_tavg_lag1w_zscore: float = 0.0,
        detection_timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Score a single `(district, disease, reporting_period)` observation across all 4 methods."""
        det_ts = detection_timestamp or datetime.now().strftime("%Y-%m-%d")

        if district in self.zero_record_districts or district not in self.supported_districts:
            return {
                "district": district,
                "disease": disease,
                "reporting_period": reporting_period,
                "is_supported": False,
                "support_status": "UNSUPPORTED_DISTRICT_ZERO_RECORDS_2023_2025",
                "anomaly_score": None,
                "is_anomaly": False,
                "anomaly_severity_tier": "UNSUPPORTED",
                "signal_types": [],
                "historical_baseline": None,
                "triggering_features": [],
                "detection_timestamp": det_ts,
                "method": "UNSUPPORTED",
                "version": self.metadata["model_version"],
                "interpretation_note": ANOMALY_DISCLAIMER,
            }

        if disease in self.zero_record_diseases or (
            disease not in self.supported_diseases and disease not in self.limited_diseases
        ):
            return {
                "district": district,
                "disease": disease,
                "reporting_period": reporting_period,
                "is_supported": False,
                "support_status": "UNSUPPORTED_DISEASE_ZERO_RECORDS_IN_IDSP",
                "anomaly_score": None,
                "is_anomaly": False,
                "anomaly_severity_tier": "UNSUPPORTED",
                "signal_types": [],
                "historical_baseline": None,
                "triggering_features": [],
                "detection_timestamp": det_ts,
                "method": "UNSUPPORTED",
                "version": self.metadata["model_version"],
                "interpretation_note": ANOMALY_DISCLAIMER,
            }

        db = self.bundle.disease_baselines.get(
            disease,
            {"mean": 25.0, "std": 25.0, "median": 15.0, "mad": 5.0, "log_median": 2.77, "log_mad": 0.35, "train_cfr": 0.01, "n_train": 1},
        )
        sb = self.bundle.seasonal_baselines.get(f"{disease}||{season}", db)
        ddb = self.bundle.district_disease_baselines.get(f"{district}||{disease}", {"mean": sb["mean"]})

        s_mean = float(sb["mean"])
        s_std = max(float(sb["std"]), 3.0)
        s_med = float(sb["median"])
        s_mad = max(float(sb["mad"]), 1.5)
        s_log_med = float(sb["log_median"])
        s_log_mad = max(float(sb["log_mad"]), 0.20)

        r_mean = float(rolling_prior_mean) if rolling_prior_mean is not None else float(ddb.get("mean", s_mean))
        r_std = max(float(rolling_prior_std) if rolling_prior_std is not None else s_std, 3.0)

        cases_t = float(observed_cases)
        log_cases_t = float(np.log1p(max(0.0, cases_t)))

        # 1. Seasonal Baseline Deviation
        sz = (cases_t - s_mean) / s_std
        s_score = float(1.0 / (1.0 + np.exp(-1.2 * (sz - 1.5))))
        s_flag = int(sz > 2.0)

        # 2. Rolling Statistical Detector
        rz = (cases_t - r_mean) / r_std
        r_score = float(1.0 / (1.0 + np.exp(-1.2 * (rz - 1.5))))
        r_flag = int(rz > 2.0)

        # 3. Robust MAD Z-Score
        rob_z_raw = 0.6745 * (cases_t - s_med) / s_mad
        rob_z_log = 0.6745 * (log_cases_t - s_log_med) / s_log_mad
        rob_z = max(rob_z_raw, rob_z_log) if rob_z_raw > 0 else rob_z_log
        rob_score = float(1.0 / (1.0 + np.exp(-1.0 * (rob_z - 2.0))))
        rob_flag = int(rob_z > 2.5)

        # 4. Mortality Signal
        cfr_t = 0.0
        m_flag = 0
        m_score = 0.0
        if observed_deaths is not None and pd.notna(observed_deaths):
            d_val = float(observed_deaths)
            cfr_t = d_val / max(cases_t, 1.0)
            ref_cfr = max(float(db.get("train_cfr", 0.01)), 0.005)
            if d_val >= 2.0 or (d_val >= 1.0 and cfr_t >= max(0.05, 2.5 * ref_cfr)):
                m_flag = 1
                m_score = min(1.0, 0.55 + 0.15 * d_val + 1.5 * cfr_t)

        vel_ratio = cases_t / max(r_mean, 1.0)
        dist_ratio = cases_t / max(float(ddb.get("mean", s_mean)), 1.0)

        # 5. Isolation Forest Score
        feat_row = pd.DataFrame(
            [
                {
                    "log1p_cases_t": log_cases_t,
                    "seasonal_zscore": sz,
                    "rolling_zscore": rz,
                    "robust_mad_zscore": rob_z,
                    "case_velocity_ratio": vel_ratio,
                    "district_burden_ratio": dist_ratio,
                    "observed_cfr": cfr_t,
                    "outbreak_events_in_period": outbreak_clusters_in_period,
                    "periods_since_prev_outbreak": periods_since_prev_outbreak,
                    "climate_rain_lag1w_zscore_vs_train": climate_rain_lag1w_zscore,
                    "climate_tavg_lag1w_zscore_vs_train": climate_tavg_lag1w_zscore,
                    "rain_mm_deviation_expanding_season_abs_t": 0.0,
                    "temp_diurnal_range_mean_t": 12.0,
                    "reporting_coverage_factor": 1.0,
                }
            ]
        )
        X_trans = self.preprocessor.transform(feat_row[self.bundle.iforest_features])
        if_score = float(self.bundle.score_iforest(X_trans)[0])
        if_flag = int(self.bundle.isolation_forest.predict(X_trans)[0] == -1)

        comp_score = max(
            0.30 * rob_score + 0.25 * s_score + 0.25 * r_score + 0.20 * if_score,
            0.85 * m_score,
        )
        comp_score = round(float(comp_score), 4)
        n_flagged = s_flag + r_flag + rob_flag + if_flag
        is_anom = bool(comp_score >= self.bundle.score_threshold or n_flagged >= 2 or m_flag == 1)

        signals = []
        triggers = []
        if r_flag == 1 or vel_ratio >= 2.2:
            signals.append("SUDDEN_CASE_INCREASE")
            triggers.append(f"cases={int(cases_t)} vs rolling_ref={r_mean:.1f} (rolling_z={rz:+.2f})")
        if dist_ratio >= 2.2 or outbreak_clusters_in_period >= 2:
            signals.append("UNUSUAL_DISTRICT_PATTERN")
            triggers.append(f"district_ratio={dist_ratio:.2f}x prior mean, clusters={outbreak_clusters_in_period}")
        if s_flag == 1 or rob_flag == 1:
            signals.append("UNEXPECTED_SEASONAL_DEVIATION")
            triggers.append(f"seasonal_z={sz:+.2f}, robust_mad_z={rob_z:+.2f} vs {season} median={s_med:.1f}")
        if m_flag == 1:
            signals.append("UNUSUAL_MORTALITY_PATTERN")
            triggers.append(f"observed_deaths={int(observed_deaths)} (CFR={cfr_t:.1%})")
        if not triggers:
            triggers.append(f"cases={int(cases_t)} within expected seasonal baseline (median={s_med:.1f})")

        if not is_anom:
            tier = "NORMAL_BASELINE"
        elif comp_score >= self.bundle.high_priority_threshold or m_flag == 1 or n_flagged >= 3:
            tier = "HIGH_PRIORITY_INVESTIGATION_SIGNAL"
        else:
            tier = "ELEVATED_WATCH_SIGNAL"

        # WHO EWARS-inspired threshold evaluation as additional evidence source
        ewars_alert = self.ewars_engine.evaluate_observation(
            district=district,
            disease=disease,
            reporting_period=reporting_period,
            observed_cases=cases_t,
            season=season,
        )
        if ewars_alert.alert_status != "NORMAL_BASELINE":
            signals.append("EWARS_THRESHOLD_ALERT")
            triggers.append(f"EWARS: {', '.join(ewars_alert.triggered_methods)} (severity={ewars_alert.alert_severity})")

        return {
            "district": district,
            "disease": disease,
            "reporting_period": reporting_period,
            "is_supported": True,
            "support_status": "SUPPORTED_VALIDATED" if disease in self.supported_diseases else "LIMITED_SAMPLE_GROUP_BASELINE",
            "anomaly_score": comp_score,
            "is_anomaly": is_anom,
            "anomaly_severity_tier": tier,
            "signal_types": signals,
            "historical_baseline": {
                "season": season,
                "seasonal_mean": round(s_mean, 2),
                "seasonal_std": round(s_std, 2),
                "robust_median": round(s_med, 2),
                "robust_mad": round(s_mad, 2),
                "rolling_reference_mean": round(r_mean, 2),
            },
            "component_scores": {
                "seasonal_zscore": round(sz, 4),
                "rolling_zscore": round(rz, 4),
                "robust_mad_zscore": round(rob_z, 4),
                "isolation_forest_score": round(if_score, 4),
                "methods_flagged_count": int(n_flagged),
            },
            "ewars_threshold_results": ewars_alert.to_dict(),
            "triggering_features": triggers,
            "detection_timestamp": det_ts,
            "method": "Composite_Consensus_Engine (Seasonal + Rolling + Robust_MAD + IsolationForest + EWARS_Thresholds)",
            "version": self.metadata["model_version"],
            "interpretation_note": ANOMALY_DISCLAIMER,
        }

    def evaluate_ewars_thresholds(
        self,
        district: str,
        disease: str,
        reporting_period: str,
        observed_cases: float,
        prior_cases_history: Optional[List[float]] = None,
        lagged_alerts_history: Optional[List[bool]] = None,
        season: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate observation against WHO EWARS-inspired threshold methodologies."""
        return self.ewars_engine.evaluate_observation(
            district=district,
            disease=disease,
            reporting_period=reporting_period,
            observed_cases=observed_cases,
            prior_cases_history=prior_cases_history,
            lagged_alerts_history=lagged_alerts_history,
            season=season,
        ).to_dict()

    def lookup_period_result(
        self,
        district: str,
        disease: str,
        reporting_period: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Look up a precomputed or live-evaluated `(district, disease, reporting_period)` record."""
        if DEFAULT_RESULTS_PATH.exists():
            df = pd.read_csv(DEFAULT_RESULTS_PATH)
            sub = df[(df["district"] == district) & (df["disease"] == disease)]
            if reporting_period and not sub.empty:
                sub = sub[sub["reporting_period"] == reporting_period]
            if not sub.empty:
                r = sub.iloc[-1]
                return {
                    "district": str(r["district"]),
                    "disease": str(r["disease"]),
                    "reporting_period": str(r["reporting_period"]),
                    "is_supported": True,
                    "support_status": "SUPPORTED_VALIDATED" if disease in self.supported_diseases else "LIMITED_SAMPLE_GROUP_BASELINE",
                    "anomaly_score": float(r["anomaly_score"]),
                    "is_anomaly": bool(r["is_anomaly"]),
                    "anomaly_severity_tier": str(r["anomaly_severity_tier"]),
                    "signal_types": [s for s in str(r["signal_types"]).split(";") if s and s != "NONE_WITHIN_BASELINE"],
                    "historical_baseline": str(r["historical_baseline"]),
                    "triggering_features": [t.strip() for t in str(r["triggering_features"]).split("|")],
                    "detection_timestamp": str(r["detection_timestamp"]),
                    "method": str(r["method"]),
                    "version": str(r["model_version"]),
                    "interpretation_note": str(r["interpretation_note"]),
                }
        return self.evaluate_observation(
            district=district,
            disease=disease,
            reporting_period=reporting_period or "2025-W52",
            observed_cases=20.0,
        )


def detect_district_disease_anomaly(
    district: str,
    disease: str,
    reporting_period: Optional[str] = None,
    observed_cases: Optional[float] = None,
    observed_deaths: Optional[float] = None,
    season: str = "Monsoon",
) -> Dict[str, Any]:
    """Functional entrypoint for FastAPI anomaly detection routes."""
    service = DiseaseAnomalyDetectionService()
    if observed_cases is not None:
        return service.evaluate_observation(
            district=district,
            disease=disease,
            reporting_period=reporting_period or "2025-W52",
            observed_cases=observed_cases,
            observed_deaths=observed_deaths,
            season=season,
        )
    return service.lookup_period_result(district, disease, reporting_period)
