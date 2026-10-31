"""Local Feature Attribution, Forecast Decomposition & Anomaly Evidence Engine (Phase 13 Sections 4–7).

Generates verified, record-level explanations for:
1. Outbreak Risk Classification (`phase5-v1.0.0`):
   - Explains why each district-disease-week received its calibrated risk estimate.
   - Quantifies signed contributions from historical disease lags, recent case growth,
     rainfall/temperature lags, spatial neighbors, and seasonality.
   - Flags whether missing input features (e.g., first-outbreak lags or missing climate station)
     affected the record.
2. Multi-Horizon Disease Case Forecasts (`phase6-v1.0.0`):
   - Decomposes 7d, 14d, and 30d forecasts into historical trend/baseline, lagged cases,
     seasonality, and climate contributions, plus conformal uncertainty and residual explanations.
3. Anomaly Signals (`phase7-v1.0.0`):
   - Explains every flagged anomaly via observed value, expected baseline, deviation z-scores,
     detection method, threshold, and historical context.
"""
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb

from src.engines.explainability.model_inspector import (
    MANDATORY_XAI_NON_CAUSAL_DISCLAIMER,
    PROJ,
    classify_feature_domain,
    clean_transformed_feature_name,
)
from src.features.geographic_features import DISTRICT_ID_MAP


def compute_local_risk_explanations(
    proj_dir: Path = PROJ, bundles_dict: Dict[str, Any] = None
) -> pd.DataFrame:
    """Generate local risk explanations across all 511 records in risk_classification_dataset.csv."""
    rc_bundle = bundles_dict["rc_bundle"]
    rc_prep = bundles_dict["rc_prep"]
    rc_meta = bundles_dict["rc_meta"]

    schema = json.loads(
        (proj_dir / "models" / "risk_classification" / "feature_schema.json").read_text(encoding="utf-8")
    )
    df_rc = pd.read_csv(proj_dir / "data" / "final_training" / "risk_classification_dataset.csv")
    raw_features = schema["categorical_features"] + schema["numerical_features"]

    X_trans = rc_prep.transform(df_rc[raw_features])
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
    X_trans = np.asarray(X_trans, dtype=float)

    feat_names = [clean_transformed_feature_name(f) for f in rc_bundle.feature_names_out]
    feat_domains = [classify_feature_domain(f) for f in feat_names]

    # Calibrated probabilities from primary model (HistGradientBoosting + Platt calibrator)
    cal_probs = rc_bundle.predict_calibrated_proba(X_trans)
    raw_probs = rc_bundle.predict_raw_proba(X_trans)

    # Exact TreeSHAP log-odds contributions from trained XGBoost_Classifier
    xgb_est = rc_bundle.estimators_dict["XGBoost_Classifier"]
    shap_contribs = xgb_est.get_booster().predict(xgb.DMatrix(X_trans), pred_contribs=True)
    shap_vals = np.asarray(shap_contribs[:, :-1], dtype=float)
    base_margin = np.asarray(shap_contribs[:, -1], dtype=float)

    supported_p75 = set(rc_meta.get("supported_diseases_primary_p75", []))
    gen_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    rows: List[Dict[str, Any]] = []
    for i, r in df_rc.iterrows():
        d_name = str(r["district_std"])
        dis_name = str(r["disease_std"])
        yr = int(r["report_year"])
        wk = int(r["report_week"])
        rep_p = f"{yr}-W{wk:02d}"
        pred_id = f"RISK_{ DISTRICT_ID_MAP.get(d_name, 'MH') }_{rep_p}_{i:03d}"

        is_val_dis = dis_name in supported_p75
        c_prob = float(cal_probs[i])
        r_tier = rc_bundle.classify_risk_tier(c_prob, is_validated_disease=is_val_dis)

        row_shap = shap_vals[i]
        dom_sums: Dict[str, float] = {
            "historical_disease_lags_and_baselines": 0.0,
            "recent_case_growth_velocity": 0.0,
            "rainfall_and_temperature_climate": 0.0,
            "seasonality_and_calendar": 0.0,
            "spatial_neighbor_and_geography": 0.0,
            "disease_and_district_context": 0.0,
        }
        for j, sv in enumerate(row_shap):
            dom_sums[feat_domains[j]] += float(sv)

        # Top 5 contributing features by |SHAP|
        top_idx = np.argsort(np.abs(row_shap))[::-1][:5]
        top_feats_list = []
        contrib_dict = {}
        for j in top_idx:
            fn = feat_names[j]
            sv = round(float(row_shap[j]), 4)
            top_feats_list.append(fn)
            contrib_dict[fn] = sv

        # Check missingness in raw input features
        raw_sub = r[raw_features]
        miss_total = int(raw_sub.isna().sum())
        miss_clim = int(
            sum(
                1
                for k in raw_features
                if pd.isna(r[k]) and classify_feature_domain(k) == "rainfall_and_temperature_climate"
            )
        )
        miss_lags = int(
            sum(
                1
                for k in raw_features
                if pd.isna(r[k])
                and classify_feature_domain(k)
                in ("historical_disease_lags_and_baselines", "recent_case_growth_velocity")
            )
        )
        affected_by_missing = bool(miss_total > 0)

        hist_c = round(dom_sums["historical_disease_lags_and_baselines"], 4)
        grow_c = round(dom_sums["recent_case_growth_velocity"], 4)
        clim_c = round(dom_sums["rainfall_and_temperature_climate"], 4)
        seas_c = round(dom_sums["seasonality_and_calendar"], 4)
        spat_c = round(dom_sums["spatial_neighbor_and_geography"], 4)

        plain_summary = (
            f"{d_name} ({dis_name}, {rep_p}) received a calibrated outbreak risk probability of "
            f"{c_prob:.3f} (Tier: {r_tier}; raw score={float(raw_probs[i]):.3f}, optimal threshold={rc_bundle.optimal_threshold:.3f}). "
            f"Top drivers: {', '.join(f'{k} ({v:+.3f})' for k, v in list(contrib_dict.items())[:3])}. "
            f"Domain log-odds contributions — Historical disease lags: {hist_c:+.3f}, "
            f"Recent case growth: {grow_c:+.3f}, Rainfall/Temperature lags: {clim_c:+.3f}, "
            f"Seasonality: {seas_c:+.3f}, Spatial neighbors: {spat_c:+.3f}. "
            f"Missing input features imputed via training medians: {miss_total} "
            f"(lag features missing={miss_lags}, climate features missing={miss_clim})."
        )

        lim_note = (
            f"Non-causal statistical attribution ({rc_bundle.model_version}). "
            + (
                f"Disease '{dis_name}' has <5 training observations; tier is flagged UNVALIDATED_LIMITED_SAMPLE. "
                if not is_val_dis
                else ""
            )
            + (
                f"{miss_lags} historical lag features were NaN (first/episodic outbreak week) and imputed using training medians."
                if miss_lags > 0
                else "All historical lag and climate features were observed."
            )
        )

        rows.append(
            {
                "prediction_or_anomaly_id": pred_id,
                "explanation_type": "OUTBREAK_RISK_CLASSIFICATION",
                "district_id": DISTRICT_ID_MAP.get(d_name, "UNKNOWN"),
                "district": d_name,
                "disease": dis_name,
                "reporting_period": rep_p,
                "chronological_split": str(r["split"]),
                "model_version": rc_bundle.model_version,
                "explanation_method": (
                    "Calibrated HistGradientBoosting Probability + Exact XGBoost TreeSHAP Log-Odds Decomposition"
                ),
                "model_estimated_risk_prob": round(c_prob, 4),
                "risk_category": r_tier,
                "observed_target_elevated_p75": (
                    int(r["target_severe_outbreak_p75"])
                    if pd.notna(r["target_severe_outbreak_p75"])
                    else -1
                ),
                "important_features": json.dumps(top_feats_list),
                "contribution_values": json.dumps(contrib_dict),
                "base_margin_log_odds": round(float(base_margin[i]), 4),
                "historical_disease_contribution": hist_c,
                "case_growth_contribution": grow_c,
                "climate_contribution": clim_c,
                "seasonal_contribution": seas_c,
                "spatial_neighbor_contribution": spat_c,
                "missing_features_count": miss_total,
                "missing_lag_features_count": miss_lags,
                "missing_climate_features_count": miss_clim,
                "affected_by_missing_data": affected_by_missing,
                "plain_language_summary": plain_summary,
                "data_limitations": lim_note,
                "scientific_disclaimer": MANDATORY_XAI_NON_CAUSAL_DISCLAIMER,
                "generation_timestamp": gen_ts,
            }
        )

    return pd.DataFrame(rows)


def compute_local_forecast_explanations(
    proj_dir: Path = PROJ, bundles_dict: Dict[str, Any] = None
) -> pd.DataFrame:
    """Generate local forecast explanations across 7d, 14d, and 30d horizons."""
    fc_bundle = bundles_dict["fc_bundle"]
    df_fc = pd.read_csv(proj_dir / "data" / "final_training" / "forecasting_dataset.csv")

    gen_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    rows: List[Dict[str, Any]] = []

    for h_key in ["7d", "14d", "30d"]:
        hb = fc_bundle.horizon_bundles[h_key]
        prep = hb["preprocessor"]
        sel_feats = hb["selected_features"]

        valid_df = df_fc[df_fc["target_cases_t"].notna()].reset_index(drop=True)
        X_trans = prep.transform(valid_df[sel_feats])
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        X_trans = np.asarray(X_trans, dtype=float)

        feat_names = [clean_transformed_feature_name(f) for f in prep.get_feature_names_out()]
        feat_domains = [classify_feature_domain(f) for f in feat_names]

        # Primary model forecast + intervals
        fc_out = fc_bundle.predict_with_intervals(X_trans, horizon=h_key)
        pt_fc = fc_out["point_forecast"]
        lo80 = fc_out["lower_80"]
        hi80 = fc_out["upper_80"]

        # Exact TreeSHAP on log1p(cases) from XGBoost_Lagged_Regressor
        xgb_est = hb["ml_models"]["XGBoost_Lagged_Regressor"]
        shap_contribs = xgb_est.get_booster().predict(xgb.DMatrix(X_trans), pred_contribs=True)
        shap_vals = np.asarray(shap_contribs[:, :-1], dtype=float)

        for i, r in valid_df.iterrows():
            d_name = str(r["district_std"])
            dis_name = str(r["disease_std"])
            yr = int(r["report_year"])
            wk = int(r["report_week"])
            rep_p = f"{yr}-W{wk:02d}"
            pred_id = f"FC_{h_key.upper()}_{DISTRICT_ID_MAP.get(d_name, 'MH')}_{rep_p}_{i:03d}"

            obs_val = float(r["target_cases_t"])
            pred_val = float(pt_fc[i])
            resid_val = round(obs_val - pred_val, 2)
            pi80_w = round(float(hi80[i] - lo80[i]), 2)

            row_shap = shap_vals[i]
            dom_sums: Dict[str, float] = {
                "historical_disease_lags_and_baselines": 0.0,
                "recent_case_growth_velocity": 0.0,
                "rainfall_and_temperature_climate": 0.0,
                "seasonality_and_calendar": 0.0,
                "spatial_neighbor_and_geography": 0.0,
                "disease_and_district_context": 0.0,
            }
            for j, sv in enumerate(row_shap):
                dom_sums[feat_domains[j]] += float(sv)

            top_idx = np.argsort(np.abs(row_shap))[::-1][:5]
            top_feats = [feat_names[j] for j in top_idx]
            contrib_dict = {feat_names[j]: round(float(row_shap[j]), 4) for j in top_idx}

            miss_cnt = int(r[sel_feats].isna().sum())
            hist_c = round(
                dom_sums["historical_disease_lags_and_baselines"] + dom_sums["disease_and_district_context"], 4
            )
            lag_grow_c = round(dom_sums["recent_case_growth_velocity"] + dom_sums["spatial_neighbor_and_geography"], 4)
            clim_c = round(dom_sums["rainfall_and_temperature_climate"], 4)
            seas_c = round(dom_sums["seasonality_and_calendar"], 4)

            within_80 = bool(lo80[i] <= obs_val <= hi80[i])
            resid_note = (
                f"Observed={obs_val:.0f} vs Forecast={pred_val:.1f} (residual={resid_val:+.1f}; "
                f"{'inside' if within_80 else 'outside'} 80% PI [{lo80[i]:.1f}, {hi80[i]:.1f}])."
            )

            plain_summary = (
                f"{h_key} ({hb['horizon_weeks']}w-ahead) forecast for {d_name} — {dis_name} ({rep_p}): "
                f"predicted {pred_val:.1f} cases (80% PI: [{lo80[i]:.1f}, {hi80[i]:.1f}], width={pi80_w:.1f}). "
                f"Log-scale TreeSHAP decomposition — Historical baseline/lags: {hist_c:+.3f}, "
                f"Case momentum/spatial: {lag_grow_c:+.3f}, Climate lags: {clim_c:+.3f}, "
                f"Seasonality: {seas_c:+.3f}. {resid_note}"
            )

            rows.append(
                {
                    "prediction_or_anomaly_id": pred_id,
                    "explanation_type": f"DISEASE_FORECAST_{h_key.upper()}",
                    "horizon": h_key,
                    "horizon_weeks": int(hb["horizon_weeks"]),
                    "district_id": DISTRICT_ID_MAP.get(d_name, "UNKNOWN"),
                    "district": d_name,
                    "disease": dis_name,
                    "reporting_period": rep_p,
                    "chronological_split": str(r["split"]),
                    "model_version": fc_bundle.model_version,
                    "explanation_method": (
                        f"{fc_bundle.selected_model_by_horizon[h_key]} Conformal Interval + XGBoost TreeSHAP Log1p Decomposition"
                    ),
                    "predicted_cases": pred_val,
                    "observed_future_cases": obs_val,
                    "forecast_residual": resid_val,
                    "pi_80_lower": float(lo80[i]),
                    "pi_80_upper": float(hi80[i]),
                    "pi_80_uncertainty_width": pi80_w,
                    "within_80pct_interval": within_80,
                    "important_features": json.dumps(top_feats),
                    "contribution_values": json.dumps(contrib_dict),
                    "historical_trend_contribution": hist_c,
                    "lagged_case_contribution": lag_grow_c,
                    "climate_feature_contribution": clim_c,
                    "seasonal_contribution": seas_c,
                    "missing_features_count": miss_cnt,
                    "affected_by_missing_data": bool(miss_cnt > 0),
                    "plain_language_summary": plain_summary,
                    "data_limitations": (
                        f"Horizon {h_key} strictly excludes lags < {hb['horizon_weeks']}w to prevent leakage; "
                        f"{miss_cnt} missing input features imputed via training medians."
                    ),
                    "scientific_disclaimer": MANDATORY_XAI_NON_CAUSAL_DISCLAIMER,
                    "generation_timestamp": gen_ts,
                }
            )

    return pd.DataFrame(rows)


def compute_local_anomaly_explanations(
    proj_dir: Path = PROJ, bundles_dict: Dict[str, Any] = None
) -> pd.DataFrame:
    """Generate structured explanation records for all anomaly detection outputs (Phase 7 Section 6 & 7)."""
    ad_bundle = bundles_dict["ad_bundle"]
    anom_df = pd.read_csv(proj_dir / "reports" / "anomaly_detection_results.csv")
    gen_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    rows: List[Dict[str, Any]] = []
    for i, r in anom_df.iterrows():
        d_name = str(r["district"])
        dis_name = str(r["disease"])
        rep_p = str(r["reporting_period"])
        anom_id = f"ANOM_{DISTRICT_ID_MAP.get(d_name, 'MH')}_{rep_p}_{i:03d}"

        obs_cases = int(r["observed_cases"])
        obs_deaths = float(r["observed_deaths"]) if pd.notna(r["observed_deaths"]) else 0.0
        a_score = float(r["anomaly_score"])
        is_anom = bool(r["is_anomaly"])
        tier = str(r["anomaly_severity_tier"])
        sig_types = str(r["signal_types"])
        hist_base = str(r["historical_baseline"])
        trig_feats = str(r["triggering_features"])

        sz = float(r["seasonal_zscore"]) if pd.notna(r["seasonal_zscore"]) else 0.0
        rz = float(r["rolling_zscore"]) if pd.notna(r["rolling_zscore"]) else 0.0
        mz = float(r["robust_mad_zscore"]) if pd.notna(r["robust_mad_zscore"]) else 0.0
        if_sc = float(r["isolation_forest_score"]) if pd.notna(r["isolation_forest_score"]) else 0.0
        m_cnt = int(r["methods_flagged_count"])

        contrib_dict = {
            "seasonal_zscore": round(sz, 3),
            "rolling_zscore": round(rz, 3),
            "robust_mad_zscore": round(mz, 3),
            "isolation_forest_score": round(if_sc, 3),
        }
        imp_feats = [t.strip() for t in trig_feats.split(";") if t.strip()]

        plain_summary = (
            f"{d_name} — {dis_name} ({rep_p}): Observed={obs_cases} cases ({int(obs_deaths)} deaths) vs. "
            f"Expected Baseline [{hist_base}]. Composite Anomaly Score={a_score:.3f} "
            f"(Flagged={is_anom}, Tier={tier}, Signals={sig_types}). "
            f"Method deviations: Seasonal Z={sz:+.2f}, Rolling Z={rz:+.2f}, Robust MAD Z={mz:+.2f}, "
            f"Isolation Forest={if_sc:.3f} ({m_cnt}/4 methods triggered; threshold={ad_bundle.score_threshold}). "
            f"Triggering evidence: {trig_feats}."
        )

        rows.append(
            {
                "prediction_or_anomaly_id": anom_id,
                "explanation_type": "DISEASE_ANOMALY_DETECTION",
                "district_id": DISTRICT_ID_MAP.get(d_name, "UNKNOWN"),
                "district": d_name,
                "disease": dis_name,
                "reporting_period": rep_p,
                "chronological_split": str(r["split"]),
                "model_version": str(r["model_version"]),
                "explanation_method": str(r["method"]),
                "is_anomaly": is_anom,
                "anomaly_severity_tier": tier,
                "signal_types": sig_types,
                "observed_value": obs_cases,
                "observed_deaths": obs_deaths,
                "expected_baseline": hist_base,
                "deviation_seasonal_z": round(sz, 4),
                "deviation_rolling_z": round(rz, 4),
                "deviation_robust_mad_z": round(mz, 4),
                "isolation_forest_score": round(if_sc, 4),
                "composite_anomaly_score": round(a_score, 4),
                "threshold_applied": f"composite_score >= {ad_bundle.score_threshold} (high_priority >= {ad_bundle.high_priority_threshold}) or >=2 method triggers",
                "important_features": json.dumps(imp_feats),
                "contribution_values": json.dumps(contrib_dict),
                "supporting_historical_context": (
                    f"Reporting coverage factor={r['reporting_coverage_factor']}, "
                    f"empirical reporting delay={r['empirical_reporting_delay_days']} days."
                ),
                "plain_language_summary": plain_summary,
                "data_limitations": str(r["interpretation_note"]),
                "scientific_disclaimer": MANDATORY_XAI_NON_CAUSAL_DISCLAIMER,
                "generation_timestamp": gen_ts,
            }
        )

    return pd.DataFrame(rows)
