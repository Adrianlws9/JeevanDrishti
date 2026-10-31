"""Global Model Explainability Engine (Phase 13 Section 2 & 3).

Generates global feature rankings, importance scores, direction of influence, and
domain-level attribution across:
1. Phase 5 Outbreak Risk Classification (`phase5-v1.0.0`):
   - Logistic Regression standardized coefficients (β_j) & Odds Ratios
   - Random Forest Gini impurity importance
   - XGBoost Exact TreeSHAP (`pred_contribs=True`) + signed direction of influence
   - HistGradientBoosting (primary model) held-out Permutation Feature Importance
2. Phase 6 Multi-Horizon Disease Forecasting (`phase6-v1.0.0` for `7d`, `14d`, `30d`):
   - XGBoost Exact TreeSHAP (`pred_contribs=True`) + domain share decomposition
   - Random Forest impurity importance
   - Ridge Regression standardized coefficients
   - HistGradientBoosting (primary model) held-out Permutation Feature Importance
3. Phase 7 Isolation Forest Anomaly Detector (`phase7-v1.0.0`):
   - 1-SD Feature Perturbation Sensitivity across all 14 anomaly input features.
"""
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
import xgboost as xgb

from src.engines.explainability.model_inspector import (
    MANDATORY_XAI_NON_CAUSAL_DISCLAIMER,
    PROJ,
    classify_feature_domain,
    clean_transformed_feature_name,
)


def _compute_feature_shap_direction(x_col: np.ndarray, shap_col: np.ndarray) -> str:
    """Determine whether higher feature values increase or decrease model output."""
    if np.std(x_col) < 1e-9 or np.std(shap_col) < 1e-9:
        return "NEUTRAL_OR_CONSTANT"
    corr = float(np.corrcoef(x_col, shap_col)[0, 1])
    if np.isnan(corr) or abs(corr) < 0.05:
        return "NONLINEAR_MIXED_DIRECTION"
    return (
        "POSITIVE_HIGHER_VALUE_INCREASES_OUTPUT"
        if corr > 0
        else "NEGATIVE_HIGHER_VALUE_DECREASES_OUTPUT"
    )


def compute_phase5_global_explanations(
    proj_dir: Path, bundles_dict: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute global feature explanations for Phase 5 Outbreak Risk Classification models."""
    rc_bundle = bundles_dict["rc_bundle"]
    rc_prep = bundles_dict["rc_prep"]
    rc_meta = bundles_dict["rc_meta"]

    schema = json.loads(
        (proj_dir / "models" / "risk_classification" / "feature_schema.json").read_text(encoding="utf-8")
    )
    df_rc = pd.read_csv(proj_dir / "data" / "final_training" / "risk_classification_dataset.csv")

    raw_features = schema["categorical_features"] + schema["numerical_features"]
    X_all_trans = rc_prep.transform(df_rc[raw_features])
    if hasattr(X_all_trans, "toarray"):
        X_all_trans = X_all_trans.toarray()
    X_all_trans = np.asarray(X_all_trans, dtype=float)

    eval_mask = (
        df_rc["split"].isin(["validation", "test"]) & df_rc["target_severe_outbreak_p75"].notna()
    ).to_numpy()
    X_eval = X_all_trans[eval_mask]
    y_eval = df_rc.loc[eval_mask, "target_severe_outbreak_p75"].to_numpy(dtype=int)

    feat_names = [clean_transformed_feature_name(f) for f in rc_bundle.feature_names_out]
    n_feat = len(feat_names)

    # 1. Logistic Regression standardized coefficients
    lr_est = rc_bundle.estimators_dict["Logistic_Regression_Baseline"]
    lr_coefs = np.asarray(lr_est.coef_[0], dtype=float)

    # 2. Random Forest Gini importance
    rf_est = rc_bundle.estimators_dict["Random_Forest_Classifier"]
    rf_imp = np.asarray(rf_est.feature_importances_, dtype=float)

    # 3. XGBoost Exact TreeSHAP (log-odds margin contributions)
    xgb_est = rc_bundle.estimators_dict["XGBoost_Classifier"]
    dmat = xgb.DMatrix(X_all_trans)
    shap_contribs = xgb_est.get_booster().predict(dmat, pred_contribs=True)
    shap_vals = np.asarray(shap_contribs[:, :-1], dtype=float)
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

    # 4. HistGradientBoosting (primary model) Permutation Importance on held-out eval split
    hgb_est = rc_bundle.primary_estimator
    perm_res = permutation_importance(
        hgb_est,
        X_eval,
        y_eval,
        scoring="average_precision",
        n_repeats=8,
        random_state=42,
        n_jobs=1,
    )
    hgb_perm_imp = np.maximum(0.0, np.asarray(perm_res.importances_mean, dtype=float))

    gen_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    rows: List[Dict[str, Any]] = []
    for j in range(n_feat):
        fname = feat_names[j]
        dom = classify_feature_domain(fname)
        shap_dir = _compute_feature_shap_direction(X_all_trans[:, j], shap_vals[:, j])
        lr_dir = (
            "POSITIVE_HIGHER_VALUE_INCREASES_RISK"
            if lr_coefs[j] > 0.01
            else (
                "NEGATIVE_HIGHER_VALUE_DECREASES_RISK"
                if lr_coefs[j] < -0.01
                else "NEAR_ZERO_LINEAR_EFFECT"
            )
        )
        rows.append(
            {
                "phase": "Phase 5",
                "engine_domain": "Outbreak Risk Classification",
                "horizon_or_task": "binary_elevated_outbreak_burden_p75",
                "model_version": rc_bundle.model_version,
                "feature_name": fname,
                "feature_domain": dom,
                "xgb_mean_abs_treeshap": round(float(mean_abs_shap[j]), 6),
                "xgb_treeshap_direction": shap_dir,
                "hgb_primary_permutation_importance_pr_auc": round(float(hgb_perm_imp[j]), 6),
                "rf_gini_importance": round(float(rf_imp[j]), 6),
                "logreg_standardized_coef_beta": round(float(lr_coefs[j]), 6),
                "logreg_odds_ratio": round(float(np.exp(np.clip(lr_coefs[j], -10, 10))), 4),
                "logreg_direction": lr_dir,
                "composite_rank_score": round(
                    float(
                        (mean_abs_shap[j] / (np.max(mean_abs_shap) + 1e-9)) * 0.40
                        + (hgb_perm_imp[j] / (np.max(hgb_perm_imp) + 1e-9)) * 0.30
                        + (rf_imp[j] / (np.max(rf_imp) + 1e-9)) * 0.15
                        + (abs(lr_coefs[j]) / (np.max(np.abs(lr_coefs)) + 1e-9)) * 0.15
                    ),
                    6,
                ),
                "dataset_context": f"{rc_meta['source_dataset_path']} (Train=356, Val=79, Test=76)",
                "non_causal_disclaimer": MANDATORY_XAI_NON_CAUSAL_DISCLAIMER,
                "generation_timestamp": gen_ts,
            }
        )

    df_out = (
        pd.DataFrame(rows)
        .sort_values("composite_rank_score", ascending=False)
        .reset_index(drop=True)
    )
    df_out["global_importance_rank"] = np.arange(1, len(df_out) + 1)

    domain_shares = (
        df_out.groupby("feature_domain")["xgb_mean_abs_treeshap"].sum()
        / (df_out["xgb_mean_abs_treeshap"].sum() + 1e-9)
        * 100.0
    ).round(2).to_dict()

    summary = {
        "model_version": rc_bundle.model_version,
        "primary_model": rc_bundle.primary_model_name,
        "optimal_decision_threshold": rc_bundle.optimal_threshold,
        "top_10_features": df_out.head(10)[
            [
                "global_importance_rank",
                "feature_name",
                "feature_domain",
                "xgb_mean_abs_treeshap",
                "hgb_primary_permutation_importance_pr_auc",
                "rf_gini_importance",
                "logreg_standardized_coef_beta",
                "xgb_treeshap_direction",
            ]
        ].to_dict(orient="records"),
        "domain_treeshap_share_pct": domain_shares,
    }
    return df_out, summary


def compute_phase6_global_explanations(
    proj_dir: Path, bundles_dict: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute global feature explanations across all 3 forecast horizons ('7d', '14d', '30d')."""
    fc_bundle = bundles_dict["fc_bundle"]
    fc_meta = bundles_dict["fc_meta"]
    df_fc = pd.read_csv(proj_dir / "data" / "final_training" / "forecasting_dataset.csv")

    gen_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    all_rows: List[Dict[str, Any]] = []
    horizon_summaries: Dict[str, Any] = {}

    for h_key in ["7d", "14d", "30d"]:
        hb = fc_bundle.horizon_bundles[h_key]
        prep = hb["preprocessor"]
        sel_feats = hb["selected_features"]

        valid_df = df_fc[df_fc["target_cases_t"].notna()].reset_index(drop=True)
        X_trans = prep.transform(valid_df[sel_feats])
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        X_trans = np.asarray(X_trans, dtype=float)

        eval_mask = valid_df["split"].isin(["validation", "test"]).to_numpy()
        X_eval = X_trans[eval_mask]
        y_eval_log = np.log1p(valid_df.loc[eval_mask, "target_cases_t"].to_numpy(dtype=float))

        feat_names = [clean_transformed_feature_name(f) for f in prep.get_feature_names_out()]
        n_feat = len(feat_names)

        # 1. XGBoost Exact TreeSHAP
        xgb_est = hb["ml_models"]["XGBoost_Lagged_Regressor"]
        shap_contribs = xgb_est.get_booster().predict(xgb.DMatrix(X_trans), pred_contribs=True)
        shap_vals = np.asarray(shap_contribs[:, :-1], dtype=float)
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

        # 2. Random Forest Impurity Importance
        rf_est = hb["ml_models"]["RandomForest_Lagged_Regressor"]
        rf_imp = np.asarray(rf_est.feature_importances_, dtype=float)

        # 3. Ridge Standardized Coefficients
        ridge_est = hb["ml_models"]["Ridge_Lagged_Regressor"]
        ridge_coefs = np.asarray(ridge_est.coef_, dtype=float)

        # 4. HistGradientBoosting (primary model) Permutation Importance on held-out eval split
        hgb_est = hb["ml_models"]["HistGradientBoosting_Regressor"]
        perm_res = permutation_importance(
            hgb_est,
            X_eval,
            y_eval_log,
            scoring="neg_mean_absolute_error",
            n_repeats=6,
            random_state=42,
            n_jobs=1,
        )
        hgb_perm_imp = np.maximum(0.0, np.asarray(perm_res.importances_mean, dtype=float))

        h_rows: List[Dict[str, Any]] = []
        for j in range(n_feat):
            fname = feat_names[j]
            dom = classify_feature_domain(fname)
            shap_dir = _compute_feature_shap_direction(X_trans[:, j], shap_vals[:, j])
            h_rows.append(
                {
                    "phase": "Phase 6",
                    "engine_domain": "Time-Series Disease Forecasting",
                    "horizon_or_task": f"forecast_{h_key}_{hb['horizon_weeks']}w_ahead",
                    "model_version": fc_bundle.model_version,
                    "feature_name": fname,
                    "feature_domain": dom,
                    "xgb_mean_abs_treeshap": round(float(mean_abs_shap[j]), 6),
                    "xgb_treeshap_direction": shap_dir,
                    "hgb_primary_permutation_importance_pr_auc": round(float(hgb_perm_imp[j]), 6),
                    "rf_gini_importance": round(float(rf_imp[j]), 6),
                    "logreg_standardized_coef_beta": round(float(ridge_coefs[j]), 6),
                    "logreg_odds_ratio": np.nan,
                    "logreg_direction": (
                        "POSITIVE_INCREASES_FORECAST_CASES"
                        if ridge_coefs[j] > 0.005
                        else (
                            "NEGATIVE_DECREASES_FORECAST_CASES"
                            if ridge_coefs[j] < -0.005
                            else "NEAR_ZERO_LINEAR_EFFECT"
                        )
                    ),
                    "composite_rank_score": round(
                        float(
                            (mean_abs_shap[j] / (np.max(mean_abs_shap) + 1e-9)) * 0.45
                            + (hgb_perm_imp[j] / (np.max(hgb_perm_imp) + 1e-9)) * 0.30
                            + (rf_imp[j] / (np.max(rf_imp) + 1e-9)) * 0.15
                            + (abs(ridge_coefs[j]) / (np.max(np.abs(ridge_coefs)) + 1e-9)) * 0.10
                        ),
                        6,
                    ),
                    "dataset_context": f"{fc_meta['source_dataset_path']} (Horizon={h_key}, N={len(valid_df)})",
                    "non_causal_disclaimer": MANDATORY_XAI_NON_CAUSAL_DISCLAIMER,
                    "generation_timestamp": gen_ts,
                }
            )

        h_df = (
            pd.DataFrame(h_rows)
            .sort_values("composite_rank_score", ascending=False)
            .reset_index(drop=True)
        )
        h_df["global_importance_rank"] = np.arange(1, len(h_df) + 1)
        all_rows.extend(h_df.to_dict(orient="records"))

        dom_shares = (
            h_df.groupby("feature_domain")["xgb_mean_abs_treeshap"].sum()
            / (h_df["xgb_mean_abs_treeshap"].sum() + 1e-9)
            * 100.0
        ).round(2).to_dict()

        horizon_summaries[h_key] = {
            "selected_primary_model": fc_bundle.selected_model_by_horizon[h_key],
            "horizon_weeks": hb["horizon_weeks"],
            "n_transformed_features": n_feat,
            "domain_treeshap_share_pct": dom_shares,
            "top_8_features": h_df.head(8)[
                [
                    "global_importance_rank",
                    "feature_name",
                    "feature_domain",
                    "xgb_mean_abs_treeshap",
                    "hgb_primary_permutation_importance_pr_auc",
                    "rf_gini_importance",
                    "xgb_treeshap_direction",
                ]
            ].to_dict(orient="records"),
        }

    return pd.DataFrame(all_rows), {
        "model_version": fc_bundle.model_version,
        "horizons": horizon_summaries,
    }


def compute_phase7_global_explanations(
    proj_dir: Path, bundles_dict: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute global feature sensitivity for the Phase 7 Isolation Forest anomaly detector."""
    ad_bundle = bundles_dict["ad_bundle"]
    ad_meta = bundles_dict["ad_meta"]
    df_ad = pd.read_csv(proj_dir / "data" / "final_training" / "anomaly_detection_dataset.csv")
    df_res = pd.read_csv(proj_dir / "reports" / "anomaly_detection_results.csv")

    if_feats = ad_bundle.iforest_features
    if_df = pd.DataFrame(
        {
            "log1p_cases_t": np.log1p(df_res["observed_cases"].to_numpy(dtype=float)),
            "seasonal_zscore": df_res["seasonal_zscore"].to_numpy(dtype=float),
            "rolling_zscore": df_res["rolling_zscore"].to_numpy(dtype=float),
            "robust_mad_zscore": df_res["robust_mad_zscore"].to_numpy(dtype=float),
            "case_velocity_ratio": df_res["observed_cases"].to_numpy(dtype=float)
            / np.maximum(df_ad["prev_outbreak_roll_mean_4"].fillna(15.0).to_numpy(dtype=float), 1.0),
            "district_burden_ratio": df_res["observed_cases"].to_numpy(dtype=float)
            / np.maximum(df_ad["district_disease_prior_mean_cases"].fillna(15.0).to_numpy(dtype=float), 1.0),
            "observed_cfr": df_res["observed_deaths"].fillna(0.0).to_numpy(dtype=float)
            / np.maximum(df_res["observed_cases"].to_numpy(dtype=float), 1.0),
            "outbreak_events_in_period": df_res["outbreak_clusters_in_period"].to_numpy(dtype=float),
            "periods_since_prev_outbreak": df_ad["periods_since_prev_outbreak"].to_numpy(dtype=float),
            "climate_rain_lag1w_zscore_vs_train": df_ad["climate_rain_lag1w_zscore_vs_train"].to_numpy(dtype=float),
            "climate_tavg_lag1w_zscore_vs_train": df_ad["climate_tavg_lag1w_zscore_vs_train"].to_numpy(dtype=float),
            "rain_mm_deviation_expanding_season_abs_t": df_ad["rain_mm_deviation_expanding_season_abs_t"].to_numpy(
                dtype=float
            ),
            "temp_diurnal_range_mean_t": df_ad["temp_diurnal_range_mean_t"].to_numpy(dtype=float),
            "reporting_coverage_factor": df_res["reporting_coverage_factor"].to_numpy(dtype=float),
        }
    )

    X_trans = ad_bundle.preprocessor.transform(if_df[if_feats])
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
    X_trans = np.asarray(X_trans, dtype=float)

    base_scores = ad_bundle.score_iforest(X_trans)
    gen_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    rows: List[Dict[str, Any]] = []

    for j, fname in enumerate(if_feats):
        X_pert = X_trans.copy()
        col_std = float(np.std(X_trans[:, j])) or 1.0
        X_pert[:, j] = X_pert[:, j] + col_std
        pert_scores = ad_bundle.score_iforest(X_pert)
        mean_abs_shift = float(np.mean(np.abs(pert_scores - base_scores)))
        signed_shift = float(np.mean(pert_scores - base_scores))

        rows.append(
            {
                "phase": "Phase 7",
                "engine_domain": "Disease Anomaly Detection (Isolation Forest)",
                "horizon_or_task": "unsupervised_multivariate_anomaly_score",
                "model_version": ad_bundle.model_version,
                "feature_name": fname,
                "feature_domain": classify_feature_domain(fname),
                "xgb_mean_abs_treeshap": round(mean_abs_shift, 6),
                "xgb_treeshap_direction": (
                    "POSITIVE_PLUS_1SD_INCREASES_ANOMALY_SCORE"
                    if signed_shift > 0
                    else "NEGATIVE_PLUS_1SD_DECREASES_ANOMALY_SCORE"
                ),
                "hgb_primary_permutation_importance_pr_auc": round(mean_abs_shift, 6),
                "rf_gini_importance": round(abs(signed_shift), 6),
                "logreg_standardized_coef_beta": round(signed_shift, 6),
                "logreg_odds_ratio": np.nan,
                "logreg_direction": "PERTURBATION_SENSITIVITY_ON_IFOREST_SCORE",
                "composite_rank_score": round(mean_abs_shift, 6),
                "dataset_context": f"{ad_meta['source_dataset_path']} (N={len(df_ad)}, 14 Isolation Forest features)",
                "non_causal_disclaimer": MANDATORY_XAI_NON_CAUSAL_DISCLAIMER,
                "generation_timestamp": gen_ts,
            }
        )

    df_out = (
        pd.DataFrame(rows)
        .sort_values("composite_rank_score", ascending=False)
        .reset_index(drop=True)
    )
    df_out["global_importance_rank"] = np.arange(1, len(df_out) + 1)

    return df_out, {
        "model_version": ad_bundle.model_version,
        "score_threshold": ad_bundle.score_threshold,
        "high_priority_threshold": ad_bundle.high_priority_threshold,
        "top_features_by_perturbation_sensitivity": df_out.head(10)[
            [
                "global_importance_rank",
                "feature_name",
                "feature_domain",
                "composite_rank_score",
                "xgb_treeshap_direction",
            ]
        ].to_dict(orient="records"),
    }


def compute_all_global_explanations(
    proj_dir: Path = PROJ, bundles_dict: Dict[str, Any] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute and combine global explanations across Phase 5, Phase 6, and Phase 7."""
    df_p5, sum_p5 = compute_phase5_global_explanations(proj_dir, bundles_dict)
    df_p6, sum_p6 = compute_phase6_global_explanations(proj_dir, bundles_dict)
    df_p7, sum_p7 = compute_phase7_global_explanations(proj_dir, bundles_dict)

    combined_df = pd.concat([df_p5, df_p6, df_p7], ignore_index=True)
    global_summary = {
        "generated_at": datetime.now().isoformat(),
        "non_causal_disclaimer": MANDATORY_XAI_NON_CAUSAL_DISCLAIMER,
        "phase5_risk_classification": sum_p5,
        "phase6_disease_forecasting": sum_p6,
        "phase7_anomaly_detection": sum_p7,
    }
    return combined_df, global_summary
