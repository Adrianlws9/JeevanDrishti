"""Step 4 of Phase 8: Advanced ML Feature Importance, Exact TreeSHAP, and Partial Dependence Analysis.

Implements:
1. Random Forest feature importance (impurity & permutation importance).
2. XGBoost feature importance (gain & weight).
3. Exact TreeSHAP explanations (`xgboost.Booster.predict(DMatrix, pred_contribs=True)`),
   computing mean(|SHAP|) and directional sign while controlling for seasonality
   (`week_sin`, `week_cos`) and geography (`district_lat`, `district_lon`).
4. 1D Partial Dependence Analysis (PDP) across key climate covariates holding
   seasonal and geographic confounders constant.

Strict Scientific Rule:
Feature importance, TreeSHAP values, and Partial Dependence curves measure predictive
contribution inside the fitted model and do NOT establish physical or epidemiological causation.
"""
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
import xgboost as xgb
from xgboost import XGBRegressor

from . import CAUSALITY_DISCLAIMER

PROJ = Path(__file__).resolve().parents[3]
CLIMATE_INTEL_DATA_DIR = PROJ / "data" / "climate_intelligence"
CLIMATE_INTEL_REPORTS_DIR = PROJ / "reports" / "climate_intelligence"

CLIMATE_CANDIDATE_COLS = [
    "rain_mm_sum_t",
    "rain_mm_sum_lag_1w",
    "rain_mm_sum_lag_2w",
    "rain_mm_sum_lag_4w",
    "rain_mm_sum_lag_8w",
    "rain_mm_accum_prev_2w",
    "rain_mm_accum_prev_4w",
    "rain_mm_accum_prev_8w",
    "rain_mm_deviation_expanding_season_abs_t",
    "tmin_c_mean_t",
    "tmin_c_mean_lag_1w",
    "tmin_c_mean_lag_4w",
    "tmin_c_mean_lag_8w",
    "tmax_c_mean_t",
    "tmax_c_mean_lag_1w",
    "tmax_c_mean_lag_4w",
    "tmax_c_mean_lag_8w",
    "tavg_c_mean_lag_1w",
    "tavg_c_mean_lag_4w",
    "temp_diurnal_range_mean_t",
    "temp_diurnal_range_lag_1w",
    "temp_diurnal_range_lag_4w",
]

CONFOUNDER_COLS = [
    "week_sin",
    "week_cos",
    "district_lat",
    "district_lon",
]


def _fit_rf_xgb_and_shap(
    df_sub: pd.DataFrame,
    target_col: str,
    task_name: str,
    feature_cols: List[str],
) -> Tuple[List[Dict[str, Any]], XGBRegressor, np.ndarray, np.ndarray]:
    """Fit Random Forest and XGBoost on chronological train split (period_idx <= 84) and compute
    RF impurity, RF out-of-sample permutation importance, XGBoost gain, and exact TreeSHAP."""
    work = df_sub[df_sub[target_col].notna()].copy().reset_index(drop=True)
    imp = SimpleImputer(strategy="median")

    tr_mask = (work["period_idx"] <= 84).to_numpy()
    ho_mask = (work["period_idx"] > 84).to_numpy()
    if tr_mask.sum() < 15:
        tr_mask = np.ones(len(work), dtype=bool)
        ho_mask = np.ones(len(work), dtype=bool)

    X_tr = imp.fit_transform(work.loc[tr_mask, feature_cols])
    X_ho = imp.transform(work.loc[ho_mask, feature_cols])
    X_all = imp.transform(work[feature_cols])

    y_tr = work.loc[tr_mask, target_col].to_numpy(dtype=float)
    y_ho = work.loc[ho_mask, target_col].to_numpy(dtype=float)

    rf = RandomForestRegressor(
        n_estimators=150,
        max_depth=5,
        min_samples_leaf=5,
        n_jobs=1,
        random_state=42,
    )
    rf.fit(X_tr, y_tr)
    rf_imp = rf.feature_importances_
    perm = permutation_importance(
        rf,
        X_ho if len(y_ho) >= 8 else X_tr,
        y_ho if len(y_ho) >= 8 else y_tr,
        n_repeats=5,
        n_jobs=1,
        random_state=42,
    )
    rf_perm = np.maximum(0.0, perm.importances_mean)

    xgb_reg = XGBRegressor(
        n_estimators=80,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=3.0,
        n_jobs=1,
        random_state=42,
    )
    xgb_reg.fit(X_tr, y_tr)
    xgb_gain = xgb_reg.feature_importances_

    # Exact native TreeSHAP via XGBoost booster (`pred_contribs=True` returns shape [n_samples, n_features + 1])
    booster = xgb_reg.get_booster()
    dmat_all = xgb.DMatrix(X_all, feature_names=feature_cols)
    shap_contribs = booster.predict(dmat_all, pred_contribs=True)
    shap_vals = shap_contribs[:, :-1]  # last column is expected value bias
    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

    # Normalize importances for clean comparison
    rf_imp_norm = rf_imp / max(rf_imp.sum(), 1e-9)
    rf_perm_norm = rf_perm / max(rf_perm.sum(), 1e-9)
    xgb_gain_norm = xgb_gain / max(xgb_gain.sum(), 1e-9)
    shap_share = mean_abs_shap / max(mean_abs_shap.sum(), 1e-9)

    rows: List[Dict[str, Any]] = []
    for idx, f_name in enumerate(feature_cols):
        f_vals = X_all[:, idx]
        s_vals = shap_vals[:, idx]
        if np.std(f_vals) > 1e-8 and np.std(s_vals) > 1e-8:
            dir_corr = float(np.corrcoef(f_vals, s_vals)[0, 1])
        else:
            dir_corr = 0.0

        is_confounder = f_name in CONFOUNDER_COLS
        consensus_score = (
            0.35 * float(shap_share[idx])
            + 0.30 * float(xgb_gain_norm[idx])
            + 0.20 * float(rf_imp_norm[idx])
            + 0.15 * float(rf_perm_norm[idx])
        )
        rows.append(
            {
                "analysis_task": task_name,
                "target_variable": target_col,
                "n_samples": int(len(work)),
                "feature_name": f_name,
                "feature_role": "Seasonal_or_Geographic_Confounder_Control" if is_confounder else "Climate_Variable",
                "rf_impurity_importance": round(float(rf_imp_norm[idx]), 4),
                "rf_permutation_importance": round(float(rf_perm_norm[idx]), 4),
                "xgb_gain_importance": round(float(xgb_gain_norm[idx]), 4),
                "mean_abs_treeshap": round(float(mean_abs_shap[idx]), 5),
                "treeshap_share": round(float(shap_share[idx]), 4),
                "shap_feature_direction_corr": round(dir_corr, 4),
                "consensus_importance_score": round(consensus_score, 4),
                "non_causal_interpretation_note": CAUSALITY_DISCLAIMER,
            }
        )
    rows.sort(key=lambda r: r["consensus_importance_score"], reverse=True)
    return rows, xgb_reg, X_all, shap_vals


def compute_partial_dependence_curves(
    xgb_reg: XGBRegressor,
    X_all: np.ndarray,
    feature_cols: List[str],
    task_name: str,
    pdp_features: List[str],
    n_grid: int = 20,
) -> List[Dict[str, Any]]:
    """Compute 1D Partial Dependence profiles holding all other covariates (including seasonality) at observed values."""
    pdp_rows: List[Dict[str, Any]] = []
    for f_name in pdp_features:
        if f_name not in feature_cols:
            continue
        f_idx = feature_cols.index(f_name)
        col_vals = X_all[:, f_idx]
        grid = np.linspace(np.percentile(col_vals, 5), np.percentile(col_vals, 95), n_grid)
        for g_val in grid:
            X_mod = X_all.copy()
            X_mod[:, f_idx] = g_val
            preds = xgb_reg.predict(X_mod)
            pdp_rows.append(
                {
                    "analysis_task": task_name,
                    "climate_feature": f_name,
                    "grid_value": round(float(g_val), 3),
                    "mean_predicted_log1p_cases": round(float(np.mean(preds)), 4),
                    "mean_predicted_cases_equiv": round(float(np.expm1(np.mean(preds))), 2),
                    "p25_predicted_log1p": round(float(np.percentile(preds, 25)), 4),
                    "p75_predicted_log1p": round(float(np.percentile(preds, 75)), 4),
                    "confounders_held_constant": "week_sin, week_cos, district_lat, district_lon, other climate lags",
                    "interpretation_note": CAUSALITY_DISCLAIMER,
                }
            )
    return pdp_rows


def run_advanced_ml_and_shap_analysis(
    aligned_outbreaks: pd.DataFrame,
    dw_panel: pd.DataFrame,
) -> Dict[str, Any]:
    """Execute Random Forest, XGBoost, Exact TreeSHAP, and Partial Dependence analyses."""
    all_features = CLIMATE_CANDIDATE_COLS + CONFOUNDER_COLS
    all_imp_rows: List[Dict[str, Any]] = []
    all_pdp_rows: List[Dict[str, Any]] = []

    pdp_key_vars = [
        "rain_mm_accum_prev_4w",
        "rain_mm_sum_lag_1w",
        "tmin_c_mean_lag_1w",
        "tmax_c_mean_lag_1w",
        "temp_diurnal_range_lag_1w",
    ]

    # 1. Pooled All-Outbreaks Severity (`log1p_cases`, n = 508)
    rows_pool, xgb_pool, X_pool, shap_pool = _fit_rf_xgb_and_shap(
        aligned_outbreaks,
        "log1p_cases",
        "Pooled_Outbreak_Severity_All_Diseases",
        all_features,
    )
    all_imp_rows.extend(rows_pool)
    all_pdp_rows.extend(
        compute_partial_dependence_curves(
            xgb_pool, X_pool, all_features, "Pooled_Outbreak_Severity_All_Diseases", pdp_key_vars
        )
    )

    # 2. Disease-specific RF + XGBoost + TreeSHAP + PDP for major diseases
    disease_shap_store: Dict[str, Any] = {
        "Pooled_All_Diseases": {
            "feature_cols": all_features,
            "X_all": X_pool,
            "shap_vals": shap_pool,
        }
    }
    for dis in ("Dengue", "Acute Diarrhoeal Disease", "Food Poisoning", "Malaria", "Chikungunya"):
        sub_d = aligned_outbreaks[aligned_outbreaks["disease_std"] == dis]
        if len(sub_d) >= 25:
            r_d, xgb_d, X_d, sh_d = _fit_rf_xgb_and_shap(
                sub_d, "log1p_cases", f"Disease_Specific_{dis}", all_features
            )
            all_imp_rows.extend(r_d)
            all_pdp_rows.extend(
                compute_partial_dependence_curves(
                    xgb_d, X_d, all_features, f"Disease_Specific_{dis}", pdp_key_vars
                )
            )
            disease_shap_store[dis] = {
                "feature_cols": all_features,
                "X_all": X_d,
                "shap_vals": sh_d,
            }

    # 3. Full District-Week Surveillance Panel (35 districts x 121 published IDSP weeks = 4,235 rows)
    pub_dw = dw_panel[dw_panel["state_idsp_bulletin_published_in_week"] == 1].copy()
    r_dw, xgb_dw, X_dw, sh_dw = _fit_rf_xgb_and_shap(
        pub_dw,
        "log1p_total_outbreak_cases",
        "District_Week_Panel_All_Diseases_Burden",
        all_features,
    )
    all_imp_rows.extend(r_dw)

    imp_df = pd.DataFrame(all_imp_rows)
    pdp_df = pd.DataFrame(all_pdp_rows)

    for out_dir in (CLIMATE_INTEL_DATA_DIR, CLIMATE_INTEL_REPORTS_DIR):
        imp_df.to_csv(out_dir / "climate_feature_importance_and_shap.csv", index=False)
        pdp_df.to_csv(out_dir / "partial_dependence_profiles.csv", index=False)

    return {
        "imp_df": imp_df,
        "pdp_df": pdp_df,
        "disease_shap_store": disease_shap_store,
    }
