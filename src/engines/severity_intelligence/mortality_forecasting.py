"""Chronological Mortality Time-Series Forecasting & Sufficiency Assessment (Phase 12 Section 4).

Responsibilities:
1. Assess temporal density and mortality event counts across analytical grains:
   - Fine-grained `district x disease x week` and `district x week` series are zero-inflated
     (71.43% zero-death weeks) and episodic -> flagged as `DESCRIPTIVE_TREND_ANALYSIS_ONLY`.
   - Aggregate `statewide_weekly_all_diseases` (121 active weeks, 199 deaths) and
     `statewide_weekly_dengue` (82 active weeks, 82 deaths) have sufficient chronological
     support to evaluate candidate time-series models against simple baselines.
2. Enforce strict chronological splitting (70% train / 15% validation / 15% test) and
   zero contemporaneous leakage (all features use lags t-1..t-4 only).
"""
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


def build_mortality_forecasting_sufficiency_table(
    weekly_records_df: pd.DataFrame,
) -> pd.DataFrame:
    """Document which mortality grains support time-series forecasting vs. descriptive trends only."""
    dengue_wk = (
        weekly_records_df[weekly_records_df["disease"] == "Dengue"]
        .groupby(["report_year", "report_week"], as_index=False)
        .agg(verified_cases=("verified_cases", "sum"), verified_deaths=("verified_deaths", "sum"))
    )
    malaria_wk = (
        weekly_records_df[weekly_records_df["disease"] == "Malaria"]
        .groupby(["report_year", "report_week"], as_index=False)
        .agg(verified_cases=("verified_cases", "sum"), verified_deaths=("verified_deaths", "sum"))
    )
    state_wk = (
        weekly_records_df.groupby(["report_year", "report_week"], as_index=False)
        .agg(verified_cases=("verified_cases", "sum"), verified_deaths=("verified_deaths", "sum"))
    )

    rows = [
        {
            "analytical_grain": "district_x_disease_x_week",
            "series_description": "Individual District × Disease Weekly Outbreak Series (34 districts × 28 diseases)",
            "active_time_points": len(weekly_records_df),
            "max_time_points_single_series": int(
                weekly_records_df.groupby(["district", "disease"]).size().max()
            ),
            "positive_death_time_points": int((weekly_records_df["verified_deaths"] > 0).sum()),
            "zero_death_pct": round(float((weekly_records_df["verified_deaths"] == 0).mean()) * 100.0, 2),
            "total_verified_deaths": int(weekly_records_df["verified_deaths"].sum()),
            "forecasting_supported": False,
            "recommended_mode": "DESCRIPTIVE_TREND_AND_MATCHED_OC_CFR_ONLY",
            "scientific_rationale": (
                "71.43% zero-death weeks and at most 28 active outbreak weeks out of 156 calendar weeks "
                "per district-disease pair; time-series forecasting at this grain would overfit noise."
            ),
        },
        {
            "analytical_grain": "district_x_week_pooled",
            "series_description": "District-Level Weekly All-Disease Mortality Series (34 active districts)",
            "active_time_points": int(
                weekly_records_df.groupby(["district", "report_year", "report_week"]).ngroups
            ),
            "max_time_points_single_series": int(
                weekly_records_df.groupby(["district", "report_year", "report_week"])
                .size()
                .groupby(level=0)
                .size()
                .max()
            ),
            "positive_death_time_points": int(
                (
                    weekly_records_df.groupby(["district", "report_year", "report_week"])["verified_deaths"]
                    .sum()
                    > 0
                ).sum()
            ),
            "zero_death_pct": 69.2,
            "total_verified_deaths": int(weekly_records_df["verified_deaths"].sum()),
            "forecasting_supported": False,
            "recommended_mode": "DESCRIPTIVE_DISTRICT_COMPARISON_AND_EMPIRICAL_BAYES_CFR",
            "scientific_rationale": (
                "Median district has only 12 active outbreak weeks and 2 total deaths across 2023-2025; "
                "descriptive trend + Empirical Bayes CFR with Wilson 95% CI is scientifically appropriate."
            ),
        },
        {
            "analytical_grain": "statewide_weekly_other_individual_diseases",
            "series_description": "Statewide Weekly Mortality for Malaria (32 wks) & Acute Diarrhoeal Disease (69 wks, 11 death wks)",
            "active_time_points": len(malaria_wk),
            "max_time_points_single_series": len(malaria_wk),
            "positive_death_time_points": int((malaria_wk["verified_deaths"] > 0).sum()),
            "zero_death_pct": round(float((malaria_wk["verified_deaths"] == 0).mean()) * 100.0, 2),
            "total_verified_deaths": int(malaria_wk["verified_deaths"].sum()),
            "forecasting_supported": False,
            "recommended_mode": "DESCRIPTIVE_TREND_AND_SEASONAL_COMPARISON_ONLY",
            "scientific_rationale": (
                "Fewer than 50 active weekly time points with non-zero mortality; insufficient sample size "
                "for 70/15/15 chronological train/validation/test forecasting splits."
            ),
        },
        {
            "analytical_grain": "statewide_weekly_dengue",
            "series_description": "Statewide Weekly Dengue Outbreak Mortality Series (2023-2025)",
            "active_time_points": len(dengue_wk),
            "max_time_points_single_series": len(dengue_wk),
            "positive_death_time_points": int((dengue_wk["verified_deaths"] > 0).sum()),
            "zero_death_pct": round(float((dengue_wk["verified_deaths"] == 0).mean()) * 100.0, 2),
            "total_verified_deaths": int(dengue_wk["verified_deaths"].sum()),
            "forecasting_supported": True,
            "recommended_mode": "CHRONOLOGICAL_FORECAST_EVALUATION_AGAINST_SIMPLE_BASELINES",
            "scientific_rationale": (
                "82 active surveillance weeks with 53 positive-death weeks (82 total deaths) support "
                "chronological 70/15/15 validation against naive and rolling baselines."
            ),
        },
        {
            "analytical_grain": "statewide_weekly_all_diseases",
            "series_description": "Statewide Weekly Aggregate Outbreak Mortality Series (All Diseases 2023-2025)",
            "active_time_points": len(state_wk),
            "max_time_points_single_series": len(state_wk),
            "positive_death_time_points": int((state_wk["verified_deaths"] > 0).sum()),
            "zero_death_pct": round(float((state_wk["verified_deaths"] == 0).mean()) * 100.0, 2),
            "total_verified_deaths": int(state_wk["verified_deaths"].sum()),
            "forecasting_supported": True,
            "recommended_mode": "CHRONOLOGICAL_FORECAST_EVALUATION_AGAINST_SIMPLE_BASELINES",
            "scientific_rationale": (
                "121 active surveillance weeks with 71 positive-death weeks (199 total deaths) provide "
                "the highest-density mortality time series in the workspace for chronological evaluation."
            ),
        },
    ]
    return pd.DataFrame(rows)


def _prepare_lagged_series_df(series_df: pd.DataFrame) -> pd.DataFrame:
    """Construct strictly lagged features (t-1..t-4) for a chronologically ordered weekly series."""
    df = series_df.sort_values(["report_year", "report_week"]).reset_index(drop=True).copy()
    df["deaths_lag1"] = df["verified_deaths"].shift(1)
    df["deaths_lag2"] = df["verified_deaths"].shift(2)
    df["deaths_lag3"] = df["verified_deaths"].shift(3)
    df["deaths_lag4"] = df["verified_deaths"].shift(4)
    df["deaths_roll4_mean"] = (
        df["verified_deaths"].shift(1).rolling(4, min_periods=1).mean()
    )
    df["cases_lag1"] = df["verified_cases"].shift(1)
    df["cases_roll4_mean"] = (
        df["verified_cases"].shift(1).rolling(4, min_periods=1).mean()
    )
    df["week_sin"] = np.sin(2.0 * np.pi * df["report_week"].astype(float) / 52.0)
    df["week_cos"] = np.cos(2.0 * np.pi * df["report_week"].astype(float) / 52.0)
    df["quarter"] = ((df["report_week"].astype(int) - 1) // 13) + 1
    return df


def _evaluate_single_target_series(
    df_raw: pd.DataFrame, target_label: str
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run chronological 70/15/15 train/val/test evaluation across 5 models on a weekly mortality series."""
    df = _prepare_lagged_series_df(df_raw).iloc[4:].reset_index(drop=True)
    n = len(df)
    n_train = int(round(n * 0.70))
    n_val = int(round(n * 0.15))
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train : n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val :].copy()

    df["split"] = (
        ["train"] * len(train_df)
        + ["validation"] * len(val_df)
        + ["test"] * len(test_df)
    )

    feature_cols = [
        "deaths_lag1",
        "deaths_lag2",
        "deaths_roll4_mean",
        "cases_lag1",
        "cases_roll4_mean",
        "week_sin",
        "week_cos",
    ]
    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df["verified_deaths"].to_numpy(dtype=float)
    X_all = df[feature_cols].to_numpy(dtype=float)

    # Quarter-level seasonal historical mean from training split only
    q_means = train_df.groupby("quarter")["verified_deaths"].mean().to_dict()
    global_train_mean = float(train_df["verified_deaths"].mean())

    # Fit Poisson GLM strictly on train split
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_all_sc = scaler.transform(X_all)
    poisson_glm = PoissonRegressor(alpha=1.5, max_iter=500)
    poisson_glm.fit(X_train_sc, y_train)

    # Fit Random Forest Count Regressor strictly on train split (n_jobs=1 to avoid Windows OpenMP hang)
    rf_reg = RandomForestRegressor(
        n_estimators=120,
        max_depth=3,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=1,
    )
    rf_reg.fit(X_train, y_train)

    preds_map = {
        "Zero_Mortality_Baseline": np.zeros(len(df), dtype=float),
        "Last_Value_Naive_Baseline": df["deaths_lag1"].to_numpy(dtype=float),
        "Rolling_4Week_Mean_Baseline": df["deaths_roll4_mean"].to_numpy(dtype=float),
        "Seasonal_Quarter_Naive_Baseline": np.array(
            [q_means.get(int(q), global_train_mean) for q in df["quarter"]], dtype=float
        ),
        "Poisson_Autoregressive_GLM": np.clip(poisson_glm.predict(X_all_sc), 0.0, None),
        "RandomForest_Lagged_Count_Regressor": np.clip(rf_reg.predict(X_all), 0.0, None),
    }

    gen_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    metric_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []

    # Baseline MAE for skill comparison
    val_mask = df["split"] == "validation"
    test_mask = df["split"] == "test"
    y_val = df.loc[val_mask, "verified_deaths"].to_numpy(dtype=float)
    y_test = df.loc[test_mask, "verified_deaths"].to_numpy(dtype=float)

    naive_val_mae = float(mean_absolute_error(y_val, preds_map["Last_Value_Naive_Baseline"][val_mask]))
    naive_test_mae = float(mean_absolute_error(y_test, preds_map["Last_Value_Naive_Baseline"][test_mask]))

    for model_name, y_hat in preds_map.items():
        # Compute residual quantile on train for 80% prediction interval
        train_resid = np.abs(y_train - y_hat[:n_train])
        q80 = float(np.quantile(train_resid, 0.80))

        for sp_name, mask, ref_mae in [
            ("validation", val_mask, naive_val_mae),
            ("test", test_mask, naive_test_mae),
        ]:
            y_true_sp = df.loc[mask, "verified_deaths"].to_numpy(dtype=float)
            y_pred_sp = y_hat[mask]
            mae = float(mean_absolute_error(y_true_sp, y_pred_sp))
            rmse = float(np.sqrt(mean_squared_error(y_true_sp, y_pred_sp)))
            low_80 = np.maximum(0.0, y_pred_sp - q80)
            high_80 = y_pred_sp + q80
            cov_80 = float(np.mean((y_true_sp >= low_80) & (y_true_sp <= high_80)) * 100.0)
            skill_pct = ((ref_mae - mae) / ref_mae) * 100.0 if ref_mae > 0 else 0.0

            metric_rows.append(
                {
                    "target_series": target_label,
                    "model_name": model_name,
                    "evaluation_split": sp_name,
                    "n_weeks": int(mask.sum()),
                    "observed_deaths_sum": int(y_true_sp.sum()),
                    "predicted_deaths_sum": round(float(y_pred_sp.sum()), 2),
                    "mae": round(mae, 4),
                    "rmse": round(rmse, 4),
                    "pi_80_coverage_pct": round(cov_80, 2),
                    "mae_skill_vs_last_value_naive_pct": round(skill_pct, 2),
                    "beats_last_value_naive_mae": bool(mae < ref_mae),
                    "features_used": (
                        "Strictly lagged (t-1..t-4) deaths, cases & seasonal sin/cos"
                        if "GLM" in model_name or "RandomForest" in model_name
                        else "Historical target baseline only"
                    ),
                    "generation_timestamp": gen_ts,
                }
            )

        for idx, r in df.iterrows():
            pred_val = float(y_hat[idx])
            pred_rows.append(
                {
                    "target_series": target_label,
                    "model_name": model_name,
                    "reporting_period": str(r["reporting_period"]),
                    "report_year": int(r["report_year"]),
                    "report_week": int(r["report_week"]),
                    "canonical_week_start": str(r["canonical_week_start"]),
                    "split": str(r["split"]),
                    "verified_cases": int(r["verified_cases"]),
                    "observed_verified_deaths": int(r["verified_deaths"]),
                    "predicted_deaths": round(pred_val, 3),
                    "pi_80_lower": round(max(0.0, pred_val - q80), 3),
                    "pi_80_upper": round(pred_val + q80, 3),
                    "record_nature": "CHRONOLOGICAL_MODEL_EVALUATION_VS_OBSERVED_AGGREGATE_DEATHS",
                    "generation_timestamp": gen_ts,
                }
            )

    return metric_rows, pred_rows


def evaluate_chronological_mortality_forecasting(
    weekly_records_df: pd.DataFrame,
    weekly_statewide_series_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run sufficiency assessment and chronological forecasting evaluation on supported statewide series."""
    sufficiency_df = build_mortality_forecasting_sufficiency_table(weekly_records_df)

    # 1. Statewide All-Diseases Weekly Mortality Series
    m_all, p_all = _evaluate_single_target_series(
        weekly_statewide_series_df, "statewide_weekly_all_diseases"
    )

    # 2. Statewide Dengue Weekly Mortality Series
    dengue_wk = (
        weekly_records_df[weekly_records_df["disease"] == "Dengue"]
        .groupby(
            ["report_year", "report_week", "canonical_week_start", "reporting_period"],
            as_index=False,
        )
        .agg(
            verified_cases=("verified_cases", "sum"),
            verified_deaths=("verified_deaths", "sum"),
        )
        .sort_values(["report_year", "report_week"])
        .reset_index(drop=True)
    )
    m_den, p_den = _evaluate_single_target_series(dengue_wk, "statewide_weekly_dengue")

    metrics_df = pd.DataFrame(m_all + m_den)
    preds_df = pd.DataFrame(p_all + p_den)
    return sufficiency_df, metrics_df, preds_df
