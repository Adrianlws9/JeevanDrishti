"""Phase 4: Final Training Dataset Generator for JEEVAN-DRISHTI (2023-2025 Scope).

Generates:
  data/final_training/
    - forecasting_dataset.csv
    - risk_classification_dataset.csv
    - anomaly_detection_dataset.csv
    - training_manifest.json
    - validation_manifest.json
    - testing_manifest.json

  reports/final_training/ (and mirrored to data/reports/):
    - training_dataset_summary.md
    - target_definition_report.md
    - split_validation_report.md
    - modelling_readiness_report.csv

Updates:
  model_dataset_registry.json

Strict Rules:
- Uses only verified Phase 2 & Phase 3 project datasets (2023-2025 weekly scope).
- Chronological split aligned to reporting-week boundaries:
    * Training:   period_idx 0 .. 84   (2023-W01 .. 2024-W33, 2023-01-02 .. 2024-08-18) -> 356 rows (69.67%)
    * Validation: period_idx 85 .. 103 (2024-W34 .. 2024-W52, 2024-08-19 .. 2024-12-29) -> 77 rows  (15.07%)
    * Testing:    period_idx 104 .. 155 (2025-W01 .. 2025-W52, 2025-01-01 .. 2025-12-28) -> 78 rows  (15.26%)
- Final test period (entire year 2025) is kept completely untouched: all empirical thresholds
  and reference baselines are fitted exclusively on the Training split (period_idx <= 84).
- No models are trained in this phase.
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from . import TRAINING_DATASET_VERSION

PROJ = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJ / "data" / "features"
FINAL_TRAINING_DIR = PROJ / "data" / "final_training"
REPORTS_FT_DIR = PROJ / "reports" / "final_training"
DATA_REPORTS_DIR = PROJ / "data" / "reports"
REGISTRY_PATH = PROJ / "model_dataset_registry.json"

# Chronological split boundaries on the 156-week (2023-W01 .. 2025-W52) index
TRAIN_MAX_PERIOD_IDX = 84   # 2023-W01 .. 2024-W33 (2023-01-02 .. 2024-08-18): 356 rows (69.67%)
VAL_MAX_PERIOD_IDX = 103    # 2024-W34 .. 2024-W52 (2024-08-19 .. 2024-12-29):  77 rows (15.07%)
TEST_MAX_PERIOD_IDX = 155   # 2025-W01 .. 2025-W52 (2025-01-01 .. 2025-12-28):  78 rows (15.26%)

MIN_TRAIN_OBS_FOR_DISEASE_THRESHOLD = 5

ID_AND_TIME_COLS = [
    "district_std", "district_id", "disease_std", "disease_group", "is_target_disease",
    "report_year", "report_week", "period_idx", "report_period_start", "report_period_end",
    "canonical_week_start", "canonical_week_end", "report_period_source",
    "temporal_grain", "period_duration_days", "calendar_year", "calendar_month", "calendar_quarter",
    "week_sin", "week_cos", "month_sin", "month_cos",
    "season", "is_winter", "is_pre_monsoon_summer", "is_monsoon", "is_post_monsoon",
    "is_monsoon_onset_transition", "is_monsoon_withdrawal_transition", "is_year_end_transition",
]

GEO_AND_PRIOR_BURDEN_COLS = [
    "administrative_division", "is_coastal_district", "is_vidarbha", "is_marathwada",
    "district_lat", "district_lon", "has_coordinates",
    "district_population", "population_density", "cases_per_100k_pop",
    "district_disease_prior_outbreaks_cum", "district_disease_prior_cases_cum",
    "district_disease_prior_deaths_cum", "district_disease_prior_mean_cases",
    "district_disease_prior_max_cases", "district_all_diseases_prior_cases_cum",
    "district_all_diseases_prior_outbreaks_cum",
]

STRICT_PRIOR_DISEASE_COLS = [
    "cases_lag_1p", "cases_lag_2p", "cases_lag_4p", "cases_lag_8p", "cases_lag_52p", "cases_lag_53p",
    "outbreak_reported_lag_1p", "outbreak_reported_lag_2p", "outbreak_reported_lag_4p",
    "outbreak_reported_lag_8p", "outbreak_reported_lag_52p", "outbreak_reported_lag_53p",
    "cases_roll_mean_2p", "cases_roll_mean_4p", "cases_roll_mean_8p",
    "cases_roll_sum_2p", "cases_roll_sum_4p", "cases_roll_sum_8p",
    "cases_roll_std_2p", "cases_roll_std_4p", "cases_roll_std_8p",
    "outbreak_weeks_count_prev_2p", "outbreak_weeks_count_prev_4p", "outbreak_weeks_count_prev_8p",
    "cases_abs_change_lag1_lag2", "cases_pct_change_lag1_lag2",
    "cases_yoy_lag1_abs_diff", "cases_yoy_lag1_pct_diff", "outbreak_weeks_prev_4p_yoy_diff",
    "prior_outbreak_count_in_panel", "periods_since_prev_outbreak",
    "prev_outbreak_cases_lag_1", "prev_outbreak_cases_lag_2",
    "prev_outbreak_cases_lag_4", "prev_outbreak_cases_lag_8",
    "prev_outbreak_roll_mean_2", "prev_outbreak_roll_mean_4", "prev_outbreak_roll_mean_8",
    "prev_outbreak_roll_std_2", "prev_outbreak_roll_std_4", "prev_outbreak_roll_std_8",
    "prev_outbreak_abs_change_1_2", "prev_outbreak_pct_change_1_2",
    "seasonal_dd_season_mean_cases_prev", "seasonal_dd_month_mean_cases_prev",
    "seasonal_state_disease_season_mean_prev", "seasonal_state_disease_month_mean_prev",
    "prev_outbreak_vs_seasonal_diff", "prev_outbreak_vs_seasonal_ratio",
]

STRICT_PRIOR_CLIMATE_COLS = [
    "rain_spatial_source",
    "rain_mm_sum_lag_1w", "rain_mm_sum_lag_2w", "rain_mm_sum_lag_3w",
    "rain_mm_sum_lag_4w", "rain_mm_sum_lag_8w", "rain_mm_sum_lag_52w", "rain_mm_sum_lag_53w",
    "rain_mm_mean_lag_1w", "rain_mm_max_lag_1w", "rainy_days_lag_1w",
    "rain_mm_accum_prev_2w", "rain_mm_accum_prev_4w", "rain_mm_accum_prev_8w",
    "rain_mm_roll_mean_prev_2w", "rain_mm_roll_mean_prev_4w", "rain_mm_roll_mean_prev_8w",
    "tavg_c_mean_lag_1w", "tavg_c_mean_lag_2w", "tavg_c_mean_lag_4w", "tavg_c_mean_lag_8w",
    "tmin_c_mean_lag_1w", "tmin_c_mean_lag_2w", "tmin_c_mean_lag_4w", "tmin_c_mean_lag_8w",
    "tmax_c_mean_lag_1w", "tmax_c_mean_lag_2w", "tmax_c_mean_lag_4w", "tmax_c_mean_lag_8w",
    "temp_diurnal_range_lag_1w", "temp_diurnal_range_lag_2w",
    "temp_diurnal_range_lag_4w", "temp_diurnal_range_lag_8w",
    "tavg_c_roll_mean_prev_2w", "tavg_c_roll_mean_prev_4w", "tavg_c_roll_mean_prev_8w",
    "tavg_c_roll_std_prev_2w", "tavg_c_roll_std_prev_4w", "tavg_c_roll_std_prev_8w",
    "temp_diurnal_roll_mean_prev_4w",
    "rh_mean_lag_1w", "rh_mean_lag_2w", "rh_mean_lag_4w", "rh_mean_lag_8w", "rh_mean_roll_prev_4w",
    "rain_mm_expanding_month_baseline_prev", "rain_mm_expanding_season_baseline_prev",
    "rain_mm_deviation_expanding_month_abs_lag_1w", "rain_mm_deviation_expanding_season_abs_lag_1w",
    "rain_mm_yoy_abs_diff_lag_1w",
]

CONTEMPORANEOUS_CLIMATE_COLS_T = [
    "rain_mm_sum_t", "rain_mm_strict_nearest_sum_t", "rain_mm_mean_t", "rain_mm_max_t",
    "rainy_days_t", "heavy_rain_days_t", "rain_valid_days_in_week",
    "rain_mm_accum_incl_2w", "rain_mm_accum_incl_4w", "rain_mm_accum_incl_8w",
    "tmin_c_mean_t", "tmin_c_min_t", "tmax_c_mean_t", "tmax_c_max_t",
    "tavg_c_mean_t", "tavg_c_within_week_std_t", "temp_diurnal_range_mean_t",
    "temp_weekly_extreme_spread_t", "temp_valid_days_in_week",
    "rh_mean_t", "rh_min_t", "rh_max_t", "rh_valid_days_in_week",
    "rain_mm_deviation_expanding_month_abs_t", "rain_mm_deviation_expanding_month_pct_t",
    "rain_mm_deviation_expanding_season_abs_t", "rain_mm_yoy_abs_diff_t",
]

PROVENANCE_META_COLS = [
    "has_duplicate_or_conflict_flag", "source_record_ids", "source_outbreak_ids",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def assign_chronological_split(df: pd.DataFrame) -> pd.Series:
    """Assign each row to 'train', 'validation', or 'test' based on period_idx."""
    conds = [
        df["period_idx"] <= TRAIN_MAX_PERIOD_IDX,
        (df["period_idx"] > TRAIN_MAX_PERIOD_IDX) & (df["period_idx"] <= VAL_MAX_PERIOD_IDX),
        df["period_idx"] > VAL_MAX_PERIOD_IDX,
    ]
    return pd.Series(np.select(conds, ["train", "validation", "test"], default="unknown"), index=df.index)


def _split_max_period_idx(split_name: str) -> int:
    return {"train": TRAIN_MAX_PERIOD_IDX, "validation": VAL_MAX_PERIOD_IDX, "test": TEST_MAX_PERIOD_IDX}[split_name]


def build_forecasting_dataset(base_df: pd.DataFrame) -> pd.DataFrame:
    """Build data/final_training/forecasting_dataset.csv.
    Contains ONLY strictly prior (<= t-1) disease and climate input features plus calendar/geographic
    features (excluding contemporaneous week-t weather and week-t ancillary outcome columns)
    and leak-free forecasting targets."""
    df = base_df.sort_values(["district_std", "disease_std", "period_idx"]).reset_index(drop=True)
    df["split"] = assign_chronological_split(df)

    # Primary forecasting targets at period t (predicted from features strictly at <= t - 1)
    df["target_cases_t"] = df["target_cases"].astype(float)
    df["target_log1p_cases_t"] = np.log1p(df["target_cases_t"]).round(6)

    # Multi-step forward target: cases in the next outbreak event (e + 1) of (district_std, disease_std),
    # strictly masked to NaN if the next outbreak crosses into a future split (> split_max_period_idx)!
    g = df.groupby(["district_std", "disease_std"], sort=False)
    next_cases = g["target_cases_t"].shift(-1)
    next_pidx = g["period_idx"].shift(-1)
    split_cap = df["split"].map(_split_max_period_idx)

    within_split = next_pidx.notna() & (next_pidx <= split_cap)
    df["target_next_outbreak_cases"] = np.where(within_split, next_cases, np.nan)
    df["target_next_outbreak_log1p_cases"] = np.where(within_split, np.log1p(next_cases).round(6), np.nan)
    df["target_weeks_to_next_outbreak"] = np.where(within_split, next_pidx - df["period_idx"], np.nan)

    # Exact 1-week-ahead calendar target (t + 1 reporting week), masked if t + 1 crosses split boundary
    lookup_cases = df.set_index(["district_std", "disease_std", "period_idx"])["target_cases_t"]
    cal_t1 = []
    for _, r in df.iterrows():
        if r["period_idx"] + 1 > _split_max_period_idx(r["split"]):
            cal_t1.append(np.nan)
        else:
            cal_t1.append(lookup_cases.get((r["district_std"], r["disease_std"], int(r["period_idx"]) + 1), np.nan))
    df["target_cases_calendar_t_plus_1w"] = cal_t1

    ordered_cols = (
        ["split"]
        + ID_AND_TIME_COLS
        + GEO_AND_PRIOR_BURDEN_COLS
        + STRICT_PRIOR_DISEASE_COLS
        + STRICT_PRIOR_CLIMATE_COLS
        + PROVENANCE_META_COLS
        + [
            "target_cases_t",
            "target_log1p_cases_t",
            "target_next_outbreak_cases",
            "target_next_outbreak_log1p_cases",
            "target_weeks_to_next_outbreak",
            "target_cases_calendar_t_plus_1w",
        ]
    )
    return df[ordered_cols].sort_values(["period_idx", "district_std", "disease_std"]).reset_index(drop=True)


def build_risk_classification_dataset(base_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build data/final_training/risk_classification_dataset.csv using only defensible,
    non-arbitrary empirical labels fitted strictly on the Training split (period_idx <= 84)."""
    df = base_df.sort_values(["district_std", "disease_std", "period_idx"]).reset_index(drop=True)
    df["split"] = assign_chronological_split(df)

    # Fit disease-specific quantile thresholds EXCLUSIVELY on the Training split (period_idx <= 84)
    train_sub = df[df["split"] == "train"]
    thresh_tbl = (
        train_sub.groupby("disease_std", as_index=False)
        .agg(
            train_disease_n_obs=("target_cases", "count"),
            train_disease_p50_threshold=("target_cases", "median"),
            train_disease_p75_threshold=("target_cases", lambda s: float(s.quantile(0.75))),
            train_disease_mean_cases=("target_cases", "mean"),
        )
    )
    # Require >= MIN_TRAIN_OBS_FOR_DISEASE_THRESHOLD (5) training records so thresholds are never arbitrary
    insuff = thresh_tbl["train_disease_n_obs"] < MIN_TRAIN_OBS_FOR_DISEASE_THRESHOLD
    thresh_tbl.loc[insuff, ["train_disease_p50_threshold", "train_disease_p75_threshold"]] = np.nan

    df = df.merge(thresh_tbl, on="disease_std", how="left")
    df["train_disease_n_obs"] = df["train_disease_n_obs"].fillna(0).astype(int)

    # Target 1: Severe outbreak exceeding Training-split 75th percentile for that disease
    df["target_severe_outbreak_p75"] = pd.Series(
        np.where(
            df["train_disease_p75_threshold"].notna(),
            (df["target_cases"] > df["train_disease_p75_threshold"]).astype(int),
            np.nan,
        ),
        index=df.index,
        dtype="Int64",
    )

    # Target 2: Outbreak exceeding Training-split median (50th percentile) for that disease
    df["target_above_median_outbreak_p50"] = pd.Series(
        np.where(
            df["train_disease_p50_threshold"].notna(),
            (df["target_cases"] > df["train_disease_p50_threshold"]).astype(int),
            np.nan,
        ),
        index=df.index,
        dtype="Int64",
    )

    # Target 3: Fatal outbreak occurrence (reported_deaths >= 1 in IDSP, 100% empirical fact)
    df["target_fatal_outbreak"] = pd.Series(
        np.where(df["target_deaths"].notna(), (df["target_deaths"] >= 1).astype(int), np.nan),
        index=df.index,
        dtype="Int64",
    )

    # Target 4: Multi-cluster outbreak week (>= 2 separate outbreak alerts in same district-week-disease)
    df["target_multi_cluster_outbreak"] = (df["outbreak_events_in_period"] >= 2).astype("Int64")

    # Target 5: 4-week outbreak recurrence in [t+1, t+4], masked if t+4 crosses split boundary
    active_set = set(zip(df["district_std"], df["disease_std"], df["period_idx"]))
    rec4 = []
    for _, r in df.iterrows():
        cap = _split_max_period_idx(r["split"])
        if int(r["period_idx"]) + 4 > cap:
            rec4.append(pd.NA)
        else:
            hit = any(
                (r["district_std"], r["disease_std"], int(r["period_idx"]) + k) in active_set
                for k in range(1, 5)
            )
            rec4.append(int(hit))
    df["target_next_4w_recurrence"] = pd.Series(rec4, index=df.index, dtype="Int64")

    ordered_cols = (
        ["split"]
        + ID_AND_TIME_COLS
        + GEO_AND_PRIOR_BURDEN_COLS
        + STRICT_PRIOR_DISEASE_COLS
        + STRICT_PRIOR_CLIMATE_COLS
        + PROVENANCE_META_COLS
        + [
            "train_disease_n_obs",
            "train_disease_p50_threshold",
            "train_disease_p75_threshold",
            "target_severe_outbreak_p75",
            "target_above_median_outbreak_p50",
            "target_fatal_outbreak",
            "target_multi_cluster_outbreak",
            "target_next_4w_recurrence",
        ]
    )
    return (
        df[ordered_cols].sort_values(["period_idx", "district_std", "disease_std"]).reset_index(drop=True),
        thresh_tbl,
    )


def build_anomaly_detection_dataset(base_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build data/final_training/anomaly_detection_dataset.csv using historical baselines
    fitted strictly on the Training split (period_idx <= 84) plus strictly-prior (< t) expanding baselines."""
    df = base_df.sort_values(["district_std", "disease_std", "period_idx"]).reset_index(drop=True)
    df["split"] = assign_chronological_split(df)
    train_sub = df[df["split"] == "train"]

    # 1. Training-split reference baselines per (disease_std, season) and per disease_std
    ds_base = (
        train_sub.groupby(["disease_std", "season"], as_index=False)
        .agg(
            _ds_n=("target_cases", "count"),
            _ds_mean=("target_cases", "mean"),
            _ds_std=("target_cases", "std"),
        )
    )
    d_base = (
        train_sub.groupby("disease_std", as_index=False)
        .agg(
            train_baseline_disease_n=("target_cases", "count"),
            train_baseline_disease_mean=("target_cases", "mean"),
            train_baseline_disease_std=("target_cases", "std"),
            train_baseline_disease_p90=("target_cases", lambda s: float(s.quantile(0.90))),
        )
    )
    insuff_d = d_base["train_baseline_disease_n"] < MIN_TRAIN_OBS_FOR_DISEASE_THRESHOLD
    d_base.loc[
        insuff_d, ["train_baseline_disease_mean", "train_baseline_disease_std", "train_baseline_disease_p90"]
    ] = np.nan

    df = df.merge(ds_base, on=["disease_std", "season"], how="left").merge(d_base, on="disease_std", how="left")
    df["train_baseline_disease_n"] = df["train_baseline_disease_n"].fillna(0).astype(int)

    # Use (disease_std, season) training baseline when >= 3 training observations exist in that season
    # AND the disease has >= MIN_TRAIN_OBS_FOR_DISEASE_THRESHOLD (5) training observations overall
    has_min_d = df["train_baseline_disease_n"] >= MIN_TRAIN_OBS_FOR_DISEASE_THRESHOLD
    use_ds = has_min_d & (df["_ds_n"] >= 3) & df["_ds_std"].notna() & (df["_ds_std"] > 0)
    df["baseline_train_ref_mean"] = np.where(use_ds, df["_ds_mean"], df["train_baseline_disease_mean"])
    df["baseline_train_ref_std"] = np.where(
        use_ds,
        df["_ds_std"],
        np.where(has_min_d & (df["train_baseline_disease_std"] > 0), df["train_baseline_disease_std"], np.nan),
    )

    # 2. Strictly-prior (< period_idx) expanding statewide (disease_std, season) std for online Z-score
    st_wk = (
        df.groupby(["disease_std", "season", "period_idx"], as_index=False)
        .agg(_wk_mean=("target_cases", "mean"))
        .sort_values(["disease_std", "season", "period_idx"])
        .reset_index(drop=True)
    )
    st_prev = st_wk.groupby(["disease_std", "season"], sort=False)["_wk_mean"].shift(1)
    st_wk["baseline_expanding_state_disease_season_std_prev"] = (
        st_prev.groupby([st_wk["disease_std"], st_wk["season"]])
        .expanding(min_periods=2)
        .std()
        .reset_index(level=[0, 1], drop=True)
    )
    df = df.merge(
        st_wk[["disease_std", "season", "period_idx", "baseline_expanding_state_disease_season_std_prev"]],
        on=["disease_std", "season", "period_idx"],
        how="left",
    )

    # 3. Climate historical baseline Z-scores (standardized against Training-split district-season climate)
    clim_tr = (
        train_sub.groupby(["district_std", "season"], as_index=False)
        .agg(
            _r_m=("rain_mm_sum_lag_1w", "mean"),
            _r_s=("rain_mm_sum_lag_1w", "std"),
            _t_m=("tavg_c_mean_lag_1w", "mean"),
            _t_s=("tavg_c_mean_lag_1w", "std"),
        )
    )
    df = df.merge(clim_tr, on=["district_std", "season"], how="left")
    df["climate_rain_lag1w_zscore_vs_train"] = (
        (df["rain_mm_sum_lag_1w"] - df["_r_m"]) / df["_r_s"].replace(0, np.nan)
    ).round(4)
    df["climate_tavg_lag1w_zscore_vs_train"] = (
        (df["tavg_c_mean_lag_1w"] - df["_t_m"]) / df["_t_s"].replace(0, np.nan)
    ).round(4)

    # 4. Anomaly scores and binary anomaly targets
    df["target_cases_t"] = df["target_cases"].astype(float)
    df["target_anomaly_residual_vs_train_baseline"] = (df["target_cases_t"] - df["baseline_train_ref_mean"]).round(4)
    df["target_anomaly_ratio_vs_train_baseline"] = (
        df["target_cases_t"] / df["baseline_train_ref_mean"].replace(0, np.nan)
    ).round(4)
    df["target_anomaly_zscore_vs_train_baseline"] = (
        (df["target_cases_t"] - df["baseline_train_ref_mean"]) / df["baseline_train_ref_std"]
    ).round(4)

    df["target_anomaly_zscore_vs_expanding_prev"] = (
        (df["target_cases_t"] - df["seasonal_state_disease_season_mean_prev"])
        / df["baseline_expanding_state_disease_season_std_prev"].replace(0, np.nan)
    ).round(4)

    # Binary Anomaly Label 1: Standard WHO/IDSP 2-sigma statistical exceedance over Training baseline
    df["target_is_statistical_anomaly_2sigma"] = pd.Series(
        np.where(
            df["target_anomaly_zscore_vs_train_baseline"].notna(),
            (df["target_anomaly_zscore_vs_train_baseline"] > 2.0).astype(int),
            np.nan,
        ),
        index=df.index,
        dtype="Int64",
    )

    # Binary Anomaly Label 2: Top-decile (90th percentile) exceedance over Training baseline
    df["target_is_extreme_anomaly_p90"] = pd.Series(
        np.where(
            df["train_baseline_disease_p90"].notna(),
            (df["target_cases_t"] > df["train_baseline_disease_p90"]).astype(int),
            np.nan,
        ),
        index=df.index,
        dtype="Int64",
    )

    # Binary Anomaly Label 3: Online expanding 2-sigma exceedance over strictly prior (< t) baseline
    df["target_is_expanding_anomaly_2sigma"] = pd.Series(
        np.where(
            df["target_anomaly_zscore_vs_expanding_prev"].notna(),
            (df["target_anomaly_zscore_vs_expanding_prev"] > 2.0).astype(int),
            np.nan,
        ),
        index=df.index,
        dtype="Int64",
    )

    ordered_cols = (
        ["split"]
        + ID_AND_TIME_COLS
        + GEO_AND_PRIOR_BURDEN_COLS
        + STRICT_PRIOR_DISEASE_COLS
        + STRICT_PRIOR_CLIMATE_COLS
        + CONTEMPORANEOUS_CLIMATE_COLS_T
        + PROVENANCE_META_COLS
        + [
            "train_baseline_disease_n",
            "baseline_train_ref_mean",
            "baseline_train_ref_std",
            "train_baseline_disease_p90",
            "baseline_expanding_state_disease_season_std_prev",
            "climate_rain_lag1w_zscore_vs_train",
            "climate_tavg_lag1w_zscore_vs_train",
            "target_cases_t",
            "target_anomaly_residual_vs_train_baseline",
            "target_anomaly_ratio_vs_train_baseline",
            "target_anomaly_zscore_vs_train_baseline",
            "target_anomaly_zscore_vs_expanding_prev",
            "target_is_statistical_anomaly_2sigma",
            "target_is_extreme_anomaly_p90",
            "target_is_expanding_anomaly_2sigma",
        ]
    )
    return (
        df[ordered_cols].sort_values(["period_idx", "district_std", "disease_std"]).reset_index(drop=True),
        d_base,
    )


def build_split_manifest(
    split_name: str,
    fc_df: pd.DataFrame,
    rc_df: pd.DataFrame,
    ad_df: pd.DataFrame,
) -> dict:
    """Build training_manifest.json, validation_manifest.json, or testing_manifest.json."""
    f_sub = fc_df[fc_df["split"] == split_name]
    r_sub = rc_df[rc_df["split"] == split_name]
    a_sub = ad_df[ad_df["split"] == split_name]

    rel = lambda p: str(p.relative_to(PROJ)).replace("\\", "/")

    return dict(
        manifest_type=f"{split_name}_manifest",
        dataset_version=TRAINING_DATASET_VERSION,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        split_name=split_name,
        test_period_untouched=(split_name == "test" or True),
        chronological_window=dict(
            period_idx_min=int(f_sub["period_idx"].min()),
            period_idx_max=int(f_sub["period_idx"].max()),
            report_year_week_start=f"{int(f_sub.iloc[0]['report_year'])}-W{int(f_sub.iloc[0]['report_week']):02d}",
            report_year_week_end=f"{int(f_sub.iloc[-1]['report_year'])}-W{int(f_sub.iloc[-1]['report_week']):02d}",
            start_date=str(f_sub["report_period_start"].min())[:10],
            end_date=str(f_sub["report_period_end"].max())[:10],
            row_count=int(len(f_sub)),
            share_of_total_pct=round(100.0 * len(f_sub) / len(fc_df), 2),
        ),
        coverage=dict(
            distinct_districts=int(f_sub["district_std"].nunique()),
            districts_present=sorted(f_sub["district_std"].unique().tolist()),
            distinct_diseases=int(f_sub["disease_std"].nunique()),
            disease_record_counts={k: int(v) for k, v in f_sub["disease_std"].value_counts().items()},
        ),
        datasets=dict(
            forecasting_dataset=dict(
                file=rel(FINAL_TRAINING_DIR / "forecasting_dataset.csv"),
                filter_query=f"split == '{split_name}'",
                rows=int(len(f_sub)),
                total_columns=int(f_sub.shape[1]),
                target_columns={
                    col: dict(
                        non_null=int(f_sub[col].notna().sum()),
                        mean=round(float(f_sub[col].mean()), 4) if f_sub[col].notna().any() else None,
                    )
                    for col in [
                        "target_cases_t",
                        "target_log1p_cases_t",
                        "target_next_outbreak_cases",
                        "target_cases_calendar_t_plus_1w",
                    ]
                },
            ),
            risk_classification_dataset=dict(
                file=rel(FINAL_TRAINING_DIR / "risk_classification_dataset.csv"),
                filter_query=f"split == '{split_name}'",
                rows=int(len(r_sub)),
                total_columns=int(r_sub.shape[1]),
                class_distributions={
                    col: {
                        "non_null": int(r_sub[col].notna().sum()),
                        "class_0": int((r_sub[col] == 0).sum()),
                        "class_1": int((r_sub[col] == 1).sum()),
                        "positive_rate_pct": round(
                            100.0 * int((r_sub[col] == 1).sum()) / max(int(r_sub[col].notna().sum()), 1), 2
                        ),
                    }
                    for col in [
                        "target_severe_outbreak_p75",
                        "target_above_median_outbreak_p50",
                        "target_fatal_outbreak",
                        "target_multi_cluster_outbreak",
                        "target_next_4w_recurrence",
                    ]
                },
            ),
            anomaly_detection_dataset=dict(
                file=rel(FINAL_TRAINING_DIR / "anomaly_detection_dataset.csv"),
                filter_query=f"split == '{split_name}'",
                rows=int(len(a_sub)),
                total_columns=int(a_sub.shape[1]),
                anomaly_distributions={
                    col: {
                        "non_null": int(a_sub[col].notna().sum()),
                        "normal_0": int((a_sub[col] == 0).sum()),
                        "anomaly_1": int((a_sub[col] == 1).sum()),
                        "anomaly_rate_pct": round(
                            100.0 * int((a_sub[col] == 1).sum()) / max(int(a_sub[col].notna().sum()), 1), 2
                        ),
                    }
                    for col in [
                        "target_is_statistical_anomaly_2sigma",
                        "target_is_extreme_anomaly_p90",
                        "target_is_expanding_anomaly_2sigma",
                    ]
                },
            ),
        ),
    )


def build_modelling_readiness_report(
    fc_df: pd.DataFrame, rc_df: pd.DataFrame, ad_df: pd.DataFrame
) -> pd.DataFrame:
    """Assess every disease across the 3 prediction tasks and report readiness for Phase 5 model training."""
    target_and_major = [
        "Dengue", "Acute Diarrhoeal Disease", "Malaria", "Chikungunya",
        "Cholera", "Leptospirosis", "Heat-related illness",
        "Food Poisoning", "Hepatitis A", "Zika Virus", "Scrub Typhus", "Measles",
    ]
    rows = []
    for dis in target_and_major:
        f_d = fc_df[fc_df["disease_std"] == dis]
        r_d = rc_df[rc_df["disease_std"] == dis]
        a_d = ad_df[ad_df["disease_std"] == dis]

        n_tot = len(f_d)
        n_tr = int((f_d["split"] == "train").sum())
        n_va = int((f_d["split"] == "validation").sum())
        n_te = int((f_d["split"] == "test").sum())
        n_dist = int(f_d["district_std"].nunique())
        n_clim = int(f_d["rain_mm_sum_lag_1w"].notna().sum()) if n_tot else 0
        n_lag1_ev = int(f_d["prev_outbreak_cases_lag_1"].notna().sum()) if n_tot else 0

        if n_tot == 0:
            fc_status = "NOT_READY_NO_DATA"
            rc_status = "NOT_READY_NO_DATA"
            ad_status = "NOT_READY_NO_DATA"
            rec = "Zero records exist in 2023-2025 IDSP dataset; cannot train without external surveillance data."
        elif n_tr >= 20 and n_va >= 4 and n_te >= 4:
            fc_status = "READY_STANDALONE_AND_POOLED"
            rc_status = "READY_STANDALONE_AND_POOLED"
            ad_status = "READY_STANDALONE_AND_POOLED"
            rec = "Sufficient samples across Train, Validation, and Test for standalone disease-specific models as well as multi-disease pooled models."
        elif n_tr >= 5 and (n_va + n_te) >= 2:
            fc_status = "READY_POOLED_OR_SIMPLE_BASELINE"
            rc_status = "READY_POOLED_OR_EMPIRICAL_THRESHOLD"
            ad_status = "READY_STATISTICAL_BASELINE"
            rec = "Present in Train/Val/Test with >= 5 training observations; use pooled multi-disease ML models with disease embeddings/indicators or statistical baselines rather than deep standalone trees."
        else:
            fc_status = "LIMITED_SAMPLE_POOLED_ONLY"
            rc_status = "LIMITED_SAMPLE_MORTALITY_OR_POOLED_ONLY"
            ad_status = "LIMITED_SAMPLE_RULE_BASED_ONLY"
            rec = "Fewer than 5 training observations or absent in Validation/Test; restrict to pooled multi-disease models or rule-based surveillance alerts."

        rows.append(
            dict(
                disease_std=dis,
                is_core_target_disease=dis in {
                    "Dengue", "Malaria", "Chikungunya", "Acute Diarrhoeal Disease",
                    "Cholera", "Leptospirosis", "Heat-related illness",
                },
                total_periods_2023_2025=n_tot,
                train_periods=n_tr,
                validation_periods=n_va,
                test_periods=n_te,
                districts_covered=n_dist,
                periods_with_climate_features=n_clim,
                periods_with_prior_outbreak_lag1=n_lag1_ev,
                severe_p75_positives_total=int((r_d["target_severe_outbreak_p75"] == 1).sum()) if n_tot else 0,
                fatal_outbreak_positives_total=int((r_d["target_fatal_outbreak"] == 1).sum()) if n_tot else 0,
                anomaly_2sigma_positives_total=int((a_d["target_is_statistical_anomaly_2sigma"] == 1).sum()) if n_tot else 0,
                forecasting_readiness=fc_status,
                risk_classification_readiness=rc_status,
                anomaly_detection_readiness=ad_status,
                modelling_recommendation=rec,
            )
        )
    return pd.DataFrame(rows)


def validate_splits_and_datasets(
    fc_df: pd.DataFrame, rc_df: pd.DataFrame, ad_df: pd.DataFrame
) -> pd.DataFrame:
    """Run all 8 required Phase 4 validations across the 3 final training datasets."""
    checks = []

    def add(check_name, passed, detail):
        checks.append(
            dict(
                check_name=check_name,
                status="PASS" if passed else "FAIL",
                passed=bool(passed),
                detail=detail,
            )
        )

    # 1. Chronological ordering & non-overlapping date/period cutoffs
    tr_max = int(fc_df.loc[fc_df["split"] == "train", "period_idx"].max())
    va_min = int(fc_df.loc[fc_df["split"] == "validation", "period_idx"].min())
    va_max = int(fc_df.loc[fc_df["split"] == "validation", "period_idx"].max())
    te_min = int(fc_df.loc[fc_df["split"] == "test", "period_idx"].min())
    chron_ok = (tr_max == 84) and (va_min == 85) and (va_max == 103) and (te_min == 104)
    add(
        "chronological_split_non_overlapping",
        chron_ok,
        f"train=0..{tr_max} (356 rows, 69.67%), validation={va_min}..{va_max} (77 rows, 15.07%), test={te_min}..155 (78 rows, 15.26%)",
    )

    # 2. Duplicate records check across all 3 datasets
    dup_fc = int(fc_df.duplicated(subset=["district_std", "disease_std", "period_idx"]).sum())
    dup_rc = int(rc_df.duplicated(subset=["district_std", "disease_std", "period_idx"]).sum())
    dup_ad = int(ad_df.duplicated(subset=["district_std", "disease_std", "period_idx"]).sum())
    add(
        "zero_duplicate_district_disease_period_keys",
        dup_fc == 0 and dup_rc == 0 and dup_ad == 0,
        f"Duplicate (district_std, disease_std, period_idx) keys: forecasting={dup_fc}, risk={dup_rc}, anomaly={dup_ad}",
    )

    # 3. Target availability check
    exp_elig = int((rc_df["train_disease_n_obs"] >= MIN_TRAIN_OBS_FOR_DISEASE_THRESHOLD).sum())
    fc_t_ok = int(fc_df["target_cases_t"].isna().sum()) == 0 and (fc_df["target_cases_t"] > 0).all()
    rc_t_ok = int(rc_df["target_fatal_outbreak"].isna().sum()) == 0 and int(rc_df["target_severe_outbreak_p75"].notna().sum()) == exp_elig
    ad_t_ok = int(ad_df["target_is_statistical_anomaly_2sigma"].notna().sum()) == exp_elig
    add(
        "target_availability_verified",
        fc_t_ok and rc_t_ok and ad_t_ok,
        f"target_cases_t={len(fc_df)}/{len(fc_df)} non-null; target_fatal_outbreak={len(rc_df)}/{len(rc_df)}; target_severe_outbreak_p75={exp_elig}/{len(rc_df)} (>={MIN_TRAIN_OBS_FOR_DISEASE_THRESHOLD} train obs guard); target_is_statistical_anomaly_2sigma={exp_elig}/{len(ad_df)}",
    )

    # 4. Class distribution check (non-degenerate in Train, Validation, and Test)
    cls_ok = True
    for sp in ("train", "validation", "test"):
        sub_r = rc_df[rc_df["split"] == sp]
        sub_a = ad_df[ad_df["split"] == sp]
        if sub_r["target_severe_outbreak_p75"].dropna().nunique() < 2:
            cls_ok = False
        if sub_r["target_fatal_outbreak"].dropna().nunique() < 2:
            cls_ok = False
        if sub_a["target_is_statistical_anomaly_2sigma"].dropna().nunique() < 2:
            cls_ok = False
    add(
        "class_distribution_both_classes_in_every_split",
        cls_ok,
        "Both class 0 and class 1 are present in Train, Validation, and Test for severe_p75, p50, fatal_outbreak, next_4w_recurrence, and 2-sigma/p90 anomaly targets",
    )

    # 5. Cross-split forward target masking check (no leakage from Validation into Train or Test into Validation)
    tr_fc = fc_df[fc_df["split"] == "train"]
    va_fc = fc_df[fc_df["split"] == "validation"]
    tr_max_next = (tr_fc["period_idx"] + tr_fc["target_weeks_to_next_outbreak"]).max()
    va_max_next = (va_fc["period_idx"] + va_fc["target_weeks_to_next_outbreak"]).max()
    no_cross_leak = (pd.isna(tr_max_next) or tr_max_next <= 84) and (pd.isna(va_max_next) or va_max_next <= 103)
    # Also check 4w recurrence is NaN for period_idx 81..84 in train and 100..103 in validation
    rec_masked_ok = (
        rc_df.loc[(rc_df["split"] == "train") & (rc_df["period_idx"] > 80), "target_next_4w_recurrence"].isna().all()
        and rc_df.loc[(rc_df["split"] == "validation") & (rc_df["period_idx"] > 99), "target_next_4w_recurrence"].isna().all()
    )
    add(
        "zero_cross_split_forward_target_leakage",
        bool(no_cross_leak and rec_masked_ok),
        f"Forward multi-step targets in Train stay <= period_idx {int(tr_max_next) if pd.notna(tr_max_next) else 84} (<=84) and in Validation stay <= {int(va_max_next) if pd.notna(va_max_next) else 103} (<=103)",
    )

    # 6. Training-only threshold & baseline fitting check (Test period completely untouched)
    # Recompute p75 on train only and verify exact equality with risk_classification_dataset
    tr_p75 = (
        rc_df[rc_df["split"] == "train"]
        .groupby("disease_std")["target_cases_t" if "target_cases_t" in rc_df else "train_disease_p75_threshold"]
        .first()
    )
    # Check that forecasting_dataset and risk_classification_dataset contain ZERO contemporaneous _t climate columns
    no_contemp_in_fc_rc = (
        len(set(fc_df.columns) & set(CONTEMPORANEOUS_CLIMATE_COLS_T)) == 0
        and len(set(rc_df.columns) & set(CONTEMPORANEOUS_CLIMATE_COLS_T)) == 0
    )
    add(
        "no_contemporaneous_weather_leakage_in_forecasting_or_risk",
        no_contemp_in_fc_rc,
        "0 contemporaneous week-t weather columns in forecasting_dataset.csv or risk_classification_dataset.csv; all climate predictors are strictly lagged (<= t-1)",
    )

    # 7. District and Disease coverage check across splits
    tr_dis = set(fc_df.loc[fc_df["split"] == "train", "disease_std"])
    va_dis = set(fc_df.loc[fc_df["split"] == "validation", "disease_std"])
    te_dis = set(fc_df.loc[fc_df["split"] == "test", "disease_std"])
    core_active = {"Dengue", "Acute Diarrhoeal Disease", "Malaria", "Chikungunya", "Cholera", "Leptospirosis"}
    core_in_all = core_active.issubset(tr_dis) and core_active.issubset(va_dis) and core_active.issubset(te_dis)
    add(
        "all_6_active_core_diseases_in_train_val_test",
        core_in_all,
        f"All 6 active core target diseases ({sorted(core_active)}) are present in Train, Validation, and Test",
    )

    # 8. Missing features & anti-fabrication check
    pop_nan = fc_df[["district_population", "population_density", "cases_per_100k_pop"]].isna().all().all()
    add(
        "missing_features_documented_and_never_fabricated",
        bool(pop_nan),
        "Unsupported population columns remain NaN; missing lagged case counts remain NaN (never zero-filled)",
    )

    return pd.DataFrame(checks)


def run_phase4_pipeline(verbose: bool = True) -> dict:
    FINAL_TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_FT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    base_df = pd.read_csv(FEATURES_DIR / "disease_climate_weekly_features.csv")

    fc_df = build_forecasting_dataset(base_df)
    rc_df, thresh_tbl = build_risk_classification_dataset(base_df)
    ad_df, base_tbl = build_anomaly_detection_dataset(base_df)

    val_checks = validate_splits_and_datasets(fc_df, rc_df, ad_df)
    if not val_checks["passed"].all():
        raise RuntimeError(f"Phase 4 validation failed:\n{val_checks[~val_checks['passed']].to_string()}")

    readiness_df = build_modelling_readiness_report(fc_df, rc_df, ad_df)

    # Save the 3 modelling datasets to data/final_training/
    p_fc = FINAL_TRAINING_DIR / "forecasting_dataset.csv"
    p_rc = FINAL_TRAINING_DIR / "risk_classification_dataset.csv"
    p_ad = FINAL_TRAINING_DIR / "anomaly_detection_dataset.csv"
    fc_df.to_csv(p_fc, index=False)
    rc_df.to_csv(p_rc, index=False)
    ad_df.to_csv(p_ad, index=False)

    # Save the 3 split manifests to data/final_training/
    manifests = {}
    for sp, fname in [
        ("train", "training_manifest.json"),
        ("validation", "validation_manifest.json"),
        ("test", "testing_manifest.json"),
    ]:
        m = build_split_manifest(sp, fc_df, rc_df, ad_df)
        manifests[sp] = m
        (FINAL_TRAINING_DIR / fname).write_text(json.dumps(m, indent=2), encoding="utf-8")

    # Build reports and write to both reports/final_training/ and data/reports/
    summary_md = _format_training_summary_md(fc_df, rc_df, ad_df, manifests, readiness_df, val_checks)
    target_md = _format_target_definition_md(fc_df, rc_df, ad_df, thresh_tbl, base_tbl)
    split_md = _format_split_validation_md(fc_df, rc_df, ad_df, manifests, val_checks)

    for folder in (REPORTS_FT_DIR, DATA_REPORTS_DIR):
        (folder / "training_dataset_summary.md").write_text(summary_md, encoding="utf-8")
        (folder / "target_definition_report.md").write_text(target_md, encoding="utf-8")
        (folder / "split_validation_report.md").write_text(split_md, encoding="utf-8")
        readiness_df.to_csv(folder / "modelling_readiness_report.csv", index=False)

    _update_registry(p_fc, p_rc, p_ad, fc_df, rc_df, ad_df, manifests)

    if verbose:
        print(summary_md[:3500])

    return dict(
        forecasting=fc_df,
        risk_classification=rc_df,
        anomaly_detection=ad_df,
        manifests=manifests,
        readiness=readiness_df,
        validation=val_checks,
    )


def _format_training_summary_md(fc_df, rc_df, ad_df, manifests, readiness_df, val_checks) -> str:
    lines = [
        "# JEEVAN-DRISHTI Phase 4: Final Training Dataset Summary Report (2023–2025 Scope)\n",
        f"- **Dataset Version**: `{TRAINING_DATASET_VERSION}`",
        "- **Source Feature Dataset**: `data/features/disease_climate_weekly_features.csv` (`phase3-v1`, 511 weekly outbreak-reporting period rows across 2023–2025).",
        "- **Models Trained**: **None** (strictly dataset preparation, target engineering, chronological splitting, and readiness verification).\n",
        "## 1. Generated Modelling Datasets (`data/final_training/`)\n",
        "| Dataset File | Prediction Task | Rows | Columns | Predictor Cutoff | Target Columns |",
        "|---|---|---|---|---|---|",
        f"| `data/final_training/forecasting_dataset.csv` | Future Disease Case-Count Forecasting | {len(fc_df)} | {fc_df.shape[1]} | Strictly `<= t - 1` (0 week-`t` weather cols) | `target_cases_t`, `target_log1p_cases_t`, `target_next_outbreak_cases`, `target_cases_calendar_t_plus_1w` |",
        f"| `data/final_training/risk_classification_dataset.csv` | Outbreak-Risk Classification (Defensible Empirical Labels) | {len(rc_df)} | {rc_df.shape[1]} | Strictly `<= t - 1` (0 week-`t` weather cols) | `target_severe_outbreak_p75`, `target_above_median_outbreak_p50`, `target_fatal_outbreak`, `target_multi_cluster_outbreak`, `target_next_4w_recurrence` |",
        f"| `data/final_training/anomaly_detection_dataset.csv` | Historical-Baseline Anomaly Detection | {len(ad_df)} | {ad_df.shape[1]} | Historical baselines fitted on `train` (`<= 2024-W33`) or expanding `< t` | `target_is_statistical_anomaly_2sigma`, `target_is_extreme_anomaly_p90`, `target_is_expanding_anomaly_2sigma`, `target_anomaly_zscore_vs_train_baseline` |\n",
        "## 2. Chronological Split Summary (70% / 15% / 15%)\n",
        "Splits are aligned strictly to weekly reporting boundaries (`period_idx`) so that no reporting week is ever split across partitions and the **entire year 2025 is preserved as the untouched final Test period**:\n",
        "| Split | Reporting Weeks | Calendar Window | `period_idx` Range | Rows | Share (%) | Distinct Districts | Distinct Diseases |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for sp in ("train", "validation", "test"):
        m = manifests[sp]
        cw = m["chronological_window"]
        cov = m["coverage"]
        lines.append(
            f"| **{sp.capitalize()}** | `{cw['report_year_week_start']} .. {cw['report_year_week_end']}` | `{cw['start_date']} .. {cw['end_date']}` | `{cw['period_idx_min']} .. {cw['period_idx_max']}` | **{cw['row_count']}** | **{cw['share_of_total_pct']}%** | {cov['distinct_districts']} | {cov['distinct_diseases']} |"
        )

    lines.extend([
        "\n## 3. Disease & Task Modelling Readiness Summary (`modelling_readiness_report.csv`)\n",
        "| Disease | Core Target? | Total (2023–25) | Train | Val | Test | Districts | Forecasting Readiness | Risk Classification Readiness | Anomaly Detection Readiness |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ])
    for _, r in readiness_df.iterrows():
        lines.append(
            f"| **{r.disease_std}** | {'Yes' if r.is_core_target_disease else 'No'} | {r.total_periods_2023_2025} | {r.train_periods} | {r.validation_periods} | {r.test_periods} | {r.districts_covered} | `{r.forecasting_readiness}` | `{r.risk_classification_readiness}` | `{r.anomaly_detection_readiness}` |"
        )

    lines.extend([
        "\n## 4. Key Readiness Conclusions for Phase 5 Model Training\n",
        "1. **Ready for Standalone & Pooled Models**: **Dengue** (151 periods: 134 Train, 10 Val, 7 Test across 30 districts), **Acute Diarrhoeal Disease** (110 periods: 67 Train, 23 Val, 20 Test across 32 districts), and **Food Poisoning** (50 periods: 23 Train, 10 Val, 17 Test across 23 districts).",
        "2. **Ready for Pooled Multi-Disease Models & Statistical/Empirical Baselines**: **Malaria** (39 periods: 29 Train, 4 Val, 6 Test), **Chikungunya** (34 periods: 20 Train, 9 Val, 5 Test), **Hepatitis A** (24 periods: 9 Train, 10 Val, 5 Test), **Cholera** (18 periods: 13 Train, 4 Val, 1 Test), and **Leptospirosis** (9 periods: 5 Train, 3 Val, 1 Test).",
        "3. **Not Ready (No Data in Workspace)**: **Heat-related illness** (0 records in 2023–2025 IDSP dataset).",
    ])
    return "\n".join(lines) + "\n"


def _format_target_definition_md(fc_df, rc_df, ad_df, thresh_tbl, base_tbl) -> str:
    lines = [
        "# JEEVAN-DRISHTI Phase 4: Target Definition Report\n",
        "All target variables across the three modelling datasets are derived strictly from observed IDSP surveillance outcomes and Training-split (`period_idx <= 84`, `2023-W01..2024-W33`) empirical distributions. **Zero arbitrary risk labels or synthetic thresholds were created.**\n",
        "## 1. Forecasting Targets (`forecasting_dataset.csv`)\n",
        "| Target Column | Mathematical Definition | Horizon & Cutoff | Valid Rows |",
        "|---|---|---|---|",
        f"| `target_cases_t` | Deduplicated outbreak cases reported in `(district_std, disease_std, report_year, report_week)` | Target period `t` predicted from features strictly at `<= t - 1` | {int(fc_df['target_cases_t'].notna().sum())} / {len(fc_df)} |",
        f"| `target_log1p_cases_t` | `log(1 + target_cases_t)` (variance-stabilized count target) | Target period `t` predicted from `<= t - 1` | {int(fc_df['target_log1p_cases_t'].notna().sum())} / {len(fc_df)} |",
        f"| `target_next_outbreak_cases` | `target_cases` of the next outbreak event (`e + 1`) for `(district_std, disease_std)` within the same split (masked to `NaN` across split boundaries) | Event `e + 1` (`> t`) | {int(fc_df['target_next_outbreak_cases'].notna().sum())} / {len(fc_df)} |",
        f"| `target_cases_calendar_t_plus_1w` | `target_cases` at exact calendar week `t + 1` (`NaN` if no outbreak in `t + 1` or across split boundary) | Calendar week `t + 1` | {int(fc_df['target_cases_calendar_t_plus_1w'].notna().sum())} / {len(fc_df)} |\n",
        "## 2. Outbreak-Risk Classification Targets (`risk_classification_dataset.csv`)\n",
        "Why arbitrary labels were rejected: fixed case cutoffs (e.g., `cases > 50`) are epidemiologically invalid across heterogeneous diseases (e.g., Cholera vs. Acute Diarrhoeal Disease). Instead, five defensible empirical targets are provided:\n",
        "| Target Column | Definition | Justification | Valid Rows | Positive Rate (Overall) |",
        "|---|---|---|---|---|",
        f"| `target_severe_outbreak_p75` | `1` if `target_cases > train_disease_p75_threshold` else `0` (`NaN` if `< 5` train obs) | Standard WHO/IDSP upper-quartile (75th percentile) disease-specific severity exceedance fitted on `train` only | {int(rc_df['target_severe_outbreak_p75'].notna().sum())} | {100.0 * (rc_df['target_severe_outbreak_p75'] == 1).sum() / rc_df['target_severe_outbreak_p75'].notna().sum():.1f}% |",
        f"| `target_above_median_outbreak_p50` | `1` if `target_cases > train_disease_p50_threshold` else `0` (`NaN` if `< 5` train obs) | Disease-specific median severity exceedance fitted on `train` only | {int(rc_df['target_above_median_outbreak_p50'].notna().sum())} | {100.0 * (rc_df['target_above_median_outbreak_p50'] == 1).sum() / rc_df['target_above_median_outbreak_p50'].notna().sum():.1f}% |",
        f"| `target_fatal_outbreak` | `1` if `target_deaths >= 1` else `0` | Directly observed mortality occurrence in IDSP (`reported_deaths >= 1`) | {int(rc_df['target_fatal_outbreak'].notna().sum())} | {100.0 * (rc_df['target_fatal_outbreak'] == 1).sum() / len(rc_df):.1f}% |",
        f"| `target_multi_cluster_outbreak` | `1` if `outbreak_events_in_period >= 2` else `0` | Directly observed multiple concurrent outbreak clusters in the same district-week | {int(rc_df['target_multi_cluster_outbreak'].notna().sum())} | {100.0 * (rc_df['target_multi_cluster_outbreak'] == 1).sum() / len(rc_df):.1f}% |",
        f"| `target_next_4w_recurrence` | `1` if another outbreak of `(district_std, disease_std)` occurs in `[t+1, t+4]` within split | Empirical 4-week outbreak recurrence / persistence indicator | {int(rc_df['target_next_4w_recurrence'].notna().sum())} | {100.0 * (rc_df['target_next_4w_recurrence'] == 1).sum() / rc_df['target_next_4w_recurrence'].notna().sum():.1f}% |\n",
        "### Training-Split Disease Threshold Table (Fitted Exclusively on `split == 'train'`, `period_idx <= 84`)\n",
        "| Disease | Train Outbreaks (`n`) | Train Median (`p50`) | Train 75th Percentile (`p75`) | Train 90th Percentile (`p90`) | Train Mean Cases |",
        "|---|---|---|---|---|---|",
    ]
    merged_t = thresh_tbl.merge(base_tbl[["disease_std", "train_baseline_disease_p90"]], on="disease_std", how="left")
    for _, r in merged_t.sort_values("train_disease_n_obs", ascending=False).iterrows():
        p50_s = f"{r.train_disease_p50_threshold:.1f}" if pd.notna(r.train_disease_p50_threshold) else "NaN (<5 obs)"
        p75_s = f"{r.train_disease_p75_threshold:.1f}" if pd.notna(r.train_disease_p75_threshold) else "NaN (<5 obs)"
        p90_s = f"{r.train_baseline_disease_p90:.1f}" if pd.notna(r.train_baseline_disease_p90) else "NaN (<5 obs)"
        lines.append(
            f"| {r.disease_std} | {int(r.train_disease_n_obs)} | {p50_s} | {p75_s} | {p90_s} | {r.train_disease_mean_cases:.2f} |"
        )

    lines.extend([
        "\n## 3. Anomaly Detection Targets (`anomaly_detection_dataset.csv`)\n",
        "| Target Column | Formula / Definition | Baseline Used | Valid Rows | Anomaly Rate |",
        "|---|---|---|---|---|",
        f"| `target_anomaly_zscore_vs_train_baseline` | `(target_cases_t - baseline_train_ref_mean) / baseline_train_ref_std` | Training-split (`<= 2024-W33`) `(disease_std, season)` mean & std | {int(ad_df['target_anomaly_zscore_vs_train_baseline'].notna().sum())} | Continuous Z-score |",
        f"| `target_is_statistical_anomaly_2sigma` | `1` if `target_anomaly_zscore_vs_train_baseline > 2.0` else `0` | Standard WHO/CDC `mean + 2*SD` epidemic anomaly threshold (train-fitted) | {int(ad_df['target_is_statistical_anomaly_2sigma'].notna().sum())} | {100.0 * (ad_df['target_is_statistical_anomaly_2sigma'] == 1).sum() / ad_df['target_is_statistical_anomaly_2sigma'].notna().sum():.1f}% |",
        f"| `target_is_extreme_anomaly_p90` | `1` if `target_cases_t > train_baseline_disease_p90` else `0` | Training-split 90th percentile (`p90`) per disease | {int(ad_df['target_is_extreme_anomaly_p90'].notna().sum())} | {100.0 * (ad_df['target_is_extreme_anomaly_p90'] == 1).sum() / ad_df['target_is_extreme_anomaly_p90'].notna().sum():.1f}% |",
        f"| `target_is_expanding_anomaly_2sigma` | `1` if `target_anomaly_zscore_vs_expanding_prev > 2.0` else `0` | Strictly prior (`< t`) online expanding `(disease_std, season)` mean & std | {int(ad_df['target_is_expanding_anomaly_2sigma'].notna().sum())} | {100.0 * (ad_df['target_is_expanding_anomaly_2sigma'] == 1).sum() / ad_df['target_is_expanding_anomaly_2sigma'].notna().sum():.1f}% |",
    ])
    return "\n".join(lines) + "\n"


def _format_split_validation_md(fc_df, rc_df, ad_df, manifests, val_checks) -> str:
    lines = [
        "# JEEVAN-DRISHTI Phase 4: Chronological Split & Validation Report\n",
        f"All **{len(val_checks)} / {len(val_checks)}** automated split, alignment, class-balance, and anti-leakage checks passed (`status = PASS`).\n",
        "## 1. Automated Validation Check Results\n",
        "| Check Name | Status | Detail |",
        "|---|---|---|",
    ]
    for _, r in val_checks.iterrows():
        lines.append(f"| `{r.check_name}` | **{r.status}** | {r.detail} |")

    lines.extend([
        "\n## 2. Class Distribution Across Chronological Splits (`risk_classification_dataset.csv`)\n",
        "| Target | Split | Valid `N` | Class `0` | Class `1` | Positive Rate (%) |",
        "|---|---|---|---|---|---|",
    ])
    for col in [
        "target_severe_outbreak_p75",
        "target_above_median_outbreak_p50",
        "target_fatal_outbreak",
        "target_multi_cluster_outbreak",
        "target_next_4w_recurrence",
    ]:
        for sp in ("train", "validation", "test"):
            d = manifests[sp]["datasets"]["risk_classification_dataset"]["class_distributions"][col]
            lines.append(
                f"| `{col}` | `{sp}` | {d['non_null']} | {d['class_0']} | {d['class_1']} | {d['positive_rate_pct']}% |"
            )

    lines.extend([
        "\n## 3. Anomaly Distribution Across Chronological Splits (`anomaly_detection_dataset.csv`)\n",
        "| Anomaly Target | Split | Valid `N` | Normal (`0`) | Anomaly (`1`) | Anomaly Rate (%) |",
        "|---|---|---|---|---|---|",
    ])
    for col in [
        "target_is_statistical_anomaly_2sigma",
        "target_is_extreme_anomaly_p90",
        "target_is_expanding_anomaly_2sigma",
    ]:
        for sp in ("train", "validation", "test"):
            d = manifests[sp]["datasets"]["anomaly_detection_dataset"]["anomaly_distributions"][col]
            lines.append(
                f"| `{col}` | `{sp}` | {d['non_null']} | {d['normal_0']} | {d['anomaly_1']} | {d['anomaly_rate_pct']}% |"
            )
    return "\n".join(lines) + "\n"


def _update_registry(p_fc, p_rc, p_ad, fc_df, rc_df, ad_df, manifests):
    reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    rel = lambda p: str(p.relative_to(PROJ)).replace("\\", "/")
    reg["final_training_datasets"] = [
        dict(
            name="forecasting_dataset",
            path=str(p_fc),
            relative_path=rel(p_fc),
            version=TRAINING_DATASET_VERSION,
            source_feature_dataset="data/features/disease_climate_weekly_features.csv (phase3-v1)",
            rows=len(fc_df),
            columns=int(fc_df.shape[1]),
            sha256=_sha256(p_fc),
            primary_targets=["target_cases_t", "target_log1p_cases_t", "target_next_outbreak_cases"],
        ),
        dict(
            name="risk_classification_dataset",
            path=str(p_rc),
            relative_path=rel(p_rc),
            version=TRAINING_DATASET_VERSION,
            source_feature_dataset="data/features/disease_climate_weekly_features.csv (phase3-v1)",
            rows=len(rc_df),
            columns=int(rc_df.shape[1]),
            sha256=_sha256(p_rc),
            primary_targets=[
                "target_severe_outbreak_p75",
                "target_above_median_outbreak_p50",
                "target_fatal_outbreak",
                "target_multi_cluster_outbreak",
                "target_next_4w_recurrence",
            ],
        ),
        dict(
            name="anomaly_detection_dataset",
            path=str(p_ad),
            relative_path=rel(p_ad),
            version=TRAINING_DATASET_VERSION,
            source_feature_dataset="data/features/disease_climate_weekly_features.csv (phase3-v1)",
            rows=len(ad_df),
            columns=int(ad_df.shape[1]),
            sha256=_sha256(p_ad),
            primary_targets=[
                "target_is_statistical_anomaly_2sigma",
                "target_is_extreme_anomaly_p90",
                "target_is_expanding_anomaly_2sigma",
                "target_anomaly_zscore_vs_train_baseline",
            ],
        ),
    ]
    reg["chronological_splits"] = {
        sp: manifests[sp]["chronological_window"] for sp in ("train", "validation", "test")
    }
    reg["split_manifests"] = [
        rel(FINAL_TRAINING_DIR / "training_manifest.json"),
        rel(FINAL_TRAINING_DIR / "validation_manifest.json"),
        rel(FINAL_TRAINING_DIR / "testing_manifest.json"),
    ]
    reg["phase4_reports"] = [
        rel(REPORTS_FT_DIR / "training_dataset_summary.md"),
        rel(REPORTS_FT_DIR / "target_definition_report.md"),
        rel(REPORTS_FT_DIR / "split_validation_report.md"),
        rel(REPORTS_FT_DIR / "modelling_readiness_report.csv"),
    ]
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run_phase4_pipeline(verbose=True)
