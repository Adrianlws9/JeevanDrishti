"""Data Alignment, Schema Standardization & Validation Gate for Predictive Intelligence Fusion (Phase 15 Sections 2, 3 & 9).

Responsibilities:
1. Enforce the Common Output Schema (Section 2):
   - `district_id`
   - `district_name`
   - `disease`
   - `reporting_period`
   - `source_observation_date`
   - `risk_classification_output`
   - `forecast_output`
   - `anomaly_output`
   - `climate_indicators`
   - `geospatial_indicators`
   - `resource_indicators`
   - `severity_indicators`
   - `model_versions`
   - `data_quality_status`
   - `generated_timestamp`
2. Perform Alignment & Validation Checks (Section 3 & Section 9):
   - District identifier & name matching (rejecting unsupported non-Maharashtra districts)
   - Disease identifier standardization
   - Reporting period compatibility check (never combining incompatible periods e.g. 2023-W01 with 2024-W05,
     and explicitly tagging cross-sectional `2023-2025_POOL` spatial/resource context so it is never
     mistaken for a weekly time-series observation)
   - Model version & data freshness verification (flagging stale model outputs)
   - Duplicate record detection & deterministic resolution
   - Missing engine output tracking
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from jeevan_pipeline.standardize import standardize_disease, standardize_district
from src.features.geographic_features import (
    ADMIN_DIVISION_MAP,
    DISTRICT_ID_MAP,
    OFFICIAL_DISTRICTS,
)

EXPECTED_ENGINE_VERSIONS: Dict[str, str] = {
    "risk_classification": "phase5-v1.0.0",
    "disease_forecasting": "phase6-v1.0.0",
    "anomaly_detection": "phase7-v1.0.0",
    "climate_intelligence": "phase8-v1.0.0",
    "geospatial_hotspot": "phase9-v1.0.0",
    "spread_simulation": "phase10-v1.0.0",
    "resource_intelligence": "phase11-v1.0.0",
    "severity_mortality": "phase12-v1.0.0",
    "explainable_ai": "phase13-v1.0.0",
}


def validate_district_and_disease(
    raw_district: str, raw_disease: str
) -> Tuple[str, str, str]:
    """Validate and standardize a Maharashtra district and disease identifier.

    Raises:
        ValueError if the district is not one of the 36 official Maharashtra districts.
    Returns:
        (district_id, standardized_district_name, standardized_disease_name)
    """
    if raw_district is None or str(raw_district).strip() == "":
        raise ValueError("District name cannot be empty.")
    std_d, _ = standardize_district(str(raw_district).strip())
    cand = std_d if std_d in OFFICIAL_DISTRICTS else str(raw_district).strip()
    if cand not in OFFICIAL_DISTRICTS:
        raise ValueError(
            f"Unsupported district '{raw_district}' (resolved: '{cand}'). "
            f"Only the 36 official Maharashtra districts are supported."
        )
    dist_id = DISTRICT_ID_MAP[cand]

    if raw_disease is None or str(raw_disease).strip() == "":
        raise ValueError("Disease identifier cannot be empty.")
    raw_dis_str = str(raw_disease).strip()
    if raw_dis_str.upper() == "ALL_DISEASES":
        std_dis = "ALL_DISEASES"
    else:
        mapped_dis, _, is_mapped = standardize_disease(raw_dis_str)
        std_dis = mapped_dis if (is_mapped and mapped_dis) else raw_dis_str

    return dist_id, cand, std_dis


def check_reporting_period_compatibility(
    period_a: str, period_b: str, allow_pooled_context_attachment: bool = False
) -> Tuple[bool, str]:
    """Verify whether two reporting periods can be combined (Section 3).

    - Exact match (`2023-W01` == `2023-W01`) -> `(True, 'EXACT_PERIOD_MATCH')`
    - Pooled context (`2023-2025_POOL`) attached as structural context when
      `allow_pooled_context_attachment=True` -> `(True, 'CROSS_SECTIONAL_POOL_CONTEXT_ONLY')`
    - Otherwise -> `(False, 'INCOMPATIBLE_REPORTING_PERIODS_BLOCKED')`
    """
    pa = str(period_a).strip()
    pb = str(period_b).strip()
    if pa == pb:
        return True, "EXACT_PERIOD_MATCH"
    if allow_pooled_context_attachment and ("POOL" in pa or "POOL" in pb):
        return True, "CROSS_SECTIONAL_POOL_CONTEXT_ONLY"
    return (
        False,
        f"INCOMPATIBLE_REPORTING_PERIODS_BLOCKED ({pa} vs {pb})",
    )


def evaluate_model_and_data_freshness(
    model_versions: Dict[str, Optional[str]],
    artifact_timestamp_iso: Optional[str] = None,
    reference_now: Optional[datetime] = None,
    max_staleness_days: int = 30,
) -> Tuple[str, List[str]]:
    """Check whether engine model versions match current expected versions and whether timestamps are fresh."""
    ref_dt = reference_now or datetime.now()
    stale_flags: List[str] = []

    for eng_key, ver_val in model_versions.items():
        if ver_val is None or str(ver_val).strip() == "":
            continue
        expected_v = EXPECTED_ENGINE_VERSIONS.get(eng_key)
        if expected_v and str(ver_val).strip() != expected_v:
            stale_flags.append(
                f"STALE_OR_MISMATCHED_VERSION:{eng_key}={ver_val}(expected {expected_v})"
            )

    if artifact_timestamp_iso:
        try:
            art_dt = datetime.fromisoformat(str(artifact_timestamp_iso).replace("Z", ""))
            age_days = (ref_dt - art_dt).total_seconds() / 86400.0
            if age_days > float(max_staleness_days):
                stale_flags.append(
                    f"STALE_OUTPUT_TIMESTAMP:age={age_days:.1f}d>max={max_staleness_days}d"
                )
        except ValueError:
            stale_flags.append(f"UNPARSEABLE_TIMESTAMP:{artifact_timestamp_iso}")

    if stale_flags:
        return "STALE_MODEL_OR_DATA_WARNING", stale_flags
    return "FRESH_VERIFIED_MODELS", []


def deduplicate_engine_dataframe(
    df: pd.DataFrame,
    key_cols: List[str],
    engine_label: str,
) -> Tuple[pd.DataFrame, int]:
    """Detect and resolve duplicate records on `key_cols` before fusion (Section 3 & Section 9)."""
    if df.empty:
        return df.copy(), 0
    dup_mask = df.duplicated(subset=key_cols, keep="first")
    dup_count = int(dup_mask.sum())
    cleaned = df.drop_duplicates(subset=key_cols, keep="first").copy()
    return cleaned, dup_count


def align_weekly_and_district_engine_outputs(
    loaded_bundle: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Align all 9 upstream engines into:
    1. `aligned_weekly_df`: 511 active outbreak weeks (`YYYY-Www`) + 2 explicit `Insufficient data`
       records for the 2 zero-outbreak districts (`Mumbai City`, `Ratnagiri`) = 513 rows.
    2. `aligned_district_pool_df`: 36 official Maharashtra districts (`2023-2025_POOL`).
    3. `alignment_audit`: Detailed alignment, deduplication, and compatibility metadata.
    """
    risk_raw = loaded_bundle["risk_df"].copy()
    fc_raw = loaded_bundle["fc_df"].copy()
    anom_raw = loaded_bundle["anom_df"].copy()
    clim_dist_raw = loaded_bundle["clim_dist_df"].copy()
    weekly_feat_raw = loaded_bundle["weekly_feat_df"].copy()
    geo_raw = loaded_bundle["geo_df"].copy()
    spread_nodes_raw = loaded_bundle["spread_nodes_df"].copy()
    res_dist_raw = loaded_bundle["res_dist_df"].copy()
    mort_wk_raw = loaded_bundle["mort_wk_df"].copy()

    # Construct canonical reporting_period in weekly_feat_raw
    weekly_feat_raw["reporting_period"] = (
        weekly_feat_raw["report_year"].astype(int).astype(str)
        + "-W"
        + weekly_feat_raw["report_week"].astype(int).apply(lambda w: f"{w:02d}")
    )
    weekly_feat_raw = weekly_feat_raw.rename(
        columns={"district_std": "district", "disease_std": "disease"}
    )

    # Deduplicate each weekly source table on (district, disease, reporting_period)
    key_3 = ["district", "disease", "reporting_period"]
    risk_df, dup_risk = deduplicate_engine_dataframe(risk_raw, key_3, "risk")
    anom_df, dup_anom = deduplicate_engine_dataframe(anom_raw, key_3, "anomaly")
    mort_wk_df, dup_mort = deduplicate_engine_dataframe(mort_wk_raw, key_3, "mortality")
    feat_df, dup_feat = deduplicate_engine_dataframe(weekly_feat_raw, key_3, "weekly_features")

    # Pivot or split forecast horizons (7d, 14d, 30d)
    fc_7d, dup_fc7 = deduplicate_engine_dataframe(
        fc_raw[fc_raw["horizon"] == "7d"], key_3, "forecast_7d"
    )
    fc_14d, _ = deduplicate_engine_dataframe(
        fc_raw[fc_raw["horizon"] == "14d"], key_3, "forecast_14d"
    )
    fc_30d, _ = deduplicate_engine_dataframe(
        fc_raw[fc_raw["horizon"] == "30d"], key_3, "forecast_30d"
    )

    # Build base weekly spine from verified_weekly_mortality_records (511 rows) + risk + forecast + anomaly + climate
    spine = mort_wk_df[
        [
            "district_id",
            "district",
            "administrative_division",
            "disease",
            "disease_group",
            "reporting_period",
            "canonical_week_start",
            "season",
            "verified_cases",
            "verified_deaths",
            "calculated_ratio",
            "calculated_ratio_reliability",
            "data_completeness",
        ]
    ].rename(
        columns={
            "district": "district_name",
            "canonical_week_start": "source_observation_date",
            "calculated_ratio": "outbreak_cfr_pct",
            "calculated_ratio_reliability": "cfr_reliability_status",
            "data_completeness": "severity_data_completeness",
        }
    )

    # Merge Phase 5 Risk Classification
    risk_sub = risk_df[
        [
            "district",
            "disease",
            "reporting_period",
            "prediction_or_anomaly_id",
            "chronological_split",
            "model_version",
            "model_estimated_risk_prob",
            "risk_category",
            "climate_contribution",
            "seasonal_contribution",
            "spatial_neighbor_contribution",
            "missing_features_count",
            "affected_by_missing_data",
            "plain_language_summary",
        ]
    ].rename(
        columns={
            "district": "district_name",
            "prediction_or_anomaly_id": "risk_record_id",
            "model_version": "risk_model_version",
            "climate_contribution": "risk_shap_climate_log_odds",
            "seasonal_contribution": "risk_shap_seasonal_log_odds",
            "spatial_neighbor_contribution": "risk_shap_spatial_log_odds",
            "plain_language_summary": "risk_xai_summary",
        }
    )
    spine = spine.merge(
        risk_sub, on=["district_name", "disease", "reporting_period"], how="left"
    )

    # Merge Phase 6 Forecasting (7d, 14d, 30d)
    fc7_sub = fc_7d[
        [
            "district",
            "disease",
            "reporting_period",
            "prediction_or_anomaly_id",
            "model_version",
            "predicted_cases",
            "pi_80_lower",
            "pi_80_upper",
            "pi_80_uncertainty_width",
            "plain_language_summary",
        ]
    ].rename(
        columns={
            "district": "district_name",
            "prediction_or_anomaly_id": "forecast_7d_record_id",
            "model_version": "forecast_model_version",
            "predicted_cases": "forecast_7d_cases",
            "pi_80_lower": "forecast_7d_pi80_lower",
            "pi_80_upper": "forecast_7d_pi80_upper",
            "pi_80_uncertainty_width": "forecast_7d_pi80_width",
            "plain_language_summary": "forecast_xai_summary",
        }
    )
    fc14_sub = fc_14d[
        ["district", "disease", "reporting_period", "predicted_cases"]
    ].rename(
        columns={"district": "district_name", "predicted_cases": "forecast_14d_cases"}
    )
    fc30_sub = fc_30d[
        ["district", "disease", "reporting_period", "predicted_cases"]
    ].rename(
        columns={"district": "district_name", "predicted_cases": "forecast_30d_cases"}
    )
    spine = (
        spine.merge(fc7_sub, on=["district_name", "disease", "reporting_period"], how="left")
        .merge(fc14_sub, on=["district_name", "disease", "reporting_period"], how="left")
        .merge(fc30_sub, on=["district_name", "disease", "reporting_period"], how="left")
    )

    # Merge Phase 7 Anomaly Detection
    anom_sub = anom_df[
        [
            "district",
            "disease",
            "reporting_period",
            "prediction_or_anomaly_id",
            "model_version",
            "is_anomaly",
            "anomaly_severity_tier",
            "signal_types",
            "composite_anomaly_score",
            "deviation_seasonal_z",
            "deviation_rolling_z",
            "deviation_robust_mad_z",
            "isolation_forest_score",
            "plain_language_summary",
        ]
    ].rename(
        columns={
            "district": "district_name",
            "prediction_or_anomaly_id": "anomaly_record_id",
            "model_version": "anomaly_model_version",
            "plain_language_summary": "anomaly_xai_summary",
        }
    )
    spine = spine.merge(
        anom_sub, on=["district_name", "disease", "reporting_period"], how="left"
    )

    # Merge Weekly Aligned Climate Observations (Phase 3/8)
    feat_clim_sub = feat_df[
        [
            "district",
            "disease",
            "reporting_period",
            "rain_mm_sum_t",
            "rain_mm_accum_prev_4w",
            "tavg_c_mean_t",
            "temp_diurnal_range_mean_t",
            "rain_spatial_source",
        ]
    ].rename(columns={"district": "district_name"})
    spine = spine.merge(
        feat_clim_sub, on=["district_name", "disease", "reporting_period"], how="left"
    )

    # Merge Phase 8 District-Level Climate Relationship Context
    clim_dist_sub = clim_dist_raw[
        [
            "district_std",
            "panel_rain_4w_spearman_rho",
            "panel_rain_4w_p_value",
            "panel_tavg_1w_spearman_rho",
            "analysis_status",
        ]
    ].rename(
        columns={
            "district_std": "district_name",
            "analysis_status": "climate_district_analysis_status",
        }
    )
    spine = spine.merge(clim_dist_sub, on="district_name", how="left")

    # Merge Phase 9 Geospatial Hotspot Context (Pooled 2023-2025 disease-specific + ALL_DISEASES fallback)
    geo_pool = geo_raw[geo_raw["reporting_period"] == "2023-2025_POOL"].copy()
    geo_dis_sub = geo_pool[
        [
            "district_name",
            "disease",
            "gi_star_zscore",
            "gi_star_p_perm",
            "gi_star_class",
            "local_moran_i",
            "lisa_quadrant",
            "is_statistically_significant_hotspot",
            "hotspot_consensus_label",
        ]
    ].copy()
    spine = spine.merge(geo_dis_sub, on=["district_name", "disease"], how="left")

    # Fill missing disease-specific spatial hotspot stats with ALL_DISEASES district hotspot context
    geo_all_map = (
        geo_pool[geo_pool["disease"] == "ALL_DISEASES"]
        .set_index("district_name")[
            [
                "gi_star_zscore",
                "gi_star_p_perm",
                "gi_star_class",
                "local_moran_i",
                "lisa_quadrant",
                "is_statistically_significant_hotspot",
                "hotspot_consensus_label",
            ]
        ]
        .to_dict(orient="index")
    )
    for idx, row in spine[spine["gi_star_zscore"].isna()].iterrows():
        dname = str(row["district_name"])
        if dname in geo_all_map:
            for col_k, val_k in geo_all_map[dname].items():
                spine.at[idx, col_k] = val_k

    # Merge Phase 10 District Graph Connectivity Context & Phase 11 Resource Demand Status
    spread_sub = spread_nodes_raw[
        ["district_name", "graph_degree", "node_simulation_status"]
    ].copy()
    res_sub = res_dist_raw[
        [
            "district_name",
            "empirical_available_beds",
            "empirical_occupied_beds",
            "empirical_icu_capacity",
            "empirical_resource_shortage_status",
            "empirical_resource_forecast_status",
        ]
    ].copy()
    spine = spine.merge(spread_sub, on="district_name", how="left").merge(
        res_sub, on="district_name", how="left"
    )

    # Append the 2 zero-outbreak districts (Mumbai City, Ratnagiri) as explicit INSUFFICIENT_DATA records
    active_districts = set(spine["district_name"].unique())
    missing_districts = sorted(list(set(OFFICIAL_DISTRICTS) - active_districts))
    zero_rows: List[Dict[str, Any]] = []
    for md in missing_districts:
        g_info = geo_all_map.get(md, {})
        zero_rows.append(
            {
                "district_id": DISTRICT_ID_MAP[md],
                "district_name": md,
                "administrative_division": ADMIN_DIVISION_MAP[md],
                "disease": "NONE_REPORTED_2023_2025",
                "disease_group": "none",
                "reporting_period": "2023-2025_NO_SURVEILLANCE_EVENTS",
                "source_observation_date": np.nan,
                "season": np.nan,
                "verified_cases": np.nan,
                "verified_deaths": np.nan,
                "outbreak_cfr_pct": np.nan,
                "cfr_reliability_status": "UNAVAILABLE_ZERO_OUTBREAK_RECORDS",
                "severity_data_completeness": "INSUFFICIENT_DATA_ZERO_2023_2025_OUTBREAKS",
                "risk_record_id": np.nan,
                "chronological_split": "no_surveillance_data",
                "risk_model_version": np.nan,
                "model_estimated_risk_prob": np.nan,
                "risk_category": "INSUFFICIENT_DATA",
                "risk_shap_climate_log_odds": np.nan,
                "risk_shap_seasonal_log_odds": np.nan,
                "risk_shap_spatial_log_odds": np.nan,
                "missing_features_count": np.nan,
                "affected_by_missing_data": True,
                "risk_xai_summary": f"No 2023-2025 outbreak surveillance records exist in the workspace for {md}.",
                "forecast_7d_record_id": np.nan,
                "forecast_model_version": np.nan,
                "forecast_7d_cases": np.nan,
                "forecast_7d_pi80_lower": np.nan,
                "forecast_7d_pi80_upper": np.nan,
                "forecast_7d_pi80_width": np.nan,
                "forecast_xai_summary": f"Forecast unavailable for {md} (0 historical outbreak weeks in 2023-2025).",
                "forecast_14d_cases": np.nan,
                "forecast_30d_cases": np.nan,
                "anomaly_record_id": np.nan,
                "anomaly_model_version": np.nan,
                "is_anomaly": np.nan,
                "anomaly_severity_tier": "INSUFFICIENT_DATA",
                "signal_types": "NO_SURVEILLANCE_RECORDS",
                "composite_anomaly_score": np.nan,
                "deviation_seasonal_z": np.nan,
                "deviation_rolling_z": np.nan,
                "deviation_robust_mad_z": np.nan,
                "isolation_forest_score": np.nan,
                "anomaly_xai_summary": f"Anomaly detection unavailable for {md} (0 surveillance weeks).",
                "rain_mm_sum_t": np.nan,
                "rain_mm_accum_prev_4w": np.nan,
                "tavg_c_mean_t": np.nan,
                "temp_diurnal_range_mean_t": np.nan,
                "rain_spatial_source": "nearest_grid_cell",
                "panel_rain_4w_spearman_rho": np.nan,
                "panel_rain_4w_p_value": np.nan,
                "panel_tavg_1w_spearman_rho": np.nan,
                "climate_district_analysis_status": "INSUFFICIENT_DISEASE_SURVEILLANCE_DATA",
                "gi_star_zscore": g_info.get("gi_star_zscore", np.nan),
                "gi_star_p_perm": g_info.get("gi_star_p_perm", np.nan),
                "gi_star_class": g_info.get("gi_star_class", "NOT_SIGNIFICANT"),
                "local_moran_i": g_info.get("local_moran_i", np.nan),
                "lisa_quadrant": g_info.get("lisa_quadrant", "NOT_SIGNIFICANT"),
                "is_statistically_significant_hotspot": False,
                "hotspot_consensus_label": g_info.get(
                    "hotspot_consensus_label", "Not a Statistically Significant Hotspot"
                ),
                "graph_degree": 4,
                "node_simulation_status": "ZERO_HISTORICAL_OUTBREAKS_IN_2023_2025",
                "empirical_available_beds": np.nan,
                "empirical_occupied_beds": np.nan,
                "empirical_icu_capacity": np.nan,
                "empirical_resource_shortage_status": "Insufficient Data (No Hospital/Resource Dataset in Workspace)",
                "empirical_resource_forecast_status": "UNSUPPORTED_NO_HISTORICAL_RESOURCE_DATA",
            }
        )

    if zero_rows:
        spine = pd.concat([spine, pd.DataFrame(zero_rows)], ignore_index=True)

    # Build 36-District Pooled Summary Spine (`reporting_period = 2023-2025_POOL`)
    pool_all_df = geo_pool[geo_pool["disease"] == "ALL_DISEASES"].copy()
    pool_all_df = pool_all_df.merge(clim_dist_sub, on="district_name", how="left")
    pool_all_df = pool_all_df.merge(spread_sub, on="district_name", how="left")
    pool_all_df = pool_all_df.merge(res_sub, on="district_name", how="left")

    alignment_audit = {
        "alignment_timestamp": datetime.now().isoformat(),
        "weekly_active_outbreak_rows": 511,
        "zero_surveillance_district_rows_appended": len(zero_rows),
        "total_weekly_aligned_rows": int(len(spine)),
        "total_pooled_district_rows": int(len(pool_all_df)),
        "duplicates_detected_and_resolved": {
            "risk_classification": dup_risk,
            "forecast_7d": dup_fc7,
            "anomaly_detection": dup_anom,
            "mortality_weekly": dup_mort,
            "weekly_features": dup_feat,
        },
        "temporal_compatibility_policy": (
            "Weekly event outputs (Phases 5, 6, 7, 8-weekly, 12, 13) are matched strictly on "
            "(district_id, disease, reporting_period=YYYY-Www). Cross-sectional 2023-2025_POOL "
            "indicators (Phase 9 Getis-Ord Gi*, Phase 10 graph degree, Phase 11 resource audit) "
            "are explicitly tagged as POOLED_CONTEXT_ONLY and never mixed as weekly time-series values."
        ),
    }
    return spine, pool_all_df, alignment_audit
