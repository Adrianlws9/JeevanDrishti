"""Frontend Integration Builder for 'Why was this district flagged?' Panel (Phase 13 Section 8).

Synthesizes verified explanations across:
- Phase 5 Outbreak Risk Classification (`phase5-v1.0.0`)
- Phase 6 Multi-Horizon Disease Forecasting (`phase6-v1.0.0`)
- Phase 7 Disease Anomaly Detection (`phase7-v1.0.0`)
- Phase 9 Geospatial Hotspot Detection (`phase9-v1.0.0`)
- Phase 12 Aggregate Mortality & OC-CFR (`phase12-v1.0.0`)

Produces structured records for all 36 Maharashtra districts containing:
1. Main evidence
2. Historical comparison
3. Important contributing features
4. Data quality
5. Model version
6. Scientific limitations
"""
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.engines.explainability.model_inspector import (
    ENGINE_VERSION,
    MANDATORY_XAI_NON_CAUSAL_DISCLAIMER,
    PROJ,
)
from src.features.geographic_features import (
    ADMIN_DIVISION_MAP,
    DISTRICT_ID_MAP,
    OFFICIAL_DISTRICTS,
)


def build_why_district_flagged_panels(
    proj_dir: Path,
    local_risk_df: pd.DataFrame,
    local_fc_df: pd.DataFrame,
    local_anom_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build the 'Why was this district flagged?' panel dataset and JSON payload for all 36 districts."""
    geo_df = pd.read_csv(proj_dir / "data" / "geospatial" / "district_disease_map_summary.csv")
    geo_pool = (
        geo_df[
            (geo_df["disease"] == "ALL_DISEASES")
            & (geo_df["reporting_period"] == "2023-2025_POOL")
        ]
        .set_index("district_name")
        .to_dict(orient="index")
    )
    # Also find the strongest disease-specific or pooled Getis-Ord Gi* hotspot per district in 2023-2025_POOL
    geo_pool_all_layers = geo_df[geo_df["reporting_period"] == "2023-2025_POOL"].copy()
    best_hotspot_map: Dict[str, Dict[str, Any]] = {}
    for d_n, grp in geo_pool_all_layers.groupby("district_name"):
        valid_g = grp[grp["gi_star_zscore"].notna()].sort_values(
            ["is_statistically_significant_hotspot", "gi_star_zscore"],
            ascending=[False, False],
        )
        if not valid_g.empty:
            top_g = valid_g.iloc[0]
            best_hotspot_map[str(d_n)] = {
                "is_statistically_significant_hotspot": bool(top_g["is_statistically_significant_hotspot"]),
                "hotspot_consensus_label": f"{top_g['hotspot_consensus_label']} ({top_g['disease']})",
                "gi_star_zscore": float(top_g["gi_star_zscore"]),
                "gi_star_p_perm": float(top_g["gi_star_p_perm"]),
            }

    mort_df = pd.read_csv(proj_dir / "data" / "severity_intelligence" / "district_mortality_summary.csv")
    mort_map = mort_df.set_index("district").to_dict(orient="index")

    fc7_df = local_fc_df[local_fc_df["horizon"] == "7d"]
    gen_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    panel_rows: List[Dict[str, Any]] = []
    panel_json_list: List[Dict[str, Any]] = []

    for d_name in OFFICIAL_DISTRICTS:
        d_id = DISTRICT_ID_MAP[d_name]
        div = ADMIN_DIVISION_MAP[d_name]
        gp = geo_pool.get(d_name, {})
        mp = mort_map.get(d_name, {})

        r_sub = local_risk_df[local_risk_df["district"] == d_name]
        f_sub = fc7_df[fc7_df["district"] == d_name]
        a_sub = local_anom_df[local_anom_df["district"] == d_name]

        if r_sub.empty:
            panel_obj = {
                "panel_title": "Why was this district flagged?",
                "district_id": d_id,
                "district_name": d_name,
                "administrative_division": div,
                "is_flagged": False,
                "flag_priority_tier": "INSUFFICIENT_SURVEILLANCE_DATA",
                "main_evidence": (
                    f"{d_name} has 0 reported IDSP outbreak clusters in the 2023–2025 workspace dataset. "
                    f"No risk, anomaly, or hotspot flags are generated because the engine never fabricates "
                    f"surveillance records."
                ),
                "historical_comparison": {
                    "observed_2023_2025_outbreak_weeks": 0,
                    "observed_2023_2025_total_cases": None,
                    "observed_2023_2025_mean_weekly_cases": None,
                    "historical_2009_2022_total_cases": (
                        int(gp["historical_2009_2022_total_cases"])
                        if pd.notna(gp.get("historical_2009_2022_total_cases"))
                        else None
                    ),
                },
                "important_contributing_features": [],
                "domain_contributions_mean_log_odds": {},
                "data_quality": {
                    "status": "INSUFFICIENT_DATA_ZERO_2023_2025_OUTBREAKS",
                    "missing_features_note": "No 2023–2025 outbreak rows to evaluate.",
                    "climate_coverage": "NASA POWER & ERA5 available; relative humidity restricted to Pune.",
                },
                "model_versions": {
                    "xai_engine": ENGINE_VERSION,
                    "risk_classifier": "phase5-v1.0.0",
                    "forecaster": "phase6-v1.0.0",
                    "anomaly_detector": "phase7-v1.0.0",
                    "geospatial_hotspot": "phase9-v1.0.0",
                },
                "limitations": MANDATORY_XAI_NON_CAUSAL_DISCLAIMER,
                "generation_timestamp": gen_ts,
            }
            panel_json_list.append(panel_obj)
            panel_rows.append(
                {
                    "panel_title": "Why was this district flagged?",
                    "district_id": d_id,
                    "district": d_name,
                    "administrative_division": div,
                    "is_flagged": False,
                    "flag_priority_tier": "INSUFFICIENT_SURVEILLANCE_DATA",
                    "peak_risk_probability": np.nan,
                    "peak_risk_category": "Insufficient Data",
                    "peak_risk_disease_and_period": "N/A",
                    "flagged_anomaly_weeks_count": 0,
                    "peak_anomaly_score": np.nan,
                    "geospatial_hotspot_class": str(gp.get("hotspot_consensus_label", "Insufficient Data")),
                    "gi_star_zscore": np.nan,
                    "forecast_7d_mean_cases": np.nan,
                    "main_evidence": panel_obj["main_evidence"],
                    "historical_comparison": "0 outbreak weeks in 2023–2025",
                    "important_contributing_features": "[]",
                    "historical_disease_mean_shap": np.nan,
                    "case_growth_mean_shap": np.nan,
                    "climate_mean_shap": np.nan,
                    "seasonal_mean_shap": np.nan,
                    "data_quality": panel_obj["data_quality"]["status"],
                    "model_version": (
                        "phase5-v1.0.0 | phase6-v1.0.0 | phase7-v1.0.0 | phase9-v1.0.0 | phase13-v1.0.0"
                    ),
                    "limitations": MANDATORY_XAI_NON_CAUSAL_DISCLAIMER,
                    "generation_timestamp": gen_ts,
                }
            )
            continue

        # Peak risk record for this district
        peak_r_idx = r_sub["model_estimated_risk_prob"].idxmax()
        peak_r = r_sub.loc[peak_r_idx]
        peak_prob = float(peak_r["model_estimated_risk_prob"])
        peak_cat = str(peak_r["risk_category"])
        peak_dis_per = f"{peak_r['disease']} ({peak_r['reporting_period']})"

        # Anomaly summary for this district
        anom_flagged = a_sub[a_sub["is_anomaly"] == True]
        n_anom = len(anom_flagged)
        max_anom_score = float(a_sub["composite_anomaly_score"].max())
        peak_a = a_sub.loc[a_sub["composite_anomaly_score"].idxmax()]

        # Geospatial hotspot status (strongest across pooled & disease-specific layers)
        bh = best_hotspot_map.get(d_name, gp)
        is_hotspot = bool(bh.get("is_statistically_significant_hotspot", False))
        hotspot_lbl = str(bh.get("hotspot_consensus_label", "Not Significant"))
        gi_z = float(bh["gi_star_zscore"]) if pd.notna(bh.get("gi_star_zscore")) else 0.0
        gi_p = float(bh["gi_star_p_perm"]) if pd.notna(bh.get("gi_star_p_perm")) else 1.0

        # Forecast 7d summary
        fc7_mean = float(f_sub["predicted_cases"].mean()) if not f_sub.empty else np.nan
        fc7_max = float(f_sub["predicted_cases"].max()) if not f_sub.empty else np.nan

        # Determine overall flag status & priority tier
        high_risk_wks = int((r_sub["risk_category"] == "High").sum())
        mod_risk_wks = int((r_sub["risk_category"] == "Moderate").sum())
        is_flagged = bool(high_risk_wks > 0 or n_anom > 0 or is_hotspot)

        if high_risk_wks > 0 and (n_anom >= 3 or is_hotspot):
            flag_tier = "MULTI_ENGINE_HIGH_PRIORITY_FLAG"
        elif high_risk_wks > 0 or is_hotspot or n_anom >= 2:
            flag_tier = "ELEVATED_SURVEILLANCE_FLAG"
        elif mod_risk_wks > 0 or n_anom == 1:
            flag_tier = "MODERATE_WATCH_FLAG"
        else:
            flag_tier = "BASELINE_SURVEILLANCE_NO_ELEVATED_FLAG"

        # Aggregate top contributing features from the district's peak risk event + district mean domain SHAP
        peak_contribs = json.loads(str(peak_r["contribution_values"]))
        top_feat_items = [
            {"feature": k, "log_odds_treeshap": v} for k, v in peak_contribs.items()
        ]

        hist_shap = round(float(r_sub["historical_disease_contribution"].mean()), 4)
        grow_shap = round(float(r_sub["case_growth_contribution"].mean()), 4)
        clim_shap = round(float(r_sub["climate_contribution"].mean()), 4)
        seas_shap = round(float(r_sub["seasonal_contribution"].mean()), 4)
        spat_shap = round(float(r_sub["spatial_neighbor_contribution"].mean()), 4)

        obs_wks = int(gp.get("observed_outbreak_weeks", len(r_sub)))
        obs_cases = int(gp.get("observed_cases", r_sub["observed_target_elevated_p75"].sum()))
        obs_mean = float(gp.get("observed_mean_cases_per_outbreak_week", 0.0))
        obs_deaths = int(mp.get("verified_deaths", 0)) if pd.notna(mp.get("verified_deaths")) else 0

        main_ev = (
            f"Risk Engine (phase5-v1.0.0): Peak calibrated risk={peak_prob:.3f} ({peak_cat}) during "
            f"{peak_dis_per} ({high_risk_wks} High-risk & {mod_risk_wks} Moderate-risk outbreak weeks). "
            f"Anomaly Engine (phase7-v1.0.0): {n_anom}/{len(a_sub)} outbreak weeks flagged "
            f"(peak score={max_anom_score:.3f} on {peak_a['disease']} in {peak_a['reporting_period']}, "
            f"Observed={int(peak_a['observed_value'])} cases vs. {peak_a['expected_baseline']}). "
            f"Geospatial Engine (phase9-v1.0.0): {hotspot_lbl} (Getis-Ord Gi* z={gi_z:+.2f}, p_perm={gi_p:.3f}). "
            f"Forecast Engine (phase6-v1.0.0): 7-day mean predicted cases={fc7_mean:.1f} (peak={fc7_max:.1f})."
        )

        hist_comp_str = (
            f"2023–2025 Verified Surveillance: {obs_wks} outbreak weeks, {obs_cases} cases "
            f"(mean {obs_mean:.1f} cases/outbreak-week), and {obs_deaths} verified deaths. "
            f"Peak event ({peak_dis_per}) top feature attributions: "
            + ", ".join(f"{k}={v:+.3f}" for k, v in peak_contribs.items())
            + "."
        )

        miss_rate = float(r_sub["affected_by_missing_data"].mean()) * 100.0
        dq_note = (
            f"{str(gp.get('data_quality_status', 'VERIFIED'))} | "
            f"Outbreak weeks with >=1 median-imputed lag feature: {miss_rate:.1f}% | "
            f"Relative humidity: {'Observed (Pune station)' if d_name == 'Pune' else 'Unavailable (Pune-only)'}."
        )

        panel_obj = {
            "panel_title": "Why was this district flagged?",
            "district_id": d_id,
            "district_name": d_name,
            "administrative_division": div,
            "is_flagged": is_flagged,
            "flag_priority_tier": flag_tier,
            "main_evidence": main_ev,
            "historical_comparison": {
                "summary_text": hist_comp_str,
                "observed_2023_2025_outbreak_weeks": obs_wks,
                "observed_2023_2025_total_cases": obs_cases,
                "observed_2023_2025_mean_weekly_cases": round(obs_mean, 2),
                "observed_2023_2025_verified_deaths": obs_deaths,
                "peak_risk_event": peak_dis_per,
                "peak_risk_calibrated_probability": round(peak_prob, 4),
                "high_risk_weeks_count": high_risk_wks,
                "flagged_anomaly_weeks_count": n_anom,
                "geospatial_gi_star_zscore": round(gi_z, 4),
                "geospatial_hotspot_label": hotspot_lbl,
            },
            "important_contributing_features": top_feat_items,
            "domain_contributions_mean_log_odds": {
                "historical_disease_lags_and_baselines": hist_shap,
                "recent_case_growth_velocity": grow_shap,
                "rainfall_and_temperature_climate": clim_shap,
                "seasonality_and_calendar": seas_shap,
                "spatial_neighbor_and_geography": spat_shap,
            },
            "data_quality": {
                "status": str(gp.get("data_quality_status", "VERIFIED")),
                "pct_records_with_imputed_initial_lags": round(miss_rate, 2),
                "details": dq_note,
            },
            "model_versions": {
                "xai_engine": ENGINE_VERSION,
                "risk_classifier": str(peak_r["model_version"]),
                "forecaster": "phase6-v1.0.0",
                "anomaly_detector": str(peak_a["model_version"]),
                "geospatial_hotspot": "phase9-v1.0.0",
                "severity_mortality": "phase12-v1.0.0",
            },
            "limitations": MANDATORY_XAI_NON_CAUSAL_DISCLAIMER,
            "generation_timestamp": gen_ts,
        }
        panel_json_list.append(panel_obj)

        panel_rows.append(
            {
                "panel_title": "Why was this district flagged?",
                "district_id": d_id,
                "district": d_name,
                "administrative_division": div,
                "is_flagged": is_flagged,
                "flag_priority_tier": flag_tier,
                "peak_risk_probability": round(peak_prob, 4),
                "peak_risk_category": peak_cat,
                "peak_risk_disease_and_period": peak_dis_per,
                "flagged_anomaly_weeks_count": n_anom,
                "peak_anomaly_score": round(max_anom_score, 4),
                "geospatial_hotspot_class": hotspot_lbl,
                "gi_star_zscore": round(gi_z, 4),
                "forecast_7d_mean_cases": round(fc7_mean, 2) if pd.notna(fc7_mean) else np.nan,
                "main_evidence": main_ev,
                "historical_comparison": hist_comp_str,
                "important_contributing_features": json.dumps(top_feat_items),
                "historical_disease_mean_shap": hist_shap,
                "case_growth_mean_shap": grow_shap,
                "climate_mean_shap": clim_shap,
                "seasonal_mean_shap": seas_shap,
                "data_quality": dq_note,
                "model_version": (
                    f"{peak_r['model_version']} | phase6-v1.0.0 | {peak_a['model_version']} | phase9-v1.0.0 | {ENGINE_VERSION}"
                ),
                "limitations": MANDATORY_XAI_NON_CAUSAL_DISCLAIMER,
                "generation_timestamp": gen_ts,
            }
        )

    panels_df = pd.DataFrame(panel_rows).sort_values(
        ["is_flagged", "peak_risk_probability", "flagged_anomaly_weeks_count"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    payload_dict = {
        "panel_title": "Why was this district flagged?",
        "engine_version": ENGINE_VERSION,
        "generated_at": datetime.now().isoformat(),
        "scientific_limitations_banner": MANDATORY_XAI_NON_CAUSAL_DISCLAIMER,
        "total_districts": len(panel_json_list),
        "flagged_districts_count": int(panels_df["is_flagged"].sum()),
        "districts": panel_json_list,
    }
    return panels_df, payload_dict
