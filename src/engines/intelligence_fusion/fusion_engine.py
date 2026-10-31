"""Transparent Rule-Based Predictive Intelligence Fusion Engine (Phase 15 Sections 4–7).

Responsibilities:
1. Fuse aligned multi-engine records using a transparent, non-averaging rule hierarchy.
2. Assign one of the 5 canonical Section 5 signal statuses:
   - `No unusual signal detected`
   - `Elevated signal`
   - `Multiple indicators aligned`
   - `Conflicting indicators`
   - `Insufficient data`
3. Preserve engine disagreements explicitly (Section 6) with methods, reporting periods, and
   both engines' outputs intact, marking conflicting records for human epidemiological review.
4. Provide a traceable `"Why this result?"` explanation (Section 7) linked to exact upstream
   engine record IDs and model versions.
"""
from datetime import datetime
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.engines.intelligence_fusion.alignment_validator import (
    EXPECTED_ENGINE_VERSIONS,
    evaluate_model_and_data_freshness,
    validate_district_and_disease,
)
from src.engines.intelligence_fusion.engine_inspector import (
    FUSION_ENGINE_VERSION,
    MANDATORY_FUSION_GOVERNANCE_DISCLAIMER,
)
from src.features.geographic_features import ADMIN_DIVISION_MAP

CANONICAL_SIGNAL_STATUSES = (
    "No unusual signal detected",
    "Elevated signal",
    "Multiple indicators aligned",
    "Conflicting indicators",
    "Insufficient data",
)

OPTIMAL_RISK_PROB_THRESHOLD = 0.265
ANOMALY_COMPOSITE_THRESHOLD = 0.55


def fuse_single_aligned_record(
    row: Dict[str, Any],
    reference_now: Optional[datetime] = None,
    max_staleness_days: int = 30,
) -> Dict[str, Any]:
    """Apply transparent rule-based fusion to a single aligned weekly or pooled record.

    Returns a dictionary satisfying both the Common Output Schema (Section 2) and the
    Evidence-Based District Intelligence Summary (Sections 5, 6, 7).
    """
    gen_ts = (reference_now or datetime.now()).isoformat()

    # Validate district and disease
    dist_id, dist_name, std_disease = validate_district_and_disease(
        row.get("district_name") or row.get("district"),
        row.get("disease"),
    )
    rep_period = str(row.get("reporting_period", "UNKNOWN"))
    obs_date = (
        str(row.get("source_observation_date"))
        if pd.notna(row.get("source_observation_date"))
        else None
    )

    # Extract raw quantities safely
    obs_cases = (
        float(row["verified_cases"])
        if pd.notna(row.get("verified_cases"))
        else (
            float(row["observed_cases"])
            if pd.notna(row.get("observed_cases"))
            else None
        )
    )
    obs_deaths = (
        float(row["verified_deaths"])
        if pd.notna(row.get("verified_deaths"))
        else (
            float(row["observed_deaths"])
            if pd.notna(row.get("observed_deaths"))
            else None
        )
    )
    cfr_pct = (
        float(row["outbreak_cfr_pct"])
        if pd.notna(row.get("outbreak_cfr_pct"))
        else (
            round(obs_deaths / obs_cases * 100.0, 2)
            if (obs_cases and obs_cases > 0 and obs_deaths is not None)
            else None
        )
    )

    risk_prob = (
        float(row["model_estimated_risk_prob"])
        if pd.notna(row.get("model_estimated_risk_prob"))
        else (
            float(row["model_estimated_risk_max"])
            if pd.notna(row.get("model_estimated_risk_max"))
            else None
        )
    )
    risk_cat = (
        str(row["risk_category"])
        if pd.notna(row.get("risk_category"))
        else "UNAVAILABLE"
    )

    fc7 = (
        float(row["forecast_7d_cases"])
        if pd.notna(row.get("forecast_7d_cases"))
        else (
            float(row["forecast_7d_mean_cases"])
            if pd.notna(row.get("forecast_7d_mean_cases"))
            else None
        )
    )
    fc14 = (
        float(row["forecast_14d_cases"])
        if pd.notna(row.get("forecast_14d_cases"))
        else (
            float(row["forecast_14d_mean_cases"])
            if pd.notna(row.get("forecast_14d_mean_cases"))
            else None
        )
    )
    fc30 = (
        float(row["forecast_30d_cases"])
        if pd.notna(row.get("forecast_30d_cases"))
        else (
            float(row["forecast_30d_mean_cases"])
            if pd.notna(row.get("forecast_30d_mean_cases"))
            else None
        )
    )
    fc7_low = (
        float(row["forecast_7d_pi80_lower"])
        if pd.notna(row.get("forecast_7d_pi80_lower"))
        else None
    )
    fc7_high = (
        float(row["forecast_7d_pi80_upper"])
        if pd.notna(row.get("forecast_7d_pi80_upper"))
        else None
    )

    is_anom_raw = row.get("is_anomaly")
    anom_score = (
        float(row["composite_anomaly_score"])
        if pd.notna(row.get("composite_anomaly_score"))
        else (
            float(row["anomaly_max_score"])
            if pd.notna(row.get("anomaly_max_score"))
            else None
        )
    )
    if pd.notna(is_anom_raw):
        is_anom = bool(is_anom_raw)
    elif anom_score is not None:
        is_anom = bool(anom_score >= ANOMALY_COMPOSITE_THRESHOLD)
    else:
        is_anom = None

    anom_tier = (
        str(row["anomaly_severity_tier"])
        if pd.notna(row.get("anomaly_severity_tier"))
        else (
            str(row["anomaly_status"])
            if pd.notna(row.get("anomaly_status"))
            else "UNAVAILABLE"
        )
    )
    anom_signals = (
        str(row["signal_types"])
        if pd.notna(row.get("signal_types"))
        else "NONE"
    )

    rain_t = (
        float(row["rain_mm_sum_t"])
        if pd.notna(row.get("rain_mm_sum_t"))
        else None
    )
    rain_4w = (
        float(row["rain_mm_accum_prev_4w"])
        if pd.notna(row.get("rain_mm_accum_prev_4w"))
        else None
    )
    tavg_t = (
        float(row["tavg_c_mean_t"])
        if pd.notna(row.get("tavg_c_mean_t"))
        else None
    )
    season_str = str(row["season"]) if pd.notna(row.get("season")) else "All_Seasons"
    clim_shap = (
        float(row["risk_shap_climate_log_odds"])
        if pd.notna(row.get("risk_shap_climate_log_odds"))
        else None
    )
    clim_rho = (
        float(row["panel_rain_4w_spearman_rho"])
        if pd.notna(row.get("panel_rain_4w_spearman_rho"))
        else None
    )

    gi_z = (
        float(row["gi_star_zscore"])
        if pd.notna(row.get("gi_star_zscore"))
        else None
    )
    gi_p = (
        float(row["gi_star_p_perm"])
        if pd.notna(row.get("gi_star_p_perm"))
        else None
    )
    is_hotspot = (
        bool(row["is_statistically_significant_hotspot"])
        if pd.notna(row.get("is_statistically_significant_hotspot"))
        else False
    )
    gi_class = (
        str(row["gi_star_class"])
        if pd.notna(row.get("gi_star_class"))
        else "NOT_SIGNIFICANT"
    )
    graph_deg = (
        int(row["graph_degree"]) if pd.notna(row.get("graph_degree")) else 0
    )

    # Check model versions & freshness
    model_versions_dict = {
        "risk_classification": (
            str(row["risk_model_version"])
            if pd.notna(row.get("risk_model_version"))
            else EXPECTED_ENGINE_VERSIONS["risk_classification"]
        ),
        "disease_forecasting": (
            str(row["forecast_model_version"])
            if pd.notna(row.get("forecast_model_version"))
            else EXPECTED_ENGINE_VERSIONS["disease_forecasting"]
        ),
        "anomaly_detection": (
            str(row["anomaly_model_version"])
            if pd.notna(row.get("anomaly_model_version"))
            else EXPECTED_ENGINE_VERSIONS["anomaly_detection"]
        ),
        "climate_intelligence": EXPECTED_ENGINE_VERSIONS["climate_intelligence"],
        "geospatial_hotspot": EXPECTED_ENGINE_VERSIONS["geospatial_hotspot"],
        "spread_simulation": EXPECTED_ENGINE_VERSIONS["spread_simulation"],
        "resource_intelligence": EXPECTED_ENGINE_VERSIONS["resource_intelligence"],
        "severity_mortality": EXPECTED_ENGINE_VERSIONS["severity_mortality"],
        "explainable_ai": EXPECTED_ENGINE_VERSIONS["explainable_ai"],
        "intelligence_fusion": FUSION_ENGINE_VERSION,
    }
    freshness_status, stale_warnings = evaluate_model_and_data_freshness(
        model_versions=model_versions_dict,
        artifact_timestamp_iso=row.get("artifact_timestamp_iso"),
        reference_now=reference_now,
        max_staleness_days=max_staleness_days,
    )

    # Evaluate data completeness across the 4 primary empirical pillars
    available_primary_pillars = 0
    missing_engines: List[str] = []
    if risk_prob is not None:
        available_primary_pillars += 1
    else:
        missing_engines.append("Risk_Classification_Engine")

    if fc7 is not None:
        available_primary_pillars += 1
    else:
        missing_engines.append("Time_Series_Forecasting_Engine")

    if anom_score is not None and is_anom is not None:
        available_primary_pillars += 1
    else:
        missing_engines.append("Anomaly_Detection_Engine")

    if obs_cases is not None and obs_deaths is not None:
        available_primary_pillars += 1
    else:
        missing_engines.append("Severity_Mortality_Engine")

    # ------------------------------------------------------------------
    # Transparent Rule-Based Signal Evaluation (Sections 4, 5, 6)
    # ------------------------------------------------------------------
    supporting_evidence: List[str] = []
    conflicting_evidence: List[str] = []
    uncertainty_limitations: List[str] = []

    if stale_warnings:
        uncertainty_limitations.extend(stale_warnings)

    # Case 0: Insufficient data (e.g., Mumbai City, Ratnagiri, or <2 primary engines available)
    if (
        std_disease == "NONE_REPORTED_2023_2025"
        or "NO_SURVEILLANCE_EVENTS" in rep_period
        or available_primary_pillars < 2
    ):
        overall_status = "Insufficient data"
        review_priority = "PRIORITY_DATA_GAP_AUDIT"
        disagreement_flag = False
        disagreement_type = "NONE_INSUFFICIENT_DATA"
        uncertainty_limitations.append(
            f"Insufficient primary analytical outputs ({available_primary_pillars}/4 primary pillars available; "
            f"missing: {', '.join(missing_engines) if missing_engines else 'zero historical surveillance events in 2023-2025'})."
        )
        why_explanation = (
            f"{dist_name} ({std_disease}, {rep_period}) is classified as 'Insufficient data' because "
            f"only {available_primary_pillars}/4 primary analytical pillars are available "
            f"({', '.join(missing_engines) if missing_engines else '0 surveillance events in 2023-2025'}). "
            f"No outbreak or elevated signal is inferred when empirical surveillance is absent."
        )
    else:
        # Pillar 1: Risk Classification Flag (Reliability weight reduced if UNVALIDATED_LIMITED_SAMPLE)
        risk_unvalidated = "UNVALIDATED" in risk_cat.upper()
        risk_elevated = bool(
            risk_prob is not None
            and (
                risk_prob >= OPTIMAL_RISK_PROB_THRESHOLD
                or risk_cat.upper() in {"HIGH", "HIGH_RISK", "ELEVATED"}
            )
            and not risk_unvalidated
        )
        if risk_unvalidated:
            uncertainty_limitations.append(
                f"Risk Classifier tier for '{std_disease}' is marked UNVALIDATED_LIMITED_SAMPLE (<5 training outbreak weeks)."
            )
        if risk_elevated and risk_prob is not None:
            supporting_evidence.append(
                f"[Risk Classification ({model_versions_dict['risk_classification']}, {rep_period})]: "
                f"Flagged elevated outbreak risk (prob={risk_prob:.3f} >= threshold {OPTIMAL_RISK_PROB_THRESHOLD}, tier={risk_cat})."
            )

        # Pillar 2: Forecast Trajectory Flag
        fc_increasing_or_high = False
        if fc7 is not None:
            if obs_cases is not None and obs_cases > 0 and fc7 >= obs_cases * 1.15 and fc7 >= 15.0:
                fc_increasing_or_high = True
                supporting_evidence.append(
                    f"[Time-Series Forecasting ({model_versions_dict['disease_forecasting']}, {rep_period})]: "
                    f"7d forecast projects upward growth ({fc7:.1f} cases vs {obs_cases:.0f} observed; "
                    f"14d={fc14 if fc14 is not None else 0:.1f}, 30d={fc30 if fc30 is not None else 0:.1f})."
                )
            elif fc14 is not None and fc14 >= fc7 * 1.10 and fc7 >= 20.0:
                fc_increasing_or_high = True
                supporting_evidence.append(
                    f"[Time-Series Forecasting ({model_versions_dict['disease_forecasting']}, {rep_period})]: "
                    f"Multi-horizon trajectory expands from 7d={fc7:.1f} to 14d={fc14:.1f} cases."
                )
            elif fc7 >= 45.0:
                fc_increasing_or_high = True
                supporting_evidence.append(
                    f"[Time-Series Forecasting ({model_versions_dict['disease_forecasting']}, {rep_period})]: "
                    f"High projected 7d case volume ({fc7:.1f} cases/week)."
                )

        # Pillar 3: Anomaly Detection Flag
        anom_elevated = bool(
            (is_anom is True)
            or (anom_score is not None and anom_score >= ANOMALY_COMPOSITE_THRESHOLD)
        )
        if anom_elevated and anom_score is not None:
            supporting_evidence.append(
                f"[Anomaly Detection ({model_versions_dict['anomaly_detection']}, {rep_period})]: "
                f"Flagged unusual activity (composite_score={anom_score:.3f} >= {ANOMALY_COMPOSITE_THRESHOLD}, "
                f"tier={anom_tier}, signals={anom_signals})."
            )

        # Pillar 4: Verified Severity / Mortality Flag
        mort_elevated = bool(obs_deaths is not None and obs_deaths >= 1.0)
        if mort_elevated and obs_deaths is not None:
            cfr_str = f"{cfr_pct:.2f}%" if cfr_pct is not None else "N/A"
            supporting_evidence.append(
                f"[Severity & Mortality ({model_versions_dict['severity_mortality']}, {rep_period})]: "
                f"Verified outbreak mortality recorded (deaths={int(obs_deaths)}, cases={int(obs_cases or 0)}, "
                f"Outbreak-Cluster CFR={cfr_str})."
            )

        # Contextual Tier 2 Indicators: Climate & Geospatial Hotspot
        clim_supportive = False
        if (clim_shap is not None and clim_shap >= 0.05) or (
            rain_4w is not None and rain_4w >= 50.0
        ):
            clim_supportive = True
            supporting_evidence.append(
                f"[Climate Intelligence ({model_versions_dict['climate_intelligence']}, {rep_period})]: "
                f"Supportive environmental context (4w_rain={rain_4w if rain_4w is not None else 0.0:.1f}mm, "
                f"season={season_str}, climate_SHAP_log_odds={clim_shap if clim_shap is not None else 0.0:+.3f}; "
                f"correlational only)."
            )

        if is_hotspot and gi_z is not None:
            supporting_evidence.append(
                f"[Geospatial Hotspot ({model_versions_dict['geospatial_hotspot']}, 2023-2025_POOL context)]: "
                f"Statistically significant spatial cluster (Getis-Ord Gi*={gi_z:+.2f}, p={gi_p if gi_p is not None else 0.0:.3f}, "
                f"class={gi_class})."
            )

        # ------------------------------------------------------------------
        # Engine Disagreement Detection (Section 6)
        # ------------------------------------------------------------------
        primary_flags = {
            "Risk_Classifier": risk_elevated,
            "Forecaster": fc_increasing_or_high,
            "Anomaly_Detector": anom_elevated,
            "Mortality_Engine": mort_elevated,
        }
        elevated_primary_names = [k for k, v in primary_flags.items() if v]
        elevated_primary_count = len(elevated_primary_names)

        disagreement_reasons: List[str] = []

        # Disagreement Pattern A: Anomaly Detector flags strong unusual spike/mortality,
        # but Risk Classifier AND Forecaster both indicate low/declining baseline
        if anom_elevated and not risk_elevated and not fc_increasing_or_high:
            if risk_prob is not None and fc7 is not None:
                disagreement_reasons.append(
                    f"ANOMALY_VS_PREDICTIVE_DISAGREEMENT: Anomaly Detector ({model_versions_dict['anomaly_detection']}, {rep_period}) "
                    f"flagged unusual activity (score={anom_score:.3f}, signals={anom_signals}), whereas "
                    f"Risk Classifier ({model_versions_dict['risk_classification']}, prob={risk_prob:.3f} < {OPTIMAL_RISK_PROB_THRESHOLD}) "
                    f"and 7d Forecaster ({model_versions_dict['disease_forecasting']}, predicted={fc7:.1f} vs observed={obs_cases or 0:.0f}) "
                    f"did not project sustained multi-week escalation."
                )

        # Disagreement Pattern B: Risk Classifier or Forecaster projects high risk/growth,
        # but Anomaly Detector reports strictly normal baseline (score < 0.30) and 0 deaths
        if (risk_elevated or fc_increasing_or_high) and not anom_elevated and not mort_elevated:
            if anom_score is not None and anom_score < 0.30:
                disagreement_reasons.append(
                    f"PREDICTIVE_VS_ANOMALY_DISAGREEMENT: Predictive engine(s) ({', '.join(elevated_primary_names)} in {rep_period}) "
                    f"indicated elevated trajectory/risk (risk_prob={risk_prob if risk_prob is not None else 0:.3f}, "
                    f"fc7={fc7 if fc7 is not None else 0:.1f}), whereas Anomaly Detector ({model_versions_dict['anomaly_detection']}) "
                    f"found current cases within normal historical seasonal bounds (composite_score={anom_score:.3f} < 0.30, deaths=0)."
                )

        # Disagreement Pattern C: High Forecast volume/growth vs Low Risk Classifier probability
        if fc_increasing_or_high and not risk_elevated and risk_prob is not None and risk_prob < 0.21:
            disagreement_reasons.append(
                f"FORECAST_VS_RISK_DISAGREEMENT: 7d Forecaster ({model_versions_dict['disease_forecasting']}) "
                f"projects {fc7:.1f} cases, whereas Risk Classifier ({model_versions_dict['risk_classification']}) "
                f"assigned low severe-outbreak probability ({risk_prob:.3f})."
            )

        conflicting_evidence.extend(disagreement_reasons)
        disagreement_flag = len(disagreement_reasons) > 0
        disagreement_type = (
            disagreement_reasons[0].split(":")[0]
            if disagreement_flag
            else "NO_PRIMARY_ENGINE_CONFLICT"
        )

        # ------------------------------------------------------------------
        # Assign Canonical Signal Status & Documented Review Priority (Section 5)
        # ------------------------------------------------------------------
        if elevated_primary_count == 0:
            if is_hotspot:
                overall_status = "Elevated signal"
                review_priority = "PRIORITY_3_ROUTINE_ENHANCED_MONITORING"
            else:
                overall_status = "No unusual signal detected"
                review_priority = "PRIORITY_4_STANDARD_SURVEILLANCE"
        elif elevated_primary_count >= 2:
            # When >= 2 primary empirical pillars align, check if there is still a sharp 1-vs-2 conflict
            if elevated_primary_count == 2 and len(disagreement_reasons) >= 1 and not mort_elevated:
                overall_status = "Multiple indicators aligned"
                review_priority = "PRIORITY_2_TARGETED_ANALYST_REVIEW"
            else:
                overall_status = "Multiple indicators aligned"
                review_priority = (
                    "PRIORITY_1_IMMEDIATE_EPIDEMIOLOGICAL_REVIEW"
                    if (mort_elevated or elevated_primary_count >= 3 or "HIGH_PRIORITY" in anom_tier)
                    else "PRIORITY_2_TARGETED_ANALYST_REVIEW"
                )
        else:
            # Exactly 1 primary pillar elevated
            if disagreement_flag:
                overall_status = "Conflicting indicators"
                review_priority = "PRIORITY_2_TARGETED_ANALYST_REVIEW"
            else:
                overall_status = "Elevated signal"
                review_priority = (
                    "PRIORITY_2_TARGETED_ANALYST_REVIEW"
                    if (mort_elevated or is_hotspot or clim_supportive)
                    else "PRIORITY_3_ROUTINE_ENHANCED_MONITORING"
                )

        # Add standard workspace limitations
        uncertainty_limitations.append(
            "Hospital bed occupancy, ICU capacity, and clinical severity counts are unavailable (NaN) in the workspace."
        )
        if pd.notna(row.get("missing_features_count")) and float(row["missing_features_count"]) > 0:
            uncertainty_limitations.append(
                f"{int(float(row['missing_features_count']))} historical lag/climate features were NaN and median-imputed using training statistics."
            )

        # Build traceable "Why this result?" plain-language explanation (Section 7)
        why_parts = [
            f"Fused Status for {dist_name} ({std_disease}, {rep_period}): '{overall_status}' "
            f"[Review Priority: {review_priority} | Primary Pillars Elevated: {elevated_primary_count}/4 ({', '.join(elevated_primary_names) or 'None'})]."
        ]
        if supporting_evidence:
            why_parts.append("Supporting Evidence: " + " | ".join(supporting_evidence))
        else:
            why_parts.append(
                f"All primary indicators remained within baseline thresholds "
                f"(risk_prob={risk_prob if risk_prob is not None else 0:.3f}, "
                f"forecast_7d={fc7 if fc7 is not None else 0:.1f}, "
                f"anomaly_score={anom_score if anom_score is not None else 0:.3f}, "
                f"verified_deaths={int(obs_deaths or 0)})."
            )
        if conflicting_evidence:
            why_parts.append("Conflicting Evidence Preserved for Analyst Review: " + " | ".join(conflicting_evidence))
        why_parts.append(
            "Traceability IDs: "
            f"risk_id={row.get('risk_record_id', 'POOLED')}, "
            f"forecast_id={row.get('forecast_7d_record_id', 'POOLED')}, "
            f"anomaly_id={row.get('anomaly_record_id', 'POOLED')}."
        )
        why_explanation = " ".join(why_parts)

    # Assemble Common Output Schema (Section 2) + Evidence-Based Intelligence Summary (Sections 5, 6, 7)
    data_quality_status = (
        "INSUFFICIENT_SURVEILLANCE_DATA"
        if overall_status == "Insufficient data"
        else (
            "STALE_MODEL_WARNING_CHECK_VERSIONS"
            if freshness_status != "FRESH_VERIFIED_MODELS"
            else "VERIFIED_SURVEILLANCE_WITH_DOCUMENTED_RESOURCE_SEVERITY_GAPS"
        )
    )

    fused_record: Dict[str, Any] = {
        # Section 2 Common Output Schema
        "district_id": dist_id,
        "district_name": dist_name,
        "administrative_division": ADMIN_DIVISION_MAP.get(dist_name, "Unknown"),
        "disease": std_disease,
        "reporting_period": rep_period,
        "source_observation_date": obs_date,
        "risk_classification_output": {
            "available": risk_prob is not None,
            "record_id": str(row["risk_record_id"]) if pd.notna(row.get("risk_record_id")) else None,
            "calibrated_risk_probability": round(risk_prob, 4) if risk_prob is not None else None,
            "decision_threshold": OPTIMAL_RISK_PROB_THRESHOLD,
            "risk_category": risk_cat,
            "flagged_elevated": bool(
                risk_prob is not None
                and risk_prob >= OPTIMAL_RISK_PROB_THRESHOLD
                and "UNVALIDATED" not in risk_cat.upper()
            ),
            "model_version": model_versions_dict["risk_classification"],
        },
        "forecast_output": {
            "available": fc7 is not None,
            "record_id": (
                str(row["forecast_7d_record_id"])
                if pd.notna(row.get("forecast_7d_record_id"))
                else None
            ),
            "forecast_7d_cases": round(fc7, 2) if fc7 is not None else None,
            "forecast_14d_cases": round(fc14, 2) if fc14 is not None else None,
            "forecast_30d_cases": round(fc30, 2) if fc30 is not None else None,
            "pi_80_lower": round(fc7_low, 2) if fc7_low is not None else None,
            "pi_80_upper": round(fc7_high, 2) if fc7_high is not None else None,
            "model_version": model_versions_dict["disease_forecasting"],
        },
        "anomaly_output": {
            "available": anom_score is not None,
            "record_id": (
                str(row["anomaly_record_id"])
                if pd.notna(row.get("anomaly_record_id"))
                else None
            ),
            "is_anomaly": is_anom,
            "composite_anomaly_score": round(anom_score, 4) if anom_score is not None else None,
            "decision_threshold": ANOMALY_COMPOSITE_THRESHOLD,
            "anomaly_severity_tier": anom_tier,
            "signal_types": anom_signals,
            "model_version": model_versions_dict["anomaly_detection"],
        },
        "climate_indicators": {
            "available": (rain_t is not None) or (clim_rho is not None),
            "season": season_str,
            "weekly_rainfall_mm": round(rain_t, 2) if rain_t is not None else None,
            "prev_4w_rainfall_accum_mm": round(rain_4w, 2) if rain_4w is not None else None,
            "weekly_mean_temp_c": round(tavg_t, 2) if tavg_t is not None else None,
            "district_rain_4w_spearman_rho": round(clim_rho, 4) if clim_rho is not None else None,
            "risk_shap_climate_log_odds": round(clim_shap, 4) if clim_shap is not None else None,
            "non_causal_note": "Correlational environmental context only; never implies biological causation.",
            "model_version": model_versions_dict["climate_intelligence"],
        },
        "geospatial_indicators": {
            "available": gi_z is not None,
            "temporal_scope": "2023-2025_POOL_CROSS_SECTIONAL_CONTEXT",
            "gi_star_zscore": round(gi_z, 4) if gi_z is not None else None,
            "gi_star_p_perm": round(gi_p, 4) if gi_p is not None else None,
            "gi_star_class": gi_class,
            "is_statistically_significant_hotspot": is_hotspot,
            "adjacent_neighbors_degree": graph_deg,
            "model_version": model_versions_dict["geospatial_hotspot"],
        },
        "resource_indicators": {
            "empirical_data_available": False,
            "empirical_available_beds": None,
            "empirical_occupied_beds": None,
            "empirical_icu_capacity": None,
            "resource_shortage_status": "Insufficient Data (No Hospital/Resource Dataset in Workspace)",
            "decision_support_calculator_available": True,
            "model_version": model_versions_dict["resource_intelligence"],
        },
        "severity_indicators": {
            "available": obs_cases is not None and obs_deaths is not None,
            "verified_outbreak_cases": obs_cases,
            "verified_outbreak_deaths": obs_deaths,
            "outbreak_cluster_cfr_pct": round(cfr_pct, 4) if cfr_pct is not None else None,
            "cfr_reliability_status": str(row.get("cfr_reliability_status", "UNAVAILABLE")),
            "clinical_hospital_admissions_count": None,
            "clinical_severe_cases_count": None,
            "model_version": model_versions_dict["severity_mortality"],
        },
        "model_versions": model_versions_dict,
        "data_quality_status": data_quality_status,
        "generated_timestamp": gen_ts,
        # Section 5, 6, 7 Evidence-Based Intelligence Summary & Disagreement Handling
        "overall_signal_status": overall_status,
        "automated_outbreak_declaration": False,
        "recommended_review_priority": review_priority,
        "primary_pillars_available_count": available_primary_pillars,
        "supporting_analytical_evidence": supporting_evidence,
        "conflicting_evidence": conflicting_evidence,
        "disagreement_flag": disagreement_flag,
        "disagreement_type": disagreement_type,
        "data_completeness_summary": (
            f"{available_primary_pillars}/4 primary analytical pillars available; "
            "empirical hospital beds and clinical severity counts are NaN across workspace."
        ),
        "model_freshness_status": freshness_status,
        "uncertainty_and_limitations": uncertainty_limitations,
        "why_this_result_explanation": why_explanation,
        "governance_disclaimer": MANDATORY_FUSION_GOVERNANCE_DISCLAIMER,
    }
    return fused_record


def flatten_fused_record_for_csv(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a nested `fused_record` dictionary into a tabular CSV row while preserving all fields."""
    r_out = rec["risk_classification_output"]
    f_out = rec["forecast_output"]
    a_out = rec["anomaly_output"]
    c_out = rec["climate_indicators"]
    g_out = rec["geospatial_indicators"]
    res_out = rec["resource_indicators"]
    s_out = rec["severity_indicators"]

    return {
        "district_id": rec["district_id"],
        "district_name": rec["district_name"],
        "administrative_division": rec["administrative_division"],
        "disease": rec["disease"],
        "reporting_period": rec["reporting_period"],
        "source_observation_date": rec["source_observation_date"],
        "overall_signal_status": rec["overall_signal_status"],
        "recommended_review_priority": rec["recommended_review_priority"],
        "automated_outbreak_declaration": rec["automated_outbreak_declaration"],
        "primary_pillars_available_count": rec["primary_pillars_available_count"],
        "disagreement_flag": rec["disagreement_flag"],
        "disagreement_type": rec["disagreement_type"],
        # Risk Classification Output
        "risk_available": r_out["available"],
        "risk_record_id": r_out["record_id"],
        "risk_calibrated_probability": r_out["calibrated_risk_probability"],
        "risk_category": r_out["risk_category"],
        "risk_flagged_elevated": r_out["flagged_elevated"],
        # Forecast Output
        "forecast_available": f_out["available"],
        "forecast_record_id": f_out["record_id"],
        "forecast_7d_cases": f_out["forecast_7d_cases"],
        "forecast_14d_cases": f_out["forecast_14d_cases"],
        "forecast_30d_cases": f_out["forecast_30d_cases"],
        "forecast_7d_pi80_lower": f_out["pi_80_lower"],
        "forecast_7d_pi80_upper": f_out["pi_80_upper"],
        # Anomaly Output
        "anomaly_available": a_out["available"],
        "anomaly_record_id": a_out["record_id"],
        "anomaly_flagged": a_out["is_anomaly"],
        "anomaly_composite_score": a_out["composite_anomaly_score"],
        "anomaly_severity_tier": a_out["anomaly_severity_tier"],
        "anomaly_signal_types": a_out["signal_types"],
        # Climate Indicators
        "climate_available": c_out["available"],
        "season": c_out["season"],
        "weekly_rainfall_mm": c_out["weekly_rainfall_mm"],
        "prev_4w_rainfall_accum_mm": c_out["prev_4w_rainfall_accum_mm"],
        "weekly_mean_temp_c": c_out["weekly_mean_temp_c"],
        "district_rain_4w_spearman_rho": c_out["district_rain_4w_spearman_rho"],
        "risk_shap_climate_log_odds": c_out["risk_shap_climate_log_odds"],
        # Geospatial Indicators
        "geospatial_available": g_out["available"],
        "gi_star_zscore": g_out["gi_star_zscore"],
        "gi_star_p_perm": g_out["gi_star_p_perm"],
        "gi_star_class": g_out["gi_star_class"],
        "is_statistically_significant_hotspot": g_out["is_statistically_significant_hotspot"],
        "adjacent_neighbors_degree": g_out["adjacent_neighbors_degree"],
        # Resource Indicators
        "empirical_resource_data_available": res_out["empirical_data_available"],
        "empirical_available_beds": res_out["empirical_available_beds"],
        "resource_shortage_status": res_out["resource_shortage_status"],
        # Severity Indicators
        "severity_available": s_out["available"],
        "verified_outbreak_cases": s_out["verified_outbreak_cases"],
        "verified_outbreak_deaths": s_out["verified_outbreak_deaths"],
        "outbreak_cluster_cfr_pct": s_out["outbreak_cluster_cfr_pct"],
        "cfr_reliability_status": s_out["cfr_reliability_status"],
        # Evidence, Conflicts, Versions, Freshness & Explainability
        "supporting_analytical_evidence": " || ".join(rec["supporting_analytical_evidence"]),
        "conflicting_evidence": " || ".join(rec["conflicting_evidence"]),
        "uncertainty_and_limitations": " || ".join(rec["uncertainty_and_limitations"]),
        "why_this_result_explanation": rec["why_this_result_explanation"],
        "model_versions_json": json.dumps(rec["model_versions"]),
        "model_freshness_status": rec["model_freshness_status"],
        "data_quality_status": rec["data_quality_status"],
        "generated_timestamp": rec["generated_timestamp"],
        "governance_disclaimer": rec["governance_disclaimer"],
    }


def fuse_dataframe_records(
    aligned_df: pd.DataFrame,
    reference_now: Optional[datetime] = None,
    max_staleness_days: int = 30,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """Run `fuse_single_aligned_record` across every row of an aligned DataFrame."""
    nested_records: List[Dict[str, Any]] = []
    flat_rows: List[Dict[str, Any]] = []

    for _, row in aligned_df.iterrows():
        rec = fuse_single_aligned_record(
            row.to_dict(),
            reference_now=reference_now,
            max_staleness_days=max_staleness_days,
        )
        nested_records.append(rec)
        flat_rows.append(flatten_fused_record_for_csv(rec))

    return nested_records, pd.DataFrame(flat_rows)
