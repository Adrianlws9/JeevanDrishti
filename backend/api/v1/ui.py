"""
Frontend Data API (Phase 19)
============================
Read-only endpoints under /api/v1/ui/* that expose the REAL analytical artifacts
(CSV/JSON files produced by Phases 1-18) in shapes convenient for the frontend.

Rules:
- Nothing is invented. Missing values are returned as null (JSON) and rendered as
  "Not Available" / "Insufficient Data" by the frontend.
- Files are loaded once and cached in memory; polling never triggers model inference.
"""

import json
import math
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from backend.config import settings
from backend.database.connection import get_db_connection

router = APIRouter(prefix="/ui", tags=["Frontend Data (real artifacts)"])

D = settings.DATA_DIR
NO_OUTBREAK_PERIOD = "2023-2025_NO_REPORTED_OUTBREAKS"


# ----------------------------------------------------------------------------- helpers
def _clean_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if hasattr(v, "item"):
        try:
            v = v.item()
        except Exception:
            pass
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
    return v


def records(df: pd.DataFrame, cols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    if cols:
        df = df[[c for c in cols if c in df.columns]]
    out = []
    for row in df.to_dict("records"):
        out.append({k: _clean_value(v) for k, v in row.items()})
    return out


@lru_cache(maxsize=None)
def load(rel: str, usecols: Optional[tuple] = None) -> pd.DataFrame:
    p = D / rel
    if not p.is_file():
        return pd.DataFrame()
    return pd.read_csv(p, usecols=list(usecols) if usecols else None)


@lru_cache(maxsize=1)
def points() -> List[Dict[str, Any]]:
    p = D / "geospatial" / "maharashtra_district_points.geojson"
    gj = json.loads(p.read_text(encoding="utf-8"))
    out = []
    for f in gj["features"]:
        pr = f["properties"]
        g = f.get("geometry")
        out.append({
            "district_id": f.get("id"),
            "district_name": pr.get("district_name"),
            "division": pr.get("administrative_division"),
            "lon": g["coordinates"][0] if g else None,
            "lat": g["coordinates"][1] if g else None,
            "has_geometry": bool(g),
        })
    return out


@lru_cache(maxsize=1)
def registry() -> Dict[str, Any]:
    p = settings.MODEL_REGISTRY_JSON
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def map_records() -> pd.DataFrame:
    return load("geospatial/district_period_map_records.csv")


def real_obs() -> pd.DataFrame:
    df = map_records()
    return df[df["reporting_period"] != NO_OUTBREAK_PERIOD]


def _period_filter(df: pd.DataFrame, period: Optional[str]) -> pd.DataFrame:
    if period and period.upper() != "ALL":
        if len(period) == 4:  # year
            return df[df["reporting_period"].str.startswith(period)]
        return df[df["reporting_period"] == period]
    return df


# ----------------------------------------------------------------------------- bootstrap
@router.get("/bootstrap")
def bootstrap():
    obs = real_obs()
    diseases = (
        obs.groupby("disease")
        .agg(records=("observed_cases", "size"), cases=("observed_cases", "sum"),
             districts=("district_name", "nunique"))
        .reset_index().sort_values("cases", ascending=False)
    )
    periods = sorted(obs["reporting_period"].unique().tolist())
    active = set(obs["district_name"].unique())
    pts = points()
    for p in pts:
        p["has_surveillance_records"] = p["district_name"] in active
    return {
        "districts": pts,
        "diseases": records(diseases),
        "periods": periods,
        "years": sorted({p[:4] for p in periods}),
        "geometry_note": (
            "District boundary polygons are not present in the project workspace. "
            "The map shows verified district centroid points only; no boundaries are drawn or invented."
        ),
    }


# ----------------------------------------------------------------------------- command centre
@router.get("/command-center")
def command_center(period: Optional[str] = Query(None)):
    obs = real_obs()
    sel = _period_filter(obs, period)
    flagged_anom = sel[sel["is_anomaly"] == True]  # noqa: E712
    risk_hi = sel[sel["risk_category"].isin(["High"])]
    elevated_districts = set(flagged_anom["district_name"]) | set(risk_hi["district_name"])

    all_rows = map_records()
    complete = (all_rows["data_quality_status"] == "COMPLETE_GEOCODED_SURVEILLANCE_RECORD").mean() if len(all_rows) else None

    reg = registry()
    status_dist = reg.get("status_distribution", {})

    latest_start = obs["canonical_week_start"].max() if len(obs) else None
    latest_period = obs["reporting_period"].max() if len(obs) else None

    meta = {}
    with get_db_connection() as conn:
        for r in conn.execute("SELECT key, value, updated_at FROM system_metadata").fetchall():
            meta[r["key"]] = {"value": r["value"], "updated_at": r["updated_at"]}
        last_ing = conn.execute(
            "SELECT filename, status, completed_at FROM ingestion_audit_logs WHERE status='SUCCESS' ORDER BY id DESC LIMIT 1"
        ).fetchone()

    return {
        "selected_period": period or "ALL",
        "districts_with_surveillance_records": int(sel["district_name"].nunique()),
        "districts_in_registry": len(points()),
        "diseases_in_selection": int(sel["disease"].nunique()),
        "diseases_in_dataset": int(obs["disease"].nunique()),
        "total_observed_cases": float(sel["observed_cases"].sum()) if len(sel) else None,
        "total_observed_deaths": float(sel["observed_deaths"].sum()) if len(sel) else None,
        "outbreak_week_records": int(len(sel)),
        "districts_with_elevated_signals": len(elevated_districts),
        "elevated_district_names": sorted(elevated_districts),
        "active_anomaly_signals": int(len(flagged_anom)),
        "latest_source_period": latest_period,
        "latest_source_week_start": latest_start,
        "last_successful_ingestion": (
            {"filename": last_ing["filename"], "completed_at": last_ing["completed_at"]} if last_ing else None
        ),
        "metadata_timestamps": meta,
        "model_status": {
            "registered_models": reg.get("total_registered_models"),
            "status_distribution": status_dist,
            "registry_version": reg.get("registry_version"),
        },
        "data_completeness_pct": round(float(complete) * 100, 1) if complete is not None else None,
        "completeness_definition": "Share of district-disease-week rows with complete geocoded surveillance records.",
        "reporting_note": (
            "Rows exist only for weeks in which an outbreak was reported. A missing week is NOT a zero-case week."
        ),
    }


# ----------------------------------------------------------------------------- map
@router.get("/map")
def map_view(disease: str = "ALL", period: str = "ALL"):
    obs = real_obs()
    if disease != "ALL":
        obs = obs[obs["disease"] == disease]
    sel = _period_filter(obs, period)

    # hotspot statistics (pooled period or year) from the precomputed summary
    hs = load("geospatial/district_disease_map_summary.csv")
    hp = "2023-2025_POOL" if period == "ALL" else (period if len(period) == 4 else None)
    hrows = pd.DataFrame()
    if hp:
        hrows = hs[(hs["disease"] == ("ALL_DISEASES" if disease == "ALL" else disease)) & (hs["reporting_period"] == hp)]
    hmap = {r["district_name"]: r for r in records(hrows)} if len(hrows) else {}

    out = []
    for p in points():
        name = p["district_name"]
        g = sel[sel["district_name"] == name]
        row = {**p, "n_records": int(len(g))}
        if len(g) == 0:
            row.update(status="INSUFFICIENT_DATA", observed_cases=None, forecast_7d=None,
                       anomaly_flagged_weeks=None, anomaly_max_score=None,
                       risk_probability=None, risk_category=None)
        else:
            risk = g["calibrated_risk_probability"].dropna()
            cats = [c for c in g["risk_category"].dropna().unique() if c in ("High", "Moderate", "Low")]
            order = {"High": 3, "Moderate": 2, "Low": 1}
            top_cat = max(cats, key=lambda c: order[c]) if cats else None
            row.update(
                status="OK",
                observed_cases=float(g["observed_cases"].sum()),
                observed_deaths=float(g["observed_deaths"].sum()),
                forecast_7d=float(g["forecast_7d_cases"].mean()) if g["forecast_7d_cases"].notna().any() else None,
                anomaly_flagged_weeks=int((g["is_anomaly"] == True).sum()),  # noqa: E712
                anomaly_max_score=float(g["anomaly_score"].max()) if g["anomaly_score"].notna().any() else None,
                risk_probability=float(risk.max()) if len(risk) else None,
                risk_category=top_cat,
            )
        h = hmap.get(name)
        row["gi_star_zscore"] = h.get("gi_star_zscore") if h else None
        row["gi_star_class"] = h.get("gi_star_class") if h else None
        row["hotspot_label"] = h.get("hotspot_consensus_label") if h else None
        row["is_significant_hotspot"] = h.get("is_statistically_significant_hotspot") if h else None
        out.append(row)
    return {"disease": disease, "period": period, "districts": out,
            "hotspot_available": bool(hp and len(hrows))}


# ----------------------------------------------------------------------------- series (observed + anomaly + forecast)
def _expected_from_text(s: Any) -> Optional[float]:
    if not isinstance(s, str):
        return None
    m = re.search(r"seasonal_median=([0-9.]+)", s)
    return float(m.group(1)) if m else None


@router.get("/series")
def series(district: str, disease: str):
    obs = real_obs()
    g = obs[(obs["district_name"] == district) & (obs["disease"] == disease)].sort_values("reporting_period")
    if g.empty:
        return {"district": district, "disease": disease, "status": "INSUFFICIENT_DATA", "points": []}
    return {
        "district": district, "disease": disease, "status": "OK",
        "points": records(g, ["reporting_period", "canonical_week_start", "observed_cases", "observed_deaths",
                              "forecast_7d_cases", "forecast_7d_lower_80", "forecast_7d_upper_80",
                              "is_anomaly", "anomaly_score", "calibrated_risk_probability", "risk_category"]),
        "note": "Only weeks with reported outbreak records are shown. Gaps are not zeros.",
    }


@router.get("/forecast")
def forecast(district: str, disease: str, horizon_days: int = 7):
    if horizon_days not in (7, 14, 30):
        raise HTTPException(400, "horizon_days must be 7, 14 or 30")
    avp = load("reports/actual_vs_predicted.csv")
    g = avp[(avp["district_std"] == district) & (avp["disease_std"] == disease) & (avp["horizon_days"] == horizon_days)]
    perf = load("reports/forecasting_performance.csv")
    perf = perf[(perf["horizon_days"] == horizon_days)]
    reg_models = [m for m in registry().get("models", []) if m.get("model_type") == "Forecasting"]
    info = [{"model_id": m["model_id"], "model_name": m["model_name"], "model_version": m["model_version"],
             "model_status": m["model_status"], "training_period": m["training_period"],
             "testing_period": m["testing_period"]} for m in reg_models]
    if g.empty:
        return {"status": "UNSUPPORTED", "reason": "No forecast was produced for this district-disease pair "
                "(insufficient training history). No forecast is shown.", "points": [],
                "performance": records(perf), "models": info}
    g = g.sort_values(["report_year", "report_week"])
    pts = records(g, ["split", "report_year", "report_week", "canonical_week_start", "actual_cases",
                      "predicted_cases", "prediction_lower_80", "prediction_upper_80", "inside_pi_80",
                      "last_value_baseline_pred", "seasonal_naive_baseline_pred", "model_name", "model_version"])
    return {
        "status": "OK", "horizon_days": horizon_days, "points": pts, "performance": records(perf), "models": info,
        "note": ("Predictions are model forecasts issued at each origin week and compared with later observations. "
                 "'test' split rows are held-out. No forward forecast beyond the last ingested week is available."),
    }


# ----------------------------------------------------------------------------- disease explorer
@router.get("/disease/{disease}")
def disease_view(disease: str, period: str = "ALL"):
    obs = real_obs()
    g = obs[obs["disease"] == disease]
    if g.empty:
        raise HTTPException(404, f"No records for disease '{disease}'")
    g = _period_filter(g, period)
    weekly = g.groupby("reporting_period").agg(cases=("observed_cases", "sum"), deaths=("observed_deaths", "sum"),
                                              districts=("district_name", "nunique")).reset_index()
    by_dist = g.groupby("district_name").agg(cases=("observed_cases", "sum"), deaths=("observed_deaths", "sum"),
                                            outbreak_weeks=("observed_cases", "size")).reset_index().sort_values("cases", ascending=False)
    corr = load("climate_intelligence/disease_climate_correlations.csv")
    corr = corr[corr["disease_std"] == disease]
    mort = load("severity_intelligence/disease_mortality_summary.csv")
    mort = mort[(mort["disease"] == disease) & (mort["district"] == "ALL_MAHARASHTRA_ACTIVE_DISTRICTS")]
    all_g = obs[obs["disease"] == disease]
    return {
        "disease": disease, "period": period,
        "weekly": records(weekly), "districts": records(by_dist),
        "coverage": {
            "districts_with_records": int(all_g["district_name"].nunique()),
            "districts_total": len(points()),
            "outbreak_week_records": int(len(all_g)),
            "first_period": all_g["reporting_period"].min(), "last_period": all_g["reporting_period"].max(),
        },
        "climate_association": records(corr), "mortality": records(mort),
        "label": "OBSERVED surveillance counts (IDSP weekly outbreak reports).",
    }


# ----------------------------------------------------------------------------- anomalies
@router.get("/anomalies")
def anomalies(district: Optional[str] = None, disease: Optional[str] = None,
              flagged_only: bool = True, limit: int = 300):
    df = load("reports/anomaly_detection_results.csv")
    if district:
        df = df[df["district"] == district]
    if disease:
        df = df[df["disease"] == disease]
    total_all = len(df)
    if flagged_only:
        df = df[df["is_anomaly"] == True]  # noqa: E712
    df = df.sort_values("reporting_period")
    timeline = df.groupby("reporting_period").size().reset_index(name="flagged")
    df = df.copy()
    df["expected_cases"] = df["triggering_features"].apply(_expected_from_text)
    detail = df.sort_values("anomaly_score", ascending=False).head(limit)
    return {
        "total_records_in_scope": int(total_all), "flagged_count": int(len(df)),
        "timeline": records(timeline),
        "records": records(detail, ["district", "disease", "reporting_period", "split", "observed_cases",
                                    "expected_cases", "anomaly_score", "is_anomaly", "anomaly_severity_tier",
                                    "signal_types", "historical_baseline", "triggering_features", "seasonal_zscore",
                                    "rolling_zscore", "robust_mad_zscore", "isolation_forest_score",
                                    "methods_flagged_count", "reporting_coverage_factor", "method", "model_version",
                                    "interpretation_note"]),
        "caution": "An elevated signal is a prompt for investigation, not proof of an outbreak.",
    }


# ----------------------------------------------------------------------------- climate
@router.get("/climate")
def climate(district: Optional[str] = None, disease: Optional[str] = None):
    corr = load("climate_intelligence/disease_climate_correlations.csv")
    lag = load("climate_intelligence/lag_analysis_results.csv")
    lag_sel = pd.DataFrame()
    if disease:
        lag_sel = lag[(lag["disease_or_series"] == disease)]
        lag_sel = lag_sel[lag_sel["climate_group"].isin(lag_sel["climate_group"].unique())]
    ts = []
    if district:
        cf = load("features/district_week_climate_features_2023_2025.csv",
                  ("district_std", "report_year", "report_week", "canonical_week_start", "rain_mm_sum_t",
                   "tavg_c_mean_t", "rh_mean_t"))
        g = cf[cf["district_std"] == district].sort_values("canonical_week_start")
        ts = records(g)
    cases = []
    if district and disease:
        s = real_obs()
        s = s[(s["district_name"] == district) & (s["disease"] == disease)]
        cases = records(s, ["canonical_week_start", "observed_cases"])
    return {
        "correlations": records(corr),
        "lags": records(lag_sel, ["disease_or_series", "climate_variable", "climate_group", "lag_weeks", "n_obs",
                                  "spearman_rho", "spearman_p", "spearman_fdr_q", "statistical_association"]) if len(lag_sel) else [],
        "climate_series": ts, "case_series": cases,
        "humidity_note": "Relative humidity is available only where a station dataset exists; null values mean not available.",
        "association_warning": "These are statistical associations in aggregated surveillance data. They do not establish causation.",
    }


# ----------------------------------------------------------------------------- hotspots
@router.get("/hotspots")
def hotspots(disease: str = "ALL_DISEASES", period: str = "2023-2025_POOL"):
    g = load("geospatial/global_spatial_autocorrelation_summary.csv")
    g = g[(g["disease"] == disease) & (g["reporting_period"] == period)]
    hs = load("geospatial/district_disease_map_summary.csv")
    hs = hs[(hs["disease"] == disease) & (hs["reporting_period"] == period)]
    cl = load("geospatial/spatial_clusters_summary.csv")
    cl = cl[(cl["disease"] == disease) & (cl["reporting_period"] == period)]
    diseases = sorted(load("geospatial/global_spatial_autocorrelation_summary.csv")["disease"].unique().tolist())
    periods = sorted(load("geospatial/global_spatial_autocorrelation_summary.csv")["reporting_period"].unique().tolist())
    return {
        "disease": disease, "period": period, "available_diseases": diseases, "available_periods": periods,
        "global_statistics": records(g, ["indicator", "spatial_weights", "n_geocoded_districts", "morans_i",
                                         "p_value_permutation", "is_significant_p05", "spatial_pattern"]),
        "districts": records(hs, ["district_name", "administrative_division", "observed_cases", "gi_star_zscore",
                                  "gi_star_p_perm", "gi_star_class", "local_moran_i", "lisa_quadrant",
                                  "is_statistically_significant_hotspot", "hotspot_consensus_label"]),
        "clusters": records(cl, ["cluster_id", "member_districts_count", "member_districts", "max_gi_star_zscore",
                                 "contains_verified_hotspot_p05"]),
        "caution": "Spatial clustering of reported counts does not prove cross-district transmission or causation.",
    }


# ----------------------------------------------------------------------------- scenarios
@router.get("/scenarios")
def scenarios():
    cat = load("intervention_scenarios/intervention_scenario_summary_catalog.csv")
    return {
        "catalog": records(cat, ["scenario_id", "scenario_name", "intervention_type", "selected_disease",
                                 "focal_districts", "scenario_duration_weeks", "intervention_start_week",
                                 "user_assumed_transmission_dampening_factor",
                                 "unmitigated_baseline_cumulative_cases", "scenario_hypothetical_cumulative_cases",
                                 "hypothetical_case_difference_vs_baseline",
                                 "hypothetical_relative_difference_vs_baseline_pct", "model_or_calculation_method",
                                 "governance_disclaimer"]),
        "label": "HYPOTHETICAL SCENARIOS - not observations and not validated predictions of intervention effect.",
    }


@router.get("/scenarios/{scenario_id}/trajectory")
def scenario_trajectory(scenario_id: str):
    t = load("intervention_scenarios/intervention_scenario_weekly_trajectories.csv")
    g = t[t["scenario_id"] == scenario_id]
    if g.empty:
        raise HTTPException(404, "Unknown scenario")
    agg = g.groupby("simulation_week").agg(
        baseline=("unmitigated_baseline_simulated_cases", "sum"),
        scenario=("hypothetical_surveillance_reported_cases", "sum")).reset_index()
    return {"scenario_id": scenario_id, "weeks": records(agg), "label": "SIMULATED"}


@router.get("/scenario-calc")
def scenario_calc(district: str, disease: str, weeks: int = Query(12, ge=2, le=26),
                  weekly_growth: float = Query(1.05, ge=0.8, le=1.5),
                  dampening: float = Query(0.25, ge=0.0, le=0.9),
                  start_week: int = Query(3, ge=1, le=26), ramp_weeks: int = Query(2, ge=0, le=8),
                  anchor_weeks: int = Query(4, ge=1, le=8)):
    """Transparent calculator: all growth/dampening values are USER ASSUMPTIONS, not estimates."""
    s = real_obs()
    s = s[(s["district_name"] == district) & (s["disease"] == disease)].sort_values("reporting_period")
    if s.empty:
        return {"status": "INSUFFICIENT_DATA", "reason": "No observed records for this district-disease pair."}
    anchor = float(s["observed_cases"].tail(anchor_weeks).mean())
    base, scen = [], []
    b = sc = anchor
    for w in range(1, weeks + 1):
        b = b * weekly_growth
        if w >= start_week:
            ramp = 1.0 if ramp_weeks == 0 else min(1.0, (w - start_week + 1) / (ramp_weeks + 1))
            g_eff = 1 + (weekly_growth - 1) * (1 - dampening * ramp) if weekly_growth > 1 else weekly_growth
        else:
            g_eff = weekly_growth
        sc = sc * g_eff
        base.append(round(b, 2)); scen.append(round(sc, 2))
    return {
        "status": "OK", "label": "HYPOTHETICAL SCENARIO - SIMULATED, NOT OBSERVED",
        "anchor_mean_weekly_cases": round(anchor, 2),
        "anchor_definition": f"Mean of the last {anchor_weeks} observed outbreak-week records for this pair.",
        "weeks": [{"week": i + 1, "baseline": base[i], "scenario": scen[i]} for i in range(weeks)],
        "cumulative_baseline": round(sum(base), 1), "cumulative_scenario": round(sum(scen), 1),
        "hypothetical_difference": round(sum(base) - sum(scen), 1),
        "assumptions": {"weekly_growth_multiplier": weekly_growth, "transmission_dampening": dampening,
                        "intervention_start_week": start_week, "ramp_weeks": ramp_weeks},
        "disclaimer": "Parameters are user-chosen assumptions. No empirical intervention-effectiveness data exist in the workspace.",
    }


# ----------------------------------------------------------------------------- resources
@router.get("/resources")
def resources():
    audit = load("resource_intelligence/resource_data_completeness_audit.csv")
    summ = load("resource_intelligence/district_resource_summary.csv")
    res_rows = audit[~audit["domain_category"].str.contains("Epidemiological|Upstream", case=False, na=False)]
    any_present = bool(res_rows["present_in_workspace"].astype(str).str.lower().eq("true").any()) if len(res_rows) else False
    return {
        "empirical_resource_data_available": any_present,
        "audit": records(audit, ["variable_label", "domain_category", "present_in_workspace", "district_coverage",
                                 "missingness_pct", "scientific_safeguard"]),
        "district_demand": records(summ, ["district_name", "observed_2023_2025_total_outbreak_cases",
                                         "observed_2023_2025_peak_weekly_cases", "upstream_phase6_forecast_7d_mean_cases",
                                         "empirical_available_beds", "empirical_resource_shortage_status",
                                         "data_completeness_status"]),
        "message": ("No hospital bed, ICU, medicine or testing-capacity dataset exists in the workspace. "
                    "Capacity is shown as Not Available; only epidemiological demand indicators are listed."),
    }


# ----------------------------------------------------------------------------- explain / fusion
@router.get("/explain")
def explain(district: str):
    p = load("explanations/why_district_flagged_panels.csv")
    g = p[p["district"] == district]
    if g.empty:
        return {"status": "INSUFFICIENT_DATA", "district": district}
    return {"status": "OK", "panel": records(g)[0]}


@router.get("/explain/flagged")
def explain_flagged():
    p = load("explanations/why_district_flagged_panels.csv")
    return {"districts": records(p, ["district", "is_flagged", "flag_priority_tier"])}


@router.get("/fusion")
def fusion(district: Optional[str] = None):
    pooled = load("fused_intelligence/fused_district_pooled_intelligence_36.csv")
    cols = ["district_id", "district_name", "administrative_division", "overall_signal_status",
            "recommended_review_priority", "automated_outbreak_declaration", "disagreement_flag", "disagreement_type",
            "risk_available", "risk_calibrated_probability", "risk_category", "risk_flagged_elevated",
            "forecast_available", "forecast_7d_cases", "forecast_14d_cases", "forecast_30d_cases",
            "anomaly_available", "anomaly_flagged", "anomaly_composite_score", "anomaly_severity_tier",
            "climate_available", "weekly_rainfall_mm", "prev_4w_rainfall_accum_mm", "weekly_mean_temp_c",
            "district_rain_4w_spearman_rho", "geospatial_available", "gi_star_zscore", "gi_star_class",
            "is_statistically_significant_hotspot", "empirical_resource_data_available", "resource_shortage_status",
            "severity_available", "verified_outbreak_cases", "verified_outbreak_deaths", "outbreak_cluster_cfr_pct",
            "supporting_analytical_evidence", "conflicting_evidence", "uncertainty_and_limitations",
            "why_this_result_explanation", "model_versions_json", "data_quality_status", "governance_disclaimer"]
    q = load("fused_intelligence/engine_disagreement_review_queue.csv")
    result = {
        "districts": records(pooled, cols),
        "disagreement_queue_size": int(len(q)),
        "automated_outbreak_declarations": int(pooled["automated_outbreak_declaration"].astype(str).str.lower().eq("true").sum()),
    }
    if district:
        result["districts"] = [r for r in result["districts"] if r["district_name"] == district]
    return result


# ----------------------------------------------------------------------------- models
@router.get("/models")
def models():
    reg = registry()
    out = []
    for m in reg.get("models", []):
        out.append({k: m.get(k) for k in ("model_id", "model_name", "engine_name", "model_type", "disease",
                                            "supported_districts_count", "unsupported_districts", "training_period",
                                            "validation_period", "testing_period", "model_version", "evaluation_metrics",
                                            "model_status", "status_justification", "production_readiness_disclaimer",
                                            "artifact_path", "artifact_exists", "dataset_version", "feature_version")})
    return {"registry_version": reg.get("registry_version"), "generated": reg.get("generated_timestamp"),
            "status_distribution": reg.get("status_distribution", {}), "models": out}


# ----------------------------------------------------------------------------- data quality
@router.get("/quality")
def quality():
    inv = load("reports/dataset_inventory.csv")
    cov = load("reports/district_coverage_report.csv")
    miss = load("reports/missing_data_report.csv")
    miss = miss[miss["n_missing"] > 0].sort_values("pct_missing", ascending=False).head(40)
    pre = load("reports/preprocessing_log.csv")
    outc = load("severity_intelligence/outcome_data_availability_audit.csv")
    with get_db_connection() as conn:
        src = [dict(r) for r in conn.execute("SELECT * FROM data_sources").fetchall()]
        meta = {r["key"]: {"value": r["value"], "updated_at": r["updated_at"]}
                for r in conn.execute("SELECT * FROM system_metadata").fetchall()}
        ing = [dict(r) for r in conn.execute(
            "SELECT ingestion_id, filename, status, stage_completed, records_total, new_records_inserted, completed_at "
            "FROM ingestion_audit_logs ORDER BY id DESC LIMIT 15").fetchall()]
    reg = registry()
    lineage = [{"model_id": m["model_id"], "model_name": m["model_name"], "dataset_version": m.get("dataset_version"),
                "datasets": m.get("dataset_filenames")} for m in reg.get("models", [])]
    return {
        "sources_registered": src,
        "dataset_inventory": records(inv, ["filename", "original_source", "rows", "columns", "role", "notes"]),
        "district_coverage": records(cov, ["district", "hist_records", "cont_records", "cont_first_report_week",
                                           "cont_years_present", "cwprs_humidity_station", "notes"]),
        "missing_data": records(miss),
        "outcome_variable_availability": records(outc),
        "preprocessing_steps": records(pre, ["step_id", "dataset", "step", "action", "rows_before", "rows_after", "detail"]),
        "ingestion_history": ing, "system_metadata": meta, "model_lineage": lineage,
    }

@router.get("/district/{district}")
def district_profile(district: str):
    obs = real_obs()
    g = obs[obs["district_name"] == district]
    pt = next((p for p in points() if p["district_name"] == district), None)
    if pt is None:
        raise HTTPException(404, "Unknown district")
    if g.empty:
        return {"district": pt, "status": "INSUFFICIENT_DATA", "diseases": [],
                "message": "No surveillance outbreak records exist for this district in 2023-2025."}
    d = g.groupby("disease").agg(cases=("observed_cases", "sum"), deaths=("observed_deaths", "sum"),
                                 outbreak_weeks=("observed_cases", "size"),
                                 flagged_weeks=("is_anomaly", "sum"),
                                 first=("reporting_period", "min"), last=("reporting_period", "max")).reset_index()
    d = d.sort_values("cases", ascending=False)
    return {"district": pt, "status": "OK", "diseases": records(d),
            "total_cases": float(g["observed_cases"].sum()), "total_deaths": float(g["observed_deaths"].sum())}


@router.get("/talukas")
def district_talukas(district: str = Query("Pune", description="District Name")):
    """Returns taluka-level surveillance intelligence for District Health Officers (DHO)."""
    from backend.services.taluka_service import get_district_taluka_intelligence
    return get_district_taluka_intelligence(district)


