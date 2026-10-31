"""Geospatial Disease Mapping, Multi-Engine Layer Integration, and Hotspot Calculation (Phase 9).

Integrates strictly separated layers across all 36 Maharashtra districts:
1. Observed case counts (`2023-2025` IDSP weekly surveillance).
2. Disease incidence rates (`NaN` — explicitly flagged as unavailable without population denominator).
3. Historical averages (`2009-2022` EpiClim historical baseline + `2023-2025` expanding baseline).
4. Forecast values from Prompt 6 (`7d`, `14d`, `30d` horizons — never mixed with observed cases).
5. Anomaly indicators from Prompt 7 (`anomaly_score`, `anomaly_severity_tier`, `signal_types`).
6. Model-estimated outbreak risk from Prompt 5 (`calibrated_probability`, `risk_category`).
7. Spatial autocorrelation & hotspot statistics (`Global Moran's I`, `Local Moran's I LISA`, `Getis-Ord Gi*`).

Districts with insufficient information (`Mumbai Suburban` missing coordinates; `Mumbai City` and
`Ratnagiri` having 0 reported outbreaks in 2023-2025; or diseases with <5 active districts) are
explicitly marked `"Insufficient Data"`.
"""
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.engines.geospatial.spatial_statistics import (
    NON_TRANSMISSION_DISCLAIMER,
    SpatialWeightsBundle,
    compute_global_morans_i,
    compute_local_morans_lisa_and_gi_star,
    identify_connected_spatial_clusters,
)
from src.features.geographic_features import (
    ADMIN_DIVISION_MAP,
    DISTRICT_ID_MAP,
    OFFICIAL_DISTRICTS,
)
from src.risk_classification.predictor import OutbreakRiskPredictor

PROJ = Path(__file__).resolve().parents[3]


def _score_all_phase5_risk_records(proj_dir: Path = PROJ) -> pd.DataFrame:
    """Run Phase 5 OutbreakRiskPredictor across risk_classification_dataset.csv to obtain
    strictly separated risk probabilities and categories per (district_std, disease_std, reporting_period)."""
    risk_ds_path = proj_dir / "data" / "final_training" / "risk_classification_dataset.csv"
    risk_df = pd.read_csv(risk_ds_path)
    predictor = OutbreakRiskPredictor(proj_dir / "models" / "risk_classification")
    preds = predictor.predict_dataframe(risk_df)

    rows = []
    for idx, p in enumerate(preds):
        r_row = risk_df.iloc[idx]
        rows.append(
            {
                "district_std": str(r_row["district_std"]),
                "disease_std": str(r_row["disease_std"]),
                "report_year": int(r_row["report_year"]),
                "report_week": int(r_row["report_week"]),
                "reporting_period": f"{int(r_row['report_year'])}-W{int(r_row['report_week']):02d}",
                "risk_support_status": str(p.get("support_status", "UNSUPPORTED")),
                "model_estimated_risk": (
                    float(p["model_estimated_risk"])
                    if p.get("model_estimated_risk") is not None
                    else np.nan
                ),
                "calibrated_risk_probability": (
                    float(p["calibrated_probability"])
                    if p.get("calibrated_probability") is not None
                    else np.nan
                ),
                "risk_category": str(p.get("risk_category") or "Insufficient Data"),
                "risk_model_version": str(p.get("model_version", "phase5-v1.0.0")),
            }
        )
    return pd.DataFrame(rows)


def build_weekly_period_map_records(
    crosswalk_df: pd.DataFrame, proj_dir: Path = PROJ
) -> pd.DataFrame:
    """Build weekly map-ready records combining observed counts, historical baseline,
    Phase 6 forecasts, Phase 7 anomalies, and Phase 5 risk estimates while keeping all layers separate.
    Also appends explicit 'Insufficient Data' sentinel records for districts lacking 2023-2025 data
    or coordinates so all 36 Maharashtra districts are represented in the map table.
    """
    weekly_feat = pd.read_csv(proj_dir / "data" / "features" / "disease_climate_weekly_features.csv")
    weekly_feat["reporting_period"] = (
        weekly_feat["report_year"].astype(int).astype(str)
        + "-W"
        + weekly_feat["report_week"].astype(int).map(lambda w: f"{w:02d}")
    )

    # Historical 2009-2022 district x disease mean baseline
    cleaned_dis = pd.read_csv(proj_dir / "data" / "processed" / "cleaned_disease_data.csv")
    hist_df = cleaned_dis[cleaned_dis["source_dataset"] == "historical_epiclim"]
    hist_dd_mean = (
        hist_df.groupby(["district_std", "disease_std"])["cases"]
        .mean()
        .round(2)
        .to_dict()
    )

    # Phase 6 forecasts (pivot 7d, 14d, 30d horizons per district_std, disease_std, report_year, report_week)
    fc_df = pd.read_csv(proj_dir / "reports" / "actual_vs_predicted.csv")
    fc_7d = fc_df[fc_df["horizon"] == "7d"].set_index(
        ["district_std", "disease_std", "report_year", "report_week"]
    )
    fc_14d = fc_df[fc_df["horizon"] == "14d"].set_index(
        ["district_std", "disease_std", "report_year", "report_week"]
    )
    fc_30d = fc_df[fc_df["horizon"] == "30d"].set_index(
        ["district_std", "disease_std", "report_year", "report_week"]
    )

    # Phase 7 anomalies
    anom_df = pd.read_csv(proj_dir / "reports" / "anomaly_detection_results.csv")
    anom_lookup = anom_df.set_index(["district", "disease", "reporting_period"]).to_dict(orient="index")

    # Phase 5 risk estimates
    risk_df = _score_all_phase5_risk_records(proj_dir)
    risk_lookup = risk_df.set_index(["district_std", "disease_std", "reporting_period"]).to_dict(
        orient="index"
    )

    cw_lookup = crosswalk_df.set_index("district_name").to_dict(orient="index")

    records: List[Dict[str, Any]] = []
    active_districts = set()

    for _, row in weekly_feat.sort_values(["report_year", "report_week", "district_std", "disease_std"]).iterrows():
        d = str(row["district_std"])
        dis = str(row["disease_std"])
        yr = int(row["report_year"])
        wk = int(row["report_week"])
        rp = str(row["reporting_period"])
        active_districts.add(d)

        cw = cw_lookup[d]
        key_fc = (d, dis, yr, wk)
        f7 = fc_7d.loc[key_fc] if key_fc in fc_7d.index else None
        f14 = fc_14d.loc[key_fc] if key_fc in fc_14d.index else None
        f30 = fc_30d.loc[key_fc] if key_fc in fc_30d.index else None

        if isinstance(f7, pd.DataFrame):
            f7 = f7.iloc[0]
        if isinstance(f14, pd.DataFrame):
            f14 = f14.iloc[0]
        if isinstance(f30, pd.DataFrame):
            f30 = f30.iloc[0]

        anom = anom_lookup.get((d, dis, rp), {})
        risk = risk_lookup.get((d, dis, rp), {})

        has_coords = bool(cw["has_verified_coordinates"])
        dq_status = (
            "COMPLETE_GEOCODED_SURVEILLANCE_RECORD"
            if has_coords
            else "Insufficient Data (Missing District Coordinates for Spatial Mapping)"
        )

        records.append(
            {
                "district_name": d,
                "district_id": DISTRICT_ID_MAP[d],
                "administrative_division": ADMIN_DIVISION_MAP[d],
                "latitude": cw["latitude"],
                "longitude": cw["longitude"],
                "disease": dis,
                "reporting_period": rp,
                "report_year": yr,
                "report_week": wk,
                "canonical_week_start": str(row["canonical_week_start"]),
                # Layer 1: Observed surveillance counts (NEVER mixed with forecasts)
                "observed_cases": float(row["target_cases"]),
                "observed_deaths": float(row["target_deaths"]) if pd.notna(row["target_deaths"]) else np.nan,
                "observed_outbreak_clusters": int(row["outbreak_events_in_period"]),
                # Layer 2: Incidence rate (explicitly unavailable without population denominator)
                "incidence_rate": np.nan,
                "incidence_rate_status": "UNAVAILABLE_NO_POPULATION_DENOMINATOR",
                # Layer 3: Historical averages
                "historical_expanding_mean_cases_2023_2025": (
                    round(float(row["district_disease_prior_mean_cases"]), 2)
                    if pd.notna(row.get("district_disease_prior_mean_cases"))
                    else np.nan
                ),
                "historical_2009_2022_reference_mean_cases": hist_dd_mean.get((d, dis), np.nan),
                # Layer 4: Phase 6 Forecasts (strictly separate from observed_cases)
                "forecast_value": round(float(f7["predicted_cases"]), 2) if f7 is not None else np.nan,
                "forecast_7d_cases": round(float(f7["predicted_cases"]), 2) if f7 is not None else np.nan,
                "forecast_7d_lower_80": round(float(f7["prediction_lower_80"]), 2) if f7 is not None else np.nan,
                "forecast_7d_upper_80": round(float(f7["prediction_upper_80"]), 2) if f7 is not None else np.nan,
                "forecast_14d_cases": round(float(f14["predicted_cases"]), 2) if f14 is not None else np.nan,
                "forecast_30d_cases": round(float(f30["predicted_cases"]), 2) if f30 is not None else np.nan,
                "forecast_record_type": "MODEL_FORECAST_SEPARATE_FROM_OBSERVED" if f7 is not None else "NO_FORECAST",
                # Layer 5: Phase 7 Anomaly Detection
                "anomaly_status": str(anom.get("anomaly_severity_tier", "Insufficient Data")),
                "is_anomaly": bool(anom.get("is_anomaly", False)),
                "anomaly_score": round(float(anom["anomaly_score"]), 4) if "anomaly_score" in anom else np.nan,
                "anomaly_signal_types": str(anom.get("signal_types", "NONE")),
                # Layer 6: Phase 5 Outbreak Risk Classification
                "model_estimated_risk": risk.get("model_estimated_risk", np.nan),
                "calibrated_risk_probability": risk.get("calibrated_risk_probability", np.nan),
                "risk_category": str(risk.get("risk_category", "Insufficient Data")),
                "risk_support_status": str(risk.get("risk_support_status", "UNSUPPORTED")),
                # Layer 7: Hotspot statistic placeholder at weekly grain (populated from district x disease summary)
                "hotspot_statistic": np.nan,
                "hotspot_status": "Computed at District x Disease Spatial Panel Grain",
                "data_quality_status": dq_status,
                "layer_separation_guard": "STRICTLY_SEPARATED_OBSERVED_FORECAST_ANOMALY_RISK",
            }
        )

    # Append explicit "Insufficient Data" records for the 2 zero-outbreak districts (Mumbai City, Ratnagiri)
    for d in OFFICIAL_DISTRICTS:
        if d not in active_districts:
            cw = cw_lookup[d]
            records.append(
                {
                    "district_name": d,
                    "district_id": DISTRICT_ID_MAP[d],
                    "administrative_division": ADMIN_DIVISION_MAP[d],
                    "latitude": cw["latitude"],
                    "longitude": cw["longitude"],
                    "disease": "ALL_DISEASES",
                    "reporting_period": "2023-2025_NO_REPORTED_OUTBREAKS",
                    "report_year": np.nan,
                    "report_week": np.nan,
                    "canonical_week_start": "",
                    "observed_cases": np.nan,
                    "observed_deaths": np.nan,
                    "observed_outbreak_clusters": 0,
                    "incidence_rate": np.nan,
                    "incidence_rate_status": "UNAVAILABLE_NO_POPULATION_DENOMINATOR",
                    "historical_expanding_mean_cases_2023_2025": np.nan,
                    "historical_2009_2022_reference_mean_cases": np.nan,
                    "forecast_value": np.nan,
                    "forecast_7d_cases": np.nan,
                    "forecast_7d_lower_80": np.nan,
                    "forecast_7d_upper_80": np.nan,
                    "forecast_14d_cases": np.nan,
                    "forecast_30d_cases": np.nan,
                    "forecast_record_type": "INSUFFICIENT_DATA",
                    "anomaly_status": "Insufficient Data",
                    "is_anomaly": False,
                    "anomaly_score": np.nan,
                    "anomaly_signal_types": "Insufficient Data",
                    "model_estimated_risk": np.nan,
                    "calibrated_risk_probability": np.nan,
                    "risk_category": "Insufficient Data",
                    "risk_support_status": "UNSUPPORTED_ZERO_RECORDS_IN_2023_2025",
                    "hotspot_statistic": np.nan,
                    "hotspot_status": "Insufficient Data",
                    "data_quality_status": "Insufficient Data",
                    "layer_separation_guard": "STRICTLY_SEPARATED_OBSERVED_FORECAST_ANOMALY_RISK",
                }
            )

    return pd.DataFrame(records)


def compute_spatial_hotspot_summaries(
    weekly_map_df: pd.DataFrame,
    crosswalk_df: pd.DataFrame,
    weights_bundle: SpatialWeightsBundle,
    proj_dir: Path = PROJ,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute district-level disease mapping and spatial hotspot statistics (Global Moran's I,
    Local Moran's I LISA, Getis-Ord Gi*, and connected spatial clusters) across all 36 districts.

    Slices analysed:
    1. `ALL_DISEASES` (`2023-2025_POOL`, `2023`, `2024`, `2025`, and `2009-2022_HISTORICAL_BASELINE`).
    2. All 12 supported individual diseases (`2023-2025_POOL`) with >= 8 records / >= 5 active districts.
    3. Rare/unsupported diseases (`< 5` active districts) explicitly included and marked `"Insufficient Data"`.
    """
    cleaned_dis = pd.read_csv(proj_dir / "data" / "processed" / "cleaned_disease_data.csv")
    hist_df = cleaned_dis[cleaned_dis["source_dataset"] == "historical_epiclim"]
    active_wk = weekly_map_df[weekly_map_df["observed_cases"].notna()].copy()

    # Held-out test forecasts per district & disease
    fc_df = pd.read_csv(proj_dir / "reports" / "actual_vs_predicted.csv")
    fc_test_7d = fc_df[(fc_df["split"] == "test") & (fc_df["horizon"] == "7d")]
    fc_all_7d = fc_df[fc_df["horizon"] == "7d"]
    fc_all_14d = fc_df[fc_df["horizon"] == "14d"]
    fc_all_30d = fc_df[fc_df["horizon"] == "30d"]

    cw_lookup = crosswalk_df.set_index("district_name").to_dict(orient="index")

    # Define analysis slices: (disease_label, period_label, sub_df_active, is_historical_slice)
    slices: List[Tuple[str, str, pd.DataFrame, bool]] = [
        ("ALL_DISEASES", "2023-2025_POOL", active_wk, False),
        ("ALL_DISEASES", "2023", active_wk[active_wk["report_year"] == 2023], False),
        ("ALL_DISEASES", "2024", active_wk[active_wk["report_year"] == 2024], False),
        ("ALL_DISEASES", "2025", active_wk[active_wk["report_year"] == 2025], False),
        ("ALL_DISEASES", "2009-2022_HISTORICAL_BASELINE", hist_df, True),
    ]

    # Individual diseases in 2023-2025
    all_diseases_23_25 = sorted(active_wk["disease"].unique())
    for dis in all_diseases_23_25:
        slices.append((dis, "2023-2025_POOL", active_wk[active_wk["disease"] == dis], False))

    summary_rows: List[Dict[str, Any]] = []
    global_moran_rows: List[Dict[str, Any]] = []
    cluster_rows: List[Dict[str, Any]] = []

    for dis_label, period_label, sub_df, is_hist in slices:
        # Aggregate per district across all 36 official districts
        if is_hist:
            dist_cases = sub_df.groupby("district_std")["cases"].sum().to_dict()
            dist_deaths = sub_df.groupby("district_std")["deaths"].sum(min_count=1).to_dict()
            dist_weeks = sub_df.groupby("district_std").size().to_dict()
            dist_mean_cases = sub_df.groupby("district_std")["cases"].mean().to_dict()
        else:
            dist_cases = sub_df.groupby("district_name")["observed_cases"].sum().to_dict()
            dist_deaths = sub_df.groupby("district_name")["observed_deaths"].sum(min_count=1).to_dict()
            dist_weeks = sub_df.groupby("district_name").size().to_dict()
            dist_mean_cases = sub_df.groupby("district_name")["observed_cases"].mean().to_dict()

        # Historical 2009-2022 reference for this disease
        if dis_label == "ALL_DISEASES":
            hist_ref_sum = hist_df.groupby("district_std")["cases"].sum().to_dict()
            hist_ref_mean = hist_df.groupby("district_std")["cases"].mean().to_dict()
        else:
            h_sub = hist_df[hist_df["disease_std"] == dis_label]
            hist_ref_sum = h_sub.groupby("district_std")["cases"].sum().to_dict()
            hist_ref_mean = h_sub.groupby("district_std")["cases"].mean().to_dict()

        # Forecast, Anomaly, and Risk aggregations (only for 2023-2025 slices)
        if not is_hist:
            if dis_label == "ALL_DISEASES":
                f7_sub = fc_all_7d if period_label == "2023-2025_POOL" else fc_all_7d[fc_all_7d["report_year"] == int(period_label)]
                f14_sub = fc_all_14d if period_label == "2023-2025_POOL" else fc_all_14d[fc_all_14d["report_year"] == int(period_label)]
                f30_sub = fc_all_30d if period_label == "2023-2025_POOL" else fc_all_30d[fc_all_30d["report_year"] == int(period_label)]
                ftest_sub = fc_test_7d
            else:
                f7_sub = fc_all_7d[fc_all_7d["disease_std"] == dis_label]
                f14_sub = fc_all_14d[fc_all_14d["disease_std"] == dis_label]
                f30_sub = fc_all_30d[fc_all_30d["disease_std"] == dis_label]
                ftest_sub = fc_test_7d[fc_test_7d["disease_std"] == dis_label]

            fc7_mean = f7_sub.groupby("district_std")["predicted_cases"].mean().to_dict()
            fc14_mean = f14_sub.groupby("district_std")["predicted_cases"].mean().to_dict()
            fc30_mean = f30_sub.groupby("district_std")["predicted_cases"].mean().to_dict()
            fctest_sum = ftest_sub.groupby("district_std")["predicted_cases"].sum().to_dict()

            anom_cnt = sub_df.groupby("district_name")["is_anomaly"].sum().to_dict()
            anom_max = sub_df.groupby("district_name")["anomaly_score"].max().to_dict()
            risk_mean = sub_df.groupby("district_name")["calibrated_risk_probability"].mean().to_dict()
            risk_max = sub_df.groupby("district_name")["calibrated_risk_probability"].max().to_dict()
        else:
            fc7_mean, fc14_mean, fc30_mean, fctest_sum = {}, {}, {}, {}
            anom_cnt, anom_max, risk_mean, risk_max = {}, {}, {}, {}

        n_active_geocoded = sum(
            1 for d in weights_bundle.districts if dist_weeks.get(d, 0) > 0
        )
        spatial_supported = bool(n_active_geocoded >= 5)

        if spatial_supported:
            # Build vector over the 35 geocoded districts
            # Districts with 0 reported outbreaks for a specific disease have 0 reported cases in IDSP,
            # while Mumbai City & Ratnagiri are flagged in data_quality_status as Insufficient Data.
            vec_cases = np.array(
                [float(dist_cases.get(d, 0.0)) for d in weights_bundle.districts], dtype=float
            )
            vec_log = np.log1p(vec_cases)

            gm_raw = compute_global_morans_i(vec_cases, weights_bundle.W_row_std, n_permutations=999)
            gm_log = compute_global_morans_i(vec_log, weights_bundle.W_row_std, n_permutations=999)
            gm_idw = compute_global_morans_i(vec_cases, weights_bundle.W_idw_row_std, n_permutations=999)

            global_moran_rows.append(
                {
                    "disease": dis_label,
                    "reporting_period": period_label,
                    "indicator": "observed_cases_raw",
                    "spatial_weights": "W_knn4_symmetric_row_standardized",
                    "n_geocoded_districts": len(weights_bundle.districts),
                    "n_active_districts": n_active_geocoded,
                    **gm_raw,
                    "morans_i_log1p_sensitivity": gm_log["morans_i"],
                    "p_perm_log1p_sensitivity": gm_log["p_value_permutation"],
                    "morans_i_idw220km_sensitivity": gm_idw["morans_i"],
                    "p_perm_idw220km_sensitivity": gm_idw["p_value_permutation"],
                    "scientific_note": NON_TRANSMISSION_DISCLAIMER,
                }
            )

            lisa_gi_df = compute_local_morans_lisa_and_gi_star(
                vec_cases, weights_bundle, n_permutations=999
            )
            lisa_lookup = lisa_gi_df.set_index("district_name").to_dict(orient="index")

            clusters = identify_connected_spatial_clusters(lisa_gi_df, weights_bundle)
            for c in clusters:
                cluster_rows.append(
                    {
                        "disease": dis_label,
                        "reporting_period": period_label,
                        **c,
                    }
                )
        else:
            lisa_lookup = {}

        # Also evaluate Global Moran's I on forecasts, anomalies, and risk for ALL_DISEASES 2023-2025_POOL
        if dis_label == "ALL_DISEASES" and period_label == "2023-2025_POOL":
            for ind_name, ind_dict in [
                ("forecast_7d_mean_cases", fc7_mean),
                ("anomaly_weeks_count", anom_cnt),
                ("anomaly_max_score", anom_max),
                ("calibrated_risk_probability_mean", risk_mean),
            ]:
                v_ind = np.array(
                    [float(ind_dict.get(d, 0.0)) if pd.notna(ind_dict.get(d, 0.0)) else 0.0 for d in weights_bundle.districts],
                    dtype=float,
                )
                gm_ind = compute_global_morans_i(v_ind, weights_bundle.W_row_std, n_permutations=999)
                gm_ind_idw = compute_global_morans_i(v_ind, weights_bundle.W_idw_row_std, n_permutations=999)
                global_moran_rows.append(
                    {
                        "disease": "ALL_DISEASES",
                        "reporting_period": "2023-2025_POOL",
                        "indicator": ind_name,
                        "spatial_weights": "W_knn4_symmetric_row_standardized",
                        "n_geocoded_districts": len(weights_bundle.districts),
                        "n_active_districts": sum(1 for v in v_ind if v > 0),
                        **gm_ind,
                        "morans_i_log1p_sensitivity": np.nan,
                        "p_perm_log1p_sensitivity": np.nan,
                        "morans_i_idw220km_sensitivity": gm_ind_idw["morans_i"],
                        "p_perm_idw220km_sensitivity": gm_ind_idw["p_value_permutation"],
                        "scientific_note": NON_TRANSMISSION_DISCLAIMER,
                    }
                )

        # Assemble 36-district rows for this slice
        for d in OFFICIAL_DISTRICTS:
            cw = cw_lookup[d]
            has_coords = bool(cw["has_verified_coordinates"])
            n_wk = int(dist_weeks.get(d, 0))
            obs_c = float(dist_cases[d]) if d in dist_cases else np.nan
            obs_d = float(dist_deaths[d]) if d in dist_deaths and pd.notna(dist_deaths[d]) else np.nan
            mean_c = round(float(dist_mean_cases[d]), 2) if d in dist_mean_cases else np.nan

            # Determine Data Quality Status ("Insufficient Data" when lacking coordinates, 0 total records, or <5 active districts)
            if not has_coords:
                dq_status = "Insufficient Data (Missing Coordinates)"
            elif not is_hist and int(cw["idsp_2023_2025_dedup_outbreak_weeks"]) == 0:
                dq_status = "Insufficient Data (0 Surveillance Outbreaks in 2023-2025)"
            elif not spatial_supported:
                dq_status = "Insufficient Data (<5 Active Districts for Spatial Hotspot Test)"
            elif n_wk == 0:
                dq_status = "Zero Reported Outbreaks for Slice (Geocoded)"
            else:
                dq_status = "Supported (Complete Geocoded Surveillance)"

            lg = lisa_lookup.get(d, {})
            if not has_coords or not spatial_supported or (not is_hist and int(cw["idsp_2023_2025_dedup_outbreak_weeks"]) == 0):
                gi_z = np.nan
                gi_p = np.nan
                gi_cls = "Insufficient Data"
                lm_i = np.nan
                lm_p = np.nan
                lisa_q = "Insufficient Data"
                is_hot = False
                hot_label = "Insufficient Data"
            else:
                gi_z = lg.get("gi_star_zscore", np.nan)
                gi_p = lg.get("gi_star_p_perm", np.nan)
                gi_cls = str(lg.get("gi_star_class", "NOT_SIGNIFICANT"))
                lm_i = lg.get("local_moran_i", np.nan)
                lm_p = lg.get("local_moran_p_perm", np.nan)
                lisa_q = str(lg.get("lisa_quadrant", "NOT_SIGNIFICANT"))
                is_hot = bool(lg.get("is_statistically_significant_hotspot", False))
                hot_label = str(lg.get("hotspot_consensus_label", "Not a Statistically Significant Hotspot"))

            # Anomaly status summary
            if is_hist or dq_status.startswith("Insufficient Data"):
                a_status = "Insufficient Data"
            elif int(anom_cnt.get(d, 0)) > 0:
                a_status = f"ANOMALY_DETECTED ({int(anom_cnt[d])} week(s), max_score={anom_max.get(d, 0.0):.2f})"
            elif n_wk > 0:
                a_status = "NORMAL_BASELINE (0 Anomalies)"
            else:
                a_status = "Insufficient Data"

            # Risk category summary
            r_prob = risk_mean.get(d, np.nan)
            r_max = risk_max.get(d, np.nan)
            if pd.isna(r_prob) or dq_status.startswith("Insufficient Data"):
                r_cat = "Insufficient Data"
            elif r_max >= 0.55:
                r_cat = "HIGH_RISK"
            elif r_max >= 0.25:
                r_cat = "MODERATE_RISK"
            else:
                r_cat = "LOW_RISK"

            summary_rows.append(
                {
                    "district_name": d,
                    "district_id": DISTRICT_ID_MAP[d],
                    "administrative_division": ADMIN_DIVISION_MAP[d],
                    "latitude": cw["latitude"],
                    "longitude": cw["longitude"],
                    "disease": dis_label,
                    "reporting_period": period_label,
                    # Layer 1: Observed cases
                    "observed_outbreak_weeks": n_wk,
                    "observed_cases": obs_c,
                    "observed_deaths": obs_d,
                    "observed_mean_cases_per_outbreak_week": mean_c,
                    # Layer 2: Incidence rate (explicitly NaN)
                    "incidence_rate": np.nan,
                    "incidence_rate_status": "UNAVAILABLE_NO_POPULATION_DENOMINATOR",
                    # Layer 3: Historical averages
                    "historical_2009_2022_total_cases": float(hist_ref_sum[d]) if d in hist_ref_sum else np.nan,
                    "historical_2009_2022_mean_cases": round(float(hist_ref_mean[d]), 2) if d in hist_ref_mean else np.nan,
                    # Layer 4: Phase 6 Forecasts (strictly separate from observed_cases)
                    "forecast_value": round(float(fc7_mean[d]), 2) if d in fc7_mean and pd.notna(fc7_mean[d]) else np.nan,
                    "forecast_7d_mean_cases": round(float(fc7_mean[d]), 2) if d in fc7_mean and pd.notna(fc7_mean[d]) else np.nan,
                    "forecast_14d_mean_cases": round(float(fc14_mean[d]), 2) if d in fc14_mean and pd.notna(fc14_mean[d]) else np.nan,
                    "forecast_30d_mean_cases": round(float(fc30_mean[d]), 2) if d in fc30_mean and pd.notna(fc30_mean[d]) else np.nan,
                    "forecast_heldout_test_7d_total_cases": round(float(fctest_sum[d]), 2) if d in fctest_sum and pd.notna(fctest_sum[d]) else np.nan,
                    # Layer 5: Phase 7 Anomaly indicators
                    "anomaly_status": a_status,
                    "anomaly_weeks_count": int(anom_cnt.get(d, 0)) if not is_hist else np.nan,
                    "anomaly_max_score": round(float(anom_max[d]), 4) if d in anom_max and pd.notna(anom_max[d]) else np.nan,
                    # Layer 6: Phase 5 Outbreak Risk indicators
                    "model_estimated_risk_mean": round(float(r_prob), 4) if pd.notna(r_prob) else np.nan,
                    "model_estimated_risk_max": round(float(r_max), 4) if pd.notna(r_max) else np.nan,
                    "risk_category": r_cat,
                    # Layer 7: Hotspot statistics (Getis-Ord Gi* & Local Moran's I LISA)
                    "hotspot_statistic": gi_z,
                    "gi_star_zscore": gi_z,
                    "gi_star_p_perm": gi_p,
                    "gi_star_class": gi_cls,
                    "local_moran_i": lm_i,
                    "local_moran_p_perm": lm_p,
                    "lisa_quadrant": lisa_q,
                    "is_statistically_significant_hotspot": is_hot,
                    "hotspot_consensus_label": hot_label,
                    "data_quality_status": dq_status,
                    "non_transmission_disclaimer": NON_TRANSMISSION_DISCLAIMER,
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    global_moran_df = pd.DataFrame(global_moran_rows)
    clusters_df = pd.DataFrame(cluster_rows)

    return summary_df, global_moran_df, clusters_df


def build_district_drilldown_profiles(
    crosswalk_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    weekly_map_df: pd.DataFrame,
    weights_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Build structured district-level drill-down profiles for all 36 Maharashtra districts."""
    profiles: Dict[str, Any] = {}
    w_lookup = weights_df.set_index("district_name").to_dict(orient="index")
    pool_all = summary_df[
        (summary_df["disease"] == "ALL_DISEASES")
        & (summary_df["reporting_period"] == "2023-2025_POOL")
    ].set_index("district_name").to_dict(orient="index")

    for _, cw in crosswalk_df.iterrows():
        d = str(cw["district_name"])
        pa = pool_all.get(d, {})
        w_info = w_lookup.get(d, {})

        # Disease breakdown for this district in 2023-2025
        d_dis = summary_df[
            (summary_df["district_name"] == d)
            & (summary_df["reporting_period"] == "2023-2025_POOL")
            & (summary_df["disease"] != "ALL_DISEASES")
            & (summary_df["observed_outbreak_weeks"] > 0)
        ].sort_values("observed_cases", ascending=False)

        disease_breakdown = []
        for _, r in d_dis.iterrows():
            disease_breakdown.append(
                {
                    "disease": str(r["disease"]),
                    "observed_outbreak_weeks": int(r["observed_outbreak_weeks"]),
                    "observed_cases": float(r["observed_cases"]) if pd.notna(r["observed_cases"]) else None,
                    "observed_deaths": float(r["observed_deaths"]) if pd.notna(r["observed_deaths"]) else None,
                    "historical_2009_2022_mean_cases": (
                        float(r["historical_2009_2022_mean_cases"])
                        if pd.notna(r["historical_2009_2022_mean_cases"])
                        else None
                    ),
                    "forecast_7d_mean_cases": (
                        float(r["forecast_7d_mean_cases"])
                        if pd.notna(r["forecast_7d_mean_cases"])
                        else None
                    ),
                    "anomaly_weeks_count": int(r["anomaly_weeks_count"]) if pd.notna(r["anomaly_weeks_count"]) else 0,
                    "model_estimated_risk_max": (
                        float(r["model_estimated_risk_max"])
                        if pd.notna(r["model_estimated_risk_max"])
                        else None
                    ),
                    "gi_star_zscore": float(r["gi_star_zscore"]) if pd.notna(r["gi_star_zscore"]) else None,
                    "gi_star_class": str(r["gi_star_class"]),
                    "lisa_quadrant": str(r["lisa_quadrant"]),
                    "is_statistically_significant_hotspot": bool(r["is_statistically_significant_hotspot"]),
                    "data_quality_status": str(r["data_quality_status"]),
                }
            )

        # Recent 10 weekly records for drill-down timeline
        d_wk = (
            weekly_map_df[
                (weekly_map_df["district_name"] == d) & (weekly_map_df["observed_cases"].notna())
            ]
            .sort_values(["report_year", "report_week"], ascending=False)
            .head(10)
        )
        recent_weeks = []
        for _, rw in d_wk.iterrows():
            recent_weeks.append(
                {
                    "reporting_period": str(rw["reporting_period"]),
                    "disease": str(rw["disease"]),
                    "observed_cases": float(rw["observed_cases"]),
                    "forecast_7d_cases": (
                        float(rw["forecast_7d_cases"]) if pd.notna(rw["forecast_7d_cases"]) else None
                    ),
                    "anomaly_status": str(rw["anomaly_status"]),
                    "risk_category": str(rw["risk_category"]),
                }
            )

        profiles[d] = {
            "district_id": str(cw["district_id"]),
            "district_name": d,
            "administrative_division": str(cw["administrative_division"]),
            "coordinates": {
                "latitude": float(cw["latitude"]) if pd.notna(cw["latitude"]) else None,
                "longitude": float(cw["longitude"]) if pd.notna(cw["longitude"]) else None,
                "has_verified_coordinates": bool(cw["has_verified_coordinates"]),
            },
            "data_quality_status": str(pa.get("data_quality_status", cw["data_quality_status"])),
            "boundary_and_historical_note": str(cw["boundary_and_historical_note"]),
            "spatial_neighbors": {
                "degree_symmetric_knn": int(w_info.get("degree_symmetric_knn", 0)),
                "neighbor_districts": str(w_info.get("neighbor_district_names", "")),
                "neighbor_distances_km": str(w_info.get("neighbor_distances_km", "")),
            },
            "overall_2023_2025_summary": {
                "observed_outbreak_weeks": int(pa.get("observed_outbreak_weeks", 0)),
                "observed_cases": (
                    float(pa["observed_cases"]) if pd.notna(pa.get("observed_cases")) else None
                ),
                "incidence_rate": None,
                "incidence_rate_status": "UNAVAILABLE_NO_POPULATION_DENOMINATOR",
                "historical_2009_2022_total_cases": (
                    float(pa["historical_2009_2022_total_cases"])
                    if pd.notna(pa.get("historical_2009_2022_total_cases"))
                    else None
                ),
                "forecast_7d_mean_cases": (
                    float(pa["forecast_7d_mean_cases"])
                    if pd.notna(pa.get("forecast_7d_mean_cases"))
                    else None
                ),
                "anomaly_status": str(pa.get("anomaly_status", "Insufficient Data")),
                "risk_category": str(pa.get("risk_category", "Insufficient Data")),
                "gi_star_zscore": (
                    float(pa["gi_star_zscore"]) if pd.notna(pa.get("gi_star_zscore")) else None
                ),
                "gi_star_class": str(pa.get("gi_star_class", "Insufficient Data")),
                "lisa_quadrant": str(pa.get("lisa_quadrant", "Insufficient Data")),
                "is_statistically_significant_hotspot": bool(
                    pa.get("is_statistically_significant_hotspot", False)
                ),
                "hotspot_consensus_label": str(pa.get("hotspot_consensus_label", "Insufficient Data")),
            },
            "disease_breakdown": disease_breakdown,
            "recent_surveillance_weeks": recent_weeks,
            "non_transmission_disclaimer": NON_TRANSMISSION_DISCLAIMER,
        }

    return profiles
