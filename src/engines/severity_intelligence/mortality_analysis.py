"""Disease-Wise, District-Wise, and Temporal/Seasonal Mortality & CFR Analysis (Phase 12).

Implements:
- Compatible Outbreak-Cluster Case Fatality Ratio (`OC-CFR %`) calculation strictly where
  `verified_deaths` and `verified_cases` come from the same deduplicated outbreak strata.
- Exact Wilson score 95% confidence intervals (`[lower_pct, upper_pct]`) and Empirical Bayes
  shrinkage CFR to prevent over-interpreting small denominators ($N < 30$).
- Explicit `NaN` columns for unavailable clinical severity, hospital admissions, complications,
  and recovery counts.
"""
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.features.geographic_features import (
    ADMIN_DIVISION_MAP,
    DISTRICT_ID_MAP,
    OFFICIAL_DISTRICTS,
)


def wilson_score_interval_pct(
    deaths: float, cases: float, z: float = 1.959963984540054
) -> Tuple[float, float]:
    """Compute Wilson score 95% confidence interval (in %) for binomial fatality ratio."""
    if pd.isna(deaths) or pd.isna(cases) or cases <= 0:
        return (np.nan, np.nan)
    n = float(cases)
    k = min(max(float(deaths), 0.0), n)
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half_width = (z * np.sqrt((p * (1.0 - p) / n) + ((z * z) / (4.0 * n * n)))) / denom
    low = max(0.0, (center - half_width) * 100.0)
    high = min(100.0, (center + half_width) * 100.0)
    return (round(low, 4), round(high, 4))


def compute_disease_mortality_summary(weekly_records_df: pd.DataFrame) -> pd.DataFrame:
    """Compute disease-wise mortality trends and Outbreak-Cluster CFR across all 28 diseases (2023-2025)."""
    gen_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    statewide_cases = float(weekly_records_df["verified_cases"].sum())
    statewide_deaths = float(weekly_records_df["verified_deaths"].sum())
    prior_cfr = statewide_deaths / statewide_cases if statewide_cases > 0 else 0.01
    pseudocount_m = 50.0

    rows: List[Dict[str, Any]] = []
    for dis_name, grp in weekly_records_df.groupby("disease"):
        n_wks = len(grp)
        n_dist = int(grp["district"].nunique())
        n_zero = int((grp["verified_deaths"] == 0).sum())
        n_pos = int((grp["verified_deaths"] > 0).sum())
        tot_cases = float(grp["verified_cases"].sum())
        tot_deaths = float(grp["verified_deaths"].sum())
        dis_grp = str(grp["disease_group"].iloc[0])

        crude_cfr_pct = round((tot_deaths / tot_cases) * 100.0, 4) if tot_cases > 0 else np.nan
        ci_low, ci_high = wilson_score_interval_pct(tot_deaths, tot_cases)
        eb_cfr_pct = (
            round(((tot_deaths + pseudocount_m * prior_cfr) / (tot_cases + pseudocount_m)) * 100.0, 4)
            if tot_cases > 0
            else np.nan
        )

        if tot_cases >= 100:
            denom_status = "HIGH_CONFIDENCE_DENOMINATOR_N_GE_100"
        elif tot_cases >= 30:
            denom_status = "MODERATE_DENOMINATOR_N_30_TO_99"
        else:
            denom_status = "SMALL_DENOMINATOR_UNSTABLE_RATIO_N_LT_30"

        rows.append(
            {
                "district": "ALL_MAHARASHTRA_ACTIVE_DISTRICTS",
                "disease": str(dis_name),
                "disease_group": dis_grp,
                "reporting_period": "2023-2025_POOL",
                "outbreak_weeks_count": n_wks,
                "affected_districts_count": n_dist,
                "zero_death_outbreak_weeks": n_zero,
                "positive_death_outbreak_weeks": n_pos,
                "positive_death_weeks_pct": round((n_pos / n_wks) * 100.0, 2),
                "verified_cases": int(tot_cases),
                "verified_deaths": int(tot_deaths),
                "severity_count": np.nan,
                "hospital_admissions_count": np.nan,
                "complications_count": np.nan,
                "recovery_count": np.nan,
                "calculated_ratio": crude_cfr_pct,
                "wilson_95ci_lower_pct": ci_low,
                "wilson_95ci_upper_pct": ci_high,
                "empirical_bayes_shrunk_cfr_pct": eb_cfr_pct,
                "denominator_stability_status": denom_status,
                "mean_deaths_per_outbreak_week": round(float(grp["verified_deaths"].mean()), 4),
                "max_deaths_in_single_outbreak_week": int(grp["verified_deaths"].max()),
                "source": "IDSP Weekly Outbreak Surveillance (2023-2025 Deduplicated)",
                "data_completeness": (
                    "100% Verified Outbreak Cases & Deaths | Clinical Severity/Admissions Unavailable (NaN)"
                ),
                "analysis_method": "Pooled Outbreak-Cluster CFR + Wilson 95% CI + Empirical Bayes Shrinkage (M=50)",
                "generation_timestamp": gen_ts,
            }
        )

    out_df = pd.DataFrame(rows).sort_values(
        ["verified_deaths", "verified_cases"], ascending=[False, False]
    ).reset_index(drop=True)
    return out_df


def compute_district_mortality_summary(weekly_records_df: pd.DataFrame) -> pd.DataFrame:
    """Compute district-wise mortality and severity summary across all 36 official Maharashtra districts."""
    gen_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    statewide_cases = float(weekly_records_df["verified_cases"].sum())
    statewide_deaths = float(weekly_records_df["verified_deaths"].sum())
    prior_cfr = statewide_deaths / statewide_cases if statewide_cases > 0 else 0.01
    pseudocount_m = 50.0

    rows: List[Dict[str, Any]] = []
    for d_name in OFFICIAL_DISTRICTS:
        grp = weekly_records_df[weekly_records_df["district"] == d_name]
        if grp.empty:
            rows.append(
                {
                    "district_id": DISTRICT_ID_MAP[d_name],
                    "district": d_name,
                    "administrative_division": ADMIN_DIVISION_MAP[d_name],
                    "disease": "ALL_DISEASES",
                    "reporting_period": "2023-2025_POOL",
                    "outbreak_weeks_count": 0,
                    "distinct_diseases_count": 0,
                    "zero_death_outbreak_weeks": np.nan,
                    "positive_death_outbreak_weeks": np.nan,
                    "verified_cases": np.nan,
                    "verified_deaths": np.nan,
                    "severity_count": np.nan,
                    "hospital_admissions_count": np.nan,
                    "complications_count": np.nan,
                    "recovery_count": np.nan,
                    "calculated_ratio": np.nan,
                    "wilson_95ci_lower_pct": np.nan,
                    "wilson_95ci_upper_pct": np.nan,
                    "empirical_bayes_shrunk_cfr_pct": np.nan,
                    "mean_deaths_per_outbreak_week": np.nan,
                    "max_deaths_in_single_outbreak_week": np.nan,
                    "dominant_fatal_disease": "Insufficient Data (0 Surveillance Outbreaks in 2023-2025)",
                    "dominant_fatal_disease_deaths": np.nan,
                    "mortality_burden_tier": "Insufficient Data",
                    "source": "IDSP Weekly Outbreak Surveillance (2023-2025 Deduplicated)",
                    "data_completeness": "Insufficient Data (0 Reported Outbreaks in 2023-2025 & No Hospital/Severity Data)",
                    "analysis_method": "Explicit Insufficient Data Preservation (Never Zero-Filled)",
                    "generation_timestamp": gen_ts,
                }
            )
            continue

        n_wks = len(grp)
        n_dis = int(grp["disease"].nunique())
        n_zero = int((grp["verified_deaths"] == 0).sum())
        n_pos = int((grp["verified_deaths"] > 0).sum())
        tot_cases = float(grp["verified_cases"].sum())
        tot_deaths = float(grp["verified_deaths"].sum())

        crude_cfr_pct = round((tot_deaths / tot_cases) * 100.0, 4) if tot_cases > 0 else np.nan
        ci_low, ci_high = wilson_score_interval_pct(tot_deaths, tot_cases)
        eb_cfr_pct = (
            round(((tot_deaths + pseudocount_m * prior_cfr) / (tot_cases + pseudocount_m)) * 100.0, 4)
            if tot_cases > 0
            else np.nan
        )

        by_dis_deaths = grp.groupby("disease")["verified_deaths"].sum().sort_values(ascending=False)
        if tot_deaths > 0:
            dom_fatal_dis = str(by_dis_deaths.index[0])
            dom_fatal_cnt = int(by_dis_deaths.iloc[0])
        else:
            dom_fatal_dis = "Zero Recorded Outbreak Deaths (2023-2025)"
            dom_fatal_cnt = 0

        if tot_deaths >= 10:
            tier = "HIGH_OBSERVED_OUTBREAK_MORTALITY_GE_10_DEATHS"
        elif tot_deaths >= 3:
            tier = "MODERATE_OBSERVED_OUTBREAK_MORTALITY_3_TO_9_DEATHS"
        elif tot_deaths >= 1:
            tier = "LOW_OBSERVED_OUTBREAK_MORTALITY_1_TO_2_DEATHS"
        else:
            tier = "ZERO_OBSERVED_OUTBREAK_DEATHS_IN_ACTIVE_SURVEILLANCE"

        rows.append(
            {
                "district_id": DISTRICT_ID_MAP[d_name],
                "district": d_name,
                "administrative_division": ADMIN_DIVISION_MAP[d_name],
                "disease": "ALL_DISEASES",
                "reporting_period": "2023-2025_POOL",
                "outbreak_weeks_count": n_wks,
                "distinct_diseases_count": n_dis,
                "zero_death_outbreak_weeks": n_zero,
                "positive_death_outbreak_weeks": n_pos,
                "verified_cases": int(tot_cases),
                "verified_deaths": int(tot_deaths),
                "severity_count": np.nan,
                "hospital_admissions_count": np.nan,
                "complications_count": np.nan,
                "recovery_count": np.nan,
                "calculated_ratio": crude_cfr_pct,
                "wilson_95ci_lower_pct": ci_low,
                "wilson_95ci_upper_pct": ci_high,
                "empirical_bayes_shrunk_cfr_pct": eb_cfr_pct,
                "mean_deaths_per_outbreak_week": round(float(grp["verified_deaths"].mean()), 4),
                "max_deaths_in_single_outbreak_week": int(grp["verified_deaths"].max()),
                "dominant_fatal_disease": dom_fatal_dis,
                "dominant_fatal_disease_deaths": dom_fatal_cnt,
                "mortality_burden_tier": tier,
                "source": "IDSP Weekly Outbreak Surveillance (2023-2025 Deduplicated)",
                "data_completeness": (
                    "100% Verified Outbreak Cases & Deaths | Clinical Severity/Admissions Unavailable (NaN)"
                ),
                "analysis_method": "District Pooled Outbreak-Cluster CFR + Wilson 95% CI + Empirical Bayes Shrinkage",
                "generation_timestamp": gen_ts,
            }
        )

    out_df = pd.DataFrame(rows).sort_values(
        ["verified_deaths", "verified_cases", "district"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    return out_df


def compute_temporal_and_seasonal_mortality_trends(
    weekly_records_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Compute (1) annual mortality trends (2023-2025), (2) seasonal mortality trends,
    and (3) weekly statewide mortality time series (121 active weeks).
    """
    gen_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # 1. Annual mortality trends (Statewide ALL_DISEASES + Top 4 fatal diseases per year)
    annual_rows: List[Dict[str, Any]] = []
    for yr, grp in weekly_records_df.groupby("report_year"):
        tot_c = float(grp["verified_cases"].sum())
        tot_d = float(grp["verified_deaths"].sum())
        ci_l, ci_u = wilson_score_interval_pct(tot_d, tot_c)
        annual_rows.append(
            {
                "district": "ALL_MAHARASHTRA_ACTIVE_DISTRICTS",
                "disease": "ALL_DISEASES",
                "reporting_period": f"{int(yr)}_ANNUAL",
                "report_year": int(yr),
                "outbreak_weeks_count": len(grp),
                "positive_death_weeks": int((grp["verified_deaths"] > 0).sum()),
                "verified_cases": int(tot_c),
                "verified_deaths": int(tot_d),
                "severity_count": np.nan,
                "calculated_ratio": round((tot_d / tot_c) * 100.0, 4) if tot_c > 0 else np.nan,
                "wilson_95ci_lower_pct": ci_l,
                "wilson_95ci_upper_pct": ci_u,
                "source": "IDSP Weekly Outbreak Surveillance (2023-2025 Deduplicated)",
                "data_completeness": "Verified Aggregate Annual Mortality | Clinical Severity Unavailable (NaN)",
                "analysis_method": "Annual Stratified Outbreak Mortality & OC-CFR",
                "generation_timestamp": gen_ts,
            }
        )
        for dis_name in ["Dengue", "Malaria", "Acute Encephalitis Syndrome", "Acute Diarrhoeal Disease"]:
            d_grp = grp[grp["disease"] == dis_name]
            if d_grp.empty:
                continue
            dc = float(d_grp["verified_cases"].sum())
            dd = float(d_grp["verified_deaths"].sum())
            dl, du = wilson_score_interval_pct(dd, dc)
            annual_rows.append(
                {
                    "district": "ALL_MAHARASHTRA_ACTIVE_DISTRICTS",
                    "disease": dis_name,
                    "reporting_period": f"{int(yr)}_ANNUAL",
                    "report_year": int(yr),
                    "outbreak_weeks_count": len(d_grp),
                    "positive_death_weeks": int((d_grp["verified_deaths"] > 0).sum()),
                    "verified_cases": int(dc),
                    "verified_deaths": int(dd),
                    "severity_count": np.nan,
                    "calculated_ratio": round((dd / dc) * 100.0, 4) if dc > 0 else np.nan,
                    "wilson_95ci_lower_pct": dl,
                    "wilson_95ci_upper_pct": du,
                    "source": "IDSP Weekly Outbreak Surveillance (2023-2025 Deduplicated)",
                    "data_completeness": "Verified Aggregate Annual Mortality | Clinical Severity Unavailable (NaN)",
                    "analysis_method": "Annual Stratified Outbreak Mortality & OC-CFR",
                    "generation_timestamp": gen_ts,
                }
            )
    annual_df = pd.DataFrame(annual_rows)

    # 2. Seasonal mortality distribution across IMD seasons
    seasonal_rows: List[Dict[str, Any]] = []
    season_groups = []
    for seas, grp in weekly_records_df.groupby("season"):
        season_groups.append(grp["verified_deaths"].to_numpy(dtype=float))
        sc = float(grp["verified_cases"].sum())
        sd = float(grp["verified_deaths"].sum())
        sl, su = wilson_score_interval_pct(sd, sc)
        seasonal_rows.append(
            {
                "district": "ALL_MAHARASHTRA_ACTIVE_DISTRICTS",
                "disease": "ALL_DISEASES",
                "reporting_period": f"2023-2025_{seas}",
                "season": str(seas),
                "outbreak_weeks_count": len(grp),
                "positive_death_weeks": int((grp["verified_deaths"] > 0).sum()),
                "positive_death_weeks_pct": round(float((grp["verified_deaths"] > 0).mean()) * 100.0, 2),
                "verified_cases": int(sc),
                "verified_deaths": int(sd),
                "share_of_statewide_deaths_pct": round(
                    (sd / float(weekly_records_df["verified_deaths"].sum())) * 100.0, 2
                ),
                "severity_count": np.nan,
                "calculated_ratio": round((sd / sc) * 100.0, 4) if sc > 0 else np.nan,
                "wilson_95ci_lower_pct": sl,
                "wilson_95ci_upper_pct": su,
                "mean_deaths_per_outbreak_week": round(float(grp["verified_deaths"].mean()), 4),
                "source": "IDSP Weekly Outbreak Surveillance (2023-2025 Deduplicated)",
                "data_completeness": "Verified Seasonal Aggregate Mortality | Clinical Severity Unavailable (NaN)",
                "analysis_method": "Seasonal Stratified Mortality & Kruskal-Wallis Nonparametric Test",
                "generation_timestamp": gen_ts,
            }
        )
    seasonal_df = pd.DataFrame(seasonal_rows).sort_values("verified_deaths", ascending=False).reset_index(drop=True)

    kw_stat, kw_p = stats.kruskal(*season_groups) if len(season_groups) >= 2 else (np.nan, np.nan)

    # 3. Weekly Statewide Mortality Time Series (121 active surveillance weeks)
    wk_rows: List[Dict[str, Any]] = []
    for (yr, wk, c_start, rep_p), grp in weekly_records_df.groupby(
        ["report_year", "report_week", "canonical_week_start", "reporting_period"]
    ):
        wc = float(grp["verified_cases"].sum())
        wd = float(grp["verified_deaths"].sum())
        wl, wu = wilson_score_interval_pct(wd, wc)
        dengue_d = float(grp.loc[grp["disease"] == "Dengue", "verified_deaths"].sum())
        malaria_d = float(grp.loc[grp["disease"] == "Malaria", "verified_deaths"].sum())
        aes_d = float(grp.loc[grp["disease"] == "Acute Encephalitis Syndrome", "verified_deaths"].sum())
        add_d = float(grp.loc[grp["disease"] == "Acute Diarrhoeal Disease", "verified_deaths"].sum())
        other_d = max(0.0, wd - (dengue_d + malaria_d + aes_d + add_d))

        wk_rows.append(
            {
                "district": "ALL_MAHARASHTRA_ACTIVE_DISTRICTS",
                "disease": "ALL_DISEASES",
                "reporting_period": str(rep_p),
                "report_year": int(yr),
                "report_week": int(wk),
                "canonical_week_start": str(c_start),
                "active_outbreak_clusters": len(grp),
                "active_districts_count": int(grp["district"].nunique()),
                "verified_cases": int(wc),
                "verified_deaths": int(wd),
                "dengue_deaths": int(dengue_d),
                "malaria_deaths": int(malaria_d),
                "aes_deaths": int(aes_d),
                "add_deaths": int(add_d),
                "other_disease_deaths": int(other_d),
                "severity_count": np.nan,
                "calculated_ratio": round((wd / wc) * 100.0, 4) if wc > 0 else np.nan,
                "wilson_95ci_lower_pct": wl,
                "wilson_95ci_upper_pct": wu,
                "source": "IDSP Weekly Outbreak Surveillance (2023-2025 Deduplicated)",
                "data_completeness": "Verified Weekly Statewide Aggregate | Clinical Severity Unavailable (NaN)",
                "analysis_method": "Weekly Statewide Aggregate Mortality & OC-CFR",
                "generation_timestamp": gen_ts,
            }
        )
    weekly_series_df = (
        pd.DataFrame(wk_rows).sort_values(["report_year", "report_week"]).reset_index(drop=True)
    )
    weekly_series_df["rolling_4wk_mean_deaths"] = (
        weekly_series_df["verified_deaths"].rolling(4, min_periods=1).mean().round(4)
    )

    temporal_stats = {
        "kruskal_wallis_seasonal_deaths_stat": round(float(kw_stat), 4) if pd.notna(kw_stat) else None,
        "kruskal_wallis_seasonal_deaths_pvalue": round(float(kw_p), 6) if pd.notna(kw_p) else None,
        "active_surveillance_weeks_count": len(weekly_series_df),
        "weeks_with_positive_deaths": int((weekly_series_df["verified_deaths"] > 0).sum()),
    }
    return annual_df, seasonal_df, weekly_series_df, temporal_stats
