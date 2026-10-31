"""Geographic, demographic, and district-level historical disease burden features (2023-2025 scope).

Rules:
- Scoped strictly to the 2023-2025 weekly datasets.
- Do NOT calculate population-normalized rates without a valid population denominator.
  Because no district population/census dataset exists in the project workspace,
  district_population, population_density, and cases_per_100k_pop are explicitly set to NaN
  and documented as unsupported without an external denominator file.
- District-level historical disease burden is computed using a strictly expanding window
  over prior reporting weeks (< period_idx) within 2023-2025 so zero future information leaks.
"""
import numpy as np
import pandas as pd

OFFICIAL_DISTRICTS = [
    "Ahmednagar", "Akola", "Amravati", "Beed", "Bhandara", "Buldhana", "Chandrapur",
    "Chhatrapati Sambhajinagar", "Dharashiv", "Dhule", "Gadchiroli", "Gondia", "Hingoli",
    "Jalgaon", "Jalna", "Kolhapur", "Latur", "Mumbai City", "Mumbai Suburban", "Nagpur",
    "Nanded", "Nandurbar", "Nashik", "Palghar", "Parbhani", "Pune", "Raigad", "Ratnagiri",
    "Sangli", "Satara", "Sindhudurg", "Solapur", "Thane", "Wardha", "Washim", "Yavatmal",
]

DISTRICT_ID_MAP = {d: f"MH_D{i + 1:02d}" for i, d in enumerate(OFFICIAL_DISTRICTS)}

ADMIN_DIVISION_MAP = {
    "Mumbai City": "Konkan", "Mumbai Suburban": "Konkan", "Thane": "Konkan", "Palghar": "Konkan",
    "Raigad": "Konkan", "Ratnagiri": "Konkan", "Sindhudurg": "Konkan",
    "Pune": "Pune", "Satara": "Pune", "Sangli": "Pune", "Solapur": "Pune", "Kolhapur": "Pune",
    "Nashik": "Nashik", "Dhule": "Nashik", "Nandurbar": "Nashik", "Jalgaon": "Nashik", "Ahmednagar": "Nashik",
    "Chhatrapati Sambhajinagar": "Marathwada", "Jalna": "Marathwada", "Beed": "Marathwada",
    "Dharashiv": "Marathwada", "Latur": "Marathwada", "Nanded": "Marathwada", "Hingoli": "Marathwada", "Parbhani": "Marathwada",
    "Amravati": "Amravati_Vidarbha_West", "Akola": "Amravati_Vidarbha_West", "Yavatmal": "Amravati_Vidarbha_West",
    "Buldhana": "Amravati_Vidarbha_West", "Washim": "Amravati_Vidarbha_West",
    "Nagpur": "Nagpur_Vidarbha_East", "Wardha": "Nagpur_Vidarbha_East", "Bhandara": "Nagpur_Vidarbha_East",
    "Gondia": "Nagpur_Vidarbha_East", "Chandrapur": "Nagpur_Vidarbha_East", "Gadchiroli": "Nagpur_Vidarbha_East",
}

COASTAL_DISTRICTS = {
    "Mumbai City", "Mumbai Suburban", "Thane", "Palghar", "Raigad", "Ratnagiri", "Sindhudurg"
}


def build_district_static_table(cleaned_climate_df: pd.DataFrame) -> pd.DataFrame:
    """Build static geographic lookup table for all 36 Maharashtra districts from cleaned_climate_data.csv."""
    coords = cleaned_climate_df.groupby("district_std")[["lat", "lon"]].first().to_dict(orient="index")

    rows = []
    for d in OFFICIAL_DISTRICTS:
        c = coords.get(d, {})
        lat = c.get("lat", np.nan)
        lon = c.get("lon", np.nan)
        div = ADMIN_DIVISION_MAP[d]
        rows.append(
            dict(
                district_std=d,
                district_id=DISTRICT_ID_MAP[d],
                administrative_division=div,
                is_coastal_district=int(d in COASTAL_DISTRICTS),
                is_vidarbha=int("Vidarbha" in div),
                is_marathwada=int(div == "Marathwada"),
                district_lat=lat,
                district_lon=lon,
                has_coordinates=int(pd.notna(lat) and pd.notna(lon)),
                # Demographic denominators are absent from the project workspace; never fabricated
                district_population=np.nan,
                population_density=np.nan,
            )
        )
    return pd.DataFrame(rows)


def add_geographic_features_weekly(
    weekly_df: pd.DataFrame, cleaned_climate_df: pd.DataFrame
) -> pd.DataFrame:
    """Attach static geographic features, strictly expanding prior-week (< period_idx)
    district disease burden within 2023-2025, and explicit NaN population-normalized
    incidence columns."""
    static_tbl = build_district_static_table(cleaned_climate_df)
    out = weekly_df.merge(static_tbl, on="district_std", how="left")

    if "target_cases" in out.columns and "disease_std" in out.columns:
        # 1. District x Disease expanding historical burden strictly prior to current week (shift(1))
        out = out.sort_values(["district_std", "disease_std", "period_idx"]).reset_index(drop=True)
        g_dd = out.groupby(["district_std", "disease_std"], sort=False)

        prior_cases = g_dd["target_cases"].shift(1)
        prior_events = g_dd["outbreak_events_in_period"].shift(1)
        prior_deaths = g_dd["target_deaths"].shift(1)

        dd_keys = [out["district_std"], out["disease_std"]]
        out["district_disease_prior_outbreaks_cum"] = prior_events.groupby(dd_keys).cumsum()
        out["district_disease_prior_cases_cum"] = prior_cases.groupby(dd_keys).cumsum()
        out["district_disease_prior_deaths_cum"] = prior_deaths.groupby(dd_keys).cumsum()
        out["district_disease_prior_mean_cases"] = (
            prior_cases.groupby(dd_keys).expanding(min_periods=1).mean().reset_index(level=[0, 1], drop=True)
        )
        out["district_disease_prior_max_cases"] = (
            prior_cases.groupby(dd_keys).expanding(min_periods=1).max().reset_index(level=[0, 1], drop=True)
        )

        # 2. District overall expanding historical burden across all diseases strictly prior to period_idx
        dist_wk = (
            out.groupby(["district_std", "period_idx"], as_index=False)
            .agg(
                _wk_dist_cases=("target_cases", "sum"),
                _wk_dist_events=("outbreak_events_in_period", "sum"),
            )
            .sort_values(["district_std", "period_idx"])
            .reset_index(drop=True)
        )
        gd = dist_wk.groupby("district_std", sort=False)
        dist_wk["district_all_diseases_prior_cases_cum"] = (
            gd["_wk_dist_cases"].shift(1).groupby(dist_wk["district_std"]).cumsum()
        )
        dist_wk["district_all_diseases_prior_outbreaks_cum"] = (
            gd["_wk_dist_events"].shift(1).groupby(dist_wk["district_std"]).cumsum()
        )
        out = out.merge(
            dist_wk[
                [
                    "district_std",
                    "period_idx",
                    "district_all_diseases_prior_cases_cum",
                    "district_all_diseases_prior_outbreaks_cum",
                ]
            ],
            on=["district_std", "period_idx"],
            how="left",
        )

    # Never calculate population-normalized incidence without a valid population denominator
    out["cases_per_100k_pop"] = np.nan
    return out
