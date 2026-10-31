"""Step 1 & 2 of Phase 8: Inspect Available Data and Align Climate & Disease Datasets.

Reads verified project datasets:
- `data/processed/cleaned_climate_data.csv` (daily district climate 2023-2025)
- `data/processed/cleaned_disease_data.csv` (weekly 2023-2025 + historical 2009-2022)
- `data/features/disease_climate_weekly_features.csv` (511 weekly outbreak records 2023-2025)
- `data/features/district_week_climate_features.csv` (5,616 district-week climate panel 2023-2025)

Generates aligned datasets in `data/climate_intelligence/`:
1. `aligned_weekly_climate_disease_outbreaks.csv` (weekly outbreak case records aligned to climate lags)
2. `aligned_district_week_surveillance_panel.csv` (complete 35-district x 156-week panel with active-week outbreak indicators and climate lags)
3. `aligned_monthly_climate_disease_panel.csv` (35-district x 36-month panel with monthly rainfall accumulation, temperature means, and disease burdens)
4. `data_coverage_and_alignment_audit.json`
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from . import CAUSALITY_DISCLAIMER, CLIMATE_INTELLIGENCE_VERSION

PROJ = Path(__file__).resolve().parents[3]
CLEANED_CLIMATE_PATH = PROJ / "data" / "processed" / "cleaned_climate_data.csv"
CLEANED_DISEASE_PATH = PROJ / "data" / "processed" / "cleaned_disease_data.csv"
WEEKLY_OUTBREAK_FEAT_PATH = PROJ / "data" / "features" / "disease_climate_weekly_features.csv"
DISTRICT_WEEK_CLIMATE_PATH = PROJ / "data" / "features" / "district_week_climate_features.csv"

CLIMATE_INTEL_DATA_DIR = PROJ / "data" / "climate_intelligence"
CLIMATE_INTEL_REPORTS_DIR = PROJ / "reports" / "climate_intelligence"

VECTOR_BORNE_DISEASES = {
    "Dengue",
    "Malaria",
    "Chikungunya",
    "Zika Virus",
    "Dengue + Chikungunya",
    "Dengue + Malaria",
    "Japanese Encephalitis",
    "Scrub Typhus",
}

WATER_FOOD_BORNE_DISEASES = {
    "Acute Diarrhoeal Disease",
    "Cholera",
    "Food Poisoning",
    "Hepatitis A",
    "Hepatitis A + E",
    "Acute Hepatitis (unspecified)",
    "Typhoid",
    "Dysentery",
    "Leptospirosis",
    "Jaundice",
}


def build_aligned_climate_disease_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Inspect existing datasets and build aligned weekly outbreak, district-week, and district-month panels."""
    CLIMATE_INTEL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CLIMATE_INTEL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    daily_clim = pd.read_csv(CLEANED_CLIMATE_PATH)
    clean_dis = pd.read_csv(CLEANED_DISEASE_PATH)
    wk_outbreaks = pd.read_csv(WEEKLY_OUTBREAK_FEAT_PATH)
    dw_clim = pd.read_csv(DISTRICT_WEEK_CLIMATE_PATH)

    # 1. Aligned Weekly Outbreak Panel (`aligned_weekly_climate_disease_outbreaks.csv`)
    # Filter to rows with valid climate coordinates (`has_coordinates == True` and `rain_mm_sum_t` non-null)
    # while explicitly logging the 3 `Mumbai Suburban` rows that lack coordinates.
    wk_outbreaks["log1p_cases"] = np.log1p(wk_outbreaks["target_cases"].clip(lower=0.0))
    wk_outbreaks["is_vector_borne"] = wk_outbreaks["disease_std"].isin(VECTOR_BORNE_DISEASES).astype(int)
    wk_outbreaks["is_water_food_borne"] = wk_outbreaks["disease_std"].isin(WATER_FOOD_BORNE_DISEASES).astype(int)

    aligned_outbreaks = wk_outbreaks[wk_outbreaks["has_coordinates"] == True].copy().reset_index(drop=True)
    aligned_outbreaks.to_csv(
        CLIMATE_INTEL_DATA_DIR / "aligned_weekly_climate_disease_outbreaks.csv", index=False
    )

    # 2. Complete District-Week Surveillance & Climate Panel (`aligned_district_week_surveillance_panel.csv`)
    # Aggregate weekly outbreak counts & case totals per (district_std, report_year, report_week)
    active_weeks_in_state = set(
        zip(wk_outbreaks["report_year"].astype(int), wk_outbreaks["report_week"].astype(int))
    )

    dw_agg = (
        wk_outbreaks.groupby(["district_std", "report_year", "report_week"], as_index=False)
        .agg(
            total_outbreak_events=("outbreak_events_in_period", "sum"),
            total_outbreak_cases=("target_cases", "sum"),
            total_outbreak_deaths=("target_deaths", lambda s: float(s.dropna().sum()) if len(s.dropna()) > 0 else 0.0),
            vector_borne_cases=("target_cases", lambda s: float(s[wk_outbreaks.loc[s.index, "is_vector_borne"] == 1].sum())),
            vector_borne_outbreaks=("is_vector_borne", "sum"),
            water_borne_cases=("target_cases", lambda s: float(s[wk_outbreaks.loc[s.index, "is_water_food_borne"] == 1].sum())),
            water_borne_outbreaks=("is_water_food_borne", "sum"),
            dengue_cases=("target_cases", lambda s: float(s[wk_outbreaks.loc[s.index, "disease_std"] == "Dengue"].sum())),
            add_cases=("target_cases", lambda s: float(s[wk_outbreaks.loc[s.index, "disease_std"] == "Acute Diarrhoeal Disease"].sum())),
            malaria_cases=("target_cases", lambda s: float(s[wk_outbreaks.loc[s.index, "disease_std"] == "Malaria"].sum())),
        )
    )

    dw_panel = dw_clim[dw_clim["has_coordinates"] == True].copy()
    dw_panel = dw_panel.merge(
        dw_agg, on=["district_std", "report_year", "report_week"], how="left"
    )
    dw_panel["state_idsp_bulletin_published_in_week"] = [
        int((int(y), int(w)) in active_weeks_in_state)
        for y, w in zip(dw_panel["report_year"], dw_panel["report_week"])
    ]

    # Only fill 0 for outbreak occurrence in weeks where the state IDSP bulletin was actually published;
    # weeks with no state IDSP bulletin (`state_idsp_bulletin_published_in_week == 0`) remain NaN so silent
    # unparsed weeks are never fabricated as zero-outbreak weeks!
    pub_mask = dw_panel["state_idsp_bulletin_published_in_week"] == 1
    for c_zero in [
        "total_outbreak_events",
        "total_outbreak_cases",
        "total_outbreak_deaths",
        "vector_borne_cases",
        "vector_borne_outbreaks",
        "water_borne_cases",
        "water_borne_outbreaks",
        "dengue_cases",
        "add_cases",
        "malaria_cases",
    ]:
        dw_panel[c_zero] = np.where(pub_mask, dw_panel[c_zero].fillna(0.0), np.nan)

    dw_panel["any_outbreak_occurred"] = np.where(
        pub_mask, (dw_panel["total_outbreak_events"] >= 1).astype(float), np.nan
    )
    dw_panel["vector_borne_outbreak_occurred"] = np.where(
        pub_mask, (dw_panel["vector_borne_outbreaks"] >= 1).astype(float), np.nan
    )
    dw_panel["water_borne_outbreak_occurred"] = np.where(
        pub_mask, (dw_panel["water_borne_outbreaks"] >= 1).astype(float), np.nan
    )
    dw_panel["log1p_total_outbreak_cases"] = np.log1p(dw_panel["total_outbreak_cases"])
    dw_panel.to_csv(
        CLIMATE_INTEL_DATA_DIR / "aligned_district_week_surveillance_panel.csv", index=False
    )

    # 3. Monthly Aggregated Climate-Disease Panel (`aligned_monthly_climate_disease_panel.csv`)
    # Aggregate daily climate to (district_std, year, month) and match to monthly outbreak aggregations
    daily_clim["date_dt"] = pd.to_datetime(daily_clim["date"])
    daily_clim["year"] = daily_clim["date_dt"].dt.year
    daily_clim["month"] = daily_clim["date_dt"].dt.month
    daily_clim["tavg_c"] = 0.5 * (daily_clim["tmin_c"] + daily_clim["tmax_c"])
    daily_clim["diurnal_range_c"] = daily_clim["tmax_c"] - daily_clim["tmin_c"]

    m_clim = (
        daily_clim[daily_clim["district_std"] != "Mumbai Suburban"]
        .groupby(["district_std", "year", "month"], as_index=False)
        .agg(
            monthly_rain_mm_sum=("rain_mm", lambda s: float(s.dropna().sum()) if len(s.dropna()) > 0 else np.nan),
            monthly_rain_mm_mean=("rain_mm", "mean"),
            monthly_rainy_days=("rain_mm", lambda s: int((s.dropna() >= 2.5).sum())),
            monthly_tmin_c_mean=("tmin_c", "mean"),
            monthly_tmax_c_mean=("tmax_c", "mean"),
            monthly_tavg_c_mean=("tavg_c", "mean"),
            monthly_diurnal_range_c_mean=("diurnal_range_c", "mean"),
            monthly_rh_mean=("rh_mean", "mean"),
        )
        .sort_values(["district_std", "year", "month"])
        .reset_index(drop=True)
    )
    # Add 1-month and 2-month lagged monthly climate variables per district
    mg = m_clim.groupby("district_std", sort=False)
    for col in [
        "monthly_rain_mm_sum",
        "monthly_tmin_c_mean",
        "monthly_tmax_c_mean",
        "monthly_tavg_c_mean",
        "monthly_diurnal_range_c_mean",
    ]:
        m_clim[f"{col}_lag_1m"] = mg[col].shift(1)
        m_clim[f"{col}_lag_2m"] = mg[col].shift(2)

    m_dis = (
        aligned_outbreaks.groupby(["district_std", "calendar_year", "calendar_month"], as_index=False)
        .agg(
            monthly_outbreak_records=("target_cases", "count"),
            monthly_total_cases=("target_cases", "sum"),
            monthly_total_deaths=("target_deaths", lambda s: float(s.dropna().sum()) if len(s.dropna()) > 0 else 0.0),
            monthly_vector_borne_cases=("target_cases", lambda s: float(s[aligned_outbreaks.loc[s.index, "is_vector_borne"] == 1].sum())),
            monthly_water_borne_cases=("target_cases", lambda s: float(s[aligned_outbreaks.loc[s.index, "is_water_food_borne"] == 1].sum())),
        )
        .rename(columns={"calendar_year": "year", "calendar_month": "month"})
    )
    m_panel = m_clim.merge(m_dis, on=["district_std", "year", "month"], how="left")
    m_panel.to_csv(
        CLIMATE_INTEL_DATA_DIR / "aligned_monthly_climate_disease_panel.csv", index=False
    )

    # 4. Data Coverage and Alignment Audit JSON
    audit = {
        "engine_version": CLIMATE_INTELLIGENCE_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "causality_disclaimer": CAUSALITY_DISCLAIMER,
        "climate_variables_inventory": {
            "rainfall_mm": {
                "source": "IMD 0.25x0.25 deg daily gridded NetCDF (imdweb_5dgbAskl.nc)",
                "date_range": "2023-01-01 to 2025-12-31 (1,096 days)",
                "native_frequency": "daily (aggregated to weekly totals/means/rainy-days and 1..8 week lags)",
                "district_coverage": "35 / 36 Maharashtra districts (34 strict nearest grid cell + Gadchiroli via 3x3 neighborhood fallback; Mumbai Suburban has no coordinates)",
                "usable_for_statewide_analysis": True,
            },
            "temperature_tmin_tmax_tavg_diurnal": {
                "source": "IMD 1.0x1.0 deg daily gridded NetCDF (imdweb_Oe9H04Bg.nc, imdweb_tXexQbB6.nc)",
                "date_range": "2023-01-01 to 2025-12-31 (1,096 days)",
                "native_frequency": "daily (aggregated to weekly means, extremes, diurnal range, and 1..8 week lags)",
                "district_coverage": "35 / 36 Maharashtra districts (Mumbai Suburban missing coordinates)",
                "note_on_tavg": "No standalone tavg raw file; derived as 0.5 * (tmin_c + tmax_c)",
                "usable_for_statewide_analysis": True,
            },
            "relative_humidity_pct": {
                "source": "CWPRS Khadakvasla station telemetry (humid_tel_hr_cwprs_mh_2021_2025.csv)",
                "date_range": "2024-05-11 to 2025-06-09 (395 valid daily aggregates after removing <0 or >100 sensor spikes)",
                "native_frequency": "sub-hourly telemetry aggregated to daily and weekly means",
                "district_coverage": "1 / 36 districts (Pune ONLY; 0% coverage in the other 35 districts)",
                "usable_for_statewide_analysis": False,
                "usable_for_pune_sub_analysis": True,
            },
        },
        "disease_surveillance_inventory": {
            "contemporary_2023_2025": {
                "source": "maharashtra_outbreaks_2023_2025.csv -> data/features/disease_climate_weekly_features.csv",
                "native_frequency": "weekly IDSP outbreak reports (W01..W52)",
                "active_reporting_weeks": "121 / 156 weeks (39/52 in 2023, 52/52 in 2024, 30/52 in 2025)",
                "deduplicated_district_disease_week_rows": int(len(wk_outbreaks)),
                "climate_aligned_outbreak_rows": int(len(aligned_outbreaks)),
                "unaligned_outbreak_rows_mumbai_suburban": int(len(wk_outbreaks) - len(aligned_outbreaks)),
                "districts_with_outbreaks": int(wk_outbreaks["district_std"].nunique()),
                "zero_outbreak_districts_2023_2025": ["Mumbai City", "Ratnagiri"],
                "diseases_observed": int(wk_outbreaks["disease_std"].nunique()),
                "unsupported_zero_record_diseases": ["Heat-related illness"],
            },
            "historical_2009_2022_note": {
                "rows": int((clean_dis["source_dataset"] == "historical_2009_2022").sum()),
                "usable_with_2023_2025_imd_climate": False,
                "reason": "Zero temporal overlap between 2009-2022 EpiClim records and 2023-2025 IMD/CWPRS climate files.",
            },
        },
        "aligned_panels_generated": {
            "aligned_weekly_climate_disease_outbreaks": {
                "path": "data/climate_intelligence/aligned_weekly_climate_disease_outbreaks.csv",
                "rows": int(len(aligned_outbreaks)),
            },
            "aligned_district_week_surveillance_panel": {
                "path": "data/climate_intelligence/aligned_district_week_surveillance_panel.csv",
                "rows": int(len(dw_panel)),
                "published_bulletin_district_weeks": int(pub_mask.sum()),
            },
            "aligned_monthly_climate_disease_panel": {
                "path": "data/climate_intelligence/aligned_monthly_climate_disease_panel.csv",
                "rows": int(len(m_panel)),
            },
        },
    }

    for out_dir in (CLIMATE_INTEL_DATA_DIR, CLIMATE_INTEL_REPORTS_DIR):
        with open(out_dir / "data_coverage_and_alignment_audit.json", "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2)

    return aligned_outbreaks, dw_panel, m_panel, audit
