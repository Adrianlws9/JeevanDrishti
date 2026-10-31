"""Climate feature generation aligned to the 2023-2025 weekly disease reporting periods.

Alignment & Anti-Leakage Rules:
- Strictly scoped to 2023-2025 (156 reporting weeks: 2023-W01 .. 2025-W52).
- Each reporting period (report_year, report_week) is mapped to its exact [report_period_start, report_period_end]
  window (from IDSP source where reported, or the verified canonical 7-day Monday-Sunday window).
- Daily IMD gridded rainfall, Tmin, Tmax, derived Tavg, and CWPRS Pune daily humidity from
  data/processed/cleaned_climate_data.csv are aggregated over [report_period_start, report_period_end]
  for each of the 36 Maharashtra districts.
- For Gadchiroli, whose strict nearest 0.25-deg rainfall grid node (19.75N, 80.25E) lies on the
  masked eastern grid border, we preserve the strict nearest column (rain_mm_strict_nearest_sum_t = NaN)
  and extract the immediate adjacent unmasked Gadchiroli interior land node (19.75N, 80.00E, 0.162 deg
  from the district point) directly from data/raw/imdweb_5dgbAskl.nc, tagging
  rain_spatial_source = 'adjacent_land_cell_19.75N_80.00E'. Mumbai Suburban has no coordinates in
  any project file and remains strictly NaN (rain_spatial_source = 'none_no_coordinates').
- All rolling and seasonal-baseline climate calculations use strictly past/expanding windows (<= t-1)
  so no future climate observation (> t) ever leaks into period t.
"""
from pathlib import Path
import netCDF4 as nc
import numpy as np
import pandas as pd

from .temporal_features import build_weekly_calendar, add_temporal_features_weekly
from .geographic_features import OFFICIAL_DISTRICTS


def _load_gadchiroli_adjacent_land_rain(raw_rain_nc_path: Path) -> pd.DataFrame:
    """Extract daily rainfall for Gadchiroli's adjacent interior land cell (19.75N, 80.00E)
    from data/raw/imdweb_5dgbAskl.nc (0.162 deg from Gadchiroli point 19.759N, 80.162E)."""
    with nc.Dataset(raw_rain_nc_path) as ds:
        g = ds.groups["rain"]
        lat = np.asarray(g["lat"][:], dtype="float64")
        lon = np.asarray(g["lon"][:], dtype="float64")
        t_days = np.asarray(g["time"][:], dtype="float64")
        dates = pd.Timestamp("2023-01-01") + pd.to_timedelta(t_days, unit="D")
        i = int(np.abs(lat - 19.75).argmin())
        j = int(np.abs(lon - 80.00).argmin())
        arr = np.ma.filled(g["rain"][:, i, j].astype("float64"), np.nan)
    return pd.DataFrame({"date": dates, "gadchiroli_adj_rain_mm": arr})


def build_weekly_period_lookup(cleaned_disease_df: pd.DataFrame) -> pd.DataFrame:
    """Build the 156-week lookup table (2023-W01 .. 2025-W52) using exact source
    report_period_start/end where present in idsp_weekly and canonical 7-day Monday-Sunday
    dates otherwise."""
    cal = build_weekly_calendar(2023, 2025)
    idsp = cleaned_disease_df[cleaned_disease_df["source_dataset"] == "idsp_weekly"].copy()
    idsp["report_period_start"] = pd.to_datetime(idsp["report_period_start"])
    idsp["report_period_end"] = pd.to_datetime(idsp["report_period_end"])

    obs_periods = (
        idsp.groupby(["report_year", "report_week"], as_index=False)
        .agg(
            src_start=("report_period_start", "first"),
            src_end=("report_period_end", "first"),
            report_period_source=("report_period_source", "first"),
        )
    )
    cal = cal.merge(obs_periods, on=["report_year", "report_week"], how="left")
    cal["report_period_start"] = cal["src_start"].fillna(cal["canonical_week_start"])
    cal["report_period_end"] = cal["src_end"].fillna(cal["canonical_week_end"])
    cal["report_period_source"] = cal["report_period_source"].fillna("canonical_calendar_no_idsp_rows")
    return cal.drop(columns=["src_start", "src_end"]).sort_values("period_idx").reset_index(drop=True)


def build_district_week_climate_features(
    cleaned_climate_df: pd.DataFrame,
    cleaned_disease_df: pd.DataFrame,
    raw_rain_nc_path: Path,
) -> pd.DataFrame:
    """Aggregate 2023-2025 daily climate data to the 36 districts x 156 reporting weeks grid and
    compute lagged, accumulated, variation, and deviation features."""
    clim = cleaned_climate_df.copy()
    clim["date"] = pd.to_datetime(clim["date"])
    clim["rain_mm_strict_nearest"] = clim["rain_mm"]

    # Rescue Gadchiroli rainfall from immediate interior land cell (19.75N, 80.00E) with explicit provenance
    gad_rain = _load_gadchiroli_adjacent_land_rain(raw_rain_nc_path)
    clim = clim.merge(gad_rain, on="date", how="left")
    is_gad = clim["district_std"] == "Gadchiroli"
    clim.loc[is_gad & clim["rain_mm"].isna(), "rain_mm"] = clim.loc[
        is_gad & clim["rain_mm"].isna(), "gadchiroli_adj_rain_mm"
    ]
    clim["rain_spatial_source"] = np.where(
        clim["district_std"] == "Mumbai Suburban",
        "none_no_coordinates",
        np.where(is_gad, "adjacent_land_cell_19.75N_80.00E", "nearest_grid_cell"),
    )
    clim["diurnal_temp_range_c"] = clim["tmax_c"] - clim["tmin_c"]
    clim["is_rainy_day_ge_2_5mm"] = np.where(
        clim["rain_mm"].notna(), (clim["rain_mm"] >= 2.5).astype(float), np.nan
    )
    clim["is_heavy_rain_day_ge_64_5mm"] = np.where(
        clim["rain_mm"].notna(), (clim["rain_mm"] >= 64.5).astype(float), np.nan
    )

    periods = build_weekly_period_lookup(cleaned_disease_df)

    # Map each daily date in 2023-01-01..2025-12-31 to the reporting week(s) it belongs to
    date_to_period = []
    for _, row in periods.iterrows():
        days = pd.date_range(row["report_period_start"], row["report_period_end"], freq="D")
        for d in days:
            date_to_period.append((d, int(row["period_idx"])))
    d2p_df = pd.DataFrame(date_to_period, columns=["date", "period_idx"])

    merged = d2p_df.merge(clim, on="date", how="inner")
    grp = merged.groupby(["district_std", "period_idx"], as_index=False)

    wk_clim = grp.agg(
        rain_mm_sum_t=("rain_mm", lambda s: s.sum(min_count=1)),
        rain_mm_strict_nearest_sum_t=("rain_mm_strict_nearest", lambda s: s.sum(min_count=1)),
        rain_mm_mean_t=("rain_mm", "mean"),
        rain_mm_max_t=("rain_mm", "max"),
        rainy_days_t=("is_rainy_day_ge_2_5mm", lambda s: s.sum(min_count=1)),
        heavy_rain_days_t=("is_heavy_rain_day_ge_64_5mm", lambda s: s.sum(min_count=1)),
        rain_valid_days_in_week=("rain_mm", "count"),
        tmin_c_mean_t=("tmin_c", "mean"),
        tmin_c_min_t=("tmin_c", "min"),
        tmax_c_mean_t=("tmax_c", "mean"),
        tmax_c_max_t=("tmax_c", "max"),
        tavg_c_mean_t=("tavg_derived_c", "mean"),
        tavg_c_within_week_std_t=("tavg_derived_c", "std"),
        temp_diurnal_range_mean_t=("diurnal_temp_range_c", "mean"),
        temp_valid_days_in_week=("tavg_derived_c", "count"),
        rh_mean_t=("rh_mean", "mean"),
        rh_min_t=("rh_min", "min"),
        rh_max_t=("rh_max", "max"),
        rh_valid_days_in_week=("rh_mean", "count"),
        rain_spatial_source=("rain_spatial_source", "first"),
    )
    wk_clim["temp_weekly_extreme_spread_t"] = wk_clim["tmax_c_max_t"] - wk_clim["tmin_c_min_t"]

    # Ensure full 36 districts x 156 periods grid
    full_idx = pd.MultiIndex.from_product(
        [OFFICIAL_DISTRICTS, periods["period_idx"].tolist()], names=["district_std", "period_idx"]
    ).to_frame().reset_index(drop=True)
    wk = full_idx.merge(periods, on="period_idx", how="left").merge(
        wk_clim, on=["district_std", "period_idx"], how="left"
    )
    wk = add_temporal_features_weekly(wk)
    wk = wk.sort_values(["district_std", "period_idx"]).reset_index(drop=True)

    # ---- Lagged & Rolling climate features per district (strictly ordered by period_idx)
    g = wk.groupby("district_std", sort=False)

    for lag in (1, 2, 3, 4, 8, 52, 53):
        wk[f"rain_mm_sum_lag_{lag}w"] = g["rain_mm_sum_t"].shift(lag)
    wk["rain_mm_mean_lag_1w"] = g["rain_mm_mean_t"].shift(1)
    wk["rain_mm_max_lag_1w"] = g["rain_mm_max_t"].shift(1)
    wk["rainy_days_lag_1w"] = g["rainy_days_t"].shift(1)

    for lag in (1, 2, 4, 8):
        wk[f"tavg_c_mean_lag_{lag}w"] = g["tavg_c_mean_t"].shift(lag)
        wk[f"tmin_c_mean_lag_{lag}w"] = g["tmin_c_mean_t"].shift(lag)
        wk[f"tmax_c_mean_lag_{lag}w"] = g["tmax_c_mean_t"].shift(lag)
        wk[f"temp_diurnal_range_lag_{lag}w"] = g["temp_diurnal_range_mean_t"].shift(lag)
        wk[f"rh_mean_lag_{lag}w"] = g["rh_mean_t"].shift(lag)

    # Accumulated rainfall over 2, 4, and 8 periods:
    # (1) Strictly prior to period t (forecasting cutoff t-1): uses shift(1) before rolling
    rain_prev = g["rain_mm_sum_t"].shift(1)
    tavg_prev = g["tavg_c_mean_t"].shift(1)
    dtr_prev = g["temp_diurnal_range_mean_t"].shift(1)
    rh_prev = g["rh_mean_t"].shift(1)

    for win in (2, 4, 8):
        wk[f"rain_mm_accum_prev_{win}w"] = (
            rain_prev.groupby(wk["district_std"]).rolling(win, min_periods=win).sum().reset_index(level=0, drop=True)
        )
        wk[f"rain_mm_roll_mean_prev_{win}w"] = (
            rain_prev.groupby(wk["district_std"]).rolling(win, min_periods=win).mean().reset_index(level=0, drop=True)
        )
        wk[f"tavg_c_roll_mean_prev_{win}w"] = (
            tavg_prev.groupby(wk["district_std"]).rolling(win, min_periods=win).mean().reset_index(level=0, drop=True)
        )
        wk[f"tavg_c_roll_std_prev_{win}w"] = (
            tavg_prev.groupby(wk["district_std"]).rolling(win, min_periods=2).std().reset_index(level=0, drop=True)
        )
        # (2) Inclusive of current period t (concurrent surveillance cutoff t): [t-win+1, t]
        wk[f"rain_mm_accum_incl_{win}w"] = (
            g["rain_mm_sum_t"].rolling(win, min_periods=win).sum().reset_index(level=0, drop=True)
        )

    wk["temp_diurnal_roll_mean_prev_4w"] = (
        dtr_prev.groupby(wk["district_std"]).rolling(4, min_periods=4).mean().reset_index(level=0, drop=True)
    )
    wk["rh_mean_roll_prev_4w"] = (
        rh_prev.groupby(wk["district_std"]).rolling(4, min_periods=1).mean().reset_index(level=0, drop=True)
    )

    # ---- Leak-free Expanding Seasonal Rainfall Baseline within 2023-2025
    # 1. Expanding monthly rainfall average per (district_std, calendar_month) using strictly prior weeks (< period_idx)
    wk = wk.sort_values(["district_std", "period_idx"]).reset_index(drop=True)
    dm_prev_rain = wk.groupby(["district_std", "calendar_month"], sort=False)["rain_mm_sum_t"].shift(1)
    wk["rain_mm_expanding_month_baseline_prev"] = (
        dm_prev_rain.groupby([wk["district_std"], wk["calendar_month"]])
        .expanding(min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    # 2. Expanding seasonal rainfall average per (district_std, season) using strictly prior weeks (< period_idx)
    ds_prev_rain = wk.groupby(["district_std", "season"], sort=False)["rain_mm_sum_t"].shift(1)
    wk["rain_mm_expanding_season_baseline_prev"] = (
        ds_prev_rain.groupby([wk["district_std"], wk["season"]])
        .expanding(min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    # Deviation from expanding monthly & seasonal baselines (current period t and lagged t-1)
    wk["rain_mm_deviation_expanding_month_abs_t"] = (
        wk["rain_mm_sum_t"] - wk["rain_mm_expanding_month_baseline_prev"]
    )
    wk["rain_mm_deviation_expanding_month_pct_t"] = (
        (wk["rain_mm_sum_t"] - wk["rain_mm_expanding_month_baseline_prev"])
        / (wk["rain_mm_expanding_month_baseline_prev"].replace(0, np.nan))
        * 100.0
    )
    wk["rain_mm_deviation_expanding_month_abs_lag_1w"] = (
        wk["rain_mm_sum_lag_1w"] - wk["rain_mm_expanding_month_baseline_prev"]
    )
    wk["rain_mm_deviation_expanding_season_abs_t"] = (
        wk["rain_mm_sum_t"] - wk["rain_mm_expanding_season_baseline_prev"]
    )
    wk["rain_mm_deviation_expanding_season_abs_lag_1w"] = (
        wk["rain_mm_sum_lag_1w"] - wk["rain_mm_expanding_season_baseline_prev"]
    )

    # Year-over-Year (52-week) rainfall comparison
    wk["rain_mm_yoy_abs_diff_t"] = wk["rain_mm_sum_t"] - wk["rain_mm_sum_lag_52w"]
    wk["rain_mm_yoy_abs_diff_lag_1w"] = wk["rain_mm_sum_lag_1w"] - wk["rain_mm_sum_lag_53w"]

    return wk.sort_values(["district_std", "period_idx"]).reset_index(drop=True)
