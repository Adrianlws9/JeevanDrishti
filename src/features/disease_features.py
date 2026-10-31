"""Disease history feature generation for the 2023-2025 IDSP weekly outbreak panel
(lags, rolling statistics, changes, seasonal baselines, YoY).

Strict Anti-Leakage & Anti-Fabrication Rules:
1. Scoped strictly to 2023-2025 weekly data (2023-W01 .. 2025-W52).
2. Missing / un-reported weeks are NEVER filled with zero cases. They remain NaN.
3. Every input feature for period t uses ONLY observations from strictly prior periods (<= t-1).
4. Target variables at period t (target_cases, target_deaths, outbreak_events_in_period) are
   NEVER used in any lag, rolling average, rolling std, percentage change, absolute change,
   seasonal baseline, or YoY input feature at period t.
"""
import numpy as np
import pandas as pd

from .temporal_features import build_weekly_calendar, add_temporal_features_weekly


def _dedup_and_aggregate_weekly_idsp(cleaned_disease_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 2023-2025 IDSP weekly outbreak records to the
    (district_std, disease_std, report_year, report_week) reporting-period grain
    while preserving raw sums and conflict flags."""
    idsp = cleaned_disease_df[cleaned_disease_df["source_dataset"] == "idsp_weekly"].copy()
    idsp["report_year"] = idsp["report_year"].astype(int)
    idsp["report_week"] = idsp["report_week"].astype(int)
    idsp["report_period_start"] = pd.to_datetime(idsp["report_period_start"])
    idsp["report_period_end"] = pd.to_datetime(idsp["report_period_end"])

    # Within a single (district_std, disease_std, report_year, report_week), if two rows have the
    # exact same (district_std, disease_std, cases) and were flagged as conflict/possible_duplicate
    # on consecutive PDF pages (e.g., Beed 2023-W42 5 cases repeated on p.15 and p.16),
    # we compute both the conservative deduplicated sum and the unadjusted raw sum.
    idsp = idsp.sort_values(["report_year", "report_week", "district_std", "disease_std", "record_id"])
    dup_same_cases = idsp.duplicated(
        subset=["report_year", "report_week", "district_std", "disease_std", "cases"], keep="first"
    ) & (idsp["duplicate_status"] != "unique")
    idsp["_cases_dedup"] = np.where(dup_same_cases, 0.0, idsp["cases"].astype(float))

    grp = idsp.groupby(["district_std", "disease_std", "report_year", "report_week"], as_index=False)
    agg = grp.agg(
        report_period_start=("report_period_start", "first"),
        report_period_end=("report_period_end", "first"),
        report_period_source=("report_period_source", "first"),
        disease_group=("disease_group", "first"),
        is_target_disease=("is_target_disease", "first"),
        target_cases=("_cases_dedup", "sum"),
        target_cases_unadjusted_sum=("cases", "sum"),
        target_cases_max_single_outbreak=("cases", "max"),
        target_deaths=("deaths", lambda s: s.sum(min_count=1)),
        outbreak_events_in_period=("record_id", "count"),
        distinct_outbreak_ids=("outbreak_id", "nunique"),
        has_duplicate_or_conflict_flag=("duplicate_status", lambda s: int((s != "unique").any())),
        source_record_ids=("record_id", lambda s: ";".join(s.astype(str))),
        source_outbreak_ids=("outbreak_id", lambda s: ";".join(s.dropna().astype(str).unique())),
    )
    return agg


def build_weekly_disease_features(cleaned_disease_df: pd.DataFrame) -> pd.DataFrame:
    """Build all disease history features for the 2023-2025 IDSP weekly outbreak panel."""
    agg = _dedup_and_aggregate_weekly_idsp(cleaned_disease_df)
    agg = add_temporal_features_weekly(agg)

    # Build a complete 156-week index per observed (district_std, disease_std) pair so calendar-week
    # lags (t-1, t-2, t-4, t-8, t-52, t-53) and rolling windows [t-W, t-1] align to exact calendar weeks
    # WITHOUT filling missing weeks with 0.
    cal = build_weekly_calendar(2023, 2025)[["period_idx"]]
    dd_pairs = agg[["district_std", "disease_std"]].drop_duplicates()
    grid = dd_pairs.merge(cal, how="cross").merge(
        agg[["district_std", "disease_std", "period_idx", "target_cases"]],
        on=["district_std", "disease_std", "period_idx"],
        how="left",
    )
    grid = grid.sort_values(["district_std", "disease_std", "period_idx"]).reset_index(drop=True)
    grid["has_outbreak"] = grid["target_cases"].notna().astype(int)

    g = grid.groupby(["district_std", "disease_std"], sort=False)

    # 1. Exact Calendar-Week Lags (NaN when no outbreak was reported in week t - k)
    for lag in (1, 2, 4, 8, 52, 53):
        grid[f"cases_lag_{lag}p"] = g["target_cases"].shift(lag)
        grid[f"outbreak_reported_lag_{lag}p"] = g["has_outbreak"].shift(lag)

    # 2. Rolling averages, sums, stds over [t-W, t-1] (strictly shifted by 1 period so t is excluded!)
    s_prev = g["target_cases"].shift(1)
    h_prev = g["has_outbreak"].shift(1)
    dd_keys = [grid["district_std"], grid["disease_std"]]

    for win in (2, 4, 8):
        grid[f"cases_roll_mean_{win}p"] = (
            s_prev.groupby(dd_keys).rolling(win, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
        )
        grid[f"cases_roll_sum_{win}p"] = (
            s_prev.groupby(dd_keys).rolling(win, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
        )
        grid[f"cases_roll_std_{win}p"] = (
            s_prev.groupby(dd_keys).rolling(win, min_periods=2).std().reset_index(level=[0, 1], drop=True)
        )
        grid[f"outbreak_weeks_count_prev_{win}p"] = (
            h_prev.groupby(dd_keys).rolling(win, min_periods=1).sum().reset_index(level=[0, 1], drop=True)
        )

    # 3. Leak-free absolute and percentage change at t-1 vs t-2 (never uses target_cases at t!)
    grid["cases_abs_change_lag1_lag2"] = grid["cases_lag_1p"] - grid["cases_lag_2p"]
    grid["cases_pct_change_lag1_lag2"] = (
        (grid["cases_lag_1p"] - grid["cases_lag_2p"]) / grid["cases_lag_2p"].replace(0, np.nan) * 100.0
    )

    # 4. Leak-free Year-over-Year (YoY) comparisons on the weekly grid
    grid["cases_yoy_lag1_abs_diff"] = grid["cases_lag_1p"] - grid["cases_lag_53p"]
    grid["cases_yoy_lag1_pct_diff"] = (
        (grid["cases_lag_1p"] - grid["cases_lag_53p"]) / grid["cases_lag_53p"].replace(0, np.nan) * 100.0
    )
    prev4_yoy = grid.groupby(["district_std", "disease_std"], sort=False)["outbreak_weeks_count_prev_4p"].shift(52)
    grid["outbreak_weeks_prev_4p_yoy_diff"] = grid["outbreak_weeks_count_prev_4p"] - prev4_yoy

    cal_feat_cols = [c for c in grid.columns if c not in ("target_cases", "has_outbreak")]
    out = agg.merge(grid[cal_feat_cols], on=["district_std", "disease_std", "period_idx"], how="left")

    # 5. Event-Sequence Lag Features (across strictly prior observed outbreaks for that district & disease in 2023-2025)
    out = out.sort_values(["district_std", "disease_std", "period_idx"]).reset_index(drop=True)
    ge = out.groupby(["district_std", "disease_std"], sort=False)

    out["prior_outbreak_count_in_panel"] = ge.cumcount()
    out["periods_since_prev_outbreak"] = out["period_idx"] - ge["period_idx"].shift(1)

    for lag in (1, 2, 4, 8):
        out[f"prev_outbreak_cases_lag_{lag}"] = ge["target_cases"].shift(lag)

    ev_prev = ge["target_cases"].shift(1)
    ev_keys = [out["district_std"], out["disease_std"]]
    for win in (2, 4, 8):
        out[f"prev_outbreak_roll_mean_{win}"] = (
            ev_prev.groupby(ev_keys).rolling(win, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
        )
        out[f"prev_outbreak_roll_std_{win}"] = (
            ev_prev.groupby(ev_keys).rolling(win, min_periods=2).std().reset_index(level=[0, 1], drop=True)
        )

    out["prev_outbreak_abs_change_1_2"] = out["prev_outbreak_cases_lag_1"] - out["prev_outbreak_cases_lag_2"]
    out["prev_outbreak_pct_change_1_2"] = (
        (out["prev_outbreak_cases_lag_1"] - out["prev_outbreak_cases_lag_2"])
        / out["prev_outbreak_cases_lag_2"].replace(0, np.nan)
        * 100.0
    )

    # 6. Leak-Free Expanding Seasonal Averages within 2023-2025 (strictly prior periods < period_idx)
    # 6a. District x Disease x Season expanding mean over strictly prior outbreaks
    out = out.sort_values(["district_std", "disease_std", "period_idx"]).reset_index(drop=True)
    dds_prev = out.groupby(["district_std", "disease_std", "season"], sort=False)["target_cases"].shift(1)
    out["seasonal_dd_season_mean_cases_prev"] = (
        dds_prev.groupby([out["district_std"], out["disease_std"], out["season"]])
        .expanding(min_periods=1)
        .mean()
        .reset_index(level=[0, 1, 2], drop=True)
    )

    # 6b. District x Disease x Calendar-Month expanding mean over strictly prior outbreaks
    ddm_prev = out.groupby(["district_std", "disease_std", "calendar_month"], sort=False)["target_cases"].shift(1)
    out["seasonal_dd_month_mean_cases_prev"] = (
        ddm_prev.groupby([out["district_std"], out["disease_std"], out["calendar_month"]])
        .expanding(min_periods=1)
        .mean()
        .reset_index(level=[0, 1, 2], drop=True)
    )

    # 6c. Statewide Disease x Season and Disease x Month expanding mean over strictly prior weeks (< period_idx)
    st_season_wk = (
        out.groupby(["disease_std", "season", "period_idx"], as_index=False)
        .agg(_wk_sum=("target_cases", "sum"), _wk_cnt=("target_cases", "count"))
        .sort_values(["disease_std", "season", "period_idx"])
        .reset_index(drop=True)
    )
    gs = st_season_wk.groupby(["disease_std", "season"], sort=False)
    cum_s_sum = gs["_wk_sum"].shift(1).groupby([st_season_wk["disease_std"], st_season_wk["season"]]).cumsum()
    cum_s_cnt = gs["_wk_cnt"].shift(1).groupby([st_season_wk["disease_std"], st_season_wk["season"]]).cumsum()
    st_season_wk["seasonal_state_disease_season_mean_prev"] = cum_s_sum / cum_s_cnt.replace(0, np.nan)

    st_month_wk = (
        out.groupby(["disease_std", "calendar_month", "period_idx"], as_index=False)
        .agg(_wk_sum=("target_cases", "sum"), _wk_cnt=("target_cases", "count"))
        .sort_values(["disease_std", "calendar_month", "period_idx"])
        .reset_index(drop=True)
    )
    gm = st_month_wk.groupby(["disease_std", "calendar_month"], sort=False)
    cum_m_sum = gm["_wk_sum"].shift(1).groupby([st_month_wk["disease_std"], st_month_wk["calendar_month"]]).cumsum()
    cum_m_cnt = gm["_wk_cnt"].shift(1).groupby([st_month_wk["disease_std"], st_month_wk["calendar_month"]]).cumsum()
    st_month_wk["seasonal_state_disease_month_mean_prev"] = cum_m_sum / cum_m_cnt.replace(0, np.nan)

    out = out.merge(
        st_season_wk[["disease_std", "season", "period_idx", "seasonal_state_disease_season_mean_prev"]],
        on=["disease_std", "season", "period_idx"],
        how="left",
    ).merge(
        st_month_wk[["disease_std", "calendar_month", "period_idx", "seasonal_state_disease_month_mean_prev"]],
        on=["disease_std", "calendar_month", "period_idx"],
        how="left",
    )

    out["prev_outbreak_vs_seasonal_diff"] = (
        out["prev_outbreak_cases_lag_1"] - out["seasonal_dd_season_mean_cases_prev"]
    )
    out["prev_outbreak_vs_seasonal_ratio"] = (
        out["prev_outbreak_cases_lag_1"] / out["seasonal_dd_season_mean_cases_prev"].replace(0, np.nan)
    )

    return out.sort_values(["report_year", "report_week", "district_std", "disease_std"]).reset_index(drop=True)
