"""Temporal feature generation for the 2023-2025 weekly reporting periods.

Rules:
- Strictly scoped to 2023-2025 weekly data (2023-W01 .. 2025-W52, 156 reporting weeks).
- Never assume weekly data is daily data; period duration and weekly grain are explicit.
- All temporal features depend strictly on the reporting period's calendar coordinates.
"""
import numpy as np
import pandas as pd

# Canonical Monday anchors verified in Phase 2 for the 156 consecutive reporting weeks of 2023-2025
WEEK1_ANCHORS = {
    2023: pd.Timestamp("2023-01-02"),
    2024: pd.Timestamp("2024-01-01"),
    2025: pd.Timestamp("2024-12-30"),
}


def build_weekly_calendar(start_year: int = 2023, end_year: int = 2025) -> pd.DataFrame:
    """Return the 156-week consecutive reporting calendar for 2023-W01 .. 2025-W52."""
    rows = []
    idx = 0
    for yr in range(start_year, end_year + 1):
        anchor = WEEK1_ANCHORS[yr]
        for wk in range(1, 53):
            w_start = anchor + pd.Timedelta(days=(wk - 1) * 7)
            w_end = w_start + pd.Timedelta(days=6)
            rows.append(
                dict(
                    period_idx=idx,
                    report_year=yr,
                    report_week=wk,
                    canonical_week_start=w_start,
                    canonical_week_end=w_end,
                )
            )
            idx += 1
    return pd.DataFrame(rows)


def classify_maharashtra_season(month: pd.Series) -> pd.Series:
    """IMD 4-season classification for Maharashtra:
    - Winter: Jan-Feb (1, 2)
    - Pre-Monsoon_Summer: Mar-May (3, 4, 5)
    - Southwest_Monsoon: Jun-Sep (6, 7, 8, 9)
    - Post-Monsoon: Oct-Dec (10, 11, 12)
    """
    conds = [
        month.isin([1, 2]),
        month.isin([3, 4, 5]),
        month.isin([6, 7, 8, 9]),
        month.isin([10, 11, 12]),
    ]
    choices = ["Winter", "Pre-Monsoon_Summer", "Southwest_Monsoon", "Post-Monsoon"]
    return pd.Series(np.select(conds, choices, default="Unknown"), index=month.index, dtype="object")


def add_temporal_features_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal features to a 2023-2025 weekly dataframe that has report_year, report_week,
    report_period_start, report_period_end."""
    out = df.copy()
    p_start = pd.to_datetime(out["report_period_start"])
    p_end = pd.to_datetime(out["report_period_end"])
    p_mid = p_start + (p_end - p_start) // 2

    out["temporal_grain"] = "weekly"
    out["period_duration_days"] = ((p_end - p_start).dt.days + 1).astype("Int64")
    out["period_idx"] = ((out["report_year"].astype(int) - 2023) * 52 + (out["report_week"].astype(int) - 1)).astype(int)
    out["calendar_year"] = p_mid.dt.year.astype(int)
    out["calendar_month"] = p_mid.dt.month.astype(int)
    out["calendar_quarter"] = p_mid.dt.quarter.astype(int)

    # Cyclical encodings for week-of-year (52) and month-of-year (12)
    wk_rad = 2.0 * np.pi * (out["report_week"].astype(float) - 1.0) / 52.0
    mo_rad = 2.0 * np.pi * (out["calendar_month"].astype(float) - 1.0) / 12.0
    out["week_sin"] = np.sin(wk_rad).round(6)
    out["week_cos"] = np.cos(wk_rad).round(6)
    out["month_sin"] = np.sin(mo_rad).round(6)
    out["month_cos"] = np.cos(mo_rad).round(6)

    # Seasonal and transition indicators
    out["season"] = classify_maharashtra_season(out["calendar_month"])
    out["is_winter"] = (out["season"] == "Winter").astype(int)
    out["is_pre_monsoon_summer"] = (out["season"] == "Pre-Monsoon_Summer").astype(int)
    out["is_monsoon"] = (out["season"] == "Southwest_Monsoon").astype(int)
    out["is_post_monsoon"] = (out["season"] == "Post-Monsoon").astype(int)

    # Peak transmission / transition windows in Maharashtra
    out["is_monsoon_onset_transition"] = out["report_week"].isin([21, 22, 23, 24, 25]).astype(int)
    out["is_monsoon_withdrawal_transition"] = out["report_week"].isin([38, 39, 40, 41, 42, 43]).astype(int)
    out["is_year_end_transition"] = out["report_week"].isin([50, 51, 52, 1, 2]).astype(int)
    return out
