"""JEEVAN-DRISHTI Phase 6: Time-Series Disease Forecasting Engine.

Provides multi-horizon weekly case-count forecasting (7-day = 1 week, 14-day = 2 weeks,
30-day = 4 weeks) with conformal uncertainty intervals, rolling-origin backtesting,
and a reusable FastAPI-ready forecasting service for Maharashtra (2023-2025 scope).
"""

FORECAST_MODEL_VERSION = "phase6-v1.0.0"

HORIZON_MAP = {
    "7d": {
        "horizon_days": 7,
        "horizon_weeks": 1,
        "label": "7_days_1_reporting_week_ahead",
        "description": "1 native IDSP reporting week ahead (7 days) from cutoff t-1; uses features strictly <= t-1.",
    },
    "14d": {
        "horizon_days": 14,
        "horizon_weeks": 2,
        "label": "14_days_2_reporting_weeks_ahead",
        "description": "2 native IDSP reporting weeks ahead (14 days) from cutoff t-2; excludes lag-1 features.",
    },
    "30d": {
        "horizon_days": 30,
        "horizon_weeks": 4,
        "label": "30_days_4_reporting_weeks_ahead",
        "description": "4 native IDSP reporting weeks ahead (28 days, closest weekly multiple to 30 days) from cutoff t-4; excludes lags 1..3.",
    },
}
