"""Main orchestrator for the JEEVAN-DRISHTI Phase 3 Feature Engineering Engine (2023-2025 scope).

Run:
    python -m src.features.feature_pipeline
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from . import FEATURE_ENGINE_VERSION, FEATURE_DATASET_VERSION
from .disease_features import build_weekly_disease_features
from .climate_features import build_district_week_climate_features
from .geographic_features import add_geographic_features_weekly
from .feature_validation import (
    run_leakage_and_math_validation,
    build_missing_feature_report,
    build_feature_coverage_report,
)

PROJ = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJ / "data" / "processed"
RAW_DIR = PROJ / "data" / "raw"
FEATURES_DIR = PROJ / "data" / "features"
REPORTS_FE_DIR = PROJ / "reports" / "feature_engineering"
REGISTRY_PATH = PROJ / "model_dataset_registry.json"


class FeatureLog:
    def __init__(self):
        self.rows = []

    def add(self, step_name: str, module: str, input_rows: int, output_rows: int, features_added: int, detail: str):
        self.rows.append(
            dict(
                step_id=len(self.rows) + 1,
                timestamp=datetime.now().isoformat(timespec="seconds"),
                step_name=step_name,
                module=module,
                input_rows=input_rows,
                output_rows=output_rows,
                features_added=features_added,
                detail=detail,
            )
        )

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_feature_dictionary() -> pd.DataFrame:
    """Return the complete feature dictionary with descriptions, formulas, availability cutoffs,
    and missing-value treatment for every column in the 2023-2025 feature tables."""
    entries = [
        # Identifiers & Targets
        ("district_std", "Identifier", "Canonical Maharashtra district name (36 official districts)", "Identity", "Known at prediction time", "Never missing"),
        ("district_id", "Geographic", "Deterministic district identifier code (MH_D01 .. MH_D36)", "Lookup(district_std)", "Known at prediction time", "Never missing"),
        ("disease_std", "Identifier", "Standardized disease category from Phase 2", "Identity", "Known at prediction time", "Never missing in disease feature table"),
        ("disease_group", "Identifier", "Disease grouping: target, combined, or other", "Lookup(disease_std)", "Known at prediction time", "Never missing"),
        ("is_target_disease", "Identifier", "Boolean indicator for the 7 core JEEVAN-DRISHTI target diseases", "disease_std in TARGET_DISEASES", "Known at prediction time", "Never missing"),
        ("report_year", "Temporal", "IDSP reporting year (strictly 2023, 2024, or 2025)", "Reporting calendar", "Known at prediction time", "Never missing"),
        ("report_week", "Temporal", "IDSP reporting week within year (1 .. 52)", "Reporting calendar", "Known at prediction time", "Never missing"),
        ("period_idx", "Temporal", "Consecutive weekly reporting index across 2023-2025 (0 .. 155)", "(report_year - 2023)*52 + (report_week - 1)", "Known at prediction time", "Never missing"),
        ("report_period_start", "Temporal", "Start date of reporting week (YYYY-MM-DD)", "Source period start or verified Monday anchor", "Known at prediction time", "Never missing"),
        ("report_period_end", "Temporal", "End date of reporting week (YYYY-MM-DD)", "Source period end or verified Sunday anchor", "Known at prediction time", "Never missing"),
        ("canonical_week_start", "Temporal", "Canonical 7-day Monday start date for (report_year, report_week)", "WEEK1_ANCHOR + (report_week - 1)*7 days", "Known at prediction time", "Never missing"),
        ("canonical_week_end", "Temporal", "Canonical 7-day Sunday end date for (report_year, report_week)", "canonical_week_start + 6 days", "Known at prediction time", "Never missing"),
        ("report_period_source", "Metadata", "Origin of report_period_start/end (source, derived_week_convention, or canonical_calendar)", "Provenance tag", "Known at prediction time", "Never missing"),
        ("target_cases", "Target (Outcome)", "Deduplicated outbreak cases reported for (district_std, disease_std) in week t", "sum(cases) after same-page duplicate collision guard", "Observed at end of period t (TARGET ONLY - DO NOT USE AS INPUT FEATURE)", "Never missing in observed outbreak rows"),
        ("target_cases_unadjusted_sum", "Target (Outcome)", "Raw unadjusted sum of cases across all rows in week t", "sum(cases)", "Observed at end of period t (TARGET ONLY)", "Never missing"),
        ("target_cases_max_single_outbreak", "Target (Outcome)", "Maximum cases in a single outbreak alert within week t", "max(cases)", "Observed at end of period t (TARGET ONLY)", "Never missing"),
        ("target_deaths", "Target (Outcome)", "Total reported deaths in week t for (district_std, disease_std)", "sum(deaths, min_count=1)", "Observed at end of period t (TARGET ONLY)", "Kept as NaN if deaths were not reported; never zero-filled"),
        ("outbreak_events_in_period", "Target (Outcome)", "Number of outbreak alert rows in week t for (district_std, disease_std)", "count(record_id)", "Observed at end of period t (TARGET ONLY)", "Never missing"),
        ("distinct_outbreak_ids", "Metadata", "Count of distinct outbreak_id codes in week t", "nunique(outbreak_id)", "Observed at end of period t", "Never missing"),
        ("has_duplicate_or_conflict_flag", "Metadata", "1 if any constituent row in week t had a Phase 2 duplicate/conflict flag", "max(duplicate_status != 'unique')", "Observed at end of period t", "Never missing"),
        ("source_record_ids", "Metadata", "Semicolon-delimited Phase 2 record_id provenance list", "join(record_id)", "Provenance metadata", "Never missing"),
        ("source_outbreak_ids", "Metadata", "Semicolon-delimited IDSP outbreak_id provenance list", "join(outbreak_id)", "Provenance metadata", "Never missing"),

        # Temporal Features (Section 2C)
        ("temporal_grain", "Temporal", "Explicit temporal resolution indicator ('weekly')", "Constant 'weekly'", "Known at prediction time", "Never missing"),
        ("period_duration_days", "Temporal", "Duration of the reporting period in days (7 for standard weeks; 5-6 for 2 boundary weeks)", "(report_period_end - report_period_start).days + 1", "Known at prediction time", "Never missing"),
        ("calendar_year", "Temporal", "Calendar year of the reporting week midpoint", "midpoint(start, end).year", "Known at prediction time", "Never missing"),
        ("calendar_month", "Temporal", "Calendar month (1..12) of the reporting week midpoint", "midpoint(start, end).month", "Known at prediction time", "Never missing"),
        ("calendar_quarter", "Temporal", "Calendar quarter (1..4) of the reporting week midpoint", "midpoint(start, end).quarter", "Known at prediction time", "Never missing"),
        ("week_sin", "Temporal", "Sine cyclical encoding of report_week (52-week cycle)", "sin(2*pi*(report_week - 1)/52)", "Known at prediction time", "Never missing"),
        ("week_cos", "Temporal", "Cosine cyclical encoding of report_week (52-week cycle)", "cos(2*pi*(report_week - 1)/52)", "Known at prediction time", "Never missing"),
        ("month_sin", "Temporal", "Sine cyclical encoding of calendar_month (12-month cycle)", "sin(2*pi*(calendar_month - 1)/12)", "Known at prediction time", "Never missing"),
        ("month_cos", "Temporal", "Cosine cyclical encoding of calendar_month (12-month cycle)", "cos(2*pi*(calendar_month - 1)/12)", "Known at prediction time", "Never missing"),
        ("season", "Temporal", "Maharashtra IMD season (Winter, Pre-Monsoon_Summer, Southwest_Monsoon, Post-Monsoon)", "Lookup(calendar_month)", "Known at prediction time", "Never missing"),
        ("is_winter", "Temporal", "1 if Winter (Jan-Feb), else 0", "I(calendar_month in {1,2})", "Known at prediction time", "Never missing"),
        ("is_pre_monsoon_summer", "Temporal", "1 if Pre-Monsoon Summer (Mar-May), else 0", "I(calendar_month in {3,4,5})", "Known at prediction time", "Never missing"),
        ("is_monsoon", "Temporal", "Monsoon-period indicator: 1 if Southwest Monsoon (Jun-Sep), else 0", "I(calendar_month in {6,7,8,9})", "Known at prediction time", "Never missing"),
        ("is_post_monsoon", "Temporal", "1 if Post-Monsoon (Oct-Dec), else 0", "I(calendar_month in {10,11,12})", "Known at prediction time", "Never missing"),
        ("is_monsoon_onset_transition", "Temporal", "Seasonal transition indicator for Southwest Monsoon onset (weeks 21-25, late May to mid-Jun)", "I(report_week in 21..25)", "Known at prediction time", "Never missing"),
        ("is_monsoon_withdrawal_transition", "Temporal", "Seasonal transition indicator for Monsoon withdrawal (weeks 38-43, late Sep to Oct)", "I(report_week in 38..43)", "Known at prediction time", "Never missing"),
        ("is_year_end_transition", "Temporal", "Year-end transition indicator (weeks 50-52 and 1-2)", "I(report_week in {50,51,52,1,2})", "Known at prediction time", "Never missing"),

        # Disease History Features — Calendar-Week Aligned (Section 2A)
        ("cases_lag_1p", "Disease History", "Observed outbreak cases 1 reporting week earlier (t - 1)", "target_cases[t - 1]", "Available at t - 1 (strictly prior to period t)", "NaN if no outbreak was reported in week t - 1; never zero-filled"),
        ("cases_lag_2p", "Disease History", "Observed outbreak cases 2 reporting weeks earlier (t - 2)", "target_cases[t - 2]", "Available at t - 2", "NaN if no outbreak was reported in week t - 2; never zero-filled"),
        ("cases_lag_4p", "Disease History", "Observed outbreak cases 4 reporting weeks earlier (t - 4)", "target_cases[t - 4]", "Available at t - 4", "NaN if no outbreak was reported in week t - 4; never zero-filled"),
        ("cases_lag_8p", "Disease History", "Observed outbreak cases 8 reporting weeks earlier (t - 8)", "target_cases[t - 8]", "Available at t - 8", "NaN if no outbreak was reported in week t - 8; never zero-filled"),
        ("cases_lag_52p", "Disease History", "Year-over-Year baseline: observed outbreak cases 52 reporting weeks earlier (t - 52)", "target_cases[t - 52]", "Available at t - 52", "NaN in 2023 or if no outbreak was reported in week t - 52"),
        ("cases_lag_53p", "Disease History", "Observed outbreak cases 53 reporting weeks earlier (t - 53), used for lag-1 YoY diff", "target_cases[t - 53]", "Available at t - 53", "NaN in 2023 or if no outbreak was reported in week t - 53"),
        ("outbreak_reported_lag_1p", "Disease History", "Binary indicator: 1 if an outbreak was reported in week t - 1, else 0", "I(target_cases[t - 1] is not NaN)", "Available at t - 1", "NaN only if t - 1 < 2023-W01"),
        ("outbreak_reported_lag_2p", "Disease History", "Binary indicator: 1 if an outbreak was reported in week t - 2, else 0", "I(target_cases[t - 2] is not NaN)", "Available at t - 2", "NaN only if t - 2 < 2023-W01"),
        ("outbreak_reported_lag_4p", "Disease History", "Binary indicator: 1 if an outbreak was reported in week t - 4, else 0", "I(target_cases[t - 4] is not NaN)", "Available at t - 4", "NaN only if t - 4 < 2023-W01"),
        ("outbreak_reported_lag_8p", "Disease History", "Binary indicator: 1 if an outbreak was reported in week t - 8, else 0", "I(target_cases[t - 8] is not NaN)", "Available at t - 8", "NaN only if t - 8 < 2023-W01"),
        ("outbreak_reported_lag_52p", "Disease History", "Binary YoY indicator: 1 if an outbreak was reported in week t - 52, else 0", "I(target_cases[t - 52] is not NaN)", "Available at t - 52", "NaN for first 52 weeks (2023)"),
        ("outbreak_reported_lag_53p", "Disease History", "Binary indicator: 1 if an outbreak was reported in week t - 53, else 0", "I(target_cases[t - 53] is not NaN)", "Available at t - 53", "NaN for first 53 weeks"),
        ("cases_roll_mean_2p", "Disease History", "Rolling average of observed cases over prior 2 calendar weeks [t-2, t-1]", "mean(target_cases[t-2 : t-1])", "Available at t - 1 (shifted by 1 week; excludes t)", "NaN if no outbreak was reported in [t-2, t-1]"),
        ("cases_roll_mean_4p", "Disease History", "Rolling average of observed cases over prior 4 calendar weeks [t-4, t-1]", "mean(target_cases[t-4 : t-1])", "Available at t - 1 (shifted by 1 week; excludes t)", "NaN if no outbreak was reported in [t-4, t-1]"),
        ("cases_roll_mean_8p", "Disease History", "Rolling average of observed cases over prior 8 calendar weeks [t-8, t-1]", "mean(target_cases[t-8 : t-1])", "Available at t - 1 (shifted by 1 week; excludes t)", "NaN if no outbreak was reported in [t-8, t-1]"),
        ("cases_roll_sum_2p", "Disease History", "Rolling sum of observed cases over prior 2 calendar weeks [t-2, t-1]", "sum(target_cases[t-2 : t-1])", "Available at t - 1", "NaN if no outbreak was reported in [t-2, t-1]"),
        ("cases_roll_sum_4p", "Disease History", "Rolling sum of observed cases over prior 4 calendar weeks [t-4, t-1]", "sum(target_cases[t-4 : t-1])", "Available at t - 1", "NaN if no outbreak was reported in [t-4, t-1]"),
        ("cases_roll_sum_8p", "Disease History", "Rolling sum of observed cases over prior 8 calendar weeks [t-8, t-1]", "sum(target_cases[t-8 : t-1])", "Available at t - 1", "NaN if no outbreak was reported in [t-8, t-1]"),
        ("cases_roll_std_2p", "Disease History", "Rolling sample standard deviation of observed cases over [t-2, t-1]", "std(target_cases[t-2 : t-1], ddof=1)", "Available at t - 1", "NaN if < 2 observed outbreak weeks in [t-2, t-1]"),
        ("cases_roll_std_4p", "Disease History", "Rolling sample standard deviation of observed cases over [t-4, t-1]", "std(target_cases[t-4 : t-1], ddof=1)", "Available at t - 1", "NaN if < 2 observed outbreak weeks in [t-4, t-1]"),
        ("cases_roll_std_8p", "Disease History", "Rolling sample standard deviation of observed cases over [t-8, t-1]", "std(target_cases[t-8 : t-1], ddof=1)", "Available at t - 1", "NaN if < 2 observed outbreak weeks in [t-8, t-1]"),
        ("outbreak_weeks_count_prev_2p", "Disease History", "Count of outbreak-reporting weeks in prior 2 calendar weeks [t-2, t-1]", "sum(has_outbreak[t-2 : t-1])", "Available at t - 1", "NaN only at period_idx == 0"),
        ("outbreak_weeks_count_prev_4p", "Disease History", "Count of outbreak-reporting weeks in prior 4 calendar weeks [t-4, t-1]", "sum(has_outbreak[t-4 : t-1])", "Available at t - 1", "NaN only at period_idx == 0"),
        ("outbreak_weeks_count_prev_8p", "Disease History", "Count of outbreak-reporting weeks in prior 8 calendar weeks [t-8, t-1]", "sum(has_outbreak[t-8 : t-1])", "Available at t - 1", "NaN only at period_idx == 0"),
        ("cases_abs_change_lag1_lag2", "Disease History", "Leak-free absolute change in cases between week t-2 and week t-1", "cases_lag_1p - cases_lag_2p", "Available at t - 1 (excludes target at t)", "NaN unless both t-1 and t-2 had reported outbreaks"),
        ("cases_pct_change_lag1_lag2", "Disease History", "Leak-free percentage change in cases between week t-2 and week t-1 (%)", "(cases_lag_1p - cases_lag_2p) / cases_lag_2p * 100", "Available at t - 1 (excludes target at t)", "NaN unless both t-1 and t-2 had reported outbreaks"),
        ("cases_yoy_lag1_abs_diff", "Disease History", "Year-over-Year absolute case difference at week t-1 vs week t-53", "cases_lag_1p - cases_lag_53p", "Available at t - 1", "NaN unless both t-1 and t-53 had reported outbreaks"),
        ("cases_yoy_lag1_pct_diff", "Disease History", "Year-over-Year percentage case difference at week t-1 vs week t-53 (%)", "(cases_lag_1p - cases_lag_53p) / cases_lag_53p * 100", "Available at t - 1", "NaN unless both t-1 and t-53 had reported outbreaks"),
        ("outbreak_weeks_prev_4p_yoy_diff", "Disease History", "YoY difference in 4-week outbreak-week frequency compared to 52 weeks earlier", "outbreak_weeks_count_prev_4p[t] - outbreak_weeks_count_prev_4p[t-52]", "Available at t - 1", "NaN in first 52 weeks (2023)"),

        # Disease History Features — Event-Sequence Aligned (Section 2A)
        ("prior_outbreak_count_in_panel", "Disease History", "Number of strictly prior outbreak weeks observed for (district_std, disease_std) in 2023-2025", "cumcount() prior to t", "Available at t - 1", "Never missing (0 for first outbreak of that disease in that district)"),
        ("periods_since_prev_outbreak", "Disease History", "Number of reporting weeks elapsed since the previous outbreak of (district_std, disease_std)", "period_idx[e] - period_idx[e - 1]", "Available at t - 1", "NaN for the first outbreak of (district_std, disease_std)"),
        ("prev_outbreak_cases_lag_1", "Disease History", "Cases in the immediately preceding outbreak (e - 1) of (district_std, disease_std)", "target_cases[e - 1]", "Available at e - 1 <= t - 1", "NaN if < 1 prior outbreak in 2023-2025"),
        ("prev_outbreak_cases_lag_2", "Disease History", "Cases in the 2nd preceding outbreak (e - 2) of (district_std, disease_std)", "target_cases[e - 2]", "Available at e - 2 <= t - 2", "NaN if < 2 prior outbreaks in 2023-2025"),
        ("prev_outbreak_cases_lag_4", "Disease History", "Cases in the 4th preceding outbreak (e - 4) of (district_std, disease_std)", "target_cases[e - 4]", "Available at e - 4 <= t - 4", "NaN if < 4 prior outbreaks in 2023-2025"),
        ("prev_outbreak_cases_lag_8", "Disease History", "Cases in the 8th preceding outbreak (e - 8) of (district_std, disease_std)", "target_cases[e - 8]", "Available at e - 8 <= t - 8", "NaN if < 8 prior outbreaks in 2023-2025"),
        ("prev_outbreak_roll_mean_2", "Disease History", "Rolling average of cases across the prior 2 observed outbreaks [e-2, e-1]", "mean(target_cases[e-2 : e-1])", "Available at e - 1 <= t - 1", "NaN if 0 prior outbreaks"),
        ("prev_outbreak_roll_mean_4", "Disease History", "Rolling average of cases across the prior 4 observed outbreaks [e-4, e-1]", "mean(target_cases[e-4 : e-1])", "Available at e - 1 <= t - 1", "NaN if 0 prior outbreaks"),
        ("prev_outbreak_roll_mean_8", "Disease History", "Rolling average of cases across the prior 8 observed outbreaks [e-8, e-1]", "mean(target_cases[e-8 : e-1])", "Available at e - 1 <= t - 1", "NaN if 0 prior outbreaks"),
        ("prev_outbreak_roll_std_2", "Disease History", "Rolling standard deviation across the prior 2 observed outbreaks [e-2, e-1]", "std(target_cases[e-2 : e-1], ddof=1)", "Available at e - 1 <= t - 1", "NaN if < 2 prior outbreaks"),
        ("prev_outbreak_roll_std_4", "Disease History", "Rolling standard deviation across the prior 4 observed outbreaks [e-4, e-1]", "std(target_cases[e-4 : e-1], ddof=1)", "Available at e - 1 <= t - 1", "NaN if < 2 prior outbreaks"),
        ("prev_outbreak_roll_std_8", "Disease History", "Rolling standard deviation across the prior 8 observed outbreaks [e-8, e-1]", "std(target_cases[e-8 : e-1], ddof=1)", "Available at e - 1 <= t - 1", "NaN if < 2 prior outbreaks"),
        ("prev_outbreak_abs_change_1_2", "Disease History", "Leak-free absolute change between the 2nd prior outbreak (e-2) and 1st prior outbreak (e-1)", "prev_outbreak_cases_lag_1 - prev_outbreak_cases_lag_2", "Available at e - 1 <= t - 1", "NaN if < 2 prior outbreaks"),
        ("prev_outbreak_pct_change_1_2", "Disease History", "Leak-free percentage change between outbreak (e-2) and outbreak (e-1) (%)", "(prev_outbreak_cases_lag_1 - prev_outbreak_cases_lag_2) / prev_outbreak_cases_lag_2 * 100", "Available at e - 1 <= t - 1", "NaN if < 2 prior outbreaks"),
        ("seasonal_dd_season_mean_cases_prev", "Disease History", "Expanding seasonal average of cases for (district_std, disease_std, season) over strictly prior weeks", "expanding_mean(target_cases[s < t]) within (district, disease, season)", "Available at t - 1", "NaN if no prior outbreak in that district, disease, and season"),
        ("seasonal_dd_month_mean_cases_prev", "Disease History", "Expanding monthly average of cases for (district_std, disease_std, calendar_month) over strictly prior weeks", "expanding_mean(target_cases[s < t]) within (district, disease, month)", "Available at t - 1", "NaN if no prior outbreak in that district, disease, and month"),
        ("seasonal_state_disease_season_mean_prev", "Disease History", "Statewide expanding seasonal average of cases for (disease_std, season) over strictly prior weeks (< t)", "cum_cases_prior / cum_outbreaks_prior within (disease, season)", "Available at t - 1", "NaN only on the first week a disease appears in a season"),
        ("seasonal_state_disease_month_mean_prev", "Disease History", "Statewide expanding monthly average of cases for (disease_std, calendar_month) over strictly prior weeks (< t)", "cum_cases_prior / cum_outbreaks_prior within (disease, month)", "Available at t - 1", "NaN only on the first week a disease appears in a calendar month"),
        ("prev_outbreak_vs_seasonal_diff", "Disease History", "Difference between latest prior outbreak (e-1) and expanding district-season average", "prev_outbreak_cases_lag_1 - seasonal_dd_season_mean_cases_prev", "Available at t - 1", "NaN if no prior seasonal baseline"),
        ("prev_outbreak_vs_seasonal_ratio", "Disease History", "Ratio of latest prior outbreak (e-1) to expanding district-season average", "prev_outbreak_cases_lag_1 / seasonal_dd_season_mean_cases_prev", "Available at t - 1", "NaN if no prior seasonal baseline"),

        # Geographic, Demographic & District Burden Features (Section 2D)
        ("administrative_division", "Geographic", "Maharashtra revenue division (Konkan, Pune, Nashik, Marathwada, Amravati_Vidarbha_West, Nagpur_Vidarbha_East)", "Lookup(district_std)", "Known at prediction time", "Never missing"),
        ("is_coastal_district", "Geographic", "1 if Konkan coastal district, else 0", "I(district_std in COASTAL_DISTRICTS)", "Known at prediction time", "Never missing"),
        ("is_vidarbha", "Geographic", "1 if district is in Vidarbha (Amravati or Nagpur division), else 0", "I('Vidarbha' in administrative_division)", "Known at prediction time", "Never missing"),
        ("is_marathwada", "Geographic", "1 if district is in Marathwada division, else 0", "I(administrative_division == 'Marathwada')", "Known at prediction time", "Never missing"),
        ("district_lat", "Geographic", "District reference latitude (degrees North) from existing project data", "Lookup(district_std)", "Known at prediction time", "NaN for Mumbai Suburban (no coordinates in workspace; not invented)"),
        ("district_lon", "Geographic", "District reference longitude (degrees East) from existing project data", "Lookup(district_std)", "Known at prediction time", "NaN for Mumbai Suburban (no coordinates in workspace; not invented)"),
        ("has_coordinates", "Geographic", "1 if district has reference coordinates in workspace, else 0", "I(district_lat is not NaN)", "Known at prediction time", "Never missing"),
        ("district_population", "Demographic (Unsupported)", "District population denominator (UNSUPPORTED: no population file in project workspace)", "NaN (never fabricated)", "Unsupported without external data", "100% NaN by rule: no population dataset in workspace"),
        ("population_density", "Demographic (Unsupported)", "District population density per sq km (UNSUPPORTED: no population/area file in project workspace)", "NaN (never fabricated)", "Unsupported without external data", "100% NaN by rule: no population dataset in workspace"),
        ("cases_per_100k_pop", "Demographic (Unsupported)", "Population-normalized disease incidence per 100k (UNSUPPORTED without valid population denominator)", "NaN (never fabricated)", "Unsupported without external data", "100% NaN by rule: do not calculate normalized rates without valid denominator"),
        ("district_disease_prior_outbreaks_cum", "Geographic / Burden", "Cumulative outbreak events of (district_std, disease_std) in 2023-2025 strictly prior to week t", "cumsum(outbreak_events_in_period[s < t])", "Available at t - 1", "NaN on first appearance of (district_std, disease_std)"),
        ("district_disease_prior_cases_cum", "Geographic / Burden", "Cumulative cases of (district_std, disease_std) in 2023-2025 strictly prior to week t", "cumsum(target_cases[s < t])", "Available at t - 1", "NaN on first appearance of (district_std, disease_std)"),
        ("district_disease_prior_deaths_cum", "Geographic / Burden", "Cumulative deaths of (district_std, disease_std) in 2023-2025 strictly prior to week t", "cumsum(target_deaths[s < t])", "Available at t - 1", "NaN on first appearance"),
        ("district_disease_prior_mean_cases", "Geographic / Burden", "Expanding mean outbreak size for (district_std, disease_std) strictly prior to week t", "expanding_mean(target_cases[s < t])", "Available at t - 1", "NaN on first appearance"),
        ("district_disease_prior_max_cases", "Geographic / Burden", "Expanding maximum outbreak size for (district_std, disease_std) strictly prior to week t", "expanding_max(target_cases[s < t])", "Available at t - 1", "NaN on first appearance"),
        ("district_all_diseases_prior_cases_cum", "Geographic / Burden", "Cumulative cases across ALL diseases in district_std strictly prior to week t", "cumsum(district_weekly_cases[s < t])", "Available at t - 1", "NaN on the first outbreak week of that district"),
        ("district_all_diseases_prior_outbreaks_cum", "Geographic / Burden", "Cumulative outbreak alerts across ALL diseases in district_std strictly prior to week t", "cumsum(district_weekly_events[s < t])", "Available at t - 1", "NaN on the first outbreak week of that district"),

        # Climate Features (Section 2B)
        ("rain_mm_sum_t", "Climate", "Current-period total weekly rainfall (mm) during [report_period_start, report_period_end]", "sum(daily rain_mm in week t)", "Observed at end of week t", "NaN only for Mumbai Suburban (no coordinates)"),
        ("rain_mm_strict_nearest_sum_t", "Climate", "Weekly rainfall (mm) using strict unadjusted nearest grid node (NaN for Gadchiroli border node)", "sum(daily rain_mm_strict_nearest in week t)", "Observed at end of week t", "NaN for Mumbai Suburban and Gadchiroli border node"),
        ("rain_mm_mean_t", "Climate", "Current-period mean daily rainfall (mm/day) in week t", "mean(daily rain_mm in week t)", "Observed at end of week t", "NaN only for Mumbai Suburban"),
        ("rain_mm_max_t", "Climate", "Current-period maximum single-day rainfall (mm/day) in week t", "max(daily rain_mm in week t)", "Observed at end of week t", "NaN only for Mumbai Suburban"),
        ("rainy_days_t", "Climate", "Count of rainy days (>= 2.5 mm) in week t", "sum(I(daily rain_mm >= 2.5))", "Observed at end of week t", "NaN only for Mumbai Suburban"),
        ("heavy_rain_days_t", "Climate", "Count of heavy rainfall days (>= 64.5 mm) in week t", "sum(I(daily rain_mm >= 64.5))", "Observed at end of week t", "NaN only for Mumbai Suburban"),
        ("rain_valid_days_in_week", "Climate Metadata", "Number of valid daily rainfall observations in week t", "count(daily rain_mm)", "Observed at end of week t", "0 for Mumbai Suburban"),
        ("rain_spatial_source", "Climate Metadata", "Spatial extraction provenance for rainfall (nearest_grid_cell, adjacent_land_cell_19.75N_80.00E, or none_no_coordinates)", "Provenance tag", "Known at prediction time", "Never missing"),
        ("rain_mm_sum_lag_1w", "Climate", "Previous-period total weekly rainfall (mm) in week t - 1", "rain_mm_sum_t[t - 1]", "Available at t - 1", "NaN at period_idx == 0 or Mumbai Suburban"),
        ("rain_mm_sum_lag_2w", "Climate", "Lagged weekly rainfall (mm) in week t - 2", "rain_mm_sum_t[t - 2]", "Available at t - 2", "NaN for period_idx < 2 or Mumbai Suburban"),
        ("rain_mm_sum_lag_3w", "Climate", "Lagged weekly rainfall (mm) in week t - 3", "rain_mm_sum_t[t - 3]", "Available at t - 3", "NaN for period_idx < 3 or Mumbai Suburban"),
        ("rain_mm_sum_lag_4w", "Climate", "Lagged weekly rainfall (mm) in week t - 4", "rain_mm_sum_t[t - 4]", "Available at t - 4", "NaN for period_idx < 4 or Mumbai Suburban"),
        ("rain_mm_sum_lag_8w", "Climate", "Lagged weekly rainfall (mm) in week t - 8", "rain_mm_sum_t[t - 8]", "Available at t - 8", "NaN for period_idx < 8 or Mumbai Suburban"),
        ("rain_mm_sum_lag_52w", "Climate", "Prior-year same-week rainfall (mm) in week t - 52", "rain_mm_sum_t[t - 52]", "Available at t - 52", "NaN for period_idx < 52 or Mumbai Suburban"),
        ("rain_mm_sum_lag_53w", "Climate", "Weekly rainfall (mm) in week t - 53", "rain_mm_sum_t[t - 53]", "Available at t - 53", "NaN for period_idx < 53 or Mumbai Suburban"),
        ("rain_mm_mean_lag_1w", "Climate", "Previous-period mean daily rainfall (mm/day) in week t - 1", "rain_mm_mean_t[t - 1]", "Available at t - 1", "NaN at period_idx == 0 or Mumbai Suburban"),
        ("rain_mm_max_lag_1w", "Climate", "Previous-period peak daily rainfall (mm/day) in week t - 1", "rain_mm_max_t[t - 1]", "Available at t - 1", "NaN at period_idx == 0 or Mumbai Suburban"),
        ("rainy_days_lag_1w", "Climate", "Previous-period count of rainy days (>= 2.5 mm) in week t - 1", "rainy_days_t[t - 1]", "Available at t - 1", "NaN at period_idx == 0 or Mumbai Suburban"),
        ("rain_mm_accum_prev_2w", "Climate", "Rainfall accumulated over prior 2 weeks [t-2, t-1] (mm, strictly prior to t)", "sum(rain_mm_sum_t[t-2 : t-1])", "Available at t - 1", "NaN for period_idx < 2 or Mumbai Suburban"),
        ("rain_mm_accum_prev_4w", "Climate", "Rainfall accumulated over prior 4 weeks [t-4, t-1] (mm, strictly prior to t)", "sum(rain_mm_sum_t[t-4 : t-1])", "Available at t - 1", "NaN for period_idx < 4 or Mumbai Suburban"),
        ("rain_mm_accum_prev_8w", "Climate", "Rainfall accumulated over prior 8 weeks [t-8, t-1] (mm, strictly prior to t)", "sum(rain_mm_sum_t[t-8 : t-1])", "Available at t - 1", "NaN for period_idx < 8 or Mumbai Suburban"),
        ("rain_mm_roll_mean_prev_2w", "Climate", "Rolling mean weekly rainfall over prior 2 weeks [t-2, t-1] (mm/week)", "mean(rain_mm_sum_t[t-2 : t-1])", "Available at t - 1", "NaN for period_idx < 2 or Mumbai Suburban"),
        ("rain_mm_roll_mean_prev_4w", "Climate", "Rolling mean weekly rainfall over prior 4 weeks [t-4, t-1] (mm/week)", "mean(rain_mm_sum_t[t-4 : t-1])", "Available at t - 1", "NaN for period_idx < 4 or Mumbai Suburban"),
        ("rain_mm_roll_mean_prev_8w", "Climate", "Rolling mean weekly rainfall over prior 8 weeks [t-8, t-1] (mm/week)", "mean(rain_mm_sum_t[t-8 : t-1])", "Available at t - 1", "NaN for period_idx < 8 or Mumbai Suburban"),
        ("rain_mm_accum_incl_2w", "Climate", "Rainfall accumulated over 2 weeks inclusive of current week [t-1, t] (mm)", "sum(rain_mm_sum_t[t-1 : t])", "Observed at end of week t", "NaN for period_idx < 1 or Mumbai Suburban"),
        ("rain_mm_accum_incl_4w", "Climate", "Rainfall accumulated over 4 weeks inclusive of current week [t-3, t] (mm)", "sum(rain_mm_sum_t[t-3 : t])", "Observed at end of week t", "NaN for period_idx < 3 or Mumbai Suburban"),
        ("rain_mm_accum_incl_8w", "Climate", "Rainfall accumulated over 8 weeks inclusive of current week [t-7, t] (mm)", "sum(rain_mm_sum_t[t-7 : t])", "Observed at end of week t", "NaN for period_idx < 7 or Mumbai Suburban"),
        ("tmin_c_mean_t", "Climate", "Weekly average of daily minimum temperature (deg C) in week t", "mean(daily tmin_c in week t)", "Observed at end of week t", "NaN only for Mumbai Suburban"),
        ("tmin_c_min_t", "Climate", "Lowest daily minimum temperature (deg C) in week t", "min(daily tmin_c in week t)", "Observed at end of week t", "NaN only for Mumbai Suburban"),
        ("tmax_c_mean_t", "Climate", "Weekly average of daily maximum temperature (deg C) in week t", "mean(daily tmax_c in week t)", "Observed at end of week t", "NaN only for Mumbai Suburban"),
        ("tmax_c_max_t", "Climate", "Highest daily maximum temperature (deg C) in week t", "max(daily tmax_c in week t)", "Observed at end of week t", "NaN only for Mumbai Suburban"),
        ("tavg_c_mean_t", "Climate", "Weekly average temperature (deg C, from derived daily (tmin+tmax)/2) in week t", "mean(daily tavg_derived_c in week t)", "Observed at end of week t", "NaN only for Mumbai Suburban"),
        ("tavg_c_within_week_std_t", "Climate", "Within-week day-to-day standard deviation of average temperature (deg C)", "std(daily tavg_derived_c in week t)", "Observed at end of week t", "NaN only for Mumbai Suburban"),
        ("temp_diurnal_range_mean_t", "Climate", "Temperature variation: weekly mean diurnal temperature range (tmax - tmin, deg C)", "mean(daily tmax_c - tmin_c in week t)", "Observed at end of week t", "NaN only for Mumbai Suburban"),
        ("temp_weekly_extreme_spread_t", "Climate", "Temperature variation: weekly extreme spread (max(tmax) - min(tmin), deg C)", "tmax_c_max_t - tmin_c_min_t", "Observed at end of week t", "NaN only for Mumbai Suburban"),
        ("temp_valid_days_in_week", "Climate Metadata", "Number of valid daily temperature observations in week t", "count(daily tavg_derived_c)", "Observed at end of week t", "0 for Mumbai Suburban"),
        ("tavg_c_mean_lag_1w", "Climate", "Lagged weekly average temperature (deg C) in week t - 1", "tavg_c_mean_t[t - 1]", "Available at t - 1", "NaN at period_idx == 0 or Mumbai Suburban"),
        ("tavg_c_mean_lag_2w", "Climate", "Lagged weekly average temperature (deg C) in week t - 2", "tavg_c_mean_t[t - 2]", "Available at t - 2", "NaN for period_idx < 2 or Mumbai Suburban"),
        ("tavg_c_mean_lag_4w", "Climate", "Lagged weekly average temperature (deg C) in week t - 4", "tavg_c_mean_t[t - 4]", "Available at t - 4", "NaN for period_idx < 4 or Mumbai Suburban"),
        ("tavg_c_mean_lag_8w", "Climate", "Lagged weekly average temperature (deg C) in week t - 8", "tavg_c_mean_t[t - 8]", "Available at t - 8", "NaN for period_idx < 8 or Mumbai Suburban"),
        ("tmin_c_mean_lag_1w", "Climate", "Lagged weekly minimum temperature (deg C) in week t - 1", "tmin_c_mean_t[t - 1]", "Available at t - 1", "NaN at period_idx == 0 or Mumbai Suburban"),
        ("tmin_c_mean_lag_2w", "Climate", "Lagged weekly minimum temperature (deg C) in week t - 2", "tmin_c_mean_t[t - 2]", "Available at t - 2", "NaN for period_idx < 2 or Mumbai Suburban"),
        ("tmin_c_mean_lag_4w", "Climate", "Lagged weekly minimum temperature (deg C) in week t - 4", "tmin_c_mean_t[t - 4]", "Available at t - 4", "NaN for period_idx < 4 or Mumbai Suburban"),
        ("tmin_c_mean_lag_8w", "Climate", "Lagged weekly minimum temperature (deg C) in week t - 8", "tmin_c_mean_t[t - 8]", "Available at t - 8", "NaN for period_idx < 8 or Mumbai Suburban"),
        ("tmax_c_mean_lag_1w", "Climate", "Lagged weekly maximum temperature (deg C) in week t - 1", "tmax_c_mean_t[t - 1]", "Available at t - 1", "NaN at period_idx == 0 or Mumbai Suburban"),
        ("tmax_c_mean_lag_2w", "Climate", "Lagged weekly maximum temperature (deg C) in week t - 2", "tmax_c_mean_t[t - 2]", "Available at t - 2", "NaN for period_idx < 2 or Mumbai Suburban"),
        ("tmax_c_mean_lag_4w", "Climate", "Lagged weekly maximum temperature (deg C) in week t - 4", "tmax_c_mean_t[t - 4]", "Available at t - 4", "NaN for period_idx < 4 or Mumbai Suburban"),
        ("tmax_c_mean_lag_8w", "Climate", "Lagged weekly maximum temperature (deg C) in week t - 8", "tmax_c_mean_t[t - 8]", "Available at t - 8", "NaN for period_idx < 8 or Mumbai Suburban"),
        ("temp_diurnal_range_lag_1w", "Climate", "Lagged weekly mean diurnal temperature range (deg C) in week t - 1", "temp_diurnal_range_mean_t[t - 1]", "Available at t - 1", "NaN at period_idx == 0 or Mumbai Suburban"),
        ("temp_diurnal_range_lag_2w", "Climate", "Lagged weekly mean diurnal temperature range (deg C) in week t - 2", "temp_diurnal_range_mean_t[t - 2]", "Available at t - 2", "NaN for period_idx < 2 or Mumbai Suburban"),
        ("temp_diurnal_range_lag_4w", "Climate", "Lagged weekly mean diurnal temperature range (deg C) in week t - 4", "temp_diurnal_range_mean_t[t - 4]", "Available at t - 4", "NaN for period_idx < 4 or Mumbai Suburban"),
        ("temp_diurnal_range_lag_8w", "Climate", "Lagged weekly mean diurnal temperature range (deg C) in week t - 8", "temp_diurnal_range_mean_t[t - 8]", "Available at t - 8", "NaN for period_idx < 8 or Mumbai Suburban"),
        ("tavg_c_roll_mean_prev_2w", "Climate", "Rolling average temperature over prior 2 weeks [t-2, t-1] (deg C)", "mean(tavg_c_mean_t[t-2 : t-1])", "Available at t - 1", "NaN for period_idx < 2 or Mumbai Suburban"),
        ("tavg_c_roll_mean_prev_4w", "Climate", "Rolling average temperature over prior 4 weeks [t-4, t-1] (deg C)", "mean(tavg_c_mean_t[t-4 : t-1])", "Available at t - 1", "NaN for period_idx < 4 or Mumbai Suburban"),
        ("tavg_c_roll_mean_prev_8w", "Climate", "Rolling average temperature over prior 8 weeks [t-8, t-1] (deg C)", "mean(tavg_c_mean_t[t-8 : t-1])", "Available at t - 1", "NaN for period_idx < 8 or Mumbai Suburban"),
        ("tavg_c_roll_std_prev_2w", "Climate", "Rolling temperature standard deviation over prior 2 weeks [t-2, t-1] (deg C)", "std(tavg_c_mean_t[t-2 : t-1])", "Available at t - 1", "NaN for period_idx < 2 or Mumbai Suburban"),
        ("tavg_c_roll_std_prev_4w", "Climate", "Rolling temperature standard deviation over prior 4 weeks [t-4, t-1] (deg C)", "std(tavg_c_mean_t[t-4 : t-1])", "Available at t - 1", "NaN for period_idx < 4 or Mumbai Suburban"),
        ("tavg_c_roll_std_prev_8w", "Climate", "Rolling temperature standard deviation over prior 8 weeks [t-8, t-1] (deg C)", "std(tavg_c_mean_t[t-8 : t-1])", "Available at t - 1", "NaN for period_idx < 8 or Mumbai Suburban"),
        ("temp_diurnal_roll_mean_prev_4w", "Climate", "Rolling 4-week mean diurnal temperature range over [t-4, t-1] (deg C)", "mean(temp_diurnal_range_mean_t[t-4 : t-1])", "Available at t - 1", "NaN for period_idx < 4 or Mumbai Suburban"),
        ("rh_mean_t", "Climate", "Weekly average relative humidity (%) in week t (CWPRS Pune station only)", "mean(daily rh_mean in week t)", "Observed at end of week t", "Available ONLY for Pune during 2024-W19..2025-W24; NaN for all other districts/weeks"),
        ("rh_min_t", "Climate", "Weekly minimum relative humidity (%) in week t (Pune only)", "min(daily rh_min in week t)", "Observed at end of week t", "Pune only (2024-W19..2025-W24); NaN elsewhere"),
        ("rh_max_t", "Climate", "Weekly maximum relative humidity (%) in week t (Pune only)", "max(daily rh_max in week t)", "Observed at end of week t", "Pune only (2024-W19..2025-W24); NaN elsewhere"),
        ("rh_valid_days_in_week", "Climate Metadata", "Count of days with valid CWPRS humidity telemetry in week t", "count(daily rh_mean)", "Observed at end of week t", "0 outside Pune 2024-W19..2025-W24"),
        ("rh_mean_lag_1w", "Climate", "Lagged weekly relative humidity (%) in week t - 1 (Pune only)", "rh_mean_t[t - 1]", "Available at t - 1", "Pune only; NaN elsewhere"),
        ("rh_mean_lag_2w", "Climate", "Lagged weekly relative humidity (%) in week t - 2 (Pune only)", "rh_mean_t[t - 2]", "Available at t - 2", "Pune only; NaN elsewhere"),
        ("rh_mean_lag_4w", "Climate", "Lagged weekly relative humidity (%) in week t - 4 (Pune only)", "rh_mean_t[t - 4]", "Available at t - 4", "Pune only; NaN elsewhere"),
        ("rh_mean_lag_8w", "Climate", "Lagged weekly relative humidity (%) in week t - 8 (Pune only)", "rh_mean_t[t - 8]", "Available at t - 8", "Pune only; NaN elsewhere"),
        ("rh_mean_roll_prev_4w", "Climate", "Rolling 4-week average relative humidity over [t-4, t-1] (%) (Pune only)", "mean(rh_mean_t[t-4 : t-1])", "Available at t - 1", "Pune only; NaN elsewhere"),
        ("rain_mm_expanding_month_baseline_prev", "Climate", "Expanding historical monthly rainfall average (mm/week) for (district_std, calendar_month) over prior weeks < t", "expanding_mean(rain_mm_sum_t[s < t]) within (district, month)", "Available at t - 1", "NaN on first week of each calendar month in 2023 or Mumbai Suburban"),
        ("rain_mm_expanding_season_baseline_prev", "Climate", "Expanding historical seasonal rainfall average (mm/week) for (district_std, season) over prior weeks < t", "expanding_mean(rain_mm_sum_t[s < t]) within (district, season)", "Available at t - 1", "NaN on first week of each season in 2023 or Mumbai Suburban"),
        ("rain_mm_deviation_expanding_month_abs_t", "Climate", "Rainfall deviation (mm) in week t from expanding district-month baseline", "rain_mm_sum_t - rain_mm_expanding_month_baseline_prev", "Observed at end of week t", "NaN on first week of month in 2023 or Mumbai Suburban"),
        ("rain_mm_deviation_expanding_month_pct_t", "Climate", "Percentage rainfall deviation (%) in week t from expanding district-month baseline", "(rain_mm_sum_t - baseline) / baseline * 100", "Observed at end of week t", "NaN when baseline is 0/NaN or Mumbai Suburban"),
        ("rain_mm_deviation_expanding_month_abs_lag_1w", "Climate", "Lagged rainfall deviation (mm) in week t - 1 from expanding district-month baseline", "rain_mm_sum_lag_1w - rain_mm_expanding_month_baseline_prev", "Available at t - 1 (strictly prior to t)", "NaN on first week of month in 2023 or Mumbai Suburban"),
        ("rain_mm_deviation_expanding_season_abs_t", "Climate", "Rainfall deviation (mm) in week t from expanding district-season baseline", "rain_mm_sum_t - rain_mm_expanding_season_baseline_prev", "Observed at end of week t", "NaN on first week of season in 2023 or Mumbai Suburban"),
        ("rain_mm_deviation_expanding_season_abs_lag_1w", "Climate", "Lagged rainfall deviation (mm) in week t - 1 from expanding district-season baseline", "rain_mm_sum_lag_1w - rain_mm_expanding_season_baseline_prev", "Available at t - 1 (strictly prior to t)", "NaN on first week of season in 2023 or Mumbai Suburban"),
        ("rain_mm_yoy_abs_diff_t", "Climate", "Year-over-Year rainfall difference (mm) between week t and week t - 52", "rain_mm_sum_t - rain_mm_sum_lag_52w", "Observed at end of week t", "NaN in 2023 (period_idx < 52) or Mumbai Suburban"),
        ("rain_mm_yoy_abs_diff_lag_1w", "Climate", "Lagged Year-over-Year rainfall difference (mm) between week t - 1 and week t - 53", "rain_mm_sum_lag_1w - rain_mm_sum_lag_53w", "Available at t - 1", "NaN for period_idx < 53 or Mumbai Suburban"),
    ]
    return pd.DataFrame(
        entries,
        columns=[
            "feature_name",
            "category",
            "description",
            "formula",
            "availability_cutoff",
            "missing_handling_note",
        ],
    )


def run_feature_pipeline(verbose: bool = True) -> dict:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_FE_DIR.mkdir(parents=True, exist_ok=True)
    # Remove any legacy pre-2023 feature file if created before scope refinement
    legacy_hist = FEATURES_DIR / "historical_disease_features.csv"
    if legacy_hist.exists():
        legacy_hist.unlink()

    flog = FeatureLog()

    # 1. Load Phase 2 cleaned datasets
    cleaned_disease = pd.read_csv(PROCESSED_DIR / "cleaned_disease_data.csv")
    cleaned_climate = pd.read_csv(PROCESSED_DIR / "cleaned_climate_data.csv")
    raw_rain_nc = RAW_DIR / "imdweb_5dgbAskl.nc"

    idsp_rows = int((cleaned_disease["source_dataset"] == "idsp_weekly").sum())
    flog.add(
        "load_phase2_inputs",
        "feature_pipeline",
        idsp_rows + len(cleaned_climate),
        idsp_rows + len(cleaned_climate),
        0,
        f"Loaded 2023-2025 IDSP weekly rows={idsp_rows} and 2023-2025 daily climate rows={len(cleaned_climate)}",
    )

    # 2. Build 2023-2025 weekly disease history + temporal features
    weekly_dis = build_weekly_disease_features(cleaned_disease)
    flog.add(
        "build_weekly_disease_and_temporal_features",
        "disease_features / temporal_features",
        idsp_rows,
        len(weekly_dis),
        len(weekly_dis.columns),
        "Aggregated 570 IDSP rows (2023-2025) to (district_std, disease_std, report_year, report_week) and computed calendar-week lags, event-sequence lags, rolling stats, changes, expanding seasonal baselines, and YoY features",
    )

    # 3. Add geographic & expanding 2023-2025 district burden features
    cols_before_geo = len(weekly_dis.columns)
    weekly_dis_geo = add_geographic_features_weekly(weekly_dis, cleaned_climate)
    flog.add(
        "add_geographic_and_burden_features",
        "geographic_features",
        len(weekly_dis),
        len(weekly_dis_geo),
        len(weekly_dis_geo.columns) - cols_before_geo,
        "Added district IDs, administrative divisions, coastal/regional indicators, coordinates, expanding prior-week district disease burden, and explicit NaN population-normalized columns",
    )

    # 4. Build 2023-2025 district x reporting-week climate features (36 districts x 156 weeks = 5,616 rows)
    climate_wk = build_district_week_climate_features(cleaned_climate, cleaned_disease, raw_rain_nc)
    climate_wk_full = add_geographic_features_weekly(climate_wk, cleaned_climate)
    flog.add(
        "build_district_week_climate_features",
        "climate_features",
        len(cleaned_climate),
        len(climate_wk_full),
        len(climate_wk_full.columns),
        "Aggregated 39,456 daily climate rows (2023-2025) to 36 districts x 156 reporting weeks (5,616 rows); rescued Gadchiroli rainfall from adjacent interior land cell (19.75N, 80.00E); computed current, lagged, accumulated (2,4,8w), diurnal/spread, and expanding seasonal deviation features",
    )

    # 5. Align and join climate features onto the 2023-2025 weekly disease outbreak panel
    clim_only_cols = [
        c for c in climate_wk.columns
        if c not in weekly_dis_geo.columns or c in ("district_std", "period_idx")
    ]
    weekly_joined = weekly_dis_geo.merge(
        climate_wk[clim_only_cols], on=["district_std", "period_idx"], how="left"
    )
    weekly_joined = weekly_joined.sort_values(
        ["report_year", "report_week", "district_std", "disease_std"]
    ).reset_index(drop=True)
    flog.add(
        "join_disease_and_climate_weekly_2023_2025",
        "feature_pipeline",
        len(weekly_dis_geo),
        len(weekly_joined),
        len(weekly_joined.columns) - len(weekly_dis_geo.columns),
        f"Aligned 2023-2025 weekly disease panel ({len(weekly_joined)} rows) with district-week climate features on (district_std, period_idx); total columns={len(weekly_joined.columns)}",
    )

    # 6. Run automated mathematical & synthetic perturbation leakage validation
    fdict = build_feature_dictionary()
    val_df = run_leakage_and_math_validation(
        weekly_joined, climate_wk_full, cleaned_disease, cleaned_climate, raw_rain_nc
    )
    all_passed = bool(val_df["passed"].all())
    flog.add(
        "validate_features_and_leakage",
        "feature_validation",
        len(weekly_joined),
        len(val_df),
        0,
        f"Executed {len(val_df)} automated checks (lag alignment, rolling shift, period match, synthetic future perturbation, anti-fabrication); all_passed={all_passed}",
    )
    if not all_passed:
        failed = val_df[~val_df["passed"]]
        raise RuntimeError(f"Feature validation failed:\n{failed.to_string()}")

    # 7. Save processed feature datasets to data/features/
    p_weekly = FEATURES_DIR / "disease_climate_weekly_features.csv"
    p_weekly_2325 = FEATURES_DIR / "disease_climate_weekly_features_2023_2025.csv"
    p_clim_wk = FEATURES_DIR / "district_week_climate_features.csv"
    p_clim_wk_2325 = FEATURES_DIR / "district_week_climate_features_2023_2025.csv"

    weekly_joined.to_csv(p_weekly, index=False, date_format="%Y-%m-%d")
    weekly_joined.to_csv(p_weekly_2325, index=False, date_format="%Y-%m-%d")
    climate_wk_full.to_csv(p_clim_wk, index=False, date_format="%Y-%m-%d")
    climate_wk_full.to_csv(p_clim_wk_2325, index=False, date_format="%Y-%m-%d")

    # 8. Build and write reports to reports/feature_engineering/
    miss_df = build_missing_feature_report(
        {
            "disease_climate_weekly_features_2023_2025": weekly_joined,
            "district_week_climate_features_2023_2025": climate_wk_full,
        },
        fdict,
    )
    cov_df = build_feature_coverage_report(weekly_joined, climate_wk_full)

    p_fdict_csv = REPORTS_FE_DIR / "feature_dictionary.csv"
    p_fdict_md = REPORTS_FE_DIR / "feature_dictionary.md"
    p_flog = REPORTS_FE_DIR / "feature_generation_log.csv"
    p_miss = REPORTS_FE_DIR / "missing_feature_report.csv"
    p_leak_csv = REPORTS_FE_DIR / "leakage_validation_report.csv"
    p_leak_md = REPORTS_FE_DIR / "leakage_validation_report.md"
    p_cov = REPORTS_FE_DIR / "feature_coverage_report.csv"
    p_sum = REPORTS_FE_DIR / "feature_engineering_summary.md"

    fdict.to_csv(p_fdict_csv, index=False)
    miss_df.to_csv(p_miss, index=False)
    val_df.to_csv(p_leak_csv, index=False)
    cov_df.to_csv(p_cov, index=False)

    # Write markdown dictionary, leakage report, and summary
    p_fdict_md.write_text(_format_dictionary_md(fdict), encoding="utf-8")
    p_leak_md.write_text(_format_leakage_md(val_df), encoding="utf-8")
    summary_md = _format_summary_md(weekly_joined, climate_wk_full, fdict, val_df, cov_df, miss_df)
    p_sum.write_text(summary_md, encoding="utf-8")

    flog.add(
        "write_feature_datasets_and_reports",
        "feature_pipeline",
        len(weekly_joined),
        len(weekly_joined),
        0,
        f"Saved feature datasets to data/features/ and reports to reports/feature_engineering/",
    )
    flog.to_df().to_csv(p_flog, index=False)

    # 9. Update model_dataset_registry.json
    _update_registry(
        [
            ("disease_climate_weekly_features_2023_2025", p_weekly, weekly_joined, "Integrated 2023-2025 weekly disease outbreak + climate + temporal + geographic feature matrix"),
            ("district_week_climate_features_2023_2025", p_clim_wk, climate_wk_full, "Complete 2023-2025 36-district x 156-week climate, temporal, and geographic feature grid"),
        ],
        [p_fdict_csv, p_fdict_md, p_flog, p_miss, p_leak_csv, p_leak_md, p_cov, p_sum],
    )

    if verbose:
        print(summary_md[:3500])

    return {
        "weekly_joined": weekly_joined,
        "climate_weekly": climate_wk_full,
        "validation": val_df,
        "coverage": cov_df,
        "missing": miss_df,
        "dictionary": fdict,
    }


def _format_dictionary_md(fdict: pd.DataFrame) -> str:
    lines = [
        "# JEEVAN-DRISHTI Phase 3: Feature Dictionary (2023–2025 Scope)\n",
        f"Total documented columns: **{len(fdict)}**.\n",
        "| Feature Name | Category | Description | Formula | Availability Cutoff | Missing-Value Handling |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in fdict.iterrows():
        lines.append(
            f"| `{r.feature_name}` | {r.category} | {r.description} | `{r.formula}` | {r.availability_cutoff} | {r.missing_handling_note} |"
        )
    return "\n".join(lines) + "\n"


def _format_leakage_md(val_df: pd.DataFrame) -> str:
    lines = [
        "# JEEVAN-DRISHTI Phase 3: Data Leakage & Mathematical Validation Report (2023–2025)\n",
        f"All **{len(val_df)} / {len(val_df)}** automated validation checks passed (`status = PASS`).\n",
        "| Category | Check Name | Status | Observed Detail | Requirement Reference |",
        "|---|---|---|---|---|",
    ]
    for _, r in val_df.iterrows():
        lines.append(
            f"| {r.category} | `{r.check_name}` | **{r.status}** | {r.observed_detail} | {r.requirement_ref} |"
        )
    return "\n".join(lines) + "\n"


def _format_summary_md(
    weekly_joined: pd.DataFrame,
    climate_wk: pd.DataFrame,
    fdict: pd.DataFrame,
    val_df: pd.DataFrame,
    cov_df: pd.DataFrame,
    miss_df: pd.DataFrame,
) -> str:
    lines = [
        "# JEEVAN-DRISHTI Phase 3: Feature Engineering Summary Report (2023–2025 Scope)\n",
        f"- **Engine Version**: `{FEATURE_ENGINE_VERSION}` | **Dataset Version**: `{FEATURE_DATASET_VERSION}`",
        "- **Temporal Scope**: **Strictly 2023–2025** (`2023-W01` to `2025-W52`, 156 consecutive 7-day reporting weeks).",
        "- **Entrypoint**: `python -m src.features.feature_pipeline`",
        "- **Models Trained**: **None** (strictly feature engineering and validation per Phase 3 instructions).\n",
        "## 1. Datasets Used & Generated\n",
        "### Input Processed Datasets Used (2023–2025)",
        f"- `data/processed/cleaned_disease_data.csv` (`source_dataset == 'idsp_weekly'`, 570 cleaned outbreak records across 2023–2025, aggregated to **{len(weekly_joined)}** distinct `(district_std, disease_std, report_year, report_week)` reporting-period observations).",
        "- `data/processed/cleaned_climate_data.csv` (39,456 daily district-climate observations from `2023-01-01` to `2025-12-31` across 36 districts, plus `data/raw/imdweb_5dgbAskl.nc` for Gadchiroli's adjacent interior land cell `19.75°N, 80.00°E`).\n",
        "### Output Feature Datasets (`data/features/`)",
        f"1. `data/features/disease_climate_weekly_features.csv` (also saved as `disease_climate_weekly_features_2023_2025.csv`): **{len(weekly_joined):,} rows × {weekly_joined.shape[1]} columns** — Every observed 2023–2025 `(district_std, disease_std, report_year, report_week)` outbreak period aligned with disease history, climate, temporal, and geographic features.",
        f"2. `data/features/district_week_climate_features.csv` (also saved as `district_week_climate_features_2023_2025.csv`): **{len(climate_wk):,} rows × {climate_wk.shape[1]} columns** — Complete 36-district × 156-week (2023–2025) weekly climate, temporal, and geographic feature grid.\n",
        "## 2. Summary of Engineered Features by Category\n",
        "| Category | Count | Key Engineered Features |",
        "|---|---|---|",
    ]
    for cat, grp in fdict.groupby("category", sort=False):
        sample_cols = ", ".join(f"`{c}`" for c in grp["feature_name"].head(6).tolist())
        if len(grp) > 6:
            sample_cols += f", … (+{len(grp) - 6} more)"
        lines.append(f"| **{cat}** | {len(grp)} | {sample_cols} |")

    lines.extend([
        "\n## 3. Target Disease Coverage in 2023–2025 (`disease_climate_weekly_features.csv`)\n",
        "| Target Disease | 2023–2025 Weekly Outbreak Periods | Districts with Outbreaks | With Event Lag-1 (`prev_outbreak_cases_lag_1`) | With Calendar Lag-1w (`cases_lag_1p`) | With Rainfall Features | With Temp Features | With Pune RH Features |",
        "|---|---|---|---|---|---|---|---|",
    ])
    dis_cov = cov_df[cov_df["scope_type"].str.startswith("Target Disease")]
    for _, r in dis_cov.iterrows():
        lines.append(
            f"| **{r.scope_name}** | {r.weekly_2023_2025_periods} | {r.districts_with_outbreaks} | {r.periods_with_prev_outbreak_lag1} | {r.periods_with_calendar_lag1p} | {r.periods_with_rain_features} | {r.periods_with_temp_features} | {r.periods_with_humidity_features} |"
        )

    lines.extend([
        "\n## 4. Data Leakage Prevention & Validation Results\n",
        f"All **{len(val_df)} / {len(val_df)}** automated checks in `reports/feature_engineering/leakage_validation_report.csv` passed:\n",
        "- **Strict Prediction Cutoff (`<= t - 1`)**: All disease lag features (`cases_lag_1p..53p`, `prev_outbreak_cases_lag_1..8`), rolling averages/stds (`cases_roll_mean_2p..8p`, `prev_outbreak_roll_mean_2..8`), absolute/percentage changes (`cases_abs_change_lag1_lag2`, `prev_outbreak_abs_change_1_2`), expanding seasonal baselines (`seasonal_dd_season_mean_cases_prev`, `seasonal_state_disease_season_mean_prev`), and expanding district burdens (`district_disease_prior_cases_cum`) apply `.shift(1)` prior to aggregation so period `t` outcomes are **never** included.",
        "- **Synthetic Future Perturbation Proof**: Adding `+50,000` cases to all outbreak records at or after `2024-W27` (`period_idx >= 78`) produced **0.000000** difference across all 45 disease-history and district-burden input features on all rows `period_idx <= 78`.",
        "- **Climate Perturbation Proof**: Adding `+500 mm` rainfall and `+15 °C` temperature to daily climate on or after `2024-07-01` produced **0.000000** difference across all weekly climate features for weeks ending `< 2024-07-01`.\n",
        "## 5. Unsupported Features & Scientific Limitations\n",
        "1. **Population & Population-Normalized Incidence (`district_population`, `population_density`, `cases_per_100k_pop`) — UNSUPPORTED**:",
        "   - No district population or census dataset exists in the project workspace. Per Prompt 3 (*'Use only the existing project datasets. Do not download new datasets or invent missing records'* and *'Do not calculate population-normalized rates without a valid population denominator'*), these three columns are preserved in the schema as `NaN`.",
        "2. **Heat-Related Illness — UNSUPPORTED (0 Records)**:",
        "   - Zero records exist for `Heat-related illness` in the 2023–2025 IDSP dataset. No synthetic records were created.",
        "3. **Mumbai Suburban Climate — UNSUPPORTED (No Coordinates in Workspace)**:",
        "   - `Mumbai Suburban` has 3 outbreak rows in 2023–2025 (`Zika Virus`, `Food Poisoning`, `Acute Diarrhoeal Disease`), but no latitude/longitude coordinates exist for `Mumbai Suburban` in any project file. Its climate features remain `NaN` (`rain_spatial_source = 'none_no_coordinates'`).",
        "4. **Relative Humidity Statewide Coverage — LIMITED TO PUNE (2024-W19 to 2025-W24)**:",
        "   - `humid_tel_hr_cwprs_mh_2021_2025.csv` contains telemetry for a single station in `Pune` from `2024-05-11` to `2025-06-09` (57 reporting weeks). For all other 35 districts and outside that date window, `rh_mean_t` and its lags remain `NaN`.",
        "5. **Gadchiroli Rainfall Grid Boundary Note**:",
        "   - Gadchiroli's reference coordinate (`19.759°N, 80.162°E`) is closest to `(19.75°N, 80.25°E)`, which lies on the masked eastern border of `imdweb_5dgbAskl.nc`. We extracted Gadchiroli's immediate unmasked interior land cell `(19.75°N, 80.00°E, 0.162° away)` from `data/raw/imdweb_5dgbAskl.nc`, tagged `rain_spatial_source = 'adjacent_land_cell_19.75N_80.00E'`, and also retained `rain_mm_strict_nearest_sum_t` (`NaN`) for full transparency.",
        "6. **Episodic Outbreak Reporting vs. Routine Surveillance**:",
        "   - IDSP reports outbreak clusters rather than routine weekly zero-counts. Calendar-week lags (`cases_lag_1p..8p`) are `NaN` whenever the preceding calendar week had no reported outbreak (never zero-filled), while binary indicators (`outbreak_reported_lag_1p..8p`, `outbreak_weeks_count_prev_2p..8p`) and event-sequence lags (`prev_outbreak_cases_lag_1..8`) provide dense history without fabricating zeros.\n",
    ])
    return "\n".join(lines)


def _update_registry(feature_tables: list, report_paths: list[Path]):
    reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    rel = lambda p: str(p.relative_to(PROJ)).replace("\\", "/")
    reg["feature_datasets"] = [
        dict(
            name=name,
            path=str(p),
            relative_path=rel(p),
            version=FEATURE_DATASET_VERSION,
            engine_version=FEATURE_ENGINE_VERSION,
            temporal_scope="2023-2025",
            rows=len(df),
            columns=int(df.shape[1]),
            sha256=_sha256(p),
            description=desc,
        )
        for name, p, df, desc in feature_tables
    ]
    reg["feature_engineering_reports"] = [
        dict(name=p.name, path=str(p), relative_path=rel(p), version=FEATURE_DATASET_VERSION)
        for p in report_paths
    ]
    reg["feature_engine"] = dict(
        package="src.features",
        entrypoint="python -m src.features.feature_pipeline",
        tests="python -m unittest discover -s tests -v",
        version=FEATURE_ENGINE_VERSION,
        dataset_version=FEATURE_DATASET_VERSION,
        temporal_scope="2023-2025 (156 reporting weeks: 2023-W01 .. 2025-W52)",
        last_run=datetime.now().isoformat(timespec="seconds"),
    )
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2), encoding="utf-8")


if __name__ == "__main__":
    run_feature_pipeline(verbose=True)
