"""Hypothetical Healthcare Resource Demand Scenario Calculator (Phase 11 Section 5).

Because no empirical hospital admissions, bed occupancy, ICU capacity, testing volume,
or medicine stock datasets exist in the project workspace, this module implements a
transparent, user-editable What-If Scenario Calculator.

Strict Rules:
- Every assumption (`hypothetical_case_increase_pct`, `assumed_admission_proportion`,
  `assumed_icu_proportion_of_admissions`, `assumed_avg_length_of_stay_days`,
  `assumed_tests_per_case`, `assumed_available_general_beds`, etc.) is explicitly visible
  and editable in the output schema.
- Every output record is labelled `HYPOTHETICAL_SCENARIO_NOT_ACTUAL_CAPACITY_OR_FORECAST`.
- Never claims that assumed capacities represent real Maharashtra hospital inventories.
"""
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from jeevan_pipeline.standardize import standardize_district
from src.engines.resource_intelligence.data_availability import MANDATORY_RESOURCE_DISCLAIMER
from src.features.geographic_features import OFFICIAL_DISTRICTS


ENGINE_VERSION = "phase11-v1.0.0"

VALID_ANCHOR_MODES = {
    "user_entered_cases",
    "observed_peak_weekly_cases",
    "observed_mean_weekly_cases",
    "phase6_forecast_7d_cases",
}


@dataclass
class ResourceScenarioAssumptions:
    """Explicit, user-editable assumptions for hypothetical resource demand calculation."""
    scenario_id: str
    scenario_name: str
    scenario_description: str
    case_load_anchor_mode: str = "phase6_forecast_7d_cases"
    user_entered_baseline_cases: Optional[float] = None
    hypothetical_case_increase_pct: float = 25.0         # [-90.0, 1000.0] %
    assumed_admission_proportion: float = 0.15           # [0.0, 1.0]
    assumed_icu_proportion_of_admissions: float = 0.10   # [0.0, 1.0]
    assumed_avg_length_of_stay_days: float = 4.5         # [1.0, 30.0] days
    assumed_tests_per_case: float = 3.0                  # [0.0, 25.0] kits/case (incl. suspect screening)
    assumed_iv_fluid_units_per_admission: float = 4.0    # [0.0, 50.0] units/admission
    assumed_treatment_courses_per_case: float = 1.0      # [0.0, 10.0] courses/case
    assumed_available_general_beds: Optional[float] = 50.0
    assumed_available_icu_beds: Optional[float] = 8.0
    assumed_available_test_kits: Optional[float] = 400.0
    assumed_available_medicine_units: Optional[float] = 300.0

    def validate(self) -> "ResourceScenarioAssumptions":
        """Validate all user-entered scenario assumptions."""
        if not self.scenario_id or not str(self.scenario_id).strip():
            raise ValueError("scenario_id must be a non-empty string.")
        if self.case_load_anchor_mode not in VALID_ANCHOR_MODES:
            raise ValueError(
                f"Invalid case_load_anchor_mode '{self.case_load_anchor_mode}'. "
                f"Must be one of {sorted(VALID_ANCHOR_MODES)}."
            )
        if self.case_load_anchor_mode == "user_entered_cases":
            if self.user_entered_baseline_cases is None or float(self.user_entered_baseline_cases) < 0:
                raise ValueError(
                    "user_entered_baseline_cases must be non-negative when case_load_anchor_mode='user_entered_cases'."
                )
        if not (-90.0 <= float(self.hypothetical_case_increase_pct) <= 1000.0):
            raise ValueError("hypothetical_case_increase_pct must be between -90.0% and +1000.0%.")
        if not (0.0 <= float(self.assumed_admission_proportion) <= 1.0):
            raise ValueError("assumed_admission_proportion must be in [0.0, 1.0].")
        if not (0.0 <= float(self.assumed_icu_proportion_of_admissions) <= 1.0):
            raise ValueError("assumed_icu_proportion_of_admissions must be in [0.0, 1.0].")
        if not (1.0 <= float(self.assumed_avg_length_of_stay_days) <= 30.0):
            raise ValueError("assumed_avg_length_of_stay_days must be in [1.0, 30.0] days.")
        if not (0.0 <= float(self.assumed_tests_per_case) <= 25.0):
            raise ValueError("assumed_tests_per_case must be in [0.0, 25.0].")
        if not (0.0 <= float(self.assumed_iv_fluid_units_per_admission) <= 50.0):
            raise ValueError("assumed_iv_fluid_units_per_admission must be in [0.0, 50.0].")
        if not (0.0 <= float(self.assumed_treatment_courses_per_case) <= 10.0):
            raise ValueError("assumed_treatment_courses_per_case must be in [0.0, 10.0].")

        for cap_name in [
            "assumed_available_general_beds",
            "assumed_available_icu_beds",
            "assumed_available_test_kits",
            "assumed_available_medicine_units",
        ]:
            val = getattr(self, cap_name)
            if val is not None and float(val) <= 0:
                raise ValueError(f"{cap_name} must be strictly positive when provided (got {val}).")

        return self


def compute_single_resource_scenario_row(
    district_name: str,
    district_id: str,
    division: str,
    baseline_cases: Optional[float],
    assumptions: ResourceScenarioAssumptions,
    calc_timestamp: str,
) -> Dict[str, Any]:
    """Compute hypothetical resource demand and capacity strain for a single district and scenario."""
    assumptions.validate()

    if baseline_cases is None or pd.isna(baseline_cases):
        return {
            "scenario_id": assumptions.scenario_id,
            "scenario_name": assumptions.scenario_name,
            "district_id": district_id,
            "district_name": district_name,
            "administrative_division": division,
            "case_load_anchor_mode": assumptions.case_load_anchor_mode,
            "anchor_baseline_cases": np.nan,
            "hypothetical_case_increase_pct": assumptions.hypothetical_case_increase_pct,
            "hypothetical_projected_weekly_cases": np.nan,
            "assumed_admission_proportion": assumptions.assumed_admission_proportion,
            "hypothetical_weekly_admissions": np.nan,
            "assumed_icu_proportion_of_admissions": assumptions.assumed_icu_proportion_of_admissions,
            "assumed_avg_length_of_stay_days": assumptions.assumed_avg_length_of_stay_days,
            "hypothetical_total_bed_days_required": np.nan,
            "hypothetical_concurrent_general_beds_required": np.nan,
            "hypothetical_concurrent_icu_beds_required": np.nan,
            "hypothetical_diagnostic_tests_required": np.nan,
            "hypothetical_iv_fluid_units_required": np.nan,
            "hypothetical_treatment_courses_required": np.nan,
            "user_assumed_available_general_beds": assumptions.assumed_available_general_beds,
            "user_assumed_available_icu_beds": assumptions.assumed_available_icu_beds,
            "hypothetical_general_bed_utilization_ratio": np.nan,
            "hypothetical_icu_bed_utilization_ratio": np.nan,
            "hypothetical_general_bed_gap": np.nan,
            "hypothetical_icu_bed_gap": np.nan,
            "hypothetical_shortage_indicator": "Insufficient Data (No Baseline Surveillance Cases for Anchor)",
            "output_nature": "HYPOTHETICAL_SCENARIO_NOT_ACTUAL_CAPACITY_OR_FORECAST",
            "model_version": ENGINE_VERSION,
            "calculation_timestamp": calc_timestamp,
        }

    base_c = float(baseline_cases)
    proj_cases = base_c * (1.0 + float(assumptions.hypothetical_case_increase_pct) / 100.0)
    proj_adm = proj_cases * float(assumptions.assumed_admission_proportion)
    los = float(assumptions.assumed_avg_length_of_stay_days)
    icu_prop = float(assumptions.assumed_icu_proportion_of_admissions)

    tot_bed_days = proj_adm * los
    gen_beds_req = (proj_adm * (1.0 - icu_prop) * los) / 7.0
    icu_beds_req = (proj_adm * icu_prop * los) / 7.0

    tests_req = proj_cases * float(assumptions.assumed_tests_per_case)
    iv_req = proj_adm * float(assumptions.assumed_iv_fluid_units_per_admission)
    med_req = proj_cases * float(assumptions.assumed_treatment_courses_per_case)

    gen_cap = assumptions.assumed_available_general_beds
    icu_cap = assumptions.assumed_available_icu_beds

    gen_util = (gen_beds_req / float(gen_cap)) if gen_cap is not None else np.nan
    icu_util = (icu_beds_req / float(icu_cap)) if icu_cap is not None else np.nan
    gen_gap = (float(gen_cap) - gen_beds_req) if gen_cap is not None else np.nan
    icu_gap = (float(icu_cap) - icu_beds_req) if icu_cap is not None else np.nan

    if gen_cap is None and icu_cap is None:
        shortage_ind = "HYPOTHETICAL_DEMAND_ONLY_NO_ASSUMED_CAPACITY"
    elif (pd.notna(gen_util) and gen_util > 1.0) or (pd.notna(icu_util) and icu_util > 1.0):
        shortage_ind = "HYPOTHETICAL_CAPACITY_DEFICIT_EXCEEDS_100PCT_ASSUMED_BEDS"
    elif (pd.notna(gen_util) and gen_util >= 0.80) or (pd.notna(icu_util) and icu_util >= 0.80):
        shortage_ind = "HYPOTHETICAL_HIGH_STRAIN_80_TO_100PCT_ASSUMED_BEDS"
    elif (pd.notna(gen_util) and gen_util >= 0.50) or (pd.notna(icu_util) and icu_util >= 0.50):
        shortage_ind = "HYPOTHETICAL_MODERATE_LOAD_50_TO_80PCT_ASSUMED_BEDS"
    else:
        shortage_ind = "HYPOTHETICAL_WITHIN_ASSUMED_CAPACITY_LT50PCT"

    return {
        "scenario_id": assumptions.scenario_id,
        "scenario_name": assumptions.scenario_name,
        "district_id": district_id,
        "district_name": district_name,
        "administrative_division": division,
        "case_load_anchor_mode": assumptions.case_load_anchor_mode,
        "anchor_baseline_cases": round(base_c, 2),
        "hypothetical_case_increase_pct": round(float(assumptions.hypothetical_case_increase_pct), 1),
        "hypothetical_projected_weekly_cases": round(proj_cases, 2),
        "assumed_admission_proportion": round(float(assumptions.assumed_admission_proportion), 4),
        "hypothetical_weekly_admissions": round(proj_adm, 2),
        "assumed_icu_proportion_of_admissions": round(icu_prop, 4),
        "assumed_avg_length_of_stay_days": round(los, 2),
        "hypothetical_total_bed_days_required": round(tot_bed_days, 2),
        "hypothetical_concurrent_general_beds_required": round(gen_beds_req, 2),
        "hypothetical_concurrent_icu_beds_required": round(icu_beds_req, 2),
        "hypothetical_diagnostic_tests_required": round(tests_req, 1),
        "hypothetical_iv_fluid_units_required": round(iv_req, 1),
        "hypothetical_treatment_courses_required": round(med_req, 1),
        "user_assumed_available_general_beds": float(gen_cap) if gen_cap is not None else np.nan,
        "user_assumed_available_icu_beds": float(icu_cap) if icu_cap is not None else np.nan,
        "hypothetical_general_bed_utilization_ratio": round(float(gen_util), 4) if pd.notna(gen_util) else np.nan,
        "hypothetical_icu_bed_utilization_ratio": round(float(icu_util), 4) if pd.notna(icu_util) else np.nan,
        "hypothetical_general_bed_gap": round(float(gen_gap), 2) if pd.notna(gen_gap) else np.nan,
        "hypothetical_icu_bed_gap": round(float(icu_gap), 2) if pd.notna(icu_gap) else np.nan,
        "hypothetical_shortage_indicator": shortage_ind,
        "output_nature": "HYPOTHETICAL_SCENARIO_NOT_ACTUAL_CAPACITY_OR_FORECAST",
        "model_version": ENGINE_VERSION,
        "calculation_timestamp": calc_timestamp,
    }


def build_default_resource_scenarios() -> List[ResourceScenarioAssumptions]:
    """Define 4 transparent benchmark resource planning scenarios covering different disease profiles and surges."""
    return [
        ResourceScenarioAssumptions(
            scenario_id="RES_SCEN_01_FORECAST_BASELINE_MODERATE",
            scenario_name="Phase 6 7-Day Case Forecast Anchor + 0% Surge (Moderate Vector-Borne Profile)",
            scenario_description=(
                "Anchored to each district's Phase 6 7-day forecast mean cases (+0% surge), assuming "
                "15% inpatient admission rate, 10% ICU rate among admissions, 4.5-day stay, and "
                "hypothetical reference ward capacity of 40 general beds / 6 ICU beds."
            ),
            case_load_anchor_mode="phase6_forecast_7d_cases",
            hypothetical_case_increase_pct=0.0,
            assumed_admission_proportion=0.15,
            assumed_icu_proportion_of_admissions=0.10,
            assumed_avg_length_of_stay_days=4.5,
            assumed_tests_per_case=3.0,
            assumed_iv_fluid_units_per_admission=4.0,
            assumed_treatment_courses_per_case=1.0,
            assumed_available_general_beds=40.0,
            assumed_available_icu_beds=6.0,
        ),
        ResourceScenarioAssumptions(
            scenario_id="RES_SCEN_02_MONSOON_SURGE_PLUS_50PCT",
            scenario_name="Phase 6 7-Day Forecast Anchor + 50% Monsoon Surge (Elevated Admission Profile)",
            scenario_description=(
                "Hypothetical +50% case surge over Phase 6 7-day forecast mean cases, assuming "
                "22% inpatient admission rate, 12% ICU rate, 5.0-day stay, against hypothetical "
                "reference ward capacity of 40 general beds / 6 ICU beds."
            ),
            case_load_anchor_mode="phase6_forecast_7d_cases",
            hypothetical_case_increase_pct=50.0,
            assumed_admission_proportion=0.22,
            assumed_icu_proportion_of_admissions=0.12,
            assumed_avg_length_of_stay_days=5.0,
            assumed_tests_per_case=4.0,
            assumed_iv_fluid_units_per_admission=5.0,
            assumed_treatment_courses_per_case=1.2,
            assumed_available_general_beds=40.0,
            assumed_available_icu_beds=6.0,
        ),
        ResourceScenarioAssumptions(
            scenario_id="RES_SCEN_03_HISTORICAL_PEAK_STRESS_TEST",
            scenario_name="Observed 2023-2025 Peak Weekly Outbreak Load + 25% Surge (Acute Outbreak Stress Test)",
            scenario_description=(
                "Anchored to each district's highest observed weekly outbreak case count in 2023-2025 "
                "plus a hypothetical +25% surge, assuming 20% admission rate, 10% ICU rate, 4.0-day stay, "
                "and hypothetical district isolation ward capacity of 50 general beds / 8 ICU beds."
            ),
            case_load_anchor_mode="observed_peak_weekly_cases",
            hypothetical_case_increase_pct=25.0,
            assumed_admission_proportion=0.20,
            assumed_icu_proportion_of_admissions=0.10,
            assumed_avg_length_of_stay_days=4.0,
            assumed_tests_per_case=3.5,
            assumed_iv_fluid_units_per_admission=6.0,
            assumed_treatment_courses_per_case=1.0,
            assumed_available_general_beds=50.0,
            assumed_available_icu_beds=8.0,
        ),
        ResourceScenarioAssumptions(
            scenario_id="RES_SCEN_04_ACUTE_DIARRHOEAL_REHYDRATION_FOCUS",
            scenario_name="Observed Mean Weekly Outbreak Load + 40% Surge (Water-Borne Rehydration Focus)",
            scenario_description=(
                "Anchored to each district's observed mean weekly outbreak cases (+40% surge), "
                "assuming 18% inpatient observation/admission rate, 5% ICU rate, shorter 2.5-day stay, "
                "and high IV/ORS fluid requirement (8 units/admission) against 40 general beds / 6 ICU beds."
            ),
            case_load_anchor_mode="observed_mean_weekly_cases",
            hypothetical_case_increase_pct=40.0,
            assumed_admission_proportion=0.18,
            assumed_icu_proportion_of_admissions=0.05,
            assumed_avg_length_of_stay_days=2.5,
            assumed_tests_per_case=2.0,
            assumed_iv_fluid_units_per_admission=8.0,
            assumed_treatment_courses_per_case=1.5,
            assumed_available_general_beds=40.0,
            assumed_available_icu_beds=6.0,
        ),
    ]


def evaluate_scenarios_across_districts(
    district_summary_df: pd.DataFrame,
    scenarios: List[ResourceScenarioAssumptions],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run all benchmark hypothetical resource scenarios across all 36 Maharashtra districts."""
    calc_ts = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    eval_rows: List[Dict[str, Any]] = []
    catalog_rows: List[Dict[str, Any]] = []

    for sc in scenarios:
        sc.validate()
        sc_rows = []
        for _, d_row in district_summary_df.iterrows():
            if sc.case_load_anchor_mode == "user_entered_cases":
                base_c = sc.user_entered_baseline_cases
            elif sc.case_load_anchor_mode == "observed_peak_weekly_cases":
                base_c = d_row["observed_2023_2025_peak_weekly_cases"]
            elif sc.case_load_anchor_mode == "observed_mean_weekly_cases":
                base_c = d_row["observed_2023_2025_mean_cases_per_outbreak_week"]
            else:
                base_c = d_row["upstream_phase6_forecast_7d_mean_cases"]

            row_out = compute_single_resource_scenario_row(
                district_name=str(d_row["district_name"]),
                district_id=str(d_row["district_id"]),
                division=str(d_row["administrative_division"]),
                baseline_cases=base_c,
                assumptions=sc,
                calc_timestamp=calc_ts,
            )
            eval_rows.append(row_out)
            sc_rows.append(row_out)

        sc_df = pd.DataFrame(sc_rows)
        valid_sc = sc_df[sc_df["hypothetical_projected_weekly_cases"].notna()]
        n_deficit = int(
            (
                valid_sc["hypothetical_shortage_indicator"]
                == "HYPOTHETICAL_CAPACITY_DEFICIT_EXCEEDS_100PCT_ASSUMED_BEDS"
            ).sum()
        )
        n_high_strain = int(
            (
                valid_sc["hypothetical_shortage_indicator"]
                == "HYPOTHETICAL_HIGH_STRAIN_80_TO_100PCT_ASSUMED_BEDS"
            ).sum()
        )

        catalog_rows.append(
            {
                **asdict(sc),
                "evaluated_districts_with_anchor": len(valid_sc),
                "insufficient_data_districts": len(sc_df) - len(valid_sc),
                "statewide_hypothetical_weekly_cases": round(
                    float(valid_sc["hypothetical_projected_weekly_cases"].sum()), 1
                ),
                "statewide_hypothetical_weekly_admissions": round(
                    float(valid_sc["hypothetical_weekly_admissions"].sum()), 1
                ),
                "statewide_hypothetical_general_beds_required": round(
                    float(valid_sc["hypothetical_concurrent_general_beds_required"].sum()), 1
                ),
                "statewide_hypothetical_icu_beds_required": round(
                    float(valid_sc["hypothetical_concurrent_icu_beds_required"].sum()), 1
                ),
                "statewide_hypothetical_diagnostic_tests_required": round(
                    float(valid_sc["hypothetical_diagnostic_tests_required"].sum()), 1
                ),
                "statewide_hypothetical_iv_fluid_units_required": round(
                    float(valid_sc["hypothetical_iv_fluid_units_per_admission"].sum() if "hypothetical_iv_fluid_units_per_admission" in valid_sc else valid_sc["hypothetical_iv_fluid_units_required"].sum()), 1
                ),
                "districts_exceeding_100pct_assumed_capacity": n_deficit,
                "districts_80_to_100pct_assumed_capacity": n_high_strain,
                "output_nature": "HYPOTHETICAL_SCENARIO_NOT_ACTUAL_CAPACITY_OR_FORECAST",
                "mandatory_disclaimer": MANDATORY_RESOURCE_DISCLAIMER,
                "model_version": ENGINE_VERSION,
                "calculation_timestamp": calc_ts,
            }
        )

    return pd.DataFrame(catalog_rows), pd.DataFrame(eval_rows)
