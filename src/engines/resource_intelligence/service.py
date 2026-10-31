"""FastAPI-Ready Service Layer for Phase 11 Healthcare Resource Demand Intelligence Engine."""
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from jeevan_pipeline.standardize import standardize_district
from src.engines.resource_intelligence.data_availability import MANDATORY_RESOURCE_DISCLAIMER
from src.engines.resource_intelligence.pipeline import _clean_nan_for_json
from src.engines.resource_intelligence.scenario_calculator import (
    ENGINE_VERSION,
    ResourceScenarioAssumptions,
    compute_single_resource_scenario_row,
)

PROJ = Path(__file__).resolve().parents[3]


class HealthcareResourceDemandService:
    """Service layer exposing data availability, district resource status, frontend payload,
    and custom user-editable hypothetical scenario calculation.
    """

    def __init__(self, proj_dir: Path = PROJ) -> None:
        self.proj_dir = proj_dir
        self.data_dir = proj_dir / "data" / "resource_intelligence"
        self.completeness_df = pd.read_csv(self.data_dir / "resource_data_completeness_audit.csv")
        self.district_summary_df = pd.read_csv(self.data_dir / "district_resource_summary.csv")
        self.catalog_df = pd.read_csv(self.data_dir / "hypothetical_resource_scenarios_catalog.csv")
        self.eval_df = pd.read_csv(self.data_dir / "hypothetical_district_scenario_evaluations.csv")
        self.frontend_payload: Dict[str, Any] = json.loads(
            (self.data_dir / "frontend_integration_payload.json").read_text(encoding="utf-8")
        )

    def get_frontend_integration_payload(self) -> Dict[str, Any]:
        """Return the complete 6-section frontend integration payload."""
        return self.frontend_payload

    def get_data_completeness_audit(self) -> List[Dict[str, Any]]:
        """Return the 12-row data completeness audit table."""
        return _clean_nan_for_json(self.completeness_df.to_dict(orient="records"))

    def get_district_resource_status(self, district_name: str) -> Dict[str, Any]:
        """Return empirical resource status (explicit None/NaN) + upstream epidemiological anchors for a district."""
        std_d, _ = standardize_district(district_name)
        sub = self.district_summary_df[self.district_summary_df["district_name"] == std_d]
        if sub.empty:
            raise ValueError(f"Unknown Maharashtra district '{district_name}' (standardized: '{std_d}').")
        rec = _clean_nan_for_json(sub.iloc[0].to_dict())
        rec["mandatory_disclaimer"] = MANDATORY_RESOURCE_DISCLAIMER
        return rec

    def run_custom_hypothetical_scenario(
        self,
        district_name: str,
        case_load_anchor_mode: str = "phase6_forecast_7d_cases",
        user_entered_baseline_cases: Optional[float] = None,
        hypothetical_case_increase_pct: float = 25.0,
        assumed_admission_proportion: float = 0.15,
        assumed_icu_proportion_of_admissions: float = 0.10,
        assumed_avg_length_of_stay_days: float = 4.5,
        assumed_tests_per_case: float = 3.0,
        assumed_iv_fluid_units_per_admission: float = 4.0,
        assumed_treatment_courses_per_case: float = 1.0,
        assumed_available_general_beds: Optional[float] = 40.0,
        assumed_available_icu_beds: Optional[float] = 6.0,
        scenario_id: str = "CUSTOM_USER_WHAT_IF_SCENARIO",
        scenario_name: str = "User-Defined Hypothetical Resource Scenario",
    ) -> Dict[str, Any]:
        """Run a custom, user-editable hypothetical scenario for any Maharashtra district."""
        std_d, _ = standardize_district(district_name)
        sub = self.district_summary_df[self.district_summary_df["district_name"] == std_d]
        if sub.empty:
            raise ValueError(f"Unknown Maharashtra district '{district_name}' (standardized: '{std_d}').")
        d_row = sub.iloc[0]

        assumptions = ResourceScenarioAssumptions(
            scenario_id=scenario_id,
            scenario_name=scenario_name,
            scenario_description="Custom user-entered hypothetical scenario via HealthcareResourceDemandService.",
            case_load_anchor_mode=case_load_anchor_mode,
            user_entered_baseline_cases=user_entered_baseline_cases,
            hypothetical_case_increase_pct=hypothetical_case_increase_pct,
            assumed_admission_proportion=assumed_admission_proportion,
            assumed_icu_proportion_of_admissions=assumed_icu_proportion_of_admissions,
            assumed_avg_length_of_stay_days=assumed_avg_length_of_stay_days,
            assumed_tests_per_case=assumed_tests_per_case,
            assumed_iv_fluid_units_per_admission=assumed_iv_fluid_units_per_admission,
            assumed_treatment_courses_per_case=assumed_treatment_courses_per_case,
            assumed_available_general_beds=assumed_available_general_beds,
            assumed_available_icu_beds=assumed_available_icu_beds,
        ).validate()

        if assumptions.case_load_anchor_mode == "user_entered_cases":
            base_c = assumptions.user_entered_baseline_cases
        elif assumptions.case_load_anchor_mode == "observed_peak_weekly_cases":
            base_c = d_row["observed_2023_2025_peak_weekly_cases"]
        elif assumptions.case_load_anchor_mode == "observed_mean_weekly_cases":
            base_c = d_row["observed_2023_2025_mean_cases_per_outbreak_week"]
        else:
            base_c = d_row["upstream_phase6_forecast_7d_mean_cases"]

        row_out = compute_single_resource_scenario_row(
            district_name=str(d_row["district_name"]),
            district_id=str(d_row["district_id"]),
            division=str(d_row["administrative_division"]),
            baseline_cases=base_c,
            assumptions=assumptions,
            calc_timestamp=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        )
        res = _clean_nan_for_json(row_out)
        res["mandatory_disclaimer"] = MANDATORY_RESOURCE_DISCLAIMER
        res["model_version"] = ENGINE_VERSION
        return res
