"""FastAPI-Ready Service Layer for Phase 12 Disease Severity & Mortality Intelligence Engine."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from jeevan_pipeline.standardize import standardize_disease, standardize_district
from src.engines.severity_intelligence.data_preparation import (
    ENGINE_VERSION,
    MANDATORY_SEVERITY_PRIVACY_AND_SAFETY_DISCLAIMER,
)
from src.engines.severity_intelligence.pipeline import _clean_nan_for_json

PROJ = Path(__file__).resolve().parents[3]


class DiseaseSeverityMortalityService:
    """Service layer exposing aggregate disease mortality, district mortality, temporal trends,
    and chronological forecast evaluations while strictly blocking individual prognosis queries.
    """

    def __init__(self, proj_dir: Path = PROJ) -> None:
        self.proj_dir = proj_dir
        self.data_dir = proj_dir / "data" / "severity_intelligence"
        self.availability_df = pd.read_csv(self.data_dir / "outcome_data_availability_audit.csv")
        self.weekly_records_df = pd.read_csv(self.data_dir / "verified_weekly_mortality_records.csv")
        self.disease_summary_df = pd.read_csv(self.data_dir / "disease_mortality_summary.csv")
        self.district_summary_df = pd.read_csv(self.data_dir / "district_mortality_summary.csv")
        self.seasonal_df = pd.read_csv(self.data_dir / "seasonal_mortality_trends.csv")
        self.sufficiency_df = pd.read_csv(self.data_dir / "mortality_forecasting_sufficiency_audit.csv")
        self.forecast_metrics_df = pd.read_csv(self.data_dir / "mortality_forecast_model_comparison.csv")
        self.summary_payload: Dict[str, Any] = json.loads(
            (self.data_dir / "severity_mortality_summary_payload.json").read_text(encoding="utf-8")
        )

    def get_summary_payload(self) -> Dict[str, Any]:
        """Return the complete Phase 12 summary payload."""
        return self.summary_payload

    def get_disease_mortality_profile(self, disease_name: str) -> Dict[str, Any]:
        """Return verified aggregate mortality & Outbreak-Cluster CFR profile for a disease."""
        std_dis = standardize_disease(disease_name)[0]
        target = std_dis if std_dis else disease_name
        sub = self.disease_summary_df[
            self.disease_summary_df["disease"].str.lower() == str(target).strip().lower()
        ]
        if sub.empty:
            return {
                "disease_queried": disease_name,
                "status": "UNSUPPORTED_OR_ZERO_OUTBREAKS_IN_2023_2025",
                "mandatory_disclaimer": MANDATORY_SEVERITY_PRIVACY_AND_SAFETY_DISCLAIMER,
                "engine_version": ENGINE_VERSION,
            }
        rec = _clean_nan_for_json(sub.iloc[0].to_dict())
        rec["mandatory_disclaimer"] = MANDATORY_SEVERITY_PRIVACY_AND_SAFETY_DISCLAIMER
        return rec

    def get_district_mortality_profile(self, district_name: str) -> Dict[str, Any]:
        """Return verified aggregate mortality & Outbreak-Cluster CFR profile for a Maharashtra district."""
        std_d, _ = standardize_district(district_name)
        sub = self.district_summary_df[self.district_summary_df["district"] == std_d]
        if sub.empty:
            raise ValueError(f"Unknown Maharashtra district '{district_name}' (standardized: '{std_d}').")
        rec = _clean_nan_for_json(sub.iloc[0].to_dict())
        rec["mandatory_disclaimer"] = MANDATORY_SEVERITY_PRIVACY_AND_SAFETY_DISCLAIMER
        return rec

    def get_weekly_records(
        self,
        district_name: Optional[str] = None,
        disease_name: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Query verified deduplicated weekly aggregate mortality records (2023-2025)."""
        df = self.weekly_records_df.copy()
        if district_name:
            std_d, _ = standardize_district(district_name)
            df = df[df["district"] == std_d]
        if disease_name:
            std_dis = standardize_disease(disease_name)[0]
            target = std_dis if std_dis else disease_name
            df = df[df["disease"].str.lower() == str(target).strip().lower()]
        return _clean_nan_for_json(df.to_dict(orient="records"))
