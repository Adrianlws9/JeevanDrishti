"""FastAPI-Ready Service Layer for Phase 13 Centralized Explainable AI (XAI) Engine."""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from jeevan_pipeline.standardize import standardize_disease, standardize_district
from src.engines.explainability.model_inspector import (
    ENGINE_VERSION,
    MANDATORY_XAI_NON_CAUSAL_DISCLAIMER,
    PROJ,
)

class ExplainableAIService:
    """Service layer exposing global model explanations, local risk/forecast/anomaly explanations,
    and the 'Why was this district flagged?' frontend panel.
    """

    def __init__(self, proj_dir: Path = PROJ) -> None:
        self.proj_dir = proj_dir
        self.data_dir = proj_dir / "data" / "explanations"
        self.inventory_df = pd.read_csv(self.data_dir / "model_explainability_inventory.csv")
        self.global_df = pd.read_csv(self.data_dir / "global_feature_explanations.csv")
        self.local_risk_df = pd.read_csv(self.data_dir / "local_risk_explanations.csv")
        self.local_fc_df = pd.read_csv(self.data_dir / "local_forecast_explanations.csv")
        self.local_anom_df = pd.read_csv(self.data_dir / "local_anomaly_explanations.csv")
        self.panels_payload: Dict[str, Any] = json.loads(
            (self.data_dir / "why_district_flagged_panels.json").read_text(encoding="utf-8")
        )
        self.global_summary: Dict[str, Any] = json.loads(
            (self.data_dir / "global_model_explanations_summary.json").read_text(encoding="utf-8")
        )

    def get_why_district_flagged_panel(self, district_name: str) -> Dict[str, Any]:
        """Return the 'Why was this district flagged?' panel payload for a Maharashtra district."""
        std_d, _ = standardize_district(district_name)
        for d_obj in self.panels_payload["districts"]:
            if d_obj["district_name"] == std_d:
                return d_obj
        raise ValueError(f"Unknown Maharashtra district '{district_name}' (standardized: '{std_d}').")

    def get_global_model_explanations(self, phase: Optional[str] = None) -> Dict[str, Any]:
        """Return global feature explanations and model versions."""
        if phase:
            sub = self.global_df[self.global_df["phase"].str.lower() == phase.strip().lower()]
            return {
                "engine_version": ENGINE_VERSION,
                "phase_queried": phase,
                "records": sub.head(30).to_dict(orient="records"),
                "scientific_disclaimer": MANDATORY_XAI_NON_CAUSAL_DISCLAIMER,
            }
        return self.global_summary

    def get_local_explanations(
        self,
        district_name: str,
        disease_name: Optional[str] = None,
        reporting_period: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return matched local risk, forecast (7d/14d/30d), and anomaly explanations for a query."""
        std_d, _ = standardize_district(district_name)
        r_sub = self.local_risk_df[self.local_risk_df["district"] == std_d]
        f_sub = self.local_fc_df[self.local_fc_df["district"] == std_d]
        a_sub = self.local_anom_df[self.local_anom_df["district"] == std_d]

        if disease_name:
            std_dis = standardize_disease(disease_name)[0] or disease_name
            r_sub = r_sub[r_sub["disease"].str.lower() == str(std_dis).lower()]
            f_sub = f_sub[f_sub["disease"].str.lower() == str(std_dis).lower()]
            a_sub = a_sub[a_sub["disease"].str.lower() == str(std_dis).lower()]

        if reporting_period:
            r_sub = r_sub[r_sub["reporting_period"] == reporting_period]
            f_sub = f_sub[f_sub["reporting_period"] == reporting_period]
            a_sub = a_sub[a_sub["reporting_period"] == reporting_period]

        return {
            "district": std_d,
            "disease_filter": disease_name,
            "reporting_period_filter": reporting_period,
            "xai_engine_version": ENGINE_VERSION,
            "scientific_disclaimer": MANDATORY_XAI_NON_CAUSAL_DISCLAIMER,
            "risk_explanations": r_sub.to_dict(orient="records"),
            "forecast_explanations": f_sub.to_dict(orient="records"),
            "anomaly_explanations": a_sub.to_dict(orient="records"),
        }
