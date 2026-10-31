"""Reusable Climate-Disease Intelligence Service for FastAPI Integration."""
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from . import CAUSALITY_DISCLAIMER, CLIMATE_INTELLIGENCE_VERSION

PROJ = Path(__file__).resolve().parents[3]
CLIMATE_INTEL_DATA_DIR = PROJ / "data" / "climate_intelligence"


class ClimateDiseaseIntelligenceService:
    """Provides structured queries over precomputed climate-disease correlations, lags,
    regressions, and TreeSHAP importances with mandatory non-causal interpretation metadata."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else CLIMATE_INTEL_DATA_DIR
        self.disease_corr = pd.read_csv(self.data_dir / "disease_climate_correlations.csv")
        self.district_rel = pd.read_csv(self.data_dir / "district_climate_relationships.csv")
        self.lag_df = pd.read_csv(self.data_dir / "lag_analysis_results.csv")
        self.reg_df = pd.read_csv(self.data_dir / "regression_analysis_results.csv")
        self.imp_df = pd.read_csv(self.data_dir / "climate_feature_importance_and_shap.csv")
        with open(self.data_dir / "data_coverage_and_alignment_audit.json", "r", encoding="utf-8") as f:
            self.audit = json.load(f)

    def get_disease_climate_profile(self, disease: str) -> Dict[str, Any]:
        """Return statistical, lagged, and TreeSHAP climate associations for a disease."""
        sub = self.disease_corr[self.disease_corr["disease_std"] == disease]
        if sub.empty:
            return {
                "disease": disease,
                "is_supported": False,
                "analysis_status": "UNSUPPORTED_DISEASE",
                "causality_disclaimer": CAUSALITY_DISCLAIMER,
                "engine_version": CLIMATE_INTELLIGENCE_VERSION,
            }
        row = sub.iloc[0].to_dict()
        lags = self.lag_df[self.lag_df["disease_or_series"] == disease].to_dict(orient="records")
        shap_sub = self.imp_df[self.imp_df["analysis_task"] == f"Disease_Specific_{disease}"]
        if shap_sub.empty:
            shap_sub = self.imp_df[self.imp_df["analysis_task"] == "Pooled_Outbreak_Severity_All_Diseases"]
        return {
            "disease": disease,
            "is_supported": row["analysis_status"] == "SUPPORTED_STATISTICAL_ANALYSIS",
            "summary": row,
            "lag_correlations": lags[:12],
            "top_shap_climate_features": shap_sub.head(10).to_dict(orient="records"),
            "causality_disclaimer": CAUSALITY_DISCLAIMER,
            "engine_version": CLIMATE_INTELLIGENCE_VERSION,
        }

    def get_district_climate_profile(self, district: str) -> Dict[str, Any]:
        """Return climate-disease relationship record for a Maharashtra district."""
        sub = self.district_rel[self.district_rel["district_std"] == district]
        if sub.empty:
            return {
                "district": district,
                "is_supported": False,
                "analysis_status": "UNSUPPORTED_DISTRICT",
                "causality_disclaimer": CAUSALITY_DISCLAIMER,
                "engine_version": CLIMATE_INTELLIGENCE_VERSION,
            }
        row = sub.iloc[0].to_dict()
        return {
            "district": district,
            "is_supported": "SUPPORTED" in str(row["analysis_status"]) or "LIMITED" in str(row["analysis_status"]),
            "profile": row,
            "causality_disclaimer": CAUSALITY_DISCLAIMER,
            "engine_version": CLIMATE_INTELLIGENCE_VERSION,
        }
