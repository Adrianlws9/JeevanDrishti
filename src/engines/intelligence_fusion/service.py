"""FastAPI-Ready Query & On-Demand Fusion Service for Phase 15 Predictive Intelligence Fusion Engine."""
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.engines.intelligence_fusion.alignment_validator import (
    align_weekly_and_district_engine_outputs,
    check_reporting_period_compatibility,
    deduplicate_engine_dataframe,
    evaluate_model_and_data_freshness,
    validate_district_and_disease,
)
from src.engines.intelligence_fusion.engine_inspector import (
    FUSION_ENGINE_VERSION,
    MANDATORY_FUSION_GOVERNANCE_DISCLAIMER,
    PROJ,
    inspect_and_load_upstream_engines,
)
from src.engines.intelligence_fusion.fusion_engine import (
    CANONICAL_SIGNAL_STATUSES,
    fuse_dataframe_records,
    fuse_single_aligned_record,
)


class PredictiveIntelligenceFusionService:
    """High-level query and custom record fusion service for JEEVAN-DRISHTI Phase 15."""

    def __init__(self, proj_dir: Path = PROJ) -> None:
        self.proj_dir = proj_dir
        self.data_dir = proj_dir / "data" / "fused_intelligence"
        self.audit_df, self.loaded_bundle = inspect_and_load_upstream_engines(proj_dir)
        self.aligned_weekly_df, self.aligned_pool_df, self.align_meta = (
            align_weekly_and_district_engine_outputs(self.loaded_bundle)
        )
        self.weekly_nested, self.weekly_fused_df = fuse_dataframe_records(
            self.aligned_weekly_df
        )
        self.pool_nested, self.pool_fused_df = fuse_dataframe_records(
            self.aligned_pool_df
        )

    def get_upstream_engines_audit(self) -> Dict[str, Any]:
        """Return inspection status of all 9 integrated analytical engines."""
        return {
            "fusion_engine_version": FUSION_ENGINE_VERSION,
            "governance_disclaimer": MANDATORY_FUSION_GOVERNANCE_DISCLAIMER,
            "canonical_signal_statuses": list(CANONICAL_SIGNAL_STATUSES),
            "engines": self.audit_df.to_dict(orient="records"),
            "alignment_metadata": self.align_meta,
        }

    def get_district_pooled_intelligence(self, district: str) -> Dict[str, Any]:
        """Return the unified 2023-2025_POOL intelligence record for any Maharashtra district (supports historical aliases)."""
        dist_id, std_dist, _ = validate_district_and_disease(district, "ALL_DISEASES")
        for rec in self.pool_nested:
            if rec["district_id"] == dist_id or rec["district_name"] == std_dist:
                return rec
        raise KeyError(f"District '{std_dist}' not found in pooled intelligence records.")

    def query_weekly_fused_intelligence(
        self,
        district: Optional[str] = None,
        disease: Optional[str] = None,
        reporting_period: Optional[str] = None,
        signal_status: Optional[str] = None,
        disagreement_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Filter weekly fused intelligence records by district, disease, reporting period, or signal status."""
        std_dist = None
        if district:
            _, std_dist, _ = validate_district_and_disease(district, disease or "ALL_DISEASES")

        std_dis = None
        if disease:
            _, _, std_dis = validate_district_and_disease(district or "Pune", disease)

        results: List[Dict[str, Any]] = []
        for rec in self.weekly_nested:
            if std_dist and rec["district_name"] != std_dist:
                continue
            if std_dis and rec["disease"] != std_dis:
                continue
            if reporting_period and rec["reporting_period"] != str(reporting_period).strip():
                continue
            if signal_status and rec["overall_signal_status"] != signal_status:
                continue
            if disagreement_only and not rec["disagreement_flag"]:
                continue
            results.append(rec)
        return results

    def fuse_custom_aligned_payload(
        self,
        payload: Dict[str, Any],
        reference_now: Optional[datetime] = None,
        max_staleness_days: int = 30,
    ) -> Dict[str, Any]:
        """Validate and fuse a custom multi-engine record dictionary (used for real-time API fusion & testing)."""
        # Verify reporting period compatibility if separate engine reporting periods are passed
        eng_periods = payload.get("engine_reporting_periods", {})
        base_period = str(payload.get("reporting_period", "UNKNOWN"))
        for eng_name, p_val in eng_periods.items():
            allow_pool = eng_name in {"geospatial_hotspot", "spread_simulation", "resource_intelligence"}
            ok, reason = check_reporting_period_compatibility(
                base_period, str(p_val), allow_pooled_context_attachment=allow_pool
            )
            if not ok:
                raise ValueError(
                    f"Reporting period mismatch for engine '{eng_name}': {reason}"
                )

        return fuse_single_aligned_record(
            payload,
            reference_now=reference_now,
            max_staleness_days=max_staleness_days,
        )
