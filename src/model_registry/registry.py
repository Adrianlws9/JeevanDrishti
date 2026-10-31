"""Model Registry & Lineage Management Engine for JEEVAN-DRISHTI (Phase 16 Sections 4, 5, 6, 7 & 9).

Maintains:
- `model_registry.json` adhering to the required 15-field schema (Section 4)
- Model status categorization: Validated prototype, Experimental, Data limited, Research, Unsupported (Section 5)
- End-to-end dataset lineage linking each model back to raw source files, cleaning version,
  feature manifest, and preprocessing config (Section 6)
- Model comparison tables (Section 7)
- Updates `model_dataset_registry.json` (Section 9)
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.features.geographic_features import DISTRICT_ID_MAP, OFFICIAL_DISTRICTS
from src.model_registry.evaluator import load_and_standardize_evaluation_metrics
from src.model_registry.model_inspector import (
    ModelArtifactRecord,
    REGISTRY_ENGINE_VERSION,
    inspect_all_workspace_models,
)

PROJ = Path(__file__).resolve().parents[2]

VALID_MODEL_STATUSES = (
    "Validated prototype",
    "Experimental",
    "Data limited",
    "Research",
    "Unsupported",
)


@dataclass
class DatasetLineageRecord:
    """End-to-end dataset lineage tracking for a registered model."""

    original_source_files: List[str]
    raw_checksums_sha256: Dict[str, str]
    cleaned_dataset_version: str
    cleaned_dataset_path: str
    feature_generation_version: str
    feature_dataset_path: str
    training_dataset_manifest: str
    preprocessing_configuration: str
    model_code_version: str
    source_time_span: str


def build_model_registry(
    proj_dir: Path = PROJ,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], pd.DataFrame]:
    """Construct complete model registry database, dataset lineage map, and flat summary catalog."""
    inspected_models = inspect_all_workspace_models(proj_dir)
    (
        metrics_catalog_df,
        baseline_comp_df,
        overfit_df,
        leakage_df,
        breakdown_df,
    ) = load_and_standardize_evaluation_metrics(proj_dir)

    registry_entries: List[Dict[str, Any]] = []
    flat_catalog_rows: List[Dict[str, Any]] = []

    # Map of supported districts (34 active + 2 documented unsupported)
    active_34 = [d for d in OFFICIAL_DISTRICTS if d not in ["Mumbai City", "Ratnagiri"]]

    # Common dataset lineage definitions across phases
    lineage_phase5_6_7 = DatasetLineageRecord(
        original_source_files=[
            "data/raw/maharashtra_weekly_outbreak_records_2023_2025_ORIGINAL_UPLOAD.csv",
            "data/raw/maharashtra_climate_daily_nasa_power_raw.csv",
            "data/raw/maharashtra_climate_daily_era5_raw.csv",
        ],
        raw_checksums_sha256={
            "idsp_outbreaks_raw": "1578113d497308fc4ba0b308b97c62ac746f2ae11fc879516b8fdadcbf5a364d",
            "nasa_power_climate_raw": "698c92849e776953ee6cf027cf85fc2c7104cfbc1767d966d95324ec9d936e78",
            "era5_climate_raw": "9f70d5e2e8df3111f185df3603410f8a8bb23126be1f727c99276229158c3a9f",
        },
        cleaned_dataset_version="phase2-v1",
        cleaned_dataset_path="data/processed/cleaned_disease_data.csv",
        feature_generation_version="phase3-v1",
        feature_dataset_path="data/features/disease_climate_weekly_features.csv",
        training_dataset_manifest="data/final_training/training_manifest.json",
        preprocessing_configuration="models/risk_classification/preprocessing_pipeline.pkl",
        model_code_version="src/risk_classification/ & src/disease_forecasting/ & src/anomaly_detection/",
        source_time_span="2023-W01..2025-W52 (511 deduplicated outbreak weeks across 34 active districts)",
    )

    lineage_phase8_climate = DatasetLineageRecord(
        original_source_files=[
            "data/raw/maharashtra_weekly_outbreak_records_2023_2025_ORIGINAL_UPLOAD.csv",
            "data/raw/maharashtra_climate_daily_nasa_power_raw.csv",
        ],
        raw_checksums_sha256={
            "idsp_outbreaks_raw": "1578113d497308fc4ba0b308b97c62ac746f2ae11fc879516b8fdadcbf5a364d",
            "nasa_power_climate_raw": "698c92849e776953ee6cf027cf85fc2c7104cfbc1767d966d95324ec9d936e78",
        },
        cleaned_dataset_version="phase2-v1",
        cleaned_dataset_path="data/processed/cleaned_climate_data.csv",
        feature_generation_version="phase3-v1",
        feature_dataset_path="data/climate_intelligence/aligned_weekly_climate_disease_outbreaks.csv",
        training_dataset_manifest="data/climate_intelligence/data_coverage_and_alignment_audit.json",
        preprocessing_configuration="src/engines/climate_intelligence/data_alignment.py",
        model_code_version="src/engines/climate_intelligence/ (phase8-v1.0.0)",
        source_time_span="2023-W01..2025-W52 (121 active surveillance weeks)",
    )

    lineage_phase9_geospatial = DatasetLineageRecord(
        original_source_files=[
            "data/raw/maharashtra_weekly_outbreak_records_2023_2025_ORIGINAL_UPLOAD.csv",
            "data/raw/district_geographic_crosswalk.csv",
        ],
        raw_checksums_sha256={
            "idsp_outbreaks_raw": "1578113d497308fc4ba0b308b97c62ac746f2ae11fc879516b8fdadcbf5a364d",
        },
        cleaned_dataset_version="phase2-v1",
        cleaned_dataset_path="data/geospatial/district_geographic_crosswalk.csv",
        feature_generation_version="phase9-v1",
        feature_dataset_path="data/geospatial/spatial_weights_matrix.csv",
        training_dataset_manifest="data/geospatial/district_disease_map_summary.csv",
        preprocessing_configuration="src/engines/geospatial/crosswalk.py",
        model_code_version="src/engines/geospatial/ (phase9-v1.0.0)",
        source_time_span="2023-2025 Pooled Surveillance (36 districts, 35 geocoded centroids)",
    )

    lineage_phase12_mortality = DatasetLineageRecord(
        original_source_files=[
            "data/raw/maharashtra_weekly_outbreak_records_2023_2025_ORIGINAL_UPLOAD.csv",
        ],
        raw_checksums_sha256={
            "idsp_outbreaks_raw": "1578113d497308fc4ba0b308b97c62ac746f2ae11fc879516b8fdadcbf5a364d",
        },
        cleaned_dataset_version="phase2-v1",
        cleaned_dataset_path="data/processed/cleaned_disease_data.csv",
        feature_generation_version="phase3-v1",
        feature_dataset_path="data/severity_intelligence/verified_weekly_mortality_records.csv",
        training_dataset_manifest="data/severity_intelligence/outcome_data_availability_audit.csv",
        preprocessing_configuration="src/engines/severity_intelligence/data_preparation.py",
        model_code_version="src/engines/severity_intelligence/ (phase12-v1.0.0)",
        source_time_span="2023-W01..2025-W52 (199 verified deaths across 511 outbreak weeks)",
    )

    for m in inspected_models:
        if m.model_status not in VALID_MODEL_STATUSES:
            raise ValueError(f"Invalid model status '{m.model_status}'. Must be one of {VALID_MODEL_STATUSES}")

        # Link lineage
        if "P05" in m.model_id or "P06" in m.model_id or "P07" in m.model_id:
            lin = lineage_phase5_6_7
        elif "P08" in m.model_id:
            lin = lineage_phase8_climate
        elif "P09" in m.model_id or "P10" in m.model_id or "P14" in m.model_id:
            lin = lineage_phase9_geospatial
        elif "P12" in m.model_id:
            lin = lineage_phase12_mortality
        else:
            lin = lineage_phase5_6_7

        # Extract metrics associated with this model
        sub_metrics = metrics_catalog_df[metrics_catalog_df["model_name"].str.contains(m.model_name.split("_")[0], case=False, na=False)]
        eval_metrics_dict: Dict[str, Any] = {}
        if not sub_metrics.empty:
            for _, r_m in sub_metrics.iterrows():
                spl_k = str(r_m["split"])
                if spl_k not in eval_metrics_dict:
                    eval_metrics_dict[spl_k] = {}
                p_name = str(r_m["primary_metric_name"])
                p_val = r_m["primary_metric_value"]
                if pd.notna(p_val):
                    eval_metrics_dict[spl_k][p_name] = p_val
                if pd.notna(r_m.get("secondary_metric_value")):
                    eval_metrics_dict[spl_k][str(r_m["secondary_metric_name"])] = r_m["secondary_metric_value"]
        else:
            eval_metrics_dict = {
                "primary_metric": {m.primary_task_metric: m.primary_metric_value},
                "baseline_metric": m.baseline_metric_value,
                "improvement_pct": m.metric_improvement_pct,
            }

        # Supported districts list
        supp_dists = active_34 if m.supported_districts_count == 34 else OFFICIAL_DISTRICTS

        entry = {
            # Required Section 4 Schema Fields
            "model_id": m.model_id,
            "model_name": m.model_name,
            "engine_phase": m.engine_phase,
            "engine_name": m.engine_name,
            "model_type": m.model_type,
            "disease": m.disease,
            "supported_districts": supp_dists,
            "supported_districts_count": len(supp_dists),
            "unsupported_districts": m.unsupported_districts,
            "dataset_filenames": m.dataset_filenames,
            "dataset_version": m.dataset_version,
            "feature_version": m.feature_version,
            "training_period": m.training_period,
            "validation_period": m.validation_period,
            "testing_period": m.testing_period,
            "training_timestamp": m.training_timestamp or datetime.now().isoformat(),
            "model_version": m.model_version,
            "evaluation_metrics": eval_metrics_dict,
            "artifact_path": m.artifact_path,
            "artifact_exists": m.artifact_exists,
            "artifact_size_bytes": m.artifact_size_bytes,
            "model_status": m.model_status,
            # Governance & Lineage Metadata (Sections 5 & 6)
            "status_justification": m.status_justification,
            "production_readiness_disclaimer": (
                "NOT APPROVED FOR AUTONOMOUS CLINICAL ACTION. This model is a "
                f"'{m.model_status}' designed strictly for epidemiological decision support."
            ),
            "dataset_lineage": asdict(lin),
        }
        registry_entries.append(entry)

        flat_catalog_rows.append(
            {
                "model_id": m.model_id,
                "model_name": m.model_name,
                "engine_phase": m.engine_phase,
                "model_type": m.model_type,
                "model_version": m.model_version,
                "model_status": m.model_status,
                "disease": m.disease,
                "supported_districts_count": len(supp_dists),
                "unsupported_districts_count": len(m.unsupported_districts),
                "training_period": m.training_period,
                "validation_period": m.validation_period,
                "testing_period": m.testing_period,
                "dataset_version": m.dataset_version,
                "primary_task_metric": m.primary_task_metric,
                "model_metric_value": m.primary_metric_value,
                "baseline_metric_value": m.baseline_metric_value,
                "improvement_vs_baseline_pct": m.metric_improvement_pct,
                "artifact_path": m.artifact_path,
                "artifact_exists": m.artifact_exists,
                "status_justification": m.status_justification,
            }
        )

    flat_catalog_df = pd.DataFrame(flat_catalog_rows)

    full_registry_dict = {
        "registry_version": REGISTRY_ENGINE_VERSION,
        "generated_timestamp": datetime.now().isoformat(),
        "total_registered_models": len(registry_entries),
        "status_distribution": flat_catalog_df["model_status"].value_counts().to_dict(),
        "type_distribution": flat_catalog_df["model_type"].value_counts().to_dict(),
        "models": registry_entries,
    }

    return registry_entries, full_registry_dict, flat_catalog_df
