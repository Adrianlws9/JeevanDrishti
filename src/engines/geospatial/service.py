"""Reusable FastAPI-Ready Geospatial Disease Hotspot Intelligence Service (Phase 9).

Provides structured queries for:
1. Maharashtra 36-district map layers (`observed_cases`, `historical_average`, `forecast_7d/14d/30d`,
   `anomaly_status`, `risk_category`, `gi_star_zscore`, `lisa_quadrant`).
2. Statistically significant hotspots (`p < 0.05` via Getis-Ord Gi* or Local Moran's I LISA).
3. District-level drill-down intelligence profiles (including alias resolution).
4. Verified 35-district Point GeoJSON payload + optional external polygon GeoJSON adapter.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from jeevan_pipeline.standardize import standardize_district
from src.engines.geospatial.geographic_standardization import attach_optional_polygon_geojson
from src.engines.geospatial.spatial_statistics import NON_TRANSMISSION_DISCLAIMER

PROJ = Path(__file__).resolve().parents[3]


class GeospatialHotspotService:
    """FastAPI-compatible service wrapper for Phase 9 Geospatial Hotspot Detection Engine."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else (PROJ / "data" / "geospatial")
        self.crosswalk_df = pd.read_csv(self.data_dir / "district_geographic_crosswalk.csv")
        self.weights_df = pd.read_csv(self.data_dir / "spatial_weights_matrix.csv")
        self.weekly_map_df = pd.read_csv(self.data_dir / "district_period_map_records.csv")
        self.summary_df = pd.read_csv(self.data_dir / "district_disease_map_summary.csv")
        self.global_moran_df = pd.read_csv(
            self.data_dir / "global_spatial_autocorrelation_summary.csv"
        )
        self.clusters_df = pd.read_csv(self.data_dir / "spatial_clusters_summary.csv")
        with open(self.data_dir / "district_drilldown_profiles.json", "r", encoding="utf-8") as f:
            self.drilldown_profiles = json.load(f)
        with open(self.data_dir / "maharashtra_district_points.geojson", "r", encoding="utf-8") as f:
            self.points_geojson = json.load(f)

    def get_district_map_layer(
        self,
        disease: str = "ALL_DISEASES",
        reporting_period: str = "2023-2025_POOL",
    ) -> Dict[str, Any]:
        """Return 36-district map records for a disease and reporting period with strict layer separation."""
        sub = self.summary_df[
            (self.summary_df["disease"] == disease)
            & (self.summary_df["reporting_period"] == reporting_period)
        ].copy()
        if sub.empty:
            return {
                "status": "UNSUPPORTED_DISEASE_OR_PERIOD",
                "disease": disease,
                "reporting_period": reporting_period,
                "records": [],
                "non_transmission_disclaimer": NON_TRANSMISSION_DISCLAIMER,
            }

        gm = self.global_moran_df[
            (self.global_moran_df["disease"] == disease)
            & (self.global_moran_df["reporting_period"] == reporting_period)
            & (self.global_moran_df["indicator"] == "observed_cases_raw")
        ]
        gm_info = gm.iloc[0].to_dict() if not gm.empty else None

        records = json.loads(sub.to_json(orient="records"))
        return {
            "status": "OK",
            "disease": disease,
            "reporting_period": reporting_period,
            "total_districts": len(records),
            "statistically_significant_hotspots_count": int(
                sub["is_statistically_significant_hotspot"].sum()
            ),
            "global_spatial_autocorrelation": gm_info,
            "records": records,
            "non_transmission_disclaimer": NON_TRANSMISSION_DISCLAIMER,
        }

    def get_verified_hotspots(
        self,
        disease: Optional[str] = None,
        reporting_period: str = "2023-2025_POOL",
    ) -> List[Dict[str, Any]]:
        """Return only districts that pass strict statistical hotspot testing (p < 0.05)."""
        sub = self.summary_df[
            (self.summary_df["reporting_period"] == reporting_period)
            & (self.summary_df["is_statistically_significant_hotspot"] == True)
        ]
        if disease:
            sub = sub[sub["disease"] == disease]
        sub = sub.sort_values(["gi_star_zscore", "observed_cases"], ascending=[False, False])
        return json.loads(sub.to_json(orient="records"))

    def get_district_drilldown(self, district_query: str) -> Dict[str, Any]:
        """Return complete multi-layer drill-down profile for a district (resolving aliases automatically)."""
        canon, map_type = standardize_district(district_query)
        if canon is None or canon not in self.drilldown_profiles:
            return {
                "status": "UNRECOGNIZED_DISTRICT",
                "queried_name": district_query,
                "message": f"'{district_query}' is not a valid Maharashtra district or documented alias.",
            }
        prof = dict(self.drilldown_profiles[canon])
        prof["queried_name"] = district_query
        prof["mapping_type"] = map_type
        prof["status"] = "OK"
        return prof

    def get_geojson_payload(
        self, external_polygon_geojson_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Return verified 35-district Point GeoJSON or join onto an optional external polygon GeoJSON."""
        if external_polygon_geojson_path is not None:
            pool_sub = self.summary_df[
                (self.summary_df["disease"] == "ALL_DISEASES")
                & (self.summary_df["reporting_period"] == "2023-2025_POOL")
            ]
            return attach_optional_polygon_geojson(pool_sub, external_polygon_geojson_path)
        return self.points_geojson
