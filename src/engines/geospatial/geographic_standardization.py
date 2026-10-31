"""Geographic Data Inspection, Crosswalk Standardization, and Geometry Adapter (Phase 9).

Responsibilities:
1. Inspect existing geographic assets in the workspace (boundary files, GeoJSON/shapefiles,
   district coordinate tables, alias crosswalks, and population files).
2. Document the absence of polygon boundary files (.geojson/.shp) and population denominator
   files without fabricating polygon boundaries or population numbers.
3. Standardize all 36 Maharashtra districts with canonical identifiers (`MH_D01..MH_D36`),
   handle alternate spellings, historical district renamings (`Aurangabad` -> `Chhatrapati Sambhajinagar`,
   `Osmanabad` -> `Dharashiv`, `Ahilyanagar` -> `Ahmednagar`), and boundary splits
   (`Palghar` carved out of `Thane` on `2014-08-01`).
4. Export a verified GeoJSON Point FeatureCollection (`maharashtra_district_points.geojson`)
   using only the 35 verified district point coordinates (with `geometry: null` for `Mumbai Suburban`),
   plus an optional adapter interface to attach external polygon boundaries if provided in the future.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from jeevan_pipeline.standardize import norm_name, repair_split_district, standardize_district
from src.features.geographic_features import (
    ADMIN_DIVISION_MAP,
    COASTAL_DISTRICTS,
    DISTRICT_ID_MAP,
    OFFICIAL_DISTRICTS,
)

PROJ = Path(__file__).resolve().parents[3]

# Documented historical renamings and boundary changes in Maharashtra
HISTORICAL_BOUNDARY_NOTES: Dict[str, str] = {
    "Chhatrapati Sambhajinagar": (
        "Renamed from Aurangabad (name change only; identical district boundary). "
        "Also repairs PDF column-split 'Chhatrapati' + 'Sambhajinagar'."
    ),
    "Dharashiv": (
        "Renamed from Osmanabad / Osmanabad (Maharashtra) (name change only; identical district boundary)."
    ),
    "Ahmednagar": (
        "Officially renamed Ahilyanagar by Maharashtra Government; canonicalized to Ahmednagar "
        "across all 2009-2025 project datasets with full alias support for 'Ahilyanagar' and 'Ahmadnagar'."
    ),
    "Palghar": (
        "Carved out of Thane district on 2014-08-01. In 2023-2025 weekly IDSP data, Palghar and Thane "
        "are reported separately. Pre-2014 historical records in maharashtra_epiclim_filtered.csv "
        "under Thane include the area later split into Palghar and are never silently merged."
    ),
    "Thane": (
        "Parent district of Palghar prior to 2014-08-01 split. Pre-2014 historical records reflect "
        "undivided Thane; 2023-2025 records reflect post-split Thane."
    ),
    "Mumbai Suburban": (
        "Present in 2023-2025 IDSP weekly outbreaks (3 outbreak weeks; repaired from column split "
        "'Mumbai' + 'Suburban'), but has NO latitude/longitude coordinates in any workspace file. "
        "Marked as Insufficient Data (No Coordinates) for spatial weights and coordinate mapping."
    ),
    "Mumbai City": (
        "Has verified coordinates (18.9734N, 72.8281E) from historical_epiclim ('Mumbai (city)'), "
        "but has 0 reported outbreak records in 2023-2025 IDSP weekly surveillance. "
        "Marked as Insufficient Data (Zero Outbreak Records in 2023-2025)."
    ),
    "Ratnagiri": (
        "Has verified coordinates (17.2826N, 73.4570E) from historical_epiclim, "
        "but has 0 reported outbreak records in 2023-2025 IDSP weekly surveillance. "
        "Marked as Insufficient Data (Zero Outbreak Records in 2023-2025)."
    ),
    "Gadchiroli": (
        "Has verified coordinates (19.7591N, 80.1623E); nearest 0.25-deg IMD rainfall grid cell is masked "
        "and uses documented 3x3 neighborhood mean fallback."
    ),
}


def inspect_workspace_geographic_assets(proj_dir: Path = PROJ) -> Dict[str, Any]:
    """Audit the project workspace for polygon boundary files, coordinate tables, crosswalks, and population data."""
    polygon_files = (
        list(proj_dir.rglob("*.geojson"))
        + list(proj_dir.rglob("*.shp"))
        + list(proj_dir.rglob("*.gpkg"))
        + list(proj_dir.rglob("*.kml"))
    )
    # Exclude our own generated point geojson if re-running pipeline
    external_polygon_files = [
        str(p.relative_to(proj_dir))
        for p in polygon_files
        if "maharashtra_district_points.geojson" not in p.name
    ]

    grid_cells_path = proj_dir / "data" / "interim" / "district_grid_cells.csv"
    alias_path = proj_dir / "data" / "reference" / "district_alias_table.csv"
    mapping_path = proj_dir / "data" / "reports" / "district_mapping.csv"

    grid_df = pd.read_csv(grid_cells_path)
    geocoded_count = int(grid_df["lat"].notna().sum())
    missing_coord_districts = grid_df.loc[grid_df["lat"].isna(), "district_std"].tolist()

    return {
        "polygon_boundary_files_found": external_polygon_files,
        "has_polygon_boundaries": len(external_polygon_files) > 0,
        "polygon_limitation_note": (
            "No district polygon boundary files (.geojson, .shp, .gpkg, .kml) exist in the project workspace. "
            "Per strict scientific rules, district boundary polygons are NOT downloaded or fabricated. "
            "Spatial weights (k-NN and inverse Haversine distance) and cartographic maps use the 35 verified "
            "district point coordinates from data/interim/district_grid_cells.csv, while exposing a clean "
            "GeoJSON polygon adapter interface for future boundary file attachment."
        ),
        "population_dataset_found": False,
        "population_limitation_note": (
            "No district population or census denominator file exists in the workspace. "
            "Incidence rates per 100,000 population are set to NaN (UNAVAILABLE_NO_POPULATION_DENOMINATOR) "
            "and never fabricated."
        ),
        "total_official_districts": len(OFFICIAL_DISTRICTS),
        "geocoded_districts_count": geocoded_count,
        "missing_coordinate_districts": missing_coord_districts,
        "coordinate_source_file": str(grid_cells_path.relative_to(proj_dir)),
        "alias_table_file": str(alias_path.relative_to(proj_dir)),
        "district_mapping_report_file": str(mapping_path.relative_to(proj_dir)),
    }


def build_geographic_crosswalk(proj_dir: Path = PROJ) -> pd.DataFrame:
    """Build the authoritative 36-district geographic standardization crosswalk for Maharashtra.

    Ensures zero duplicate district IDs, documents all aliases, historical renamings,
    boundary splits, coordinate availability, and 2023-2025 vs 2009-2022 record counts.
    """
    grid_df = pd.read_csv(proj_dir / "data" / "interim" / "district_grid_cells.csv")
    alias_df = pd.read_csv(proj_dir / "data" / "reference" / "district_alias_table.csv")
    cleaned_dis = pd.read_csv(proj_dir / "data" / "processed" / "cleaned_disease_data.csv")
    weekly_feat = pd.read_csv(proj_dir / "data" / "features" / "disease_climate_weekly_features.csv")

    # Group aliases per canonical district
    alias_map: Dict[str, List[str]] = {d: [] for d in OFFICIAL_DISTRICTS}
    for _, row in alias_df.iterrows():
        canon = str(row["canonical"]).strip()
        raw_n = str(row["raw_name"]).strip()
        kind = str(row["kind"]).strip()
        if canon in alias_map and kind == "alias":
            alias_map[canon].append(raw_n)

    # Historical (2009-2022) vs IDSP (2023-2025) counts
    hist_df = cleaned_dis[cleaned_dis["source_dataset"] == "historical_epiclim"]
    idsp_df = cleaned_dis[cleaned_dis["source_dataset"] == "idsp_weekly"]

    hist_counts = hist_df.groupby("district_std").size().to_dict()
    idsp_raw_counts = idsp_df.groupby("district_std").size().to_dict()
    idsp_dedup_counts = weekly_feat.groupby("district_std").size().to_dict()
    idsp_cases_sum = weekly_feat.groupby("district_std")["target_cases"].sum().to_dict()
    idsp_diseases_cnt = weekly_feat.groupby("district_std")["disease_std"].nunique().to_dict()

    grid_lookup = grid_df.set_index("district_std").to_dict(orient="index")

    rows = []
    for d in OFFICIAL_DISTRICTS:
        g = grid_lookup.get(d, {})
        lat = g.get("lat", np.nan)
        lon = g.get("lon", np.nan)
        has_coords = bool(pd.notna(lat) and pd.notna(lon))
        n_hist = int(hist_counts.get(d, 0))
        n_idsp_raw = int(idsp_raw_counts.get(d, 0))
        n_idsp_dedup = int(idsp_dedup_counts.get(d, 0))
        tot_cases_23_25 = float(idsp_cases_sum.get(d, 0.0)) if n_idsp_dedup > 0 else np.nan
        n_diseases_23_25 = int(idsp_diseases_cnt.get(d, 0))

        if not has_coords:
            geo_status = "INSUFFICIENT_DATA_NO_COORDINATES"
            map_status_label = "Insufficient Data (No Coordinates)"
        elif n_idsp_dedup == 0:
            geo_status = "INSUFFICIENT_DATA_ZERO_OUTBREAK_RECORDS_2023_2025"
            map_status_label = "Insufficient Data (0 Outbreaks in 2023-2025)"
        else:
            geo_status = "COMPLETE_GEOCODED_WITH_SURVEILLANCE_DATA"
            map_status_label = "Supported (Geocoded + Active Surveillance)"

        aliases_list = alias_map.get(d, [])
        if d == "Mumbai Suburban":
            aliases_list = sorted(set(aliases_list + ["Mumbai Suburban (split: 'Mumbai' + 'Suburban')"]))
        elif d == "Chhatrapati Sambhajinagar":
            aliases_list = sorted(
                set(aliases_list + ["Chhatrapati Sambhajinagar (split: 'Chhatrapati' + 'Sambhajinagar')"])
            )

        rows.append(
            {
                "district_id": DISTRICT_ID_MAP[d],
                "district_name": d,
                "administrative_division": ADMIN_DIVISION_MAP[d],
                "is_coastal": bool(d in COASTAL_DISTRICTS),
                "latitude": round(float(lat), 6) if has_coords else np.nan,
                "longitude": round(float(lon), 6) if has_coords else np.nan,
                "has_verified_coordinates": has_coords,
                "polygon_geometry_available": False,
                "documented_aliases": "; ".join(aliases_list) if aliases_list else "None (Exact Match Only)",
                "historical_2009_2022_records": n_hist,
                "idsp_2023_2025_raw_records": n_idsp_raw,
                "idsp_2023_2025_dedup_outbreak_weeks": n_idsp_dedup,
                "idsp_2023_2025_distinct_diseases": n_diseases_23_25,
                "idsp_2023_2025_total_observed_cases": tot_cases_23_25,
                "population_denominator": np.nan,
                "incidence_rate_supported": False,
                "data_quality_status": geo_status,
                "map_display_status": map_status_label,
                "boundary_and_historical_note": HISTORICAL_BOUNDARY_NOTES.get(
                    d, "Stable district boundary and canonical name across 2009-2025."
                ),
            }
        )

    crosswalk_df = pd.DataFrame(rows)
    # Strict uniqueness validation
    assert len(crosswalk_df) == 36, "Crosswalk must have exactly 36 official Maharashtra districts."
    assert crosswalk_df["district_id"].nunique() == 36, "Duplicate district_id detected."
    assert crosswalk_df["district_name"].nunique() == 36, "Duplicate district_name detected."
    return crosswalk_df


def export_district_points_geojson(
    summary_df: pd.DataFrame, output_path: Path
) -> Dict[str, Any]:
    """Export a standards-compliant GeoJSON FeatureCollection of Maharashtra districts.

    Uses Point geometries for the 35 districts with verified coordinates and `geometry: null`
    for `Mumbai Suburban` (never fabricating coordinates or polygon boundaries).
    """
    features = []
    for _, row in summary_df.iterrows():
        lat = row.get("latitude", np.nan)
        lon = row.get("longitude", np.nan)
        if pd.notna(lat) and pd.notna(lon):
            geom = {
                "type": "Point",
                "coordinates": [round(float(lon), 6), round(float(lat), 6)],
            }
        else:
            geom = None

        props = {}
        for k, v in row.items():
            if pd.isna(v):
                props[k] = None
            elif isinstance(v, (np.integer, int)):
                props[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                props[k] = round(float(v), 4)
            elif isinstance(v, (np.bool_, bool)):
                props[k] = bool(v)
            else:
                props[k] = str(v)

        features.append(
            {
                "type": "Feature",
                "id": str(row["district_id"]),
                "geometry": geom,
                "properties": props,
            }
        )

    fc = {
        "type": "FeatureCollection",
        "metadata": {
            "project": "JEEVAN-DRISHTI",
            "engine": "Phase 9 Geospatial Disease Hotspot Detection Engine",
            "geometry_type": "Verified_District_Point_Centroids_Only (No Polygon Boundary File in Workspace)",
            "crs": "EPSG:4326 (WGS84)",
            "total_districts": len(features),
            "geocoded_point_features": sum(1 for f in features if f["geometry"] is not None),
            "null_geometry_features": sum(1 for f in features if f["geometry"] is None),
            "non_fabrication_notice": (
                "District boundary polygons are absent from the project workspace and are never invented. "
                "To attach official polygon boundaries in the future, pass a 36-district GeoJSON file to "
                "attach_optional_polygon_geojson()."
            ),
        },
        "features": features,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=2)
    return fc


def attach_optional_polygon_geojson(
    district_records_df: pd.DataFrame,
    external_geojson_path: Optional[Path] = None,
    district_property_key: str = "district",
) -> Dict[str, Any]:
    """Adapter interface to join district disease/hotspot records onto an external polygon GeoJSON.

    If `external_geojson_path` is None or does not exist, returns a structured status payload
    explaining that polygon boundaries are unavailable and falls back to the verified point GeoJSON.
    """
    if external_geojson_path is None or not Path(external_geojson_path).exists():
        return {
            "status": "POLYGON_BOUNDARY_FILE_UNAVAILABLE",
            "has_polygons": False,
            "message": (
                "No external district polygon GeoJSON/shapefile was provided or found in workspace. "
                "Using verified 35-district point geometry interface without fabricating polygons."
            ),
            "matched_districts": 0,
            "unmatched_features": [],
        }

    with open(external_geojson_path, "r", encoding="utf-8") as f:
        geo = json.load(f)

    rec_lookup = district_records_df.set_index("district_name").to_dict(orient="index")
    matched = 0
    unmatched = []
    for feat in geo.get("features", []):
        props = feat.get("properties", {})
        raw_name = str(props.get(district_property_key, ""))
        canon, _ = standardize_district(raw_name)
        if canon and canon in rec_lookup:
            props.update(rec_lookup[canon])
            matched += 1
        else:
            unmatched.append(raw_name)

    return {
        "status": "POLYGON_BOUNDARY_JOINED",
        "has_polygons": True,
        "matched_districts": matched,
        "unmatched_features": unmatched,
        "geojson": geo,
    }
