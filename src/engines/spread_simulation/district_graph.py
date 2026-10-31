"""District Graph Construction & Geographic/Mobility Asset Audit (Phase 10).

Responsibilities:
1. Inspect existing project datasets for district boundaries, adjacency tables, coordinates,
   road networks, human mobility data, population denominators, and disease history.
2. Build the 36-district Maharashtra graph:
   - 35 geocoded districts connected via the validated symmetric k-NN (k=4) spatial adjacency
     edges from `data/geospatial/spatial_weights_matrix.csv` (83 undirected edges / 166 directed arcs).
   - `Mumbai Suburban` (`MH_D19`) included as an explicit isolated node (`degree = 0`) because it
     lacks coordinates in the workspace — never inventing edges or coordinates.
3. Document that empirical human mobility and road-network datasets are absent from the workspace,
   so edge weights represent geometric/topological proximity assumptions only.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.features.geographic_features import (
    ADMIN_DIVISION_MAP,
    COASTAL_DISTRICTS,
    DISTRICT_ID_MAP,
    OFFICIAL_DISTRICTS,
)

PROJ = Path(__file__).resolve().parents[3]

MANDATORY_SCENARIO_DISCLAIMER = (
    "HYPOTHETICAL SCENARIO SIMULATION ONLY — NOT A REAL OUTBREAK OR DISEASE-SPREAD PREDICTION. "
    "No empirical human mobility, road-traffic, or biological transmission parameters exist in the "
    "workspace; propagation reflects user-configured graph adjacency assumptions only."
)


def inspect_spread_simulation_assets(proj_dir: Path = PROJ) -> Dict[str, Any]:
    """Audit workspace datasets required for Prompt 10 Section 1."""
    cw_path = proj_dir / "data" / "geospatial" / "district_geographic_crosswalk.csv"
    sw_path = proj_dir / "data" / "geospatial" / "spatial_weights_matrix.csv"
    grid_path = proj_dir / "data" / "interim" / "district_grid_cells.csv"
    dis_path = proj_dir / "data" / "processed" / "cleaned_disease_data.csv"

    cw_df = pd.read_csv(cw_path)
    sw_df = pd.read_csv(sw_path)

    n_geocoded = int(cw_df["has_verified_coordinates"].sum())
    n_directed_arcs = int(sw_df["degree_symmetric_knn"].sum())

    return {
        "district_boundary_polygons_available": False,
        "validated_adjacency_file_available": True,
        "validated_adjacency_source": str(sw_path.relative_to(proj_dir).as_posix()),
        "geographic_coordinates_available": True,
        "geocoded_districts_count": n_geocoded,
        "ungeocoded_districts": ["Mumbai Suburban"],
        "coordinate_source": str(grid_path.relative_to(proj_dir).as_posix()),
        "road_network_data_available": False,
        "mobility_data_available": False,
        "population_data_available": False,
        "disease_history_available": True,
        "disease_history_source": str(dis_path.relative_to(proj_dir).as_posix()),
        "undirected_edges_count": n_directed_arcs // 2,
        "directed_arcs_count": n_directed_arcs,
        "mobility_handling_rule": (
            "No authorized human mobility, commuter flow, or road-traffic dataset exists in the workspace. "
            "Equal inter-district movement is NOT assumed, and traffic/migration numbers are NEVER invented. "
            "Simulations use documented geographic adjacency weights (distance-decayed, inverse-distance, "
            "or division-bounded) strictly for illustrative what-if scenario exploration."
        ),
        "non_prediction_disclaimer": MANDATORY_SCENARIO_DISCLAIMER,
    }


@dataclass
class DistrictGraph:
    """In-memory representation of the 36-district Maharashtra graph."""
    nodes_df: pd.DataFrame
    edges_df: pd.DataFrame
    districts: List[str]                  # All 36 official districts in canonical order
    district_ids: List[str]               # MH_D01..MH_D36
    district_to_idx: Dict[str, int]
    adj_distance_km: np.ndarray           # (36, 36) Haversine distance in km for connected edges, 0 if not connected
    adj_binary: np.ndarray                # (36, 36) 1.0 if connected in validated k-NN graph, 0.0 otherwise
    adj_same_division: np.ndarray         # (36, 36) 1.0 if connected and in same revenue division, 0.0 otherwise


def build_district_graph(proj_dir: Path = PROJ) -> DistrictGraph:
    """Construct the 36-node Maharashtra district graph from existing validated geospatial files.

    Uses `data/geospatial/district_geographic_crosswalk.csv`, `data/geospatial/spatial_weights_matrix.csv`,
    and `data/geospatial/district_disease_map_summary.csv`. Does not invent connections for `Mumbai Suburban`.
    """
    cw_df = pd.read_csv(proj_dir / "data" / "geospatial" / "district_geographic_crosswalk.csv")
    sw_df = pd.read_csv(proj_dir / "data" / "geospatial" / "spatial_weights_matrix.csv")
    sum_df = pd.read_csv(proj_dir / "data" / "geospatial" / "district_disease_map_summary.csv")

    pool_all = (
        sum_df[
            (sum_df["disease"] == "ALL_DISEASES")
            & (sum_df["reporting_period"] == "2023-2025_POOL")
        ]
        .set_index("district_name")
        .to_dict(orient="index")
    )
    cw_lookup = cw_df.set_index("district_name").to_dict(orient="index")
    sw_lookup = sw_df.set_index("district_name").to_dict(orient="index")

    districts = list(OFFICIAL_DISTRICTS)
    district_ids = [DISTRICT_ID_MAP[d] for d in districts]
    district_to_idx = {d: i for i, d in enumerate(districts)}
    n = len(districts)

    adj_binary = np.zeros((n, n), dtype=float)
    adj_distance_km = np.zeros((n, n), dtype=float)
    adj_same_division = np.zeros((n, n), dtype=float)

    node_rows: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []

    for i, d in enumerate(districts):
        cw = cw_lookup[d]
        sw = sw_lookup[d]
        pa = pool_all.get(d, {})
        has_coords = bool(cw["has_verified_coordinates"])
        deg = int(sw["degree_symmetric_knn"]) if pd.notna(sw.get("degree_symmetric_knn")) else 0

        obs_wks = int(pa.get("observed_outbreak_weeks", 0))
        obs_cases = float(pa["observed_cases"]) if pd.notna(pa.get("observed_cases")) else np.nan
        obs_mean = (
            float(pa["observed_mean_cases_per_outbreak_week"])
            if pd.notna(pa.get("observed_mean_cases_per_outbreak_week"))
            else np.nan
        )

        if not has_coords:
            elig = "ISOLATED_NO_COORDINATES_NO_PROPAGATION_EDGES"
        elif obs_wks == 0:
            elig = "GEOCODED_NODE_ZERO_OBSERVED_2023_2025_OUTBREAKS"
        else:
            elig = "ELIGIBLE_GEOCODED_SURVEILLANCE_NODE"

        node_rows.append(
            {
                "node_index": i,
                "district_id": DISTRICT_ID_MAP[d],
                "district_name": d,
                "administrative_division": ADMIN_DIVISION_MAP[d],
                "is_coastal": bool(d in COASTAL_DISTRICTS),
                "latitude": float(cw["latitude"]) if has_coords else np.nan,
                "longitude": float(cw["longitude"]) if has_coords else np.nan,
                "has_verified_coordinates": has_coords,
                "graph_degree": deg,
                "observed_2023_2025_outbreak_weeks": obs_wks,
                "observed_2023_2025_total_cases_ref": obs_cases,
                "observed_2023_2025_mean_weekly_cases_ref": obs_mean,
                "population_denominator": np.nan,
                "mobility_data_status": "UNAVAILABLE_IN_WORKSPACE",
                "node_simulation_status": elig,
            }
        )

        # Parse validated neighbors from spatial_weights_matrix.csv
        if has_coords and deg > 0:
            nbr_names = [
                s.strip()
                for s in str(sw["neighbor_district_names"]).split(";")
                if s.strip()
            ]
            nbr_dists = [
                float(s.strip())
                for s in str(sw["neighbor_distances_km"]).split(";")
                if s.strip()
            ]
            nbr_wts = [
                float(s.strip())
                for s in str(sw["row_standardized_weights"]).split(";")
                if s.strip()
            ]

            for nb_name, dist_km, r_wt in zip(nbr_names, nbr_dists, nbr_wts):
                j = district_to_idx[nb_name]
                same_div = ADMIN_DIVISION_MAP[d] == ADMIN_DIVISION_MAP[nb_name]
                adj_binary[i, j] = 1.0
                adj_distance_km[i, j] = dist_km
                adj_same_division[i, j] = 1.0 if same_div else 0.0

                edge_rows.append(
                    {
                        "edge_id": f"{DISTRICT_ID_MAP[d]}->{DISTRICT_ID_MAP[nb_name]}",
                        "source_district_id": DISTRICT_ID_MAP[d],
                        "source_district_name": d,
                        "target_district_id": DISTRICT_ID_MAP[nb_name],
                        "target_district_name": nb_name,
                        "source_division": ADMIN_DIVISION_MAP[d],
                        "target_division": ADMIN_DIVISION_MAP[nb_name],
                        "same_administrative_division": bool(same_div),
                        "great_circle_distance_km": round(dist_km, 2),
                        "topological_row_std_weight": round(r_wt, 4),
                        "inverse_distance_raw": round(1.0 / max(dist_km, 1.0), 6),
                        "edge_basis": "Validated_Symmetric_kNN4_Point_Coordinate_Adjacency",
                        "mobility_data_status": "UNAVAILABLE_GEOGRAPHIC_PROXIMITY_ONLY",
                    }
                )

    nodes_df = pd.DataFrame(node_rows)
    edges_df = pd.DataFrame(edge_rows)

    # Verify symmetry and zero self-loops
    assert np.allclose(adj_binary, adj_binary.T), "District adjacency matrix must be symmetric."
    assert np.allclose(np.diag(adj_binary), 0.0), "Self-loops are not allowed in adjacency matrix."
    assert adj_binary[district_to_idx["Mumbai Suburban"]].sum() == 0.0, (
        "Mumbai Suburban has no coordinates and must have zero invented edges."
    )

    return DistrictGraph(
        nodes_df=nodes_df,
        edges_df=edges_df,
        districts=districts,
        district_ids=district_ids,
        district_to_idx=district_to_idx,
        adj_distance_km=adj_distance_km,
        adj_binary=adj_binary,
        adj_same_division=adj_same_division,
    )


def compute_transition_weights_matrix(
    graph: DistrictGraph,
    edge_weighting_mode: str = "distance_decayed",
    neighbour_influence_decay_km: float = 120.0,
    cross_division_attenuation: float = 0.65,
) -> np.ndarray:
    """Compute row-normalized edge transition weight matrix W_mode (36 x 36) under explicit assumptions.

    Supported `edge_weighting_mode` values:
    - `'distance_decayed'`: w_ij proportional to exp(-d_ij / neighbour_influence_decay_km) for connected (i, j).
      Does NOT assume neighbours have equal movement; closer neighbours receive higher geometric weight.
    - `'inverse_distance'`: w_ij proportional to 1 / d_ij for connected (i, j).
    - `'division_bounded'`: distance-decayed weights multiplied by `cross_division_attenuation` when
      districts i and j belong to different revenue divisions.
    - `'uniform_topological'`: w_ij = 1 / degree(i) (pure topological baseline for sensitivity comparison).
    """
    valid_modes = {
        "distance_decayed",
        "inverse_distance",
        "division_bounded",
        "uniform_topological",
    }
    if edge_weighting_mode not in valid_modes:
        raise ValueError(
            f"Unsupported edge_weighting_mode '{edge_weighting_mode}'. Must be one of {sorted(valid_modes)}."
        )
    if neighbour_influence_decay_km <= 0:
        raise ValueError("neighbour_influence_decay_km must be strictly positive.")
    if not (0.0 <= cross_division_attenuation <= 1.0):
        raise ValueError("cross_division_attenuation must be in [0.0, 1.0].")

    n = len(graph.districts)
    W_raw = np.zeros((n, n), dtype=float)
    mask = graph.adj_binary > 0

    if edge_weighting_mode == "distance_decayed":
        W_raw[mask] = np.exp(-graph.adj_distance_km[mask] / float(neighbour_influence_decay_km))
    elif edge_weighting_mode == "inverse_distance":
        W_raw[mask] = 1.0 / np.maximum(graph.adj_distance_km[mask], 1.0)
    elif edge_weighting_mode == "division_bounded":
        base = np.exp(-graph.adj_distance_km[mask] / float(neighbour_influence_decay_km))
        div_factor = np.where(
            graph.adj_same_division[mask] > 0, 1.0, float(cross_division_attenuation)
        )
        W_raw[mask] = base * div_factor
    elif edge_weighting_mode == "uniform_topological":
        W_raw[mask] = 1.0

    row_sums = W_raw.sum(axis=1, keepdims=True)
    W_norm = np.divide(W_raw, row_sums, out=np.zeros_like(W_raw), where=row_sums > 0)
    return W_norm
