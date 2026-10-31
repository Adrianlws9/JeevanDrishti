"""Spatial Weights Matrix and Hotspot Statistical Methods (Phase 9).

Implements:
1. Great-circle Haversine distance matrix across the 35 geocoded Maharashtra districts.
2. Symmetric k-Nearest Neighbor (k=4, row-standardized) spatial weights matrix W_knn4
   and Inverse Distance Band (d <= 220 km, row-standardized) sensitivity matrix W_idw.
3. Global Moran's I with analytical variance + 999 Monte Carlo spatial permutations.
4. Local Moran's I (Anselin 1995 LISA) with 999 conditional permutations per district.
5. Getis-Ord Gi* (Ord & Getis 1995) using self-inclusive binary spatial weights w_ij*
   with analytical normal p-values and 999 permutation p-values.
6. Graph-connected spatial cluster identification for elevated multi-district corridors.

Strict Scientific Rules:
- Never label a district as a statistically significant hotspot unless the statistical test
  supports that claim (p < 0.05).
- Never interpret geographic proximity or spatial autocorrelation as proof of direct
  inter-district disease transmission.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


EARTH_RADIUS_KM = 6371.0088

NON_TRANSMISSION_DISCLAIMER = (
    "Spatial autocorrelation or hotspot significance indicates geographic clustering of reported "
    "surveillance counts; it does NOT prove cross-district pathogen transmission or causation."
)


def haversine_distance_matrix_km(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Compute pairwise Haversine distance matrix in kilometers for N (lat, lon) points."""
    lat_rad = np.radians(np.asarray(lats, dtype=float))
    lon_rad = np.radians(np.asarray(lons, dtype=float))
    dlat = lat_rad[:, None] - lat_rad[None, :]
    dlon = lon_rad[:, None] - lon_rad[None, :]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_rad[:, None]) * np.cos(lat_rad[None, :]) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))
    dist = EARTH_RADIUS_KM * c
    np.fill_diagonal(dist, 0.0)
    return dist


@dataclass
class SpatialWeightsBundle:
    """Container for the 35-district spatial weights matrices and metadata."""
    districts: List[str]
    district_ids: List[str]
    lats: np.ndarray
    lons: np.ndarray
    dist_km: np.ndarray
    W_binary_sym: np.ndarray       # Symmetric k-NN binary adjacency (zero diagonal)
    W_row_std: np.ndarray          # Row-standardized k-NN weights (zero diagonal, rows sum to 1)
    W_self_binary: np.ndarray      # Self-inclusive binary weights w_ij* for Getis-Ord Gi* (diagonal = 1)
    W_idw_row_std: np.ndarray      # Inverse-distance band (<=220 km) row-standardized weights
    k_neighbors: int
    idw_bandwidth_km: float


def build_spatial_weights(
    crosswalk_df: pd.DataFrame, k_neighbors: int = 4, idw_bandwidth_km: float = 220.0
) -> Tuple[SpatialWeightsBundle, pd.DataFrame, Dict[str, Any]]:
    """Construct spatial weights matrices over all geocoded Maharashtra districts (35 / 36).

    `Mumbai Suburban` is excluded from spatial weights because it has no verified coordinates.
    """
    geo_df = (
        crosswalk_df[crosswalk_df["has_verified_coordinates"]]
        .sort_values("district_name")
        .reset_index(drop=True)
    )
    districts = geo_df["district_name"].tolist()
    district_ids = geo_df["district_id"].tolist()
    lats = geo_df["latitude"].to_numpy(dtype=float)
    lons = geo_df["longitude"].to_numpy(dtype=float)
    n = len(districts)

    dist_km = haversine_distance_matrix_km(lats, lons)

    # 1. Symmetric k-NN binary adjacency (zero diagonal)
    W_knn_dir = np.zeros((n, n), dtype=float)
    for i in range(n):
        d_row = dist_km[i].copy()
        d_row[i] = np.inf
        nn_idx = np.argsort(d_row)[:k_neighbors]
        W_knn_dir[i, nn_idx] = 1.0

    W_binary_sym = np.maximum(W_knn_dir, W_knn_dir.T)
    np.fill_diagonal(W_binary_sym, 0.0)

    # Row-standardized W_knn4
    row_sums = W_binary_sym.sum(axis=1, keepdims=True)
    W_row_std = np.divide(W_binary_sym, row_sums, out=np.zeros_like(W_binary_sym), where=row_sums > 0)

    # Self-inclusive binary weights for Ord & Getis (1995) Gi*
    W_self_binary = W_binary_sym.copy()
    np.fill_diagonal(W_self_binary, 1.0)

    # 2. Secondary Inverse-Distance Band (d <= 220 km) row-standardized matrix
    W_idw_raw = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j and 0.0 < dist_km[i, j] <= idw_bandwidth_km:
                W_idw_raw[i, j] = 1.0 / dist_km[i, j]
        # Ensure no island if nearest neighbor > bandwidth
        if W_idw_raw[i].sum() == 0:
            d_row = dist_km[i].copy()
            d_row[i] = np.inf
            nn_idx = np.argsort(d_row)[:k_neighbors]
            W_idw_raw[i, nn_idx] = 1.0 / dist_km[i, nn_idx]
    idw_sums = W_idw_raw.sum(axis=1, keepdims=True)
    W_idw_row_std = np.divide(W_idw_raw, idw_sums, out=np.zeros_like(W_idw_raw), where=idw_sums > 0)

    # Build neighbor table for export
    edge_rows = []
    for i in range(n):
        nbr_indices = np.where(W_binary_sym[i] > 0)[0]
        nbr_indices = sorted(nbr_indices, key=lambda j: dist_km[i, j])
        nbr_names = [districts[j] for j in nbr_indices]
        nbr_ids = [district_ids[j] for j in nbr_indices]
        nbr_dists = [round(float(dist_km[i, j]), 2) for j in nbr_indices]
        nbr_wts = [round(float(W_row_std[i, j]), 4) for j in nbr_indices]
        edge_rows.append(
            {
                "district_id": district_ids[i],
                "district_name": districts[i],
                "latitude": round(float(lats[i]), 6),
                "longitude": round(float(lons[i]), 6),
                "degree_symmetric_knn": len(nbr_indices),
                "row_weight_sum": round(float(W_row_std[i].sum()), 6),
                "nearest_neighbor_distance_km": nbr_dists[0] if nbr_dists else np.nan,
                "max_knn_neighbor_distance_km": max(nbr_dists) if nbr_dists else np.nan,
                "neighbor_district_ids": "; ".join(nbr_ids),
                "neighbor_district_names": "; ".join(nbr_names),
                "neighbor_distances_km": "; ".join(str(d) for d in nbr_dists),
                "row_standardized_weights": "; ".join(str(w) for w in nbr_wts),
            }
        )

    # Add Mumbai Suburban explicitly as excluded / insufficient coordinates
    ms_row = crosswalk_df[crosswalk_df["district_name"] == "Mumbai Suburban"]
    if not ms_row.empty:
        edge_rows.append(
            {
                "district_id": str(ms_row.iloc[0]["district_id"]),
                "district_name": "Mumbai Suburban",
                "latitude": np.nan,
                "longitude": np.nan,
                "degree_symmetric_knn": 0,
                "row_weight_sum": 0.0,
                "nearest_neighbor_distance_km": np.nan,
                "max_knn_neighbor_distance_km": np.nan,
                "neighbor_district_ids": "INSUFFICIENT_DATA_NO_COORDINATES",
                "neighbor_district_names": "Insufficient Data (No Coordinates in Workspace)",
                "neighbor_distances_km": "",
                "row_standardized_weights": "",
            }
        )

    weights_df = pd.DataFrame(edge_rows)
    degrees = W_binary_sym.sum(axis=1)
    metadata = {
        "method_primary": "Symmetric k-Nearest Neighbors (k=4) on Great-Circle Haversine Distance, Row-Standardized",
        "method_secondary_sensitivity": f"Inverse Haversine Distance Band (<= {idw_bandwidth_km} km), Row-Standardized",
        "why_point_based_weights": (
            "No district polygon boundary file (.geojson/.shp) exists in the workspace. "
            "Symmetric k-NN (k=4) on verified district point coordinates guarantees a connected, "
            "island-free spatial graph across all 35 geocoded districts without inventing polygon boundaries."
        ),
        "n_total_districts": 36,
        "n_geocoded_in_W": n,
        "excluded_no_coordinates": ["Mumbai Suburban"],
        "k_nearest_parameter": k_neighbors,
        "min_neighbors_after_symmetrization": int(degrees.min()),
        "max_neighbors_after_symmetrization": int(degrees.max()),
        "mean_neighbors_after_symmetrization": round(float(degrees.mean()), 2),
        "zero_islands_verified": bool((degrees > 0).all()),
        "row_standardization_verified": bool(np.allclose(W_row_std.sum(axis=1), 1.0)),
    }

    bundle = SpatialWeightsBundle(
        districts=districts,
        district_ids=district_ids,
        lats=lats,
        lons=lons,
        dist_km=dist_km,
        W_binary_sym=W_binary_sym,
        W_row_std=W_row_std,
        W_self_binary=W_self_binary,
        W_idw_row_std=W_idw_row_std,
        k_neighbors=k_neighbors,
        idw_bandwidth_km=idw_bandwidth_km,
    )
    return bundle, weights_df, metadata


def compute_global_morans_i(
    values: np.ndarray,
    W: np.ndarray,
    n_permutations: int = 999,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Compute Global Moran's I with analytical randomization variance and Monte Carlo permutation test."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    s0 = float(W.sum())
    if n < 5 or s0 <= 0 or np.nanstd(x) < 1e-12:
        return {
            "morans_i": np.nan,
            "expected_i": -1.0 / max(n - 1, 1),
            "variance_randomization": np.nan,
            "z_score_analytical": np.nan,
            "p_value_analytical": np.nan,
            "p_value_permutation": np.nan,
            "is_significant_p05": False,
            "spatial_pattern": "INSUFFICIENT_VARIANCE_OR_DATA",
        }

    x_mean = float(np.mean(x))
    dev = x - x_mean
    denom = float(np.sum(dev ** 2))
    num = float(dev @ W @ dev)
    moran_i = (n / s0) * (num / denom)
    expected_i = -1.0 / (n - 1)

    # Analytical randomization variance (Cliff & Ord 1981)
    s1 = 0.5 * float(np.sum((W + W.T) ** 2))
    s2 = float(np.sum((W.sum(axis=1) + W.sum(axis=0)) ** 2))
    m2 = denom / n
    m4 = float(np.sum(dev ** 4)) / n
    b2 = m4 / (m2 ** 2) if m2 > 0 else 3.0

    num_var = (
        n * ((n ** 2 - 3 * n + 3) * s1 - n * s2 + 3 * (s0 ** 2))
        - b2 * ((n ** 2 - n) * s1 - 2 * n * s2 + 6 * (s0 ** 2))
    )
    den_var = (n - 1) * (n - 2) * (n - 3) * (s0 ** 2)
    var_rand = max((num_var / den_var) - (expected_i ** 2), 1e-12)
    z_analytical = (moran_i - expected_i) / np.sqrt(var_rand)
    p_analytical = float(2.0 * (1.0 - stats.norm.cdf(abs(z_analytical))))

    # Monte Carlo spatial permutation test
    rng = np.random.default_rng(random_state)
    sim_i = np.empty(n_permutations, dtype=float)
    for p in range(n_permutations):
        perm_dev = rng.permutation(dev)
        sim_i[p] = (n / s0) * (float(perm_dev @ W @ perm_dev) / denom)

    extreme_count = int(np.sum(np.abs(sim_i - expected_i) >= abs(moran_i - expected_i)))
    p_perm = float((extreme_count + 1) / (n_permutations + 1))

    is_sig = bool(p_perm < 0.05 and p_analytical < 0.05)
    if is_sig and moran_i > expected_i:
        pattern = "SIGNIFICANT_POSITIVE_SPATIAL_CLUSTERING"
    elif is_sig and moran_i < expected_i:
        pattern = "SIGNIFICANT_SPATIAL_DISPERSION"
    else:
        pattern = "RANDOM_OR_NOT_STATISTICALLY_SIGNIFICANT"

    return {
        "morans_i": round(float(moran_i), 4),
        "expected_i": round(float(expected_i), 4),
        "variance_randomization": round(float(var_rand), 6),
        "z_score_analytical": round(float(z_analytical), 4),
        "p_value_analytical": round(float(p_analytical), 4),
        "p_value_permutation": round(float(p_perm), 4),
        "is_significant_p05": is_sig,
        "spatial_pattern": pattern,
    }


def compute_local_morans_lisa_and_gi_star(
    values: np.ndarray,
    weights_bundle: SpatialWeightsBundle,
    n_permutations: int = 999,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute Local Moran's I_i (LISA) and Getis-Ord Gi* for each of the 35 geocoded districts.

    Enforces strict significance gating:
    - A district is labelled `HH_HOTSPOT` in LISA only if z_i > 0, spatial_lag > 0, x_i > 0,
      and conditional permutation p_value < 0.05.
    - A district is labelled `HOTSPOT_95_CONFIDENCE` or `HOTSPOT_99_CONFIDENCE` in Gi* only if
      Gi* >= 1.96, analytical p < 0.05, permutation p < 0.05, and x_i > 0.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    W_std = weights_bundle.W_row_std
    W_star = weights_bundle.W_self_binary

    x_mean = float(np.mean(x))
    x_std_pop = float(np.sqrt(np.mean((x - x_mean) ** 2)))  # Population std S for Gi*
    x_std_sample = float(np.std(x, ddof=1)) if n > 1 else x_std_pop

    if x_std_pop < 1e-12:
        # Constant vector -> no hotspots possible
        rows = []
        for i in range(n):
            rows.append(
                {
                    "district_id": weights_bundle.district_ids[i],
                    "district_name": weights_bundle.districts[i],
                    "value": float(x[i]),
                    "z_value": 0.0,
                    "spatial_lag_z": 0.0,
                    "local_moran_i": 0.0,
                    "local_moran_z": 0.0,
                    "local_moran_p_perm": 1.0,
                    "lisa_quadrant": "NOT_SIGNIFICANT",
                    "gi_star_zscore": 0.0,
                    "gi_star_p_analytical": 1.0,
                    "gi_star_p_perm": 1.0,
                    "gi_star_class": "NOT_SIGNIFICANT",
                    "is_statistically_significant_hotspot": False,
                    "hotspot_consensus_label": "Not a Statistically Significant Hotspot",
                }
            )
        return pd.DataFrame(rows)

    z = (x - x_mean) / x_std_sample
    spatial_lag_z = W_std @ z
    local_i = z * spatial_lag_z

    # Getis-Ord Gi* exact formula (Ord & Getis 1995)
    gi_star = np.zeros(n, dtype=float)
    for i in range(n):
        w_i = W_star[i]
        sum_w = float(np.sum(w_i))
        sum_w2 = float(np.sum(w_i ** 2))
        num = float(np.sum(w_i * x)) - x_mean * sum_w
        rad = max((n * sum_w2 - (sum_w ** 2)) / max(n - 1, 1), 1e-12)
        denom = x_std_pop * np.sqrt(rad)
        gi_star[i] = num / denom if denom > 0 else 0.0

    gi_p_analytical = 2.0 * (1.0 - stats.norm.cdf(np.abs(gi_star)))

    # Conditional permutation tests for Local Moran's I_i and Getis-Ord Gi*
    rng = np.random.default_rng(random_state)
    local_i_p_perm = np.ones(n, dtype=float)
    local_i_z_perm = np.zeros(n, dtype=float)
    gi_p_perm = np.ones(n, dtype=float)

    for i in range(n):
        other_idx = np.array([j for j in range(n) if j != i], dtype=int)
        other_z = z[other_idx]
        other_x = x[other_idx]
        w_std_other = W_std[i, other_idx]
        w_star_other = W_star[i, other_idx]
        nbr_positions = np.where(w_std_other > 0)[0]
        k_i = len(nbr_positions)

        if k_i == 0:
            continue

        sim_lags = np.empty(n_permutations, dtype=float)
        sim_gi_num = np.empty(n_permutations, dtype=float)
        sum_w_star = float(np.sum(W_star[i]))
        m_star = int(round(sum_w_star))
        sum_w2_star = float(np.sum(W_star[i] ** 2))
        gi_denom = x_std_pop * np.sqrt(max((n * sum_w2_star - (sum_w_star ** 2)) / max(n - 1, 1), 1e-12))

        for p in range(n_permutations):
            sampled_pos = rng.choice(len(other_idx), size=k_i, replace=False)
            # Local Moran conditional permutation (Anselin 1995: holds z_i fixed, permutes k_i neighbors)
            sim_lags[p] = float(np.sum(w_std_other[nbr_positions] * other_z[sampled_pos]))
            # Getis-Ord Gi* CSR permutation (Ord & Getis 1995: samples m_star = k_i + 1 districts from N)
            sampled_all = rng.choice(n, size=m_star, replace=False)
            sim_sum_x = float(np.sum(x[sampled_all]))
            sim_gi_num[p] = (sim_sum_x - x_mean * sum_w_star) / gi_denom if gi_denom > 0 else 0.0

        sim_ii = z[i] * sim_lags
        mean_sim_ii = float(np.mean(sim_ii))
        std_sim_ii = float(np.std(sim_ii))
        local_i_z_perm[i] = (local_i[i] - mean_sim_ii) / std_sim_ii if std_sim_ii > 1e-12 else 0.0
        ext_ii = int(np.sum(np.abs(sim_ii - mean_sim_ii) >= abs(local_i[i] - mean_sim_ii)))
        local_i_p_perm[i] = float((ext_ii + 1) / (n_permutations + 1))

        ext_gi = int(np.sum(np.abs(sim_gi_num) >= abs(gi_star[i])))
        gi_p_perm[i] = float((ext_gi + 1) / (n_permutations + 1))

    rows = []
    for i in range(n):
        zi = float(z[i])
        lag_zi = float(spatial_lag_z[i])
        p_lisa = float(local_i_p_perm[i])
        g_z = float(gi_star[i])
        p_gi_a = float(gi_p_analytical[i])
        p_gi_p = float(gi_p_perm[i])

        # LISA quadrant (strictly gated at p_perm < 0.05)
        if p_lisa < 0.05 and x[i] > 0 and zi > 0 and lag_zi > 0:
            lisa_q = "HH_HOTSPOT"
        elif p_lisa < 0.05 and zi < 0 and lag_zi < 0:
            lisa_q = "LL_COLDSPOT"
        elif p_lisa < 0.05 and x[i] > 0 and zi > 0 and lag_zi <= 0:
            lisa_q = "HL_SPATIAL_OUTLIER"
        elif p_lisa < 0.05 and zi <= 0 and lag_zi > 0:
            lisa_q = "LH_SPATIAL_OUTLIER"
        else:
            lisa_q = "NOT_SIGNIFICANT"

        # Getis-Ord Gi* classification (strictly gated at p_analytical < 0.05, p_perm < 0.05, and z_i > 0 for hotspots)
        if x[i] > 0 and zi > 0 and g_z >= 2.5758 and p_gi_a < 0.01 and p_gi_p < 0.05:
            gi_cls = "HOTSPOT_99_CONFIDENCE"
        elif x[i] > 0 and zi > 0 and g_z >= 1.9600 and p_gi_a < 0.05 and p_gi_p < 0.05:
            gi_cls = "HOTSPOT_95_CONFIDENCE"
        elif zi <= 0 and g_z >= 1.9600 and p_gi_a < 0.05:
            gi_cls = "LOW_BURDEN_ADJACENT_TO_HOTSPOT_NOT_HOTSPOT"
        elif x[i] > 0 and zi > 0 and g_z >= 1.6449 and p_gi_a < 0.10:
            gi_cls = "ELEVATED_WATCH_90_MARGINAL_NOT_HOTSPOT"
        elif g_z <= -2.5758 and p_gi_a < 0.01 and p_gi_p < 0.05:
            gi_cls = "COLDSPOT_99_CONFIDENCE"
        elif g_z <= -1.9600 and p_gi_a < 0.05 and p_gi_p < 0.05:
            gi_cls = "COLDSPOT_95_CONFIDENCE"
        else:
            gi_cls = "NOT_SIGNIFICANT"

        is_sig_hotspot = bool(
            gi_cls in ("HOTSPOT_99_CONFIDENCE", "HOTSPOT_95_CONFIDENCE")
            or lisa_q == "HH_HOTSPOT"
        )

        if gi_cls == "HOTSPOT_99_CONFIDENCE" and lisa_q == "HH_HOTSPOT":
            consensus = "Statistically Significant Hotspot (99% Gi* + LISA High-High, p<0.01)"
        elif is_sig_hotspot:
            consensus = f"Statistically Significant Hotspot ({gi_cls} / LISA={lisa_q}, p<0.05)"
        elif lisa_q == "HL_SPATIAL_OUTLIER":
            consensus = "Isolated High-Burden Spatial Outlier (LISA High-Low, p<0.05; Not a Contiguous Hotspot)"
        elif gi_cls == "ELEVATED_WATCH_90_MARGINAL_NOT_HOTSPOT":
            consensus = "Marginal Spatial Elevation (90% Gi* Watch Only; Not Significant at alpha=0.05)"
        elif "COLDSPOT" in gi_cls or lisa_q == "LL_COLDSPOT":
            consensus = f"Statistically Significant Coldspot ({gi_cls} / LISA={lisa_q}, p<0.05)"
        else:
            consensus = "Not a Statistically Significant Hotspot (p >= 0.05)"

        rows.append(
            {
                "district_id": weights_bundle.district_ids[i],
                "district_name": weights_bundle.districts[i],
                "value": round(float(x[i]), 4),
                "z_value": round(zi, 4),
                "spatial_lag_z": round(lag_zi, 4),
                "local_moran_i": round(float(local_i[i]), 4),
                "local_moran_z": round(float(local_i_z_perm[i]), 4),
                "local_moran_p_perm": round(p_lisa, 4),
                "lisa_quadrant": lisa_q,
                "gi_star_zscore": round(g_z, 4),
                "gi_star_p_analytical": round(p_gi_a, 4),
                "gi_star_p_perm": round(p_gi_p, 4),
                "gi_star_class": gi_cls,
                "is_statistically_significant_hotspot": is_sig_hotspot,
                "hotspot_consensus_label": consensus,
            }
        )

    return pd.DataFrame(rows)


def identify_connected_spatial_clusters(
    lisa_gi_df: pd.DataFrame, weights_bundle: SpatialWeightsBundle
) -> List[Dict[str, Any]]:
    """Identify connected multi-district spatial clusters on the k-NN graph where z_value > 0.25 or Gi* >= 1.28."""
    idx_map = {d: i for i, d in enumerate(weights_bundle.districts)}
    elevated_mask = (lisa_gi_df["z_value"] > 0.25) | (lisa_gi_df["gi_star_zscore"] >= 1.28)
    elevated_districts = set(lisa_gi_df.loc[elevated_mask, "district_name"].tolist())

    visited = set()
    clusters = []
    cluster_idx = 1

    for d in sorted(elevated_districts):
        if d in visited:
            continue
        # BFS over W_binary_sym restricted to elevated_districts
        comp = []
        queue = [d]
        visited.add(d)
        while queue:
            curr = queue.pop(0)
            comp.append(curr)
            ci = idx_map[curr]
            nbr_indices = np.where(weights_bundle.W_binary_sym[ci] > 0)[0]
            for ni in nbr_indices:
                nd = weights_bundle.districts[ni]
                if nd in elevated_districts and nd not in visited:
                    visited.add(nd)
                    queue.append(nd)

        sub = lisa_gi_df[lisa_gi_df["district_name"].isin(comp)]
        n_sig = int(sub["is_statistically_significant_hotspot"].sum())
        clusters.append(
            {
                "cluster_id": f"SPATIAL_CLUSTER_{cluster_idx:02d}",
                "member_districts_count": len(comp),
                "member_districts": "; ".join(sorted(comp)),
                "statistically_significant_hotspot_members": n_sig,
                "mean_value": round(float(sub["value"].mean()), 2),
                "max_gi_star_zscore": round(float(sub["gi_star_zscore"].max()), 3),
                "contains_verified_hotspot_p05": bool(n_sig > 0),
                "scientific_note": NON_TRANSMISSION_DISCLAIMER,
            }
        )
        cluster_idx += 1

    return clusters
