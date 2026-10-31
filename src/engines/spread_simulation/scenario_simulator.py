"""Transparent Graph-Based Scenario Simulator & Parameter Validation (Phase 10).

Implements a deterministic, auditable graph propagation calculator over the 36-district
Maharashtra network without inventing biological SEIR parameters or empirical human mobility flows.

Update Equation at step t -> t+1 (for each geocoded district i):
- Effective intervention multiplier:
    m_int(t) = (1.0 - intervention_response_coverage) if t >= intervention_start_step else 1.0
- Persisted within-district signal:
    P_i(t) = within_district_persistence_factor * m_int(t) * S_i(t)
- Incoming neighbor signal pressure (along incoming edges j -> i):
    I_i(t) = hypothetical_propagation_factor * m_int(t) * sum_j( W_mode[j, i] * S_j(t) )
- Bounded next-step hypothetical signal index (in [0, 100]):
    S_i(t + 1) = min(100.0, P_i(t) + I_i(t) * (1.0 - S_i(t) / 100.0))

Note: `W_mode[j, i]` uses the column of incoming weights from neighbor `j` into `i` normalized
so that a district surrounded by maximally active neighbors (S_j = 100) receives at most
`100 * hypothetical_propagation_factor` units of incoming pressure per step.
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from jeevan_pipeline.standardize import standardize_district
from src.engines.spread_simulation.district_graph import (
    MANDATORY_SCENARIO_DISCLAIMER,
    DistrictGraph,
    compute_transition_weights_matrix,
)


ENGINE_VERSION = "phase10-v1.0.0"


@dataclass
class ScenarioParameters:
    """Explicit, user-visible, validated scenario configuration."""
    scenario_id: str
    scenario_name: str
    scenario_description: str
    initial_affected_districts: Dict[str, float]   # District Name -> Initial Signal (0, 100]
    scenario_duration_steps: int = 10              # Steps in [1, 26] (hypothetical weeks)
    hypothetical_propagation_factor: float = 0.35  # Cross-district transfer factor in [0.0, 1.0]
    within_district_persistence_factor: float = 0.80  # Within-district retention in [0.0, 1.0]
    neighbour_influence_decay_km: float = 120.0    # Distance decay scale in [25.0, 500.0] km
    edge_weighting_mode: str = "distance_decayed"  # 'distance_decayed' | 'inverse_distance' | 'division_bounded' | 'uniform_topological'
    cross_division_attenuation: float = 0.65       # Attenuation across division borders in [0.0, 1.0]
    intervention_start_step: int = 5               # Step in [1, 26] when response starts
    intervention_response_coverage: float = 0.40   # Fractional dampening in [0.0, 1.0]
    activation_threshold: float = 5.0              # Threshold in [1.0, 50.0] for affected sequence logging

    def validate_and_canonicalize(self, graph: DistrictGraph) -> "ScenarioParameters":
        """Validate all parameter boundaries and canonicalize district names."""
        if not self.scenario_id or not str(self.scenario_id).strip():
            raise ValueError("scenario_id must be a non-empty string.")
        if not self.initial_affected_districts:
            raise ValueError("initial_affected_districts cannot be empty.")
        if not (1 <= int(self.scenario_duration_steps) <= 26):
            raise ValueError("scenario_duration_steps must be between 1 and 26.")
        if not (0.0 <= float(self.hypothetical_propagation_factor) <= 1.0):
            raise ValueError("hypothetical_propagation_factor must be in [0.0, 1.0].")
        if not (0.0 <= float(self.within_district_persistence_factor) <= 1.0):
            raise ValueError("within_district_persistence_factor must be in [0.0, 1.0].")
        if not (25.0 <= float(self.neighbour_influence_decay_km) <= 500.0):
            raise ValueError("neighbour_influence_decay_km must be in [25.0, 500.0] km.")
        if not (1 <= int(self.intervention_start_step) <= 26):
            raise ValueError("intervention_start_step must be between 1 and 26.")
        if not (0.0 <= float(self.intervention_response_coverage) <= 1.0):
            raise ValueError("intervention_response_coverage must be in [0.0, 1.0].")
        if not (1.0 <= float(self.activation_threshold) <= 50.0):
            raise ValueError("activation_threshold must be in [1.0, 50.0].")

        canon_seeds: Dict[str, float] = {}
        for raw_d, val in self.initial_affected_districts.items():
            canon_d, _ = standardize_district(raw_d)
            if canon_d is None or canon_d not in graph.district_to_idx:
                raise ValueError(f"Unrecognized Maharashtra district in initial_affected_districts: '{raw_d}'")
            fval = float(val)
            if not (0.0 < fval <= 100.0):
                raise ValueError(
                    f"Initial signal for '{canon_d}' must be in (0.0, 100.0], got {fval}."
                )
            canon_seeds[canon_d] = round(fval, 4)

        self.initial_affected_districts = canon_seeds
        return self


def run_scenario_simulation(
    graph: DistrictGraph,
    params: ScenarioParameters,
    simulation_timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a deterministic multi-step hypothetical propagation scenario over the district graph."""
    params = params.validate_and_canonicalize(graph)
    sim_ts = simulation_timestamp or datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # Compute row-normalized incoming weights W_in[i, j] (weight of neighbor j on target i)
    W_in = compute_transition_weights_matrix(
        graph,
        edge_weighting_mode=params.edge_weighting_mode,
        neighbour_influence_decay_km=params.neighbour_influence_decay_km,
        cross_division_attenuation=params.cross_division_attenuation,
    )

    n = len(graph.districts)
    T = int(params.scenario_duration_steps)
    theta = float(params.activation_threshold)

    # S_hist has shape (T + 1, n)
    S_hist = np.zeros((T + 1, n), dtype=float)
    P_hist = np.zeros((T + 1, n), dtype=float)
    I_hist = np.zeros((T + 1, n), dtype=float)
    dom_nbr_hist: List[List[str]] = [["NONE"] * n for _ in range(T + 1)]

    # Step 0 initialization
    for d_name, init_val in params.initial_affected_districts.items():
        idx = graph.district_to_idx[d_name]
        S_hist[0, idx] = float(init_val)
        P_hist[0, idx] = float(init_val)
        dom_nbr_hist[0][idx] = "INITIAL_SEED"

    first_activation_step: Dict[int, Optional[int]] = {i: None for i in range(n)}
    for i in range(n):
        if S_hist[0, i] >= theta:
            first_activation_step[i] = 0

    # Deterministic step-by-step propagation t = 0 .. T-1 (producing state at step t + 1)
    for t in range(T):
        next_step = t + 1
        m_int = (
            1.0 - float(params.intervention_response_coverage)
            if next_step >= int(params.intervention_start_step)
            else 1.0
        )
        cur_S = S_hist[t]

        for i in range(n):
            p_i = float(params.within_district_persistence_factor) * m_int * cur_S[i]
            # Incoming neighbor contributions
            nbr_contribs = W_in[i] * cur_S
            inc_raw = float(np.sum(nbr_contribs))
            i_i = float(params.hypothetical_propagation_factor) * m_int * inc_raw
            next_s = min(100.0, p_i + i_i * (1.0 - cur_S[i] / 100.0))

            S_hist[t + 1, i] = next_s
            P_hist[t + 1, i] = p_i
            I_hist[t + 1, i] = i_i

            if nbr_contribs.max() > 1e-6:
                best_j = int(np.argmax(nbr_contribs))
                dom_nbr_hist[t + 1][i] = graph.districts[best_j]
            elif cur_S[i] > 1e-6:
                dom_nbr_hist[t + 1][i] = "INTERNAL_PERSISTENCE"
            else:
                dom_nbr_hist[t + 1][i] = "NONE"

            if first_activation_step[i] is None and next_s >= theta:
                first_activation_step[i] = t + 1

    # Build ordered affected district sequence
    wavefront_steps: List[Dict[str, Any]] = []
    sequence_tokens: List[str] = []
    for step_idx in range(T + 1):
        newly_activated = [
            graph.districts[i]
            for i in range(n)
            if first_activation_step[i] == step_idx
        ]
        newly_activated_sorted = sorted(
            newly_activated,
            key=lambda d: (-S_hist[step_idx, graph.district_to_idx[d]], d),
        )
        if newly_activated_sorted:
            sequence_tokens.append(f"Step {step_idx}: " + ", ".join(newly_activated_sorted))
        wavefront_steps.append(
            {
                "simulation_step": step_idx,
                "newly_activated_districts": newly_activated_sorted,
                "active_districts_count": int(np.sum(S_hist[step_idx] >= theta)),
                "mean_simulated_signal": round(float(np.mean(S_hist[step_idx])), 3),
                "max_simulated_signal": round(float(np.max(S_hist[step_idx])), 3),
                "sum_simulated_signal": round(float(np.sum(S_hist[step_idx])), 2),
                "intervention_active": bool(step_idx >= int(params.intervention_start_step)),
            }
        )

    affected_sequence_str = " -> ".join(sequence_tokens) if sequence_tokens else "No districts reached activation threshold"

    # Build detailed step x district records DataFrame
    node_meta = graph.nodes_df.set_index("district_name").to_dict(orient="index")
    step_rows: List[Dict[str, Any]] = []

    init_districts_str = "; ".join(
        f"{d} ({v:.1f})" for d, v in params.initial_affected_districts.items()
    )

    assumptions_text = (
        f"1. Graph Adjacency: Validated 35-district k-NN (k=4) Haversine graph; Mumbai Suburban is isolated (no coordinates). "
        f"2. Edge Weighting Mode: '{params.edge_weighting_mode}' (decay={params.neighbour_influence_decay_km} km, cross_div_atten={params.cross_division_attenuation}) — represents geometric proximity, NOT empirical human mobility. "
        f"3. Propagation & Persistence: beta_prop={params.hypothetical_propagation_factor}, rho_persist={params.within_district_persistence_factor}. "
        f"4. Intervention: Starts at step {params.intervention_start_step} with response coverage eta_cov={params.intervention_response_coverage:.0%}."
    )
    limitations_text = (
        "1. Not an outbreak forecast or biological SEIR prediction. "
        "2. No commuter flow, road-network traffic, or population denominator data exist in the workspace. "
        "3. Simulated signal index (0-100) is a dimensionless hypothetical stress indicator and must never be mixed with observed IDSP case counts."
    )

    for t in range(T + 1):
        int_active = bool(t >= int(params.intervention_start_step))
        for i, d_name in enumerate(graph.districts):
            nm = node_meta[d_name]
            step_rows.append(
                {
                    "scenario_id": params.scenario_id,
                    "scenario_name": params.scenario_name,
                    "initial_district": init_districts_str,
                    "simulation_step": t,
                    "district_id": graph.district_ids[i],
                    "district_name": d_name,
                    "administrative_division": nm["administrative_division"],
                    "latitude": nm["latitude"],
                    "longitude": nm["longitude"],
                    "node_simulation_status": nm["node_simulation_status"],
                    # Simulated indicator values (strictly labelled as hypothetical simulation)
                    "simulated_signal_index": round(float(S_hist[t, i]), 4),
                    "simulated_persisted_signal": round(float(P_hist[t, i]), 4),
                    "simulated_incoming_neighbor_signal": round(float(I_hist[t, i]), 4),
                    "is_affected_above_threshold": bool(S_hist[t, i] >= theta),
                    "first_activation_step": (
                        int(first_activation_step[i])
                        if first_activation_step[i] is not None
                        else np.nan
                    ),
                    "dominant_incoming_source": dom_nbr_hist[t][i],
                    "intervention_active_at_step": int_active,
                    # Historical observed reference kept strictly separate for context only
                    "observed_2023_2025_mean_weekly_cases_ref": nm["observed_2023_2025_mean_weekly_cases_ref"],
                    "record_nature": "HYPOTHETICAL_SCENARIO_SIMULATION_NOT_OBSERVED_CASES",
                    "model_version": ENGINE_VERSION,
                    "simulation_timestamp": sim_ts,
                }
            )

    step_df = pd.DataFrame(step_rows)

    total_ever_affected = sum(1 for v in first_activation_step.values() if v is not None)
    peak_step_idx = int(np.argmax(S_hist.sum(axis=1)))

    summary_record = {
        "scenario_id": params.scenario_id,
        "scenario_name": params.scenario_name,
        "scenario_description": params.scenario_description,
        "initial_district": init_districts_str,
        "scenario_parameters": asdict(params),
        "affected_district_sequence": affected_sequence_str,
        "wavefront_timeline": wavefront_steps,
        "total_districts_ever_affected": int(total_ever_affected),
        "initial_step0_signal_sum": round(float(S_hist[0].sum()), 2),
        "peak_step": peak_step_idx,
        "peak_statewide_signal_sum": round(float(S_hist[peak_step_idx].sum()), 2),
        "final_step_signal_sum": round(float(S_hist[T].sum()), 2),
        "final_step_active_districts": int(np.sum(S_hist[T] >= theta)),
        "assumptions": assumptions_text,
        "scenario_limitations": limitations_text,
        "non_prediction_disclaimer": MANDATORY_SCENARIO_DISCLAIMER,
        "model_version": ENGINE_VERSION,
        "simulation_timestamp": sim_ts,
    }

    return {
        "summary": summary_record,
        "step_records_df": step_df,
        "S_matrix": S_hist,
    }


def build_default_benchmark_scenarios() -> List[ScenarioParameters]:
    """Define 6 benchmark hypothetical scenarios covering all required Prompt 10 cases."""
    return [
        ScenarioParameters(
            scenario_id="SCEN_01_EAST_VIDARBHA_BASELINE",
            scenario_name="Eastern Vidarbha Single-District Elevation (Delayed Intervention)",
            scenario_description=(
                "Hypothetical elevated signal seeded in Gadchiroli (85.0) under distance-decayed "
                "geographic adjacency with delayed response starting at step 8 (30% coverage)."
            ),
            initial_affected_districts={"Gadchiroli": 85.0},
            scenario_duration_steps=12,
            hypothetical_propagation_factor=0.38,
            within_district_persistence_factor=0.82,
            neighbour_influence_decay_km=120.0,
            edge_weighting_mode="distance_decayed",
            intervention_start_step=8,
            intervention_response_coverage=0.30,
            activation_threshold=5.0,
        ),
        ScenarioParameters(
            scenario_id="SCEN_02_EAST_VIDARBHA_EARLY_INTERVENTION",
            scenario_name="Eastern Vidarbha Early Intervention & High Response Coverage (Paired Comparison)",
            scenario_description=(
                "Identical initial seed in Gadchiroli (85.0) and propagation factor (0.38) as SCEN_01, "
                "but with early intervention at step 2 and 70% response coverage."
            ),
            initial_affected_districts={"Gadchiroli": 85.0},
            scenario_duration_steps=12,
            hypothetical_propagation_factor=0.38,
            within_district_persistence_factor=0.82,
            neighbour_influence_decay_km=120.0,
            edge_weighting_mode="distance_decayed",
            intervention_start_step=2,
            intervention_response_coverage=0.70,
            activation_threshold=5.0,
        ),
        ScenarioParameters(
            scenario_id="SCEN_03_KONKAN_MULTI_NODE_ELEVATION",
            scenario_name="Northern Konkan Multi-District Simultaneous Signal Elevation",
            scenario_description=(
                "Simultaneous elevated signals in neighbouring coastal districts Thane (75.0), "
                "Raigad (70.0), and Palghar (65.0) with moderate intervention at step 5 (50% coverage)."
            ),
            initial_affected_districts={"Thane": 75.0, "Raigad": 70.0, "Palghar": 65.0},
            scenario_duration_steps=12,
            hypothetical_propagation_factor=0.40,
            within_district_persistence_factor=0.80,
            neighbour_influence_decay_km=100.0,
            edge_weighting_mode="distance_decayed",
            intervention_start_step=5,
            intervention_response_coverage=0.50,
            activation_threshold=5.0,
        ),
        ScenarioParameters(
            scenario_id="SCEN_04_PUNE_KOLHAPUR_DIVISION_BOUNDED",
            scenario_name="Western Maharashtra Corridor (Division-Bounded Adjacency Assumption)",
            scenario_description=(
                "Dual seed in Kolhapur (85.0) and Pune (75.0) using division-bounded edge weights "
                "(cross-division attenuation = 0.50) and intervention at step 6 (45% coverage)."
            ),
            initial_affected_districts={"Kolhapur": 85.0, "Pune": 75.0},
            scenario_duration_steps=12,
            hypothetical_propagation_factor=0.36,
            within_district_persistence_factor=0.80,
            neighbour_influence_decay_km=110.0,
            edge_weighting_mode="division_bounded",
            cross_division_attenuation=0.50,
            intervention_start_step=6,
            intervention_response_coverage=0.45,
            activation_threshold=5.0,
        ),
        ScenarioParameters(
            scenario_id="SCEN_05_MARATHWADA_HIGH_PROPAGATION_UNMITIGATED",
            scenario_name="Marathwada High-Propagation Stress Scenario (No Intervention Baseline)",
            scenario_description=(
                "Hypothetical stress-test seeded in Nanded (90.0) with high propagation factor (0.55), "
                "inverse-distance edge weighting, and 0% intervention coverage over 12 steps."
            ),
            initial_affected_districts={"Nanded": 90.0},
            scenario_duration_steps=12,
            hypothetical_propagation_factor=0.55,
            within_district_persistence_factor=0.85,
            neighbour_influence_decay_km=140.0,
            edge_weighting_mode="inverse_distance",
            intervention_start_step=12,
            intervention_response_coverage=0.0,
            activation_threshold=5.0,
        ),
        ScenarioParameters(
            scenario_id="SCEN_06_MARATHWADA_RAPID_CONTAINMENT",
            scenario_name="Marathwada High-Propagation Stress Scenario (Rapid Containment at Step 3)",
            scenario_description=(
                "Paired comparison to SCEN_05: identical Nanded (90.0) seed and high propagation (0.55), "
                "but with rapid containment starting at step 3 (65% response coverage)."
            ),
            initial_affected_districts={"Nanded": 90.0},
            scenario_duration_steps=12,
            hypothetical_propagation_factor=0.55,
            within_district_persistence_factor=0.85,
            neighbour_influence_decay_km=140.0,
            edge_weighting_mode="inverse_distance",
            intervention_start_step=3,
            intervention_response_coverage=0.65,
            activation_threshold=5.0,
        ),
    ]


def run_parameter_sensitivity_grid(graph: DistrictGraph) -> pd.DataFrame:
    """Evaluate a structured sensitivity grid over propagation factor, intervention timing,
    response coverage, and edge weighting mode (seeded in Pune = 80.0)."""
    rows: List[Dict[str, Any]] = []
    for mode in ["distance_decayed", "inverse_distance", "division_bounded", "uniform_topological"]:
        for beta in [0.20, 0.35, 0.50]:
            for t_int in [2, 5, 9]:
                for eta in [0.0, 0.35, 0.70]:
                    p = ScenarioParameters(
                        scenario_id=f"SENS_{mode[:4].upper()}_B{int(beta*100)}_T{t_int}_E{int(eta*100)}",
                        scenario_name="Sensitivity Grid Run",
                        scenario_description="Parameter sensitivity sweep",
                        initial_affected_districts={"Pune": 80.0},
                        scenario_duration_steps=10,
                        hypothetical_propagation_factor=beta,
                        within_district_persistence_factor=0.80,
                        neighbour_influence_decay_km=120.0,
                        edge_weighting_mode=mode,
                        intervention_start_step=t_int,
                        intervention_response_coverage=eta,
                        activation_threshold=5.0,
                    )
                    res = run_scenario_simulation(graph, p)
                    s = res["summary"]
                    rows.append(
                        {
                            "seed_district": "Pune (80.0)",
                            "edge_weighting_mode": mode,
                            "hypothetical_propagation_factor": beta,
                            "intervention_start_step": t_int,
                            "intervention_response_coverage": eta,
                            "total_districts_ever_affected": s["total_districts_ever_affected"],
                            "peak_statewide_signal_sum": s["peak_statewide_signal_sum"],
                            "final_step10_signal_sum": s["final_step_signal_sum"],
                            "final_step10_active_districts": s["final_step_active_districts"],
                            "non_prediction_disclaimer": MANDATORY_SCENARIO_DISCLAIMER,
                        }
                    )
    return pd.DataFrame(rows)
