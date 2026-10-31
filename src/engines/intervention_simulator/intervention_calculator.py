"""Transparent Public-Health Intervention Scenario Calculator (Phase 14 Sections 2–6).

Implements:
- Section 2: Configurable scenario types:
    1. INCREASED_SURVEILLANCE_AND_TESTING
    2. ADDITIONAL_VECTOR_CONTROL_ACTIVITY
    3. PUBLIC_AWARENESS_CAMPAIGN
    4. INCREASED_HEALTHCARE_RESOURCE_ALLOCATION
    5. EARLY_INTERVENTION_VS_DELAYED_INTERVENTION
    6. DISTRICT_FOCUSED_RESPONSE
    7. MULTI_DISTRICT_RESPONSE
    8. BASELINE_NO_ADDITIONAL_INTERVENTION
- Section 3: Transparent Scenario Calculator (no empirical RCT dataset exists; never claims
  fixed empirical effectiveness percentages; all assumptions visible and user-editable).
- Section 4: User-configurable parameters:
    - selected_disease
    - selected_districts
    - scenario_duration_weeks
    - intervention_start_week
    - hypothetical_resource_allocation
    - simulator_assumptions
- Section 5: Structured scenario outputs (`scenario_id`, `selected_districts`, `selected_intervention`,
  `input_assumptions`, `baseline_comparison`, `simulated_indicators`, `expected_resource_requirements`,
  `model_or_calculation_method`, `generation_timestamp`, `limitations`).
- Section 6: Side-by-side comparison (`Baseline` vs. `Intervention Scenario A` vs. `Intervention Scenario B`).
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.engines.intervention_simulator.capability_inspector import (
    ENGINE_VERSION,
    MANDATORY_INTERVENTION_GOVERNANCE_DISCLAIMER,
)
from src.features.geographic_features import (
    ADMIN_DIVISION_MAP,
    DISTRICT_ID_MAP,
    OFFICIAL_DISTRICTS,
)

SUPPORTED_INTERVENTION_TYPES = (
    "BASELINE_NO_ADDITIONAL_INTERVENTION",
    "INCREASED_SURVEILLANCE_AND_TESTING",
    "ADDITIONAL_VECTOR_CONTROL_ACTIVITY",
    "PUBLIC_AWARENESS_CAMPAIGN",
    "INCREASED_HEALTHCARE_RESOURCE_ALLOCATION",
    "EARLY_INTERVENTION_VS_DELAYED_INTERVENTION",
    "DISTRICT_FOCUSED_RESPONSE",
    "MULTI_DISTRICT_RESPONSE",
)


@dataclass
class InterventionScenarioConfig:
    """User-configurable public-health intervention scenario specification (Phase 14 Sections 2–4)."""

    scenario_id: str
    scenario_name: str
    intervention_type: str
    selected_disease: str = "ALL_DISEASES"
    selected_districts: List[str] = field(default_factory=lambda: ["Pune"])
    include_adjacent_ring_districts: bool = False
    scenario_duration_weeks: int = 8
    intervention_start_week: int = 2
    intervention_ramp_up_weeks: int = 2

    # User-entered baseline growth & cross-district coupling assumptions (explicit what-if parameters)
    baseline_weekly_growth_multiplier: float = 1.08
    baseline_cross_district_coupling_beta: float = 0.06
    custom_initial_weekly_cases_per_district: Optional[float] = None

    # User-assumed hypothetical intervention modifiers (NEVER claimed as empirical efficacy)
    user_assumed_transmission_dampening_factor: float = 0.0
    user_assumed_cross_district_spillover_dampening: float = 0.0
    user_assumed_active_testing_detection_multiplier: float = 1.0

    # User-assumed healthcare resource demand & hypothetical allocation parameters
    user_assumed_admission_proportion: float = 0.18
    user_assumed_icu_proportion_of_admissions: float = 0.15
    user_assumed_avg_length_of_stay_days: float = 5.0
    user_assumed_tests_per_case: float = 8.0
    hypothetical_allocated_general_beds_per_district: Optional[float] = None
    hypothetical_allocated_icu_beds_per_district: Optional[float] = None
    analyst_notes: str = ""

    def validate(self) -> None:
        if self.intervention_type not in SUPPORTED_INTERVENTION_TYPES:
            raise ValueError(
                f"Unsupported intervention_type '{self.intervention_type}'. "
                f"Must be one of {SUPPORTED_INTERVENTION_TYPES}"
            )
        if not self.selected_districts:
            raise ValueError("selected_districts must contain at least one Maharashtra district.")
        for d in self.selected_districts:
            if d not in OFFICIAL_DISTRICTS:
                raise ValueError(f"Unknown district '{d}'. Must be one of 36 official Maharashtra districts.")
        if not (1 <= self.scenario_duration_weeks <= 52):
            raise ValueError("scenario_duration_weeks must be between 1 and 52 weeks.")
        if not (1 <= self.intervention_start_week <= self.scenario_duration_weeks + 1):
            raise ValueError("intervention_start_week must be between 1 and scenario_duration_weeks + 1.")
        if not (0 <= self.intervention_ramp_up_weeks <= 26):
            raise ValueError("intervention_ramp_up_weeks must be between 0 and 26 weeks.")
        if not (0.2 <= self.baseline_weekly_growth_multiplier <= 3.0):
            raise ValueError("baseline_weekly_growth_multiplier must be between 0.2 and 3.0.")
        if not (0.0 <= self.baseline_cross_district_coupling_beta <= 0.5):
            raise ValueError("baseline_cross_district_coupling_beta must be between 0.0 and 0.5.")
        if not (0.0 <= self.user_assumed_transmission_dampening_factor <= 0.90):
            raise ValueError("user_assumed_transmission_dampening_factor must be in [0.0, 0.90].")
        if not (0.0 <= self.user_assumed_cross_district_spillover_dampening <= 0.95):
            raise ValueError("user_assumed_cross_district_spillover_dampening must be in [0.0, 0.95].")
        if not (0.5 <= self.user_assumed_active_testing_detection_multiplier <= 5.0):
            raise ValueError("user_assumed_active_testing_detection_multiplier must be in [0.5, 5.0].")
        if not (0.0 <= self.user_assumed_admission_proportion <= 1.0):
            raise ValueError("user_assumed_admission_proportion must be in [0.0, 1.0].")
        if not (0.0 <= self.user_assumed_icu_proportion_of_admissions <= 1.0):
            raise ValueError("user_assumed_icu_proportion_of_admissions must be in [0.0, 1.0].")
        if not (0.5 <= self.user_assumed_avg_length_of_stay_days <= 60.0):
            raise ValueError("user_assumed_avg_length_of_stay_days must be in [0.5, 60.0].")
        if not (1.0 <= self.user_assumed_tests_per_case <= 200.0):
            raise ValueError("user_assumed_tests_per_case must be in [1.0, 200.0].")


def _resolve_active_and_ring_districts(
    selected_districts: List[str],
    include_ring: bool,
    adj_map: Dict[str, List[str]],
) -> Tuple[List[str], List[str], List[str]]:
    """Return (focal_districts, ring_districts, all_simulated_districts)."""
    focal = list(dict.fromkeys(selected_districts))
    ring: List[str] = []
    if include_ring:
        for d in focal:
            for nb in adj_map.get(d, []):
                if nb not in focal and nb not in ring:
                    ring.append(nb)
    all_dists = focal + ring
    return focal, ring, all_dists


def _lookup_district_baseline_weekly_cases(
    district: str,
    disease: str,
    pool_df: pd.DataFrame,
    custom_override: Optional[float] = None,
) -> Tuple[float, str, float, str]:
    """Lookup verified initial weekly baseline anchor for (district, disease) from Phase 6/9 summary.

    Returns:
        (initial_weekly_cases, anchor_source, observed_total_cases, hotspot_status)
    """
    if custom_override is not None and custom_override >= 0:
        return (
            float(custom_override),
            "USER_CUSTOM_INITIAL_WEEKLY_CASES_OVERRIDE",
            np.nan,
            "CUSTOM_OVERRIDE",
        )

    sub = pool_df[(pool_df["district_name"] == district) & (pool_df["disease"] == disease)]
    if sub.empty and disease != "ALL_DISEASES":
        sub = pool_df[(pool_df["district_name"] == district) & (pool_df["disease"] == "ALL_DISEASES")]

    if not sub.empty:
        row = sub.iloc[0]
        obs_total = float(row["observed_cases"]) if pd.notna(row["observed_cases"]) else 0.0
        hotspot_lbl = str(row.get("gi_star_class", "NOT_SIGNIFICANT"))
        fc7 = row.get("forecast_7d_mean_cases", np.nan)
        obs_mean = row.get("observed_mean_cases_per_outbreak_week", np.nan)

        if pd.notna(fc7) and float(fc7) > 0:
            return float(fc7), "PHASE6_7D_FORECAST_MEAN_ANCHOR", obs_total, hotspot_lbl
        if pd.notna(obs_mean) and float(obs_mean) > 0:
            return float(obs_mean), "PHASE2_OBSERVED_OUTBREAK_WEEKLY_MEAN_ANCHOR", obs_total, hotspot_lbl
        if obs_total > 0:
            wks = max(float(row.get("observed_outbreak_weeks", 1.0)), 1.0)
            return float(obs_total / wks), "PHASE2_OBSERVED_CASES_PER_WEEK_ANCHOR", obs_total, hotspot_lbl

    # Fallback if district had 0 outbreaks for that specific disease in 2023-2025 (e.g. Mumbai City / Ratnagiri)
    all_sub = pool_df[(pool_df["district_name"] == district) & (pool_df["disease"] == "ALL_DISEASES")]
    if not all_sub.empty:
        r0 = all_sub.iloc[0]
        obs_mean0 = r0.get("observed_mean_cases_per_outbreak_week", np.nan)
        if pd.notna(obs_mean0) and float(obs_mean0) > 0:
            return (
                float(obs_mean0) * 0.35,
                "SCALED_ALL_DISEASES_DISTRICT_MEAN_FALLBACK",
                0.0,
                str(r0.get("gi_star_class", "NOT_SIGNIFICANT")),
            )

    return 10.0, "HYPOTHETICAL_SEED_10_CASES_ZERO_SURVEILLANCE_DISTRICT", 0.0, "NO_2023_2025_OUTBREAKS"


def simulate_intervention_scenario(
    config: InterventionScenarioConfig,
    pool_df: pd.DataFrame,
    adj_map: Dict[str, List[str]],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Run transparent parametric intervention simulation and compare against an unmitigated baseline
    over the exact same district set, disease, and horizon.

    Mathematical Formulation (Transparent Parametric What-If Calculator):
    Let D be the set of simulated districts (focal + optional adjacent ring).
    For each district i in D and week t in {1, ..., T}:
      1. Unmitigated Baseline Trajectory B_{i,t}:
         - Week 1: B_{i,1} = B_{i,0} (verified forecast/historical weekly anchor)
         - Week t > 1:
             local_growth = B_{i,t-1} * g_0
             neighbor_spillover = beta_0 * mean_{j in N(i) cap D}(B_{j,t-1})
             B_{i,t} = local_growth + neighbor_spillover
      2. Intervention Ramp Function r(t):
         - For t < t_start: r(t) = 0.0
         - For t >= t_start:
             r(t) = 1.0 if ramp_weeks == 0 else min(1.0, (t - t_start + 1) / ramp_weeks)
      3. Hypothetical Intervention Underlying Trajectory U_{i,t}:
         - Effective local growth multiplier:
             g_eff(t) = g_0 * (1.0 - r(t) * d_trans)
         - Effective cross-district spillover coupling:
             beta_eff(t) = beta_0 * (1.0 - r(t) * d_spill)
         - Week 1:
             U_{i,1} = B_{i,0} * (1.0 - r(1) * d_trans)
         - Week t > 1:
             U_{i,t} = U_{i,t-1} * g_eff(t) + beta_eff(t) * mean_{j in N(i) cap D}(U_{j,t-1})
      4. Surveillance / Active Case-Finding Reported Signal S_{i,t}:
         - When active testing / surveillance is scaled up (`m_test > 1.0`), short-term reported
           case detection increases via active case-finding while early isolation dampens subsequent
           underlying transmission:
             S_{i,t} = U_{i,t} * (1.0 + r(t) * (m_test - 1.0) * 0.25)
      5. Hypothetical Resource Requirements (Phase 11 Little's Law Coupling):
         - Admissions_{i,t} = U_{i,t} * p_adm
         - Concurrent_General_Beds_{i,t} = Admissions_{i,t} * (1 - p_icu) * (ALOS / 7.0)
         - Concurrent_ICU_Beds_{i,t} = Admissions_{i,t} * p_icu * (ALOS / 7.0)
         - Weekly_Diagnostic_Tests_{i,t} = U_{i,t} * tests_per_case * (1.0 + r(t) * (m_test - 1.0))
    """
    config.validate()
    focal_dists, ring_dists, sim_dists = _resolve_active_and_ring_districts(
        config.selected_districts, config.include_adjacent_ring_districts, adj_map
    )

    # Anchor lookup per district
    anchors: Dict[str, float] = {}
    anchor_sources: Dict[str, str] = {}
    obs_totals: Dict[str, float] = {}
    hotspot_labels: Dict[str, str] = {}

    for d in sim_dists:
        b0, src, obs_tot, hs_lbl = _lookup_district_baseline_weekly_cases(
            d,
            config.selected_disease,
            pool_df,
            custom_override=config.custom_initial_weekly_cases_per_district,
        )
        # Ring districts that were not primary focal districts start at 50% of their baseline anchor
        # if they are included purely as a buffer ring
        if d in ring_dists and config.custom_initial_weekly_cases_per_district is None:
            b0 = round(b0 * 0.65, 4)
        anchors[d] = b0
        anchor_sources[d] = src
        obs_totals[d] = obs_tot
        hotspot_labels[d] = hs_lbl

    T = config.scenario_duration_weeks
    g0 = config.baseline_weekly_growth_multiplier
    beta0 = config.baseline_cross_district_coupling_beta
    d_trans = config.user_assumed_transmission_dampening_factor
    d_spill = config.user_assumed_cross_district_spillover_dampening
    m_test = config.user_assumed_active_testing_detection_multiplier

    p_adm = config.user_assumed_admission_proportion
    p_icu = config.user_assumed_icu_proportion_of_admissions
    alos = config.user_assumed_avg_length_of_stay_days
    tpc = config.user_assumed_tests_per_case
    alloc_gen = config.hypothetical_allocated_general_beds_per_district
    alloc_icu = config.hypothetical_allocated_icu_beds_per_district

    prev_B: Dict[str, float] = {}
    prev_U: Dict[str, float] = {}
    weekly_rows: List[Dict[str, Any]] = []

    for t in range(1, T + 1):
        # Compute intervention activation ramp r(t) in [0, 1]
        if t < config.intervention_start_week:
            r_t = 0.0
        else:
            if config.intervention_ramp_up_weeks <= 0:
                r_t = 1.0
            else:
                r_t = min(
                    1.0,
                    float(t - config.intervention_start_week + 1)
                    / float(config.intervention_ramp_up_weeks),
                )

        curr_B: Dict[str, float] = {}
        curr_U: Dict[str, float] = {}

        for d in sim_dists:
            # Active neighbors inside simulated subgraph
            nbs_in_sim = [nb for nb in adj_map.get(d, []) if nb in sim_dists]
            # Whether district receives direct focal intervention or ring spillover intervention
            is_focal = d in focal_dists
            dist_r_t = r_t if (is_focal or config.include_adjacent_ring_districts) else 0.0

            if t == 1:
                b_val = anchors[d]
                u_val = anchors[d] * (1.0 - dist_r_t * d_trans * 0.5)
            else:
                nb_mean_B = (
                    float(np.mean([prev_B[nb] for nb in nbs_in_sim])) if nbs_in_sim else 0.0
                )
                b_val = prev_B[d] * g0 + beta0 * nb_mean_B

                g_eff = g0 * (1.0 - dist_r_t * d_trans)
                beta_eff = beta0 * (1.0 - dist_r_t * d_spill)
                nb_mean_U = (
                    float(np.mean([prev_U[nb] for nb in nbs_in_sim])) if nbs_in_sim else 0.0
                )
                u_val = max(0.0, prev_U[d] * g_eff + beta_eff * nb_mean_U)

            curr_B[d] = b_val
            curr_U[d] = u_val

            # Active surveillance reported signal vs underlying cases
            reported_S = u_val * (1.0 + dist_r_t * (m_test - 1.0) * 0.25)
            hyp_delta_vs_baseline = u_val - b_val
            hyp_delta_pct = ((u_val - b_val) / b_val * 100.0) if b_val > 0 else 0.0

            # Resource demand under Baseline vs Scenario
            base_adm = b_val * p_adm
            base_gen_beds = base_adm * (1.0 - p_icu) * (alos / 7.0)
            base_icu_beds = base_adm * p_icu * (alos / 7.0)
            base_tests = b_val * tpc

            scen_adm = u_val * p_adm
            scen_gen_beds = scen_adm * (1.0 - p_icu) * (alos / 7.0)
            scen_icu_beds = scen_adm * p_icu * (alos / 7.0)
            scen_tests = u_val * tpc * (1.0 + dist_r_t * (m_test - 1.0))

            gen_bed_gap = (
                (alloc_gen - scen_gen_beds) if alloc_gen is not None else np.nan
            )
            icu_bed_gap = (
                (alloc_icu - scen_icu_beds) if alloc_icu is not None else np.nan
            )
            if alloc_gen is not None:
                bed_status = (
                    "HYPOTHETICAL_CAPACITY_ADEQUATE"
                    if gen_bed_gap >= 0 and (alloc_icu is None or icu_bed_gap >= 0)
                    else "HYPOTHETICAL_BED_DEFICIT_ALERT"
                )
            else:
                bed_status = "UNAVAILABLE_NO_ALLOCATED_BED_INPUT"

            weekly_rows.append(
                {
                    "scenario_id": config.scenario_id,
                    "scenario_name": config.scenario_name,
                    "intervention_type": config.intervention_type,
                    "selected_disease": config.selected_disease,
                    "district_name": d,
                    "district_id": DISTRICT_ID_MAP.get(d, "UNKNOWN"),
                    "administrative_division": ADMIN_DIVISION_MAP.get(d, "Unknown"),
                    "district_role_in_scenario": "FOCAL_TARGET_DISTRICT" if is_focal else "ADJACENT_RING_BUFFER_DISTRICT",
                    "simulation_week": t,
                    "intervention_active_flag": bool(t >= config.intervention_start_week),
                    "intervention_ramp_fraction": round(dist_r_t, 4),
                    "observed_historical_2023_2025_cases_reference_only": obs_totals[d],
                    "baseline_anchor_weekly_cases_w1": round(anchors[d], 4),
                    "baseline_anchor_source": anchor_sources[d],
                    "unmitigated_baseline_simulated_cases": round(b_val, 4),
                    "hypothetical_intervention_underlying_cases": round(u_val, 4),
                    "hypothetical_surveillance_reported_cases": round(reported_S, 4),
                    "hypothetical_case_difference_vs_baseline": round(hyp_delta_vs_baseline, 4),
                    "hypothetical_relative_difference_vs_baseline_pct": round(hyp_delta_pct, 2),
                    "baseline_concurrent_general_beds": round(base_gen_beds, 4),
                    "scenario_concurrent_general_beds": round(scen_gen_beds, 4),
                    "baseline_concurrent_icu_beds": round(base_icu_beds, 4),
                    "scenario_concurrent_icu_beds": round(scen_icu_beds, 4),
                    "baseline_weekly_diagnostic_tests": round(base_tests, 2),
                    "scenario_weekly_diagnostic_tests": round(scen_tests, 2),
                    "hypothetical_allocated_general_beds": alloc_gen if alloc_gen is not None else np.nan,
                    "hypothetical_allocated_icu_beds": alloc_icu if alloc_icu is not None else np.nan,
                    "hypothetical_general_bed_surplus_or_deficit": (
                        round(gen_bed_gap, 4) if pd.notna(gen_bed_gap) else np.nan
                    ),
                    "hypothetical_icu_bed_surplus_or_deficit": (
                        round(icu_bed_gap, 4) if pd.notna(icu_bed_gap) else np.nan
                    ),
                    "hypothetical_capacity_comparison_status": bed_status,
                    "record_nature_label": "HYPOTHETICAL_SIMULATED_TRAJECTORY_NEVER_MIX_WITH_OBSERVED",
                    "governance_disclaimer": MANDATORY_INTERVENTION_GOVERNANCE_DISCLAIMER,
                }
            )

        prev_B = curr_B
        prev_U = curr_U

    traj_df = pd.DataFrame(weekly_rows)

    # Aggregate summary metrics (Section 5 required output fields)
    total_base_cases = float(traj_df["unmitigated_baseline_simulated_cases"].sum())
    total_scen_cases = float(traj_df["hypothetical_intervention_underlying_cases"].sum())
    total_reported_cases = float(traj_df["hypothetical_surveillance_reported_cases"].sum())
    cum_diff_cases = total_scen_cases - total_base_cases
    cum_diff_pct = (cum_diff_cases / total_base_cases * 100.0) if total_base_cases > 0 else 0.0

    # Weekly totals across all simulated districts to identify peak week burden
    wk_grp = traj_df.groupby("simulation_week", as_index=False).agg(
        base_wk_cases=("unmitigated_baseline_simulated_cases", "sum"),
        scen_wk_cases=("hypothetical_intervention_underlying_cases", "sum"),
        base_wk_gen_beds=("baseline_concurrent_general_beds", "sum"),
        scen_wk_gen_beds=("scenario_concurrent_general_beds", "sum"),
        base_wk_icu_beds=("baseline_concurrent_icu_beds", "sum"),
        scen_wk_icu_beds=("scenario_concurrent_icu_beds", "sum"),
        base_wk_tests=("baseline_weekly_diagnostic_tests", "sum"),
        scen_wk_tests=("scenario_weekly_diagnostic_tests", "sum"),
    )

    peak_base_cases = float(wk_grp["base_wk_cases"].max())
    peak_scen_cases = float(wk_grp["scen_wk_cases"].max())
    peak_scen_gen_beds = float(wk_grp["scen_wk_gen_beds"].max())
    peak_scen_icu_beds = float(wk_grp["scen_wk_icu_beds"].max())
    total_scen_tests = float(wk_grp["scen_wk_tests"].sum())

    total_alloc_gen = (
        float(alloc_gen * len(sim_dists)) if alloc_gen is not None else None
    )
    total_alloc_icu = (
        float(alloc_icu * len(sim_dists)) if alloc_icu is not None else None
    )

    summary_output: Dict[str, Any] = {
        "scenario_id": config.scenario_id,
        "scenario_name": config.scenario_name,
        "selected_districts": {
            "focal_districts": focal_dists,
            "adjacent_ring_districts": ring_dists,
            "all_simulated_districts": sim_dists,
            "simulated_districts_count": len(sim_dists),
        },
        "selected_intervention": {
            "intervention_type": config.intervention_type,
            "selected_disease": config.selected_disease,
            "scenario_duration_weeks": config.scenario_duration_weeks,
            "intervention_start_week": config.intervention_start_week,
            "intervention_ramp_up_weeks": config.intervention_ramp_up_weeks,
            "include_adjacent_ring_districts": config.include_adjacent_ring_districts,
            "analyst_notes": config.analyst_notes,
        },
        "input_assumptions": {
            "assumption_transparency_status": "USER_CONFIGURABLE_WHAT_IF_ASSUMPTIONS_NO_EMPIRICAL_EFFICACY_CLAIMED",
            "baseline_weekly_growth_multiplier": g0,
            "baseline_cross_district_coupling_beta": beta0,
            "user_assumed_transmission_dampening_factor": d_trans,
            "user_assumed_cross_district_spillover_dampening": d_spill,
            "user_assumed_active_testing_detection_multiplier": m_test,
            "user_assumed_admission_proportion": p_adm,
            "user_assumed_icu_proportion_of_admissions": p_icu,
            "user_assumed_avg_length_of_stay_days": alos,
            "user_assumed_tests_per_case": tpc,
            "hypothetical_allocated_general_beds_per_district": alloc_gen,
            "hypothetical_allocated_icu_beds_per_district": alloc_icu,
        },
        "baseline_comparison": {
            "unmitigated_baseline_cumulative_cases": round(total_base_cases, 2),
            "scenario_hypothetical_cumulative_cases": round(total_scen_cases, 2),
            "hypothetical_cumulative_case_difference_vs_baseline": round(cum_diff_cases, 2),
            "hypothetical_relative_difference_vs_baseline_pct": round(cum_diff_pct, 2),
            "unmitigated_baseline_peak_weekly_cases": round(peak_base_cases, 2),
            "scenario_hypothetical_peak_weekly_cases": round(peak_scen_cases, 2),
            "hypothetical_peak_weekly_case_difference": round(peak_scen_cases - peak_base_cases, 2),
            "baseline_peak_concurrent_general_beds": round(float(wk_grp["base_wk_gen_beds"].max()), 2),
            "scenario_peak_concurrent_general_beds": round(peak_scen_gen_beds, 2),
        },
        "simulated_indicators": {
            "hypothetical_cumulative_underlying_cases": round(total_scen_cases, 2),
            "hypothetical_cumulative_surveillance_reported_cases": round(total_reported_cases, 2),
            "hypothetical_peak_weekly_underlying_cases": round(peak_scen_cases, 2),
            "hypothetical_mean_weekly_underlying_cases": round(total_scen_cases / float(T), 2),
            "peak_burden_simulation_week": int(
                wk_grp.loc[wk_grp["scen_wk_cases"].idxmax(), "simulation_week"]
            ),
        },
        "expected_resource_requirements": {
            "hypothetical_peak_concurrent_general_beds_required": round(peak_scen_gen_beds, 2),
            "hypothetical_peak_concurrent_icu_beds_required": round(peak_scen_icu_beds, 2),
            "hypothetical_cumulative_diagnostic_tests_required": round(total_scen_tests, 2),
            "hypothetical_total_allocated_general_beds": total_alloc_gen,
            "hypothetical_total_allocated_icu_beds": total_alloc_icu,
            "hypothetical_peak_general_bed_surplus_or_deficit": (
                round(total_alloc_gen - peak_scen_gen_beds, 2)
                if total_alloc_gen is not None
                else None
            ),
            "hypothetical_peak_icu_bed_surplus_or_deficit": (
                round(total_alloc_icu - peak_scen_icu_beds, 2)
                if total_alloc_icu is not None
                else None
            ),
        },
        "model_or_calculation_method": (
            "Transparent Parametric What-If Scenario Calculator coupled with Phase 6 Forecast Baseline "
            "Anchors, Phase 10 Verified 83-Edge District Adjacency Graph, and Phase 11 Little's Law "
            "Resource Demand Formulas (Engine Version: " + ENGINE_VERSION + ")"
        ),
        "generation_timestamp": datetime.now().isoformat(),
        "limitations": [
            "No empirical intervention RCT or observational effectiveness dataset exists in the project workspace.",
            "All transmission dampening, spillover dampening, and active testing multipliers are explicit user-configurable what-if assumptions, NOT validated empirical efficacy percentages.",
            "No real hospital bed, ICU, or medicine stock inventory exists in the workspace; capacity gap figures reflect user-entered hypothetical allocations only.",
            "Cross-district coupling uses verified topological adjacency (83 undirected edges), not real-time human mobility.",
            "Simulated indicators are strictly separated from observed surveillance records and must never be interpreted as clinical advice or guaranteed outbreak outcomes.",
        ],
        "governance_disclaimer": MANDATORY_INTERVENTION_GOVERNANCE_DISCLAIMER,
    }
    return summary_output, traj_df


def compare_baseline_vs_scenarios_a_and_b(
    comparison_id: str,
    comparison_title: str,
    baseline_cfg: InterventionScenarioConfig,
    scenario_a_cfg: InterventionScenarioConfig,
    scenario_b_cfg: InterventionScenarioConfig,
    pool_df: pd.DataFrame,
    adj_map: Dict[str, List[str]],
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Compare `Baseline` vs. `Intervention Scenario A` vs. `Intervention Scenario B` (Phase 14 Section 6).

    Returns:
        - comparison_summary_dict
        - weekly_side_by_side_df (by simulation_week)
        - district_side_by_side_df (by district_name)
    """
    base_sum, base_traj = simulate_intervention_scenario(baseline_cfg, pool_df, adj_map)
    scen_a_sum, scen_a_traj = simulate_intervention_scenario(scenario_a_cfg, pool_df, adj_map)
    scen_b_sum, scen_b_traj = simulate_intervention_scenario(scenario_b_cfg, pool_df, adj_map)

    # Weekly side-by-side comparison table
    w_base = base_traj.groupby("simulation_week", as_index=False).agg(
        baseline_cases=("hypothetical_intervention_underlying_cases", "sum"),
        baseline_reported_cases=("hypothetical_surveillance_reported_cases", "sum"),
        baseline_general_beds=("scenario_concurrent_general_beds", "sum"),
        baseline_icu_beds=("scenario_concurrent_icu_beds", "sum"),
        baseline_diagnostic_tests=("scenario_weekly_diagnostic_tests", "sum"),
    )
    w_a = scen_a_traj.groupby("simulation_week", as_index=False).agg(
        scenario_a_cases=("hypothetical_intervention_underlying_cases", "sum"),
        scenario_a_reported_cases=("hypothetical_surveillance_reported_cases", "sum"),
        scenario_a_general_beds=("scenario_concurrent_general_beds", "sum"),
        scenario_a_icu_beds=("scenario_concurrent_icu_beds", "sum"),
        scenario_a_diagnostic_tests=("scenario_weekly_diagnostic_tests", "sum"),
    )
    w_b = scen_b_traj.groupby("simulation_week", as_index=False).agg(
        scenario_b_cases=("hypothetical_intervention_underlying_cases", "sum"),
        scenario_b_reported_cases=("hypothetical_surveillance_reported_cases", "sum"),
        scenario_b_general_beds=("scenario_concurrent_general_beds", "sum"),
        scenario_b_icu_beds=("scenario_concurrent_icu_beds", "sum"),
        scenario_b_diagnostic_tests=("scenario_weekly_diagnostic_tests", "sum"),
    )

    wk_cmp = w_base.merge(w_a, on="simulation_week", how="outer").merge(
        w_b, on="simulation_week", how="outer"
    )
    wk_cmp["comparison_id"] = comparison_id
    wk_cmp["comparison_title"] = comparison_title
    wk_cmp["selected_disease"] = baseline_cfg.selected_disease
    wk_cmp["scenario_a_id"] = scenario_a_cfg.scenario_id
    wk_cmp["scenario_a_name"] = scenario_a_cfg.scenario_name
    wk_cmp["scenario_b_id"] = scenario_b_cfg.scenario_id
    wk_cmp["scenario_b_name"] = scenario_b_cfg.scenario_name
    wk_cmp["scenario_a_vs_baseline_case_diff"] = (
        wk_cmp["scenario_a_cases"] - wk_cmp["baseline_cases"]
    ).round(4)
    wk_cmp["scenario_b_vs_baseline_case_diff"] = (
        wk_cmp["scenario_b_cases"] - wk_cmp["baseline_cases"]
    ).round(4)
    wk_cmp["scenario_a_vs_scenario_b_case_diff"] = (
        wk_cmp["scenario_a_cases"] - wk_cmp["scenario_b_cases"]
    ).round(4)
    wk_cmp["simulation_label"] = "HYPOTHETICAL_SCENARIO_COMPARISON_NOT_EMPIRICAL_PREDICTION"

    # District-wise cumulative comparison table
    d_base = base_traj.groupby(
        ["district_name", "district_id", "administrative_division"], as_index=False
    ).agg(
        observed_historical_2023_2025_cases=("observed_historical_2023_2025_cases_reference_only", "first"),
        baseline_cumulative_cases=("hypothetical_intervention_underlying_cases", "sum"),
        baseline_peak_general_beds=("scenario_concurrent_general_beds", "max"),
    )
    d_a = scen_a_traj.groupby("district_name", as_index=False).agg(
        scenario_a_role=("district_role_in_scenario", "first"),
        scenario_a_cumulative_cases=("hypothetical_intervention_underlying_cases", "sum"),
        scenario_a_peak_general_beds=("scenario_concurrent_general_beds", "max"),
        scenario_a_peak_icu_beds=("scenario_concurrent_icu_beds", "max"),
    )
    d_b = scen_b_traj.groupby("district_name", as_index=False).agg(
        scenario_b_role=("district_role_in_scenario", "first"),
        scenario_b_cumulative_cases=("hypothetical_intervention_underlying_cases", "sum"),
        scenario_b_peak_general_beds=("scenario_concurrent_general_beds", "max"),
        scenario_b_peak_icu_beds=("scenario_concurrent_icu_beds", "max"),
    )

    dist_cmp = d_base.merge(d_a, on="district_name", how="outer").merge(
        d_b, on="district_name", how="outer"
    )
    dist_cmp["comparison_id"] = comparison_id
    dist_cmp["comparison_title"] = comparison_title
    dist_cmp["selected_disease"] = baseline_cfg.selected_disease
    dist_cmp["scenario_a_vs_baseline_diff"] = (
        dist_cmp["scenario_a_cumulative_cases"] - dist_cmp["baseline_cumulative_cases"]
    ).round(2)
    dist_cmp["scenario_b_vs_baseline_diff"] = (
        dist_cmp["scenario_b_cumulative_cases"] - dist_cmp["baseline_cumulative_cases"]
    ).round(2)
    dist_cmp["scenario_a_vs_scenario_b_diff"] = (
        dist_cmp["scenario_a_cumulative_cases"] - dist_cmp["scenario_b_cumulative_cases"]
    ).round(2)
    dist_cmp["record_nature_label"] = "HYPOTHETICAL_SIMULATION_SEPARATED_FROM_OBSERVED_RECORDS"

    comparison_summary = {
        "comparison_id": comparison_id,
        "comparison_title": comparison_title,
        "selected_disease": baseline_cfg.selected_disease,
        "baseline": base_sum,
        "scenario_a": scen_a_sum,
        "scenario_b": scen_b_sum,
        "head_to_head_metrics": {
            "baseline_cumulative_cases": base_sum["simulated_indicators"][
                "hypothetical_cumulative_underlying_cases"
            ],
            "scenario_a_cumulative_cases": scen_a_sum["simulated_indicators"][
                "hypothetical_cumulative_underlying_cases"
            ],
            "scenario_b_cumulative_cases": scen_b_sum["simulated_indicators"][
                "hypothetical_cumulative_underlying_cases"
            ],
            "scenario_a_vs_baseline_cases_delta": scen_a_sum["baseline_comparison"][
                "hypothetical_cumulative_case_difference_vs_baseline"
            ],
            "scenario_b_vs_baseline_cases_delta": scen_b_sum["baseline_comparison"][
                "hypothetical_cumulative_case_difference_vs_baseline"
            ],
            "scenario_a_vs_scenario_b_cases_delta": round(
                scen_a_sum["simulated_indicators"]["hypothetical_cumulative_underlying_cases"]
                - scen_b_sum["simulated_indicators"]["hypothetical_cumulative_underlying_cases"],
                2,
            ),
            "baseline_peak_general_beds": base_sum["expected_resource_requirements"][
                "hypothetical_peak_concurrent_general_beds_required"
            ],
            "scenario_a_peak_general_beds": scen_a_sum["expected_resource_requirements"][
                "hypothetical_peak_concurrent_general_beds_required"
            ],
            "scenario_b_peak_general_beds": scen_b_sum["expected_resource_requirements"][
                "hypothetical_peak_concurrent_general_beds_required"
            ],
        },
        "governance_disclaimer": MANDATORY_INTERVENTION_GOVERNANCE_DISCLAIMER,
    }
    return comparison_summary, wk_cmp, dist_cmp


def build_benchmark_intervention_scenarios() -> Tuple[
    List[InterventionScenarioConfig],
    List[Tuple[str, str, InterventionScenarioConfig, InterventionScenarioConfig, InterventionScenarioConfig]],
]:
    """Define the 8 benchmark scenarios covering all 7 scenario types in Prompt 14 Section 2,
    plus 3 structured `Baseline vs. Scenario A vs. Scenario B` comparison suites.
    """
    # 1. Baseline (No Additional Intervention) — Dengue in Pune + Nashik + Thane + Mumbai Suburban
    s0_baseline_dengue = InterventionScenarioConfig(
        scenario_id="SCEN_00_BASELINE_DENGUE_URBAN_CLUSTER",
        scenario_name="Baseline (No Additional Intervention — Dengue Urban Cluster)",
        intervention_type="BASELINE_NO_ADDITIONAL_INTERVENTION",
        selected_disease="Dengue",
        selected_districts=["Pune", "Nashik", "Thane", "Mumbai Suburban"],
        include_adjacent_ring_districts=False,
        scenario_duration_weeks=8,
        intervention_start_week=2,
        intervention_ramp_up_weeks=2,
        baseline_weekly_growth_multiplier=1.10,
        baseline_cross_district_coupling_beta=0.06,
        user_assumed_transmission_dampening_factor=0.0,
        user_assumed_cross_district_spillover_dampening=0.0,
        user_assumed_active_testing_detection_multiplier=1.0,
        user_assumed_admission_proportion=0.22,
        user_assumed_icu_proportion_of_admissions=0.15,
        user_assumed_avg_length_of_stay_days=5.5,
        user_assumed_tests_per_case=10.0,
        hypothetical_allocated_general_beds_per_district=10.0,
        hypothetical_allocated_icu_beds_per_district=2.0,
        analyst_notes="Unmitigated reference trajectory under user-assumed 10% weekly growth.",
    )

    # 2. Scenario Type 1: Increased Surveillance and Testing
    s1_surveillance_testing = InterventionScenarioConfig(
        scenario_id="SCEN_01_INCREASED_SURVEILLANCE_AND_TESTING",
        scenario_name="Increased Surveillance & Active Testing (What-If 2.0x Testing + Early Isolation)",
        intervention_type="INCREASED_SURVEILLANCE_AND_TESTING",
        selected_disease="Dengue",
        selected_districts=["Pune", "Nashik", "Thane", "Mumbai Suburban"],
        include_adjacent_ring_districts=False,
        scenario_duration_weeks=8,
        intervention_start_week=2,
        intervention_ramp_up_weeks=2,
        baseline_weekly_growth_multiplier=1.10,
        baseline_cross_district_coupling_beta=0.06,
        user_assumed_transmission_dampening_factor=0.18,
        user_assumed_cross_district_spillover_dampening=0.15,
        user_assumed_active_testing_detection_multiplier=2.0,
        user_assumed_admission_proportion=0.22,
        user_assumed_icu_proportion_of_admissions=0.15,
        user_assumed_avg_length_of_stay_days=5.5,
        user_assumed_tests_per_case=10.0,
        hypothetical_allocated_general_beds_per_district=10.0,
        hypothetical_allocated_icu_beds_per_district=2.0,
        analyst_notes=(
            "Hypothetical 2.0x active testing scale-up starting Week 2; increases short-term "
            "surveillance detection while user-assumed early isolation dampens growth by 18%."
        ),
    )

    # 3. Scenario Type 2: Additional Vector-Control Activity (Early Week 2 start)
    s2_vector_control_early = InterventionScenarioConfig(
        scenario_id="SCEN_02_ADDITIONAL_VECTOR_CONTROL_EARLY_W2",
        scenario_name="Additional Vector-Control Activity (Early Start Week 2, User-Assumed 28% Dampening)",
        intervention_type="ADDITIONAL_VECTOR_CONTROL_ACTIVITY",
        selected_disease="Dengue",
        selected_districts=["Pune", "Nashik", "Thane", "Mumbai Suburban"],
        include_adjacent_ring_districts=False,
        scenario_duration_weeks=8,
        intervention_start_week=2,
        intervention_ramp_up_weeks=2,
        baseline_weekly_growth_multiplier=1.10,
        baseline_cross_district_coupling_beta=0.06,
        user_assumed_transmission_dampening_factor=0.28,
        user_assumed_cross_district_spillover_dampening=0.20,
        user_assumed_active_testing_detection_multiplier=1.1,
        user_assumed_admission_proportion=0.22,
        user_assumed_icu_proportion_of_admissions=0.15,
        user_assumed_avg_length_of_stay_days=5.5,
        user_assumed_tests_per_case=10.0,
        hypothetical_allocated_general_beds_per_district=10.0,
        hypothetical_allocated_icu_beds_per_district=2.0,
        analyst_notes=(
            "Hypothetical larval source reduction & vector control initiated at Week 2 with 2-week ramp-up."
        ),
    )

    # 4. Scenario Type 5: Delayed Intervention (Delayed Week 5 start with identical assumptions)
    s3_vector_control_delayed = InterventionScenarioConfig(
        scenario_id="SCEN_03_DELAYED_INTERVENTION_START_W5",
        scenario_name="Delayed Vector-Control Response (Delayed Start Week 5, User-Assumed 28% Dampening)",
        intervention_type="EARLY_INTERVENTION_VS_DELAYED_INTERVENTION",
        selected_disease="Dengue",
        selected_districts=["Pune", "Nashik", "Thane", "Mumbai Suburban"],
        include_adjacent_ring_districts=False,
        scenario_duration_weeks=8,
        intervention_start_week=5,
        intervention_ramp_up_weeks=2,
        baseline_weekly_growth_multiplier=1.10,
        baseline_cross_district_coupling_beta=0.06,
        user_assumed_transmission_dampening_factor=0.28,
        user_assumed_cross_district_spillover_dampening=0.20,
        user_assumed_active_testing_detection_multiplier=1.1,
        user_assumed_admission_proportion=0.22,
        user_assumed_icu_proportion_of_admissions=0.15,
        user_assumed_avg_length_of_stay_days=5.5,
        user_assumed_tests_per_case=10.0,
        hypothetical_allocated_general_beds_per_district=10.0,
        hypothetical_allocated_icu_beds_per_district=2.0,
        analyst_notes=(
            "Identical hypothetical dampening (28%) as SCEN_02, but delayed from Week 2 to Week 5."
        ),
    )

    # 5. Scenario Type 3: Public-Awareness Campaign (Waterborne ADD in Pune, Sangli, Satara, Solapur)
    s4_public_awareness = InterventionScenarioConfig(
        scenario_id="SCEN_04_PUBLIC_AWARENESS_WASH_CAMPAIGN",
        scenario_name="Public-Awareness & WASH Campaign (Acute Diarrhoeal Disease — Pune Division)",
        intervention_type="PUBLIC_AWARENESS_CAMPAIGN",
        selected_disease="Acute Diarrhoeal Disease",
        selected_districts=["Pune", "Satara", "Sangli", "Solapur", "Kolhapur"],
        include_adjacent_ring_districts=False,
        scenario_duration_weeks=8,
        intervention_start_week=2,
        intervention_ramp_up_weeks=3,
        baseline_weekly_growth_multiplier=1.08,
        baseline_cross_district_coupling_beta=0.05,
        user_assumed_transmission_dampening_factor=0.20,
        user_assumed_cross_district_spillover_dampening=0.15,
        user_assumed_active_testing_detection_multiplier=1.15,
        user_assumed_admission_proportion=0.14,
        user_assumed_icu_proportion_of_admissions=0.08,
        user_assumed_avg_length_of_stay_days=3.5,
        user_assumed_tests_per_case=5.0,
        hypothetical_allocated_general_beds_per_district=12.0,
        hypothetical_allocated_icu_beds_per_district=2.0,
        analyst_notes=(
            "Hypothetical community hygiene, safe-water advisories, and ORS awareness campaign (3-week ramp)."
        ),
    )

    # 6. Scenario Type 4: Increased Healthcare Resource Allocation
    s5_resource_allocation = InterventionScenarioConfig(
        scenario_id="SCEN_05_INCREASED_HEALTHCARE_RESOURCE_ALLOCATION",
        scenario_name="Increased Healthcare Surge Bed & ICU Allocation (18 General + 4 ICU Beds/District)",
        intervention_type="INCREASED_HEALTHCARE_RESOURCE_ALLOCATION",
        selected_disease="Dengue",
        selected_districts=["Pune", "Nashik", "Thane", "Mumbai Suburban"],
        include_adjacent_ring_districts=False,
        scenario_duration_weeks=8,
        intervention_start_week=2,
        intervention_ramp_up_weeks=1,
        baseline_weekly_growth_multiplier=1.10,
        baseline_cross_district_coupling_beta=0.06,
        user_assumed_transmission_dampening_factor=0.10,
        user_assumed_cross_district_spillover_dampening=0.10,
        user_assumed_active_testing_detection_multiplier=1.25,
        user_assumed_admission_proportion=0.22,
        user_assumed_icu_proportion_of_admissions=0.15,
        user_assumed_avg_length_of_stay_days=4.5,
        user_assumed_tests_per_case=10.0,
        hypothetical_allocated_general_beds_per_district=18.0,
        hypothetical_allocated_icu_beds_per_district=4.0,
        analyst_notes=(
            "Hypothetical expansion of dedicated ward capacity from 10 to 18 general beds and 2 to 4 ICU beds per district."
        ),
    )

    # 7. Scenario Type 6: District-Focused Response (Single-District Focused vs Ring)
    s6_district_focused = InterventionScenarioConfig(
        scenario_id="SCEN_06_DISTRICT_FOCUSED_RESPONSE_GADCHIROLI",
        scenario_name="District-Focused Response Only (Gadchiroli Focal District, Ring Unmitigated)",
        intervention_type="DISTRICT_FOCUSED_RESPONSE",
        selected_disease="Malaria",
        selected_districts=["Gadchiroli"],
        include_adjacent_ring_districts=False,
        scenario_duration_weeks=8,
        intervention_start_week=2,
        intervention_ramp_up_weeks=2,
        baseline_weekly_growth_multiplier=1.09,
        baseline_cross_district_coupling_beta=0.08,
        user_assumed_transmission_dampening_factor=0.30,
        user_assumed_cross_district_spillover_dampening=0.10,
        user_assumed_active_testing_detection_multiplier=1.3,
        user_assumed_admission_proportion=0.20,
        user_assumed_icu_proportion_of_admissions=0.14,
        user_assumed_avg_length_of_stay_days=5.0,
        user_assumed_tests_per_case=12.0,
        hypothetical_allocated_general_beds_per_district=14.0,
        hypothetical_allocated_icu_beds_per_district=2.5,
        analyst_notes=(
            "Single-district focused response in Gadchiroli with contiguous Vidarbha corridor (Chandrapur, Gondia, Bhandara) also tracked."
        ),
    )

    # 8. Scenario Type 7: Multi-District Coordinated Ring Response
    s7_multi_district_ring = InterventionScenarioConfig(
        scenario_id="SCEN_07_MULTI_DISTRICT_RING_RESPONSE_VIDARBHA",
        scenario_name="Multi-District Coordinated Ring Response (Gadchiroli + Contiguous Vidarbha Ring)",
        intervention_type="MULTI_DISTRICT_RESPONSE",
        selected_disease="Malaria",
        selected_districts=["Gadchiroli", "Chandrapur", "Gondia", "Bhandara"],
        include_adjacent_ring_districts=False,
        scenario_duration_weeks=8,
        intervention_start_week=2,
        intervention_ramp_up_weeks=2,
        baseline_weekly_growth_multiplier=1.09,
        baseline_cross_district_coupling_beta=0.08,
        user_assumed_transmission_dampening_factor=0.30,
        user_assumed_cross_district_spillover_dampening=0.55,
        user_assumed_active_testing_detection_multiplier=1.3,
        user_assumed_admission_proportion=0.20,
        user_assumed_icu_proportion_of_admissions=0.14,
        user_assumed_avg_length_of_stay_days=5.0,
        user_assumed_tests_per_case=12.0,
        hypothetical_allocated_general_beds_per_district=14.0,
        hypothetical_allocated_icu_beds_per_district=2.5,
        analyst_notes=(
            "Coordinated multi-district intervention across Gadchiroli and its contiguous neighbours "
            "(Chandrapur, Gondia, Bhandara) with 55% hypothetical cross-border spillover dampening."
        ),
    )

    benchmark_configs = [
        s0_baseline_dengue,
        s1_surveillance_testing,
        s2_vector_control_early,
        s3_vector_control_delayed,
        s4_public_awareness,
        s5_resource_allocation,
        s6_district_focused,
        s7_multi_district_ring,
    ]

    # Define 3 structured `Baseline vs. Scenario A vs. Scenario B` comparison suites (Section 6)
    # Suite 1: Early Intervention (Week 2) vs. Delayed Intervention (Week 5)
    suite1_early_vs_delayed = (
        "CMP_01_EARLY_VS_DELAYED_VECTOR_CONTROL",
        "Comparison 1: Baseline vs. Early Vector-Control (Week 2) vs. Delayed Response (Week 5)",
        s0_baseline_dengue,
        s2_vector_control_early,
        s3_vector_control_delayed,
    )

    # Suite 2: Surveillance & Testing vs. Surge Healthcare Resource Allocation
    suite2_surveillance_vs_resource = (
        "CMP_02_SURVEILLANCE_TESTING_VS_RESOURCE_ALLOCATION",
        "Comparison 2: Baseline vs. Active Surveillance/Testing (Scenario A) vs. Surge Bed Allocation (Scenario B)",
        s0_baseline_dengue,
        s1_surveillance_testing,
        s5_resource_allocation,
    )

    # Suite 3: Single-District Focused vs. Multi-District Ring Response (over same 4-district Vidarbha subgraph)
    s_vid_base = InterventionScenarioConfig(
        scenario_id="SCEN_VIDARBHA_BASELINE",
        scenario_name="Baseline (Unmitigated Vidarbha 4-District Malaria Corridor)",
        intervention_type="BASELINE_NO_ADDITIONAL_INTERVENTION",
        selected_disease="Malaria",
        selected_districts=["Gadchiroli", "Chandrapur", "Gondia", "Bhandara"],
        include_adjacent_ring_districts=False,
        scenario_duration_weeks=8,
        intervention_start_week=2,
        intervention_ramp_up_weeks=2,
        baseline_weekly_growth_multiplier=1.09,
        baseline_cross_district_coupling_beta=0.08,
        user_assumed_transmission_dampening_factor=0.0,
        user_assumed_cross_district_spillover_dampening=0.0,
        user_assumed_active_testing_detection_multiplier=1.0,
        user_assumed_admission_proportion=0.20,
        user_assumed_icu_proportion_of_admissions=0.14,
        user_assumed_avg_length_of_stay_days=5.0,
        user_assumed_tests_per_case=12.0,
        hypothetical_allocated_general_beds_per_district=14.0,
        hypothetical_allocated_icu_beds_per_district=2.5,
    )
    s_vid_single = InterventionScenarioConfig(
        scenario_id="SCEN_VIDARBHA_MODERATE_SINGLE_FOCUS",
        scenario_name="Scenario A: Moderate Local Response Without Cross-District Spillover Control",
        intervention_type="DISTRICT_FOCUSED_RESPONSE",
        selected_disease="Malaria",
        selected_districts=["Gadchiroli", "Chandrapur", "Gondia", "Bhandara"],
        include_adjacent_ring_districts=False,
        scenario_duration_weeks=8,
        intervention_start_week=2,
        intervention_ramp_up_weeks=2,
        baseline_weekly_growth_multiplier=1.09,
        baseline_cross_district_coupling_beta=0.08,
        user_assumed_transmission_dampening_factor=0.15,
        user_assumed_cross_district_spillover_dampening=0.0,
        user_assumed_active_testing_detection_multiplier=1.15,
        user_assumed_admission_proportion=0.20,
        user_assumed_icu_proportion_of_admissions=0.14,
        user_assumed_avg_length_of_stay_days=5.0,
        user_assumed_tests_per_case=12.0,
        hypothetical_allocated_general_beds_per_district=14.0,
        hypothetical_allocated_icu_beds_per_district=2.5,
    )
    suite3_district_vs_multi = (
        "CMP_03_LOCAL_FOCUS_VS_MULTI_DISTRICT_COORDINATED_RING",
        "Comparison 3: Baseline vs. Uncoordinated Local Response (Scenario A) vs. Multi-District Ring Response (Scenario B)",
        s_vid_base,
        s_vid_single,
        s7_multi_district_ring,
    )

    comparison_suites = [
        suite1_early_vs_delayed,
        suite2_surveillance_vs_resource,
        suite3_district_vs_multi,
    ]
    return benchmark_configs, comparison_suites
