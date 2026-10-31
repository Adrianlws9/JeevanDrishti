"""Master Pipeline for Public-Health Intervention Scenario Simulator (Phase 14).

Executes:
1. Upstream capability inspection across all 7 analytical engines (Phases 6–12)
2. 8 Benchmark Intervention Scenarios covering all 7 required scenario types in Prompt 14 Section 2
3. 3 Structured `Baseline vs. Scenario A vs. Scenario B` comparison suites (Prompt 14 Section 6)
4. 5 Static PNG Visualizations + Standalone Interactive HTML Simulator (Prompt 14 Section 7)
5. Persistence of structured CSV/JSON artifacts in `data/intervention_scenarios/` and
   comprehensive Markdown report in `reports/intervention_simulator/intervention_scenario_simulator_report.md`.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.engines.intervention_simulator.capability_inspector import (
    ENGINE_VERSION,
    MANDATORY_INTERVENTION_GOVERNANCE_DISCLAIMER,
    PROJ,
    inspect_upstream_project_capabilities,
)
from src.engines.intervention_simulator.intervention_calculator import (
    build_benchmark_intervention_scenarios,
    compare_baseline_vs_scenarios_a_and_b,
    simulate_intervention_scenario,
)
from src.engines.intervention_simulator.visualizations import (
    generate_intervention_visualizations,
)


def run_intervention_simulator_pipeline(proj_dir: Path = PROJ) -> Dict[str, Any]:
    """Execute end-to-end Phase 14 Public-Health Intervention Scenario Simulator pipeline."""
    data_dir = proj_dir / "data" / "intervention_scenarios"
    report_dir = proj_dir / "reports" / "intervention_simulator"
    fig_dir = report_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Inspect upstream project capabilities
    capability_df, pool_df, adj_map, cap_meta = inspect_upstream_project_capabilities(proj_dir)
    cap_csv = data_dir / "upstream_capability_inspection.csv"
    capability_df.to_csv(cap_csv, index=False)

    # 2. Run all 8 benchmark scenarios
    benchmark_configs, comparison_suites = build_benchmark_intervention_scenarios()
    scenario_summaries: List[Dict[str, Any]] = []
    all_traj_dfs: List[pd.DataFrame] = []
    summary_flat_rows: List[Dict[str, Any]] = []

    for cfg in benchmark_configs:
        s_out, traj_df = simulate_intervention_scenario(cfg, pool_df, adj_map)
        scenario_summaries.append(s_out)
        all_traj_dfs.append(traj_df)

        summary_flat_rows.append(
            {
                "scenario_id": s_out["scenario_id"],
                "scenario_name": s_out["scenario_name"],
                "intervention_type": s_out["selected_intervention"]["intervention_type"],
                "selected_disease": s_out["selected_intervention"]["selected_disease"],
                "focal_districts": ", ".join(s_out["selected_districts"]["focal_districts"]),
                "simulated_districts_count": s_out["selected_districts"]["simulated_districts_count"],
                "scenario_duration_weeks": s_out["selected_intervention"]["scenario_duration_weeks"],
                "intervention_start_week": s_out["selected_intervention"]["intervention_start_week"],
                "intervention_ramp_up_weeks": s_out["selected_intervention"]["intervention_ramp_up_weeks"],
                "baseline_weekly_growth_multiplier": s_out["input_assumptions"][
                    "baseline_weekly_growth_multiplier"
                ],
                "user_assumed_transmission_dampening_factor": s_out["input_assumptions"][
                    "user_assumed_transmission_dampening_factor"
                ],
                "user_assumed_cross_district_spillover_dampening": s_out["input_assumptions"][
                    "user_assumed_cross_district_spillover_dampening"
                ],
                "user_assumed_active_testing_detection_multiplier": s_out["input_assumptions"][
                    "user_assumed_active_testing_detection_multiplier"
                ],
                "unmitigated_baseline_cumulative_cases": s_out["baseline_comparison"][
                    "unmitigated_baseline_cumulative_cases"
                ],
                "scenario_hypothetical_cumulative_cases": s_out["simulated_indicators"][
                    "hypothetical_cumulative_underlying_cases"
                ],
                "hypothetical_surveillance_reported_cases": s_out["simulated_indicators"][
                    "hypothetical_cumulative_surveillance_reported_cases"
                ],
                "hypothetical_case_difference_vs_baseline": s_out["baseline_comparison"][
                    "hypothetical_cumulative_case_difference_vs_baseline"
                ],
                "hypothetical_relative_difference_vs_baseline_pct": s_out["baseline_comparison"][
                    "hypothetical_relative_difference_vs_baseline_pct"
                ],
                "hypothetical_peak_concurrent_general_beds_required": s_out[
                    "expected_resource_requirements"
                ]["hypothetical_peak_concurrent_general_beds_required"],
                "hypothetical_peak_concurrent_icu_beds_required": s_out[
                    "expected_resource_requirements"
                ]["hypothetical_peak_concurrent_icu_beds_required"],
                "hypothetical_cumulative_diagnostic_tests_required": s_out[
                    "expected_resource_requirements"
                ]["hypothetical_cumulative_diagnostic_tests_required"],
                "hypothetical_total_allocated_general_beds": s_out[
                    "expected_resource_requirements"
                ]["hypothetical_total_allocated_general_beds"],
                "hypothetical_peak_general_bed_surplus_or_deficit": s_out[
                    "expected_resource_requirements"
                ]["hypothetical_peak_general_bed_surplus_or_deficit"],
                "model_or_calculation_method": s_out["model_or_calculation_method"],
                "generation_timestamp": s_out["generation_timestamp"],
                "governance_disclaimer": s_out["governance_disclaimer"],
            }
        )

    all_traj_df = pd.concat(all_traj_dfs, ignore_index=True)
    summary_flat_df = pd.DataFrame(summary_flat_rows)

    traj_csv = data_dir / "intervention_scenario_weekly_trajectories.csv"
    summary_csv = data_dir / "intervention_scenario_summary_catalog.csv"
    all_traj_df.to_csv(traj_csv, index=False)
    summary_flat_df.to_csv(summary_csv, index=False)

    # 3. Run the 3 structured `Baseline vs. Scenario A vs. Scenario B` comparison suites
    cmp_summaries: List[Dict[str, Any]] = []
    wk_cmp_dfs: List[pd.DataFrame] = []
    dist_cmp_dfs: List[pd.DataFrame] = []

    for cid, ctitle, b_cfg, a_cfg, b2_cfg in comparison_suites:
        c_sum, wk_df, dist_df = compare_baseline_vs_scenarios_a_and_b(
            cid, ctitle, b_cfg, a_cfg, b2_cfg, pool_df, adj_map
        )
        cmp_summaries.append(c_sum)
        wk_cmp_dfs.append(wk_df)
        dist_cmp_dfs.append(dist_df)

    weekly_cmp_df = pd.concat(wk_cmp_dfs, ignore_index=True)
    district_cmp_df = pd.concat(dist_cmp_dfs, ignore_index=True)

    wk_cmp_csv = data_dir / "baseline_vs_scenario_a_vs_scenario_b_weekly.csv"
    dist_cmp_csv = data_dir / "baseline_vs_scenario_a_vs_scenario_b_districts.csv"
    weekly_cmp_df.to_csv(wk_cmp_csv, index=False)
    district_cmp_df.to_csv(dist_cmp_csv, index=False)

    # Save structured JSON payload for frontend integration
    payload_json_path = data_dir / "intervention_scenarios_frontend_payload.json"
    frontend_payload = {
        "engine_version": ENGINE_VERSION,
        "generation_timestamp": datetime.now().isoformat(),
        "governance_disclaimer": MANDATORY_INTERVENTION_GOVERNANCE_DISCLAIMER,
        "capability_metadata": cap_meta,
        "benchmark_scenarios": scenario_summaries,
        "side_by_side_comparisons": cmp_summaries,
    }
    payload_json_path.write_text(json.dumps(frontend_payload, indent=2), encoding="utf-8")

    # 4. Generate static PNG figures + interactive HTML simulator
    html_sim_path = report_dir / "interactive_intervention_scenario_simulator.html"
    fig_paths = generate_intervention_visualizations(
        capability_df=capability_df,
        pool_df=pool_df,
        adj_map=adj_map,
        scenario_summaries=scenario_summaries,
        weekly_cmp_df=weekly_cmp_df,
        district_cmp_df=district_cmp_df,
        fig_dir=fig_dir,
        html_path=html_sim_path,
    )

    # 5. Write comprehensive Markdown report
    report_md_path = report_dir / "intervention_scenario_simulator_report.md"
    _write_markdown_report(
        report_md_path=report_md_path,
        capability_df=capability_df,
        summary_flat_df=summary_flat_df,
        cmp_summaries=cmp_summaries,
        fig_paths=fig_paths,
    )

    result_meta = {
        "engine_version": ENGINE_VERSION,
        "upstream_engines_inspected": int(len(capability_df) - 1),
        "benchmark_scenarios_count": len(scenario_summaries),
        "trajectory_rows_count": len(all_traj_df),
        "comparison_suites_count": len(cmp_summaries),
        "weekly_comparison_rows": len(weekly_cmp_df),
        "district_comparison_rows": len(district_cmp_df),
        "artifacts": {
            "upstream_capability_csv": str(cap_csv),
            "scenario_trajectories_csv": str(traj_csv),
            "scenario_summary_csv": str(summary_csv),
            "weekly_comparison_csv": str(wk_cmp_csv),
            "district_comparison_csv": str(dist_cmp_csv),
            "frontend_payload_json": str(payload_json_path),
            "interactive_html": str(html_sim_path),
            "markdown_report": str(report_md_path),
            **fig_paths,
        },
    }
    print(
        f"[Phase 14 Pipeline Complete] Upstream engines inspected: 7 | "
        f"Benchmark scenarios: {len(scenario_summaries)} | "
        f"Weekly trajectory rows: {len(all_traj_df)} | "
        f"Comparison suites: {len(cmp_summaries)}"
    )
    return result_meta


def _write_markdown_report(
    report_md_path: Path,
    capability_df: pd.DataFrame,
    summary_flat_df: pd.DataFrame,
    cmp_summaries: List[Dict[str, Any]],
    fig_paths: Dict[str, str],
) -> None:
    """Write the Phase 14 technical & governance evaluation report."""
    lines: List[str] = [
        "# JEEVAN-DRISHTI — Phase 14: Public-Health Intervention Scenario Simulator Report",
        "",
        f"**Engine Version**: `{ENGINE_VERSION}`  ",
        f"**Generated Timestamp**: `{datetime.now().isoformat()}`  ",
        "**Modelling Approach**: `Transparent Parametric What-If Scenario Calculator + Adjacency Ring Coupling + Little's Law Resource Demand`",
        "",
        "> [!CAUTION]",
        f"> **GOVERNANCE SAFEGUARD**: {MANDATORY_INTERVENTION_GOVERNANCE_DISCLAIMER}",
        "",
        "---",
        "",
        "## 1. Upstream Project Capability Inspection (Section 1)",
        "",
        "Before implementing any intervention simulation logic, all 7 upstream intelligence engines (Phases 6–12) and the workspace data inventory were audited:",
        "",
        "| Upstream Engine / Dataset | Version | Genuinely Available | Verified Variables Used | Role in Intervention Simulator |",
        "| :--- | :--- | :---: | :--- | :--- |",
    ]

    for _, r in capability_df.iterrows():
        av_badge = "YES" if r["genuinely_available_in_workspace"] else "**NO (0 RCT Records)**"
        lines.append(
            f"| {r['upstream_engine']} | `{r['version']}` | {av_badge} | {r['variables_available']} | {r['role_in_intervention_simulator']} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 2. Transparent Scenario Calculator Formulation (Sections 2–4)",
            "",
            "Because **zero empirical intervention efficacy or randomized controlled trial (RCT) datasets** exist in the project workspace, the simulator implements a **Transparent Parametric What-If Scenario Calculator** where every assumption is explicitly visible and user-editable:",
            "",
            "1. **Verified Baseline Caseload Anchor ($B_{i,0}$)**: Initialized from Phase 6 `forecast_7d_mean_cases` (or Phase 2 `observed_mean_cases_per_outbreak_week` fallback) for each selected district $i$ and disease.",
            "2. **Unmitigated Baseline Trajectory ($B_{i,t}$)**:",
            "   $$B_{i,t} = B_{i,t-1} \\cdot g_0 + \\beta_0 \\cdot \\frac{1}{|N(i) \\cap D|} \\sum_{j \\in N(i) \\cap D} B_{j,t-1}$$",
            "3. **Intervention Activation Ramp ($r(t) \\in [0, 1]$)**:",
            "   $$r(t) = \\begin{cases} 0 & \\text{if } t < t_{\\text{start}} \\\\ \\min\\!\\left(1,\\; \\frac{t - t_{\\text{start}} + 1}{w_{\\text{ramp}}}\\right) & \\text{if } t \\ge t_{\\text{start}} \\end{cases}$$",
            "4. **Hypothetical Intervention Underlying Trajectory ($U_{i,t}$)**:",
            "   $$U_{i,t} = U_{i,t-1} \\cdot g_0 \\cdot \\bigl(1 - r(t)\\,d_{\\text{trans}}\\bigr) + \\beta_0 \\cdot \\bigl(1 - r(t)\\,d_{\\text{spill}}\\bigr) \\cdot \\frac{1}{|N(i) \\cap D|} \\sum_{j \\in N(i) \\cap D} U_{j,t-1}$$",
            "5. **Active Surveillance Case-Finding Signal ($S_{i,t}$)**:",
            "   $$S_{i,t} = U_{i,t} \\cdot \\bigl(1 + 0.25 \\cdot r(t) \\cdot (m_{\\text{test}} - 1)\\bigr)$$",
            "6. **Hypothetical Healthcare Resource Coupling (Phase 11 Little's Law)**:",
            "   $$\\text{General Beds}_{i,t} = U_{i,t} \\cdot p_{\\text{adm}} \\cdot (1 - p_{\\text{icu}}) \\cdot \\frac{\\text{ALOS}}{7}, \\qquad \\text{ICU Beds}_{i,t} = U_{i,t} \\cdot p_{\\text{adm}} \\cdot p_{\\text{icu}} \\cdot \\frac{\\text{ALOS}}{7}$$",
            "",
            "---",
            "",
            "## 3. Benchmark Scenario Catalog Across All 7 Scenario Types (Sections 2 & 5)",
            "",
            "| Scenario ID | Intervention Type | Disease | Focal Districts | Start Wk | Assumed Dampening | Baseline Cases | Scenario Cases | Hypothetical Diff vs. Baseline | Peak Gen. Beds |",
            "| :--- | :--- | :--- | :--- | :---: | :---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for _, r in summary_flat_df.iterrows():
        lines.append(
            f"| `{r['scenario_id']}` | `{r['intervention_type']}` | {r['selected_disease']} | "
            f"{r['focal_districts']} | W{int(r['intervention_start_week'])} | "
            f"{r['user_assumed_transmission_dampening_factor']*100:.0f}% | "
            f"{r['unmitigated_baseline_cumulative_cases']:.1f} | "
            f"{r['scenario_hypothetical_cumulative_cases']:.1f} | "
            f"{r['hypothetical_case_difference_vs_baseline']:+.1f} ({r['hypothetical_relative_difference_vs_baseline_pct']:+.1f}%) | "
            f"{r['hypothetical_peak_concurrent_general_beds_required']:.1f} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 4. Side-by-Side Comparison Suites: `Baseline` vs. `Scenario A` vs. `Scenario B` (Section 6)",
            "",
            "| Comparison Suite ID | Disease | Baseline Cum. Cases | Scenario A Cum. Cases (Delta) | Scenario B Cum. Cases (Delta) | Scenario A vs. B Delta | Peak Gen. Beds (Base / A / B) |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | :---: |",
        ]
    )

    for c in cmp_summaries:
        h = c["head_to_head_metrics"]
        lines.append(
            f"| `{c['comparison_id']}` | {c['selected_disease']} | "
            f"{h['baseline_cumulative_cases']:.1f} | "
            f"{h['scenario_a_cumulative_cases']:.1f} ({h['scenario_a_vs_baseline_cases_delta']:+.1f}) | "
            f"{h['scenario_b_cumulative_cases']:.1f} ({h['scenario_b_vs_baseline_cases_delta']:+.1f}) | "
            f"{h['scenario_a_vs_scenario_b_cases_delta']:+.1f} | "
            f"{h['baseline_peak_general_beds']:.1f} / {h['scenario_a_peak_general_beds']:.1f} / {h['scenario_b_peak_general_beds']:.1f} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 5. Generated Visual & Interactive Artifacts (Section 7)",
            "",
            f"- **Figure 1 (Capability Audit)**: `{fig_paths['capability_audit_png']}`",
            f"- **Figure 2 (Baseline vs. Scenario A vs. Scenario B Timelines)**: `{fig_paths['timeline_comparison_png']}`",
            f"- **Figure 3 (District-Wise Scenario Map & Comparison)**: `{fig_paths['district_map_comparison_png']}`",
            f"- **Figure 4 (Intervention Timing vs. Dampening Sensitivity Grid)**: `{fig_paths['sensitivity_grid_png']}`",
            f"- **Figure 5 (Hypothetical Resource Allocation & Bed Strain)**: `{fig_paths['resource_allocation_png']}`",
            f"- **Interactive HTML Simulator**: `{fig_paths['interactive_html_simulator']}`",
            "",
            "---",
            "",
            "## 6. Governance & Limitations Checklist (Section 8)",
            "",
            "- **Zero Fabricated Efficacy Claims**: Every scenario table, chart, and JSON payload explicitly labels transmission dampening, spillover dampening, and testing scale-up as user-editable what-if assumptions.",
            "- **Strict Separation from Observed Records**: All weekly simulation rows carry `record_nature_label = HYPOTHETICAL_SIMULATED_TRAJECTORY_NEVER_MIX_WITH_OBSERVED` and are stored exclusively in `data/intervention_scenarios/`.",
            "- **Non-Clinical Decision Support Only**: The engine never recommends clinical treatments, never executes automated policies, and never replaces qualified public-health epidemiologists.",
        ]
    )

    report_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run_intervention_simulator_pipeline()
