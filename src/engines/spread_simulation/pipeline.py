"""End-to-End Execution Pipeline for JEEVAN-DRISHTI Phase 10 District Spread Scenario Engine."""
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.engines.spread_simulation.district_graph import (
    MANDATORY_SCENARIO_DISCLAIMER,
    DistrictGraph,
    build_district_graph,
    inspect_spread_simulation_assets,
)
from src.engines.spread_simulation.scenario_simulator import (
    ENGINE_VERSION,
    build_default_benchmark_scenarios,
    run_parameter_sensitivity_grid,
    run_scenario_simulation,
)
from src.engines.spread_simulation.visualizations import generate_all_spread_visualizations

PROJ = Path(__file__).resolve().parents[3]


def _write_spread_simulation_report(
    audit_info: Dict[str, Any],
    graph: DistrictGraph,
    all_results: List[Dict[str, Any]],
    sens_df: pd.DataFrame,
    viz_info: Dict[str, Any],
    report_path: Path,
) -> None:
    """Write the comprehensive Phase 10 District Spread Propagation & Scenario Simulation Report."""
    now_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    lines = [
        "# JEEVAN-DRISHTI Phase 10: District Spread Propagation & Scenario Simulation Report",
        "",
        f"> **MANDATORY SCIENTIFIC & NON-PREDICTION BOUNDARY**: {MANDATORY_SCENARIO_DISCLAIMER}",
        "",
        f"- **Generated At**: `{now_str}`",
        f"- **Engine Version**: `{ENGINE_VERSION}`",
        "- **Simulation Method**: Transparent, Deterministic Graph-Based Signal Propagation Calculator (`0–100` Hypothetical Stress Index) over the 36-District Maharashtra Network.",
        "",
        "## 1. Geographic, Road-Network & Mobility Asset Inspection",
        "",
        "| Data Asset | Available in Workspace? | Engine Handling & Non-Fabrication Safeguard |",
        "|---|---|---|",
        f"| **District Adjacency (`spatial_weights_matrix.csv`)** | `Yes` (`{audit_info['validated_adjacency_source']}`) | **83 undirected edges (`166` directed arcs)** across **35 geocoded districts** (`degree 4..7`, mean `4.74`). |",
        f"| **District Point Coordinates (`lat`, `lon`)** | `Yes` (`35 / 36` districts in `{audit_info['coordinate_source']}`) | `Mumbai Suburban` (`MH_D19`) has no coordinates in workspace; represented as an explicit isolated node (`degree = 0`) with zero invented edges. |",
        "| **District Polygon Boundaries (`.geojson`, `.shp`)** | `No` (`0` files) | Not downloaded or invented. |",
        "| **Road-Network Data** | `No` (`0` files) | Not downloaded or invented. |",
        "| **Human Mobility / Migration / Traffic Data** | `No` (`0` files) | **Equal movement is NOT assumed**, and traffic/migration numbers are **never invented**. Edge weights use documented geometric proximity functions strictly for illustrative scenario exploration. |",
        "| **Population Denominators** | `No` (`0` files) | Compartmental `SEIR` population rates are not fabricated; engine computes a transparent dimensionless signal index (`0–100`). |",
        f"| **Historical Disease Surveillance (`2023–2025`)** | `Yes` (`{audit_info['disease_history_source']}`) | Retained strictly as a separate reference column (`observed_2023_2025_mean_weekly_cases_ref`); **never mixed with simulated values**. |",
        "",
        "## 2. District Graph Structure & Edge Weighting Modes",
        "",
        "- **Nodes ($V$)**: `36` official Maharashtra districts (`MH_D01..MH_D36`). `35` nodes are connected in a single island-free component; `Mumbai Suburban` (`MH_D19`) has `degree = 0` (`ISOLATED_NO_COORDINATES_NO_PROPAGATION_EDGES`).",
        "- **Edges ($E$)**: `83` undirected edges (`166` directed arcs) with great-circle Haversine distance $d_{ij} \\in [36.26\\text{ km}, 178.95\\text{ km}]$ and administrative division flags.",
        "- **Configurable Edge Weighting Assumptions ($W_{\\text{mode}}$)** (row-normalized per receiving district):",
        "  1. `distance_decayed`: $w_{ij} \\propto \\exp(-d_{ij} / \\lambda_{\\text{km}})$ — closer neighbours exert stronger hypothetical geometric coupling.",
        "  2. `inverse_distance`: $w_{ij} \\propto 1 / d_{ij}$.",
        "  3. `division_bounded`: $w_{ij} \\propto \\exp(-d_{ij} / \\lambda_{\\text{km}}) \\cdot \\gamma_{\\text{cross}}$ when districts $i$ and $j$ belong to different revenue divisions.",
        "  4. `uniform_topological`: $w_{ij} = 1 / \\deg(i)$ — pure graph-topology baseline.",
        "",
        "## 3. Transparent Graph-Based Scenario Update Equation",
        "",
        "For each geocoded district $i$ and hypothetical step $t \\rightarrow t + 1$:",
        "$$m_{\\text{int}}(t+1) = \\begin{cases} 1 - \\eta_{\\text{cov}}, & \\text{if } t + 1 \\ge t_{\\text{int}} \\\\ 1.0, & \\text{if } t + 1 < t_{\\text{int}} \\end{cases}$$",
        "$$P_i(t+1) = \\rho_{\\text{persist}} \\cdot m_{\\text{int}}(t+1) \\cdot S_i(t), \\qquad I_i(t+1) = \\beta_{\\text{prop}} \\cdot m_{\\text{int}}(t+1) \\sum_{j \\in \\mathcal{N}(i)} W_{ij}^{(\\text{mode})} S_j(t)$$",
        "$$S_i(t+1) = \\min\\!\\left(100.0,\\; P_i(t+1) + I_i(t+1) \\cdot \\left(1 - \\frac{S_i(t)}{100.0}\\right)\\right)$$",
        "",
        "## 4. Benchmark Hypothetical Scenarios (`SCEN_01` – `SCEN_06`)",
        "",
        "| Scenario ID | Initial Seed District(s) | Edge Mode ($\\lambda_{\\text{km}}$) | $\\beta_{\\text{prop}}$ / $\\rho_{\\text{persist}}$ | Intervention ($t_{\\text{int}}$, $\\eta_{\\text{cov}}$) | Ever Affected ($\\ge 5.0$) | Peak Signal Sum (Step) | Final Step 12 Signal Sum |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for r in all_results:
        s = r["summary"]
        p = s["scenario_parameters"]
        lines.append(
            f"| `{s['scenario_id']}` | `{s['initial_district']}` | `{p['edge_weighting_mode']}` ({p['neighbour_influence_decay_km']:.0f}km) | `{p['hypothetical_propagation_factor']:.2f}` / `{p['within_district_persistence_factor']:.2f}` | Step `{p['intervention_start_step']}` (`{p['intervention_response_coverage']:.0%}`) | **{s['total_districts_ever_affected']}** / 35 | **{s['peak_statewide_signal_sum']:.1f}** (`t={s['peak_step']}`) | **{s['final_step_signal_sum']:.1f}** ({s['final_step_active_districts']} active) |"
        )

    lines.extend(
        [
            "",
            "### Step-by-Step Affected District Wavefront Sequences (`Activation Threshold >= 5.0`)",
            "",
        ]
    )
    for r in all_results:
        s = r["summary"]
        lines.append(f"- **`{s['scenario_id']}`** ({s['scenario_name']}):")
        lines.append(f"  - *Wavefront Sequence*: `{s['affected_district_sequence']}`")

    lines.extend(
        [
            "",
            "## 5. Before-and-After Intervention Impact Comparison",
            "",
            "1. **Eastern Vidarbha (`SCEN_01` Delayed Response vs. `SCEN_02` Early Intervention)**:",
            f"   - Under delayed intervention (`t_int = 8`, `30%` coverage), the hypothetical signal seeded in `Gadchiroli (85.0)` reaches **{all_results[0]['summary']['total_districts_ever_affected']} districts** with a peak statewide signal sum of **{all_results[0]['summary']['peak_statewide_signal_sum']:.1f}** and final step-12 sum of **{all_results[0]['summary']['final_step_signal_sum']:.1f}**.",
            f"   - With early intervention (`t_int = 2`, `70%` coverage) under identical seed and propagation assumptions, the wavefront is restricted to **{all_results[1]['summary']['total_districts_ever_affected']} districts** and the final step-12 signal sum drops to **{all_results[1]['summary']['final_step_signal_sum']:.1f}** (a **{100.0 * (1.0 - all_results[1]['summary']['final_step_signal_sum'] / max(all_results[0]['summary']['final_step_signal_sum'], 1e-6)):.1f}%** reduction at step 12).",
            "2. **Marathwada Stress Test (`SCEN_05` Unmitigated vs. `SCEN_06` Rapid Containment)**:",
            f"   - Under unmitigated high propagation (`beta_prop = 0.55`, `0%` coverage) seeded in `Nanded (90.0)`, **{all_results[4]['summary']['total_districts_ever_affected']} districts** cross the activation threshold and final step-12 signal reaches **{all_results[4]['summary']['final_step_signal_sum']:.1f}**.",
            f"   - Rapid containment at `step 3` (`65%` coverage) limits spread to **{all_results[5]['summary']['total_districts_ever_affected']} districts** and reduces final step-12 signal to **{all_results[5]['summary']['final_step_signal_sum']:.1f}** (a **{100.0 * (1.0 - all_results[5]['summary']['final_step_signal_sum'] / max(all_results[4]['summary']['final_step_signal_sum'], 1e-6)):.1f}%** reduction).",
            "",
            "## 6. Generated Interactive & Static Visualizations (`reports/spread_simulation/`)",
            "",
            f"- **Interactive HTML District Graph & Propagation Animation**: `{Path(viz_info['interactive_html']).relative_to(PROJ).as_posix()}`",
        ]
    )
    for fp in viz_info["static_figures"]:
        lines.append(f"- **Static Figure**: `{Path(fp).relative_to(PROJ).as_posix()}`")

    lines.extend(
        [
            "",
            "## 7. Scientific Limitations & Interpretation Rules",
            "",
            "1. **Hypothetical Exploration Only**: Simulated indicator values (`simulated_signal_index`) represent deterministic mathematical consequences of user-selected parameters on the Maharashtra district adjacency graph; they are **never** real outbreak predictions or forecasts.",
            "2. **Absence of Human Mobility & Road Traffic Data**: Geographic proximity does not equal human travel volume. No commuter, rail, air, or highway traffic data exist in the workspace.",
            "3. **Absence of Biological & Demographic Parameters**: No district population denominators or pathogen-specific $R_0$ parameters exist in the workspace; therefore, no compartmental `SEIR` incidence claims are made.",
            "4. **Mumbai Suburban Isolation**: Because `Mumbai Suburban` (`MH_D19`) lacks coordinates in all project datasets, it has `0` graph edges and remains at `0.0` during neighbor-driven propagation unless explicitly seeded as an isolated node.",
            "",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_spread_simulation_pipeline(proj_dir: Path = PROJ) -> Dict[str, Any]:
    """Execute the complete Phase 10 District Spread Propagation and Scenario Engine."""
    data_out = proj_dir / "data" / "scenarios"
    reports_out = proj_dir / "reports" / "spread_simulation"
    data_out.mkdir(parents=True, exist_ok=True)
    reports_out.mkdir(parents=True, exist_ok=True)

    # 1. Inspect existing geographic & mobility information
    audit_info = inspect_spread_simulation_assets(proj_dir)

    # 2. Build the 36-district graph
    graph = build_district_graph(proj_dir)

    # 3. Run the 6 benchmark scenarios
    benchmark_params = build_default_benchmark_scenarios()
    all_results = [run_scenario_simulation(graph, p) for p in benchmark_params]

    # 4. Run parameter sensitivity grid (108 combinations)
    sens_df = run_parameter_sensitivity_grid(graph)

    # 5. Consolidate outputs and save to data/scenarios/ and reports/spread_simulation/
    all_step_df = pd.concat([r["step_records_df"] for r in all_results], ignore_index=True)

    summary_rows = []
    for r in all_results:
        s = r["summary"]
        p = s["scenario_parameters"]
        summary_rows.append(
            {
                "scenario_id": s["scenario_id"],
                "scenario_name": s["scenario_name"],
                "initial_district": s["initial_district"],
                "scenario_duration_steps": p["scenario_duration_steps"],
                "hypothetical_propagation_factor": p["hypothetical_propagation_factor"],
                "within_district_persistence_factor": p["within_district_persistence_factor"],
                "neighbour_influence_decay_km": p["neighbour_influence_decay_km"],
                "edge_weighting_mode": p["edge_weighting_mode"],
                "intervention_start_step": p["intervention_start_step"],
                "intervention_response_coverage": p["intervention_response_coverage"],
                "activation_threshold": p["activation_threshold"],
                "affected_district_sequence": s["affected_district_sequence"],
                "total_districts_ever_affected": s["total_districts_ever_affected"],
                "initial_step0_signal_sum": s["initial_step0_signal_sum"],
                "peak_step": s["peak_step"],
                "peak_statewide_signal_sum": s["peak_statewide_signal_sum"],
                "final_step_signal_sum": s["final_step_signal_sum"],
                "final_step_active_districts": s["final_step_active_districts"],
                "assumptions": s["assumptions"],
                "scenario_limitations": s["scenario_limitations"],
                "non_prediction_disclaimer": s["non_prediction_disclaimer"],
                "model_version": s["model_version"],
                "simulation_timestamp": s["simulation_timestamp"],
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    csv_exports = {
        "district_graph_nodes.csv": graph.nodes_df,
        "district_graph_edges.csv": graph.edges_df,
        "scenario_summary_catalog.csv": summary_df,
        "scenario_simulation_step_records.csv": all_step_df,
        "parameter_sensitivity_grid.csv": sens_df,
    }
    for fname, df_obj in csv_exports.items():
        df_obj.to_csv(data_out / fname, index=False)
        df_obj.to_csv(reports_out / fname, index=False)

    full_json_payload = {
        "engine_version": ENGINE_VERSION,
        "generated_at": datetime.now().isoformat(),
        "non_prediction_disclaimer": MANDATORY_SCENARIO_DISCLAIMER,
        "asset_audit": audit_info,
        "scenarios": [r["summary"] for r in all_results],
    }
    for target_dir in (data_out, reports_out):
        with open(target_dir / "scenario_runs_metadata.json", "w", encoding="utf-8") as f:
            json.dump(full_json_payload, f, indent=2)

    # 6. Generate static figures + interactive HTML simulator
    viz_info = generate_all_spread_visualizations(
        graph, all_results, sens_df, reports_dir=reports_out, data_dir=data_out
    )

    # 7. Write Markdown report
    report_md = reports_out / "spread_simulation_report.md"
    _write_spread_simulation_report(
        audit_info, graph, all_results, sens_df, viz_info, report_md
    )
    shutil.copy2(report_md, proj_dir / "data" / "reports" / "spread_simulation_report.md")

    # 8. Update model_dataset_registry.json
    phase10_reg = {
        "engine_name": "District_Spread_Propagation_And_Scenario_Engine",
        "engine_version": ENGINE_VERSION,
        "generated_at": datetime.now().isoformat(),
        "non_prediction_disclaimer": MANDATORY_SCENARIO_DISCLAIMER,
        "graph_topology": {
            "total_nodes": 36,
            "connected_geocoded_nodes": 35,
            "isolated_ungeocoded_nodes": ["Mumbai Suburban"],
            "undirected_edges": int(len(graph.edges_df) // 2),
            "directed_arcs": int(len(graph.edges_df)),
            "mobility_data_in_workspace": False,
            "road_network_in_workspace": False,
        },
        "benchmark_scenarios_count": len(all_results),
        "sensitivity_grid_runs": len(sens_df),
        "output_files": {
            "nodes_csv": "data/scenarios/district_graph_nodes.csv",
            "edges_csv": "data/scenarios/district_graph_edges.csv",
            "scenario_catalog_csv": "data/scenarios/scenario_summary_catalog.csv",
            "step_records_csv": "data/scenarios/scenario_simulation_step_records.csv",
            "sensitivity_grid_csv": "data/scenarios/parameter_sensitivity_grid.csv",
            "metadata_json": "data/scenarios/scenario_runs_metadata.json",
            "interactive_simulator_html": "reports/spread_simulation/interactive_spread_scenario_simulator.html",
            "report_md": "reports/spread_simulation/spread_simulation_report.md",
            "figures": [
                str(Path(fp).relative_to(proj_dir).as_posix())
                for fp in viz_info["static_figures"]
            ],
        },
    }

    for reg_path in [
        proj_dir / "model_dataset_registry.json",
        proj_dir / "reports" / "model_dataset_registry.json",
    ]:
        if reg_path.exists():
            with open(reg_path, "r", encoding="utf-8") as f:
                reg = json.load(f)
        else:
            reg = {}
        reg["phase10_spread_simulation_engine"] = phase10_reg
        with open(reg_path, "w", encoding="utf-8") as f:
            json.dump(reg, f, indent=2)

    return {
        "engine_version": ENGINE_VERSION,
        "scenarios_executed": len(all_results),
        "sensitivity_runs": len(sens_df),
        "static_figures_count": len(viz_info["static_figures"]),
        "interactive_html": viz_info["interactive_html"],
        "report_path": str(report_md),
    }


if __name__ == "__main__":
    res = run_spread_simulation_pipeline()
    print(
        f"Phase 10 Spread Simulation Engine completed: "
        f"{res['scenarios_executed']} scenarios, {res['sensitivity_runs']} sensitivity runs, "
        f"{res['static_figures_count']} figures + interactive HTML simulator."
    )
