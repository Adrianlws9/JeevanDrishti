"""Static & Interactive Visualization Suite for Phase 10 District Spread Scenario Engine.

Generates:
1. Five high-resolution static PNG figures in `reports/spread_simulation/figures/` and `data/scenarios/figures/`:
   - `01_maharashtra_district_adjacency_graph.png`
   - `02_propagation_sequence_snapshots.png`
   - `03_scenario_timeline_and_affected_sequence.png`
   - `04_before_after_intervention_comparison.png`
   - `05_parameter_sensitivity_and_assumptions_panel.png`
2. One standalone interactive HTML/SVG/JS application (`interactive_spread_scenario_simulator.html`)
   featuring an Interactive District Graph, Step-by-Step Propagation Animation, Parameter Adjustment
   Panel, Scenario Timeline, Before-and-After Comparison, and Assumptions Panel.
"""
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.engines.geospatial.visualizations import SHORT_CODE_MAP
from src.engines.spread_simulation.district_graph import (
    MANDATORY_SCENARIO_DISCLAIMER,
    DistrictGraph,
)


DIVISION_COLORS: Dict[str, str] = {
    "Konkan": "#0284C7",
    "Pune": "#059669",
    "Nashik": "#D97706",
    "Marathwada": "#7C3AED",
    "Amravati_Vidarbha_West": "#DB2777",
    "Nagpur_Vidarbha_East": "#DC2626",
}


def plot_01_district_adjacency_graph(
    graph: DistrictGraph, out_path: Path
) -> Path:
    """Figure 1: Maharashtra 36-District Graph Structure & Edge Distance Weights."""
    fig, ax = plt.subplots(figsize=(13.5, 8.2), dpi=160)
    fig.suptitle(
        "JEEVAN-DRISHTI Phase 10: Maharashtra District Graph Topology (35 Geocoded Nodes + 83 Validated Undirected Edges)\n"
        f"[{MANDATORY_SCENARIO_DISCLAIMER}]",
        fontsize=11,
        fontweight="bold",
        y=0.98,
    )

    nodes = graph.nodes_df
    geo_nodes = nodes[nodes["has_verified_coordinates"]].copy()

    # Draw undirected edges colored by Haversine distance
    n = len(graph.districts)
    for i in range(n):
        for j in range(i + 1, n):
            if graph.adj_binary[i, j] > 0:
                ri = nodes.iloc[i]
                rj = nodes.iloc[j]
                dist_km = float(graph.adj_distance_km[i, j])
                same_div = bool(graph.adj_same_division[i, j] > 0)
                col = "#475569" if same_div else "#94A3B8"
                ls = "-" if same_div else "--"
                lw = max(0.8, 2.4 - dist_km / 110.0)
                ax.plot(
                    [float(ri["longitude"]), float(rj["longitude"])],
                    [float(ri["latitude"]), float(rj["latitude"])],
                    color=col,
                    linestyle=ls,
                    linewidth=lw,
                    alpha=0.72,
                    zorder=2,
                )

    for div_name, div_col in DIVISION_COLORS.items():
        sub = geo_nodes[geo_nodes["administrative_division"] == div_name]
        ax.scatter(
            sub["longitude"],
            sub["latitude"],
            s=95 + sub["graph_degree"] * 22,
            color=div_col,
            edgecolors="#0F172A",
            linewidths=1.2,
            label=f"{div_name} ({len(sub)} geocoded)",
            zorder=4,
        )

    for _, r in geo_nodes.iterrows():
        code = SHORT_CODE_MAP.get(str(r["district_name"]), str(r["district_name"])[:3].upper())
        deg = int(r["graph_degree"])
        ax.annotate(
            f"{code} (d={deg})",
            (float(r["longitude"]), float(r["latitude"])),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7.3,
            fontweight="bold",
            color="#0F172A",
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.78),
            zorder=5,
        )

    ax.text(
        0.02,
        0.03,
        "Graph Audit & Non-Fabrication Notes:\n"
        "• Nodes: 36 Official Maharashtra Districts (35 Geocoded, 1 Isolated)\n"
        "• Isolated Node: Mumbai Suburban (MH_D19, degree=0 — No Coordinates in Workspace)\n"
        "• Solid Edge = Intra-Division Adjacency | Dashed Edge = Cross-Division Adjacency\n"
        "• Mobility Data: UNAVAILABLE in workspace (Geometric Proximity Weights Only)",
        transform=ax.transAxes,
        fontsize=8.2,
        bbox=dict(boxstyle="round,pad=0.45", fc="#FEF3C7", ec="#D97706", lw=1.1),
        zorder=6,
    )

    ax.set_xlim(72.1, 81.1)
    ax.set_ylim(15.6, 22.1)
    ax.set_xlabel("Longitude (°E)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Latitude (°N)", fontsize=9.5, fontweight="bold")
    ax.set_title(
        "Validated Symmetric k-NN (k=4) District Graph Colored by Administrative Division & Node Degree",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=8.2, framealpha=0.94, title="Administrative Division")
    ax.grid(True, linestyle=":", alpha=0.38, color="#94A3B8")
    ax.set_facecolor("#F8FAFC")

    fig.tight_layout(rect=[0, 0.01, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_02_propagation_sequence_snapshots(
    graph: DistrictGraph,
    scen01_res: Dict[str, Any],
    out_path: Path,
) -> Path:
    """Figure 2: Step-by-Step Hypothetical Propagation Wavefront Snapshots (Steps 0, 2, 5, 10)."""
    step_df = scen01_res["step_records_df"]
    snap_steps = [0, 2, 5, 10]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12.2), dpi=160)
    fig.suptitle(
        "JEEVAN-DRISHTI Phase 10: Hypothetical Signal Propagation Sequence — SCEN_01 (Seed: Gadchiroli = 85.0)\n"
        "[WARNING: Dimensionless Hypothetical Signal Index (0–100) Under Distance-Decayed Adjacency — NOT Observed Cases]",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )

    nodes = graph.nodes_df
    n = len(graph.districts)

    for idx, t_step in enumerate(snap_steps):
        ax = axes[idx // 2, idx % 2]
        for i in range(n):
            for j in range(i + 1, n):
                if graph.adj_binary[i, j] > 0:
                    ri, rj = nodes.iloc[i], nodes.iloc[j]
                    ax.plot(
                        [float(ri["longitude"]), float(rj["longitude"])],
                        [float(ri["latitude"]), float(rj["latitude"])],
                        color="#CBD5E1",
                        linewidth=0.9,
                        zorder=1,
                    )

        sub = step_df[
            (step_df["simulation_step"] == t_step) & (step_df["latitude"].notna())
        ].copy()
        active = sub[sub["is_affected_above_threshold"]]
        inactive = sub[~sub["is_affected_above_threshold"]]

        if not inactive.empty:
            ax.scatter(
                inactive["longitude"],
                inactive["latitude"],
                marker="o",
                s=65,
                color="#E2E8F0",
                edgecolors="#64748B",
                linewidths=0.9,
                label="Below Activation Threshold (<5.0) [○]",
                zorder=3,
            )

        if not active.empty:
            sc = ax.scatter(
                active["longitude"],
                active["latitude"],
                marker="^",
                c=active["simulated_signal_index"],
                vmin=0.0,
                vmax=90.0,
                s=110 + active["simulated_signal_index"] * 2.2,
                cmap="YlOrRd",
                edgecolors="#0F172A",
                linewidths=1.2,
                label="Hypothetically Affected (>=5.0) [▲]",
                zorder=4,
            )
            cbar = fig.colorbar(sc, ax=ax, fraction=0.042, pad=0.02)
            cbar.set_label("Simulated Signal Index (0–100)", fontsize=8)

            for _, r in active.iterrows():
                code = SHORT_CODE_MAP.get(str(r["district_name"]), str(r["district_name"])[:3].upper())
                val = float(r["simulated_signal_index"])
                f_step = int(r["first_activation_step"])
                ax.annotate(
                    f"{code}:{val:.1f}(t{f_step})",
                    (float(r["longitude"]), float(r["latitude"])),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7.0,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.76),
                    zorder=5,
                )

        int_status = "INTERVENTION ACTIVE (30% dampening)" if t_step >= 8 else "PRE-INTERVENTION"
        ax.set_xlim(72.1, 81.1)
        ax.set_ylim(15.6, 22.1)
        ax.set_xlabel("Longitude (°E)", fontsize=9)
        ax.set_ylabel("Latitude (°N)", fontsize=9)
        ax.set_title(
            f"Simulation Step t = {t_step} [{int_status}]\n"
            f"Affected Districts (>=5.0): {len(active)} / 35 Geocoded | Signal Sum = {sub['simulated_signal_index'].sum():.1f}",
            fontsize=10,
            fontweight="bold",
        )
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.set_facecolor("#F8FAFC")

    fig.tight_layout(rect=[0, 0.01, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_03_scenario_timeline_and_affected_sequence(
    all_results: List[Dict[str, Any]], out_path: Path
) -> Path:
    """Figure 3: Scenario Timelines Across All 6 Benchmark Scenarios."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), dpi=160)
    fig.suptitle(
        "JEEVAN-DRISHTI Phase 10: Scenario Timeline — Simulated Signal Burden & Affected District Count (Steps 0..12)\n"
        f"[{MANDATORY_SCENARIO_DISCLAIMER}]",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )

    palette = ["#DC2626", "#16A34A", "#0284C7", "#7C3AED", "#D97706", "#0D9488"]
    markers = ["o", "s", "^", "D", "v", "P"]

    ax1, ax2 = axes[0], axes[1]
    for res, col, mk in zip(all_results, palette, markers):
        s = res["summary"]
        wf = pd.DataFrame(s["wavefront_timeline"])
        lbl = f"{s['scenario_id']} (int@{s['scenario_parameters']['intervention_start_step']}, cov={s['scenario_parameters']['intervention_response_coverage']:.0%})"
        ax1.plot(
            wf["simulation_step"],
            wf["sum_simulated_signal"],
            color=col,
            marker=mk,
            linewidth=2.0,
            markersize=6,
            label=lbl,
        )
        ax2.plot(
            wf["simulation_step"],
            wf["active_districts_count"],
            color=col,
            marker=mk,
            linewidth=2.0,
            markersize=6,
            label=lbl,
        )

    ax1.set_xlabel("Hypothetical Simulation Step t (Weeks)", fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Statewide Sum of Simulated Signal Index", fontsize=9.5, fontweight="bold")
    ax1.set_title("A. Statewide Simulated Signal Trajectory by Scenario", fontsize=10.5, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.45)
    ax1.legend(loc="upper left", fontsize=7.6, framealpha=0.92)

    ax2.set_xlabel("Hypothetical Simulation Step t (Weeks)", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Number of Districts Above Activation Threshold (>=5.0)", fontsize=9.5, fontweight="bold")
    ax2.set_title("B. Wavefront Expansion — Active Districts Count by Step", fontsize=10.5, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.45)
    ax2.legend(loc="upper left", fontsize=7.6, framealpha=0.92)

    fig.tight_layout(rect=[0, 0.01, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_04_before_after_intervention_comparison(
    all_results: List[Dict[str, Any]], out_path: Path
) -> Path:
    """Figure 4: Before-and-After Intervention Timing & Response Coverage Comparison."""
    res_map = {r["summary"]["scenario_id"]: r for r in all_results}
    s1 = res_map["SCEN_01_EAST_VIDARBHA_BASELINE"]["step_records_df"]
    s2 = res_map["SCEN_02_EAST_VIDARBHA_EARLY_INTERVENTION"]["step_records_df"]
    s5 = res_map["SCEN_05_MARATHWADA_HIGH_PROPAGATION_UNMITIGATED"]["step_records_df"]
    s6 = res_map["SCEN_06_MARATHWADA_RAPID_CONTAINMENT"]["step_records_df"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.0), dpi=160)
    fig.suptitle(
        "JEEVAN-DRISHTI Phase 10: Before-and-After Intervention Comparison (Delayed/Unmitigated vs. Early Containment)\n"
        "[Illustrates How Earlier Intervention Step & Higher Coverage Dampen Graph Signal Propagation]",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )

    # Panel A: Vidarbha Final Step (t=12) Top 10 Affected Districts Comparison
    ax1 = axes[0]
    s1_f = s1[s1["simulation_step"] == 12].set_index("district_name")["simulated_signal_index"]
    s2_f = s2[s2["simulation_step"] == 12].set_index("district_name")["simulated_signal_index"]
    top_vid = s1_f.sort_values(ascending=False).head(10).index.tolist()[::-1]

    y_pos = np.arange(len(top_vid))
    ax1.barh(
        y_pos + 0.18,
        [s1_f[d] for d in top_vid],
        height=0.35,
        color="#DC2626",
        label="SCEN_01: Delayed Response (step 8, 30% cov)",
    )
    ax1.barh(
        y_pos - 0.18,
        [s2_f[d] for d in top_vid],
        height=0.35,
        color="#16A34A",
        label="SCEN_02: Early Response (step 2, 70% cov)",
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_vid, fontsize=8.5, fontweight="bold")
    ax1.set_xlabel("Final Step (t=12) Simulated Signal Index (0–100)", fontsize=9.5, fontweight="bold")
    ax1.set_title(
        "A. Eastern Vidarbha (Seed: Gadchiroli=85)\nDelayed vs. Early Intervention at Step t=12",
        fontsize=10.5,
        fontweight="bold",
    )
    ax1.legend(loc="lower right", fontsize=8.5)
    ax1.grid(axis="x", linestyle=":", alpha=0.45)

    # Panel B: Marathwada Final Step (t=12) Top 10 Affected Districts Comparison
    ax2 = axes[1]
    s5_f = s5[s5["simulation_step"] == 12].set_index("district_name")["simulated_signal_index"]
    s6_f = s6[s6["simulation_step"] == 12].set_index("district_name")["simulated_signal_index"]
    top_mar = s5_f.sort_values(ascending=False).head(10).index.tolist()[::-1]

    y_pos2 = np.arange(len(top_mar))
    ax2.barh(
        y_pos2 + 0.18,
        [s5_f[d] for d in top_mar],
        height=0.35,
        color="#D97706",
        label="SCEN_05: Unmitigated Stress Test (0% cov)",
    )
    ax2.barh(
        y_pos2 - 0.18,
        [s6_f[d] for d in top_mar],
        height=0.35,
        color="#0284C7",
        label="SCEN_06: Rapid Containment (step 3, 65% cov)",
    )
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(top_mar, fontsize=8.5, fontweight="bold")
    ax2.set_xlabel("Final Step (t=12) Simulated Signal Index (0–100)", fontsize=9.5, fontweight="bold")
    ax2.set_title(
        "B. Marathwada Stress Test (Seed: Nanded=90)\nUnmitigated vs. Rapid Containment at Step t=12",
        fontsize=10.5,
        fontweight="bold",
    )
    ax2.legend(loc="lower right", fontsize=8.5)
    ax2.grid(axis="x", linestyle=":", alpha=0.45)

    fig.tight_layout(rect=[0, 0.01, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_05_parameter_sensitivity_and_assumptions_panel(
    sens_df: pd.DataFrame, out_path: Path
) -> Path:
    """Figure 5: Parameter Adjustment Sensitivity & Explicit Assumptions Panel."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), dpi=160)
    fig.suptitle(
        "JEEVAN-DRISHTI Phase 10: Parameter Sensitivity Matrix & Explicit Scenario Assumptions Panel\n"
        f"[{MANDATORY_SCENARIO_DISCLAIMER}]",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )

    # Panel A: Grouped bar sensitivity of Final Step Signal Sum across beta_prop and eta_cov (distance_decayed, t_int=2)
    ax1 = axes[0]
    sub = sens_df[
        (sens_df["edge_weighting_mode"] == "distance_decayed")
        & (sens_df["intervention_start_step"] == 2)
    ].copy()
    betas = sorted(sub["hypothetical_propagation_factor"].unique())
    etas = sorted(sub["intervention_response_coverage"].unique())
    x = np.arange(len(betas))
    width = 0.25
    cols = ["#DC2626", "#F59E0B", "#16A34A"]

    for idx, (eta, col) in enumerate(zip(etas, cols)):
        vals = [
            float(
                sub[
                    (sub["hypothetical_propagation_factor"] == b)
                    & (sub["intervention_response_coverage"] == eta)
                ]["peak_statewide_signal_sum"].iloc[0]
            )
            for b in betas
        ]
        ax1.bar(
            x + (idx - 1) * width,
            vals,
            width=width,
            color=col,
            edgecolor="#0F172A",
            label=f"Response Coverage η = {eta:.0%}",
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"β_prop = {b:.2f}" for b in betas], fontsize=9.5, fontweight="bold")
    ax1.set_ylabel("Peak Statewide Simulated Signal Sum", fontsize=9.5, fontweight="bold")
    ax1.set_title(
        "A. Parameter Sensitivity (Seed: Pune=80, Intervention at Step 2)\nEffect of Propagation Factor β_prop & Response Coverage η",
        fontsize=10.5,
        fontweight="bold",
    )
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(axis="y", linestyle=":", alpha=0.45)

    # Panel B: Structured Assumptions & Scientific Guardrails Card
    ax2 = axes[1]
    ax2.axis("off")
    card_text = (
        "EXPLICIT SCENARIO ENGINE ASSUMPTIONS & SCIENTIFIC BOUNDARIES\n"
        "=============================================================\n\n"
        "1. GRAPH TOPOLOGY & ADJACENCY SOURCE:\n"
        "   • 36 Maharashtra districts represented as nodes (MH_D01..MH_D36).\n"
        "   • 35 geocoded districts connected via 83 validated symmetric k-NN (k=4)\n"
        "     edges derived from verified district coordinates in district_grid_cells.csv.\n"
        "   • Mumbai Suburban (MH_D19) has NO coordinates in the workspace and is\n"
        "     kept as an isolated node (0 edges) — zero connections are invented.\n\n"
        "2. MOBILITY & ROAD-NETWORK DATA LIMITATION:\n"
        "   • Zero human mobility, commuter flow, or highway traffic datasets exist\n"
        "     in the project workspace.\n"
        "   • Equal movement between neighbours is NOT assumed. Edge weights use\n"
        "     transparent geometric decay exp(-d_km / λ_km) or division attenuation.\n\n"
        "3. TRANSPARENT GRAPH CALCULATOR vs. BIOLOGICAL COMPARTMENTAL MODELS:\n"
        "   • Because population denominators (N_i) and pathogen R_0 / serial intervals\n"
        "     are absent from the workspace, no arbitrary SEIR medical parameters\n"
        "     are fabricated.\n"
        "   • Simulated values (0–100) represent a hypothetical stress index and are\n"
        "     NEVER mixed with observed IDSP cases or Phase 6 case forecasts.\n\n"
        "4. MANDATORY INTERPRETATION RULE:\n"
        "   • Every simulation output is a 'what-if' planning exploration under\n"
        "     user-selected parameters, NEVER a confirmed outbreak-spread prediction."
    )
    ax2.text(
        0.02,
        0.96,
        card_text,
        transform=ax2.transAxes,
        va="top",
        ha="left",
        fontsize=8.6,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.6", fc="#F8FAFC", ec="#0F172A", lw=1.4),
    )

    fig.tight_layout(rect=[0, 0.01, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_interactive_html_simulator(
    graph: DistrictGraph,
    all_results: List[Dict[str, Any]],
    out_path: Path,
) -> Path:
    """Create a self-contained interactive HTML/SVG/JS District Spread & Scenario Simulator."""
    nodes_payload = []
    for _, r in graph.nodes_df.iterrows():
        nodes_payload.append(
            {
                "idx": int(r["node_index"]),
                "id": str(r["district_id"]),
                "name": str(r["district_name"]),
                "code": SHORT_CODE_MAP.get(str(r["district_name"]), str(r["district_name"])[:3].upper()),
                "division": str(r["administrative_division"]),
                "lat": float(r["latitude"]) if pd.notna(r["latitude"]) else None,
                "lon": float(r["longitude"]) if pd.notna(r["longitude"]) else None,
                "degree": int(r["graph_degree"]),
                "obs_cases_ref": float(r["observed_2023_2025_total_cases_ref"]) if pd.notna(r["observed_2023_2025_total_cases_ref"]) else None,
            }
        )

    edges_payload = []
    n = len(graph.districts)
    for i in range(n):
        for j in range(i + 1, n):
            if graph.adj_binary[i, j] > 0:
                edges_payload.append(
                    {
                        "u": i,
                        "v": j,
                        "dist_km": round(float(graph.adj_distance_km[i, j]), 2),
                        "same_div": bool(graph.adj_same_division[i, j] > 0),
                    }
                )

    presets_payload = [r["summary"]["scenario_parameters"] for r in all_results]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>JEEVAN-DRISHTI Phase 10 — Interactive District Spread Scenario Simulator</title>
<style>
  :root {{
    --bg: #0f172a; --panel: #1e293b; --border: #334155; --text: #f8fafc; --muted: #94a3b8;
    --accent: #38bdf8; --warn: #f59e0b; --danger: #ef4444; --ok: #22c55e;
  }}
  * {{ box-sizing: border-box; font-family: system-ui, -apple-system, sans-serif; }}
  body {{ margin: 0; background: var(--bg); color: var(--text); padding: 16px; }}
  .banner {{
    background: #451a03; border: 1px solid var(--warn); color: #fef3c7;
    padding: 10px 14px; border-radius: 8px; margin-bottom: 14px; font-size: 13px; font-weight: 600;
  }}
  .grid {{ display: grid; grid-template-columns: 340px 1fr 360px; gap: 14px; }}
  .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }}
  h2, h3 {{ margin: 0 0 10px 0; font-size: 15px; color: var(--accent); }}
  label {{ display: block; font-size: 12px; color: var(--muted); margin-top: 8px; }}
  select, input[type=range], button {{ width: 100%; margin-top: 4px; }}
  select, button {{
    background: #0f172a; color: var(--text); border: 1px solid var(--border);
    padding: 7px 10px; border-radius: 6px; font-size: 12.5px; cursor: pointer;
  }}
  button:hover {{ border-color: var(--accent); }}
  .btn-row {{ display: flex; gap: 6px; margin-top: 10px; }}
  .val-badge {{ float: right; color: var(--text); font-weight: 700; }}
  svg {{ width: 100%; height: 470px; background: #0b1120; border-radius: 8px; border: 1px solid var(--border); }}
  .seq-box {{
    max-height: 155px; overflow-y: auto; font-size: 12px; line-height: 1.45;
    background: #0b1120; padding: 8px; border-radius: 6px; border: 1px solid var(--border);
  }}
  .assumptions {{ font-size: 11.5px; color: var(--muted); line-height: 1.45; }}
  .bar-row {{ display: flex; align-items: center; gap: 6px; font-size: 11.5px; margin: 4px 0; }}
  .bar-fill {{ height: 10px; border-radius: 3px; }}
</style>
</head>
<body>
<div class="banner">
  ⚠️ {MANDATORY_SCENARIO_DISCLAIMER}
</div>

<div class="grid">
  <!-- Parameter Adjustment Panel -->
  <div class="card">
    <h2>1. Parameter Adjustment Panel</h2>
    <label>Preset Benchmark Scenario</label>
    <select id="presetSelect"></select>

    <label>Initial Seed District (or click node on map)</label>
    <select id="seedDistrictSelect"></select>

    <label>Initial Seed Signal (0–100) <span id="seedValLbl" class="val-badge">85</span></label>
    <input type="range" id="seedValInput" min="10" max="100" step="5" value="85">

    <label>Hypothetical Propagation Factor (β_prop) <span id="betaLbl" class="val-badge">0.38</span></label>
    <input type="range" id="betaInput" min="0.05" max="0.80" step="0.01" value="0.38">

    <label>Within-District Persistence (ρ_persist) <span id="rhoLbl" class="val-badge">0.82</span></label>
    <input type="range" id="rhoInput" min="0.20" max="0.95" step="0.02" value="0.82">

    <label>Neighbour Distance Decay Scale (λ km) <span id="decayLbl" class="val-badge">120 km</span></label>
    <input type="range" id="decayInput" min="40" max="300" step="10" value="120">

    <label>Edge Weighting Assumption</label>
    <select id="modeSelect">
      <option value="distance_decayed">Distance-Decayed Geometric Adjacency</option>
      <option value="inverse_distance">Inverse-Distance (1 / d_km)</option>
      <option value="division_bounded">Division-Bounded Adjacency</option>
      <option value="uniform_topological">Uniform Topological Degree</option>
    </select>

    <label>Intervention Start Step (t_int) <span id="tintLbl" class="val-badge">Step 8</span></label>
    <input type="range" id="tintInput" min="1" max="12" step="1" value="8">

    <label>Intervention Response Coverage (η_cov) <span id="etaLbl" class="val-badge">30%</span></label>
    <input type="range" id="etaInput" min="0" max="0.90" step="0.05" value="0.30">

    <hr style="border-color:var(--border);margin:12px 0;">
    <h3>2. Assumptions & Non-Prediction Panel</h3>
    <div class="assumptions">
      • <b>No Mobility Fabrication</b>: Human mobility & road traffic data are absent from the workspace.<br>
      • <b>Ungeocoded Node</b>: <code>Mumbai Suburban (MH_D19)</code> has no coordinates and 0 edges.<br>
      • <b>Layer Separation</b>: Simulated signal index (0–100) is never mixed with observed IDSP case counts.
    </div>
  </div>

  <!-- Interactive District Graph & Animation Panel -->
  <div class="card">
    <h2>3. Interactive Maharashtra District Graph & Propagation Animation</h2>
    <div class="btn-row" style="margin-bottom:8px;">
      <button id="playBtn">▶ Play Animation</button>
      <button id="pauseBtn">⏸ Pause</button>
      <button id="resetBtn">⏮ Reset (Step 0)</button>
    </div>
    <label>Animation Step Slider: <span id="stepLbl" class="val-badge">Step 0 / 12</span></label>
    <input type="range" id="stepSlider" min="0" max="12" step="1" value="0" style="margin-bottom:8px;">
    <svg id="graphSvg" viewBox="0 0 780 460"></svg>
    <div id="hoverInfo" style="font-size:12px;color:var(--muted);margin-top:6px;">
      Hover over any district node to view simulated signal vs. historical observed reference (strictly separated).
    </div>
  </div>

  <!-- Scenario Timeline & Before/After Panel -->
  <div class="card">
    <h2>4. Scenario Timeline (Steps 0..12)</h2>
    <svg id="timelineSvg" viewBox="0 0 320 165" style="height:165px;margin-bottom:10px;"></svg>

    <h3>5. Affected District Wavefront Sequence (≥5.0)</h3>
    <div id="seqBox" class="seq-box"></div>

    <h3 style="margin-top:12px;">6. Before vs. After Intervention (Step 12 Top Districts)</h3>
    <div id="beforeAfterBox" style="max-height:175px;overflow-y:auto;"></div>
  </div>
</div>

<script>
const NODES = {json.dumps(nodes_payload)};
const EDGES = {json.dumps(edges_payload)};
const PRESETS = {json.dumps(presets_payload)};

let animTimer = null;
let currentSim = null;

function lonToX(lon) {{ return 45 + ((lon - 72.5) / (80.6 - 72.5)) * 690; }}
function latToY(lat) {{ return 420 - ((lat - 16.0) / (21.8 - 16.0)) * 375; }}

function buildWeights(mode, decayKm) {{
  const n = NODES.length;
  const W = Array.from({{length: n}}, () => Array(n).fill(0));
  EDGES.forEach(e => {{
    let w = 1.0;
    if (mode === 'distance_decayed') w = Math.exp(-e.dist_km / decayKm);
    else if (mode === 'inverse_distance') w = 1.0 / Math.max(e.dist_km, 1.0);
    else if (mode === 'division_bounded') w = Math.exp(-e.dist_km / decayKm) * (e.same_div ? 1.0 : 0.5);
    W[e.u][e.v] = w;
    W[e.v][e.u] = w;
  }});
  for (let i = 0; i < n; i++) {{
    const s = W[i].reduce((a, b) => a + b, 0);
    if (s > 0) for (let j = 0; j < n; j++) W[i][j] /= s;
  }}
  return W;
}}

function simulateRun(seeds, T, beta, rho, decayKm, mode, tInt, eta) {{
  const n = NODES.length;
  const W = buildWeights(mode, decayKm);
  const S = Array.from({{length: T + 1}}, () => Array(n).fill(0));
  const firstStep = Array(n).fill(null);

  Object.entries(seeds).forEach(([dName, val]) => {{
    const nd = NODES.find(x => x.name === dName);
    if (nd) {{ S[0][nd.idx] = val; if (val >= 5.0) firstStep[nd.idx] = 0; }}
  }});

  for (let t = 0; t < T; t++) {{
    const mInt = (t + 1) >= tInt ? (1.0 - eta) : 1.0;
    for (let i = 0; i < n; i++) {{
      const p_i = rho * mInt * S[t][i];
      let inc = 0;
      for (let j = 0; j < n; j++) inc += W[i][j] * S[t][j];
      const i_i = beta * mInt * inc;
      const nextS = Math.min(100.0, p_i + i_i * (1.0 - S[t][i] / 100.0));
      S[t + 1][i] = nextS;
      if (firstStep[i] === null && nextS >= 5.0) firstStep[i] = t + 1;
    }}
  }}
  return {{ S, firstStep }};
}}

function signalColor(val) {{
  if (val < 5.0) return '#334155';
  if (val < 20.0) return '#facc15';
  if (val < 45.0) return '#f97316';
  return '#ef4444';
}}

function updateSimulation() {{
  const seedDist = document.getElementById('seedDistrictSelect').value;
  const seedVal = parseFloat(document.getElementById('seedValInput').value);
  const beta = parseFloat(document.getElementById('betaInput').value);
  const rho = parseFloat(document.getElementById('rhoInput').value);
  const decay = parseFloat(document.getElementById('decayInput').value);
  const mode = document.getElementById('modeSelect').value;
  const tInt = parseInt(document.getElementById('tintInput').value);
  const eta = parseFloat(document.getElementById('etaInput').value);

  document.getElementById('seedValLbl').textContent = seedVal;
  document.getElementById('betaLbl').textContent = beta.toFixed(2);
  document.getElementById('rhoLbl').textContent = rho.toFixed(2);
  document.getElementById('decayLbl').textContent = decay + ' km';
  document.getElementById('tintLbl').textContent = 'Step ' + tInt;
  document.getElementById('etaLbl').textContent = Math.round(eta * 100) + '%';

  const seeds = {{ [seedDist]: seedVal }};
  const withInt = simulateRun(seeds, 12, beta, rho, decay, mode, tInt, eta);
  const noInt = simulateRun(seeds, 12, beta, rho, decay, mode, 12, 0.0);
  currentSim = {{ withInt, noInt, tInt }};
  renderStep(parseInt(document.getElementById('stepSlider').value));
  renderTimelineAndSummary();
}}

function renderStep(step) {{
  document.getElementById('stepLbl').textContent = `Step ${{step}} / 12`;
  const svg = document.getElementById('graphSvg');
  svg.innerHTML = '';
  const S_step = currentSim.withInt.S[step];

  EDGES.forEach(e => {{
    const u = NODES[e.u], v = NODES[e.v];
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', lonToX(u.lon)); line.setAttribute('y1', latToY(u.lat));
    line.setAttribute('x2', lonToX(v.lon)); line.setAttribute('y2', latToY(v.lat));
    line.setAttribute('stroke', '#334155');
    line.setAttribute('stroke-width', '1.2');
    svg.appendChild(line);
  }});

  NODES.forEach(nd => {{
    if (nd.lat === null) return;
    const x = lonToX(nd.lon), y = latToY(nd.lat);
    const val = S_step[nd.idx];
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.style.cursor = 'pointer';

    const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    c.setAttribute('cx', x); c.setAttribute('cy', y);
    c.setAttribute('r', 7 + val * 0.12);
    c.setAttribute('fill', signalColor(val));
    c.setAttribute('stroke', val >= 5.0 ? '#f8fafc' : '#64748b');
    c.setAttribute('stroke-width', '1.4');
    g.appendChild(c);

    const t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    t.setAttribute('x', x); t.setAttribute('y', y - 10);
    t.setAttribute('text-anchor', 'middle');
    t.setAttribute('fill', '#e2e8f0');
    t.setAttribute('font-size', '9.5px');
    t.setAttribute('font-weight', 'bold');
    t.textContent = `${{nd.code}}:${{val.toFixed(0)}}`;
    g.appendChild(t);

    g.addEventListener('mouseenter', () => {{
      const obsTxt = nd.obs_cases_ref !== null ? `${{nd.obs_cases_ref}} total cases (2023-25)` : 'Insufficient Data';
      document.getElementById('hoverInfo').innerHTML =
        `<b>${{nd.name}} (${{nd.id}})</b> | Simulated Signal (Step ${{step}}): <b>${{val.toFixed(2)}}</b> | Historical Observed Ref (Separate Layer): ${'{'}obsTxt{'}'}`;
    }});
    g.addEventListener('click', () => {{
      document.getElementById('seedDistrictSelect').value = nd.name;
      updateSimulation();
    }});
    svg.appendChild(g);
  }});
}}

function renderTimelineAndSummary() {{
  const tSvg = document.getElementById('timelineSvg');
  tSvg.innerHTML = '';
  const sumsInt = currentSim.withInt.S.map(row => row.reduce((a, b) => a + b, 0));
  const sumsNo = currentSim.noInt.S.map(row => row.reduce((a, b) => a + b, 0));
  const maxS = Math.max(...sumsNo, 100);

  const ptsInt = sumsInt.map((v, t) => `${{25 + t * 22}},${{145 - (v / maxS) * 120}}`).join(' ');
  const ptsNo = sumsNo.map((v, t) => `${{25 + t * 22}},${{145 - (v / maxS) * 120}}`).join(' ');

  tSvg.innerHTML = `
    <polyline fill="none" stroke="#ef4444" stroke-width="2" stroke-dasharray="4,3" points="${{ptsNo}}"/>
    <polyline fill="none" stroke="#22c55e" stroke-width="2.5" points="${{ptsInt}}"/>
    <text x="25" y="16" fill="#ef4444" font-size="10">--- Unmitigated (No Intervention)</text>
    <text x="185" y="16" fill="#22c55e" font-size="10">─── With Intervention</text>
  `;

  // Affected sequence
  const seqLines = [];
  for (let t = 0; t <= 12; t++) {{
    const newly = NODES.filter(nd => currentSim.withInt.firstStep[nd.idx] === t).map(nd => nd.name);
    if (newly.length > 0) seqLines.push(`<div><b>Step ${{t}}:</b> ${{newly.join(', ')}}</div>`);
  }}
  document.getElementById('seqBox').innerHTML = seqLines.join('') || 'None';

  // Before vs After at Step 12
  const s12No = currentSim.noInt.S[12];
  const s12Int = currentSim.withInt.S[12];
  const ranked = NODES.map(nd => ({{ name: nd.code, noVal: s12No[nd.idx], intVal: s12Int[nd.idx] }}))
    .sort((a, b) => b.noVal - a.noVal).slice(0, 8);

  document.getElementById('beforeAfterBox').innerHTML = ranked.map(r => `
    <div class="bar-row">
      <span style="width:36px;font-weight:700;">${{r.name}}</span>
      <div style="flex:1;">
        <div class="bar-fill" style="width:${{Math.min(100, r.noVal)}}%;background:#ef4444;" title="Without Intervention: ${{r.noVal.toFixed(1)}}"></div>
        <div class="bar-fill" style="width:${{Math.min(100, r.intVal)}}%;background:#22c55e;margin-top:2px;" title="With Intervention: ${{r.intVal.toFixed(1)}}"></div>
      </div>
      <span style="width:72px;text-align:right;">${{r.noVal.toFixed(0)}} → <b>${{r.intVal.toFixed(0)}}</b></span>
    </div>
  `).join('');
}}

// Initialize selects
const seedSel = document.getElementById('seedDistrictSelect');
NODES.filter(n => n.lat !== null).forEach(n => {{
  const opt = document.createElement('option');
  opt.value = n.name; opt.textContent = `${{n.name}} (${{n.id}})`;
  seedSel.appendChild(opt);
}});
seedSel.value = 'Gadchiroli';

const presetSel = document.getElementById('presetSelect');
PRESETS.forEach((p, idx) => {{
  const opt = document.createElement('option');
  opt.value = idx; opt.textContent = `${{p.scenario_id}}`;
  presetSel.appendChild(opt);
}});

presetSel.addEventListener('change', () => {{
  const p = PRESETS[parseInt(presetSel.value)];
  const firstSeed = Object.keys(p.initial_affected_districts)[0];
  seedSel.value = firstSeed;
  document.getElementById('seedValInput').value = p.initial_affected_districts[firstSeed];
  document.getElementById('betaInput').value = p.hypothetical_propagation_factor;
  document.getElementById('rhoInput').value = p.within_district_persistence_factor;
  document.getElementById('decayInput').value = p.neighbour_influence_decay_km;
  document.getElementById('modeSelect').value = p.edge_weighting_mode;
  document.getElementById('tintInput').value = p.intervention_start_step;
  document.getElementById('etaInput').value = p.intervention_response_coverage;
  updateSimulation();
}});

['seedDistrictSelect','seedValInput','betaInput','rhoInput','decayInput','modeSelect','tintInput','etaInput'].forEach(id => {{
  document.getElementById(id).addEventListener('input', updateSimulation);
}});

document.getElementById('stepSlider').addEventListener('input', e => renderStep(parseInt(e.target.value)));
document.getElementById('playBtn').addEventListener('click', () => {{
  if (animTimer) clearInterval(animTimer);
  animTimer = setInterval(() => {{
    const sl = document.getElementById('stepSlider');
    let next = (parseInt(sl.value) + 1) % 13;
    sl.value = next;
    renderStep(next);
  }}, 650);
}});
document.getElementById('pauseBtn').addEventListener('click', () => {{ if (animTimer) clearInterval(animTimer); }});
document.getElementById('resetBtn').addEventListener('click', () => {{
  if (animTimer) clearInterval(animTimer);
  document.getElementById('stepSlider').value = 0;
  renderStep(0);
}});

updateSimulation();
</script>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_content, encoding="utf-8")
    return out_path


def generate_all_spread_visualizations(
    graph: DistrictGraph,
    all_results: List[Dict[str, Any]],
    sens_df: pd.DataFrame,
    reports_dir: Path,
    data_dir: Path,
) -> Dict[str, Any]:
    """Generate all 5 static PNG figures and the interactive HTML simulator."""
    rep_figs = reports_dir / "figures"
    dat_figs = data_dir / "figures"
    rep_figs.mkdir(parents=True, exist_ok=True)
    dat_figs.mkdir(parents=True, exist_ok=True)

    f1 = plot_01_district_adjacency_graph(
        graph, rep_figs / "01_maharashtra_district_adjacency_graph.png"
    )
    f2 = plot_02_propagation_sequence_snapshots(
        graph, all_results[0], rep_figs / "02_propagation_sequence_snapshots.png"
    )
    f3 = plot_03_scenario_timeline_and_affected_sequence(
        all_results, rep_figs / "03_scenario_timeline_and_affected_sequence.png"
    )
    f4 = plot_04_before_after_intervention_comparison(
        all_results, rep_figs / "04_before_after_intervention_comparison.png"
    )
    f5 = plot_05_parameter_sensitivity_and_assumptions_panel(
        sens_df, rep_figs / "05_parameter_sensitivity_and_assumptions_panel.png"
    )

    static_figs = [f1, f2, f3, f4, f5]
    for fp in static_figs:
        shutil.copy2(fp, dat_figs / fp.name)

    html_rep = generate_interactive_html_simulator(
        graph, all_results, reports_dir / "interactive_spread_scenario_simulator.html"
    )
    shutil.copy2(html_rep, data_dir / "interactive_spread_scenario_simulator.html")

    return {
        "static_figures": [str(fp) for fp in static_figs],
        "interactive_html": str(html_rep),
    }
