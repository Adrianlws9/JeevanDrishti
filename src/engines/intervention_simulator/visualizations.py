"""Visualization & Interactive HTML Generator for Public-Health Intervention Simulator (Phase 14 Section 7).

Generates:
1. `01_project_capability_and_intervention_support_audit.png`
2. `02_baseline_vs_scenario_a_vs_scenario_b_timelines.png`
3. `03_district_wise_scenario_comparison_map.png`
4. `04_intervention_timing_and_assumption_sensitivity_grid.png`
5. `05_hypothetical_resource_allocation_and_bed_strain_comparison.png`
6. `interactive_intervention_scenario_simulator.html`
"""
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.engines.intervention_simulator.capability_inspector import (
    MANDATORY_INTERVENTION_GOVERNANCE_DISCLAIMER,
)
from src.engines.intervention_simulator.intervention_calculator import (
    InterventionScenarioConfig,
    simulate_intervention_scenario,
)


def generate_intervention_visualizations(
    capability_df: pd.DataFrame,
    pool_df: pd.DataFrame,
    adj_map: Dict[str, List[str]],
    scenario_summaries: List[Dict[str, Any]],
    weekly_cmp_df: pd.DataFrame,
    district_cmp_df: pd.DataFrame,
    fig_dir: Path,
    html_path: Path,
) -> Dict[str, str]:
    """Render all 5 static PNG figures and the standalone interactive HTML simulator."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Figure 1: Project Capability & Intervention Support Audit
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12.5, 6.2), dpi=150)
    labels = capability_df["upstream_engine"].tolist()[::-1]
    avail = capability_df["genuinely_available_in_workspace"].tolist()[::-1]
    roles = capability_df["role_in_intervention_simulator"].tolist()[::-1]

    colors = ["#1b7837" if a else "#d73027" for a in avail]
    vals = [1.0 if a else 0.18 for a in avail]
    y_pos = np.arange(len(labels))

    bars = ax.barh(y_pos, vals, color=colors, edgecolor="#222222", height=0.62, alpha=0.88)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9.5, fontweight="bold")
    ax.set_xlim(0, 1.38)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["Unavailable (0%)", "Partial", "Verified in Workspace (100%)"], fontsize=9)
    ax.set_title(
        "Phase 14 Capability Audit: Verified Upstream Engines vs. Empirical Intervention Data Gap\n"
        "(HYPOTHETICAL SCENARIO CALCULATOR — Zero Empirical Intervention Efficacy Claimed)",
        fontsize=11.5,
        fontweight="bold",
        pad=12,
    )

    for bar, is_av, role_txt in zip(bars, avail, roles):
        badge = "VERIFIED INPUT" if is_av else "NO RCT DATA -> USER ASSUMPTION"
        short_role = role_txt if len(role_txt) <= 68 else role_txt[:65] + "..."
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2.0,
            f"[{badge}] {short_role}",
            va="center",
            ha="left",
            fontsize=8.2,
            color="#111111" if is_av else "#990000",
            fontweight="semibold",
        )

    ax.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    f1 = fig_dir / "01_project_capability_and_intervention_support_audit.png"
    fig.savefig(f1, bbox_inches="tight")
    plt.close(fig)
    saved["capability_audit_png"] = str(f1)

    # ------------------------------------------------------------------
    # Figure 2: Baseline vs. Scenario A vs. Scenario B Timelines (3 Suites)
    # ------------------------------------------------------------------
    cmp_ids = weekly_cmp_df["comparison_id"].unique().tolist()
    fig, axes = plt.subplots(1, len(cmp_ids), figsize=(16.5, 5.4), dpi=150)
    if len(cmp_ids) == 1:
        axes = [axes]

    for ax, cid in zip(axes, cmp_ids):
        sub = weekly_cmp_df[weekly_cmp_df["comparison_id"] == cid].sort_values("simulation_week")
        wks = sub["simulation_week"].values
        b_cases = sub["baseline_cases"].values
        a_cases = sub["scenario_a_cases"].values
        b2_cases = sub["scenario_b_cases"].values
        a_name = str(sub["scenario_a_name"].iloc[0]).split("(")[0].strip()
        b_name = str(sub["scenario_b_name"].iloc[0]).split("(")[0].strip()
        dis_lbl = str(sub["selected_disease"].iloc[0])

        ax.plot(wks, b_cases, "o--", color="#d73027", lw=2.3, label="Baseline (No New Intervention)")
        ax.plot(wks, a_cases, "s-", color="#1a9850", lw=2.4, label=f"Scenario A: {a_name[:28]}")
        ax.plot(wks, b2_cases, "^-", color="#2b5c8f", lw=2.4, label=f"Scenario B: {b_name[:28]}")
        ax.fill_between(wks, a_cases, b_cases, color="#1a9850", alpha=0.12)

        ax.axvline(2, color="#555555", linestyle=":", lw=1.2, label="Week 2 Intervention Marker")
        if "DELAYED" in cid:
            ax.axvline(5, color="#2b5c8f", linestyle="-.", lw=1.2, label="Week 5 Delayed Marker")

        ax.set_title(f"{cid.replace('_', ' ')}\nDisease: {dis_lbl} [HYPOTHETICAL]", fontsize=9.5, fontweight="bold")
        ax.set_xlabel("Simulation Week (t)", fontsize=9)
        ax.set_ylabel("Hypothetical Weekly Cases", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=7.6, loc="upper left")

    fig.suptitle(
        "Side-by-Side Comparison: Baseline vs. Intervention Scenario A vs. Intervention Scenario B\n"
        "(DECISION-SUPPORT SIMULATION ONLY — All Trajectories Derived From User-Editable Assumptions)",
        fontsize=12,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    f2 = fig_dir / "02_baseline_vs_scenario_a_vs_scenario_b_timelines.png"
    fig.savefig(f2, bbox_inches="tight")
    plt.close(fig)
    saved["timeline_comparison_png"] = str(f2)

    # ------------------------------------------------------------------
    # Figure 3: District-Wise Scenario Comparison Map & Bar Panel
    # ------------------------------------------------------------------
    fig, (ax_map, ax_bar) = plt.subplots(1, 2, figsize=(15.5, 6.2), dpi=150)

    # Left panel: Maharashtra 36-district centroid map highlighting focal scenarios
    all_geo = pool_df[pool_df["disease"] == "ALL_DISEASES"].copy()
    ax_map.scatter(
        all_geo["longitude"],
        all_geo["latitude"],
        s=55,
        c="#cccccc",
        edgecolors="#555555",
        alpha=0.75,
        label="Other Maharashtra Districts",
    )

    urban_dengue = ["Pune", "Nashik", "Thane", "Mumbai Suburban"]
    vid_malaria = ["Gadchiroli", "Chandrapur", "Gondia", "Bhandara"]
    u_sub = all_geo[all_geo["district_name"].isin(urban_dengue)]
    v_sub = all_geo[all_geo["district_name"].isin(vid_malaria)]

    ax_map.scatter(
        u_sub["longitude"],
        u_sub["latitude"],
        s=180,
        c="#d95f02",
        edgecolors="#111111",
        linewidth=1.5,
        label="Suite 1 & 2 Cluster (Dengue Urban 4 Districts)",
        zorder=4,
    )
    ax_map.scatter(
        v_sub["longitude"],
        v_sub["latitude"],
        s=180,
        c="#7570b3",
        edgecolors="#111111",
        linewidth=1.5,
        marker="D",
        label="Suite 3 Cluster (Malaria Vidarbha 4-District Ring)",
        zorder=4,
    )

    for _, r in pd.concat([u_sub, v_sub]).iterrows():
        ax_map.annotate(
            str(r["district_name"]),
            (float(r["longitude"]), float(r["latitude"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8.2,
            fontweight="bold",
        )

    ax_map.set_title(
        "Maharashtra District Centroid Map of Benchmark Scenario Clusters\n"
        "(Point Centroid Geometry — No Fabricated Polygons)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_map.set_xlabel("Longitude (°E)")
    ax_map.set_ylabel("Latitude (°N)")
    ax_map.grid(True, linestyle="--", alpha=0.35)
    ax_map.legend(fontsize=8, loc="lower right")

    # Right panel: District-wise Cumulative Case Comparison across Suite 1 & Suite 3
    s1_dist = district_cmp_df[
        district_cmp_df["comparison_id"] == "CMP_01_EARLY_VS_DELAYED_VECTOR_CONTROL"
    ].copy()
    s3_dist = district_cmp_df[
        district_cmp_df["comparison_id"] == "CMP_03_LOCAL_FOCUS_VS_MULTI_DISTRICT_COORDINATED_RING"
    ].copy()
    comb_dist = pd.concat([s1_dist, s3_dist], ignore_index=True)

    x_idx = np.arange(len(comb_dist))
    width = 0.26
    ax_bar.bar(
        x_idx - width,
        comb_dist["baseline_cumulative_cases"],
        width=width,
        color="#d73027",
        label="Baseline (Unmitigated)",
    )
    ax_bar.bar(
        x_idx,
        comb_dist["scenario_a_cumulative_cases"],
        width=width,
        color="#1a9850",
        label="Scenario A (Early / Local)",
    )
    ax_bar.bar(
        x_idx + width,
        comb_dist["scenario_b_cumulative_cases"],
        width=width,
        color="#2b5c8f",
        label="Scenario B (Delayed / Multi-District Ring)",
    )
    ax_bar.set_xticks(x_idx)
    ax_bar.set_xticklabels(
        [f"{r['district_name']}\n({r['selected_disease'][:7]})" for _, r in comb_dist.iterrows()],
        rotation=30,
        ha="right",
        fontsize=8.5,
    )
    ax_bar.set_ylabel("8-Week Hypothetical Cumulative Cases")
    ax_bar.set_title(
        "District-Wise Cumulative Case Comparison Across Scenarios\n"
        "(Hypothetical What-If Totals — Never Mixed With Observed Counts)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_bar.grid(axis="y", linestyle="--", alpha=0.35)
    ax_bar.legend(fontsize=8.2)

    plt.tight_layout()
    f3 = fig_dir / "03_district_wise_scenario_comparison_map.png"
    fig.savefig(f3, bbox_inches="tight")
    plt.close(fig)
    saved["district_map_comparison_png"] = str(f3)

    # ------------------------------------------------------------------
    # Figure 4: Intervention Timing vs. Dampening Sensitivity Grid
    # ------------------------------------------------------------------
    start_weeks = [1, 2, 3, 4, 5, 6]
    damp_factors = [0.0, 0.10, 0.20, 0.30, 0.40]
    grid_mat = np.zeros((len(damp_factors), len(start_weeks)))

    for i_d, d_val in enumerate(damp_factors):
        for j_w, w_val in enumerate(start_weeks):
            cfg_tmp = InterventionScenarioConfig(
                scenario_id=f"SENS_D{int(d_val*100)}_W{w_val}",
                scenario_name="Sensitivity Cell",
                intervention_type="EARLY_INTERVENTION_VS_DELAYED_INTERVENTION",
                selected_disease="Dengue",
                selected_districts=["Pune", "Nashik", "Thane", "Mumbai Suburban"],
                scenario_duration_weeks=8,
                intervention_start_week=w_val,
                intervention_ramp_up_weeks=2,
                baseline_weekly_growth_multiplier=1.10,
                baseline_cross_district_coupling_beta=0.06,
                user_assumed_transmission_dampening_factor=d_val,
                user_assumed_cross_district_spillover_dampening=d_val * 0.5,
            )
            s_tmp, _ = simulate_intervention_scenario(cfg_tmp, pool_df, adj_map)
            grid_mat[i_d, j_w] = s_tmp["simulated_indicators"]["hypothetical_cumulative_underlying_cases"]

    fig, ax = plt.subplots(figsize=(9.8, 6.0), dpi=150)
    im = ax.imshow(grid_mat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(start_weeks)))
    ax.set_xticklabels([f"Start Week {w}" for w in start_weeks], fontsize=9.5, fontweight="bold")
    ax.set_yticks(np.arange(len(damp_factors)))
    ax.set_yticklabels(
        [f"User-Assumed Dampening = {int(d*100)}%" for d in damp_factors],
        fontsize=9.5,
        fontweight="bold",
    )

    for i in range(len(damp_factors)):
        for j in range(len(start_weeks)):
            val = grid_mat[i, j]
            base_v = grid_mat[0, j]
            pct_diff = ((val - base_v) / base_v * 100.0) if base_v > 0 else 0.0
            ax.text(
                j,
                i,
                f"{val:.0f}\n({pct_diff:+.1f}%)",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color="white" if val > np.median(grid_mat) * 1.05 else "#111111",
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("8-Week Hypothetical Cumulative Cases (4-District Cluster)", fontsize=9)
    ax.set_title(
        "Intervention Timing vs. User-Assumed Dampening Sensitivity Matrix (Dengue Cluster)\n"
        "(Illustrates Parametric Sensitivity to Start Delay — NOT Empirical RCT Efficacy)",
        fontsize=11,
        fontweight="bold",
        pad=12,
    )
    plt.tight_layout()
    f4 = fig_dir / "04_intervention_timing_and_assumption_sensitivity_grid.png"
    fig.savefig(f4, bbox_inches="tight")
    plt.close(fig)
    saved["sensitivity_grid_png"] = str(f4)

    # ------------------------------------------------------------------
    # Figure 5: Hypothetical Resource Allocation & Bed Strain Comparison
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12.5, 5.8), dpi=150)
    s_names = [s["scenario_id"].replace("SCEN_0", "S").replace("_", " ")[:28] for s in scenario_summaries]
    peak_gen = [
        s["expected_resource_requirements"]["hypothetical_peak_concurrent_general_beds_required"]
        for s in scenario_summaries
    ]
    peak_icu = [
        s["expected_resource_requirements"]["hypothetical_peak_concurrent_icu_beds_required"]
        for s in scenario_summaries
    ]
    alloc_gen = [
        s["expected_resource_requirements"]["hypothetical_total_allocated_general_beds"] or 0.0
        for s in scenario_summaries
    ]

    x_pos = np.arange(len(s_names))
    w = 0.28
    ax.bar(x_pos - w, peak_gen, width=w, color="#e66101", label="Peak Required General Beds (Hypothetical)")
    ax.bar(x_pos, peak_icu, width=w, color="#b2182b", label="Peak Required ICU Beds (Hypothetical)")
    ax.bar(
        x_pos + w,
        alloc_gen,
        width=w,
        color="#2166ac",
        alpha=0.82,
        label="User-Entered Allocated General Beds (Hypothetical)",
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(s_names, rotation=25, ha="right", fontsize=8.5, fontweight="bold")
    ax.set_ylabel("Concurrent Beds across Selected Cluster")
    ax.set_title(
        "Hypothetical Peak Concurrent Bed Requirements vs. User-Entered Resource Allocation\n"
        "(Zero Actual Hospital Inventory Exists in Workspace — All Allocations Are Analyst Inputs)",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8.8)
    plt.tight_layout()
    f5 = fig_dir / "05_hypothetical_resource_allocation_and_bed_strain_comparison.png"
    fig.savefig(f5, bbox_inches="tight")
    plt.close(fig)
    saved["resource_allocation_png"] = str(f5)

    # ------------------------------------------------------------------
    # Interactive HTML Simulator (Section 7 Frontend Views)
    # ------------------------------------------------------------------
    _write_interactive_html_simulator(pool_df, adj_map, scenario_summaries, html_path)
    saved["interactive_html_simulator"] = str(html_path)
    return saved


def _write_interactive_html_simulator(
    pool_df: pd.DataFrame,
    adj_map: Dict[str, List[str]],
    scenario_summaries: List[Dict[str, Any]],
    html_path: Path,
) -> None:
    """Create a standalone interactive HTML/JS Public-Health Intervention Scenario Simulator."""
    anchors_by_dis_dist: Dict[str, Dict[str, float]] = {}
    diseases = sorted(pool_df["disease"].unique().tolist())
    districts = sorted(pool_df["district_name"].unique().tolist())

    for dis in diseases:
        anchors_by_dis_dist[dis] = {}
        sub = pool_df[pool_df["disease"] == dis]
        for _, r in sub.iterrows():
            dname = str(r["district_name"])
            fc7 = r.get("forecast_7d_mean_cases", np.nan)
            obs_m = r.get("observed_mean_cases_per_outbreak_week", np.nan)
            val = float(fc7) if (pd.notna(fc7) and float(fc7) > 0) else (
                float(obs_m) if (pd.notna(obs_m) and float(obs_m) > 0) else 8.0
            )
            anchors_by_dis_dist[dis][dname] = round(val, 2)

    payload_js = json.dumps(
        {
            "diseases": diseases,
            "districts": districts,
            "anchors": anchors_by_dis_dist,
            "adjacency": adj_map,
            "disclaimer": MANDATORY_INTERVENTION_GOVERNANCE_DISCLAIMER,
        }
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>JEEVAN-DRISHTI — Public-Health Intervention Scenario Simulator (Phase 14)</title>
<style>
  body {{ font-family: 'Segoe UI', Roboto, Arial, sans-serif; margin: 0; padding: 0; background: #f4f6f9; color: #1f2937; }}
  header {{ background: #1e293b; color: #ffffff; padding: 16px 24px; }}
  header h1 {{ margin: 0 0 6px 0; font-size: 20px; }}
  .disclaimer-banner {{ background: #fef2f2; border-left: 6px solid #dc2626; color: #7f1d1d; padding: 12px 18px; font-size: 13px; font-weight: 600; }}
  .container {{ display: grid; grid-template-columns: 370px 1fr; gap: 18px; padding: 18px 24px; }}
  .card {{ background: #ffffff; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); padding: 16px; margin-bottom: 16px; }}
  .card h2 {{ margin-top: 0; font-size: 15px; border-bottom: 1px solid #e5e7eb; padding-bottom: 8px; color: #0f172a; }}
  label {{ display: block; font-size: 12px; font-weight: 600; margin-top: 10px; color: #334155; }}
  select, input[type=number], input[type=range] {{ width: 100%; margin-top: 4px; padding: 6px; border: 1px solid #cbd5e1; border-radius: 4px; font-size: 13px; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 14px; }}
  .kpi-box {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 10px 12px; }}
  .kpi-title {{ font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 700; }}
  .kpi-val {{ font-size: 20px; font-weight: 700; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; margin-top: 8px; }}
  th, td {{ border: 1px solid #e2e8f0; padding: 7px 9px; text-align: right; }}
  th {{ background: #f1f5f9; color: #1e293b; text-align: center; }}
  td:first-child {{ text-align: left; font-weight: 600; }}
  .badge-hyp {{ background: #fef3c7; color: #92400e; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: 700; }}
</style>
</head>
<body>
<header>
  <h1>JEEVAN-DRISHTI — Public-Health Intervention Scenario Simulator (Phase 14)</h1>
  <div style="font-size:12.5px; color:#cbd5e1;">Transparent Parametric What-If Decision-Support Tool | Baseline vs. Scenario A vs. Scenario B</div>
</header>
<div class="disclaimer-banner">
  ⚠️ GOVERNANCE SAFEGUARD: {MANDATORY_INTERVENTION_GOVERNANCE_DISCLAIMER}
</div>

<div class="container">
  <!-- Parameter Control Panel -->
  <div>
    <div class="card">
      <h2>1. Parameter Control Panel (User-Editable)</h2>
      <label>Disease Profile</label>
      <select id="selDisease" onchange="runSimulation()"></select>

      <label>Focal District (Primary)</label>
      <select id="selDistrict" onchange="runSimulation()"></select>

      <label>Include Verified Contiguous Ring Neighbours (Phase 10 Graph)</label>
      <select id="selRing" onchange="runSimulation()">
        <option value="true" selected>Yes — Multi-District Ring Cluster</option>
        <option value="false">No — Single Focal District Only</option>
      </select>

      <label>Scenario Duration (Weeks): <span id="lblDur">8</span>w</label>
      <input type="range" id="rngDur" min="4" max="16" step="1" value="8" oninput="runSimulation()" />

      <label>Baseline Weekly Growth Multiplier (g₀): <span id="lblGrowth">1.10</span>x</label>
      <input type="range" id="rngGrowth" min="0.90" max="1.35" step="0.02" value="1.10" oninput="runSimulation()" />

      <hr style="margin:14px 0; border:none; border-top:1px solid #e2e8f0;" />
      <label style="color:#15803d;">Scenario A: Start Week (<span id="lblStartA">2</span>) & User-Assumed Dampening (<span id="lblDampA">28</span>%)</label>
      <input type="range" id="rngStartA" min="1" max="8" step="1" value="2" oninput="runSimulation()" />
      <input type="range" id="rngDampA" min="0" max="60" step="2" value="28" oninput="runSimulation()" />

      <label style="color:#1d4ed8;">Scenario B: Start Week (<span id="lblStartB">5</span>) & User-Assumed Dampening (<span id="lblDampB">28</span>%)</label>
      <input type="range" id="rngStartB" min="1" max="8" step="1" value="5" oninput="runSimulation()" />
      <input type="range" id="rngDampB" min="0" max="60" step="2" value="28" oninput="runSimulation()" />

      <label>User-Assumed Hospital Admission Rate (%): <span id="lblAdm">20</span>%</label>
      <input type="range" id="rngAdm" min="5" max="50" step="1" value="20" oninput="runSimulation()" />

      <label>Hypothetical Allocated General Beds per District</label>
      <input type="number" id="numAllocBeds" value="12" min="0" max="500" onchange="runSimulation()" />
    </div>

    <div class="card">
      <h2>2. Explicit Assumption Summary</h2>
      <div id="assumptionBox" style="font-size:12px; line-height:1.5;"></div>
    </div>
  </div>

  <!-- Right Main Workspace -->
  <div>
    <div class="card">
      <h2>3. Side-by-Side Comparison: Baseline vs. Scenario A vs. Scenario B <span class="badge-hyp">HYPOTHETICAL WHAT-IF</span></h2>
      <div class="kpi-grid">
        <div class="kpi-box" style="border-left:4px solid #dc2626;">
          <div class="kpi-title">Baseline Cumulative Cases</div>
          <div class="kpi-val" id="kpiBaseCases">-</div>
          <div style="font-size:11px; color:#64748b;" id="kpiBaseBeds">-</div>
        </div>
        <div class="kpi-box" style="border-left:4px solid #16a34a;">
          <div class="kpi-title">Scenario A Cumulative Cases</div>
          <div class="kpi-val" id="kpiScenACases">-</div>
          <div style="font-size:11px; color:#15803d;" id="kpiScenADiff">-</div>
        </div>
        <div class="kpi-box" style="border-left:4px solid #2563eb;">
          <div class="kpi-title">Scenario B Cumulative Cases</div>
          <div class="kpi-val" id="kpiScenBCases">-</div>
          <div style="font-size:11px; color:#1d4ed8;" id="kpiScenBDiff">-</div>
        </div>
      </div>

      <table id="tblWeekly">
        <thead>
          <tr>
            <th>Simulation Week</th>
            <th>Baseline Weekly Cases (Hypothetical)</th>
            <th>Scenario A Cases (Hypothetical)</th>
            <th>Scenario B Cases (Hypothetical)</th>
            <th>Scenario A Peak Gen. Beds</th>
            <th>Scenario B Peak Gen. Beds</th>
            <th>Allocated Beds Gap (Scen A)</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>

    <div class="card">
      <h2>4. Results & Limitations Governance Panel</h2>
      <ul style="font-size:12.5px; line-height:1.55; margin:0; padding-left:18px; color:#334155;">
        <li><strong>No Empirical Efficacy Fabrication:</strong> This simulator does NOT claim that vector control, awareness campaigns, or testing will empirically reduce cases by a fixed percentage. All percentage reductions shown above are the mathematical result of the user-selected dampening sliders.</li>
        <li><strong>Strict Separation of Records:</strong> Simulated weekly counts are never written back to or merged with verified 2023–2025 IDSP outbreak surveillance records.</li>
        <li><strong>Resource Capacity Caveat:</strong> Because no hospital bed or medicine stock inventory exists in the project workspace, bed surplus/deficit figures compare hypothetical demand against the user-entered bed allocation input.</li>
        <li><strong>Non-Clinical Decision Support:</strong> For exploratory public-health scenario planning only; never replaces epidemiological field investigation or clinical judgement.</li>
      </ul>
    </div>
  </div>
</div>

<script>
const DATA = {payload_js};

function initControls() {{
  const selDis = document.getElementById('selDisease');
  DATA.diseases.forEach(d => {{
    const opt = document.createElement('option');
    opt.value = d; opt.textContent = d;
    if (d === 'Dengue') opt.selected = true;
    selDis.appendChild(opt);
  }});
  const selDist = document.getElementById('selDistrict');
  DATA.districts.forEach(d => {{
    const opt = document.createElement('option');
    opt.value = d; opt.textContent = d;
    if (d === 'Pune') opt.selected = true;
    selDist.appendChild(opt);
  }});
  runSimulation();
}}

function simulateTrack(anchorSum, nDists, T, g0, startWk, dampPct, admRate, allocPerDist) {{
  const dFactor = dampPct / 100.0;
  const alos = 5.0;
  const icuShare = 0.15;
  let prev = anchorSum;
  const rows = [];
  for (let t = 1; t <= T; t++) {{
    const r_t = (t < startWk) ? 0.0 : Math.min(1.0, (t - startWk + 1) / 2.0);
    let cases = (t === 1) ? anchorSum * (1.0 - r_t * dFactor * 0.5) : prev * g0 * (1.0 - r_t * dFactor);
    prev = cases;
    const genBeds = cases * admRate * (1.0 - icuShare) * (alos / 7.0);
    const totalAlloc = allocPerDist * nDists;
    rows.push({{ week: t, cases: cases, genBeds: genBeds, gap: totalAlloc - genBeds }});
  }}
  return rows;
}}

function runSimulation() {{
  const dis = document.getElementById('selDisease').value;
  const focal = document.getElementById('selDistrict').value;
  const useRing = document.getElementById('selRing').value === 'true';
  const T = parseInt(document.getElementById('rngDur').value, 10);
  const g0 = parseFloat(document.getElementById('rngGrowth').value);
  const startA = parseInt(document.getElementById('rngStartA').value, 10);
  const dampA = parseFloat(document.getElementById('rngDampA').value);
  const startB = parseInt(document.getElementById('rngStartB').value, 10);
  const dampB = parseFloat(document.getElementById('rngDampB').value);
  const admPct = parseFloat(document.getElementById('rngAdm').value);
  const allocPerDist = parseFloat(document.getElementById('numAllocBeds').value || 0);

  document.getElementById('lblDur').textContent = T;
  document.getElementById('lblGrowth').textContent = g0.toFixed(2);
  document.getElementById('lblStartA').textContent = startA;
  document.getElementById('lblDampA').textContent = dampA;
  document.getElementById('lblStartB').textContent = startB;
  document.getElementById('lblDampB').textContent = dampB;
  document.getElementById('lblAdm').textContent = admPct;

  const ringDists = useRing ? (DATA.adjacency[focal] || []) : [];
  const allDists = [focal, ...ringDists];
  let anchorSum = 0.0;
  allDists.forEach((d, idx) => {{
    const a = (DATA.anchors[dis] && DATA.anchors[dis][d]) ? DATA.anchors[dis][d] : 8.0;
    anchorSum += (idx === 0) ? a : a * 0.65;
  }});

  const baseRows = simulateTrack(anchorSum, allDists.length, T, g0, 99, 0.0, admPct/100.0, allocPerDist);
  const scenARows = simulateTrack(anchorSum, allDists.length, T, g0, startA, dampA, admPct/100.0, allocPerDist);
  const scenBRows = simulateTrack(anchorSum, allDists.length, T, g0, startB, dampB, admPct/100.0, allocPerDist);

  const sumBase = baseRows.reduce((s, r) => s + r.cases, 0);
  const sumA = scenARows.reduce((s, r) => s + r.cases, 0);
  const sumB = scenBRows.reduce((s, r) => s + r.cases, 0);
  const peakBaseBeds = Math.max(...baseRows.map(r => r.genBeds));
  const peakABeds = Math.max(...scenARows.map(r => r.genBeds));
  const peakBBeds = Math.max(...scenBRows.map(r => r.genBeds));

  document.getElementById('kpiBaseCases').textContent = sumBase.toFixed(1);
  document.getElementById('kpiBaseBeds').textContent = `Peak Gen. Beds: ${{peakBaseBeds.toFixed(1)}}`;
  document.getElementById('kpiScenACases').textContent = sumA.toFixed(1);
  const diffA = sumA - sumBase;
  const pctA = sumBase > 0 ? (diffA / sumBase * 100.0) : 0.0;
  document.getElementById('kpiScenADiff').textContent = `${{diffA.toFixed(1)}} cases (${{pctA.toFixed(1)}}%) vs Baseline | Peak Beds: ${{peakABeds.toFixed(1)}}`;

  document.getElementById('kpiScenBCases').textContent = sumB.toFixed(1);
  const diffB = sumB - sumBase;
  const pctB = sumBase > 0 ? (diffB / sumBase * 100.0) : 0.0;
  document.getElementById('kpiScenBDiff').textContent = `${{diffB.toFixed(1)}} cases (${{pctB.toFixed(1)}}%) vs Baseline | Peak Beds: ${{peakBBeds.toFixed(1)}}`;

  document.getElementById('assumptionBox').innerHTML =
    `<strong>Simulated Districts (${{allDists.length}}):</strong> ${{allDists.join(', ')}}<br/>` +
    `<strong>Week 1 Anchor Sum:</strong> ${{anchorSum.toFixed(2)}} cases/week (from Phase 6/2 baseline)<br/>` +
    `<strong>Scenario A Assumption:</strong> Start Week ${{startA}}, User-Assumed Dampening ${{dampA}}%<br/>` +
    `<strong>Scenario B Assumption:</strong> Start Week ${{startB}}, User-Assumed Dampening ${{dampB}}%<br/>` +
    `<strong>Total Allocated General Beds:</strong> ${{(allocPerDist * allDists.length).toFixed(0)}} beds (${{allocPerDist}}/district)`;

  const tbody = document.querySelector('#tblWeekly tbody');
  tbody.innerHTML = '';
  for (let i = 0; i < T; i++) {{
    const tr = document.createElement('tr');
    const gapVal = scenARows[i].gap;
    const gapColor = gapVal >= 0 ? '#15803d' : '#dc2626';
    tr.innerHTML =
      `<td>Week ${{i+1}}</td>` +
      `<td>${{baseRows[i].cases.toFixed(2)}}</td>` +
      `<td>${{scenARows[i].cases.toFixed(2)}}</td>` +
      `<td>${{scenBRows[i].cases.toFixed(2)}}</td>` +
      `<td>${{scenARows[i].genBeds.toFixed(2)}}</td>` +
      `<td>${{scenBRows[i].genBeds.toFixed(2)}}</td>` +
      `<td style="color:${{gapColor}}; font-weight:700;">${{gapVal >= 0 ? '+' : ''}}${{gapVal.toFixed(2)}}</td>`;
    tbody.appendChild(tr);
  }}
}}

window.onload = initControls;
</script>
</body>
</html>
"""
    html_path.write_text(html_content, encoding="utf-8")
