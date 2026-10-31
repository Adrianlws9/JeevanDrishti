"""Static & Interactive Visualization Suite for Phase 11 Healthcare Resource Demand Engine."""
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.engines.resource_intelligence.data_availability import MANDATORY_RESOURCE_DISCLAIMER


def plot_01_resource_data_availability_audit(
    completeness_df: pd.DataFrame, out_path: Path
) -> Path:
    """Figure 1: Healthcare Resource Data Completeness Audit vs. Upstream Surveillance Drivers."""
    fig, ax = plt.subplots(figsize=(14.5, 6.8), dpi=160)
    fig.suptitle(
        "JEEVAN-DRISHTI Phase 11: Healthcare Resource Data Availability & Non-Fabrication Audit\n"
        "[Zero Hospital/Bed/Medicine Datasets Exist in Workspace — Never Invented; Upstream Case Drivers Kept Separate]",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )

    df_plot = completeness_df.iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(df_plot))
    avail_pct = 100.0 - df_plot["missingness_pct"].to_numpy(dtype=float)
    colors = ["#16A34A" if p else "#94A3B8" for p in df_plot["present_in_workspace"]]

    ax.barh(y_pos, avail_pct, color=colors, edgecolor="#0F172A", height=0.62)
    for i, r in df_plot.iterrows():
        status_txt = (
            f"PRESENT ({100.0 - r['missingness_pct']:.1f}% non-null) — {r['engine_handling_mode']}"
            if r["present_in_workspace"]
            else "UNAVAILABLE IN WORKSPACE (0% — Set to NaN; Handled via Hypothetical Scenario Calculator)"
        )
        ax.text(
            min(avail_pct[i] + 1.5, 52.0),
            i,
            status_txt,
            va="center",
            ha="left",
            fontsize=8.2,
            fontweight="bold",
            color="#0F172A",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot["variable_label"], fontsize=8.8, fontweight="bold")
    ax.set_xlim(0, 108)
    ax.set_xlabel("Workspace Data Availability (%)", fontsize=9.5, fontweight="bold")
    ax.set_title(
        "Audit of 9 Healthcare Resource Categories (0% Present) vs. 3 Upstream Epidemiological Case Drivers",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    ax.set_facecolor("#F8FAFC")

    fig.tight_layout(rect=[0, 0.01, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_02_upstream_epidemiological_demand_drivers(
    district_summary_df: pd.DataFrame, out_path: Path
) -> Path:
    """Figure 2: Upstream Epidemiological Demand Drivers by District (2023-2025)."""
    valid = (
        district_summary_df[district_summary_df["observed_2023_2025_peak_weekly_cases"].notna()]
        .sort_values("observed_2023_2025_peak_weekly_cases", ascending=False)
        .head(15)
        .iloc[::-1]
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), dpi=160)
    fig.suptitle(
        "JEEVAN-DRISHTI Phase 11: Verified Upstream Epidemiological Demand Drivers (2023–2025 IDSP Surveillance)\n"
        "[Used as Optional Anchors for the Hypothetical Scenario Calculator — Never Labelled as Hospital Admissions]",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )

    y_pos = np.arange(len(valid))
    ax1 = axes[0]
    ax1.barh(
        y_pos + 0.18,
        valid["observed_2023_2025_peak_weekly_cases"],
        height=0.35,
        color="#DC2626",
        label="Observed Peak Weekly Outbreak Cases (2023–2025)",
    )
    ax1.barh(
        y_pos - 0.18,
        valid["observed_2023_2025_mean_cases_per_outbreak_week"],
        height=0.35,
        color="#2563EB",
        label="Observed Mean Weekly Outbreak Cases (2023–2025)",
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(valid["district_name"], fontsize=8.8, fontweight="bold")
    ax1.set_xlabel("Observed Weekly Outbreak Cases", fontsize=9.5, fontweight="bold")
    ax1.set_title("A. Top 15 Districts by Peak Weekly Outbreak Caseload", fontsize=10.5, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8.5)
    ax1.grid(axis="x", linestyle=":", alpha=0.45)

    ax2 = axes[1]
    ax2.barh(
        y_pos + 0.18,
        valid["upstream_phase6_forecast_7d_mean_cases"],
        height=0.35,
        color="#D97706",
        label="Phase 6 7-Day Forecast Mean Cases",
    )
    ax2.barh(
        y_pos - 0.18,
        valid["upstream_phase6_forecast_30d_mean_cases"],
        height=0.35,
        color="#059669",
        label="Phase 6 30-Day Forecast Mean Cases",
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(valid["district_name"], fontsize=8.8, fontweight="bold")
    ax2.set_xlabel("Forecasted Weekly Outbreak Cases (Phase 6 Engine)", fontsize=9.5, fontweight="bold")
    ax2.set_title("B. Upstream Phase 6 Case Forecasts (7d vs. 30d Horizon)", fontsize=10.5, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=8.5)
    ax2.grid(axis="x", linestyle=":", alpha=0.45)

    fig.tight_layout(rect=[0, 0.01, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_03_hypothetical_resource_scenario_comparison(
    catalog_df: pd.DataFrame, out_path: Path
) -> Path:
    """Figure 3: Statewide Hypothetical Resource Requirements Across Benchmark Scenarios."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), dpi=160)
    fig.suptitle(
        "JEEVAN-DRISHTI Phase 11: Hypothetical Scenario Calculator Outputs Across 4 Planning Profiles\n"
        "[WARNING: HYPOTHETICAL SCENARIO CALCULATIONS BASED ON USER-EDITABLE ASSUMPTIONS — NOT ACTUAL CAPACITY OR FORECAST]",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )

    short_labels = [
        "RES_SCEN_01\n(7d Forecast +0%,\nadm=15%, LOS=4.5d)",
        "RES_SCEN_02\n(7d Forecast +50%,\nadm=22%, LOS=5.0d)",
        "RES_SCEN_03\n(Peak Weekly +25%,\nadm=20%, LOS=4.0d)",
        "RES_SCEN_04\n(Mean Weekly +40%,\nadm=18%, LOS=2.5d)",
    ]
    x = np.arange(len(catalog_df))

    ax1 = axes[0]
    w = 0.26
    ax1.bar(
        x - w,
        catalog_df["statewide_hypothetical_weekly_admissions"],
        width=w,
        color="#2563EB",
        edgecolor="#0F172A",
        label="Hypothetical Weekly Admissions",
    )
    ax1.bar(
        x,
        catalog_df["statewide_hypothetical_general_beds_required"],
        width=w,
        color="#D97706",
        edgecolor="#0F172A",
        label="Hypothetical Concurrent General Beds",
    )
    ax1.bar(
        x + w,
        catalog_df["statewide_hypothetical_icu_beds_required"],
        width=w,
        color="#DC2626",
        edgecolor="#0F172A",
        label="Hypothetical Concurrent ICU Beds",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, fontsize=8.5, fontweight="bold")
    ax1.set_ylabel("Hypothetical Statewide Units Required", fontsize=9.5, fontweight="bold")
    ax1.set_title("A. Hypothetical Admissions, General Beds & ICU Beds by Scenario", fontsize=10.5, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=8.5)
    ax1.grid(axis="y", linestyle=":", alpha=0.45)

    ax2 = axes[1]
    ax2.bar(
        x - 0.18,
        catalog_df["statewide_hypothetical_diagnostic_tests_required"],
        width=0.34,
        color="#7C3AED",
        edgecolor="#0F172A",
        label="Hypothetical Diagnostic Tests Required",
    )
    ax2.bar(
        x + 0.18,
        catalog_df["statewide_hypothetical_iv_fluid_units_required"],
        width=0.34,
        color="#0891B2",
        edgecolor="#0F172A",
        label="Hypothetical IV Fluid / ORS Units Required",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels, fontsize=8.5, fontweight="bold")
    ax2.set_ylabel("Hypothetical Consumable Units Required", fontsize=9.5, fontweight="bold")
    ax2.set_title("B. Hypothetical Diagnostic Tests & IV/ORS Units by Scenario", fontsize=10.5, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.grid(axis="y", linestyle=":", alpha=0.45)

    fig.tight_layout(rect=[0, 0.01, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_04_hypothetical_capacity_strain_sensitivity(
    eval_df: pd.DataFrame, out_path: Path
) -> Path:
    """Figure 4: District Hypothetical Bed Strain & Assumption Sensitivity Heatmap."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.0), dpi=160)
    fig.suptitle(
        "JEEVAN-DRISHTI Phase 11: Hypothetical Capacity Strain Comparison & Assumption Sensitivity Matrix\n"
        "[Assumed Ward Capacity Is a User-Editable Input Parameter — Never Presented as Actual Hospital Inventory]",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )

    # Panel A: Top 12 Districts under RES_SCEN_03 (Peak Weekly + 25% vs Assumed 50 General Beds)
    ax1 = axes[0]
    s3 = (
        eval_df[
            (eval_df["scenario_id"] == "RES_SCEN_03_HISTORICAL_PEAK_STRESS_TEST")
            & (eval_df["hypothetical_general_bed_utilization_ratio"].notna())
        ]
        .sort_values("hypothetical_general_bed_utilization_ratio", ascending=False)
        .head(12)
        .iloc[::-1]
    )
    y_pos = np.arange(len(s3))
    ratios_pct = s3["hypothetical_general_bed_utilization_ratio"].to_numpy(dtype=float) * 100.0
    cols = [
        "#DC2626" if r > 100.0 else ("#F59E0B" if r >= 80.0 else "#2563EB")
        for r in ratios_pct
    ]
    ax1.barh(y_pos, ratios_pct, color=cols, edgecolor="#0F172A", height=0.62)
    ax1.axvline(100.0, color="#DC2626", linestyle="--", linewidth=1.4, label="100% of User-Assumed Ward Capacity (50 beds)")
    ax1.axvline(80.0, color="#F59E0B", linestyle=":", linewidth=1.3, label="80% High-Strain Threshold")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(s3["district_name"], fontsize=8.8, fontweight="bold")
    ax1.set_xlabel("Hypothetical General Bed Utilization (% of Assumed 50-Bed Ward)", fontsize=9, fontweight="bold")
    ax1.set_title(
        "A. RES_SCEN_03 (Peak Weekly Load +25%, adm=20%, LOS=4d)\nTop 12 Districts vs. Hypothetical 50-Bed Reference Ward",
        fontsize=10,
        fontweight="bold",
    )
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(axis="x", linestyle=":", alpha=0.45)

    # Panel B: 2D Sensitivity Grid of Concurrent Beds Required for 100 Hypothetical Weekly Cases
    ax2 = axes[1]
    adm_rates = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    los_vals = [2.0, 3.0, 4.5, 6.0, 8.0]
    mat = np.zeros((len(los_vals), len(adm_rates)), dtype=float)
    for r_i, los in enumerate(los_vals):
        for c_i, adm in enumerate(adm_rates):
            # Concurrent total beds (general + ICU) per 100 weekly cases = (100 * adm * los) / 7.0
            mat[r_i, c_i] = (100.0 * adm * los) / 7.0

    im = ax2.imshow(mat, cmap="YlOrRd", aspect="auto")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.045, pad=0.02)
    cbar.set_label("Hypothetical Concurrent Beds per 100 Weekly Cases", fontsize=8.5)
    ax2.set_xticks(np.arange(len(adm_rates)))
    ax2.set_xticklabels([f"{int(a*100)}%" for a in adm_rates], fontsize=9, fontweight="bold")
    ax2.set_yticks(np.arange(len(los_vals)))
    ax2.set_yticklabels([f"{l:.1f} days" for l in los_vals], fontsize=9, fontweight="bold")
    ax2.set_xlabel("User-Assumed Inpatient Admission Proportion (α_adm)", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("User-Assumed Average Length of Stay (L_stay)", fontsize=9.5, fontweight="bold")
    ax2.set_title(
        "B. Formula Sensitivity Table (Per 100 Hypothetical Weekly Cases)\nConcurrent Beds = (Cases × α_adm × L_stay) / 7",
        fontsize=10,
        fontweight="bold",
    )
    for r_i in range(len(los_vals)):
        for c_i in range(len(adm_rates)):
            ax2.text(
                c_i,
                r_i,
                f"{mat[r_i, c_i]:.1f}",
                ha="center",
                va="center",
                fontsize=8.8,
                fontweight="bold",
                color="#0F172A" if mat[r_i, c_i] < 22 else "white",
            )

    fig.tight_layout(rect=[0, 0.01, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_interactive_resource_calculator_html(
    district_summary_df: pd.DataFrame, out_path: Path
) -> Path:
    """Generate a standalone interactive HTML/JS Healthcare Resource Scenario Calculator."""
    dist_payload = []
    for _, r in district_summary_df.iterrows():
        dist_payload.append(
            {
                "id": str(r["district_id"]),
                "name": str(r["district_name"]),
                "division": str(r["administrative_division"]),
                "peak_wk": float(r["observed_2023_2025_peak_weekly_cases"]) if pd.notna(r["observed_2023_2025_peak_weekly_cases"]) else None,
                "mean_wk": float(r["observed_2023_2025_mean_cases_per_outbreak_week"]) if pd.notna(r["observed_2023_2025_mean_cases_per_outbreak_week"]) else None,
                "fc7_wk": float(r["upstream_phase6_forecast_7d_mean_cases"]) if pd.notna(r["upstream_phase6_forecast_7d_mean_cases"]) else None,
                "dom_disease": str(r["dominant_outbreak_disease_2023_2025"]),
            }
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>JEEVAN-DRISHTI Phase 11 — Hypothetical Healthcare Resource Scenario Calculator</title>
<style>
  :root {{
    --bg: #0f172a; --panel: #1e293b; --border: #334155; --text: #f8fafc;
    --muted: #94a3b8; --accent: #38bdf8; --warn: #f59e0b; --danger: #ef4444; --ok: #22c55e;
  }}
  * {{ box-sizing: border-box; font-family: system-ui, -apple-system, sans-serif; }}
  body {{ margin: 0; background: var(--bg); color: var(--text); padding: 16px; }}
  .banner {{
    background: #451a03; border: 1px solid var(--warn); color: #fef3c7;
    padding: 10px 14px; border-radius: 8px; margin-bottom: 14px; font-size: 13px; font-weight: 600;
  }}
  .grid {{ display: grid; grid-template-columns: 380px 1fr; gap: 14px; }}
  .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }}
  h2, h3 {{ margin: 0 0 10px 0; font-size: 15px; color: var(--accent); }}
  label {{ display: block; font-size: 12px; color: var(--muted); margin-top: 8px; }}
  select, input[type=range], input[type=number] {{
    width: 100%; margin-top: 4px; background: #0f172a; color: var(--text);
    border: 1px solid var(--border); padding: 6px 8px; border-radius: 6px; font-size: 12.5px;
  }}
  .val-badge {{ float: right; color: var(--text); font-weight: 700; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-bottom: 14px; }}
  .kpi {{ background: #0b1120; border: 1px solid var(--border); border-radius: 8px; padding: 10px; }}
  .kpi .lbl {{ font-size: 11px; color: var(--muted); }}
  .kpi .num {{ font-size: 20px; font-weight: 700; margin-top: 4px; color: var(--accent); }}
  .kpi .tag {{ font-size: 10px; color: var(--warn); margin-top: 2px; }}
</style>
</head>
<body>
<div class="banner">
  ⚠️ {MANDATORY_RESOURCE_DISCLAIMER}
</div>
<div class="grid">
  <div class="card">
    <h2>1. Editable Scenario Assumptions</h2>
    <label>Maharashtra District</label>
    <select id="distSelect"></select>

    <label>Baseline Weekly Case Anchor Mode</label>
    <select id="anchorSelect">
      <option value="fc7_wk">Phase 6 7-Day Forecast Mean Cases</option>
      <option value="peak_wk">Observed 2023–2025 Peak Weekly Cases</option>
      <option value="mean_wk">Observed 2023–2025 Mean Weekly Cases</option>
      <option value="custom">User-Entered Custom Weekly Cases</option>
    </select>

    <label>User-Entered Custom Baseline Cases</label>
    <input type="number" id="customCasesInput" value="50" min="0" max="10000">

    <label>Hypothetical Case Increase (%) <span id="incLbl" class="val-badge">+25%</span></label>
    <input type="range" id="incInput" min="-50" max="200" step="5" value="25">

    <label>Assumed Admission Proportion (α_adm) <span id="admLbl" class="val-badge">15%</span></label>
    <input type="range" id="admInput" min="1" max="60" step="1" value="15">

    <label>Assumed ICU Share of Admissions (α_icu) <span id="icuLbl" class="val-badge">10%</span></label>
    <input type="range" id="icuInput" min="0" max="40" step="1" value="10">

    <label>Assumed Avg Length of Stay (Days) <span id="losLbl" class="val-badge">4.5 d</span></label>
    <input type="range" id="losInput" min="1" max="14" step="0.5" value="4.5">

    <label>Assumed Diagnostic Tests per Case <span id="testLbl" class="val-badge">3.0</span></label>
    <input type="range" id="testInput" min="0" max="10" step="0.5" value="3.0">

    <label>User-Assumed Available General Beds (Hypothetical)</label>
    <input type="number" id="genCapInput" value="40" min="1" max="5000">

    <label>User-Assumed Available ICU Beds (Hypothetical)</label>
    <input type="number" id="icuCapInput" value="6" min="1" max="1000">
  </div>

  <div class="card">
    <h2>2. Hypothetical Resource Demand & Capacity Comparison (Strictly Non-Empirical)</h2>
    <div id="statusBox" style="margin-bottom:10px;font-size:12.5px;color:var(--muted);"></div>
    <div class="kpi-grid">
      <div class="kpi">
        <div class="lbl">Hypothetical Projected Weekly Cases</div>
        <div class="num" id="kpiCases">-</div>
        <div class="tag">HYPOTHETICAL SCENARIO</div>
      </div>
      <div class="kpi">
        <div class="lbl">Hypothetical Weekly Admissions</div>
        <div class="num" id="kpiAdm">-</div>
        <div class="tag">ASSUMPTION-BASED</div>
      </div>
      <div class="kpi">
        <div class="lbl">Hypothetical Total Bed-Days</div>
        <div class="num" id="kpiBedDays">-</div>
        <div class="tag">ASSUMPTION-BASED</div>
      </div>
      <div class="kpi">
        <div class="lbl">Concurrent General Beds Required</div>
        <div class="num" id="kpiGenBeds">-</div>
        <div class="tag" id="kpiGenUtil">-</div>
      </div>
      <div class="kpi">
        <div class="lbl">Concurrent ICU Beds Required</div>
        <div class="num" id="kpiIcuBeds">-</div>
        <div class="tag" id="kpiIcuUtil">-</div>
      </div>
      <div class="kpi">
        <div class="lbl">Hypothetical Diagnostic Tests</div>
        <div class="num" id="kpiTests">-</div>
        <div class="tag">ASSUMPTION-BASED</div>
      </div>
    </div>
    <h3>3. Empirical Workspace Data Completeness Status for Selected District</h3>
    <div style="background:#0b1120;border:1px solid var(--border);border-radius:8px;padding:12px;font-size:12px;line-height:1.6;">
      • <b>Empirical Hospital Admissions:</b> <code>Insufficient Data (NaN — No Hospital Dataset in Workspace)</code><br>
      • <b>Empirical Available / Occupied Beds:</b> <code>Insufficient Data (NaN — No Bed Inventory in Workspace)</code><br>
      • <b>Empirical ICU / Ventilator Capacity:</b> <code>Insufficient Data (NaN — No ICU Dataset in Workspace)</code><br>
      • <b>Empirical Testing & Medicine Stock:</b> <code>Insufficient Data (NaN — No Supply Chain Dataset in Workspace)</code>
    </div>
  </div>
</div>
<script>
const DISTRICTS = {json.dumps(dist_payload)};
const dSel = document.getElementById('distSelect');
DISTRICTS.forEach(d => {{
  const o = document.createElement('option');
  o.value = d.name; o.textContent = `${{d.name}} (${{d.id}})`;
  dSel.appendChild(o);
}});
dSel.value = 'Pune';

function updateCalc() {{
  const dName = dSel.value;
  const dObj = DISTRICTS.find(x => x.name === dName);
  const mode = document.getElementById('anchorSelect').value;
  const inc = parseFloat(document.getElementById('incInput').value);
  const adm = parseFloat(document.getElementById('admInput').value) / 100.0;
  const icu = parseFloat(document.getElementById('icuInput').value) / 100.0;
  const los = parseFloat(document.getElementById('losInput').value);
  const testsPerCase = parseFloat(document.getElementById('testInput').value);
  const genCap = parseFloat(document.getElementById('genCapInput').value) || 1;
  const icuCap = parseFloat(document.getElementById('icuCapInput').value) || 1;

  document.getElementById('incLbl').textContent = (inc >= 0 ? '+' : '') + inc + '%';
  document.getElementById('admLbl').textContent = Math.round(adm * 100) + '%';
  document.getElementById('icuLbl').textContent = Math.round(icu * 100) + '%';
  document.getElementById('losLbl').textContent = los.toFixed(1) + ' d';
  document.getElementById('testLbl').textContent = testsPerCase.toFixed(1);

  let base = null;
  if (mode === 'custom') base = parseFloat(document.getElementById('customCasesInput').value);
  else base = dObj[mode];

  if (base === null || isNaN(base)) {{
    document.getElementById('statusBox').innerHTML =
      `<b>${{dObj.name}} (${{dObj.id}})</b>: Insufficient Surveillance Data for selected anchor (0 reported outbreaks in 2023–2025). Switch to 'User-Entered Custom Weekly Cases' to run a hypothetical scenario.`;
    ['kpiCases','kpiAdm','kpiBedDays','kpiGenBeds','kpiIcuBeds','kpiTests'].forEach(id => document.getElementById(id).textContent = 'Insuff. Data');
    return;
  }}

  const projCases = base * (1.0 + inc / 100.0);
  const projAdm = projCases * adm;
  const bedDays = projAdm * los;
  const genBeds = (projAdm * (1.0 - icu) * los) / 7.0;
  const icuBeds = (projAdm * icu * los) / 7.0;
  const tests = projCases * testsPerCase;

  document.getElementById('statusBox').innerHTML =
    `<b>${{dObj.name}} (${{dObj.id}} | ${{dObj.division}})</b> — Anchor Baseline: <b>${{base.toFixed(1)}}</b> cases/week | Dominant 2023–25 Outbreak Disease: <b>${{dObj.dom_disease}}</b>`;
  document.getElementById('kpiCases').textContent = projCases.toFixed(1);
  document.getElementById('kpiAdm').textContent = projAdm.toFixed(1);
  document.getElementById('kpiBedDays').textContent = bedDays.toFixed(1);
  document.getElementById('kpiGenBeds').textContent = genBeds.toFixed(1);
  document.getElementById('kpiGenUtil').textContent = `${{((genBeds / genCap) * 100).toFixed(1)}}% of assumed ${{genCap}} beds`;
  document.getElementById('kpiIcuBeds').textContent = icuBeds.toFixed(2);
  document.getElementById('kpiIcuUtil').textContent = `${{((icuBeds / icuCap) * 100).toFixed(1)}}% of assumed ${{icuCap}} ICU beds`;
  document.getElementById('kpiTests').textContent = tests.toFixed(0);
}}

['distSelect','anchorSelect','customCasesInput','incInput','admInput','icuInput','losInput','testInput','genCapInput','icuCapInput'].forEach(id => {{
  document.getElementById(id).addEventListener('input', updateCalc);
}});
updateCalc();
</script>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def generate_all_resource_visualizations(
    completeness_df: pd.DataFrame,
    district_summary_df: pd.DataFrame,
    catalog_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    reports_dir: Path,
    data_dir: Path,
) -> Dict[str, Any]:
    """Generate all 4 static PNG figures and the interactive HTML resource calculator."""
    rep_figs = reports_dir / "figures"
    dat_figs = data_dir / "figures"
    rep_figs.mkdir(parents=True, exist_ok=True)
    dat_figs.mkdir(parents=True, exist_ok=True)

    f1 = plot_01_resource_data_availability_audit(
        completeness_df,
        rep_figs / "01_resource_data_availability_and_completeness_audit.png",
    )
    f2 = plot_02_upstream_epidemiological_demand_drivers(
        district_summary_df,
        rep_figs / "02_upstream_epidemiological_demand_drivers_by_district.png",
    )
    f3 = plot_03_hypothetical_resource_scenario_comparison(
        catalog_df,
        rep_figs / "03_hypothetical_resource_scenario_comparison.png",
    )
    f4 = plot_04_hypothetical_capacity_strain_sensitivity(
        eval_df,
        rep_figs / "04_hypothetical_capacity_strain_sensitivity_matrix.png",
    )

    figs = [f1, f2, f3, f4]
    for fp in figs:
        shutil.copy2(fp, dat_figs / fp.name)

    html_p = generate_interactive_resource_calculator_html(
        district_summary_df,
        reports_dir / "interactive_resource_demand_calculator.html",
    )
    shutil.copy2(html_p, data_dir / "interactive_resource_demand_calculator.html")

    return {
        "static_figures": [str(fp) for fp in figs],
        "interactive_html": str(html_p),
    }
