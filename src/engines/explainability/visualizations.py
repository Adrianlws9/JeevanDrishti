"""Static & Interactive Visualization Suite for Phase 13 Explainable AI (XAI) Engine."""
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.engines.explainability.model_inspector import MANDATORY_XAI_NON_CAUSAL_DISCLAIMER


def plot_01_global_risk_classifier_explainability(
    global_df: pd.DataFrame, out_path: Path
) -> Path:
    """Figure 1: Phase 5 Outbreak Risk Classifier Multi-Method Global Feature Importance."""
    p5 = (
        global_df[global_df["phase"] == "Phase 5"]
        .sort_values("composite_rank_score", ascending=False)
        .head(14)
        .iloc[::-1]
        .reset_index(drop=True)
    )

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.0), dpi=160)
    fig.suptitle(
        "JEEVAN-DRISHTI Phase 13: Phase 5 Outbreak Risk Classifier Global Explainability (phase5-v1.0.0)\n"
        "[Comparing XGBoost Exact TreeSHAP, HistGradientBoosting Held-Out Permutation Importance, RF Gini & LogReg β]",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )

    y_pos = np.arange(len(p5))
    ax1 = axes[0]
    ax1.barh(
        y_pos + 0.18,
        p5["xgb_mean_abs_treeshap"],
        height=0.35,
        color="#DC2626",
        edgecolor="#0F172A",
        label="XGBoost Mean |TreeSHAP| (Log-Odds)",
    )
    ax1.barh(
        y_pos - 0.18,
        p5["hgb_primary_permutation_importance_pr_auc"] * 10.0,
        height=0.35,
        color="#2563EB",
        edgecolor="#0F172A",
        label="HistGradientBoosting Permutation ΔPR-AUC (×10)",
    )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(p5["feature_name"], fontsize=8.5, fontweight="bold")
    ax1.set_xlabel("Global Attribution Magnitude", fontsize=9.5, fontweight="bold")
    ax1.set_title("A. Top 14 Risk Features: Exact TreeSHAP vs. Primary HGB Permutation", fontsize=10.3, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8.3)
    ax1.grid(axis="x", linestyle=":", alpha=0.45)

    ax2 = axes[1]
    betas = p5["logreg_standardized_coef_beta"].to_numpy(dtype=float)
    cols = ["#DC2626" if b >= 0 else "#0891B2" for b in betas]
    ax2.barh(y_pos, betas, color=cols, edgecolor="#0F172A", height=0.6)
    ax2.axvline(0.0, color="#0F172A", linewidth=1.2)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(p5["feature_name"], fontsize=8.5, fontweight="bold")
    ax2.set_xlabel("Standardized Logistic Regression Coefficient (β_j)", fontsize=9.5, fontweight="bold")
    ax2.set_title(
        "B. Signed Linear Direction of Influence (Logistic Regression Baseline β_j)\n[Red = Increases Risk Log-Odds | Blue = Decreases Risk Log-Odds]",
        fontsize=10.0,
        fontweight="bold",
    )
    ax2.grid(axis="x", linestyle=":", alpha=0.45)

    fig.tight_layout(rect=[0, 0.01, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_02_multi_horizon_forecast_domain_contributions(
    global_summary: Dict[str, Any], global_df: pd.DataFrame, out_path: Path
) -> Path:
    """Figure 2: Phase 6 Multi-Horizon Forecast (7d, 14d, 30d) Feature Domain Attribution."""
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.8), dpi=160)
    fig.suptitle(
        "JEEVAN-DRISHTI Phase 13: Phase 6 Multi-Horizon Disease Forecasting Explainability (phase6-v1.0.0)\n"
        "[TreeSHAP Domain Attribution Across 7d (1w), 14d (2w), and 30d (4w) Forecast Horizons]",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )

    # Panel A: Stacked Domain Share (%) across 7d, 14d, 30d
    ax1 = axes[0]
    horizons = ["7d", "14d", "30d"]
    domains = [
        ("historical_disease_lags_and_baselines", "Historical Disease Lags & Baselines", "#2563EB"),
        ("disease_and_district_context", "Disease & District Context", "#7C3AED"),
        ("rainfall_and_temperature_climate", "Rainfall & Temperature Climate Lags", "#059669"),
        ("seasonality_and_calendar", "Seasonality & Calendar Timing", "#D97706"),
        ("recent_case_growth_velocity", "Recent Case Growth Velocity", "#DC2626"),
        ("spatial_neighbor_and_geography", "Spatial Neighbor Activity", "#0891B2"),
    ]
    x = np.arange(len(horizons))
    bottom = np.zeros(len(horizons), dtype=float)
    for d_key, d_lbl, col in domains:
        vals = np.array(
            [
                float(
                    global_summary["phase6_disease_forecasting"]["horizons"][h][
                        "domain_treeshap_share_pct"
                    ].get(d_key, 0.0)
                )
                for h in horizons
            ]
        )
        ax1.bar(x, vals, bottom=bottom, width=0.48, color=col, edgecolor="#0F172A", label=d_lbl)
        bottom += vals

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        ["7-Day (1w Ahead)\n[All Lags ≥ 1w]", "14-Day (2w Ahead)\n[Lags ≥ 2w Only]", "30-Day (4w Ahead)\n[Lags ≥ 4w Only]"],
        fontsize=9.0,
        fontweight="bold",
    )
    ax1.set_ylabel("Share of Total Mean |TreeSHAP| (%)", fontsize=9.5, fontweight="bold")
    ax1.set_ylim(0, 108)
    ax1.set_title("A. Forecast Feature Domain Attribution Share by Horizon", fontsize=10.5, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8.0)
    ax1.grid(axis="y", linestyle=":", alpha=0.45)

    # Panel B: Top 12 Features for 7d Forecast
    ax2 = axes[1]
    f7 = (
        global_df[
            (global_df["phase"] == "Phase 6")
            & (global_df["horizon_or_task"] == "forecast_7d_1w_ahead")
        ]
        .sort_values("composite_rank_score", ascending=False)
        .head(12)
        .iloc[::-1]
        .reset_index(drop=True)
    )
    y2 = np.arange(len(f7))
    ax2.barh(y2, f7["xgb_mean_abs_treeshap"], color="#059669", edgecolor="#0F172A", height=0.62)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(f7["feature_name"], fontsize=8.5, fontweight="bold")
    ax2.set_xlabel("Mean |TreeSHAP| on log(1 + Next-Week Cases)", fontsize=9.5, fontweight="bold")
    ax2.set_title("B. Top 12 Global Features for 7-Day (1-Week) Case Forecasting", fontsize=10.5, fontweight="bold")
    ax2.grid(axis="x", linestyle=":", alpha=0.45)

    fig.tight_layout(rect=[0, 0.01, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_03_local_risk_domain_waterfall(
    local_risk_df: pd.DataFrame, out_path: Path
) -> Path:
    """Figure 3: Local Risk Domain Attribution Across Top Flagged High-Risk Outbreak Weeks."""
    top12 = (
        local_risk_df.sort_values("model_estimated_risk_prob", ascending=False)
        .head(12)
        .iloc[::-1]
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=(15.5, 6.8), dpi=160)
    fig.suptitle(
        "JEEVAN-DRISHTI Phase 13: Local Risk Explanation — Domain Log-Odds Contributions for Top 12 High-Risk Weeks\n"
        "[Explaining Why Specific District-Disease-Weeks Received Elevated Calibrated Risk Scores (phase5-v1.0.0)]",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )

    y_pos = np.arange(len(top12))
    labels = [
        f"{r['district']} | {r['disease']}\n({r['reporting_period']}, P={r['model_estimated_risk_prob']:.2f})"
        for _, r in top12.iterrows()
    ]

    dom_cols = [
        ("historical_disease_contribution", "Historical Disease Lags & Baselines", "#2563EB"),
        ("case_growth_contribution", "Recent Case Growth Velocity", "#DC2626"),
        ("climate_contribution", "Rainfall & Temperature Climate Lags", "#059669"),
        ("seasonal_contribution", "Seasonality & Calendar Timing", "#D97706"),
        ("spatial_neighbor_contribution", "Spatial Neighbor Activity", "#7C3AED"),
    ]
    h = 0.15
    offsets = [-0.30, -0.15, 0.0, 0.15, 0.30]
    for (col_name, lbl, color), off in zip(dom_cols, offsets):
        ax.barh(
            y_pos + off,
            top12[col_name],
            height=h,
            color=color,
            edgecolor="#0F172A",
            label=lbl,
        )

    ax.axvline(0.0, color="#0F172A", linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.2, fontweight="bold")
    ax.set_xlabel("Signed Local TreeSHAP Log-Odds Contribution", fontsize=9.5, fontweight="bold")
    ax.set_title(
        "Signed Local Feature Domain Contributions (+ Increases Risk | - Decreases Risk)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="x", linestyle=":", alpha=0.45)

    fig.tight_layout(rect=[0, 0.01, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_04_anomaly_evidence_and_iforest_sensitivity(
    local_anom_df: pd.DataFrame, global_df: pd.DataFrame, out_path: Path
) -> Path:
    """Figure 4: Anomaly Detection Evidence Decomposition & Isolation Forest Sensitivity."""
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.8), dpi=160)
    fig.suptitle(
        "JEEVAN-DRISHTI Phase 13: Phase 7 Anomaly Detection Explainability (phase7-v1.0.0)\n"
        "[Left: Top 12 Flagged Anomalies (Z-Scores & Composite Score) | Right: Isolation Forest Feature Perturbation Sensitivity]",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )

    top_a = (
        local_anom_df[local_anom_df["is_anomaly"] == True]
        .sort_values("composite_anomaly_score", ascending=False)
        .head(12)
        .iloc[::-1]
        .reset_index(drop=True)
    )
    y1 = np.arange(len(top_a))
    ax1 = axes[0]
    ax1.barh(
        y1 + 0.18,
        np.clip(top_a["deviation_seasonal_z"], -2, 15),
        height=0.35,
        color="#DC2626",
        edgecolor="#0F172A",
        label="Seasonal Baseline Z-Score (clipped at 15)",
    )
    ax1.barh(
        y1 - 0.18,
        np.clip(top_a["deviation_robust_mad_z"], -2, 15),
        height=0.35,
        color="#D97706",
        edgecolor="#0F172A",
        label="Robust MAD Z-Score (clipped at 15)",
    )
    ax1.axvline(2.0, color="#0F172A", linestyle="--", linewidth=1.2, label="Z = 2.0 Anomaly Threshold")
    ax1.set_yticks(y1)
    ax1.set_yticklabels(
        [
            f"{r['district']} | {r['disease']}\n({r['reporting_period']}, Cases={int(r['observed_value'])})"
            for _, r in top_a.iterrows()
        ],
        fontsize=8.0,
        fontweight="bold",
    )
    ax1.set_xlabel("Baseline Deviation Z-Score", fontsize=9.5, fontweight="bold")
    ax1.set_title("A. Top 12 Flagged Anomalies: Seasonal & Robust MAD Z-Score Evidence", fontsize=10.3, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8.0)
    ax1.grid(axis="x", linestyle=":", alpha=0.45)

    # Panel B: Isolation Forest 14 Features Sensitivity
    ax2 = axes[1]
    p7 = (
        global_df[global_df["phase"] == "Phase 7"]
        .sort_values("composite_rank_score", ascending=True)
        .reset_index(drop=True)
    )
    y2 = np.arange(len(p7))
    ax2.barh(y2, p7["composite_rank_score"], color="#7C3AED", edgecolor="#0F172A", height=0.62)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(p7["feature_name"], fontsize=8.5, fontweight="bold")
    ax2.set_xlabel("Mean |Δ Isolation Forest Anomaly Score| Under +1 SD Perturbation", fontsize=9.2, fontweight="bold")
    ax2.set_title("B. Isolation Forest Feature Sensitivity (All 14 Engineered Features)", fontsize=10.3, fontweight="bold")
    ax2.grid(axis="x", linestyle=":", alpha=0.45)

    fig.tight_layout(rect=[0, 0.01, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_05_why_district_flagged_synthesis(
    panels_df: pd.DataFrame, out_path: Path
) -> Path:
    """Figure 5: Multi-Engine 'Why was this district flagged?' Synthesis Across Top 16 Districts."""
    top16 = (
        panels_df[panels_df["peak_risk_probability"].notna()]
        .head(16)
        .iloc[::-1]
        .reset_index(drop=True)
    )

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.0), dpi=160)
    fig.suptitle(
        "JEEVAN-DRISHTI Phase 13: 'Why Was This District Flagged?' Multi-Engine Evidence Synthesis\n"
        "[Combining Phase 5 Risk Probability, Phase 7 Anomaly Score, and Phase 9 Getis-Ord Gi* Hotspot Z-Score]",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )

    y_pos = np.arange(len(top16))
    ax1 = axes[0]
    ax1.barh(
        y_pos + 0.18,
        top16["peak_risk_probability"],
        height=0.35,
        color="#DC2626",
        edgecolor="#0F172A",
        label="Peak Calibrated Outbreak Risk Probability (phase5-v1.0.0)",
    )
    ax1.barh(
        y_pos - 0.18,
        top16["peak_anomaly_score"],
        height=0.35,
        color="#D97706",
        edgecolor="#0F172A",
        label="Peak Composite Anomaly Score (phase7-v1.0.0)",
    )
    ax1.axvline(0.35, color="#DC2626", linestyle="--", linewidth=1.2, label="High-Risk Cutoff (0.35)")
    ax1.axvline(0.55, color="#D97706", linestyle=":", linewidth=1.2, label="Anomaly Cutoff (0.55)")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top16["district"], fontsize=8.8, fontweight="bold")
    ax1.set_xlim(0, 1.08)
    ax1.set_xlabel("Probability / Normalized Score [0..1]", fontsize=9.5, fontweight="bold")
    ax1.set_title("A. Top 16 Flagged Districts: Peak Risk vs. Peak Anomaly Score", fontsize=10.3, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=8.0)
    ax1.grid(axis="x", linestyle=":", alpha=0.45)

    ax2 = axes[1]
    gi_vals = top16["gi_star_zscore"].to_numpy(dtype=float)
    cols = ["#DC2626" if z >= 1.65 else ("#2563EB" if z > 0 else "#64748B") for z in gi_vals]
    ax2.barh(y_pos, gi_vals, color=cols, edgecolor="#0F172A", height=0.6)
    ax2.axvline(0.0, color="#0F172A", linewidth=1.1)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(
        [f"{r['district']} ({r['geospatial_hotspot_class']})" for _, r in top16.iterrows()],
        fontsize=8.2,
        fontweight="bold",
    )
    ax2.set_xlabel("Geospatial Getis-Ord Gi* Z-Score (phase9-v1.0.0)", fontsize=9.5, fontweight="bold")
    ax2.set_title("B. Spatial Clustering Evidence (Getis-Ord Gi* Z-Score & Hotspot Status)", fontsize=10.3, fontweight="bold")
    ax2.grid(axis="x", linestyle=":", alpha=0.45)

    fig.tight_layout(rect=[0, 0.01, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_interactive_why_flagged_html(
    panel_payload: Dict[str, Any], out_path: Path
) -> Path:
    """Generate a standalone interactive HTML/JS 'Why was this district flagged?' Explorer."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>JEEVAN-DRISHTI Phase 13 — Why Was This District Flagged? (Explainable AI Panel)</title>
<style>
  :root {{
    --bg: #0f172a; --panel: #1e293b; --border: #334155; --text: #f8fafc;
    --muted: #94a3b8; --accent: #38bdf8; --warn: #f59e0b; --danger: #ef4444; --ok: #22c55e;
  }}
  * {{ box-sizing: border-box; font-family: system-ui, -apple-system, sans-serif; }}
  body {{ margin: 0; background: var(--bg); color: var(--text); padding: 16px; }}
  .banner {{
    background: #31102F; border: 1px solid var(--accent); color: #E0F2FE;
    padding: 10px 14px; border-radius: 8px; margin-bottom: 14px; font-size: 12.5px; font-weight: 600;
  }}
  .top-bar {{ display: flex; gap: 12px; align-items: center; margin-bottom: 14px; }}
  select {{
    background: var(--panel); color: var(--text); border: 1px solid var(--border);
    padding: 8px 12px; border-radius: 8px; font-size: 14px; font-weight: 600; min-width: 280px;
  }}
  .grid {{ display: grid; grid-template-columns: 1.15fr 1fr; gap: 14px; }}
  .card {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px; margin-bottom: 14px; }}
  h2 {{ margin: 0 0 8px 0; font-size: 16px; color: var(--accent); }}
  h3 {{ margin: 0 0 8px 0; font-size: 13.5px; color: #cbd5e1; text-transform: uppercase; letter-spacing: 0.5px; }}
  p {{ margin: 0; font-size: 13px; line-height: 1.55; color: #e2e8f0; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; margin-top: 6px; }}
  th, td {{ border-bottom: 1px solid var(--border); padding: 6px 8px; text-align: left; }}
  th {{ color: var(--muted); font-weight: 600; }}
  .badge {{ padding: 3px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; }}
</style>
</head>
<body>
<div class="banner">
  🔍 <b>Why was this district flagged?</b> — {MANDATORY_XAI_NON_CAUSAL_DISCLAIMER}
</div>
<div class="top-bar">
  <label style="font-weight:700;font-size:14px;">Select Maharashtra District:</label>
  <select id="distSelect"></select>
  <span id="tierBadge" class="badge"></span>
</div>
<div class="grid">
  <div>
    <div class="card">
      <h3>1. Main Multi-Engine Evidence</h3>
      <p id="mainEvidence"></p>
    </div>
    <div class="card">
      <h3>2. Historical Surveillance Comparison (2023–2025)</h3>
      <p id="histComparison"></p>
    </div>
    <div class="card">
      <h3>3. Data Quality & Coverage Assessment</h3>
      <p id="dataQuality"></p>
    </div>
  </div>
  <div>
    <div class="card">
      <h3>4. Important Contributing Features (Peak Risk Event TreeSHAP)</h3>
      <table id="featTable">
        <thead><tr><th>Feature Name</th><th>Signed Log-Odds TreeSHAP</th></tr></thead>
        <tbody></tbody>
      </table>
    </div>
    <div class="card">
      <h3>5. Domain Attribution & Model Versions</h3>
      <div id="modelVersions" style="font-size:12.5px;line-height:1.6;"></div>
      <div style="margin-top:8px;padding-top:8px;border-top:1px solid var(--border);font-size:11.5px;color:var(--muted);" id="limText"></div>
    </div>
  </div>
</div>
<script>
const PAYLOAD = {json.dumps(panel_payload)};
const sel = document.getElementById('distSelect');
PAYLOAD.districts.forEach(d => {{
  const opt = document.createElement('option');
  opt.value = d.district_name;
  opt.textContent = `${{d.district_name}} (${{d.district_id}} — ${{d.flag_priority_tier}})`;
  sel.appendChild(opt);
}});
sel.value = 'Gadchiroli';

function renderDistrict() {{
  const d = PAYLOAD.districts.find(x => x.district_name === sel.value);
  const b = document.getElementById('tierBadge');
  b.textContent = d.flag_priority_tier;
  b.style.background = d.is_flagged ? '#7f1d1d' : '#1e3a8a';
  b.style.color = '#fff';

  document.getElementById('mainEvidence').textContent = d.main_evidence;
  document.getElementById('histComparison').textContent = d.historical_comparison.summary_text || 'Insufficient 2023–2025 surveillance records.';
  document.getElementById('dataQuality').textContent = d.data_quality.details || d.data_quality.status;

  const tb = document.querySelector('#featTable tbody');
  tb.innerHTML = '';
  if (!d.important_contributing_features || d.important_contributing_features.length === 0) {{
    tb.innerHTML = '<tr><td colspan="2">No 2023–2025 outbreak events available for local feature attribution.</td></tr>';
  }} else {{
    d.important_contributing_features.forEach(item => {{
      const tr = document.createElement('tr');
      const v = item.log_odds_treeshap;
      tr.innerHTML = `<td><code>${{item.feature}}</code></td><td style="font-weight:700;color:${{v >= 0 ? '#f87171' : '#38bdf8'}}">${{v >= 0 ? '+' : ''}}${{v.toFixed(4)}}</td>`;
      tb.appendChild(tr);
    }});
  }}

  const mv = d.model_versions;
  document.getElementById('modelVersions').innerHTML =
    `<b>Model Versions Used:</b> Risk Classifier: <code>${{mv.risk_classifier}}</code> | Forecaster: <code>${{mv.forecaster}}</code> | Anomaly Detector: <code>${{mv.anomaly_detector}}</code> | Geospatial Hotspot: <code>${{mv.geospatial_hotspot}}</code> | XAI Engine: <code>${{mv.xai_engine}}</code>`;
  document.getElementById('limText').textContent = d.limitations;
}}
sel.addEventListener('change', renderDistrict);
renderDistrict();
</script>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def generate_all_explainability_visualizations(
    global_df: pd.DataFrame,
    global_summary: Dict[str, Any],
    local_risk_df: pd.DataFrame,
    local_anom_df: pd.DataFrame,
    panels_df: pd.DataFrame,
    panel_payload: Dict[str, Any],
    reports_dir: Path,
    data_dir: Path,
) -> Dict[str, Any]:
    """Generate all 5 static PNG figures and the interactive HTML 'Why was this district flagged?' explorer."""
    rep_figs = reports_dir / "figures"
    dat_figs = data_dir / "figures"
    rep_figs.mkdir(parents=True, exist_ok=True)
    dat_figs.mkdir(parents=True, exist_ok=True)

    f1 = plot_01_global_risk_classifier_explainability(
        global_df,
        rep_figs / "01_global_risk_classifier_explainability_comparison.png",
    )
    f2 = plot_02_multi_horizon_forecast_domain_contributions(
        global_summary,
        global_df,
        rep_figs / "02_global_multi_horizon_forecast_feature_contributions.png",
    )
    f3 = plot_03_local_risk_domain_waterfall(
        local_risk_df,
        rep_figs / "03_local_risk_waterfall_and_domain_attribution.png",
    )
    f4 = plot_04_anomaly_evidence_and_iforest_sensitivity(
        local_anom_df,
        global_df,
        rep_figs / "04_anomaly_evidence_and_isolation_forest_attribution.png",
    )
    f5 = plot_05_why_district_flagged_synthesis(
        panels_df,
        rep_figs / "05_why_district_flagged_synthesis_matrix.png",
    )

    figs = [f1, f2, f3, f4, f5]
    for fp in figs:
        shutil.copy2(fp, dat_figs / fp.name)

    html_p = generate_interactive_why_flagged_html(
        panel_payload,
        reports_dir / "interactive_why_district_flagged_explorer.html",
    )
    shutil.copy2(html_p, data_dir / "interactive_why_district_flagged_explorer.html")

    return {
        "static_figures": [str(fp) for fp in figs],
        "interactive_html": str(html_p),
    }
