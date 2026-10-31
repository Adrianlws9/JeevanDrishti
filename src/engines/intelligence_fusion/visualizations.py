"""Visualizations & Interactive HTML Explorer for Predictive Intelligence Fusion Engine (Phase 15).

Generates:
1. `01_upstream_engines_and_reliability_hierarchy_audit.png`
2. `02_weekly_fused_signal_status_and_priority_distribution.png`
3. `03_maharashtra_36_district_fused_intelligence_map_and_matrix.png`
4. `04_engine_disagreement_and_conflict_analysis.png`
5. `05_traceable_why_this_result_decomposition_cards.png`
6. `interactive_predictive_intelligence_fusion_explorer.html`
"""
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.engines.intelligence_fusion.engine_inspector import (
    MANDATORY_FUSION_GOVERNANCE_DISCLAIMER,
)


def generate_fusion_visualizations(
    audit_df: pd.DataFrame,
    weekly_fused_df: pd.DataFrame,
    district_pool_fused_df: pd.DataFrame,
    geo_pool_df: pd.DataFrame,
    fig_dir: Path,
    html_path: Path,
) -> Dict[str, str]:
    """Render all 5 static PNG figures and the standalone interactive HTML fusion explorer."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Figure 1: Upstream Engines & Reliability Hierarchy Audit
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13.0, 6.4), dpi=150)
    eng_names = audit_df["engine_name"].tolist()[::-1]
    tiers = audit_df["reliability_tier"].tolist()[::-1]
    versions = audit_df["model_version"].tolist()[::-1]
    roles = audit_df["fusion_role"].tolist()[::-1]

    tier_color_map = {
        "TIER_1_PRIMARY_EMPIRICAL_PREDICTIVE": "#1b7837",
        "TIER_1_PRIMARY_EMPIRICAL_DETECTION": "#1b7837",
        "TIER_1_PRIMARY_EMPIRICAL_SEVERITY": "#1b7837",
        "TIER_1_EXPLAINABILITY_TRACEABILITY_LAYER": "#2166ac",
        "TIER_2_CONTEXTUAL_ENVIRONMENTAL_ASSOCIATION": "#d95f02",
        "TIER_2_CONTEXTUAL_SPATIAL_CLUSTERING": "#d95f02",
        "TIER_3_HYPOTHETICAL_SCENARIO_CONTEXT_ONLY": "#7570b3",
        "TIER_3_CAPACITY_GAP_AND_WHAT_IF_CALCULATOR": "#7570b3",
    }
    tier_weight_map = {
        "TIER_1_PRIMARY_EMPIRICAL_PREDICTIVE": 1.0,
        "TIER_1_PRIMARY_EMPIRICAL_DETECTION": 1.0,
        "TIER_1_PRIMARY_EMPIRICAL_SEVERITY": 1.0,
        "TIER_1_EXPLAINABILITY_TRACEABILITY_LAYER": 0.92,
        "TIER_2_CONTEXTUAL_ENVIRONMENTAL_ASSOCIATION": 0.65,
        "TIER_2_CONTEXTUAL_SPATIAL_CLUSTERING": 0.65,
        "TIER_3_HYPOTHETICAL_SCENARIO_CONTEXT_ONLY": 0.35,
        "TIER_3_CAPACITY_GAP_AND_WHAT_IF_CALCULATOR": 0.35,
    }

    colors = [tier_color_map.get(t, "#555555") for t in tiers]
    vals = [tier_weight_map.get(t, 0.5) for t in tiers]
    y_pos = np.arange(len(eng_names))

    bars = ax.barh(y_pos, vals, color=colors, edgecolor="#222222", height=0.62, alpha=0.88)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{n} ({v})" for n, v in zip(eng_names, versions)], fontsize=9.2, fontweight="bold")
    ax.set_xlim(0, 1.55)
    ax.set_xticks([0.35, 0.65, 1.0])
    ax.set_xticklabels(
        [
            "Tier 3: Scenario / Capacity Context",
            "Tier 2: Correlational Climate / Spatial",
            "Tier 1: Primary Empirical Pillar",
        ],
        fontsize=8.8,
        fontweight="semibold",
    )
    ax.set_title(
        "Phase 15 Fusion Architecture: 9 Verified Upstream Analytical Engines & Reliability Hierarchy\n"
        "(Transparent Rule-Based Fusion — No Blind Score Averaging Across Incomparable Quantities)",
        fontsize=11.5,
        fontweight="bold",
        pad=12,
    )

    for bar, t_lbl, role_txt in zip(bars, tiers, roles):
        short_role = role_txt if len(role_txt) <= 68 else role_txt[:65] + "..."
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2.0,
            f"[{t_lbl[:6]}] {short_role}",
            va="center",
            ha="left",
            fontsize=8.0,
            color="#111111",
        )

    ax.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    f1 = fig_dir / "01_upstream_engines_and_reliability_hierarchy_audit.png"
    fig.savefig(f1, bbox_inches="tight")
    plt.close(fig)
    saved["engines_hierarchy_png"] = str(f1)

    # ------------------------------------------------------------------
    # Figure 2: Weekly Fused Signal Status & Review Priority Distribution
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 5.6), dpi=150)

    status_order = [
        "Multiple indicators aligned",
        "Conflicting indicators",
        "Elevated signal",
        "No unusual signal detected",
        "Insufficient data",
    ]
    status_colors = {
        "Multiple indicators aligned": "#b2182b",
        "Conflicting indicators": "#e66101",
        "Elevated signal": "#fdb863",
        "No unusual signal detected": "#2166ac",
        "Insufficient data": "#878787",
    }

    s_counts = weekly_fused_df["overall_signal_status"].value_counts()
    s_vals = [int(s_counts.get(k, 0)) for k in status_order]
    s_cols = [status_colors[k] for k in status_order]

    b1 = ax1.barh(status_order[::-1], s_vals[::-1], color=s_cols[::-1], edgecolor="#222222", height=0.58)
    for bar, v in zip(b1, s_vals[::-1]):
        pct = v / max(len(weekly_fused_df), 1) * 100.0
        ax1.text(
            bar.get_width() + 3,
            bar.get_y() + bar.get_height() / 2.0,
            f"{v} ({pct:.1f}%)",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    ax1.set_xlim(0, max(s_vals) * 1.24)
    ax1.set_title(
        "Weekly Fused Signal Status Distribution (513 Records)\n(511 Active Outbreak Weeks + 2 Zero-Surveillance Districts)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax1.set_xlabel("Number of Fused District-Disease-Week Records")
    ax1.grid(axis="x", linestyle="--", alpha=0.35)

    p_counts = weekly_fused_df["recommended_review_priority"].value_counts()
    p_labels = p_counts.index.tolist()[::-1]
    p_vals = p_counts.values.tolist()[::-1]
    b2 = ax2.barh(p_labels, p_vals, color="#386cb0", edgecolor="#222222", height=0.58, alpha=0.88)
    for bar, v in zip(b2, p_vals):
        ax2.text(
            bar.get_width() + 3,
            bar.get_y() + bar.get_height() / 2.0,
            f"{v}",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    ax2.set_xlim(0, max(p_vals) * 1.22)
    ax2.set_title(
        "Recommended Epidemiological Review Priority Breakdown\n(Documented Rule Hierarchy — Never Auto-Declares Outbreaks)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax2.set_xlabel("Number of Fused Records")
    ax2.grid(axis="x", linestyle="--", alpha=0.35)

    plt.tight_layout()
    f2 = fig_dir / "02_weekly_fused_signal_status_and_priority_distribution.png"
    fig.savefig(f2, bbox_inches="tight")
    plt.close(fig)
    saved["signal_status_distribution_png"] = str(f2)

    # ------------------------------------------------------------------
    # Figure 3: Maharashtra 36-District Fused Intelligence Map & Matrix
    # ------------------------------------------------------------------
    fig, (ax_map, ax_tbl) = plt.subplots(1, 2, figsize=(16.0, 6.4), dpi=150)

    geo_all = geo_pool_df[geo_pool_df["disease"] == "ALL_DISEASES"][
        ["district_name", "latitude", "longitude"]
    ].drop_duplicates()
    map_df = district_pool_fused_df.merge(geo_all, on="district_name", how="left")

    for st_name in status_order:
        sub = map_df[map_df["overall_signal_status"] == st_name]
        if sub.empty:
            continue
        ax_map.scatter(
            sub["longitude"],
            sub["latitude"],
            s=140,
            c=status_colors.get(st_name, "#555555"),
            edgecolors="#111111",
            linewidth=1.2,
            label=f"{st_name} ({len(sub)})",
            alpha=0.9,
        )

    for _, r in map_df.iterrows():
        if pd.notna(r.get("longitude")) and pd.notna(r.get("latitude")):
            ax_map.annotate(
                str(r["district_name"])[:12],
                (float(r["longitude"]), float(r["latitude"])),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7.3,
                fontweight="semibold",
            )

    ax_map.set_title(
        "Maharashtra 36-District Pooled Intelligence View (2023–2025_POOL)\n"
        "(District Centroid Geometry — Colored by Fused Signal Status)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_map.set_xlabel("Longitude (°E)")
    ax_map.set_ylabel("Latitude (°N)")
    ax_map.grid(True, linestyle="--", alpha=0.35)
    ax_map.legend(fontsize=8.0, loc="lower right")

    # Right panel: Top 14 Districts Multi-Engine Normalized Indicator Matrix
    top14 = (
        district_pool_fused_df.sort_values(
            by=["verified_outbreak_deaths", "verified_outbreak_cases"],
            ascending=False,
            na_position="last",
        )
        .head(14)
        .copy()
    )
    d_names = top14["district_name"].tolist()
    mat_cols = ["Risk Prob", "Forecast 7d", "Anomaly Max", "Deaths", "Hotspot Gi*"]
    mat_vals = np.zeros((len(top14), 5))
    for i_r, (_, r) in enumerate(top14.iterrows()):
        mat_vals[i_r, 0] = float(r["risk_calibrated_probability"] or 0.0)
        mat_vals[i_r, 1] = min(float(r["forecast_7d_cases"] or 0.0) / 60.0, 1.0)
        mat_vals[i_r, 2] = float(r["anomaly_composite_score"] or 0.0)
        mat_vals[i_r, 3] = min(float(r["verified_outbreak_deaths"] or 0.0) / 25.0, 1.0)
        gi_v = float(r["gi_star_zscore"] or 0.0)
        mat_vals[i_r, 4] = np.clip((gi_v + 2.0) / 5.0, 0.0, 1.0)

    im = ax_tbl.imshow(mat_vals, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=1.0)
    ax_tbl.set_xticks(np.arange(len(mat_cols)))
    ax_tbl.set_xticklabels(mat_cols, fontsize=9, fontweight="bold")
    ax_tbl.set_yticks(np.arange(len(d_names)))
    ax_tbl.set_yticklabels(
        [f"{d} [{s[:18]}]" for d, s in zip(d_names, top14["overall_signal_status"])],
        fontsize=8.2,
        fontweight="bold",
    )

    for i_r, (_, r) in enumerate(top14.iterrows()):
        raw_txts = [
            f"{float(r['risk_calibrated_probability'] or 0):.2f}",
            f"{float(r['forecast_7d_cases'] or 0):.1f}",
            f"{float(r['anomaly_composite_score'] or 0):.2f}",
            f"{int(float(r['verified_outbreak_deaths'] or 0))}",
            f"{float(r['gi_star_zscore'] or 0):+.2f}",
        ]
        for j_c, txt in enumerate(raw_txts):
            ax_tbl.text(
                j_c,
                i_r,
                txt,
                ha="center",
                va="center",
                fontsize=8.0,
                fontweight="bold",
                color="white" if mat_vals[i_r, j_c] > 0.62 else "#111111",
            )

    ax_tbl.set_title(
        "Top 14 Districts Multi-Engine Signal Profile (Raw Quantities Preserved)\n"
        "(Illustrates Why Engines Are Rule-Fused Rather Than Averaged)",
        fontsize=10.2,
        fontweight="bold",
    )
    plt.colorbar(im, ax=ax_tbl, fraction=0.035, pad=0.03, label="Relative Signal Intensity")

    plt.tight_layout()
    f3 = fig_dir / "03_maharashtra_36_district_fused_intelligence_map_and_matrix.png"
    fig.savefig(f3, bbox_inches="tight")
    plt.close(fig)
    saved["district_map_matrix_png"] = str(f3)

    # ------------------------------------------------------------------
    # Figure 4: Engine Disagreement & Conflict Analysis (Section 6)
    # ------------------------------------------------------------------
    fig, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(15.0, 5.6), dpi=150)

    d_counts = weekly_fused_df["disagreement_type"].value_counts()
    d_lbls = d_counts.index.tolist()[::-1]
    d_vals = d_counts.values.tolist()[::-1]
    d_cols = ["#d95f02" if "DISAGREEMENT" in k else "#2b8cbe" for k in d_lbls]

    bars_d = ax_c1.barh(d_lbls, d_vals, color=d_cols, edgecolor="#222222", height=0.55)
    for bar, v in zip(bars_d, d_vals):
        ax_c1.text(
            bar.get_width() + 3,
            bar.get_y() + bar.get_height() / 2.0,
            f"{v}",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    ax_c1.set_xlim(0, max(d_vals) * 1.22)
    ax_c1.set_title(
        "Engine Agreement vs. Disagreement Classification (513 Weekly Records)\n"
        "(All Disagreements Preserved Intact for Epidemiological Analyst Review)",
        fontsize=10.2,
        fontweight="bold",
    )
    ax_c1.set_xlabel("Record Count")
    ax_c1.grid(axis="x", linestyle="--", alpha=0.35)

    # Scatter plot of Risk Probability vs Anomaly Composite Score colored by Overall Signal Status
    valid_sc = weekly_fused_df.dropna(
        subset=["risk_calibrated_probability", "anomaly_composite_score"]
    )
    for st_name in status_order:
        sub_s = valid_sc[valid_sc["overall_signal_status"] == st_name]
        if sub_s.empty:
            continue
        ax_c2.scatter(
            sub_s["risk_calibrated_probability"],
            sub_s["anomaly_composite_score"],
            s=36,
            c=status_colors.get(st_name, "#555555"),
            alpha=0.78,
            edgecolors="#222222",
            linewidth=0.4,
            label=f"{st_name} (n={len(sub_s)})",
        )

    ax_c2.axvline(0.265, color="#1b7837", linestyle="--", lw=1.4, label="Risk Threshold (0.265)")
    ax_c2.axhline(0.55, color="#b2182b", linestyle="--", lw=1.4, label="Anomaly Threshold (0.55)")
    ax_c2.set_xlabel("Phase 5 Calibrated Outbreak Risk Probability")
    ax_c2.set_ylabel("Phase 7 Composite Anomaly Score")
    ax_c2.set_title(
        "Engine Disagreement Quadrants: Risk Classifier vs. Anomaly Detector\n"
        "(Upper-Left & Lower-Right Quadrants Show Preserved Inter-Engine Disagreements)",
        fontsize=10.2,
        fontweight="bold",
    )
    ax_c2.grid(True, linestyle="--", alpha=0.35)
    ax_c2.legend(fontsize=7.8, loc="upper right")

    plt.tight_layout()
    f4 = fig_dir / "04_engine_disagreement_and_conflict_analysis.png"
    fig.savefig(f4, bbox_inches="tight")
    plt.close(fig)
    saved["disagreement_analysis_png"] = str(f4)

    # ------------------------------------------------------------------
    # Figure 5: Traceable "Why This Result?" Decomposition Cards (Section 7)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14.5, 6.8), dpi=150)
    ax.axis("off")

    card_examples: List[pd.Series] = []
    for st_name in status_order:
        sub = weekly_fused_df[weekly_fused_df["overall_signal_status"] == st_name]
        if not sub.empty:
            card_examples.append(sub.iloc[0])

    ax.text(
        0.5,
        0.97,
        "Traceable 'Why This Result?' Evidence Cards Across the 5 Canonical Fused Statuses (Section 7)",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    y_top = 0.90
    card_h = 0.165
    for idx, ex in enumerate(card_examples):
        y0 = y_top - idx * (card_h + 0.015)
        st_lbl = str(ex["overall_signal_status"])
        col = status_colors.get(st_lbl, "#555555")
        rect = plt.Rectangle(
            (0.02, y0 - card_h),
            0.96,
            card_h,
            facecolor="#f8fafc",
            edgecolor=col,
            linewidth=2.2,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)

        hdr = (
            f"[{st_lbl.upper()}] {ex['district_name']} ({ex['district_id']}) — "
            f"{ex['disease']} | Period: {ex['reporting_period']} | "
            f"Priority: {ex['recommended_review_priority']}"
        )
        ax.text(
            0.035,
            y0 - 0.032,
            hdr,
            fontsize=9.3,
            fontweight="bold",
            color=col,
            transform=ax.transAxes,
        )

        why_txt = str(ex["why_this_result_explanation"])
        if len(why_txt) > 290:
            why_txt = why_txt[:287] + "..."
        ax.text(
            0.035,
            y0 - 0.095,
            f"Why this result?: {why_txt}",
            fontsize=8.1,
            color="#1e293b",
            transform=ax.transAxes,
            wrap=True,
        )
        ax.text(
            0.035,
            y0 - 0.145,
            f"Freshness: {ex['model_freshness_status']} | Quality: {ex['data_quality_status']}",
            fontsize=7.8,
            color="#475569",
            fontweight="semibold",
            transform=ax.transAxes,
        )

    plt.tight_layout()
    f5 = fig_dir / "05_traceable_why_this_result_decomposition_cards.png"
    fig.savefig(f5, bbox_inches="tight")
    plt.close(fig)
    saved["why_this_result_cards_png"] = str(f5)

    # ------------------------------------------------------------------
    # Interactive HTML Unified District Intelligence Explorer
    # ------------------------------------------------------------------
    _write_interactive_fusion_html(weekly_fused_df, district_pool_fused_df, html_path)
    saved["interactive_fusion_html"] = str(html_path)
    return saved


def _write_interactive_fusion_html(
    weekly_fused_df: pd.DataFrame,
    district_pool_fused_df: pd.DataFrame,
    html_path: Path,
) -> None:
    """Create a standalone interactive HTML/JS Unified District Intelligence Fusion Explorer."""
    pool_records = json.loads(district_pool_fused_df.to_json(orient="records"))
    weekly_records = json.loads(weekly_fused_df.to_json(orient="records"))

    html_str = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>JEEVAN-DRISHTI — Predictive Intelligence Fusion Engine Explorer (Phase 15)</title>
<style>
  body {{ font-family: 'Segoe UI', Roboto, Arial, sans-serif; margin: 0; padding: 0; background: #f1f5f9; color: #0f172a; }}
  header {{ background: #0f172a; color: #ffffff; padding: 16px 24px; }}
  header h1 {{ margin: 0 0 4px 0; font-size: 20px; }}
  .disclaimer {{ background: #fff1f2; border-left: 6px solid #e11d48; color: #881337; padding: 11px 20px; font-size: 12.5px; font-weight: 600; }}
  .wrap {{ padding: 18px 24px; }}
  .controls {{ display: flex; gap: 14px; flex-wrap: wrap; background: #ffffff; padding: 14px 18px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 16px; }}
  .controls div {{ flex: 1; min-width: 180px; }}
  label {{ display: block; font-size: 11.5px; font-weight: 700; color: #475569; text-transform: uppercase; margin-bottom: 4px; }}
  select {{ width: 100%; padding: 7px; border-radius: 4px; border: 1px solid #cbd5e1; font-size: 13px; }}
  .card {{ background: #ffffff; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); margin-bottom: 16px; }}
  .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 11.5px; font-weight: 700; }}
  .st-aligned {{ background: #fee2e2; color: #991b1b; }}
  .st-conflict {{ background: #ffedd5; color: #9a3412; }}
  .st-elevated {{ background: #fef9c3; color: #854d0e; }}
  .st-normal {{ background: #dbeafe; color: #1e40af; }}
  .st-insuff {{ background: #e2e8f0; color: #334155; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12.5px; margin-top: 10px; }}
  th, td {{ border: 1px solid #e2e8f0; padding: 8px; text-align: left; vertical-align: top; }}
  th {{ background: #f8fafc; font-weight: 700; }}
</style>
</head>
<body>
<header>
  <h1>JEEVAN-DRISHTI — Central Predictive Intelligence Fusion Engine (Phase 15)</h1>
  <div style="font-size:12.5px; color:#94a3b8;">Unified Multi-Engine District Intelligence, Disagreement Preservation & Traceable 'Why This Result?' Explanations</div>
</header>
<div class="disclaimer">
  ⚠️ GOVERNANCE SAFEGUARD: {MANDATORY_FUSION_GOVERNANCE_DISCLAIMER}
</div>
<div class="wrap">
  <div class="controls">
    <div>
      <label>View Mode (Temporal Grain)</label>
      <select id="selGrain" onchange="renderView()">
        <option value="weekly">Weekly Outbreak Events (513 Records)</option>
        <option value="pooled">36-District Pooled Intelligence (2023-2025_POOL)</option>
      </select>
    </div>
    <div>
      <label>Filter by District</label>
      <select id="selDistrict" onchange="renderView()">
        <option value="ALL">All 36 Maharashtra Districts</option>
      </select>
    </div>
    <div>
      <label>Filter by Fused Signal Status</label>
      <select id="selStatus" onchange="renderView()">
        <option value="ALL">All 5 Canonical Signal Statuses</option>
        <option value="Multiple indicators aligned">Multiple indicators aligned</option>
        <option value="Conflicting indicators">Conflicting indicators</option>
        <option value="Elevated signal">Elevated signal</option>
        <option value="No unusual signal detected">No unusual signal detected</option>
        <option value="Insufficient data">Insufficient data</option>
      </select>
    </div>
  </div>

  <div class="card">
    <h2 style="margin-top:0; font-size:16px;" id="lblCount">Fused Intelligence Records</h2>
    <table id="tblFused">
      <thead>
        <tr>
          <th>District & Period</th>
          <th>Disease</th>
          <th>Fused Signal Status & Priority</th>
          <th>Upstream Engine Outputs (Preserved)</th>
          <th>Supporting & Conflicting Evidence</th>
          <th>Traceable "Why This Result?" Explanation</th>
        </tr>
      </thead>
      <tbody></tbody>
    </table>
  </div>
</div>

<script>
const WEEKLY = {json.dumps(weekly_records)};
const POOLED = {json.dumps(pool_records)};

function badgeClass(st) {{
  if (st === 'Multiple indicators aligned') return 'badge st-aligned';
  if (st === 'Conflicting indicators') return 'badge st-conflict';
  if (st === 'Elevated signal') return 'badge st-elevated';
  if (st === 'Insufficient data') return 'badge st-insuff';
  return 'badge st-normal';
}}

function init() {{
  const selD = document.getElementById('selDistrict');
  const dists = [...new Set(POOLED.map(r => r.district_name))].sort();
  dists.forEach(d => {{
    const o = document.createElement('option');
    o.value = d; o.textContent = d;
    selD.appendChild(o);
  }});
  renderView();
}}

function renderView() {{
  const grain = document.getElementById('selGrain').value;
  const dist = document.getElementById('selDistrict').value;
  const st = document.getElementById('selStatus').value;
  const src = (grain === 'weekly') ? WEEKLY : POOLED;

  const filtered = src.filter(r => {{
    if (dist !== 'ALL' && r.district_name !== dist) return false;
    if (st !== 'ALL' && r.overall_signal_status !== st) return false;
    return true;
  }});

  document.getElementById('lblCount').textContent = `Showing ${{Math.min(filtered.length, 100)}} of ${{filtered.length}} Matching Fused Intelligence Records`;
  const tbody = document.querySelector('#tblFused tbody');
  tbody.innerHTML = '';

  filtered.slice(0, 100).forEach(r => {{
    const tr = document.createElement('tr');
    const rProb = r.risk_calibrated_probability !== null ? r.risk_calibrated_probability.toFixed(3) : 'NaN';
    const fc7 = r.forecast_7d_cases !== null ? r.forecast_7d_cases.toFixed(1) : 'NaN';
    const aScore = r.anomaly_composite_score !== null ? r.anomaly_composite_score.toFixed(3) : 'NaN';
    const deaths = r.verified_outbreak_deaths !== null ? r.verified_outbreak_deaths : 'NaN';
    const gi = r.gi_star_zscore !== null ? r.gi_star_zscore.toFixed(2) : 'NaN';

    tr.innerHTML =
      `<td><strong>${{r.district_name}}</strong> (${{r.district_id}})<br/><span style="color:#64748b;font-size:11.5px;">${{r.reporting_period}}</span></td>` +
      `<td><strong>${{r.disease}}</strong></td>` +
      `<td><span class="${{badgeClass(r.overall_signal_status)}}">${{r.overall_signal_status}}</span><br/><div style="font-size:11px;margin-top:5px;color:#334155;"><strong>Priority:</strong> ${{r.recommended_review_priority}}</div></td>` +
      `<td style="font-size:11.5px;line-height:1.45;">` +
        `• <strong>Risk Prob:</strong> ${{rProb}} (${{r.risk_category}})<br/>` +
        `• <strong>Forecast 7d:</strong> ${{fc7}} cases<br/>` +
        `• <strong>Anomaly Score:</strong> ${{aScore}} (Flag=${{r.anomaly_flagged}})<br/>` +
        `• <strong>Verified Deaths:</strong> ${{deaths}} (CFR=${{r.outbreak_cluster_cfr_pct !== null ? r.outbreak_cluster_cfr_pct + '%' : 'NaN'}})<br/>` +
        `• <strong>Hotspot Gi*:</strong> ${{gi}} (${{r.gi_star_class}})` +
      `</td>` +
      `<td style="font-size:11.5px;line-height:1.45;">` +
        `<div><strong>Supporting:</strong> ${{r.supporting_analytical_evidence || 'None'}}</div>` +
        (r.conflicting_evidence ? `<div style="color:#b91c1c;margin-top:4px;"><strong>Conflict Preserved:</strong> ${{r.conflicting_evidence}}</div>` : '') +
      `</td>` +
      `<td style="font-size:11.5px;line-height:1.45;">${{r.why_this_result_explanation}}</td>`;
    tbody.appendChild(tr);
  }});
}}

window.onload = init;
</script>
</body>
</html>
"""
    html_path.write_text(html_str, encoding="utf-8")
