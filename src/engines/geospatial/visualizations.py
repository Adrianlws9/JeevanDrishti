"""Accessible Geospatial Map Rendering & Hotspot Visualization Suite (Phase 9).

Generates 6 publication-quality, colorblind-accessible maps in `reports/geospatial/figures/`
and `data/geospatial/figures/`:
1. `01_maharashtra_district_disease_burden_map.png`
2. `02_disease_specific_hotspot_maps.png`
3. `03_historical_comparison_maps.png`
4. `04_forecast_disease_maps.png`
5. `05_anomaly_and_risk_intelligence_maps.png`
6. `06_district_drilldown_spatial_profiles.png`

Accessibility & Non-Fabrication Design:
- Official district polygon boundaries (.geojson/.shp) do not exist in the workspace and are
  NOT fabricated. Maps plot the 35 verified district point centroids connected by the
  symmetric k-NN (k=4) spatial weights graph, plus an explicit inset callout for
  `Mumbai Suburban` (missing coordinates) and `Mumbai City` / `Ratnagiri` (Insufficient Data).
- Never relies solely on color to communicate risk or hotspot status: combines colorblind-safe
  colormaps (`YlOrBr`, `Cividis`, `Viridis`, `RdBu_r`), distinct marker shapes (`^` triangle for
  statistically significant hotspot p<0.05, `s` square for High-Low outlier, `o` circle for
  non-significant, `v` downward triangle for coldspot, `X` cross for Insufficient Data),
  and direct district code + value text labels.
"""
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.engines.geospatial.spatial_statistics import SpatialWeightsBundle


SHORT_CODE_MAP: Dict[str, str] = {
    "Ahmednagar": "AHM",
    "Akola": "AKO",
    "Amravati": "AMR",
    "Beed": "BEE",
    "Bhandara": "BHA",
    "Buldhana": "BUL",
    "Chandrapur": "CHA",
    "Chhatrapati Sambhajinagar": "CSN",
    "Dharashiv": "DHA",
    "Dhule": "DHU",
    "Gadchiroli": "GAD",
    "Gondia": "GON",
    "Hingoli": "HIN",
    "Jalgaon": "JLG",
    "Jalna": "JLN",
    "Kolhapur": "KOL",
    "Latur": "LAT",
    "Mumbai City": "MCI",
    "Mumbai Suburban": "MSU",
    "Nagpur": "NAG",
    "Nanded": "NAN",
    "Nandurbar": "NDB",
    "Nashik": "NAS",
    "Palghar": "PAL",
    "Parbhani": "PAR",
    "Pune": "PUN",
    "Raigad": "RAI",
    "Ratnagiri": "RAT",
    "Sangli": "SAN",
    "Satara": "SAT",
    "Sindhudurg": "SIN",
    "Solapur": "SOL",
    "Thane": "THA",
    "Wardha": "WAR",
    "Washim": "WAS",
    "Yavatmal": "YAV",
}


def _draw_spatial_network_edges(ax: plt.Axes, weights_bundle: SpatialWeightsBundle) -> None:
    """Plot the symmetric k-NN (k=4) spatial weights graph edges across the 35 geocoded districts."""
    n = len(weights_bundle.districts)
    for i in range(n):
        for j in range(i + 1, n):
            if weights_bundle.W_binary_sym[i, j] > 0:
                ax.plot(
                    [weights_bundle.lons[i], weights_bundle.lons[j]],
                    [weights_bundle.lats[i], weights_bundle.lats[j]],
                    color="#CBD5E1",
                    linestyle="-",
                    linewidth=0.85,
                    alpha=0.65,
                    zorder=1,
                )


def _style_map_axes(ax: plt.Axes, title: str, subtitle: str = "") -> None:
    ax.set_xlim(72.1, 81.1)
    ax.set_ylim(15.6, 22.1)
    ax.set_xlabel("Longitude (°E)", fontsize=9, fontweight="bold")
    ax.set_ylabel("Latitude (°N)", fontsize=9, fontweight="bold")
    ax.set_title(title if not subtitle else f"{title}\n{subtitle}", fontsize=10.5, fontweight="bold", pad=8)
    ax.grid(True, linestyle=":", alpha=0.35, color="#94A3B8")
    ax.set_facecolor("#F8FAFC")


def plot_01_maharashtra_district_burden_map(
    summary_df: pd.DataFrame,
    weights_bundle: SpatialWeightsBundle,
    out_path: Path,
) -> Path:
    """Figure 1: Maharashtra District-Level Observed Disease Burden & Outbreak Weeks Map (2023-2025)."""
    sub = summary_df[
        (summary_df["disease"] == "ALL_DISEASES")
        & (summary_df["reporting_period"] == "2023-2025_POOL")
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2), dpi=160)
    fig.suptitle(
        "JEEVAN-DRISHTI Phase 9: Maharashtra District-Level Observed Disease Surveillance Map (2023–2025 Weekly IDSP)\n"
        "[Geometry: 35 Verified District Point Centroids + Symmetric k-NN (k=4) Spatial Graph; Zero Fabricated Polygon Boundaries]",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    # Panel A: Total Observed Cases (2023-2025)
    ax = axes[0]
    _draw_spatial_network_edges(ax, weights_bundle)
    geo_valid = sub[sub["latitude"].notna() & (sub["observed_outbreak_weeks"] > 0)].copy()
    geo_insuff = sub[sub["latitude"].notna() & (sub["observed_outbreak_weeks"] == 0)].copy()
    no_coord = sub[sub["latitude"].isna()].copy()

    sc = ax.scatter(
        geo_valid["longitude"],
        geo_valid["latitude"],
        c=geo_valid["observed_cases"],
        s=90 + geo_valid["observed_cases"] * 0.14,
        cmap="YlOrBr",
        edgecolors="#1E293B",
        linewidths=1.2,
        alpha=0.92,
        zorder=3,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.042, pad=0.02)
    cbar.set_label("Observed Cases (2023–2025 Total)", fontsize=9, fontweight="bold")

    # Plot Insufficient Data (0 outbreaks in 2023-2025: Mumbai City, Ratnagiri)
    if not geo_insuff.empty:
        ax.scatter(
            geo_insuff["longitude"],
            geo_insuff["latitude"],
            marker="X",
            s=130,
            color="#64748B",
            edgecolors="#0F172A",
            linewidths=1.3,
            zorder=4,
            label="Insufficient Data (0 Outbreaks in 2023-25)",
        )

    for _, r in sub[sub["latitude"].notna()].iterrows():
        code = SHORT_CODE_MAP.get(str(r["district_name"]), str(r["district_name"])[:3].upper())
        val_txt = (
            f"{code}:{int(r['observed_cases'])}"
            if pd.notna(r["observed_cases"]) and r["observed_outbreak_weeks"] > 0
            else f"{code}:Insuff."
        )
        ax.annotate(
            val_txt,
            (float(r["longitude"]), float(r["latitude"])),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7.2,
            fontweight="bold",
            color="#0F172A",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.72),
            zorder=5,
        )

    # Inset callout for Mumbai Suburban (No coordinates)
    ms_cases = int(no_coord["observed_cases"].iloc[0]) if not no_coord.empty and pd.notna(no_coord["observed_cases"].iloc[0]) else 0
    ms_wks = int(no_coord["observed_outbreak_weeks"].iloc[0]) if not no_coord.empty else 0
    ax.text(
        0.02,
        0.03,
        f"Ungeocoded District Callout:\n• Mumbai Suburban (MH_D19): {ms_cases} cases ({ms_wks} wks)\n  Status: Insufficient Data (No Coordinates in Workspace)\n• Incidence Rate: N/A (No Population Denominator)",
        transform=ax.transAxes,
        fontsize=7.8,
        bbox=dict(boxstyle="round,pad=0.4", fc="#FEF3C7", ec="#D97706", lw=1.0),
        zorder=6,
    )
    _style_map_axes(
        ax,
        "A. Observed Outbreak Cases by District (2023–2025)",
        "Circle Size & Color = Observed Cases | 'X' = Insufficient Data (Mumbai City, Ratnagiri)",
    )

    # Panel B: Active Outbreak Weeks & Hotspot Marker Shapes
    ax2 = axes[1]
    _draw_spatial_network_edges(ax2, weights_bundle)
    marker_specs = [
        ("HOTSPOT", geo_valid[geo_valid["is_statistically_significant_hotspot"]], "^", "#DC2626", "Significant Hotspot (p<0.05) [▲]"),
        ("OUTLIER", geo_valid[(~geo_valid["is_statistically_significant_hotspot"]) & (geo_valid["lisa_quadrant"] == "HL_SPATIAL_OUTLIER")], "s", "#D97706", "High-Low Spatial Outlier (p<0.05) [■]"),
        ("NORMAL", geo_valid[(~geo_valid["is_statistically_significant_hotspot"]) & (geo_valid["lisa_quadrant"] != "HL_SPATIAL_OUTLIER")], "o", "#2563EB", "Not Statistically Significant [●]"),
    ]
    for _, df_m, mk, col, lbl in marker_specs:
        if not df_m.empty:
            ax2.scatter(
                df_m["longitude"],
                df_m["latitude"],
                marker=mk,
                s=85 + df_m["observed_outbreak_weeks"] * 6.5,
                color=col,
                edgecolors="#0F172A",
                linewidths=1.1,
                alpha=0.90,
                label=lbl,
                zorder=4,
            )
    if not geo_insuff.empty:
        ax2.scatter(
            geo_insuff["longitude"],
            geo_insuff["latitude"],
            marker="X",
            s=130,
            color="#64748B",
            edgecolors="#0F172A",
            linewidths=1.3,
            label="Insufficient Data (0 Outbreaks) [✕]",
            zorder=4,
        )

    for _, r in sub[sub["latitude"].notna()].iterrows():
        code = SHORT_CODE_MAP.get(str(r["district_name"]), str(r["district_name"])[:3].upper())
        wks = int(r["observed_outbreak_weeks"])
        tag = "*" if r["is_statistically_significant_hotspot"] else ""
        txt = f"{code}:{wks}w{tag}" if wks > 0 else f"{code}:Insuff"
        ax2.annotate(
            txt,
            (float(r["longitude"]), float(r["latitude"])),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7.2,
            fontweight="bold",
            color="#0F172A",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.72),
            zorder=5,
        )

    ax2.legend(loc="lower right", fontsize=8, framealpha=0.92, title="Multi-Channel Shape & Color Legend")
    _style_map_axes(
        ax2,
        "B. Active Outbreak Weeks & Hotspot Significance (2023–2025)",
        "Shape + Color + '*' Tag Distinguish Verified Hotspots from Non-Significant Burden",
    )

    fig.tight_layout(rect=[0, 0.01, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_02_disease_specific_hotspot_maps(
    summary_df: pd.DataFrame,
    global_moran_df: pd.DataFrame,
    weights_bundle: SpatialWeightsBundle,
    out_path: Path,
) -> Path:
    """Figure 2: Disease-Specific Getis-Ord Gi* & Local Moran's I Hotspot Maps (4-Panel Grid)."""
    focal_diseases = ["Dengue", "Malaria", "Acute Diarrhoeal Disease", "Food Poisoning"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12.5), dpi=160)
    fig.suptitle(
        "JEEVAN-DRISHTI Phase 9: Disease-Specific Spatial Hotspot Detection (Getis-Ord Gi* & Local Moran's I LISA, 2023–2025)\n"
        "[Non-Transmission Rule: Spatial Clustering Reflects Surveillance Concentration Only — Not Direct Cross-District Transmission]",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    gm_lookup = (
        global_moran_df[
            (global_moran_df["reporting_period"] == "2023-2025_POOL")
            & (global_moran_df["indicator"] == "observed_cases_raw")
        ]
        .set_index("disease")
        .to_dict(orient="index")
    )

    for idx, dis in enumerate(focal_diseases):
        ax = axes[idx // 2, idx % 2]
        _draw_spatial_network_edges(ax, weights_bundle)
        sub = summary_df[
            (summary_df["disease"] == dis)
            & (summary_df["reporting_period"] == "2023-2025_POOL")
            & (summary_df["latitude"].notna())
        ].copy()

        gm = gm_lookup.get(dis, {})
        m_i = gm.get("morans_i", np.nan)
        m_p = gm.get("p_value_permutation", np.nan)

        # Plot each class with distinct marker shape and accessible color
        for _, r in sub.iterrows():
            lon, lat = float(r["longitude"]), float(r["latitude"])
            code = SHORT_CODE_MAP.get(str(r["district_name"]), str(r["district_name"])[:3].upper())
            gi_z = float(r["gi_star_zscore"]) if pd.notna(r["gi_star_zscore"]) else 0.0
            cases = int(r["observed_cases"]) if pd.notna(r["observed_cases"]) else 0
            lisa_q = str(r["lisa_quadrant"])
            dq = str(r["data_quality_status"])

            if dq.startswith("Insufficient Data"):
                mk, col, sz, tag = "X", "#64748B", 110, "Insuff"
            elif r["is_statistically_significant_hotspot"]:
                mk, col, sz, tag = "^", "#DC2626", 190, f"HOT(Gi*={gi_z:+.1f}*)"
            elif lisa_q == "HL_SPATIAL_OUTLIER":
                mk, col, sz, tag = "s", "#D97706", 155, f"HL(Gi*={gi_z:+.1f})"
            elif "COLDSPOT" in str(r["gi_star_class"]) or lisa_q == "LL_COLDSPOT":
                mk, col, sz, tag = "v", "#0284C7", 140, f"COLD({gi_z:+.1f})"
            elif cases > 0:
                mk, col, sz, tag = "o", "#3B82F6", 95, f"{cases}c"
            else:
                mk, col, sz, tag = "o", "#CBD5E1", 55, "0c"

            ax.scatter(lon, lat, marker=mk, s=sz, color=col, edgecolors="#0F172A", linewidths=1.0, zorder=4)
            if cases > 0 or r["is_statistically_significant_hotspot"] or dq.startswith("Insufficient Data"):
                ax.annotate(
                    f"{code}:{tag}",
                    (lon, lat),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=6.8,
                    fontweight="bold",
                    color="#0F172A",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75),
                    zorder=5,
                )

        n_hot = int(sub["is_statistically_significant_hotspot"].sum())
        _style_map_axes(
            ax,
            f"{dis} (2023–2025 Weekly IDSP)",
            f"Global Moran's I = {m_i:+.3f} (perm p = {m_p:.3f}) | Verified Hotspots (p<0.05): {n_hot}",
        )

    legend_elements = [
        Line2D([0], [0], marker="^", color="w", label="Statistically Significant Hotspot (Gi* / LISA HH, p<0.05) [▲]", markerfacecolor="#DC2626", markeredgecolor="#0F172A", markersize=10),
        Line2D([0], [0], marker="s", color="w", label="Isolated High-Burden Spatial Outlier (LISA HL, p<0.05) [■]", markerfacecolor="#D97706", markeredgecolor="#0F172A", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="Active Surveillance District — Not Significant (p>=0.05) [●]", markerfacecolor="#3B82F6", markeredgecolor="#0F172A", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Zero Reported Outbreaks for This Disease [○]", markerfacecolor="#CBD5E1", markeredgecolor="#0F172A", markersize=7),
        Line2D([0], [0], marker="X", color="w", label="Insufficient Data (Mumbai City, Ratnagiri, Mumbai Suburban) [✕]", markerfacecolor="#64748B", markeredgecolor="#0F172A", markersize=9),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_03_historical_comparison_maps(
    summary_df: pd.DataFrame,
    weights_bundle: SpatialWeightsBundle,
    out_path: Path,
) -> Path:
    """Figure 3: Historical Comparison Maps (2009-2022 Baseline vs 2023 vs 2024 vs 2025 IDSP)."""
    periods = [
        ("2009-2022_HISTORICAL_BASELINE", "A. Historical EpiClim Baseline (2009–2022)\n[Note: Pre-2014 Thane Includes Area Split to Palghar in Aug 2014]"),
        ("2023", "B. 2023 IDSP Weekly Outbreak Burden\n[Post-Split 36-District Standard]"),
        ("2024", "C. 2024 IDSP Weekly Outbreak Burden\n[Post-Split 36-District Standard]"),
        ("2025", "D. 2025 IDSP Weekly Outbreak Burden\n[Post-Split 36-District Standard]"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12.5), dpi=160)
    fig.suptitle(
        "JEEVAN-DRISHTI Phase 9: Historical Comparison Maps — 2009–2022 Reference vs. 2023, 2024 & 2025 Surveillance\n"
        "[Observed Historical & Annual Counts Kept Strictly Separate; Boundary Split Documented for Thane/Palghar]",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    for idx, (per_key, title_txt) in enumerate(periods):
        ax = axes[idx // 2, idx % 2]
        _draw_spatial_network_edges(ax, weights_bundle)
        sub = summary_df[
            (summary_df["disease"] == "ALL_DISEASES")
            & (summary_df["reporting_period"] == per_key)
            & (summary_df["latitude"].notna())
        ].copy()

        active = sub[sub["observed_outbreak_weeks"] > 0]
        zero_or_insuff = sub[sub["observed_outbreak_weeks"] == 0]

        if not active.empty:
            sc = ax.scatter(
                active["longitude"],
                active["latitude"],
                c=np.log1p(active["observed_cases"]),
                s=80 + np.sqrt(active["observed_cases"]) * 3.8,
                cmap="viridis",
                edgecolors="#0F172A",
                linewidths=1.0,
                alpha=0.92,
                zorder=4,
            )
            cbar = fig.colorbar(sc, ax=ax, fraction=0.042, pad=0.02)
            cbar.set_label("log1p(Observed Cases)", fontsize=8.5)

        if not zero_or_insuff.empty:
            ax.scatter(
                zero_or_insuff["longitude"],
                zero_or_insuff["latitude"],
                marker="X",
                s=95,
                color="#94A3B8",
                edgecolors="#1E293B",
                linewidths=1.0,
                zorder=3,
            )

        for _, r in sub.iterrows():
            code = SHORT_CODE_MAP.get(str(r["district_name"]), str(r["district_name"])[:3].upper())
            wks = int(r["observed_outbreak_weeks"])
            if wks > 0 and pd.notna(r["observed_cases"]):
                lbl = f"{code}:{int(r['observed_cases'])}"
                ax.annotate(
                    lbl,
                    (float(r["longitude"]), float(r["latitude"])),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=6.7,
                    fontweight="bold",
                    color="#0F172A",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.72),
                    zorder=5,
                )

        _style_map_axes(ax, title_txt)

    fig.tight_layout(rect=[0, 0.01, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_04_forecast_disease_maps(
    summary_df: pd.DataFrame,
    weights_bundle: SpatialWeightsBundle,
    out_path: Path,
) -> Path:
    """Figure 4: Forecast Disease Maps (Phase 6 Multi-Horizon Forecasts Kept Separate from Observed Cases)."""
    sub = summary_df[
        (summary_df["disease"] == "ALL_DISEASES")
        & (summary_df["reporting_period"] == "2023-2025_POOL")
        & (summary_df["latitude"].notna())
    ].copy()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12.2), dpi=160)
    fig.suptitle(
        "JEEVAN-DRISHTI Phase 9: Multi-Horizon Disease Forecast Maps (Phase 6 Engine — Strictly Separated from Observed Cases)\n"
        "[WARNING: Panels B, C, D Show MODEL FORECASTS (Cases/Outbreak Week), Never Labelled as Observed Cases]",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    panels = [
        ("observed_mean_cases_per_outbreak_week", "A. OBSERVED Mean Cases per Outbreak Week (2023–2025)\n[Ground-Truth IDSP Surveillance Reference]", "cividis"),
        ("forecast_7d_mean_cases", "B. MODEL FORECAST — 7-Day (1-Week Ahead) Mean Cases\n[HistGradientBoosting Regressor, Phase 6]", "YlOrRd"),
        ("forecast_14d_mean_cases", "C. MODEL FORECAST — 14-Day (2-Weeks Ahead) Mean Cases\n[Strict Lag >= 2w Cutoff, Phase 6]", "YlOrRd"),
        ("forecast_30d_mean_cases", "D. MODEL FORECAST — 30-Day (4-Weeks Ahead) Mean Cases\n[Strict Lag >= 4w Cutoff, Phase 6]", "YlOrRd"),
    ]

    for idx, (col_name, title_txt, cmap_name) in enumerate(panels):
        ax = axes[idx // 2, idx % 2]
        _draw_spatial_network_edges(ax, weights_bundle)
        valid = sub[sub[col_name].notna()].copy()
        insuff = sub[sub[col_name].isna()].copy()

        if not valid.empty:
            sc = ax.scatter(
                valid["longitude"],
                valid["latitude"],
                c=valid[col_name],
                s=85 + valid[col_name] * 2.2,
                cmap=cmap_name,
                edgecolors="#0F172A",
                linewidths=1.1,
                alpha=0.92,
                zorder=4,
            )
            cbar = fig.colorbar(sc, ax=ax, fraction=0.042, pad=0.02)
            cbar.set_label("Cases / Outbreak Week", fontsize=8.5)

        if not insuff.empty:
            ax.scatter(
                insuff["longitude"],
                insuff["latitude"],
                marker="X",
                s=115,
                color="#64748B",
                edgecolors="#0F172A",
                linewidths=1.2,
                zorder=4,
            )

        for _, r in sub.iterrows():
            code = SHORT_CODE_MAP.get(str(r["district_name"]), str(r["district_name"])[:3].upper())
            val = r[col_name]
            txt = f"{code}:{val:.1f}" if pd.notna(val) else f"{code}:Insuff"
            ax.annotate(
                txt,
                (float(r["longitude"]), float(r["latitude"])),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                fontsize=6.8,
                fontweight="bold",
                color="#0F172A",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.74),
                zorder=5,
            )

        _style_map_axes(ax, title_txt)

    fig.tight_layout(rect=[0, 0.01, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_05_anomaly_and_risk_intelligence_maps(
    summary_df: pd.DataFrame,
    weights_bundle: SpatialWeightsBundle,
    out_path: Path,
) -> Path:
    """Figure 5: Phase 7 Anomaly Detection & Phase 5 Outbreak Risk Classification Maps."""
    sub = summary_df[
        (summary_df["disease"] == "ALL_DISEASES")
        & (summary_df["reporting_period"] == "2023-2025_POOL")
        & (summary_df["latitude"].notna())
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.2), dpi=160)
    fig.suptitle(
        "JEEVAN-DRISHTI Phase 9: District Anomaly Signals (Phase 7) & Model-Estimated Outbreak Risk (Phase 5)\n"
        "[Anomalies Are Elevated Investigation Signals, Not Proof of Outbreak; Risk Probabilities Are Calibrated Estimates]",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    # Panel A: Phase 7 Anomaly Count & Max Anomaly Score
    ax = axes[0]
    _draw_spatial_network_edges(ax, weights_bundle)
    valid_a = sub[sub["anomaly_max_score"].notna()].copy()
    insuff_a = sub[sub["anomaly_max_score"].isna()].copy()

    for _, r in valid_a.iterrows():
        lon, lat = float(r["longitude"]), float(r["latitude"])
        acnt = int(r["anomaly_weeks_count"]) if pd.notna(r["anomaly_weeks_count"]) else 0
        ascore = float(r["anomaly_max_score"])
        code = SHORT_CODE_MAP.get(str(r["district_name"]), str(r["district_name"])[:3].upper())
        if acnt >= 3:
            mk, col, sz = "^", "#DC2626", 185
        elif acnt >= 1:
            mk, col, sz = "s", "#F59E0B", 145
        else:
            mk, col, sz = "o", "#10B981", 95
        ax.scatter(lon, lat, marker=mk, s=sz, color=col, edgecolors="#0F172A", linewidths=1.1, zorder=4)
        ax.annotate(
            f"{code}:{acnt}a({ascore:.2f})",
            (lon, lat),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=6.8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.74),
            zorder=5,
        )

    if not insuff_a.empty:
        ax.scatter(insuff_a["longitude"], insuff_a["latitude"], marker="X", s=125, color="#64748B", edgecolors="#0F172A", zorder=4)
        for _, r in insuff_a.iterrows():
            code = SHORT_CODE_MAP.get(str(r["district_name"]), str(r["district_name"])[:3].upper())
            ax.annotate(f"{code}:Insuff", (float(r["longitude"]), float(r["latitude"])), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6.8, fontweight="bold")

    leg_a = [
        Line2D([0], [0], marker="^", color="w", label=">=3 Anomaly Weeks [▲]", markerfacecolor="#DC2626", markeredgecolor="#0F172A", markersize=9),
        Line2D([0], [0], marker="s", color="w", label="1–2 Anomaly Weeks [■]", markerfacecolor="#F59E0B", markeredgecolor="#0F172A", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="0 Anomaly Weeks (Baseline) [●]", markerfacecolor="#10B981", markeredgecolor="#0F172A", markersize=8),
        Line2D([0], [0], marker="X", color="w", label="Insufficient Data [✕]", markerfacecolor="#64748B", markeredgecolor="#0F172A", markersize=8),
    ]
    ax.legend(handles=leg_a, loc="lower right", fontsize=8, framealpha=0.92, title="Phase 7 Anomaly Tier")
    _style_map_axes(
        ax,
        "A. Phase 7 Anomaly Detection Signals by District (2023–2025)",
        "Label = Code : Anomaly Weeks Count (Max Consensus Score)",
    )

    # Panel B: Phase 5 Calibrated Outbreak Risk Probability
    ax2 = axes[1]
    _draw_spatial_network_edges(ax2, weights_bundle)
    valid_r = sub[sub["model_estimated_risk_max"].notna()].copy()
    insuff_r = sub[sub["model_estimated_risk_max"].isna()].copy()

    for _, r in valid_r.iterrows():
        lon, lat = float(r["longitude"]), float(r["latitude"])
        rmax = float(r["model_estimated_risk_max"])
        rmean = float(r["model_estimated_risk_mean"])
        rcat = str(r["risk_category"])
        code = SHORT_CODE_MAP.get(str(r["district_name"]), str(r["district_name"])[:3].upper())
        if rcat == "HIGH_RISK":
            mk, col, sz = "^", "#B91C1C", 180
        elif rcat == "MODERATE_RISK":
            mk, col, sz = "s", "#EA580C", 140
        else:
            mk, col, sz = "o", "#2563EB", 95
        ax2.scatter(lon, lat, marker=mk, s=sz, color=col, edgecolors="#0F172A", linewidths=1.1, zorder=4)
        ax2.annotate(
            f"{code}:{rmean:.2f}",
            (lon, lat),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=6.8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.74),
            zorder=5,
        )

    if not insuff_r.empty:
        ax2.scatter(insuff_r["longitude"], insuff_r["latitude"], marker="X", s=125, color="#64748B", edgecolors="#0F172A", zorder=4)
        for _, r in insuff_r.iterrows():
            code = SHORT_CODE_MAP.get(str(r["district_name"]), str(r["district_name"])[:3].upper())
            ax2.annotate(f"{code}:Insuff", (float(r["longitude"]), float(r["latitude"])), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=6.8, fontweight="bold")

    leg_r = [
        Line2D([0], [0], marker="^", color="w", label="Peak High Risk (P >= 0.55) [▲]", markerfacecolor="#B91C1C", markeredgecolor="#0F172A", markersize=9),
        Line2D([0], [0], marker="s", color="w", label="Peak Moderate Risk (0.25 <= P < 0.55) [■]", markerfacecolor="#EA580C", markeredgecolor="#0F172A", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Low Risk (P < 0.25) [●]", markerfacecolor="#2563EB", markeredgecolor="#0F172A", markersize=8),
        Line2D([0], [0], marker="X", color="w", label="Insufficient Data [✕]", markerfacecolor="#64748B", markeredgecolor="#0F172A", markersize=8),
    ]
    ax2.legend(handles=leg_r, loc="lower right", fontsize=8, framealpha=0.92, title="Phase 5 Risk Category")
    _style_map_axes(
        ax2,
        "B. Phase 5 Calibrated Outbreak Risk Estimates (2023–2025)",
        "Label = Code : Mean Calibrated Risk Probability | Shape = Peak Risk Category",
    )

    fig.tight_layout(rect=[0, 0.01, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_06_district_drilldown_spatial_profiles(
    summary_df: pd.DataFrame,
    weights_bundle: SpatialWeightsBundle,
    out_path: Path,
) -> Path:
    """Figure 6: District-Level Drill-Down Spatial Profiles & Moran Scatterplot (2023-2025)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.0), dpi=160)
    fig.suptitle(
        "JEEVAN-DRISHTI Phase 9: District Drill-Down Disease Composition & Spatial Autocorrelation Moran Plot (2023–2025)\n"
        "[Left: Top 12 Burden Districts by Disease Breakdown | Right: Malaria Spatial Lag Moran Scatterplot (I = +0.181, p = 0.015)]",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )

    # Panel A: Top 12 districts stacked horizontal bar by major disease
    ax = axes[0]
    pool_all = summary_df[
        (summary_df["disease"] == "ALL_DISEASES")
        & (summary_df["reporting_period"] == "2023-2025_POOL")
        & (summary_df["observed_cases"].notna())
    ].sort_values("observed_cases", ascending=False).head(12)
    top_districts = pool_all["district_name"].tolist()[::-1]

    major_diseases = ["Dengue", "Acute Diarrhoeal Disease", "Malaria", "Hepatitis A", "Food Poisoning", "Chikungunya"]
    palette = ["#D97706", "#2563EB", "#DC2626", "#7C3AED", "#059669", "#0891B2"]

    left_accum = np.zeros(len(top_districts), dtype=float)
    for dis, col in zip(major_diseases, palette):
        vals = []
        for d in top_districts:
            m = summary_df[
                (summary_df["district_name"] == d)
                & (summary_df["disease"] == dis)
                & (summary_df["reporting_period"] == "2023-2025_POOL")
            ]
            v = float(m["observed_cases"].iloc[0]) if not m.empty and pd.notna(m["observed_cases"].iloc[0]) else 0.0
            vals.append(v)
        vals_arr = np.array(vals, dtype=float)
        ax.barh(top_districts, vals_arr, left=left_accum, color=col, edgecolor="white", height=0.68, label=dis)
        left_accum += vals_arr

    # Remaining other diseases
    other_vals = []
    for i, d in enumerate(top_districts):
        tot = float(pool_all[pool_all["district_name"] == d]["observed_cases"].iloc[0])
        other_vals.append(max(tot - left_accum[i], 0.0))
    ax.barh(top_districts, other_vals, left=left_accum, color="#94A3B8", edgecolor="white", height=0.68, label="Other Diseases")

    ax.set_xlabel("Observed Outbreak Cases (2023–2025)", fontsize=9.5, fontweight="bold")
    ax.set_title("A. Top 12 Maharashtra Districts — Disease Burden Drill-Down", fontsize=10.5, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)
    ax.grid(axis="x", linestyle=":", alpha=0.4)

    # Panel B: Moran Scatterplot for Malaria (strongest spatial cluster in Eastern Vidarbha)
    ax2 = axes[1]
    mal = summary_df[
        (summary_df["disease"] == "Malaria")
        & (summary_df["reporting_period"] == "2023-2025_POOL")
        & (summary_df["latitude"].notna())
    ].set_index("district_name").loc[weights_bundle.districts].reset_index()

    x = mal["observed_cases"].fillna(0.0).to_numpy(dtype=float)
    z = (x - np.mean(x)) / (np.std(x, ddof=1) if np.std(x, ddof=1) > 0 else 1.0)
    w_z = weights_bundle.W_row_std @ z

    ax2.axhline(0, color="#64748B", linestyle="--", linewidth=1.0)
    ax2.axvline(0, color="#64748B", linestyle="--", linewidth=1.0)

    for i, r in mal.iterrows():
        is_hot = bool(r["is_statistically_significant_hotspot"])
        lisa_q = str(r["lisa_quadrant"])
        if is_hot:
            mk, col, sz = "^", "#DC2626", 130
        elif lisa_q == "HL_SPATIAL_OUTLIER":
            mk, col, sz = "s", "#D97706", 110
        else:
            mk, col, sz = "o", "#2563EB", 65
        ax2.scatter(z[i], w_z[i], marker=mk, s=sz, color=col, edgecolors="#0F172A", zorder=4)
        if abs(z[i]) > 0.35 or abs(w_z[i]) > 0.35 or is_hot:
            ax2.annotate(
                f"{r['district_name']} (Gi*={r['gi_star_zscore']:+.2f})",
                (z[i], w_z[i]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=7.5,
                fontweight="bold",
            )

    # OLS slope = Global Moran's I
    z_line = np.linspace(min(z.min(), -0.5), max(z.max(), 1.0), 50)
    slope = float(np.polyfit(z, w_z, 1)[0])
    ax2.plot(z_line, slope * z_line, color="#DC2626", linewidth=1.5, linestyle="-", label=f"Moran Slope = {slope:+.3f}")
    ax2.set_xlabel("Standardized Observed Cases (z)", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Spatial Lag of Standardized Cases (W · z)", fontsize=9.5, fontweight="bold")
    ax2.set_title("B. Spatial Autocorrelation Moran Plot — Malaria (Eastern Vidarbha Cluster)", fontsize=10.5, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout(rect=[0, 0.01, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_all_geospatial_figures(
    summary_df: pd.DataFrame,
    global_moran_df: pd.DataFrame,
    weights_bundle: SpatialWeightsBundle,
    reports_fig_dir: Path,
    data_fig_dir: Path,
) -> List[str]:
    """Render all 6 geospatial intelligence figures and mirror them to data/geospatial/figures/."""
    reports_fig_dir.mkdir(parents=True, exist_ok=True)
    data_fig_dir.mkdir(parents=True, exist_ok=True)

    f1 = plot_01_maharashtra_district_burden_map(
        summary_df, weights_bundle, reports_fig_dir / "01_maharashtra_district_disease_burden_map.png"
    )
    f2 = plot_02_disease_specific_hotspot_maps(
        summary_df, global_moran_df, weights_bundle, reports_fig_dir / "02_disease_specific_hotspot_maps.png"
    )
    f3 = plot_03_historical_comparison_maps(
        summary_df, weights_bundle, reports_fig_dir / "03_historical_comparison_maps.png"
    )
    f4 = plot_04_forecast_disease_maps(
        summary_df, weights_bundle, reports_fig_dir / "04_forecast_disease_maps.png"
    )
    f5 = plot_05_anomaly_and_risk_intelligence_maps(
        summary_df, weights_bundle, reports_fig_dir / "05_anomaly_and_risk_intelligence_maps.png"
    )
    f6 = plot_06_district_drilldown_spatial_profiles(
        summary_df, weights_bundle, reports_fig_dir / "06_district_drilldown_spatial_profiles.png"
    )

    generated = [f1, f2, f3, f4, f5, f6]
    for fp in generated:
        shutil.copy2(fp, data_fig_dir / fp.name)

    return [str(fp) for fp in generated]
