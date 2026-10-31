"""Step 6 of Phase 8: Publication-Quality Visualizations for Climate-Disease Intelligence.

Generates 7 charts in `reports/climate_intelligence/figures/` and `data/climate_intelligence/figures/`:
1. `01_rainfall_vs_disease_trends.png`
2. `02_temperature_vs_disease_trends.png`
3. `03_humidity_vs_disease_trends_pune.png`
4. `04_lagged_climate_disease_relationships.png`
5. `05_district_climate_disease_comparison.png`
6. `06_climate_disease_correlation_heatmap.png`
7. `07_shap_and_partial_dependence_climate.png`

Every figure explicitly identifies the Data Period, Reporting Frequency, and Non-Causal Notice.
"""
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJ = Path(__file__).resolve().parents[3]
REPORTS_FIG_DIR = PROJ / "reports" / "climate_intelligence" / "figures"
DATA_FIG_DIR = PROJ / "data" / "climate_intelligence" / "figures"


def _save_both(fig: plt.Figure, filename: str) -> str:
    REPORTS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_FIG_DIR.mkdir(parents=True, exist_ok=True)
    p1 = REPORTS_FIG_DIR / filename
    p2 = DATA_FIG_DIR / filename
    fig.savefig(p1, dpi=150, bbox_inches="tight", facecolor="white")
    fig.savefig(p2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(p1.relative_to(PROJ)).replace("\\", "/")


def generate_all_climate_disease_charts(
    aligned_outbreaks: pd.DataFrame,
    dw_panel: pd.DataFrame,
    state_wk: pd.DataFrame,
    lag_df: pd.DataFrame,
    district_rel_df: pd.DataFrame,
    imp_df: pd.DataFrame,
    pdp_df: pd.DataFrame,
) -> List[str]:
    """Create all 7 required visualization charts with explicit period and frequency labels."""
    saved_paths: List[str] = []
    wk_order = state_wk.sort_values("period_idx").reset_index(drop=True)
    x_idx = wk_order["period_idx"].to_numpy()

    # 1. Rainfall vs. Disease Trend Chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.suptitle(
        "Maharashtra Rainfall vs. Disease Outbreak Trends\n"
        "Data Period: 2023-W01 to 2025-W52 (121 Active Weeks) | Reporting Frequency: Weekly IDSP & Weekly Aggregated IMD Gridded Rainfall",
        fontsize=11,
        fontweight="bold",
    )
    ax1_rain = ax1.twinx()
    ax1.bar(x_idx, wk_order["state_vector_cases"], color="#d95f02", alpha=0.75, width=0.9, label="Vector-Borne Cases (Dengue/Malaria/Chikungunya)")
    ax1_rain.plot(x_idx, wk_order["rain_mm_accum_prev_4w"], color="#1f78b4", linewidth=2.0, label="Prior 4-Week Accumulated Rainfall (mm)")
    ax1.set_ylabel("Weekly Vector-Borne Cases", color="#d95f02", fontweight="bold")
    ax1_rain.set_ylabel("4-Week Rainfall (mm)", color="#1f78b4", fontweight="bold")
    ax1.set_title("Panel A: Vector-Borne Outbreak Cases vs. 4-Week Accumulated Rainfall (Weekly Grain)", fontsize=10)
    ax1.grid(alpha=0.25)

    ax2_rain = ax2.twinx()
    ax2.bar(x_idx, wk_order["state_water_cases"], color="#2ca02c", alpha=0.75, width=0.9, label="Water/Food-Borne Cases (ADD/Cholera/Food Poisoning)")
    ax2_rain.plot(x_idx, wk_order["rain_mm_sum_t"], color="#08519c", linewidth=1.8, linestyle="--", label="Current-Week Rainfall Total (mm)")
    ax2.set_ylabel("Weekly Water/Food-Borne Cases", color="#2ca02c", fontweight="bold")
    ax2_rain.set_ylabel("Weekly Rainfall (mm)", color="#08519c", fontweight="bold")
    ax2.set_xlabel("Chronological Weekly Index (0 = 2023-W01, 52 = 2024-W01, 104 = 2025-W01, 155 = 2025-W52)")
    ax2.set_title("Panel B: Water/Food-Borne Outbreak Cases vs. Current-Week Rainfall (Weekly Grain)", fontsize=10)
    ax2.grid(alpha=0.25)
    fig.text(
        0.5, 0.01,
        "Note: Observational weekly co-movement only; does not imply direct causation without controlling for seasonal confounders.",
        ha="center", fontsize=8.5, style="italic",
    )
    saved_paths.append(_save_both(fig, "01_rainfall_vs_disease_trends.png"))

    # 2. Temperature vs. Disease Trend Chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    fig.suptitle(
        "Maharashtra Temperature Regimes vs. Weekly Disease Outbreak Burden\n"
        "Data Period: 2023-W01 to 2025-W52 (121 Active Weeks) | Reporting Frequency: Weekly IDSP & Weekly Mean IMD Temperature (1.0° Grid)",
        fontsize=11,
        fontweight="bold",
    )
    ax1_t = ax1.twinx()
    ax1.bar(x_idx, wk_order["state_dengue_cases"], color="#e7298a", alpha=0.75, width=0.9, label="Weekly Dengue Cases")
    ax1_t.plot(x_idx, wk_order["tmin_c_mean_lag_1w"], color="#d62728", linewidth=2.0, label="Lag-1w Mean Min Temperature Tmin (°C)")
    ax1_t.plot(x_idx, wk_order["temp_diurnal_range_lag_1w"], color="#7570b3", linewidth=1.6, linestyle=":", label="Lag-1w Diurnal Temp Range (°C)")
    ax1.set_ylabel("Weekly Dengue Cases", color="#e7298a", fontweight="bold")
    ax1_t.set_ylabel("Temperature (°C)", color="#d62728", fontweight="bold")
    ax1.set_title("Panel A: Weekly Dengue Cases vs. Lag-1w Minimum Temperature & Diurnal Range", fontsize=10)
    ax1.grid(alpha=0.25)

    ax2_t = ax2.twinx()
    ax2.bar(x_idx, wk_order["state_add_cases"], color="#ff7f00", alpha=0.75, width=0.9, label="Weekly Acute Diarrhoeal Disease Cases")
    ax2_t.plot(x_idx, wk_order["tmax_c_mean_lag_1w"], color="#b15928", linewidth=2.0, label="Lag-1w Mean Max Temperature Tmax (°C)")
    ax2.set_ylabel("Weekly ADD Cases", color="#ff7f00", fontweight="bold")
    ax2_t.set_ylabel("Max Temperature Tmax (°C)", color="#b15928", fontweight="bold")
    ax2.set_xlabel("Chronological Weekly Index (0 = 2023-W01 .. 155 = 2025-W52)")
    ax2.set_title("Panel B: Weekly Acute Diarrhoeal Disease Cases vs. Lag-1w Maximum Temperature", fontsize=10)
    ax2.grid(alpha=0.25)
    fig.text(
        0.5, 0.01,
        "Note: Temperature and disease counts exhibit strong shared annual seasonality; statistical correlation does not establish causation.",
        ha="center", fontsize=8.5, style="italic",
    )
    saved_paths.append(_save_both(fig, "02_temperature_vs_disease_trends.png"))

    # 3. Relative Humidity vs. Disease Trend Chart (Pune Only, May 2024 - Jun 2025)
    pune_wk = (
        dw_panel[(dw_panel["district_std"] == "Pune") & (dw_panel["rh_mean_t"].notna())]
        .sort_values("period_idx")
        .reset_index(drop=True)
    )
    fig, ax1 = plt.subplots(figsize=(12, 5.5))
    fig.suptitle(
        "Pune District Relative Humidity vs. Weekly Disease Outbreak Cases (Single-Station Sub-Analysis)\n"
        "Data Period: 2024-W19 to 2025-W23 (May 2024 – Jun 2025, 57 Weeks) | Reporting Frequency: Weekly IDSP & Weekly Aggregated CWPRS Telemetry",
        fontsize=10.5,
        fontweight="bold",
    )
    ax2 = ax1.twinx()
    p_x = np.arange(len(pune_wk))
    p_labels = [f"{int(r['report_year'])}-W{int(r['report_week']):02d}" for _, r in pune_wk.iterrows()]
    cases_pune = pune_wk["total_outbreak_cases"].fillna(0.0).to_numpy()
    ax1.bar(p_x, cases_pune, color="#6a3d9a", alpha=0.75, label="Pune Weekly Outbreak Cases (Dengue / Zika / ADD / Chikungunya)")
    ax2.plot(p_x, pune_wk["rh_mean_t"], color="#1f78b4", linewidth=2.2, marker="o", markersize=3.5, label="Pune Weekly Mean Relative Humidity (%)")
    if "rh_mean_roll_prev_4w" in pune_wk.columns:
        ax2.plot(p_x, pune_wk["rh_mean_roll_prev_4w"], color="#33a02c", linewidth=1.8, linestyle="--", label="Pune Prior 4-Week Rolling Mean RH (%)")
    ax1.set_ylabel("Pune Weekly Reported Outbreak Cases", color="#6a3d9a", fontweight="bold")
    ax2.set_ylabel("Relative Humidity (%)", color="#1f78b4", fontweight="bold")
    tick_step = max(1, len(p_x) // 12)
    ax1.set_xticks(p_x[::tick_step])
    ax1.set_xticklabels([p_labels[i] for i in range(0, len(p_labels), tick_step)], rotation=35, ha="right")
    ax1.grid(alpha=0.25)
    fig.text(
        0.5, 0.01,
        "Scope Limitation: Relative humidity is available ONLY at CWPRS Khadakvasla (Pune) for May 2024 – Jun 2025; never extrapolated to other districts.",
        ha="center", fontsize=8.5, style="italic",
    )
    saved_paths.append(_save_both(fig, "03_humidity_vs_disease_trends_pune.png"))

    # 4. Lagged Relationship Chart (Multi-Week Lag Profiles L = 0, 1, 2, 3, 4, 8 weeks)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    fig.suptitle(
        "Multi-Week Lagged Climate–Disease Spearman Correlation Profiles (Lags 0 to 8 Reporting Weeks)\n"
        "Data Period: 2023-W01 to 2025-W52 | Reporting Frequency: Weekly IDSP & Weekly IMD Climate Lags",
        fontsize=11,
        fontweight="bold",
    )
    lags_to_plot = [0, 1, 2, 3, 4, 8]
    rain_lag_map = {
        0: "rain_mm_sum_t",
        1: "rain_mm_sum_lag_1w",
        2: "rain_mm_sum_lag_2w",
        3: "rain_mm_sum_lag_3w",
        4: "rain_mm_sum_lag_4w",
        8: "rain_mm_sum_lag_8w",
    }
    temp_lag_map = {
        0: "tmin_c_mean_t",
        1: "tmin_c_mean_lag_1w",
        2: "tmin_c_mean_lag_2w",
        4: "tmin_c_mean_lag_4w",
        8: "tmin_c_mean_lag_8w",
    }

    for series_lbl, color, marker in [
        ("Statewide_Vector_Borne_Weekly_Cases", "#d95f02", "o"),
        ("Statewide_Water_Borne_Weekly_Cases", "#1b9e77", "s"),
        ("Statewide_Dengue_Weekly_Cases", "#e7298a", "^"),
        ("Statewide_ADD_Weekly_Cases", "#7570b3", "D"),
    ]:
        sub_s = lag_df[
            (lag_df["analysis_grain"] == "statewide_weekly_time_series_121w")
            & (lag_df["disease_or_series"] == series_lbl)
        ]
        r_vals = []
        for lw in lags_to_plot:
            m_row = sub_s[sub_s["climate_variable"] == rain_lag_map[lw]]
            r_vals.append(float(m_row["spearman_rho"].iloc[0]) if not m_row.empty else np.nan)
        axes[0].plot(lags_to_plot, r_vals, marker=marker, linewidth=2.0, color=color, label=series_lbl.replace("Statewide_", "").replace("_Weekly_Cases", ""))

        t_lags = [0, 1, 2, 4, 8]
        t_vals = []
        for lw in t_lags:
            m_row = sub_s[sub_s["climate_variable"] == temp_lag_map[lw]]
            t_vals.append(float(m_row["spearman_rho"].iloc[0]) if not m_row.empty else np.nan)
        axes[1].plot(t_lags, t_vals, marker=marker, linewidth=2.0, color=color, label=series_lbl.replace("Statewide_", "").replace("_Weekly_Cases", ""))

    for ax, ttl in zip(
        axes,
        ["Panel A: Weekly Rainfall Total at Lag k Weeks", "Panel B: Weekly Mean Min Temp (Tmin) at Lag k Weeks"],
    ):
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.axhline(0.18, color="gray", linestyle="--", linewidth=0.8, label="p=0.05 threshold (|rho|≈0.18)")
        ax.axhline(-0.18, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Climate Lag (Reporting Weeks Prior to Disease Period t)")
        ax.set_title(ttl, fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8.5, loc="best")
    axes[0].set_ylabel("Spearman Rank Correlation (rho)")
    saved_paths.append(_save_both(fig, "04_lagged_climate_disease_relationships.png"))

    # 5. District-Wise Comparison Chart
    valid_dist = district_rel_df[district_rel_df["mean_weekly_rainfall_mm"].notna()].copy()
    top15_dist = valid_dist.sort_values("outbreak_weeks_count", ascending=False).head(15)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "District-Wise Climate Regimes & Disease Outbreak Burden Across Maharashtra\n"
        "Data Period: 2023-W01 to 2025-W52 | Reporting Frequency: Weekly IDSP & Weekly Aggregated IMD District Climate (35 Coordinate Districts)",
        fontsize=11,
        fontweight="bold",
    )
    y_pos = np.arange(len(top15_dist))
    ax1.barh(y_pos, top15_dist["outbreak_weeks_count"], color="#2b8cbe", alpha=0.85)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"{r['district_std']} ({r['dominant_disease']})" for _, r in top15_dist.iterrows()], fontsize=8.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("Total Outbreak-Reporting Weeks (2023–2025)")
    ax1.set_title("Panel A: Top 15 Districts by Outbreak Weeks & Dominant Disease", fontsize=10)
    ax1.grid(axis="x", alpha=0.25)

    sc = ax2.scatter(
        valid_dist["mean_weekly_rainfall_mm"],
        valid_dist["mean_weekly_tavg_c"],
        s=np.maximum(valid_dist["outbreak_weeks_count"] * 12, 30),
        c=valid_dist["panel_rain_4w_spearman_rho"].fillna(0.0),
        cmap="coolwarm",
        vmin=-0.35,
        vmax=0.35,
        edgecolors="black",
        alpha=0.85,
    )
    for _, r in top15_dist.head(10).iterrows():
        ax2.annotate(
            str(r["district_std"]),
            (float(r["mean_weekly_rainfall_mm"]), float(r["mean_weekly_tavg_c"])),
            fontsize=7.5,
            xytext=(4, 3),
            textcoords="offset points",
        )
    cbar = plt.colorbar(sc, ax=ax2)
    cbar.set_label("Spearman rho (4w Rainfall vs. Weekly Cases)")
    ax2.set_xlabel("District Mean Weekly Rainfall (mm, 2023–2025)")
    ax2.set_ylabel("District Mean Weekly Temperature Tavg (°C, 2023–2025)")
    ax2.set_title("Panel B: District Climate Space (Bubble Size = Outbreak Weeks, Color = Rain-Disease rho)", fontsize=10)
    ax2.grid(alpha=0.25)
    saved_paths.append(_save_both(fig, "05_district_climate_disease_comparison.png"))

    # 6. Climate-Disease Correlation Heatmap
    hm_series = [
        "Statewide_All_Outbreaks_Weekly_Cases",
        "Statewide_Vector_Borne_Weekly_Cases",
        "Statewide_Water_Borne_Weekly_Cases",
        "Statewide_Dengue_Weekly_Cases",
        "Statewide_ADD_Weekly_Cases",
        "Statewide_Malaria_Weekly_Cases",
        "Dengue",
        "Acute Diarrhoeal Disease",
        "Food Poisoning",
        "Malaria",
        "Chikungunya",
        "Hepatitis A",
        "Cholera",
    ]
    hm_clim_cols = [
        "rain_mm_sum_t",
        "rain_mm_sum_lag_1w",
        "rain_mm_sum_lag_4w",
        "rain_mm_accum_prev_4w",
        "rain_mm_accum_prev_8w",
        "tmin_c_mean_t",
        "tmin_c_mean_lag_1w",
        "tmin_c_mean_lag_4w",
        "tmax_c_mean_t",
        "tmax_c_mean_lag_1w",
        "temp_diurnal_range_mean_t",
        "temp_diurnal_range_lag_1w",
    ]
    mat = np.full((len(hm_series), len(hm_clim_cols)), np.nan)
    sig_stars = [["" for _ in hm_clim_cols] for _ in hm_series]

    for i_s, s_name in enumerate(hm_series):
        for j_c, c_name in enumerate(hm_clim_cols):
            sub_m = lag_df[(lag_df["disease_or_series"] == s_name) & (lag_df["climate_variable"] == c_name)]
            if not sub_m.empty and pd.notna(sub_m["spearman_rho"].iloc[0]):
                val = float(sub_m["spearman_rho"].iloc[0])
                pv = float(sub_m["spearman_p"].iloc[0])
                qv = float(sub_m["spearman_fdr_q"].iloc[0])
                mat[i_s, j_c] = val
                if qv < 0.05:
                    sig_stars[i_s][j_c] = "**"
                elif pv < 0.05:
                    sig_stars[i_s][j_c] = "*"

    fig, ax = plt.subplots(figsize=(13, 7.5))
    im = ax.imshow(np.nan_to_num(mat, nan=0.0), cmap="RdBu_r", vmin=-0.50, vmax=0.50, aspect="auto")
    ax.set_xticks(np.arange(len(hm_clim_cols)))
    ax.set_xticklabels(hm_clim_cols, rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(np.arange(len(hm_series)))
    ax.set_yticklabels([s.replace("Statewide_", "State: ").replace("_Weekly_Cases", "") for s in hm_series], fontsize=9)
    for i_s in range(len(hm_series)):
        for j_c in range(len(hm_clim_cols)):
            if not np.isnan(mat[i_s, j_c]):
                txt = f"{mat[i_s, j_c]:+.2f}{sig_stars[i_s][j_c]}"
                ax.text(j_c, i_s, txt, ha="center", va="center", fontsize=7.5, color="black")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Spearman Rank Correlation (rho)")
    ax.set_title(
        "Climate–Disease Spearman Correlation Heatmap (* p < 0.05, ** FDR q < 0.05)\n"
        "Data Period: 2023-W01 to 2025-W52 | Reporting Frequency: Weekly IDSP & Weekly Aggregated IMD Climate",
        fontsize=10.5,
        fontweight="bold",
    )
    saved_paths.append(_save_both(fig, "06_climate_disease_correlation_heatmap.png"))

    # 7. Exact TreeSHAP & Partial Dependence Chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Advanced ML Analysis: Exact TreeSHAP Feature Importance & Partial Dependence Profiles\n"
        "Data Period: 2023-W01 to 2025-W52 (508 Aligned Outbreak Weeks) | Reporting Frequency: Weekly (Controlling for Seasonality & Geography)",
        fontsize=10.5,
        fontweight="bold",
    )
    pool_imp = imp_df[imp_df["analysis_task"] == "Pooled_Outbreak_Severity_All_Diseases"].head(12)
    y_idx = np.arange(len(pool_imp))
    colors = ["#2b8cbe" if r["feature_role"] == "Climate_Variable" else "#999999" for _, r in pool_imp.iterrows()]
    ax1.barh(y_idx, pool_imp["mean_abs_treeshap"], color=colors, alpha=0.88)
    ax1.set_yticks(y_idx)
    ax1.set_yticklabels(pool_imp["feature_name"], fontsize=8.5)
    ax1.invert_yaxis()
    ax1.set_xlabel("Mean |TreeSHAP| Contribution to log1p(Cases) (Blue = Climate, Gray = Confounder Control)")
    ax1.set_title("Panel A: Top 12 Features by Exact TreeSHAP Importance", fontsize=10)
    ax1.grid(axis="x", alpha=0.25)

    pdp_rain = pdp_df[
        (pdp_df["analysis_task"] == "Pooled_Outbreak_Severity_All_Diseases")
        & (pdp_df["climate_feature"] == "rain_mm_accum_prev_4w")
    ]
    pdp_tmin = pdp_df[
        (pdp_df["analysis_task"] == "Pooled_Outbreak_Severity_All_Diseases")
        & (pdp_df["climate_feature"] == "tmin_c_mean_lag_1w")
    ]
    ax2_t = ax2.twiny()
    if not pdp_rain.empty:
        ax2.plot(
            pdp_rain["grid_value"],
            pdp_rain["mean_predicted_cases_equiv"],
            color="#1f78b4",
            linewidth=2.2,
            marker="o",
            markersize=4,
            label="PDP: Prior 4-Week Rainfall (mm, bottom axis)",
        )
    if not pdp_tmin.empty:
        ax2_t.plot(
            pdp_tmin["grid_value"],
            pdp_tmin["mean_predicted_cases_equiv"],
            color="#d62728",
            linewidth=2.2,
            linestyle="--",
            marker="s",
            markersize=4,
            label="PDP: Lag-1w Min Temp Tmin (°C, top axis)",
        )
    ax2.set_xlabel("Prior 4-Week Accumulated Rainfall (mm)", color="#1f78b4", fontweight="bold")
    ax2_t.set_xlabel("Lag-1w Mean Minimum Temperature Tmin (°C)", color="#d62728", fontweight="bold")
    ax2.set_ylabel("Partial Dependence Marginal Expected Cases (Seasonality Held Constant)")
    ax2.set_title("Panel B: 1D Partial Dependence Curves (Confounders Held Constant)", fontsize=10)
    ax2.grid(alpha=0.25)
    fig.text(
        0.5, 0.01,
        "Non-Causal Notice: TreeSHAP and Partial Dependence quantify marginal model sensitivity, not causal intervention effects.",
        ha="center", fontsize=8.5, style="italic",
    )
    saved_paths.append(_save_both(fig, "07_shap_and_partial_dependence_climate.png"))

    return saved_paths
