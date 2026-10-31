"""Visualization Suite for Phase 12 Disease Severity & Mortality Intelligence Engine."""
from pathlib import Path
import shutil
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_01_outcome_availability_and_missing_vs_zero(
    availability_df: pd.DataFrame,
    missing_vs_zero_df: pd.DataFrame,
    out_path: Path,
) -> Path:
    """Figure 1: Outcome Category Availability & Missing (NaN) vs. Zero (0) Deaths Audit."""
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.8), dpi=160)
    fig.suptitle(
        "JEEVAN-DRISHTI Phase 12: Outcome & Severity Data Availability + Missing (NaN) vs. Zero (0) Deaths Audit\n"
        "[Aggregate Outbreak Surveillance Only — Unavailable Clinical Severity/Admissions/Recovery Kept Strictly NaN]",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )

    # Panel A: 7 Outcome Categories
    ax1 = axes[0]
    df_a = availability_df.iloc[::-1].reset_index(drop=True)
    y_pos = np.arange(len(df_a))
    pct_avail = [100.0 if a else 0.0 for a in df_a["available_in_workspace"]]
    cols = ["#16A34A" if a else "#94A3B8" for a in df_a["available_in_workspace"]]
    ax1.barh(y_pos, pct_avail, color=cols, edgecolor="#0F172A", height=0.6)
    for i, r in df_a.iterrows():
        lbl = (
            f"SUPPORTED ({r['non_null_records_2023_2025']} verified weekly rows in 2023–25)"
            if r["available_in_workspace"]
            else "UNAVAILABLE IN WORKSPACE (0% — Explicitly NaN; Never Invented)"
        )
        ax1.text(
            4.0,
            i,
            lbl,
            va="center",
            ha="left",
            fontsize=8.3,
            fontweight="bold",
            color="#0F172A",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.88),
        )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_a["outcome_category_label"], fontsize=8.8, fontweight="bold")
    ax1.set_xlim(0, 110)
    ax1.set_xlabel("2023–2025 Workspace Availability (%)", fontsize=9.5, fontweight="bold")
    ax1.set_title("A. Audit of 7 Severity & Mortality Outcome Categories", fontsize=10.5, fontweight="bold")
    ax1.grid(axis="x", linestyle=":", alpha=0.45)

    # Panel B: Missing (NaN) vs. Exact Zero (0) vs. Positive (>0) Death Records
    ax2 = axes[1]
    sub_mz = missing_vs_zero_df[
        missing_vs_zero_df["duplicate_status"].isin(["unique", "unique_district_disease_week"])
    ].reset_index(drop=True)
    labels_b = [
        "Historical EpiClim\n(2009–2022 Unique)\n[n = 1,143]",
        "IDSP Raw Unique\n(2023–2025 Unique)\n[n = 552]",
        "Canonical Weekly Grain\n(2023–2025 Deduplicated)\n[n = 511]",
    ]
    x = np.arange(len(sub_mz))
    tot = sub_mz["total_records"].to_numpy(dtype=float)
    na_pct = (sub_mz["deaths_missing_nan_count"].to_numpy(dtype=float) / tot) * 100.0
    zero_pct = (sub_mz["deaths_exact_zero_count"].to_numpy(dtype=float) / tot) * 100.0
    pos_pct = (sub_mz["deaths_positive_gt0_count"].to_numpy(dtype=float) / tot) * 100.0

    ax2.bar(x, pos_pct, width=0.52, color="#DC2626", edgecolor="#0F172A", label="Positive Deaths (≥ 1)")
    ax2.bar(
        x,
        zero_pct,
        bottom=pos_pct,
        width=0.52,
        color="#2563EB",
        edgecolor="#0F172A",
        label="Verified Zero Deaths (= 0)",
    )
    ax2.bar(
        x,
        na_pct,
        bottom=pos_pct + zero_pct,
        width=0.52,
        color="#CBD5E1",
        edgecolor="#0F172A",
        label="Missing / Unrecorded (NaN — Never Zero-Filled)",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_b, fontsize=8.8, fontweight="bold")
    ax2.set_ylabel("Percentage of Records (%)", fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, 108)
    ax2.set_title(
        "B. Strict Separation of Missing (NaN) vs. Zero (0) vs. Positive Deaths",
        fontsize=10.5,
        fontweight="bold",
    )
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(axis="y", linestyle=":", alpha=0.45)

    fig.tight_layout(rect=[0, 0.01, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_02_disease_wise_mortality_and_wilson_cfr(
    disease_summary_df: pd.DataFrame, out_path: Path
) -> Path:
    """Figure 2: Disease-Wise Total Verified Deaths & Outbreak-Cluster CFR (%) with Wilson 95% CIs."""
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.0), dpi=160)
    fig.suptitle(
        "JEEVAN-DRISHTI Phase 12: Disease-Wise Verified Outbreak Mortality & Outbreak-Cluster CFR (2023–2025)\n"
        "[Compatible Numerator/Denominator Only — Wilson 95% CIs & Empirical Bayes Shrinkage Shown for N ≥ 30 Cases]",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )

    # Panel A: Diseases with >= 1 verified death in 2023-2025
    ax1 = axes[0]
    pos_dis = (
        disease_summary_df[disease_summary_df["verified_deaths"] > 0]
        .sort_values("verified_deaths", ascending=True)
        .reset_index(drop=True)
    )
    y1 = np.arange(len(pos_dis))
    ax1.barh(y1, pos_dis["verified_deaths"], color="#DC2626", edgecolor="#0F172A", height=0.64)
    for i, r in pos_dis.iterrows():
        ax1.text(
            r["verified_deaths"] + 1.0,
            i,
            f"{int(r['verified_deaths'])} deaths / {int(r['verified_cases'])} cases ({r['positive_death_outbreak_weeks']}/{r['outbreak_weeks_count']} wks)",
            va="center",
            fontsize=8.2,
            fontweight="bold",
        )
    ax1.set_yticks(y1)
    ax1.set_yticklabels(pos_dis["disease"], fontsize=8.8, fontweight="bold")
    ax1.set_xlim(0, max(pos_dis["verified_deaths"]) * 1.38)
    ax1.set_xlabel("Total Verified Outbreak Deaths (2023–2025)", fontsize=9.5, fontweight="bold")
    ax1.set_title("A. All 15 Diseases with ≥ 1 Verified Outbreak Death (Total = 199)", fontsize=10.5, fontweight="bold")
    ax1.grid(axis="x", linestyle=":", alpha=0.45)

    # Panel B: Outbreak-Cluster CFR (%) with Wilson 95% CI for diseases with N >= 30 cases
    ax2 = axes[1]
    stable_dis = (
        disease_summary_df[disease_summary_df["verified_cases"] >= 30]
        .sort_values("calculated_ratio", ascending=True)
        .reset_index(drop=True)
    )
    y2 = np.arange(len(stable_dis))
    cfr = stable_dis["calculated_ratio"].to_numpy(dtype=float)
    low = stable_dis["wilson_95ci_lower_pct"].to_numpy(dtype=float)
    high = stable_dis["wilson_95ci_upper_pct"].to_numpy(dtype=float)
    eb = stable_dis["empirical_bayes_shrunk_cfr_pct"].to_numpy(dtype=float)

    ax2.errorbar(
        cfr,
        y2,
        xerr=[np.maximum(0.0, cfr - low), np.maximum(0.0, high - cfr)],
        fmt="o",
        color="#B91C1C",
        ecolor="#64748B",
        elinewidth=1.6,
        capsize=3.5,
        label="Crude Outbreak-Cluster CFR (%) ± Wilson 95% CI",
    )
    ax2.scatter(
        eb,
        y2,
        marker="D",
        s=42,
        color="#2563EB",
        zorder=4,
        label="Empirical Bayes Shrunk CFR (M=50 pseudocounts)",
    )
    ax2.set_yticks(y2)
    ax2.set_yticklabels(
        [f"{r['disease']} (N={int(r['verified_cases'])})" for _, r in stable_dis.iterrows()],
        fontsize=8.5,
        fontweight="bold",
    )
    ax2.set_xlabel("Outbreak-Cluster Case Fatality Ratio (%)", fontsize=9.5, fontweight="bold")
    ax2.set_title(
        "B. Outbreak-Cluster CFR (%) & Wilson 95% CIs (Diseases with N ≥ 30 Cases)",
        fontsize=10.5,
        fontweight="bold",
    )
    ax2.legend(loc="lower right", fontsize=8.3)
    ax2.grid(axis="x", linestyle=":", alpha=0.45)

    fig.tight_layout(rect=[0, 0.01, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_03_district_mortality_comparison(
    district_summary_df: pd.DataFrame, out_path: Path
) -> Path:
    """Figure 3: District-Wise Verified Outbreak Deaths & Outbreak-Cluster CFR Comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 7.0), dpi=160)
    fig.suptitle(
        "JEEVAN-DRISHTI Phase 12: District-Wise Aggregate Outbreak Mortality & Case Fatality Comparison (2023–2025)\n"
        "[34 Active Surveillance Districts; Mumbai City & Ratnagiri Marked Insufficient Data (0 Reported Outbreaks)]",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )

    top15 = (
        district_summary_df[district_summary_df["verified_deaths"].notna()]
        .sort_values(["verified_deaths", "verified_cases"], ascending=[False, False])
        .head(15)
        .iloc[::-1]
        .reset_index(drop=True)
    )
    y_pos = np.arange(len(top15))

    # Panel A: Total Verified Deaths + Dominant Fatal Disease Annotation
    ax1 = axes[0]
    ax1.barh(y_pos, top15["verified_deaths"], color="#DC2626", edgecolor="#0F172A", height=0.62)
    for i, r in top15.iterrows():
        ax1.text(
            r["verified_deaths"] + 0.5,
            i,
            f"{int(r['verified_deaths'])} deaths ({int(r['positive_death_outbreak_weeks'])}/{int(r['outbreak_weeks_count'])} wks) | Dom: {r['dominant_fatal_disease']}",
            va="center",
            fontsize=8.0,
            fontweight="bold",
        )
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top15["district"], fontsize=8.8, fontweight="bold")
    ax1.set_xlim(0, max(top15["verified_deaths"]) * 1.52)
    ax1.set_xlabel("Verified Outbreak Deaths (2023–2025)", fontsize=9.5, fontweight="bold")
    ax1.set_title("A. Top 15 Maharashtra Districts by Total Verified Outbreak Deaths", fontsize=10.5, fontweight="bold")
    ax1.grid(axis="x", linestyle=":", alpha=0.45)

    # Panel B: Crude vs Empirical Bayes CFR (%) for the same Top 15 Fatal Districts
    ax2 = axes[1]
    cfr = top15["calculated_ratio"].to_numpy(dtype=float)
    low = top15["wilson_95ci_lower_pct"].to_numpy(dtype=float)
    high = top15["wilson_95ci_upper_pct"].to_numpy(dtype=float)
    eb = top15["empirical_bayes_shrunk_cfr_pct"].to_numpy(dtype=float)

    ax2.errorbar(
        cfr,
        y_pos,
        xerr=[np.maximum(0.0, cfr - low), np.maximum(0.0, high - cfr)],
        fmt="o",
        color="#B91C1C",
        ecolor="#64748B",
        elinewidth=1.5,
        capsize=3.5,
        label="District Crude OC-CFR (%) ± Wilson 95% CI",
    )
    ax2.scatter(
        eb,
        y_pos,
        marker="D",
        s=42,
        color="#2563EB",
        zorder=4,
        label="Empirical Bayes Shrunk OC-CFR (M=50)",
    )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(
        [f"{r['district']} (N={int(r['verified_cases'])})" for _, r in top15.iterrows()],
        fontsize=8.5,
        fontweight="bold",
    )
    ax2.set_xlabel("District Outbreak-Cluster Case Fatality Ratio (%)", fontsize=9.5, fontweight="bold")
    ax2.set_title("B. District Outbreak-Cluster CFR (%) & Small-Sample EB Shrinkage", fontsize=10.5, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=8.3)
    ax2.grid(axis="x", linestyle=":", alpha=0.45)

    fig.tight_layout(rect=[0, 0.01, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_04_temporal_and_seasonal_mortality_trends(
    weekly_series_df: pd.DataFrame,
    seasonal_df: pd.DataFrame,
    temporal_stats: Dict[str, Any],
    out_path: Path,
) -> Path:
    """Figure 4: Weekly Statewide Mortality Timeline (2023-2025) & Seasonal Mortality Comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.8), dpi=160, gridspec_kw={"width_ratios": [1.45, 1.0]})
    fig.suptitle(
        "JEEVAN-DRISHTI Phase 12: Temporal & Seasonal Outbreak Mortality Trends Across Maharashtra (2023–2025)\n"
        "[121 Active Surveillance Weeks, 199 Total Verified Deaths — Stacked by Major Fatal Diseases]",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )

    ax1 = axes[0]
    x = np.arange(len(weekly_series_df))
    d_den = weekly_series_df["dengue_deaths"].to_numpy(dtype=float)
    d_mal = weekly_series_df["malaria_deaths"].to_numpy(dtype=float)
    d_aes = weekly_series_df["aes_deaths"].to_numpy(dtype=float)
    d_add = weekly_series_df["add_deaths"].to_numpy(dtype=float)
    d_oth = weekly_series_df["other_disease_deaths"].to_numpy(dtype=float)

    ax1.bar(x, d_den, color="#DC2626", width=0.85, label="Dengue (82 deaths)")
    ax1.bar(x, d_mal, bottom=d_den, color="#D97706", width=0.85, label="Malaria (40 deaths)")
    ax1.bar(
        x,
        d_aes,
        bottom=d_den + d_mal,
        color="#7C3AED",
        width=0.85,
        label="Acute Encephalitis Syndrome (26 deaths)",
    )
    ax1.bar(
        x,
        d_add,
        bottom=d_den + d_mal + d_aes,
        color="#0891B2",
        width=0.85,
        label="Acute Diarrhoeal Disease (12 deaths)",
    )
    ax1.bar(
        x,
        d_oth,
        bottom=d_den + d_mal + d_aes + d_add,
        color="#64748B",
        width=0.85,
        label="Other Outbreak Diseases (39 deaths)",
    )
    ax1.plot(
        x,
        weekly_series_df["rolling_4wk_mean_deaths"],
        color="#0F172A",
        linewidth=2.0,
        label="4-Week Rolling Mean Deaths",
    )

    tick_idx = np.linspace(0, len(weekly_series_df) - 1, 10, dtype=int)
    ax1.set_xticks(tick_idx)
    ax1.set_xticklabels(
        weekly_series_df["reporting_period"].iloc[tick_idx],
        rotation=30,
        ha="right",
        fontsize=8.3,
        fontweight="bold",
    )
    ax1.set_ylabel("Verified Outbreak Deaths per Active Week", fontsize=9.5, fontweight="bold")
    ax1.set_title("A. Weekly Statewide Outbreak Mortality Timeline (2023-W01 to 2025-W52)", fontsize=10.5, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=8.0)
    ax1.grid(axis="y", linestyle=":", alpha=0.45)

    # Panel B: Seasonal Mortality Breakdown
    ax2 = axes[1]
    sx = np.arange(len(seasonal_df))
    ax2.bar(
        sx - 0.18,
        seasonal_df["verified_deaths"],
        width=0.35,
        color="#DC2626",
        edgecolor="#0F172A",
        label="Total Verified Deaths (2023–2025)",
    )
    ax2.bar(
        sx + 0.18,
        seasonal_df["positive_death_weeks"],
        width=0.35,
        color="#2563EB",
        edgecolor="#0F172A",
        label="Outbreak Weeks with ≥ 1 Death",
    )
    for i, r in seasonal_df.iterrows():
        ax2.text(
            i - 0.18,
            r["verified_deaths"] + 1.8,
            f"{int(r['verified_deaths'])}\n({r['share_of_statewide_deaths_pct']:.1f}%)",
            ha="center",
            fontsize=8.0,
            fontweight="bold",
        )
    ax2.set_xticks(sx)
    ax2.set_xticklabels(seasonal_df["season"], rotation=15, ha="right", fontsize=8.8, fontweight="bold")
    ax2.set_ylabel("Count (Deaths / Positive-Death Weeks)", fontsize=9.5, fontweight="bold")
    kw_h = temporal_stats.get("kruskal_wallis_seasonal_deaths_stat")
    kw_p = temporal_stats.get("kruskal_wallis_seasonal_deaths_pvalue")
    ax2.set_title(
        f"B. Seasonal Mortality Distribution\n(Kruskal-Wallis H = {kw_h}, p = {kw_p})",
        fontsize=10.2,
        fontweight="bold",
    )
    ax2.set_ylim(0, max(seasonal_df["verified_deaths"]) * 1.25)
    ax2.legend(loc="upper right", fontsize=8.3)
    ax2.grid(axis="y", linestyle=":", alpha=0.45)

    fig.tight_layout(rect=[0, 0.01, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_05_chronological_mortality_forecast_validation(
    metrics_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    out_path: Path,
) -> Path:
    """Figure 5: Chronological Mortality Forecasting Evaluation vs. Simple Baselines."""
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.8), dpi=160, gridspec_kw={"width_ratios": [1.35, 1.0]})
    fig.suptitle(
        "JEEVAN-DRISHTI Phase 12: Chronological Mortality Forecasting Validation (70% Train / 15% Val / 15% Test)\n"
        "[Evaluated on Statewide Weekly Mortality Series Using Strictly Lagged (t-1..t-4) Features vs. Simple Baselines]",
        fontsize=11.5,
        fontweight="bold",
        y=0.98,
    )

    # Panel A: Trajectory on statewide_weekly_all_diseases
    ax1 = axes[0]
    p_glm = preds_df[
        (preds_df["target_series"] == "statewide_weekly_all_diseases")
        & (preds_df["model_name"] == "Poisson_Autoregressive_GLM")
    ].reset_index(drop=True)
    p_roll = preds_df[
        (preds_df["target_series"] == "statewide_weekly_all_diseases")
        & (preds_df["model_name"] == "Rolling_4Week_Mean_Baseline")
    ].reset_index(drop=True)
    p_rf = preds_df[
        (preds_df["target_series"] == "statewide_weekly_all_diseases")
        & (preds_df["model_name"] == "RandomForest_Lagged_Count_Regressor")
    ].reset_index(drop=True)

    x = np.arange(len(p_glm))
    ax1.plot(
        x,
        p_glm["observed_verified_deaths"],
        color="#0F172A",
        linewidth=1.5,
        marker="o",
        markersize=3.2,
        label="Observed Weekly Verified Deaths",
    )
    ax1.plot(
        x,
        p_roll["predicted_deaths"],
        color="#64748B",
        linestyle="--",
        linewidth=1.5,
        label="Rolling 4-Week Mean Baseline",
    )
    ax1.plot(
        x,
        p_glm["predicted_deaths"],
        color="#DC2626",
        linewidth=1.8,
        label="Poisson Autoregressive GLM (Lagged)",
    )
    ax1.plot(
        x,
        p_rf["predicted_deaths"],
        color="#2563EB",
        linewidth=1.6,
        linestyle="-.",
        label="RandomForest Lagged Count Regressor",
    )
    ax1.fill_between(
        x,
        p_glm["pi_80_lower"],
        p_glm["pi_80_upper"],
        color="#FCA5A5",
        alpha=0.28,
        label="Poisson GLM 80% Prediction Interval",
    )

    val_start = int((p_glm["split"] == "train").sum())
    test_start = val_start + int((p_glm["split"] == "validation").sum())
    ax1.axvline(val_start - 0.5, color="#D97706", linestyle="--", linewidth=1.4, label="Validation Split Start")
    ax1.axvline(test_start - 0.5, color="#059669", linestyle="--", linewidth=1.4, label="Test Split Start")

    tick_idx = np.linspace(0, len(p_glm) - 1, 9, dtype=int)
    ax1.set_xticks(tick_idx)
    ax1.set_xticklabels(
        p_glm["reporting_period"].iloc[tick_idx],
        rotation=30,
        ha="right",
        fontsize=8.3,
        fontweight="bold",
    )
    ax1.set_ylabel("Weekly Statewide Verified Deaths", fontsize=9.5, fontweight="bold")
    ax1.set_title("A. Statewide Weekly All-Disease Mortality: Observed vs. Chronological Forecasts", fontsize=10.3, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=7.8, ncol=2)
    ax1.grid(axis="y", linestyle=":", alpha=0.45)

    # Panel B: Test & Validation MAE Comparison on statewide_weekly_all_diseases
    ax2 = axes[1]
    m_all_val = metrics_df[
        (metrics_df["target_series"] == "statewide_weekly_all_diseases")
        & (metrics_df["evaluation_split"] == "validation")
    ].set_index("model_name")
    m_all_test = metrics_df[
        (metrics_df["target_series"] == "statewide_weekly_all_diseases")
        & (metrics_df["evaluation_split"] == "test")
    ].set_index("model_name")

    model_order = list(m_all_test.index)[::-1]
    ym = np.arange(len(model_order))
    val_maes = [float(m_all_val.loc[m, "mae"]) for m in model_order]
    test_maes = [float(m_all_test.loc[m, "mae"]) for m in model_order]

    ax2.barh(ym + 0.18, val_maes, height=0.35, color="#D97706", edgecolor="#0F172A", label="Validation Split MAE")
    ax2.barh(ym - 0.18, test_maes, height=0.35, color="#2563EB", edgecolor="#0F172A", label="Test Split MAE (Untouched)")
    for i, m in enumerate(model_order):
        ax2.text(test_maes[i] + 0.03, i - 0.18, f"{test_maes[i]:.3f}", va="center", fontsize=8.0, fontweight="bold")
    ax2.set_yticks(ym)
    ax2.set_yticklabels(model_order, fontsize=8.3, fontweight="bold")
    ax2.set_xlim(0, max(max(val_maes), max(test_maes)) * 1.25)
    ax2.set_xlabel("Mean Absolute Error (Deaths / Week)", fontsize=9.5, fontweight="bold")
    ax2.set_title("B. Model Comparison Against Simple Baselines (All-Diseases)", fontsize=10.3, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=8.3)
    ax2.grid(axis="x", linestyle=":", alpha=0.45)

    fig.tight_layout(rect=[0, 0.01, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_all_severity_visualizations(
    availability_df: pd.DataFrame,
    missing_vs_zero_df: pd.DataFrame,
    disease_summary_df: pd.DataFrame,
    district_summary_df: pd.DataFrame,
    weekly_series_df: pd.DataFrame,
    seasonal_df: pd.DataFrame,
    temporal_stats: Dict[str, Any],
    metrics_df: pd.DataFrame,
    preds_df: pd.DataFrame,
    reports_dir: Path,
    data_dir: Path,
) -> List[str]:
    """Generate all 5 static PNG figures in reports/severity_intelligence/figures/ and mirror to data/."""
    rep_figs = reports_dir / "figures"
    dat_figs = data_dir / "figures"
    rep_figs.mkdir(parents=True, exist_ok=True)
    dat_figs.mkdir(parents=True, exist_ok=True)

    f1 = plot_01_outcome_availability_and_missing_vs_zero(
        availability_df,
        missing_vs_zero_df,
        rep_figs / "01_outcome_data_availability_and_missing_vs_zero_audit.png",
    )
    f2 = plot_02_disease_wise_mortality_and_wilson_cfr(
        disease_summary_df,
        rep_figs / "02_disease_wise_mortality_and_wilson_cfr_intervals.png",
    )
    f3 = plot_03_district_mortality_comparison(
        district_summary_df,
        rep_figs / "03_district_mortality_comparison_and_burden_ranking.png",
    )
    f4 = plot_04_temporal_and_seasonal_mortality_trends(
        weekly_series_df,
        seasonal_df,
        temporal_stats,
        rep_figs / "04_temporal_and_seasonal_mortality_trends_2023_2025.png",
    )
    f5 = plot_05_chronological_mortality_forecast_validation(
        metrics_df,
        preds_df,
        rep_figs / "05_chronological_mortality_forecast_validation.png",
    )

    figs = [f1, f2, f3, f4, f5]
    for fp in figs:
        shutil.copy2(fp, dat_figs / fp.name)
    return [str(fp) for fp in figs]
