"""Main Pipeline & Report Generator for Phase 8: Climate-Disease Intelligence Engine."""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from . import CAUSALITY_DISCLAIMER, CLIMATE_INTELLIGENCE_VERSION
from .advanced_analysis import run_advanced_ml_and_shap_analysis
from .data_alignment import build_aligned_climate_disease_datasets
from .statistical_analysis import run_statistical_and_lag_analysis
from .visualizations import generate_all_climate_disease_charts

PROJ = Path(__file__).resolve().parents[3]
CLIMATE_INTEL_DATA_DIR = PROJ / "data" / "climate_intelligence"
CLIMATE_INTEL_REPORTS_DIR = PROJ / "reports" / "climate_intelligence"
DATA_REPORTS_DIR = PROJ / "data" / "reports"
REGISTRY_PATH = PROJ / "model_dataset_registry.json"


def run_climate_intelligence_engine() -> Dict[str, Any]:
    """Execute the end-to-end Phase 8 Climate-Disease Intelligence Engine."""
    CLIMATE_INTEL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    CLIMATE_INTEL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    aligned_outbreaks, dw_panel, m_panel, audit = build_aligned_climate_disease_datasets()
    stat_res = run_statistical_and_lag_analysis(aligned_outbreaks, dw_panel)
    adv_res = run_advanced_ml_and_shap_analysis(aligned_outbreaks, dw_panel)

    fig_paths = generate_all_climate_disease_charts(
        aligned_outbreaks=aligned_outbreaks,
        dw_panel=dw_panel,
        state_wk=stat_res["state_wk"],
        lag_df=stat_res["lag_df"],
        district_rel_df=stat_res["district_rel_df"],
        imp_df=adv_res["imp_df"],
        pdp_df=adv_res["pdp_df"],
    )

    _write_comprehensive_report(audit, stat_res, adv_res, fig_paths)
    _update_registry(audit, stat_res, adv_res, fig_paths)

    return {
        "audit": audit,
        "stat_res": stat_res,
        "adv_res": adv_res,
        "figure_paths": fig_paths,
    }


def _write_comprehensive_report(
    audit: Dict[str, Any],
    stat_res: Dict[str, pd.DataFrame],
    adv_res: Dict[str, Any],
    fig_paths: List[str],
) -> None:
    """Write `climate_disease_intelligence_report.md` to `reports/climate_intelligence/` and `data/reports/`."""
    lag_df = stat_res["lag_df"]
    dis_df = stat_res["disease_corr_df"]
    dist_df = stat_res["district_rel_df"]
    seas_df = stat_res["seasonal_df"]
    reg_df = stat_res["reg_df"]
    imp_df = adv_res["imp_df"]

    lines = [
        "# JEEVAN-DRISHTI Phase 8: Climate–Disease Intelligence Report (Maharashtra, 2023–2025)",
        "",
        f"> **SCIENTIFIC INTERPRETATION MANDATE**: {CAUSALITY_DISCLAIMER}",
        "",
        f"- **Generated At**: `{audit['generated_at']}`",
        f"- **Engine Version**: `{CLIMATE_INTELLIGENCE_VERSION}`",
        "- **Data Period**: `2023-W01 to 2025-W52` (`2023-01-01 to 2025-12-31`, `121` active IDSP reporting weeks; `2009–2022` historical EpiClim file has zero date overlap with IMD/CWPRS climate files).",
        "- **Native Reporting Frequency**: Daily IMD/CWPRS climate aggregated to **Weekly** (`W01..W52`) and **Monthly** (`36 months`) to match native weekly IDSP outbreak surveillance.",
        "",
        "## 1. Available Climate & Disease Data Coverage Audit",
        "",
        "| Variable | Source File | Native Grain | Date Coverage | Spatial Coverage | Usable Scope |",
        "|---|---|---|---|---|---|",
        "| **Rainfall (`rain_mm`)** | `imdweb_5dgbAskl.nc` (0.25° grid) | Daily | `2023-01-01..2025-12-31` | **35 / 36 districts** (`Gadchiroli` via 3×3 neighborhood fallback; `Mumbai Suburban` has no coordinates) | Statewide weekly & monthly analysis |",
        "| **Min & Max Temp (`tmin_c`, `tmax_c`, derived `tavg_c`, diurnal range)** | `imdweb_Oe9H04Bg.nc`, `imdweb_tXexQbB6.nc` (1.0° grid) | Daily | `2023-01-01..2025-12-31` | **35 / 36 districts** (`Mumbai Suburban` missing coordinates) | Statewide weekly & monthly analysis |",
        "| **Relative Humidity (`rh_mean`)** | `humid_tel_hr_cwprs_mh_2021_2025.csv` | Sub-hourly | `2024-05-11..2025-06-09` (395 valid days) | **1 / 36 districts (`Pune` ONLY)** | Pune single-district sub-analysis only |",
        "| **Disease Outbreaks (`2023–2025`)** | `maharashtra_outbreaks_2023_2025.csv` | Weekly (`W01..W52`) | `2023-W01..2025-W52` (`121` active weeks) | **34 / 36 districts** (`511` deduplicated outbreak weeks; `508` climate-aligned; `Mumbai City` & `Ratnagiri` have `0` outbreaks) | 11 diseases with $\\ge 8$ records; 17 rare/unsupported diseases |",
        "",
        "## 2. Statewide Weekly Lagged Climate–Disease Associations (`121` Active IDSP Weeks)",
        "",
        "| Series | Climate Variable | Lag (Weeks) | N | Pearson $r$ (log1p) | Spearman $\\rho$ | $p$-value | FDR $q$-value | Statistical Association Verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    sw_sub = lag_df[lag_df["analysis_grain"] == "statewide_weekly_time_series_121w"].copy()
    sw_sub["abs_rho"] = sw_sub["spearman_rho"].abs()
    top_sw = sw_sub.sort_values("abs_rho", ascending=False).head(18)
    for _, r in top_sw.iterrows():
        lines.append(
            f"| `{r['disease_or_series']}` | `{r['climate_variable']}` | {int(r['lag_weeks'])}w | {int(r['n_obs'])} | "
            f"{r['pearson_r_log1p']:+.3f} | **{r['spearman_rho']:+.3f}** | {r['spearman_p']:.4f} | "
            f"{r['spearman_fdr_q']:.4f} | {r['statistical_association']} |"
        )

    lines.extend([
        "",
        "## 3. Disease-Wise Climate Correlation Summary & Four-Tier Scientific Interpretation",
        "",
        "| Disease | Aligned Outbreaks ($N$) | Districts | Dominant Season | Best Rainfall Lag ($\\rho$, $p$) | Best Temp Lag ($\\rho$, $p$) | Status | Observed Relationship | Statistical Association | Predictive Usefulness |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ])
    for _, r in dis_df.iterrows():
        r_str = (
            f"`{r['best_rainfall_feature']}` ({r['best_rainfall_spearman_rho']:+.3f}, p={r['best_rainfall_p_value']:.3f})"
            if pd.notna(r["best_rainfall_spearman_rho"])
            else "N/A (<8 obs)"
        )
        t_str = (
            f"`{r['best_temperature_feature']}` ({r['best_temperature_spearman_rho']:+.3f}, p={r['best_temperature_p_value']:.3f})"
            if pd.notna(r["best_temperature_spearman_rho"])
            else "N/A (<8 obs)"
        )
        lines.append(
            f"| **{r['disease_std']}** | {int(r['n_climate_aligned_outbreaks'])} | {int(r['districts_affected'])} | "
            f"`{r['dominant_season']}` | {r_str} | {t_str} | `{r['analysis_status']}` | "
            f"{r['observed_relationship']} | {r['statistical_association']} | {r['predictive_usefulness']} |"
        )

    lines.extend([
        "",
        "## 4.Pune Single-Station Relative Humidity Sub-Analysis (`May 2024 – Jun 2025`)",
        "",
        "| Grain | Series | RH Variable | Lag (Weeks) | N | Spearman $\\rho$ | $p$-value | Interpretation |",
        "|---|---|---|---|---|---|---|---|",
    ])
    pune_rh_sub = lag_df[lag_df["climate_group"] == "Relative_Humidity_Pune_Only"]
    for _, r in pune_rh_sub.iterrows():
        rho_s = f"{r['spearman_rho']:+.3f}" if pd.notna(r["spearman_rho"]) else "N/A"
        p_s = f"{r['spearman_p']:.4f}" if pd.notna(r["spearman_p"]) else "N/A"
        lines.append(
            f"| `{r['analysis_grain']}` | `{r['disease_or_series']}` | `{r['climate_variable']}` | "
            f"{int(r['lag_weeks'])}w | {int(r['n_obs'])} | **{rho_s}** | {p_s} | {r['statistical_association']} |"
        )

    lines.extend([
        "",
        "## 5. Seasonal Regime Comparisons (`Kruskal-Wallis` Non-Parametric Test)",
        "",
        "| Season | Active Weeks | Mean Weekly Rain (mm) | Mean Weekly Tavg (°C) | Mean Diurnal Range (°C) | Mean Weekly Outbreaks | Mean Vector-Borne Cases | Mean Water/Food-Borne Cases | Kruskal-Wallis Vector $H$ ($p$) | Kruskal-Wallis Water $H$ ($p$) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ])
    for _, r in seas_df.iterrows():
        lines.append(
            f"| `{r['season']}` | {int(r['active_reporting_weeks'])} | {r['mean_weekly_rainfall_mm']:.2f} | "
            f"{r['mean_weekly_tavg_c']:.2f} | {r['mean_weekly_diurnal_range_c']:.2f} | "
            f"{r['mean_weekly_total_outbreaks']:.2f} | **{r['mean_weekly_vector_borne_cases']:.2f}** | "
            f"**{r['mean_weekly_water_borne_cases']:.2f}** | {r['kruskal_wallis_vector_borne_H']:.2f} ({r['kruskal_wallis_vector_borne_p']:.4f}) | "
            f"{r['kruskal_wallis_water_borne_H']:.2f} ({r['kruskal_wallis_water_borne_p']:.4f}) |"
        )

    lines.extend([
        "",
        "## 6. Confounder-Adjusted Regression Analysis (Unadjusted vs. Seasonality & Geography Controlled)",
        "",
        "| Regression Model | N | Climate Predictor | Unadjusted $\\beta$ ($p$) | Season-Adjusted $\\beta$ ($p$) | Unadj $R^2$ -> Adj $R^2$ | Survives Seasonal Control? | Scientific Interpretation |",
        "|---|---|---|---|---|---|---|---|",
    ])
    for _, r in reg_df.iterrows():
        lines.append(
            f"| `{r['regression_model']}` | {int(r['n_observations'])} | `{r['climate_predictor']}` | "
            f"{r['unadjusted_std_beta']:+.4f} ({r['unadjusted_p_value']:.4f}) | "
            f"**{r['season_adjusted_std_beta']:+.4f}** ({r['season_adjusted_p_value']:.4f}) | "
            f"{r['unadjusted_r2']:.3f} -> {r['season_adjusted_r2']:.3f} | "
            f"`{r['remains_significant_after_confounder_control']}` | {r['scientific_interpretation']} |"
        )

    lines.extend([
        "",
        "## 7. Advanced ML Feature Importance & Exact TreeSHAP Summary (`Pooled Outbreak Severity`)",
        "",
        "| Rank | Feature Name | Role | RF Impurity | RF Permutation | XGBoost Gain | Mean `|TreeSHAP|` | TreeSHAP Share | Direction Corr | Consensus Score |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ])
    pool_imp = imp_df[imp_df["analysis_task"] == "Pooled_Outbreak_Severity_All_Diseases"].head(15)
    for idx, (_, r) in enumerate(pool_imp.iterrows(), start=1):
        lines.append(
            f"| {idx} | `{r['feature_name']}` | `{r['feature_role']}` | {r['rf_impurity_importance']:.4f} | "
            f"{r['rf_permutation_importance']:.4f} | {r['xgb_gain_importance']:.4f} | "
            f"**{r['mean_abs_treeshap']:.4f}** | {r['treeshap_share']:.1%} | "
            f"{r['shap_feature_direction_corr']:+.3f} | **{r['consensus_importance_score']:.4f}** |"
        )

    lines.extend([
        "",
        "## 8. District-Wise Climate–Disease Comparison (Top 20 Districts by Outbreak Weeks)",
        "",
        "| District | Division | Outbreak Weeks | Total Cases | Dominant Disease | Mean Weekly Rain (mm) | Mean Weekly Tavg (°C) | Panel Rain-4w $\\rho$ ($p$) | Panel Tavg-1w $\\rho$ ($p$) | Status |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ])
    for _, r in dist_df.head(20).iterrows():
        rain_m = f"{r['mean_weekly_rainfall_mm']:.1f}" if pd.notna(r["mean_weekly_rainfall_mm"]) else "N/A"
        tavg_m = f"{r['mean_weekly_tavg_c']:.1f}" if pd.notna(r["mean_weekly_tavg_c"]) else "N/A"
        r_rho = f"{r['panel_rain_4w_spearman_rho']:+.3f} ({r['panel_rain_4w_p_value']:.3f})" if pd.notna(r["panel_rain_4w_spearman_rho"]) else "N/A"
        t_rho = f"{r['panel_tavg_1w_spearman_rho']:+.3f} ({r['panel_tavg_1w_p_value']:.3f})" if pd.notna(r["panel_tavg_1w_spearman_rho"]) else "N/A"
        lines.append(
            f"| **{r['district_std']}** | {r['administrative_division']} | {int(r['outbreak_weeks_count'])} | "
            f"{int(r['total_outbreak_cases'])} | {r['dominant_disease']} | {rain_m} | {tavg_m} | "
            f"{r_rho} | {t_rho} | `{r['analysis_status']}` |"
        )

    lines.extend([
        "",
        "## 9. Generated Visualizations (`reports/climate_intelligence/figures/`)",
        "",
    ])
    for fp in fig_paths:
        lines.append(f"- `{fp}`")

    lines.extend([
        "",
        "## 10. Unsupported Analyses & Methodological Limitations",
        "",
        "1. **Statewide Relative Humidity Analysis Cannot Be Performed**: `humid_tel_hr_cwprs_mh_2021_2025.csv` contains measurements for a single station (`CWPRS Khadakvasla, Pune`) from `2024-05-11` to `2025-06-09`. Extrapolating Pune's relative humidity to coastal Konkan or semi-arid Vidarbha/Marathwada would be scientifically invalid.",
        "2. **2009–2022 Historical Records Cannot Be Aligned with 2023–2025 IMD Gridded Climate**: `maharashtra_epiclim_filtered.csv` ends on `2022-06-15`, whereas all three IMD NetCDF files begin on `2023-01-01`.",
        "3. **Mumbai Suburban Climate Alignment Unavailable**: `Mumbai Suburban` (`3` outbreak records in 2023–2025) has no latitude/longitude coordinates in the project lookup table and is excluded from gridded climate joins.",
        "4. **Zero-Record Districts & Diseases**: `Mumbai City` and `Ratnagiri` have `0` reported outbreaks in the 2023–2025 IDSP dataset; `Heat-related illness` has `0` records across all disease files; 17 rare diseases have `< 8` climate-aligned records and cannot support standalone parametric lag inference.",
        f"5. **Non-Causal Interpretation Boundary**: {CAUSALITY_DISCLAIMER}",
    ])

    md_text = "\n".join(lines) + "\n"
    (CLIMATE_INTEL_REPORTS_DIR / "climate_disease_intelligence_report.md").write_text(md_text, encoding="utf-8")
    (DATA_REPORTS_DIR / "climate_disease_intelligence_report.md").write_text(md_text, encoding="utf-8")


def _update_registry(
    audit: Dict[str, Any],
    stat_res: Dict[str, pd.DataFrame],
    adv_res: Dict[str, Any],
    fig_paths: List[str],
) -> None:
    """Update `model_dataset_registry.json` with Phase 8 Climate-Disease Intelligence deliverables."""
    if not REGISTRY_PATH.exists():
        return
    reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    reg["phase8_climate_disease_intelligence_engine"] = {
        "engine_version": CLIMATE_INTELLIGENCE_VERSION,
        "generated_at": audit["generated_at"],
        "causality_disclaimer": CAUSALITY_DISCLAIMER,
        "code_package": "src/engines/climate_intelligence",
        "data_outputs": [
            "data/climate_intelligence/aligned_weekly_climate_disease_outbreaks.csv",
            "data/climate_intelligence/aligned_district_week_surveillance_panel.csv",
            "data/climate_intelligence/aligned_monthly_climate_disease_panel.csv",
            "data/climate_intelligence/disease_climate_correlations.csv",
            "data/climate_intelligence/district_climate_relationships.csv",
            "data/climate_intelligence/lag_analysis_results.csv",
            "data/climate_intelligence/seasonal_comparison_results.csv",
            "data/climate_intelligence/regression_analysis_results.csv",
            "data/climate_intelligence/climate_feature_importance_and_shap.csv",
            "data/climate_intelligence/partial_dependence_profiles.csv",
            "data/climate_intelligence/data_coverage_and_alignment_audit.json",
        ],
        "report_outputs": [
            "reports/climate_intelligence/climate_disease_intelligence_report.md",
            "reports/climate_intelligence/disease_climate_correlations.csv",
            "reports/climate_intelligence/district_climate_relationships.csv",
            "reports/climate_intelligence/lag_analysis_results.csv",
            "reports/climate_intelligence/regression_analysis_results.csv",
            "reports/climate_intelligence/climate_feature_importance_and_shap.csv",
            "reports/climate_intelligence/partial_dependence_profiles.csv",
        ],
        "visualizations": fig_paths,
    }
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2), encoding="utf-8")


if __name__ == "__main__":
    res = run_climate_intelligence_engine()
    print(f"Climate-Disease Intelligence Engine completed. Generated {len(res['figure_paths'])} figures.")
