# JEEVAN-DRISHTI — Phase 14: Public-Health Intervention Scenario Simulator Report

**Engine Version**: `phase14-v1.0.0`  
**Generated Timestamp**: `2026-10-31T17:33:32.073983`  
**Modelling Approach**: `Transparent Parametric What-If Scenario Calculator + Adjacency Ring Coupling + Little's Law Resource Demand`

> [!CAUTION]
> **GOVERNANCE SAFEGUARD**: DECISION-SUPPORT WHAT-IF SIMULATOR ONLY — NOT AN AUTOMATED PUBLIC-HEALTH AUTHORITY OR VALIDATED EFFICACY FORECAST. No empirical intervention effectiveness percentages exist in the project workspace. All simulated trajectories and resource comparisons are HYPOTHETICAL SCENARIOS derived strictly from visible, user-editable assumptions. This tool never recommends clinical treatments, never executes real-world interventions, and never replaces qualified public-health officials.

---

## 1. Upstream Project Capability Inspection (Section 1)

Before implementing any intervention simulation logic, all 7 upstream intelligence engines (Phases 6–12) and the workspace data inventory were audited:

| Upstream Engine / Dataset | Version | Genuinely Available | Verified Variables Used | Role in Intervention Simulator |
| :--- | :--- | :---: | :--- | :--- |
| Phase 6: Disease Forecasting Engine | `phase6-v1.0.0` | YES | 7d (1w), 14d (2w), and 30d (4w) forecasted weekly case counts across 34 active districts | Provides verified baseline weekly caseload anchors (B_{i,0}) for district-disease scenario trajectories. |
| Phase 7: Anomaly Detection Engine | `phase7-v1.0.0` | YES | Composite anomaly score, Seasonal Z, Rolling Z, Robust MAD Z, and 116 flagged outbreak weeks | Identifies elevated district-disease clusters to prioritize in what-if early vs. delayed response scenarios. |
| Phase 8: Climate–Disease Intelligence Engine | `phase8-v1.0.0` | YES | Seasonal Kruskal-Wallis (p=3.21e-05), lagged rainfall/temperature associations (1..8w) | Informs seasonal timing context (e.g., Monsoon/Post-Monsoon vector-control vs. Pre-Monsoon WASH). |
| Phase 9: Geospatial Hotspot Detection Engine | `phase9-v1.0.0` | YES | 36 district centroids, Global Moran's I, Local Getis-Ord Gi* z-scores (14 hotspots) | Supports district-wise scenario mapping and spatial cluster selection. |
| Phase 10: District Spread Propagation Engine | `phase10-v1.0.0` | YES | Verified 36-node, 83-edge (166 directed arcs) Maharashtra district adjacency graph | Powers multi-district ring-response scenarios and cross-border spillover dampening comparisons. |
| Phase 11: Healthcare Resource Demand Engine | `phase11-v1.0.0` | YES | Explicit NaN audit of hospital capacities + deterministic scenario calculator formulas | Computes hypothetical concurrent general beds, ICU beds, and diagnostic test requirements per scenario. |
| Phase 12: Disease Severity & Mortality Engine | `phase12-v1.0.0` | YES | Verified 2023-2025 outbreak cases (19,172), deaths (199), and Outbreak-Cluster CFR (%) | Provides historical disease-specific Outbreak-Cluster CFR context alongside simulated case trajectories. |
| Empirical Intervention Efficacy / RCT Dataset | `NONE_IN_WORKSPACE` | **NO (0 RCT Records)** | 0 empirical intervention efficacy or cost-effectiveness records exist in workspace | Blocked from fabrication — replaced by transparent, user-editable What-If Scenario Calculator parameters. |

---

## 2. Transparent Scenario Calculator Formulation (Sections 2–4)

Because **zero empirical intervention efficacy or randomized controlled trial (RCT) datasets** exist in the project workspace, the simulator implements a **Transparent Parametric What-If Scenario Calculator** where every assumption is explicitly visible and user-editable:

1. **Verified Baseline Caseload Anchor ($B_{i,0}$)**: Initialized from Phase 6 `forecast_7d_mean_cases` (or Phase 2 `observed_mean_cases_per_outbreak_week` fallback) for each selected district $i$ and disease.
2. **Unmitigated Baseline Trajectory ($B_{i,t}$)**:
   $$B_{i,t} = B_{i,t-1} \cdot g_0 + \beta_0 \cdot \frac{1}{|N(i) \cap D|} \sum_{j \in N(i) \cap D} B_{j,t-1}$$
3. **Intervention Activation Ramp ($r(t) \in [0, 1]$)**:
   $$r(t) = \begin{cases} 0 & \text{if } t < t_{\text{start}} \\ \min\!\left(1,\; \frac{t - t_{\text{start}} + 1}{w_{\text{ramp}}}\right) & \text{if } t \ge t_{\text{start}} \end{cases}$$
4. **Hypothetical Intervention Underlying Trajectory ($U_{i,t}$)**:
   $$U_{i,t} = U_{i,t-1} \cdot g_0 \cdot \bigl(1 - r(t)\,d_{\text{trans}}\bigr) + \beta_0 \cdot \bigl(1 - r(t)\,d_{\text{spill}}\bigr) \cdot \frac{1}{|N(i) \cap D|} \sum_{j \in N(i) \cap D} U_{j,t-1}$$
5. **Active Surveillance Case-Finding Signal ($S_{i,t}$)**:
   $$S_{i,t} = U_{i,t} \cdot \bigl(1 + 0.25 \cdot r(t) \cdot (m_{\text{test}} - 1)\bigr)$$
6. **Hypothetical Healthcare Resource Coupling (Phase 11 Little's Law)**:
   $$\text{General Beds}_{i,t} = U_{i,t} \cdot p_{\text{adm}} \cdot (1 - p_{\text{icu}}) \cdot \frac{\text{ALOS}}{7}, \qquad \text{ICU Beds}_{i,t} = U_{i,t} \cdot p_{\text{adm}} \cdot p_{\text{icu}} \cdot \frac{\text{ALOS}}{7}$$

---

## 3. Benchmark Scenario Catalog Across All 7 Scenario Types (Sections 2 & 5)

| Scenario ID | Intervention Type | Disease | Focal Districts | Start Wk | Assumed Dampening | Baseline Cases | Scenario Cases | Hypothetical Diff vs. Baseline | Peak Gen. Beds |
| :--- | :--- | :--- | :--- | :---: | :---: | ---: | ---: | ---: | ---: |
| `SCEN_00_BASELINE_DENGUE_URBAN_CLUSTER` | `BASELINE_NO_ADDITIONAL_INTERVENTION` | Dengue | Pune, Nashik, Thane, Mumbai Suburban | W2 | 0% | 500.9 | 500.9 | +0.0 (+0.0%) | 14.3 |
| `SCEN_01_INCREASED_SURVEILLANCE_AND_TESTING` | `INCREASED_SURVEILLANCE_AND_TESTING` | Dengue | Pune, Nashik, Thane, Mumbai Suburban | W2 | 18% | 500.9 | 263.1 | -237.9 (-47.5%) | 5.6 |
| `SCEN_02_ADDITIONAL_VECTOR_CONTROL_EARLY_W2` | `ADDITIONAL_VECTOR_CONTROL_ACTIVITY` | Dengue | Pune, Nashik, Thane, Mumbai Suburban | W2 | 28% | 500.9 | 192.0 | -308.9 (-61.7%) | 5.3 |
| `SCEN_03_DELAYED_INTERVENTION_START_W5` | `EARLY_INTERVENTION_VS_DELAYED_INTERVENTION` | Dengue | Pune, Nashik, Thane, Mumbai Suburban | W5 | 28% | 500.9 | 352.3 | -148.6 (-29.7%) | 8.1 |
| `SCEN_04_PUBLIC_AWARENESS_WASH_CAMPAIGN` | `PUBLIC_AWARENESS_CAMPAIGN` | Acute Diarrhoeal Disease | Pune, Satara, Sangli, Solapur, Kolhapur | W2 | 20% | 2867.8 | 1566.0 | -1301.8 (-45.4%) | 15.1 |
| `SCEN_05_INCREASED_HEALTHCARE_RESOURCE_ALLOCATION` | `INCREASED_HEALTHCARE_RESOURCE_ALLOCATION` | Dengue | Pune, Nashik, Thane, Mumbai Suburban | W2 | 10% | 500.9 | 330.1 | -170.9 (-34.1%) | 5.6 |
| `SCEN_06_DISTRICT_FOCUSED_RESPONSE_GADCHIROLI` | `DISTRICT_FOCUSED_RESPONSE` | Malaria | Gadchiroli | W2 | 30% | 179.9 | 70.5 | -109.4 (-60.8%) | 2.0 |
| `SCEN_07_MULTI_DISTRICT_RING_RESPONSE_VIDARBHA` | `MULTI_DISTRICT_RESPONSE` | Malaria | Gadchiroli, Chandrapur, Gondia, Bhandara | W2 | 30% | 1161.0 | 383.5 | -777.5 (-67.0%) | 9.7 |

---

## 4. Side-by-Side Comparison Suites: `Baseline` vs. `Scenario A` vs. `Scenario B` (Section 6)

| Comparison Suite ID | Disease | Baseline Cum. Cases | Scenario A Cum. Cases (Delta) | Scenario B Cum. Cases (Delta) | Scenario A vs. B Delta | Peak Gen. Beds (Base / A / B) |
| :--- | :--- | ---: | ---: | ---: | ---: | :---: |
| `CMP_01_EARLY_VS_DELAYED_VECTOR_CONTROL` | Dengue | 500.9 | 192.0 (-308.9) | 352.3 (-148.6) | -160.3 | 14.3 / 5.3 / 8.1 |
| `CMP_02_SURVEILLANCE_TESTING_VS_RESOURCE_ALLOCATION` | Dengue | 500.9 | 263.1 (-237.9) | 330.1 (-170.9) | -67.0 | 14.3 / 5.6 / 5.6 |
| `CMP_03_LOCAL_FOCUS_VS_MULTI_DISTRICT_COORDINATED_RING` | Malaria | 1161.0 | 689.1 (-472.0) | 383.5 (-777.5) | +305.6 | 29.0 / 10.9 / 9.7 |

---

## 5. Generated Visual & Interactive Artifacts (Section 7)

- **Figure 1 (Capability Audit)**: `C:\Users\Janhvi Suryavanshi\.gemini\antigravity\scratch\JEEVAN-DRISHTI\reports\intervention_simulator\figures\01_project_capability_and_intervention_support_audit.png`
- **Figure 2 (Baseline vs. Scenario A vs. Scenario B Timelines)**: `C:\Users\Janhvi Suryavanshi\.gemini\antigravity\scratch\JEEVAN-DRISHTI\reports\intervention_simulator\figures\02_baseline_vs_scenario_a_vs_scenario_b_timelines.png`
- **Figure 3 (District-Wise Scenario Map & Comparison)**: `C:\Users\Janhvi Suryavanshi\.gemini\antigravity\scratch\JEEVAN-DRISHTI\reports\intervention_simulator\figures\03_district_wise_scenario_comparison_map.png`
- **Figure 4 (Intervention Timing vs. Dampening Sensitivity Grid)**: `C:\Users\Janhvi Suryavanshi\.gemini\antigravity\scratch\JEEVAN-DRISHTI\reports\intervention_simulator\figures\04_intervention_timing_and_assumption_sensitivity_grid.png`
- **Figure 5 (Hypothetical Resource Allocation & Bed Strain)**: `C:\Users\Janhvi Suryavanshi\.gemini\antigravity\scratch\JEEVAN-DRISHTI\reports\intervention_simulator\figures\05_hypothetical_resource_allocation_and_bed_strain_comparison.png`
- **Interactive HTML Simulator**: `C:\Users\Janhvi Suryavanshi\.gemini\antigravity\scratch\JEEVAN-DRISHTI\reports\intervention_simulator\interactive_intervention_scenario_simulator.html`

---

## 6. Governance & Limitations Checklist (Section 8)

- **Zero Fabricated Efficacy Claims**: Every scenario table, chart, and JSON payload explicitly labels transmission dampening, spillover dampening, and testing scale-up as user-editable what-if assumptions.
- **Strict Separation from Observed Records**: All weekly simulation rows carry `record_nature_label = HYPOTHETICAL_SIMULATED_TRAJECTORY_NEVER_MIX_WITH_OBSERVED` and are stored exclusively in `data/intervention_scenarios/`.
- **Non-Clinical Decision Support Only**: The engine never recommends clinical treatments, never executes automated policies, and never replaces qualified public-health epidemiologists.
