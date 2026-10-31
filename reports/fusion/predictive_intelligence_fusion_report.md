# JEEVAN-DRISHTI — Phase 15: Central Predictive Intelligence Fusion Engine Report

**Engine Version**: `phase15-v1.0.0`  
**Generated Timestamp**: `2026-10-31T17:47:26.416413`  
**Fusion Architecture**: `Transparent Tiered Evidence-Based Rule Hierarchy with Explicit Disagreement Preservation`

> [!CAUTION]
> **GOVERNANCE SAFEGUARD**: EVIDENCE-BASED DECISION-SUPPORT FUSION ONLY — NOT AN AUTOMATED OUTBREAK DECLARATION. This engine combines heterogeneous analytical signals using transparent, auditable rules without blindly averaging incompatible model scores. Disagreements between engines are explicitly preserved for human epidemiological review. Clinical severity and hospital resource inventories remain unavailable (NaN) in the workspace and are never fabricated.

---

## 1. Upstream Analytical Engine Inspection (Section 1)

All **9 upstream analytical engines** (Phases 5–13) were inspected and verified before integration:

| Engine ID | Engine Name | Version | Verified Rows/Panels | Temporal Grain | Reliability Tier | Role in Fusion Engine |
| :--- | :--- | :--- | ---: | :--- | :--- | :--- |
| `ENGINE_01_RISK_CLASSIFICATION` | Risk Classification Engine (Phase 5) | `phase5-v1.0.0` | `511` | `WEEKLY_OUTBREAK_EVENT (YYYY-Www)` | `TIER_1_PRIMARY_EMPIRICAL_PREDICTIVE` | Flags elevated outbreak probability (>= 0.265 optimal F1 threshold); downgraded if disease < 5 train rows. |
| `ENGINE_02_TIME_SERIES_FORECASTING` | Time-Series Forecasting Engine (Phase 6) | `phase6-v1.0.0` | `1533` | `WEEKLY_OUTBREAK_EVENT (7d, 14d, 30d horizons)` | `TIER_1_PRIMARY_EMPIRICAL_PREDICTIVE` | Flags projected case trajectory growth or high multi-horizon case volume. |
| `ENGINE_03_ANOMALY_DETECTION` | Anomaly Detection Engine (Phase 7) | `phase7-v1.0.0` | `511` | `WEEKLY_OUTBREAK_EVENT (YYYY-Www)` | `TIER_1_PRIMARY_EMPIRICAL_DETECTION` | Flags statistically unusual weekly case spikes, spatial surges, or mortality deviations. |
| `ENGINE_04_CLIMATE_DISEASE_INTELLIGENCE` | Climate–Disease Intelligence Engine (Phase 8) | `phase8-v1.0.0` | `36` | `WEEKLY_ALIGNED_LAGS + DISTRICT_2023_2025_POOL` | `TIER_2_CONTEXTUAL_ENVIRONMENTAL_ASSOCIATION` | Provides non-causal environmental conduciveness context (rainfall accumulation & seasonal regime). |
| `ENGINE_05_GEOSPATIAL_HOTSPOT_DETECTION` | Geospatial Hotspot Detection Engine (Phase 9) | `phase9-v1.0.0` | `1188` | `POOLED_2023_2025_AND_ANNUAL_CROSS_SECTION` | `TIER_2_CONTEXTUAL_SPATIAL_CLUSTERING` | Identifies whether a district-disease pair sits inside a statistically significant spatial hotspot (p < 0.05). |
| `ENGINE_06_SPREAD_SCENARIO_SIMULATION` | District Spread Scenario Engine (Phase 10) | `phase10-v1.0.0` | `36` | `STATIC_TOPOLOGY_AND_HYPOTHETICAL_SCENARIO` | `TIER_3_HYPOTHETICAL_SCENARIO_CONTEXT_ONLY` | Provides verified topological neighbor connectivity; hypothetical spread scores are never treated as empirical signals. |
| `ENGINE_07_HEALTHCARE_RESOURCE_DEMAND` | Healthcare Resource Demand Engine (Phase 11) | `phase11-v1.0.0` | `36` | `DISTRICT_AUDIT_AND_HYPOTHETICAL_CALCULATOR` | `TIER_3_CAPACITY_GAP_AND_WHAT_IF_CALCULATOR` | Flags empirical hospital resource data unavailability (NaN) and provides hypothetical surge bed estimates. |
| `ENGINE_08_SEVERITY_AND_MORTALITY` | Disease Severity and Mortality Engine (Phase 12) | `phase12-v1.0.0` | `511` | `WEEKLY_OUTBREAK_EVENT + POOLED_2023_2025` | `TIER_1_PRIMARY_EMPIRICAL_SEVERITY` | Provides verified outbreak mortality signals and denominator-aware Outbreak-Cluster CFR context. |
| `ENGINE_09_EXPLAINABLE_AI` | Centralized Explainable AI Engine (Phase 13) | `phase13-v1.0.0` | `7` | `WEEKLY_LOCAL_ATTRIBUTIONS + DISTRICT_PANELS` | `TIER_1_EXPLAINABILITY_TRACEABILITY_LAYER` | Supplies feature-level 'Why this result?' traceability for every fused district-disease record. |

---

## 2. Common Output Schema, Alignment & Validation Gate (Sections 2 & 3)

1. **Standardized Common Output Schema**: Every fused record contains `district_id`, `district_name`, `disease`, `reporting_period`, `source_observation_date`, `risk_classification_output`, `forecast_output`, `anomaly_output`, `climate_indicators`, `geospatial_indicators`, `resource_indicators`, `severity_indicators`, `model_versions`, `data_quality_status`, and `generated_timestamp`.
2. **Strict Temporal Grain Separation**:
   - **Weekly Event Grain (`513` records)**: Matches `(district_id, disease, reporting_period=YYYY-Www)` across all `511` deduplicated `2023–2025` outbreak weeks plus `2` explicit `Insufficient data` records for the zero-surveillance districts (`Mumbai City`, `Ratnagiri`).
   - **36-District Pooled Grain (`36` records)**: Combines `2023-2025_POOL` district summaries across all `36` official Maharashtra districts.
   - **Cross-Sectional Context Tagging**: Pooled indicators (Phase 9 Getis-Ord $G_i^*$, Phase 10 graph degree, Phase 11 resource audit) are explicitly tagged as `2023-2025_POOL_CROSS_SECTIONAL_CONTEXT` when attached to weekly records so incompatible temporal periods are never silently conflated.
3. **Validation Checks Enforced**:
   - District alias standardization (`Aurangabad -> Chhatrapati Sambhajinagar`, `Osmanabad -> Dharashiv`) and rejection of non-Maharashtra districts.
   - Incompatible reporting period rejection (`check_reporting_period_compatibility`).
   - Model version & timestamp freshness verification (`evaluate_model_and_data_freshness`).
   - Pre-fusion deduplication across all upstream tables.

---

## 3. Transparent Rule-Based Fusion Architecture & Reliability Hierarchy (Sections 4 & 5)

Because a calibrated probability (`0.28`), a case count forecast (`45.2`), a composite anomaly score (`0.85`), and a spatial $G_i^*$ $z$-score (`+2.41`) represent **incomparable mathematical quantities**, this engine **never averages model scores**. Instead, it evaluates a 3-tier evidence hierarchy:

- **Tier 1 — Primary Empirical Weekly Pillars (`4` Pillars)**:
  1. **Risk Classification Pillar (`R_flag`)**: Elevated if `calibrated_risk_probability >= 0.265` (F1-optimal threshold) and disease tier is not `UNVALIDATED_LIMITED_SAMPLE` ($n < 5$ training weeks).
  2. **Time-Series Forecast Pillar (`F_flag`)**: Elevated if `forecast_7d_cases >= 1.15 * observed_cases` (with $\ge 15$ cases), or `forecast_14d_cases >= 1.10 * forecast_7d_cases` ($\ge 20$ cases), or `forecast_7d_cases >= 45.0`.
  3. **Anomaly Detection Pillar (`A_flag`)**: Elevated if `is_anomaly == True` or `composite_anomaly_score >= 0.55`.
  4. **Verified Severity / Mortality Pillar (`M_flag`)**: Elevated if `verified_outbreak_deaths >= 1`.
- **Tier 2 — Contextual Environmental & Spatial Pillars (`2` Pillars)**:
  5. **Climate Conduciveness Context (`C_context`)**: Supportive if `risk_shap_climate_log_odds >= +0.05` or `prev_4w_rainfall_accum_mm >= 50.0` (correlational context only).
  6. **Geospatial Hotspot Context (`G_context`)**: Supportive if Getis-Ord $G_i^*$ is statistically significant ($p < 0.05$).
- **Tier 3 — Capacity & Connectivity Caveats**:
  7. **Resource Demand (`Res_status`)**: Explicitly reports `Insufficient Data (No Hospital/Resource Dataset in Workspace)` (`NaN` beds/ICU) and points to the Phase 11/14 What-If Calculators.

### Canonical Signal Status Distribution (`513` Weekly Records vs. `36` Pooled District Records)

| Canonical Signal Status (Section 5) | Weekly Records (`N = 513`) | Pooled District Records (`N = 36`) | Rule Definition |
| :--- | ---: | ---: | :--- |
| `Multiple indicators aligned` | `157` | `31` | $\ge 2$ Tier-1 primary pillars simultaneously elevated |
| `Conflicting indicators` | `137` | `0` | 1 primary pillar strongly elevated while other primary pillars contradict it |
| `Elevated signal` | `51` | `1` | 1 primary or spatial hotspot indicator elevated without sharp contradiction |
| `No unusual signal detected` | `166` | `2` | All 4 primary pillars within normal/baseline bounds |
| `Insufficient data` | `2` | `2` | Zero surveillance records in `2023–2025` (`Mumbai City`, `Ratnagiri`) or $<2$ primary pillars available |

### Documented Epidemiological Review Priority Breakdown (`513` Weekly Records)

| Review Priority Tier | Weekly Count | Assignment Rule |
| :--- | ---: | :--- |
| `PRIORITY_1_IMMEDIATE_EPIDEMIOLOGICAL_REVIEW` | `136` | `Multiple indicators aligned` + (`verified_deaths >= 1` or $\ge 3$ aligned pillars or `HIGH_PRIORITY` anomaly) |
| `PRIORITY_2_TARGETED_ANALYST_REVIEW` | `195` | `Conflicting indicators` (analyst adjudication required) or `Multiple indicators aligned` (0 deaths) or `Elevated signal` with contextual support |
| `PRIORITY_3_ROUTINE_ENHANCED_MONITORING` | `14` | Single `Elevated signal` with 0 deaths |
| `PRIORITY_4_STANDARD_SURVEILLANCE` | `166` | `No unusual signal detected` |
| `PRIORITY_DATA_GAP_AUDIT` | `2` | `Insufficient data` (`Mumbai City`, `Ratnagiri`) |

---

## 4. Engine Disagreement Handling & Sample Conflict Queue (Section 6)

Across the `513` weekly records, **`237` records** exhibited explicit inter-engine disagreements (`ANOMALY_VS_PREDICTIVE_DISAGREEMENT`, `PREDICTIVE_VS_ANOMALY_DISAGREEMENT`, or `FORECAST_VS_RISK_DISAGREEMENT`). Every disagreeing record is preserved intact in `data/fused_intelligence/engine_disagreement_review_queue.csv`:

| District | Disease | Period | Status | Disagreement Type | Risk Prob | 7d Forecast | Anomaly Score | Deaths | Preserved Conflict Summary |
| :--- | :--- | :---: | :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| Thane | Dengue | `2023-W01` | `Multiple indicators aligned` | `ANOMALY_VS_PREDICTIVE_DISAGREEMENT` | `0.221` | `10.2` | `0.850` | `1` | ANOMALY_VS_PREDICTIVE_DISAGREEMENT: Anomaly Detector (phase7-v1.0.0, 2023-W01) flagged unusual activity (score=0.850, signals=UNUSUAL_M... |
| Ahmednagar | Dengue | `2023-W05` | `Multiple indicators aligned` | `ANOMALY_VS_PREDICTIVE_DISAGREEMENT` | `0.224` | `8.5` | `0.850` | `1` | ANOMALY_VS_PREDICTIVE_DISAGREEMENT: Anomaly Detector (phase7-v1.0.0, 2023-W05) flagged unusual activity (score=0.850, signals=UNUSUAL_M... |
| Dharashiv | Food Poisoning | `2023-W07` | `Conflicting indicators` | `PREDICTIVE_VS_ANOMALY_DISAGREEMENT` | `0.219` | `33.9` | `0.112` | `0` | PREDICTIVE_VS_ANOMALY_DISAGREEMENT: Predictive engine(s) (Forecaster in 2023-W07) indicated elevated trajectory/risk (risk_prob=0.219, ... |
| Pune | Measles | `2023-W13` | `Conflicting indicators` | `PREDICTIVE_VS_ANOMALY_DISAGREEMENT` | `0.268` | `15.3` | `0.157` | `0` | PREDICTIVE_VS_ANOMALY_DISAGREEMENT: Predictive engine(s) (Forecaster in 2023-W13) indicated elevated trajectory/risk (risk_prob=0.268, ... |
| Nanded | Dengue | `2023-W17` | `Conflicting indicators` | `PREDICTIVE_VS_ANOMALY_DISAGREEMENT` | `0.206` | `15.9` | `0.168` | `0` | PREDICTIVE_VS_ANOMALY_DISAGREEMENT: Predictive engine(s) (Forecaster in 2023-W17) indicated elevated trajectory/risk (risk_prob=0.206, ... |
| Yavatmal | Food Poisoning | `2023-W20` | `Conflicting indicators` | `PREDICTIVE_VS_ANOMALY_DISAGREEMENT` | `0.214` | `31.1` | `0.121` | `0` | PREDICTIVE_VS_ANOMALY_DISAGREEMENT: Predictive engine(s) (Forecaster in 2023-W20) indicated elevated trajectory/risk (risk_prob=0.214, ... |
| Gadchiroli | Malaria | `2023-W22` | `Multiple indicators aligned` | `ANOMALY_VS_PREDICTIVE_DISAGREEMENT` | `0.260` | `19.9` | `0.850` | `3` | ANOMALY_VS_PREDICTIVE_DISAGREEMENT: Anomaly Detector (phase7-v1.0.0, 2023-W22) flagged unusual activity (score=0.850, signals=SUDDEN_CA... |
| Dharashiv | Dengue | `2023-W24` | `Conflicting indicators` | `PREDICTIVE_VS_ANOMALY_DISAGREEMENT` | `0.217` | `15.9` | `0.137` | `0` | PREDICTIVE_VS_ANOMALY_DISAGREEMENT: Predictive engine(s) (Forecaster in 2023-W24) indicated elevated trajectory/risk (risk_prob=0.217, ... |
| Satara | Acute Diarrhoeal Disease | `2023-W24` | `Multiple indicators aligned` | `FORECAST_VS_RISK_DISAGREEMENT` | `0.205` | `34.8` | `0.335` | `1` | FORECAST_VS_RISK_DISAGREEMENT: 7d Forecaster (phase6-v1.0.0) projects 34.8 cases, whereas Risk Classifier (phase5-v1.0.0) assigned low ... |
| Dharashiv | Chikungunya | `2023-W25` | `Conflicting indicators` | `PREDICTIVE_VS_ANOMALY_DISAGREEMENT` | `0.234` | `16.1` | `0.108` | `0` | PREDICTIVE_VS_ANOMALY_DISAGREEMENT: Predictive engine(s) (Forecaster in 2023-W25) indicated elevated trajectory/risk (risk_prob=0.234, ... |

---

## 5. Generated Visual & Interactive Artifacts (Section 8)

- **Figure 1 (Upstream Engines & Reliability Hierarchy)**: `C:\Users\Janhvi Suryavanshi\.gemini\antigravity\scratch\JEEVAN-DRISHTI\reports\fusion\figures\01_upstream_engines_and_reliability_hierarchy_audit.png`
- **Figure 2 (Weekly Signal Status & Review Priority Distribution)**: `C:\Users\Janhvi Suryavanshi\.gemini\antigravity\scratch\JEEVAN-DRISHTI\reports\fusion\figures\02_weekly_fused_signal_status_and_priority_distribution.png`
- **Figure 3 (Maharashtra 36-District Fused Intelligence Map & Matrix)**: `C:\Users\Janhvi Suryavanshi\.gemini\antigravity\scratch\JEEVAN-DRISHTI\reports\fusion\figures\03_maharashtra_36_district_fused_intelligence_map_and_matrix.png`
- **Figure 4 (Engine Disagreement & Conflict Quadrants)**: `C:\Users\Janhvi Suryavanshi\.gemini\antigravity\scratch\JEEVAN-DRISHTI\reports\fusion\figures\04_engine_disagreement_and_conflict_analysis.png`
- **Figure 5 (Traceable 'Why This Result?' Evidence Cards)**: `C:\Users\Janhvi Suryavanshi\.gemini\antigravity\scratch\JEEVAN-DRISHTI\reports\fusion\figures\05_traceable_why_this_result_decomposition_cards.png`
- **Interactive HTML Unified Fusion Explorer**: `C:\Users\Janhvi Suryavanshi\.gemini\antigravity\scratch\JEEVAN-DRISHTI\reports\fusion\interactive_predictive_intelligence_fusion_explorer.html`

---

## 6. Uncertainty, Limitations & Governance Checklist (Section 10)

- **No Automated Outbreak Declarations**: Every record enforces `automated_outbreak_declaration = False`.
- **Zero Blind Score Averaging**: Probabilities, case counts, anomaly Z-scores, and spatial $G_i^*$ statistics are never averaged.
- **Preserved Disagreements**: Conflicting engine signals are explicitly surfaced in `conflicting_evidence` and routed to `PRIORITY_2_TARGETED_ANALYST_REVIEW`.
- **Documented Data Gaps**: Hospital bed capacities, ICU inventories, and clinical severity counts remain explicitly `NaN`/`None` across all fused records.
