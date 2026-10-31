# JEEVAN-DRISHTI Phase 10: District Spread Propagation & Scenario Simulation Report

> **MANDATORY SCIENTIFIC & NON-PREDICTION BOUNDARY**: HYPOTHETICAL SCENARIO SIMULATION ONLY — NOT A REAL OUTBREAK OR DISEASE-SPREAD PREDICTION. No empirical human mobility, road-traffic, or biological transmission parameters exist in the workspace; propagation reflects user-configured graph adjacency assumptions only.

- **Generated At**: `2026-10-31T16:30:17`
- **Engine Version**: `phase10-v1.0.0`
- **Simulation Method**: Transparent, Deterministic Graph-Based Signal Propagation Calculator (`0–100` Hypothetical Stress Index) over the 36-District Maharashtra Network.

## 1. Geographic, Road-Network & Mobility Asset Inspection

| Data Asset | Available in Workspace? | Engine Handling & Non-Fabrication Safeguard |
|---|---|---|
| **District Adjacency (`spatial_weights_matrix.csv`)** | `Yes` (`data/geospatial/spatial_weights_matrix.csv`) | **83 undirected edges (`166` directed arcs)** across **35 geocoded districts** (`degree 4..7`, mean `4.74`). |
| **District Point Coordinates (`lat`, `lon`)** | `Yes` (`35 / 36` districts in `data/interim/district_grid_cells.csv`) | `Mumbai Suburban` (`MH_D19`) has no coordinates in workspace; represented as an explicit isolated node (`degree = 0`) with zero invented edges. |
| **District Polygon Boundaries (`.geojson`, `.shp`)** | `No` (`0` files) | Not downloaded or invented. |
| **Road-Network Data** | `No` (`0` files) | Not downloaded or invented. |
| **Human Mobility / Migration / Traffic Data** | `No` (`0` files) | **Equal movement is NOT assumed**, and traffic/migration numbers are **never invented**. Edge weights use documented geometric proximity functions strictly for illustrative scenario exploration. |
| **Population Denominators** | `No` (`0` files) | Compartmental `SEIR` population rates are not fabricated; engine computes a transparent dimensionless signal index (`0–100`). |
| **Historical Disease Surveillance (`2023–2025`)** | `Yes` (`data/processed/cleaned_disease_data.csv`) | Retained strictly as a separate reference column (`observed_2023_2025_mean_weekly_cases_ref`); **never mixed with simulated values**. |

## 2. District Graph Structure & Edge Weighting Modes

- **Nodes ($V$)**: `36` official Maharashtra districts (`MH_D01..MH_D36`). `35` nodes are connected in a single island-free component; `Mumbai Suburban` (`MH_D19`) has `degree = 0` (`ISOLATED_NO_COORDINATES_NO_PROPAGATION_EDGES`).
- **Edges ($E$)**: `83` undirected edges (`166` directed arcs) with great-circle Haversine distance $d_{ij} \in [36.26\text{ km}, 178.95\text{ km}]$ and administrative division flags.
- **Configurable Edge Weighting Assumptions ($W_{\text{mode}}$)** (row-normalized per receiving district):
  1. `distance_decayed`: $w_{ij} \propto \exp(-d_{ij} / \lambda_{\text{km}})$ — closer neighbours exert stronger hypothetical geometric coupling.
  2. `inverse_distance`: $w_{ij} \propto 1 / d_{ij}$.
  3. `division_bounded`: $w_{ij} \propto \exp(-d_{ij} / \lambda_{\text{km}}) \cdot \gamma_{\text{cross}}$ when districts $i$ and $j$ belong to different revenue divisions.
  4. `uniform_topological`: $w_{ij} = 1 / \deg(i)$ — pure graph-topology baseline.

## 3. Transparent Graph-Based Scenario Update Equation

For each geocoded district $i$ and hypothetical step $t \rightarrow t + 1$:
$$m_{\text{int}}(t+1) = \begin{cases} 1 - \eta_{\text{cov}}, & \text{if } t + 1 \ge t_{\text{int}} \\ 1.0, & \text{if } t + 1 < t_{\text{int}} \end{cases}$$
$$P_i(t+1) = \rho_{\text{persist}} \cdot m_{\text{int}}(t+1) \cdot S_i(t), \qquad I_i(t+1) = \beta_{\text{prop}} \cdot m_{\text{int}}(t+1) \sum_{j \in \mathcal{N}(i)} W_{ij}^{(\text{mode})} S_j(t)$$
$$S_i(t+1) = \min\!\left(100.0,\; P_i(t+1) + I_i(t+1) \cdot \left(1 - \frac{S_i(t)}{100.0}\right)\right)$$

## 4. Benchmark Hypothetical Scenarios (`SCEN_01` – `SCEN_06`)

| Scenario ID | Initial Seed District(s) | Edge Mode ($\lambda_{\text{km}}$) | $\beta_{\text{prop}}$ / $\rho_{\text{persist}}$ | Intervention ($t_{\text{int}}$, $\eta_{\text{cov}}$) | Ever Affected ($\ge 5.0$) | Peak Signal Sum (Step) | Final Step 12 Signal Sum |
|---|---|---|---|---|---|---|---|
| `SCEN_01_EAST_VIDARBHA_BASELINE` | `Gadchiroli (85.0)` | `distance_decayed` (120km) | `0.38` / `0.82` | Step `8` (`30%`) | **6** / 35 | **148.6** (`t=7`) | **47.7** (6 active) |
| `SCEN_02_EAST_VIDARBHA_EARLY_INTERVENTION` | `Gadchiroli (85.0)` | `distance_decayed` (120km) | `0.38` / `0.82` | Step `2` (`70%`) | **2** / 35 | **90.0** (`t=1`) | **0.0** (0 active) |
| `SCEN_03_KONKAN_MULTI_NODE_ELEVATION` | `Thane (75.0); Raigad (70.0); Palghar (65.0)` | `distance_decayed` (100km) | `0.40` / `0.80` | Step `5` (`50%`) | **7** / 35 | **271.7** (`t=4`) | **3.5** (0 active) |
| `SCEN_04_PUNE_KOLHAPUR_DIVISION_BOUNDED` | `Kolhapur (85.0); Pune (75.0)` | `division_bounded` (110km) | `0.36` / `0.80` | Step `6` (`45%`) | **12** / 35 | **247.5** (`t=5`) | **8.8** (0 active) |
| `SCEN_05_MARATHWADA_HIGH_PROPAGATION_UNMITIGATED` | `Nanded (90.0)` | `inverse_distance` (140km) | `0.55` / `0.85` | Step `12` (`0%`) | **31** / 35 | **1106.9** (`t=12`) | **1106.9** (31 active) |
| `SCEN_06_MARATHWADA_RAPID_CONTAINMENT` | `Nanded (90.0)` | `inverse_distance` (140km) | `0.55` / `0.85` | Step `3` (`65%`) | **5** / 35 | **143.7** (`t=2`) | **0.1** (0 active) |

### Step-by-Step Affected District Wavefront Sequences (`Activation Threshold >= 5.0`)

- **`SCEN_01_EAST_VIDARBHA_BASELINE`** (Eastern Vidarbha Single-District Elevation (Delayed Intervention)):
  - *Wavefront Sequence*: `Step 0: Gadchiroli -> Step 1: Chandrapur -> Step 2: Bhandara, Gondia, Nagpur -> Step 4: Wardha`
- **`SCEN_02_EAST_VIDARBHA_EARLY_INTERVENTION`** (Eastern Vidarbha Early Intervention & High Response Coverage (Paired Comparison)):
  - *Wavefront Sequence*: `Step 0: Gadchiroli -> Step 1: Chandrapur`
- **`SCEN_03_KONKAN_MULTI_NODE_ELEVATION`** (Northern Konkan Multi-District Simultaneous Signal Elevation):
  - *Wavefront Sequence*: `Step 0: Thane, Raigad, Palghar -> Step 1: Mumbai City, Nashik, Pune -> Step 4: Ahmednagar`
- **`SCEN_04_PUNE_KOLHAPUR_DIVISION_BOUNDED`** (Western Maharashtra Corridor (Division-Bounded Adjacency Assumption)):
  - *Wavefront Sequence*: `Step 0: Kolhapur, Pune -> Step 1: Satara, Sangli, Sindhudurg, Ratnagiri -> Step 2: Raigad, Ahmednagar -> Step 3: Mumbai City, Thane -> Step 5: Solapur, Palghar`
- **`SCEN_05_MARATHWADA_HIGH_PROPAGATION_UNMITIGATED`** (Marathwada High-Propagation Stress Scenario (No Intervention Baseline)):
  - *Wavefront Sequence*: `Step 0: Nanded -> Step 1: Hingoli, Latur, Parbhani, Washim -> Step 3: Dharashiv -> Step 4: Yavatmal, Beed, Solapur, Akola -> Step 5: Amravati, Buldhana, Jalna -> Step 7: Chhatrapati Sambhajinagar, Ahmednagar, Wardha, Jalgaon -> Step 9: Sangli, Dhule, Nandurbar -> Step 10: Nagpur, Chandrapur, Nashik, Bhandara, Gondia -> Step 11: Pune, Gadchiroli, Satara, Kolhapur, Ratnagiri, Sindhudurg`
- **`SCEN_06_MARATHWADA_RAPID_CONTAINMENT`** (Marathwada High-Propagation Stress Scenario (Rapid Containment at Step 3)):
  - *Wavefront Sequence*: `Step 0: Nanded -> Step 1: Hingoli, Latur, Parbhani, Washim`

## 5. Before-and-After Intervention Impact Comparison

1. **Eastern Vidarbha (`SCEN_01` Delayed Response vs. `SCEN_02` Early Intervention)**:
   - Under delayed intervention (`t_int = 8`, `30%` coverage), the hypothetical signal seeded in `Gadchiroli (85.0)` reaches **6 districts** with a peak statewide signal sum of **148.6** and final step-12 sum of **47.7**.
   - With early intervention (`t_int = 2`, `70%` coverage) under identical seed and propagation assumptions, the wavefront is restricted to **2 districts** and the final step-12 signal sum drops to **0.0** (a **100.0%** reduction at step 12).
2. **Marathwada Stress Test (`SCEN_05` Unmitigated vs. `SCEN_06` Rapid Containment)**:
   - Under unmitigated high propagation (`beta_prop = 0.55`, `0%` coverage) seeded in `Nanded (90.0)`, **31 districts** cross the activation threshold and final step-12 signal reaches **1106.9**.
   - Rapid containment at `step 3` (`65%` coverage) limits spread to **5 districts** and reduces final step-12 signal to **0.1** (a **100.0%** reduction).

## 6. Generated Interactive & Static Visualizations (`reports/spread_simulation/`)

- **Interactive HTML District Graph & Propagation Animation**: `reports/spread_simulation/interactive_spread_scenario_simulator.html`
- **Static Figure**: `reports/spread_simulation/figures/01_maharashtra_district_adjacency_graph.png`
- **Static Figure**: `reports/spread_simulation/figures/02_propagation_sequence_snapshots.png`
- **Static Figure**: `reports/spread_simulation/figures/03_scenario_timeline_and_affected_sequence.png`
- **Static Figure**: `reports/spread_simulation/figures/04_before_after_intervention_comparison.png`
- **Static Figure**: `reports/spread_simulation/figures/05_parameter_sensitivity_and_assumptions_panel.png`

## 7. Scientific Limitations & Interpretation Rules

1. **Hypothetical Exploration Only**: Simulated indicator values (`simulated_signal_index`) represent deterministic mathematical consequences of user-selected parameters on the Maharashtra district adjacency graph; they are **never** real outbreak predictions or forecasts.
2. **Absence of Human Mobility & Road Traffic Data**: Geographic proximity does not equal human travel volume. No commuter, rail, air, or highway traffic data exist in the workspace.
3. **Absence of Biological & Demographic Parameters**: No district population denominators or pathogen-specific $R_0$ parameters exist in the workspace; therefore, no compartmental `SEIR` incidence claims are made.
4. **Mumbai Suburban Isolation**: Because `Mumbai Suburban` (`MH_D19`) lacks coordinates in all project datasets, it has `0` graph edges and remains at `0.0` during neighbor-driven propagation unless explicitly seeded as an isolated node.
