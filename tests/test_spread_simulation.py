"""Automated Validation Suite for JEEVAN-DRISHTI Phase 10 District Spread Scenario Engine.

Tests all required Prompt 10 Section 8 verification areas:
1. Graph construction & verified adjacency (36 nodes, 35 geocoded connected, Mumbai Suburban isolated).
2. Parameter boundaries & invalid input rejection.
3. Deterministic simulation behaviour, zero-propagation isolation, and intervention monotonicity.
4. Strict separation of simulated values from observed counts & non-prediction disclaimers.
5. Output files, static PNG figures, interactive HTML simulator, and service interface.
"""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.engines.spread_simulation.district_graph import (
    build_district_graph,
    compute_transition_weights_matrix,
)
from src.engines.spread_simulation.scenario_simulator import (
    ScenarioParameters,
    run_scenario_simulation,
)
from src.engines.spread_simulation.service import DistrictSpreadScenarioService

PROJ = Path(__file__).resolve().parents[1]
SCEN_DATA = PROJ / "data" / "scenarios"
SCEN_REP = PROJ / "reports" / "spread_simulation"


class TestPhase10SpreadSimulationEngine(unittest.TestCase):
    """Comprehensive verification suite for Phase 10 District Spread Scenario Engine."""

    @classmethod
    def setUpClass(cls):
        cls.graph = build_district_graph(PROJ)

    def test_01_graph_construction_and_adjacency(self):
        """Verify 36 nodes, symmetric 83 undirected edges (166 arcs), and zero invented edges for Mumbai Suburban."""
        g = self.graph
        self.assertEqual(len(g.nodes_df), 36)
        self.assertEqual(g.nodes_df["district_id"].nunique(), 36)
        self.assertEqual(int(g.nodes_df["has_verified_coordinates"].sum()), 35)

        # Symmetry and zero self-loops
        self.assertTrue(np.allclose(g.adj_binary, g.adj_binary.T))
        self.assertTrue(np.allclose(np.diag(g.adj_binary), 0.0))

        # Mumbai Suburban has no coordinates and must have 0 edges
        ms_idx = g.district_to_idx["Mumbai Suburban"]
        self.assertEqual(float(g.adj_binary[ms_idx].sum()), 0.0)

        # All other 35 geocoded districts have degree >= 4
        for d, idx in g.district_to_idx.items():
            if d != "Mumbai Suburban":
                self.assertGreaterEqual(int(g.adj_binary[idx].sum()), 4)

        # Verify all 4 edge weighting modes normalize rows to 1.0 for geocoded nodes and 0.0 for Mumbai Suburban
        for mode in ["distance_decayed", "inverse_distance", "division_bounded", "uniform_topological"]:
            W = compute_transition_weights_matrix(g, edge_weighting_mode=mode)
            self.assertEqual(W.shape, (36, 36))
            self.assertAlmostEqual(float(W[ms_idx].sum()), 0.0, places=6)
            for d, idx in g.district_to_idx.items():
                if d != "Mumbai Suburban":
                    self.assertAlmostEqual(float(W[idx].sum()), 1.0, places=6)

    def test_02_parameter_boundaries_and_invalid_inputs(self):
        """Verify invalid districts, out-of-range parameters, and bad modes raise ValueError."""
        g = self.graph

        # 1. Invalid district name
        with self.assertRaises(ValueError):
            ScenarioParameters(
                scenario_id="BAD_DIST",
                scenario_name="Bad District",
                scenario_description="Test",
                initial_affected_districts={"Atlantis": 50.0},
            ).validate_and_canonicalize(g)

        # 2. Empty initial_affected_districts
        with self.assertRaises(ValueError):
            ScenarioParameters(
                scenario_id="EMPTY_SEED",
                scenario_name="Empty",
                scenario_description="Test",
                initial_affected_districts={},
            ).validate_and_canonicalize(g)

        # 3. Initial signal outside (0, 100]
        with self.assertRaises(ValueError):
            ScenarioParameters(
                scenario_id="BAD_SIGNAL",
                scenario_name="Bad Signal",
                scenario_description="Test",
                initial_affected_districts={"Pune": 150.0},
            ).validate_and_canonicalize(g)

        # 4. Propagation factor outside [0, 1]
        with self.assertRaises(ValueError):
            ScenarioParameters(
                scenario_id="BAD_BETA",
                scenario_name="Bad Beta",
                scenario_description="Test",
                initial_affected_districts={"Pune": 50.0},
                hypothetical_propagation_factor=1.5,
            ).validate_and_canonicalize(g)

        # 5. Duration steps outside [1, 26]
        with self.assertRaises(ValueError):
            ScenarioParameters(
                scenario_id="BAD_DUR",
                scenario_name="Bad Duration",
                scenario_description="Test",
                initial_affected_districts={"Pune": 50.0},
                scenario_duration_steps=0,
            ).validate_and_canonicalize(g)

        # 6. Invalid edge weighting mode
        with self.assertRaises(ValueError):
            compute_transition_weights_matrix(g, edge_weighting_mode="invented_mobility")

        # 7. Historical alias ('Aurangabad') resolves cleanly to 'Chhatrapati Sambhajinagar'
        p_alias = ScenarioParameters(
            scenario_id="ALIAS_OK",
            scenario_name="Alias Test",
            scenario_description="Test",
            initial_affected_districts={"Aurangabad": 60.0},
        ).validate_and_canonicalize(g)
        self.assertIn("Chhatrapati Sambhajinagar", p_alias.initial_affected_districts)

    def test_03_deterministic_simulation_and_monotonic_behaviour(self):
        """Verify determinism, zero-propagation isolation, and intervention dampening monotonicity."""
        g = self.graph
        p1 = ScenarioParameters(
            scenario_id="DET_TEST",
            scenario_name="Determinism Test",
            scenario_description="Test",
            initial_affected_districts={"Gadchiroli": 80.0},
            scenario_duration_steps=10,
            hypothetical_propagation_factor=0.35,
            within_district_persistence_factor=0.80,
            intervention_start_step=4,
            intervention_response_coverage=0.45,
        )
        r1 = run_scenario_simulation(g, p1, simulation_timestamp="2026-10-31T16:00:00")
        r2 = run_scenario_simulation(g, p1, simulation_timestamp="2026-10-31T16:00:00")
        self.assertTrue(np.allclose(r1["S_matrix"], r2["S_matrix"]))
        self.assertEqual(
            r1["summary"]["affected_district_sequence"],
            r2["summary"]["affected_district_sequence"],
        )

        # When hypothetical_propagation_factor == 0.0, signal NEVER spreads to any other district
        p_zero_prop = ScenarioParameters(
            scenario_id="ZERO_PROP",
            scenario_name="Zero Propagation",
            scenario_description="Test",
            initial_affected_districts={"Pune": 75.0},
            scenario_duration_steps=8,
            hypothetical_propagation_factor=0.0,
            within_district_persistence_factor=0.80,
            intervention_start_step=8,
            intervention_response_coverage=0.0,
        )
        r_zero = run_scenario_simulation(g, p_zero_prop)
        self.assertEqual(r_zero["summary"]["total_districts_ever_affected"], 1)
        self.assertEqual(r_zero["summary"]["affected_district_sequence"], "Step 0: Pune")

        # Higher response coverage strictly reduces final step signal sum
        p_low_cov = ScenarioParameters(
            scenario_id="LOW_COV",
            scenario_name="Low Coverage",
            scenario_description="Test",
            initial_affected_districts={"Pune": 80.0},
            scenario_duration_steps=10,
            hypothetical_propagation_factor=0.40,
            intervention_start_step=2,
            intervention_response_coverage=0.10,
        )
        p_high_cov = ScenarioParameters(
            scenario_id="HIGH_COV",
            scenario_name="High Coverage",
            scenario_description="Test",
            initial_affected_districts={"Pune": 80.0},
            scenario_duration_steps=10,
            hypothetical_propagation_factor=0.40,
            intervention_start_step=2,
            intervention_response_coverage=0.75,
        )
        r_low = run_scenario_simulation(g, p_low_cov)
        r_high = run_scenario_simulation(g, p_high_cov)
        self.assertLess(
            r_high["summary"]["final_step_signal_sum"],
            r_low["summary"]["final_step_signal_sum"],
        )

    def test_04_outputs_visualizations_and_service(self):
        """Verify all Phase 10 CSVs, JSONs, 5 PNG figures, interactive HTML, and service exist and work."""
        required_data_files = [
            "district_graph_nodes.csv",
            "district_graph_edges.csv",
            "scenario_summary_catalog.csv",
            "scenario_simulation_step_records.csv",
            "parameter_sensitivity_grid.csv",
            "scenario_runs_metadata.json",
            "interactive_spread_scenario_simulator.html",
        ]
        for fname in required_data_files:
            self.assertTrue((SCEN_DATA / fname).exists(), f"Missing {SCEN_DATA / fname}")
            self.assertTrue((SCEN_REP / fname).exists(), f"Missing {SCEN_REP / fname}")

        required_figs = [
            "01_maharashtra_district_adjacency_graph.png",
            "02_propagation_sequence_snapshots.png",
            "03_scenario_timeline_and_affected_sequence.png",
            "04_before_after_intervention_comparison.png",
            "05_parameter_sensitivity_and_assumptions_panel.png",
        ]
        for fig_name in required_figs:
            p_rep = SCEN_REP / "figures" / fig_name
            p_dat = SCEN_DATA / "figures" / fig_name
            self.assertTrue(p_rep.exists() and p_rep.stat().st_size > 15_000)
            self.assertTrue(p_dat.exists() and p_dat.stat().st_size > 15_000)

        # Verify interactive HTML contains required interactive panels
        html_txt = (SCEN_REP / "interactive_spread_scenario_simulator.html").read_text(encoding="utf-8")
        self.assertIn("Parameter Adjustment Panel", html_txt)
        self.assertIn("Assumptions & Non-Prediction Panel", html_txt)
        self.assertIn("Before vs. After Intervention", html_txt)
        self.assertIn("HYPOTHETICAL SCENARIO SIMULATION ONLY", html_txt)

        # Test DistrictSpreadScenarioService
        svc = DistrictSpreadScenarioService(PROJ)
        topo = svc.get_graph_topology()
        self.assertEqual(topo["total_nodes"], 36)
        self.assertEqual(topo["undirected_edges"], 83)

        custom = svc.simulate_custom_scenario(
            initial_affected_districts={"Kolhapur": 80.0},
            scenario_duration_steps=8,
            hypothetical_propagation_factor=0.35,
            intervention_start_step=3,
            intervention_response_coverage=0.55,
        )
        self.assertEqual(custom["status"], "OK")
        self.assertGreater(custom["final_step_signal_reduction_pct"], 20.0)


if __name__ == "__main__":
    unittest.main()
