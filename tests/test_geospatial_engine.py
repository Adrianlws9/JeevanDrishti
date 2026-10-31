"""Automated Validation Suite for JEEVAN-DRISHTI Phase 9 Geospatial Hotspot Detection Engine.

Tests all 6 required validation areas from Prompt 9 Section 7:
1. Geographic joins & 36-district standardization (including aliases & historical names).
2. Missing geometries (`Mumbai Suburban` has no coordinates; zero polygon fabrication; `"Insufficient Data"` markings).
3. Duplicate district mappings (zero duplicate district IDs or names).
4. Correct aggregation & strict separation of observed, historical, forecast, anomaly, and risk layers.
5. Spatial statistical calculations (row-standardized W rows sum to 1.0, Global Moran's I E[I] = -1/(N-1),
   Local Moran's I and Getis-Ord Gi* strict p < 0.05 significance gating).
6. Map rendering (all 6 required PNG figures and GeoJSON/drilldown outputs exist and are valid).
"""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.engines.geospatial.service import GeospatialHotspotService

PROJ = Path(__file__).resolve().parents[1]
GEO_DATA = PROJ / "data" / "geospatial"
GEO_REP = PROJ / "reports" / "geospatial"


class TestPhase9GeospatialHotspotEngine(unittest.TestCase):
    """Comprehensive verification suite for Phase 9 Geospatial Disease Hotspot Detection Engine."""

    def test_01_geographic_joins_and_zero_duplicate_mappings(self):
        """Verify all 36 official Maharashtra districts map uniquely with documented aliases and zero duplicates."""
        cw_path = GEO_DATA / "district_geographic_crosswalk.csv"
        self.assertTrue(cw_path.exists(), "district_geographic_crosswalk.csv missing")
        cw = pd.read_csv(cw_path)

        self.assertEqual(len(cw), 36)
        self.assertEqual(cw["district_id"].nunique(), 36)
        self.assertEqual(cw["district_name"].nunique(), 36)

        # Verify historical renamings and boundary notes are documented
        csn = cw[cw["district_name"] == "Chhatrapati Sambhajinagar"].iloc[0]
        self.assertIn("Aurangabad", csn["documented_aliases"])

        dha = cw[cw["district_name"] == "Dharashiv"].iloc[0]
        self.assertIn("Osmanabad", dha["documented_aliases"])

        ahm = cw[cw["district_name"] == "Ahmednagar"].iloc[0]
        self.assertIn("Ahilyanagar", ahm["documented_aliases"])

        pal = cw[cw["district_name"] == "Palghar"].iloc[0]
        self.assertIn("2014-08-01", pal["boundary_and_historical_note"])

    def test_02_missing_geometries_and_insufficient_data_handling(self):
        """Verify no polygon boundaries or population denominators are fabricated and Insufficient Data is marked."""
        cw = pd.read_csv(GEO_DATA / "district_geographic_crosswalk.csv")
        self.assertFalse(cw["polygon_geometry_available"].any())
        self.assertTrue(cw["population_denominator"].isna().all())

        # Exactly 35 districts have verified point coordinates; Mumbai Suburban has NaN coordinates
        self.assertEqual(int(cw["has_verified_coordinates"].sum()), 35)
        ms = cw[cw["district_name"] == "Mumbai Suburban"].iloc[0]
        self.assertFalse(bool(ms["has_verified_coordinates"]))
        self.assertTrue(pd.isna(ms["latitude"]))

        # Verify GeoJSON Point FeatureCollection has 35 Point geometries and 1 null geometry
        with open(GEO_DATA / "maharashtra_district_points.geojson", "r", encoding="utf-8") as f:
            gj = json.load(f)
        self.assertEqual(len(gj["features"]), 36)
        point_feats = [ft for ft in gj["features"] if ft["geometry"] is not None]
        null_feats = [ft for ft in gj["features"] if ft["geometry"] is None]
        self.assertEqual(len(point_feats), 35)
        self.assertEqual(len(null_feats), 1)
        self.assertEqual(null_feats[0]["properties"]["district_name"], "Mumbai Suburban")

        # Verify Mumbai City, Ratnagiri, and Mumbai Suburban are marked Insufficient Data in summary
        sum_df = pd.read_csv(GEO_DATA / "district_disease_map_summary.csv")
        pool_all = sum_df[
            (sum_df["disease"] == "ALL_DISEASES")
            & (sum_df["reporting_period"] == "2023-2025_POOL")
        ].set_index("district_name")
        for d_insuff in ["Mumbai Suburban", "Mumbai City", "Ratnagiri"]:
            self.assertTrue(
                str(pool_all.loc[d_insuff, "data_quality_status"]).startswith("Insufficient Data"),
                f"{d_insuff} must be marked Insufficient Data",
            )
            self.assertEqual(pool_all.loc[d_insuff, "gi_star_class"], "Insufficient Data")

    def test_03_aggregation_and_strict_layer_separation(self):
        """Verify observed counts match source data and are strictly separated from forecasts, anomalies, and risk."""
        wk_map = pd.read_csv(GEO_DATA / "district_period_map_records.csv")
        sum_df = pd.read_csv(GEO_DATA / "district_disease_map_summary.csv")

        # Active weekly records = 511 + 2 sentinel Insufficient Data records = 513
        active_wk = wk_map[wk_map["observed_cases"].notna()]
        self.assertEqual(len(active_wk), 511)

        # Incidence rate must be strictly NaN across all records
        self.assertTrue(wk_map["incidence_rate"].isna().all())
        self.assertTrue(sum_df["incidence_rate"].isna().all())

        # Sum of observed_cases in 2023-2025_POOL ALL_DISEASES must equal sum of weekly observed_cases
        pool_all = sum_df[
            (sum_df["disease"] == "ALL_DISEASES")
            & (sum_df["reporting_period"] == "2023-2025_POOL")
        ]
        self.assertAlmostEqual(
            float(pool_all["observed_cases"].sum()),
            float(active_wk["observed_cases"].sum()),
            places=2,
        )

        # Verify forecast_value is distinct from observed_cases
        both_valid = active_wk[active_wk["forecast_value"].notna()]
        self.assertGreater(len(both_valid), 500)
        self.assertFalse(
            np.allclose(both_valid["observed_cases"], both_valid["forecast_value"]),
            "Forecast values must never be identical to or mixed with observed cases",
        )

    def test_04_spatial_weights_and_statistical_hotspot_calculations(self):
        """Verify spatial weights matrix W_knn4 properties, Global Moran's I E[I], and strict p<0.05 hotspot gating."""
        w_df = pd.read_csv(GEO_DATA / "spatial_weights_matrix.csv")
        self.assertEqual(len(w_df), 36)
        geo_w = w_df[w_df["latitude"].notna()]
        self.assertEqual(len(geo_w), 35)
        self.assertTrue((geo_w["degree_symmetric_knn"] >= 4).all())
        self.assertTrue(np.allclose(geo_w["row_weight_sum"], 1.0, atol=1e-4))

        gm_df = pd.read_csv(GEO_DATA / "global_spatial_autocorrelation_summary.csv")
        self.assertGreaterEqual(len(gm_df), 15)
        # E[I] for N=35 geocoded districts is -1/34 = -0.0294
        self.assertTrue(np.allclose(gm_df["expected_i"], -1.0 / 34.0, atol=1e-3))

        # Strict Hotspot Gating Check: Every row with is_statistically_significant_hotspot == True
        # MUST have either (gi_star_zscore >= 1.96 and gi_star_p_perm < 0.05) OR (lisa_quadrant == 'HH_HOTSPOT' and local_moran_p_perm < 0.05)
        sum_df = pd.read_csv(GEO_DATA / "district_disease_map_summary.csv")
        hot_rows = sum_df[sum_df["is_statistically_significant_hotspot"] == True]
        self.assertGreater(len(hot_rows), 0)
        for _, r in hot_rows.iterrows():
            gi_ok = (float(r["gi_star_zscore"]) >= 1.96) and (float(r["gi_star_p_perm"]) < 0.05)
            lisa_ok = (str(r["lisa_quadrant"]) == "HH_HOTSPOT") and (float(r["local_moran_p_perm"]) < 0.05)
            self.assertTrue(
                gi_ok or lisa_ok,
                f"District {r['district_name']} ({r['disease']}) marked as hotspot without p<0.05 support!",
            )
            self.assertGreater(float(r["observed_cases"]), 0.0)

    def test_05_map_rendering_and_service_queries(self):
        """Verify all 6 required map visualizations exist and GeospatialHotspotService works."""
        expected_figs = [
            "01_maharashtra_district_disease_burden_map.png",
            "02_disease_specific_hotspot_maps.png",
            "03_historical_comparison_maps.png",
            "04_forecast_disease_maps.png",
            "05_anomaly_and_risk_intelligence_maps.png",
            "06_district_drilldown_spatial_profiles.png",
        ]
        for fname in expected_figs:
            p_rep = GEO_REP / "figures" / fname
            p_dat = GEO_DATA / "figures" / fname
            self.assertTrue(p_rep.exists() and p_rep.stat().st_size > 15_000, f"Missing figure {p_rep}")
            self.assertTrue(p_dat.exists() and p_dat.stat().st_size > 15_000, f"Missing figure {p_dat}")

        svc = GeospatialHotspotService()
        dengue_map = svc.get_district_map_layer("Dengue", "2023-2025_POOL")
        self.assertEqual(dengue_map["status"], "OK")
        self.assertEqual(dengue_map["total_districts"], 36)

        # Verify alias resolution in drill-down service ('Aurangabad' -> 'Chhatrapati Sambhajinagar')
        dd = svc.get_district_drilldown("Aurangabad")
        self.assertEqual(dd["status"], "OK")
        self.assertEqual(dd["district_name"], "Chhatrapati Sambhajinagar")
        self.assertEqual(dd["mapping_type"], "alias")

        # Verify optional polygon adapter gracefully reports unavailable when no external file exists
        poly_res = svc.get_geojson_payload(external_polygon_geojson_path=PROJ / "nonexistent.geojson")
        self.assertEqual(poly_res["status"], "POLYGON_BOUNDARY_FILE_UNAVAILABLE")
        self.assertFalse(poly_res["has_polygons"])


if __name__ == "__main__":
    unittest.main()
