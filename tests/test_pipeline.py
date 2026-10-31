"""Automated validation tests for the Phase 2 pipeline.  Run:  python -m unittest discover -s tests -v

Unit tests cover the pure standardization helpers; integration tests run the pipeline and assert the invariants that
must hold for the processed data (no fabrication, no zero-filling, nothing silently dropped, raw files untouched).
"""
import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jeevan_pipeline import config as C                      # noqa: E402
from jeevan_pipeline import standardize as S                 # noqa: E402
from jeevan_pipeline import run_pipeline                     # noqa: E402


class TestStandardizationUnits(unittest.TestCase):
    def test_ordinal_dates(self):
        d = S.parse_ordinal_date(pd.Series(["2nd January 2023", "23rd March 2024", None, "garbage"]))
        self.assertEqual(d[0], pd.Timestamp("2023-01-02"))
        self.assertEqual(d[1], pd.Timestamp("2024-03-23"))
        self.assertTrue(pd.isna(d[2]) and pd.isna(d[3]))        # never guessed

    def test_week_number(self):
        self.assertEqual(int(S.parse_week_number(pd.Series(["45th week"]))[0]), 45)

    def test_district_aliases(self):
        cases = {"Aurangabad": "Chhatrapati Sambhajinagar", "Osmanabad (Maharashtra)": "Dharashiv", "Nasik": "Nashik",
                 "Sholapur": "Solapur", "Palaghar": "Palghar", "Mumbai (city)": "Mumbai City", "pune ": "Pune"}
        for raw, want in cases.items():
            self.assertEqual(S.standardize_district(raw)[0], want, raw)
        self.assertIsNone(S.standardize_district("Mumbai")[0])   # ambiguous -> must not be guessed
        self.assertIsNone(S.standardize_district("Atlantis")[0])

    def test_split_district_repair(self):
        self.assertEqual(S.repair_split_district("Mumbai", "Suburban Zika Virus"), ("Mumbai Suburban", "Zika Virus", True))
        self.assertEqual(S.repair_split_district("Chhatrapati", "Sambhajinaga r Food Poisoning"),
                         ("Chhatrapati Sambhajinagar", "Food Poisoning", True))
        self.assertEqual(S.repair_split_district("Pune", "Dengue"), ("Pune", "Dengue", False))
        self.assertFalse(S.repair_split_district("Mumbai", "Zika Virus")[2])

    def test_disease_labels(self):
        want = {"Acute Diarrheal Disease": "Acute Diarrhoeal Disease", "Acute Gastroenterit is": "Acute Diarrhoeal Disease",
                "Dengue Fever": "Dengue", "Suspected Dengue": "Dengue", "Dengue/Chikungunya": "Dengue + Chikungunya",
                "Chikungunya & Dengue": "Dengue + Chikungunya", "Dengue & Malaria (P.vivax)": "Dengue + Malaria",
                "Mixed Malaria (Plasmodiu m falciparum)": "Malaria", "Malaria (PV)": "Malaria", "Food Poisoning": "Food Poisoning",
                "Leptospirosis": "Leptospirosis"}
        for raw, std in want.items():
            self.assertEqual(S.standardize_disease(raw)[0], std, raw)
        self.assertTrue(S.standardize_disease("Suspected Dengue")[1])
        self.assertFalse(S.standardize_disease("Totally Unknown Thing")[2])   # reported as unmapped, not forced

    def test_official_list(self):
        self.assertEqual(len(set(C.OFFICIAL_DISTRICTS)), 36)

    def test_add_flag(self):
        df = pd.DataFrame({"flags": ["", "a"]})
        S.add_flag(df, pd.Series([True, True]), "b")
        self.assertEqual(df["flags"].tolist(), ["b", "a;b"])


class TestPipelineOutputs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        run_pipeline.main()
        cls.d = pd.read_csv(C.PROCESSED / "cleaned_disease_data.csv", parse_dates=["event_date", "report_period_start", "report_period_end"])
        cls.c = pd.read_csv(C.PROCESSED / "cleaned_climate_data.csv", parse_dates=["date"])
        cls.rej = pd.read_csv(C.REPORTS / "rejected_records.csv")
        cls.d["flags"] = cls.d["flags"].fillna("")          # empty string round-trips through CSV as NaN
        cls.c["flags"] = cls.c["flags"].fillna("")
        cls.log = pd.read_csv(C.REPORTS / "preprocessing_log.csv")
        cls.hist_raw = pd.read_csv(C.HIST_RAW)
        cls.idsp_raw = pd.read_csv(C.IDSP_RAW, encoding="utf-8-sig", dtype=str)

    # ---- raw data protection
    def test_raw_files_unchanged(self):
        man = pd.read_csv(C.RAW_MANIFEST)
        for _, r in man.iterrows():
            h = hashlib.sha256(Path(r.raw_copy).read_bytes()).hexdigest()
            self.assertEqual(h, r.sha256_copy, r.raw_copy)

    # ---- reconciliation: nothing silently lost, nothing invented
    def test_row_reconciliation(self):
        for ds, raw in (("historical_epiclim", self.hist_raw), ("idsp_weekly", self.idsp_raw)):
            kept = int((self.d.source_dataset == ds).sum())
            rej = int((self.rej.source_dataset == ds).sum())
            self.assertEqual(kept + rej, len(raw), ds)

    def test_no_fabricated_rows(self):
        for ds, raw in (("historical_epiclim", self.hist_raw), ("idsp_weekly", self.idsp_raw)):
            rows = self.d[self.d.source_dataset == ds].source_row
            self.assertTrue(rows.isin(raw.index).all())
            self.assertTrue(rows.is_unique)

    def test_record_ids_unique(self):
        self.assertTrue(self.d.record_id.is_unique)

    def test_rejected_have_reason_and_original(self):
        self.assertFalse(self.rej.reject_reason.isna().any())
        self.assertFalse(self.rej.original_record.isna().any())
        dups = self.rej[self.rej.reject_reason == "exact_duplicate"]
        self.assertTrue(dups.kept_record_id.isin(self.d.record_id).all())

    # ---- disease table invariants
    def test_cases_deaths_valid(self):
        self.assertFalse(self.d.cases.isna().any())
        self.assertTrue((self.d.cases > 0).all())
        self.assertTrue(((self.d.deaths.isna()) | ((self.d.deaths >= 0) & (self.d.deaths <= self.d.cases))).all())

    def test_missing_deaths_not_zero(self):
        h = self.d[self.d.source_dataset == "historical_epiclim"]
        raw_missing = self.hist_raw.Deaths.isna().sum()
        rejected_missing = self.hist_raw.loc[self.rej[self.rej.source_dataset == "historical_epiclim"].source_row].Deaths.isna().sum()
        self.assertEqual(int(h.deaths.isna().sum()), int(raw_missing - rejected_missing))
        self.assertTrue(h.loc[h.deaths.isna(), "flags"].str.contains("deaths_missing_not_zero").all())

    def test_districts_standard(self):
        self.assertTrue(self.d.district_std.isin(C.OFFICIAL_DISTRICTS).all())

    def test_source_values_preserved(self):
        i = self.d[self.d.source_dataset == "idsp_weekly"]
        orig = self.idsp_raw.loc[i.source_row, "district"].values
        self.assertTrue((i.district_source.values == orig).all())
        rep = i[i["flags"].str.contains("structural_repair_split_district")]
        self.assertEqual(len(rep), 5)
        self.assertIn(rep.district_std.iloc[0], ("Mumbai Suburban", "Chhatrapati Sambhajinagar"))

    def test_disease_labels_complete(self):
        self.assertFalse(self.d.disease_std.isna().any())
        self.assertTrue(self.d.disease_mapped.all())
        for t in ("Dengue", "Malaria", "Chikungunya", "Acute Diarrhoeal Disease", "Cholera", "Leptospirosis"):
            self.assertGreater(int((self.d.disease_std == t).sum()), 0, t)
        self.assertEqual(int((self.d.disease_std == "Heat-related illness").sum()), 0)   # absent in source, must not be created

    def test_no_exact_duplicates_remain_and_conflicts_flagged(self):
        cols = ["source_dataset", "district_source", "disease_source", "event_date", "cases", "deaths", "outbreak_id", "report_year",
                "report_week", "page", "status"]
        self.assertEqual(int(self.d.duplicated(cols).sum()), 0)
        grp = self.d[self.d.duplicate_status != "unique"]
        self.assertTrue(grp.dup_group_id.notna().all())
        self.assertTrue(grp["flags"].str.contains("duplicate_group").all())
        self.assertTrue((self.d.duplicate_status == "conflict").any())

    def test_derived_periods_are_labelled(self):
        der = self.d[self.d.report_period_source == "derived_week_convention"]
        self.assertTrue(der["flags"].str.contains("report_period_derived_week_convention").all())
        i = self.d[self.d.source_dataset == "idsp_weekly"]
        src = i[i.report_period_source == "source"]
        self.assertTrue(((src.report_period_end - src.report_period_start).dt.days <= 6).all())
        self.assertTrue(((der.report_period_end - der.report_period_start).dt.days == 6).all())

    def test_hist_event_date_valid_and_in_range(self):
        h = self.d[self.d.source_dataset == "historical_epiclim"]
        self.assertFalse(h.event_date.isna().any())
        self.assertEqual(str(h.event_date.min().date()), "2009-07-23")
        self.assertEqual(str(h.event_date.max().date()), "2022-06-15")

    # ---- climate table invariants
    def test_climate_shape_and_keys(self):
        self.assertEqual(len(self.c), 36 * 1096)
        self.assertTrue(self.c.duplicated(["date", "district_std"]).sum() == 0)
        self.assertEqual(set(self.c.district_std), set(C.OFFICIAL_DISTRICTS))
        days = self.c.date.drop_duplicates().sort_values()
        self.assertTrue((days.diff().dropna().dt.days == 1).all())
        self.assertEqual(str(days.min().date()), "2023-01-01")
        self.assertEqual(str(days.max().date()), "2025-12-31")

    def test_climate_value_ranges(self):
        self.assertTrue((self.c.rain_mm.dropna() >= 0).all())
        self.assertTrue(self.c.tmin_c.dropna().between(C.TEMP_MIN_C, C.TEMP_MAX_C).all())
        self.assertTrue(self.c.tmax_c.dropna().between(C.TEMP_MIN_C, C.TEMP_MAX_C).all())
        both = self.c.dropna(subset=["tmin_c", "tmax_c"])
        self.assertTrue((both.tmin_c <= both.tmax_c).all())
        self.assertTrue(np.allclose(both.tavg_derived_c, (both.tmin_c + both.tmax_c) / 2))
        self.assertTrue(self.c.tavg_derived_c[self.c.tmin_c.isna() | self.c.tmax_c.isna()].isna().all())

    def test_no_climate_invention(self):
        ms = self.c[self.c.district_std == "Mumbai Suburban"]
        self.assertTrue(ms[["lat", "lon", "rain_mm", "tmin_c", "tmax_c"]].isna().all().all())
        self.assertTrue(ms["flags"].str.contains("no_district_coordinates").all())

    def test_humidity_pune_only_and_valid(self):
        has = self.c[self.c.rh_n_valid.notna()]
        self.assertEqual(set(has.district_std), {"Pune"})
        self.assertTrue(has.rh_mean.dropna().between(0, 100).all())
        self.assertTrue((has.rh_min.dropna() > 0).all())
        self.assertTrue((has.rh_max.dropna() <= 100).all())
        self.assertTrue(self.c.loc[self.c.district_std != "Pune", "rh_mean"].isna().all())
        self.assertTrue(has.date.min() >= pd.Timestamp("2024-05-11") and has.date.max() <= pd.Timestamp("2025-06-09"))
        rh_rej = self.rej[self.rej.source_dataset == "humidity_cwprs"]
        self.assertEqual(len(rh_rej), 1880)

    # ---- compatibility annotations
    def test_climate_join_flags(self):
        h = self.d[self.d.source_dataset == "historical_epiclim"]
        self.assertFalse(h.climate_joinable.any())            # no date overlap with climate files
        i = self.d[self.d.source_dataset == "idsp_weekly"]
        ms = i[i.district_std == "Mumbai Suburban"]
        self.assertFalse(ms.climate_joinable.any())
        self.assertTrue(i.climate_joinable.sum() > 0)

    # ---- logs / reports / registry
    def test_log_and_reports(self):
        steps = set(self.log.step)
        for s in ("integrity_check", "district_standardization", "disease_standardization", "duplicate_resolution",
                  "invalid_record_identification", "climate_validation", "district_extraction", "compatibility_check"):
            self.assertIn(s, steps)
        self.assertEqual(list(self.log.step_id), list(range(1, len(self.log) + 1)))
        for f in ("district_mapping.csv", "missing_data_report.csv", "preprocessing_summary.md"):
            self.assertTrue((C.REPORTS / f).exists(), f)
        mr = pd.read_csv(C.REPORTS / "missing_data_report.csv")
        row = mr[(mr.dataset == "cleaned_disease_data") & (mr.column == "deaths")].iloc[0]
        self.assertIn("NOT zero", row.meaning_and_treatment)

    def test_district_mapping_complete(self):
        m = pd.read_csv(C.REPORTS / "district_mapping.csv")
        self.assertFalse(m.district_std.isna().any())
        self.assertEqual(set(m.district_std), set(C.OFFICIAL_DISTRICTS))

    def test_registry_updated(self):
        reg = json.loads(C.REGISTRY.read_text(encoding="utf-8"))
        names = {p["name"]: p for p in reg["processed_datasets"]}
        for n, f in (("cleaned_disease_data", "cleaned_disease_data.csv"), ("cleaned_climate_data", "cleaned_climate_data.csv")):
            p = names[n]
            self.assertEqual(p["version"], C.DATASET_VERSION)
            self.assertEqual(p["sha256"], hashlib.sha256((C.PROCESSED / f).read_bytes()).hexdigest())
        self.assertEqual(reg["pipeline"]["version"], C.PIPELINE_VERSION)


if __name__ == "__main__":
    unittest.main()

