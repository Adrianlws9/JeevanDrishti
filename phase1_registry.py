"""Builds model_dataset_registry.json from the Phase 1 report CSVs (read-only on sources)."""
import json, pandas as pd
P = r"C:\Users\Janhvi Suryavanshi\.gemini\antigravity\scratch\JEEVAN-DRISHTI"
R = P + r"\data\reports"
inv = pd.read_csv(R + r"\dataset_inventory.csv"); cl = pd.read_csv(R + r"\climate_coverage_report.csv")
dc = pd.read_csv(R + r"\disease_coverage_report.csv"); comp = pd.read_csv(R + r"\dataset_compatibility_report.csv")
def n(x): return None if pd.isna(x) else (int(x) if float(x).is_integer() else float(x))
def row(fn): return inv[inv.filename.str.startswith(fn)].iloc[0]
def base(fn, **kw):
    r = row(fn); d = dict(filename=fn, path=r.path, size_bytes=int(r.size_bytes), file_type=r.file_type,
                          source_identifier=r.original_source, rows=n(r.rows), columns=n(r.columns)); d.update(kw); return d
disease = {}
for _, r in dc.iterrows():
    disease.setdefault(r.disease, {})[("historical_2009_2022" if "epiclim" in r.dataset or r.dataset == "hist" else "contemporary_2023_2025")] = dict(
        records=n(r.outbreak_records), cases=n(r.reported_cases), deaths=n(r.reported_deaths), districts=n(r.districts_with_records), first=None if pd.isna(r.first_date) else r.first_date, last=None if pd.isna(r.last_date) else r.last_date)
reg = dict(
  registry_version="phase1-1.0", project="JEEVAN-DRISHTI", generated="2026-10-31", note="Facts measured from files; no cleaning, no imputation. Nothing here has been modified.",
  datasets=[
    base("maharashtra_epiclim_filtered.csv", role="disease_historical", date_start="2009-07-23", date_end="2022-06-15", frequency="event-based (outbreak_date)", districts_covered=35, districts_missing=["Mumbai Suburban"], usable_with_climate_files=False, usable_note="No date overlap with any IMD/CWPRS file"),
    base("maharashtra_weekly_outbreak_records_2023_2025.csv (user upload, raw)", role="disease_contemporary_raw_original", date_start="2023-W01", date_end="2025-W52", frequency="weekly report", note="5 rows have unquoted multi-word district names (shifted columns)"),
    base("maharashtra_outbreaks_2023_2025.csv (project copy)", role="disease_contemporary", date_start="2023-01-01(report period)", date_end="2025-12-28(report period)", frequency="weekly report; only 39/52 weeks in 2023, 52/52 in 2024, 30/52 in 2025 present", districts_covered=34, districts_missing=["Mumbai City", "Ratnagiri"], usable_with_climate_files=True, note="Derived copy, quoting fix only"),
    base("imdweb_5dgbAskl.nc", role="climate_rainfall", variable="rain", units="mm/day", date_start="2023-01-01", date_end="2025-12-31", frequency="daily", grid="0.25 deg, 30 lat x 31 lon", extent=dict(lat=[15.75, 23.0], lon=[72.75, 80.25]), nan_share_note="38% NaN (outside India/ocean)"),
    base("imdweb_Oe9H04Bg.nc", role="climate_tmin", variable="tmin", units="C", date_start="2023-01-01", date_end="2025-12-31", frequency="daily", grid="1.0 deg, 9 x 9", extent=dict(lat=[15.5, 23.5], lon=[72.5, 80.5])),
    base("imdweb_tXexQbB6.nc", role="climate_tmax", variable="tmax", units="C", date_start="2023-01-01", date_end="2025-12-31", frequency="daily", grid="1.0 deg, 9 x 9", extent=dict(lat=[15.5, 23.5], lon=[72.5, 80.5])),
    base("humid_tel_hr_cwprs_mh_2021_2025.csv", role="climate_relative_humidity", units="%", date_start="2024-05-11", date_end="2025-06-09", frequency="5-minute telemetry with gaps", spatial="single station, Pune (CWPRS Khadakvasla)", note="Filename says 2021_2025 but data is 2024-05..2025-06; 1,875 RH values outside 0-100"),
    base("sevadataset.zip", role="raw_source_pdfs_unparsed"),
    base("week1_2023.pdf", role="reference_pdf"),
  ],
  not_source_data=["district_data_availability.csv", "disease_data_availability.csv", "dataset_audit_report.md (earlier audit; contains figures that disagree with measured values)"],
  variables_not_available=["Heat-related illness (no records in either disease file)", "Average temperature (no dataset)", "Gridded relative humidity (single Pune station only)", "Leptospirosis in 2009-2022 file"],
  disease_coverage=disease,
  compatibility=comp.fillna("").to_dict(orient="records"),
)
json.dump(reg, open(P + r"\model_dataset_registry.json", "w", encoding="utf-8"), indent=2, default=str)
print("ok", len(reg["datasets"]))
