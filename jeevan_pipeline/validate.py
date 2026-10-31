"""Reporting-frequency validation, district/date compatibility checks and missing-data reporting."""
import re
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from . import config as C

SEVA_ZIP = Path(r"C:\Users\Janhvi Suryavanshi\Downloads\sevadataset.zip")   # Phase-1 inventoried raw PDF archive (read-only listing)


def _pdf_weeks():
    """{year: set(week)} of weekly PDFs present in sevadataset.zip (names only; the archive is not extracted)."""
    if not SEVA_ZIP.exists():
        return None
    out = {}
    with zipfile.ZipFile(SEVA_ZIP) as z:
        for n in z.namelist():
            m = re.search(r"sevadataset/(?:(\d{4})/)?week(\d+)", n)
            if m:
                out.setdefault(int(m.group(1) or 2023), set()).add(int(m.group(2)))   # un-foldered PDFs carry 2023 rows in the CSV
    return out


def frequency_validation(disease, climate_stats, anchors):
    rows = []
    def add(ds, check, expected, observed, status, detail=""):
        rows.append(dict(dataset=ds, check=check, expected=expected, observed=observed, status=status, detail=detail))
    i = disease[disease.source_dataset == "idsp_weekly"]
    pdfs = _pdf_weeks()
    for y in sorted(i.report_year.dropna().unique()):
        y = int(y)
        present = set(i[i.report_year == y].report_week.dropna().astype(int))
        missing = sorted(set(range(1, 53)) - present)
        det = f"weeks without any outbreak row: {missing}"
        if pdfs is not None:
            no_pdf = sorted(set(missing) - pdfs.get(y, set()))
            det += f"; of these, weeks with NO source PDF in sevadataset.zip: {no_pdf}; weeks with a PDF but no rows: {sorted(set(missing) & pdfs.get(y, set()))}"
        add("idsp_weekly", f"weeks_present_{y}", 52, len(present), "OK" if not missing else "ATTENTION",
            det + ". A week with no rows cannot be told apart from a missing report, so it is never zero-filled.")
    g = i.groupby(["report_year", "report_week"]).report_period_start.nunique()
    add("idsp_weekly", "one_period_per_year_week", "1", f"{int((g > 1).sum())} (year,week) with >1 period", "OK" if (g > 1).sum() == 0 else "ATTENTION")
    ln = (i.report_period_end - i.report_period_start).dt.days
    add("idsp_weekly", "period_length_days", 6, f"min={ln.min()} max={ln.max()}", "OK" if ln.nunique() == 1 and ln.iloc[0] == 6 else "ATTENTION",
        "report_period_end - report_period_start; short periods in source (kept as given, flagged): " + str(sorted(set(map(tuple, i[ln < 6][["report_year", "report_week"]].astype(int).values.tolist())))))
    add("idsp_weekly", "report_period_source", "source", str(i.report_period_source.value_counts().to_dict()), "ATTENTION" if (i.report_period_source != "source").any() else "OK",
        f"week-1 anchors verified per year: {climate_stats.get('anchors_text', anchors)}")
    h = disease[disease.source_dataset == "historical_epiclim"].sort_values("event_date")
    gaps = h.event_date.diff().dt.days.dropna()
    add("historical_epiclim", "reporting_frequency", "regular interval", f"event-based; median gap {gaps.median():.0f} d, max {gaps.max():.0f} d", "INFO",
        "No fixed reporting frequency; events per year: " + str(h.groupby(h.event_date.dt.year).size().to_dict()))
    for v in ("rain", "tmin", "tmax"):
        add(f"imd_{v}", "daily_continuity", C_days(climate_stats), f"{climate_stats['n_dates']} days, shape={climate_stats[v]['shape']}", "OK",
            f"{climate_stats['date_min']}..{climate_stats['date_max']}; NaN grid values (ocean mask) {climate_stats[v]['n_nan']}")
    r = climate_stats["rh"]
    add("humidity_cwprs", "native_frequency", "5 min", str(r["step_minutes"]), "ATTENTION", f"{r['days_with_data']}/{r['span_days']} calendar days have data; {r['days_low_coverage']} days below 50% coverage")
    add("humidity_cwprs", "spatial_coverage", "36 districts", "1 station (Pune)", "ATTENTION", f"{r['first']} .. {r['last']}")
    add("cross", "disease(weekly)_vs_climate(daily)", "same resolution", "weekly vs daily", "AGGREGATE_CLIMATE",
        "Climate must be aggregated to the report week (Mon-Sun) per district; weekly disease data cannot be disaggregated")
    add("cross", "historical_vs_climate", "overlap", "0 days", "INCOMPATIBLE", "historical events end 2022-06-15; climate starts 2023-01-01")
    return pd.DataFrame(rows)


def C_days(s):
    return s["n_dates"]


def annotate_and_check_compatibility(disease, climate, cell, climate_stats):
    cmin, cmax = pd.Timestamp(climate_stats["date_min"]), pd.Timestamp(climate_stats["date_max"])
    rh = climate_stats["rh"]
    rhmin, rhmax = pd.Timestamp(rh["first"]).normalize(), pd.Timestamp(rh["last"]).normalize()
    s = disease.join_date_start.fillna(disease.event_date)
    e = disease.join_date_end.fillna(disease.event_date)
    ov = (e.clip(upper=cmax) - s.clip(lower=cmin)).dt.days + 1
    disease["climate_window_overlap_days"] = ov.clip(lower=0).fillna(0).astype(int)
    c = cell.set_index("district_std")
    disease["district_has_coordinates"] = disease.district_std.map(c["lat"].notna())
    disease["climate_rain_available"] = disease.district_std.map(c["rain_valid_days"] > 0).fillna(False) & (disease.climate_window_overlap_days > 0)
    disease["climate_temp_available"] = disease.district_std.map((c["tmin_valid_days"] > 0) & (c["tmax_valid_days"] > 0)).fillna(False) & (disease.climate_window_overlap_days > 0)
    rov = (e.clip(upper=rhmax) - s.clip(lower=rhmin)).dt.days + 1
    disease["climate_rh_available"] = (disease.district_std == rh["district"]) & (rov > 0)
    disease["climate_joinable"] = disease.climate_rain_available | disease.climate_temp_available
    rows = []
    for d in C.OFFICIAL_DISTRICTS:
        x = disease[disease.district_std == d]
        ci = c.loc[d]
        issues = []
        if pd.isna(ci.get("lat")):
            issues.append("no coordinates -> no climate extraction")
        for v in ("rain", "tmin", "tmax"):
            if ci.get(f"{v}_valid_days", 0) == 0 and not pd.isna(ci.get("lat")):
                issues.append(f"nearest {v} grid cell is all-NaN")
        if (x.source_dataset == "historical_epiclim").sum() == 0:
            issues.append("no historical records")
        if (x.source_dataset == "idsp_weekly").sum() == 0:
            issues.append("no 2023-25 records")
        rows.append(dict(
            district=d, historical_records=int((x.source_dataset == "historical_epiclim").sum()),
            idsp_records=int((x.source_dataset == "idsp_weekly").sum()),
            has_coordinates=bool(pd.notna(ci.get("lat"))), rain_valid_days=int(ci.get("rain_valid_days", 0)),
            tmin_valid_days=int(ci.get("tmin_valid_days", 0)), tmax_valid_days=int(ci.get("tmax_valid_days", 0)),
            idsp_records_in_climate_window=int(((x.source_dataset == "idsp_weekly") & (x.climate_window_overlap_days > 0)).sum()),
            records_rain_joinable=int(x.climate_rain_available.sum()), records_temp_joinable=int(x.climate_temp_available.sum()),
            records_rh_joinable=int(x.climate_rh_available.sum()), issues="; ".join(issues)))
    return disease, pd.DataFrame(rows)


SEMANTICS = {
    "deaths": "Deaths not reported in source. Kept as NaN; NOT zero.",
    "Deaths": "Deaths not reported in source. Kept as NaN; NOT zero.",
    "event_date": "Outbreak start date not given in the IDSP report. Report period is used for joins instead.",
    "outbreak_start_date": "Outbreak start date not given in the IDSP report.",
    "last_reported_date": "Only populated for continuing outbreaks.",
    "status": "Only populated for a few outbreaks (Under Control / Under Surveillance).",
    "hist_LAI": "Embedded leaf-area-index covariate absent in source (undocumented origin).",
    "hist_Temp_C": "Embedded temperature absent in source.",
    "hist_preci": "Embedded precipitation absent in source.",
    "LAI": "Embedded leaf-area-index covariate absent in source.",
    "Temp": "Embedded temperature absent in source.",
    "Temp_C": "Embedded temperature absent in source.",
    "report_period_start": "Missing in source for some 2024 weeks; derived only where the week convention was verified.",
    "report_period_end": "Missing in source for some 2024 weeks; derived only where the week convention was verified.",
    "rain_mm": "NaN where district has no coordinates or nearest IMD cell is outside the India land mask.",
    "tmin_c": "NaN where district has no coordinates or nearest IMD cell is outside the India land mask.",
    "tmax_c": "NaN where district has no coordinates or nearest IMD cell is outside the India land mask.",
    "tavg_derived_c": "Derived (tmin+tmax)/2; NaN if either input is NaN.",
    "rh_mean": "Humidity exists only for Pune station, 2024-05-11..2025-06-09; elsewhere not available (NOT imputed).",
    "lat": "Coordinates only available from historical file; Mumbai Suburban has none.",
}


def missing_report(frames):
    rows = []
    for name, df in frames.items():
        n = len(df)
        for col in df.columns:
            m = int(df[col].isna().sum())
            if str(col).startswith("Unnamed"):
                continue
            rows.append(dict(dataset=name, column=col, n_rows=n, n_missing=m, pct_missing=round(100 * m / max(n, 1), 2),
                             meaning_and_treatment=SEMANTICS.get(col, "" if m == 0 else "Left as NaN; no imputation.")))
    return pd.DataFrame(rows)


