"""JEEVAN-DRISHTI Phase 1 inspection (read-only on all source data). Writes only to data/reports/."""
import os, re, zipfile, warnings
import numpy as np, pandas as pd, netCDF4 as nc
warnings.filterwarnings("ignore")

PROJ = r"C:\Users\Janhvi Suryavanshi\.gemini\antigravity\scratch\JEEVAN-DRISHTI"
DL = r"C:\Users\Janhvi Suryavanshi\Downloads"
UP = r"C:\Users\Janhvi Suryavanshi\.gemini\antigravity\brain\77ab7f37-6ad5-4925-b8cc-b64abd77919f\.user_uploaded"
OUT = os.path.join(PROJ, "data", "reports")
os.makedirs(OUT, exist_ok=True)

F = {
    "hist": DL + r"\maharashtra_epiclim_filtered.csv",
    "cont_raw": UP + r"\media_1793432423584_ffb33b26.csv",
    "cont_clean": PROJ + r"\maharashtra_outbreaks_2023_2025.csv",
    "rain": DL + r"\imdweb_5dgbAskl.nc",
    "tmin": DL + r"\imdweb_Oe9H04Bg.nc",
    "tmax": DL + r"\imdweb_tXexQbB6.nc",
    "rh": DL + r"\humid_tel_hr_cwprs_mh_2021_2025.csv",
    "pdf": PROJ + r"\sample_reports\week1_2023.pdf",
    "d_dist": PROJ + r"\district_data_availability.csv",
    "d_dis": PROJ + r"\disease_data_availability.csv",
}

OFFICIAL = ['Ahmednagar','Akola','Amravati','Beed','Bhandara','Buldhana','Chandrapur','Chhatrapati Sambhajinagar',
 'Dharashiv','Dhule','Gadchiroli','Gondia','Hingoli','Jalgaon','Jalna','Kolhapur','Latur','Mumbai City','Mumbai Suburban',
 'Nagpur','Nanded','Nandurbar','Nashik','Palghar','Parbhani','Pune','Raigad','Ratnagiri','Sangli','Satara','Sindhudurg',
 'Solapur','Thane','Wardha','Washim','Yavatmal']
assert len(OFFICIAL) == 36
ALIAS = {'aurangabad':'Chhatrapati Sambhajinagar','chhatrapati sambhajinagar':'Chhatrapati Sambhajinagar',
 'osmanabad':'Dharashiv','osmanabad (maharashtra)':'Dharashiv','dharashiv':'Dharashiv',
 'nasik':'Nashik','sholapur':'Solapur','palaghar':'Palghar','ahmadnagar':'Ahmednagar','ahilyanagar':'Ahmednagar',
 'mumbai (city)':'Mumbai City','mumbai city':'Mumbai City','mumbai suburban':'Mumbai Suburban','bid':'Beed',
 'buldana':'Buldhana','gondiya':'Gondia'}
def canon(x):
    k = str(x).strip().lower()
    if k in ALIAS: return ALIAS[k]
    for o in OFFICIAL:
        if o.lower() == k: return o
    return None  # unmapped -> flagged, never guessed

DISEASES = {
 'Dengue': r'dengue', 'Malaria': r'malaria', 'Chikungunya': r'chikungunya',
 'Acute Diarrhoeal Disease': r'diarrh|gastroenter',
 'Cholera': r'cholera', 'Leptospirosis': r'leptospir',
 'Heat-related illness': r'heat|sun.?stroke|hyperther',
}
def dis_flag(s, pat): return s.astype(str).str.contains(pat, case=False, regex=True)

def ordinal_date(s):
    return pd.to_datetime(s.astype(str).str.replace(r'(\d+)(st|nd|rd|th)', r'\1', regex=True), errors='coerce')

# ---------------- load ----------------
hist = pd.read_csv(F["hist"]); cont_raw = pd.read_csv(F["cont_raw"]); cont = pd.read_csv(F["cont_clean"])
rh = pd.read_csv(F["rh"])
hist['_d'] = pd.to_datetime(hist.outbreak_date, errors='coerce')
cont['_start'] = pd.to_datetime(cont.outbreak_start_date, errors='coerce')
cont['_ps'] = ordinal_date(cont.report_period_start.fillna('')); cont['_pe'] = ordinal_date(cont.report_period_end.fillna(''))
cont_raw['_ps'] = ordinal_date(cont_raw.report_period_start.fillna(''))
hist['std'] = hist.district.map(canon); cont['std'] = cont.district.map(canon)
rhcol = [c for c in rh.columns if 'Relative Humidity' in c][0]
rh['_t'] = pd.to_datetime(rh['Data Acquisition Time'], format='%d-%m-%Y %H:%M', errors='coerce')

def grid(path, grp, var):
    ds = nc.Dataset(path); g = ds.groups[grp]
    t = nc.num2date(g['time'][:], g['time'].units); a = g[var][:]
    lat = g['lat'][:]; lon = g['lon'][:]
    d = dict(t0=str(t[0])[:10], t1=str(t[-1])[:10], n=len(t), lat=lat, lon=lon,
             units=g[var].units, nan=int(np.ma.count_masked(a)) + int(np.isnan(np.ma.filled(a, np.nan)).sum() - np.ma.count_masked(a)),
             size=a.size, mn=float(np.nanmin(np.ma.filled(a, np.nan))), mx=float(np.nanmax(np.ma.filled(a, np.nan))),
             mean=float(np.nanmean(np.ma.filled(a, np.nan))),
             gaps=int((np.diff(g['time'][:]) != 1).sum()), dtype=str(g[var].dtype))
    return d
G = {'rain': grid(F['rain'], 'rain', 'rain'), 'tmin': grid(F['tmin'], 'tmin', 'tmin'), 'tmax': grid(F['tmax'], 'tmax', 'tmax')}
for k, g in G.items():
    g['res'] = round(float(np.round(np.diff(g['lat']).mean(), 4)), 3)

# ---------------- 1. inventory ----------------
def size(p): return os.path.getsize(p)
inv = []
def add(name, path, src, kind, rows, cols, role, notes):
    inv.append(dict(filename=name, path=path, size_bytes=size(path), file_type=kind, original_source=src, rows=rows, columns=cols, role=role, notes=notes))
add("maharashtra_epiclim_filtered.csv", F['hist'], "EpiClim-derived (IDSP outbreak records w/ ERA-style climate cols); user-supplied", "CSV", len(hist), hist.shape[1], "disease (historical)", "Event-based IDSP outbreak records 2009-2022; embedded preci/LAI/Temp columns of unstated source")
add("maharashtra_weekly_outbreak_records_2023_2025.csv (user upload, raw)", F['cont_raw'], "IDSP weekly outbreak PDFs (sevadataset/weekNN.pdf) parsed to CSV; user-supplied", "CSV", len(cont_raw), cont_raw.shape[1], "disease (contemporary)", "Original upload; unquoted multi-word districts shifted tokens into the disease column in some rows")
add("maharashtra_outbreaks_2023_2025.csv (project copy)", F['cont_clean'], "Derived from the user upload; quoting fix only", "CSV", len(cont), cont.shape[1], "disease (contemporary)", "Same 575 rows; five district names quoted. Derived copy, not an original")
add("imdweb_5dgbAskl.nc", F['rain'], "IMD gridded daily rainfall (IMD web download)", "NetCDF4 (group /rain)", G['rain']['n'], 3, "climate: rainfall", f"{G['rain']['res']} deg grid, {len(G['rain']['lat'])}x{len(G['rain']['lon'])} cells")
add("imdweb_Oe9H04Bg.nc", F['tmin'], "IMD gridded daily minimum temperature", "NetCDF4 (group /tmin)", G['tmin']['n'], 3, "climate: tmin", f"{G['tmin']['res']} deg grid, {len(G['tmin']['lat'])}x{len(G['tmin']['lon'])} cells")
add("imdweb_tXexQbB6.nc", F['tmax'], "IMD gridded daily maximum temperature", "NetCDF4 (group /tmax)", G['tmax']['n'], 3, "climate: tmax", f"{G['tmax']['res']} deg grid, {len(G['tmax']['lat'])}x{len(G['tmax']['lon'])} cells")
add("humid_tel_hr_cwprs_mh_2021_2025.csv", F['rh'], "CWPRS telemetry (India-WRIS style), station CWPRS campus, Pune", "CSV", len(rh), rh.shape[1], "climate: relative humidity", "Single station")
add("week1_2023.pdf", F['pdf'], "IDSP weekly outbreak report (sample)", "PDF (unstructured)", None, None, "reference only", "Not structured; not parsed here")
add("district_data_availability.csv", F['d_dist'], "Generated by this project (earlier audit)", "CSV", len(pd.read_csv(F['d_dist'])), pd.read_csv(F['d_dist']).shape[1], "derived report", "Not source data; excluded from record totals. Contains district-mapping errors, see summary")
add("disease_data_availability.csv", F['d_dis'], "Generated by this project (earlier audit)", "CSV", len(pd.read_csv(F['d_dis'])), pd.read_csv(F['d_dis']).shape[1], "derived report", "Not source data; excluded from record totals")
# zip archives in project dir
zips = [os.path.join(r, f) for r, _, fs in os.walk(PROJ) for f in fs if f.lower().endswith('.zip')]
zips.append(DL + r"\sevadataset.zip")  # source PDFs for the 2023-2025 CSV; listed only, never extracted
for z in zips:
    with zipfile.ZipFile(z) as zf:
        nm = zf.namelist(); npdf = sum(n.lower().endswith('.pdf') for n in nm)
        add(os.path.basename(z), z, "IDSP weekly outbreak report PDFs (source of the 2023-2025 CSV; inferred from source_pdf column paths 'sevadataset/...')", "ZIP", len(nm), None, "archive (raw source, unparsed)", f"{npdf} PDFs, not extracted or parsed in Phase 1; first entries: " + "; ".join(nm[:4]))
pd.DataFrame(inv).to_csv(os.path.join(OUT, "dataset_inventory.csv"), index=False)

# ---------------- 2. quality ----------------
q = []
def qrow(ds, col, dtype, n, miss, extra_dups, mn, mx, issues):
    q.append(dict(dataset=ds, column=col, dtype=dtype, non_null=n, missing=miss, missing_pct=round(100*miss/max(n+miss,1), 2),
                  min=mn, max=mx, issues=issues))
def table_quality(name, df, skip=()):
    for c in df.columns:
        if c.startswith('_') or c == 'std' or c in skip: continue
        s = df[c]; num = pd.api.types.is_numeric_dtype(s)
        qrow(name, c, str(s.dtype), int(s.notna().sum()), int(s.isna().sum()), None,
             s.min() if num else '', s.max() if num else '', '')
table_quality("maharashtra_epiclim_filtered.csv", hist.drop(columns=['_d']))
table_quality("maharashtra_outbreaks_2023_2025.csv", cont.drop(columns=['_start','_ps','_pe']))
table_quality("humid_tel_hr_cwprs_mh_2021_2025.csv", rh.drop(columns=['_t']))
dups = {}
dups['hist_full'] = int(hist.drop(columns=['_d','std']).duplicated().sum())
dups['hist_key'] = int(hist.duplicated(['outbreak_date','district','Disease']).sum())
dups['cont_full'] = int(cont.drop(columns=['_start','_ps','_pe','std']).duplicated().sum())
dups['cont_id'] = int(cont.outbreak_id.duplicated().sum())
dups['cont_key'] = int(cont.duplicated(['report_year','report_week','district','disease','reported_cases']).sum())
dups['rh_full'] = int(rh.drop(columns=['_t']).duplicated().sum())
dups['rh_time'] = int(rh.duplicated(['_t']).sum())
dups['rh_slno'] = int(rh.SlNo.duplicated().sum())

summ = []  # dataset-level checks
def chk(ds, check, result, severity): summ.append(dict(dataset=ds, column='(dataset-level)', dtype='', non_null='', missing='', missing_pct='', min='', max='', issues=f"{check}: {result} [{severity}]"))
H = 'maharashtra_epiclim_filtered.csv'; C = 'maharashtra_outbreaks_2023_2025.csv'; R = 'humid_tel_hr_cwprs_mh_2021_2025.csv'
chk(H, 'exact duplicate rows', dups['hist_full'], 'info'); chk(H, 'duplicate (date,district,disease)', dups['hist_key'], 'warn' if dups['hist_key'] else 'info')
chk(H, 'unparseable outbreak_date', int(hist._d.isna().sum()), 'info')
chk(H, 'Deaths > Cases', int((hist.Deaths > hist.Cases).sum()), 'warn')
chk(H, 'Cases <= 0', int((hist.Cases <= 0).sum()), 'info'); chk(H, 'negative Deaths', int((hist.Deaths < 0).sum()), 'info')
chk(H, 'year column != outbreak_date year', int((hist._d.dt.year != hist.year).sum()), 'warn')
chk(H, 'lat/lon outside Maharashtra box (15.5-22.1N, 72.6-80.9E)', int((~hist.Latitude.between(15.5,22.1) | ~hist.Longitude.between(72.6,80.9)).sum()), 'warn')
chk(H, 'Temp_C outside 0-50C', int(((hist.Temp_C < 0) | (hist.Temp_C > 50)).sum()), 'warn')
chk(H, 'Temp (K) vs Temp_C inconsistent (>0.1C)', int(((hist.Temp - 273.15 - hist.Temp_C).abs() > 0.1).sum()), 'warn')
chk(H, 'unmapped district names', "; ".join(sorted(hist[hist['std'].isna()].district.unique())) or 'none', 'info')
chk(H, 'district spelling variants (raw->canonical)', "; ".join(f"{r}->{canon(r)}" for r in sorted(hist.district.unique()) if canon(r) and canon(r) != r), 'warn')
chk(H, 'week_of_outbreak unparsable', int((~hist.week_of_outbreak.astype(str).str.match(r'^\d+(st|nd|rd|th) week$')).sum()), 'info')
chk(C, 'exact duplicate rows', dups['cont_full'], 'info'); chk(C, 'duplicate outbreak_id', dups['cont_id'], 'warn' if dups['cont_id'] else 'info')
chk(C, 'duplicate (year,week,district,disease,cases)', dups['cont_key'], 'warn' if dups['cont_key'] else 'info')
chk(C, 'Deaths > Cases', int((cont.reported_deaths > cont.reported_cases).sum()), 'warn')
chk(C, 'reported_cases <= 0', int((cont.reported_cases <= 0).sum()), 'info')
chk(C, 'missing report_period_start/end', int(cont.report_period_start.isna().sum()), 'warn')
chk(C, 'missing outbreak_start_date', int(cont.outbreak_start_date.isna().sum()), 'warn')
chk(C, 'outbreak_start_date after report_period_end', int((cont._start > cont._pe).sum()), 'warn')
chk(C, 'outbreak_start_date >30d before report_period_start', int(((cont._ps - cont._start).dt.days > 30).sum()), 'info')
chk(C, 'unmapped district names', "; ".join(sorted(cont[cont['std'].isna()].district.unique())) or 'none', 'info')
chk(C, 'district spelling variants', "; ".join(f"{r}->{canon(r)}" for r in sorted(cont.district.unique()) if canon(r) and canon(r) != r), 'warn')
chk(C, 'disease label variants (distinct raw labels)', int(cont.disease.nunique()), 'warn')
chk(C, 'disease labels with broken tokens (split words / district fragments)', "; ".join(sorted([d for d in cont.disease.unique() if re.search(r'Plasmodiu m|Gastroenterit is|Gastroenter itis|Sambhajinag|Suburban', d)])) or 'none', 'warn')
raw_bad = cont_raw[cont_raw.disease.str.contains('Suburban|Sambhajinag', na=False)]
chk("maharashtra_weekly_outbreak_records_2023_2025.csv (raw upload)", 'rows with district fragment in disease column (needs fix)', int(len(raw_bad)), 'warn')
# weeks present
wk = cont.groupby('report_year').report_week.nunique()
chk(C, 'distinct report weeks per year (of 52/53)', "; ".join(f"{y}:{n}" for y, n in wk.items()), 'warn')
chk(C, 'status column populated', int(cont.status.notna().sum()), 'info')
chk(R, 'exact duplicate rows', dups['rh_full'], 'info'); chk(R, 'duplicate timestamps', dups['rh_time'], 'warn' if dups['rh_time'] else 'info')
chk(R, 'unparseable timestamps', int(rh._t.isna().sum()), 'info')
chk(R, 'RH outside 0-100', int(((rh[rhcol] < 0) | (rh[rhcol] > 100)).sum()), 'warn')
chk(R, 'RH == 0 or ==100 (possible sensor artefacts)', int(((rh[rhcol] == 0) | (rh[rhcol] == 100)).sum()), 'info')
step = rh.sort_values('_t')._t.diff().dt.total_seconds().div(60).value_counts().head(5)
chk(R, 'timestamp step minutes (top)', "; ".join(f"{k:g}min x{v}" for k, v in step.items()), 'info')
span_days = (rh._t.max() - rh._t.min()).days + 1
chk(R, 'days with any data / calendar days in span', f"{rh._t.dt.normalize().nunique()}/{span_days}", 'warn')
chk(R, 'filename year range vs actual data', f"filename 2021_2025; data {rh._t.min():%Y-%m-%d} to {rh._t.max():%Y-%m-%d}", 'warn')
chk(R, 'stations', rh.Station.nunique(), 'warn'); chk(R, 'districts', "; ".join(rh.District.unique()), 'warn')
for k, nm in [('rain','imdweb_5dgbAskl.nc'),('tmin','imdweb_Oe9H04Bg.nc'),('tmax','imdweb_tXexQbB6.nc')]:
    g = G[k]
    qrow(nm, k, g['dtype'], g['size'] - g['nan'], g['nan'], None, round(g['mn'],2), round(g['mx'],2), f"units={g['units']}; mean={g['mean']:.2f}; non-consecutive time steps={g['gaps']}")
chk('imdweb_*.nc', 'caveat', 'fill/NaN counts reported per file; negative-rainfall / tmin>tmax checks below', 'info')
with nc.Dataset(F['tmin']) as a, nc.Dataset(F['tmax']) as b:
    tn = np.ma.filled(a['tmin']['tmin'][:], np.nan); tx = np.ma.filled(b['tmax']['tmax'][:], np.nan)
    chk('imdweb_Oe9H04Bg.nc / imdweb_tXexQbB6.nc', 'cells where tmin > tmax', int(np.nansum(tn > tx)), 'warn')
with nc.Dataset(F['rain']) as a:
    rr = np.ma.filled(a['rain']['rain'][:], np.nan)
    chk('imdweb_5dgbAskl.nc', 'negative rainfall values', int(np.nansum(rr < 0)), 'warn')
    chk('imdweb_5dgbAskl.nc', 'ocean/outside-India NaN cells (share)', f"{np.isnan(rr).mean():.1%}", 'info')
pd.concat([pd.DataFrame(q), pd.DataFrame(summ)]).to_csv(os.path.join(OUT, "dataset_quality_report.csv"), index=False)

# ---------------- 3. disease coverage ----------------
dr = []
for dname, pat in DISEASES.items():
    for ds, df, dc, cc, dd, tcol in [('maharashtra_epiclim_filtered.csv (2009-2022)', hist, 'Disease', 'Cases', 'Deaths', '_d'),
                                      ('maharashtra_outbreaks_2023_2025.csv (2023-2025)', cont, 'disease', 'reported_cases', 'reported_deaths', '_ps')]:
        m = df[dis_flag(df[dc], pat)]
        if dname == 'Acute Diarrhoeal Disease': m = m[~dis_flag(m[dc], 'food poison')]
        dr.append(dict(disease=dname, dataset=ds, outbreak_records=len(m), reported_cases=int(m[cc].sum()) if len(m) else None,
                       reported_deaths=int(m[dd].sum()) if len(m) else None,
                       districts_with_records=m['std'].nunique() if len(m) else 0,
                       first_date=str(m[tcol].min())[:10] if len(m) else None, last_date=str(m[tcol].max())[:10] if len(m) else None,
                       distinct_raw_labels="; ".join(sorted(m[dc].unique())) if len(m) else '',
                       coverage_note=('NO RECORDS FOUND in this dataset' if len(m) == 0 else 'outbreak/cluster reports only - not total incidence; absent week != zero cases')))
# extra categories noted
for extra, pat in [('Food poisoning (not requested)', 'food poison'), ('Zika (not requested)', 'zika'), ('Hepatitis A/E (not requested)', 'hepatitis'), ('Typhoid (not requested)', 'typhoid')]:
    for ds, df, dc, cc, dd in [('hist', hist, 'Disease', 'Cases', 'Deaths'), ('cont', cont, 'disease', 'reported_cases', 'reported_deaths')]:
        m = df[dis_flag(df[dc], pat)]
        if len(m): dr.append(dict(disease=extra, dataset=ds, outbreak_records=len(m), reported_cases=int(m[cc].sum()), reported_deaths=int(m[dd].sum()), districts_with_records=m['std'].nunique(), first_date='', last_date='', distinct_raw_labels='', coverage_note='additional category present'))
pd.DataFrame(dr).to_csv(os.path.join(OUT, "disease_coverage_report.csv"), index=False)
# all raw labels (appendix) printed to console
# ---------------- 4. district coverage ----------------
cd = []
rh_dist = set(rh.District.unique())
for o in OFFICIAL:
    h = hist[hist['std'] == o]; c = cont[cont['std'] == o]
    cd.append(dict(district=o,
        hist_records=len(h), hist_raw_names="; ".join(sorted(h.district.unique())), hist_first=str(h._d.min())[:10] if len(h) else '', hist_last=str(h._d.max())[:10] if len(h) else '',
        cont_records=len(c), cont_raw_names="; ".join(sorted(c.district.unique())), cont_first_report_week=f"{c.report_year.min()}-W{int(c[c.report_year==c.report_year.min()].report_week.min()):02d}" if len(c) else '',
        cont_years_present="; ".join(map(str, sorted(c.report_year.unique()))) if len(c) else '',
        in_any_disease_dataset='yes' if (len(h) or len(c)) else 'NO',
        in_both_disease_datasets='yes' if (len(h) and len(c)) else 'no',
        cwprs_humidity_station='yes (Pune only; 1 station)' if o == 'Pune' and 'Pune' in rh_dist else 'no',
        imd_grid_coverage_note='state-wide grid; district extraction not yet done (needs boundaries/centroids)',
        notes={'Mumbai City':'Historical file labels it "Mumbai (city)"; no 2023-25 records under Mumbai City', 'Mumbai Suburban':'No historical records; 2023-25 records only',
               'Chhatrapati Sambhajinagar':'Historical name Aurangabad; renamed 2023', 'Dharashiv':'Historical name Osmanabad; renamed 2023', 'Ahmednagar':'Officially renamed Ahilyanagar (2025)',
               'Palghar':'Carved from Thane in 2014; pre-2014 Palghar events (if any) sit under Thane in official records', 'Solapur':'Historical spellings Sholapur/Solapur', 'Nashik':'Historical spellings Nasik/Nashik'}.get(o, '')))
unm = sorted(set(hist[hist['std'].isna()].district) | set(cont[cont['std'].isna()].district))
cd.append(dict(district='[UNMAPPED NAMES]', hist_raw_names="; ".join(unm) or 'none'))
pd.DataFrame(cd).to_csv(os.path.join(OUT, "district_coverage_report.csv"), index=False)

# ---------------- 5. climate coverage ----------------
cl = []
for k, nm, var in [('rain','imdweb_5dgbAskl.nc','Rainfall'), ('tmin','imdweb_Oe9H04Bg.nc','Minimum temperature'), ('tmax','imdweb_tXexQbB6.nc','Maximum temperature')]:
    g = G[k]
    cl.append(dict(variable=var, file=nm, available='yes', units=g['units'], temporal_resolution='daily', start=g['t0'], end=g['t1'], n_time_steps=g['n'],
        lat_min=round(float(g['lat'].min()),3), lat_max=round(float(g['lat'].max()),3), lon_min=round(float(g['lon'].min()),3), lon_max=round(float(g['lon'].max()),3),
        grid_res_deg=g['res'], grid_shape=f"{len(g['lat'])}x{len(g['lon'])}", spatial_scope='gridded, Maharashtra bounding box', value_min=round(g['mn'],2), value_max=round(g['mx'],2), missing_values=g['nan'],
        note=''))
cl.append(dict(variable='Average temperature', file='(none)', available='NOT AVAILABLE as a dataset', note='Could be derived as (tmin+tmax)/2 only at the coarse 1-degree grid; not provided. Historical CSV has a separate Temp/Temp_C column of undocumented source/definition'))
cl.append(dict(variable='Relative humidity', file='humid_tel_hr_cwprs_mh_2021_2025.csv', available='partial: 1 station', units='%', temporal_resolution='sub-hourly telemetry (~20 min)', start=str(rh._t.min())[:10], end=str(rh._t.max())[:10],
    n_time_steps=len(rh), lat_min=float(rh.Latitude.min()), lat_max=float(rh.Latitude.max()), lon_min=float(rh.Longitude.min()), lon_max=float(rh.Longitude.max()), spatial_scope='single point: CWPRS campus, Khadakvasla, Haveli, Pune',
    value_min=float(rh[rhcol].min()), value_max=float(rh[rhcol].max()), missing_values=int(rh[rhcol].isna().sum()), note='Not representative of other districts; data covers 2024-05-11 onward despite filename saying 2021-2025'))
cl.append(dict(variable='Embedded climate columns in historical disease CSV (preci, LAI, Temp, Temp_C)', file=H, available='yes, per outbreak row', start=str(hist._d.min())[:10], end=str(hist._d.max())[:10],
    spatial_scope='point per district (Latitude/Longitude columns)', value_min=f"preci {hist.preci.min():.2f}; Temp_C {hist.Temp_C.min():.1f}", value_max=f"preci {hist.preci.max():.2f}; Temp_C {hist.Temp_C.max():.1f}",
    missing_values=f"preci {int(hist.preci.isna().sum())}; Temp_C {int(hist.Temp_C.isna().sum())}", note='Source/units of preci, LAI, Temp undocumented; only for 2009-2022 event dates; no humidity'))
# do district points fall within grid?
pts = hist.drop_duplicates('std').dropna(subset=['std'])[['std','Latitude','Longitude']]
inside = pts[(pts.Latitude.between(G['rain']['lat'].min(), G['rain']['lat'].max())) & (pts.Longitude.between(G['rain']['lon'].min(), G['rain']['lon'].max())) &
             (pts.Latitude.between(G['tmin']['lat'].min(), G['tmin']['lat'].max())) & (pts.Longitude.between(G['tmin']['lon'].min(), G['tmin']['lon'].max()))]
cl.append(dict(variable='District-point containment check', file='historical CSV lat/lon vs NetCDF extents', available=f"{len(inside)}/{len(pts)} district points inside both rain and temperature grid extents", note='Only districts with a lat/lon in the historical file can be tested; Mumbai Suburban has none'))
pd.DataFrame(cl).to_csv(os.path.join(OUT, "climate_coverage_report.csv"), index=False)

# ---------------- 6. compatibility ----------------
def ov(a0, a1, b0, b1):
    s = max(pd.Timestamp(a0), pd.Timestamp(b0)); e = min(pd.Timestamp(a1), pd.Timestamp(b1))
    return max((e - s).days + 1, 0), str(s)[:10] if e >= s else '', str(e)[:10] if e >= s else ''
hr = (hist._d.min(), hist._d.max()); cr = (cont._ps.min(), cont._pe.max()); rr_ = (pd.Timestamp(G['rain']['t0']), pd.Timestamp(G['rain']['t1'])); hh = (rh._t.min().normalize(), rh._t.max().normalize())
comp = []
def crow(a, b, keys, days_rng, freq_a, freq_b, ok, notes):
    comp.append(dict(dataset_a=a, dataset_b=b, join_keys=keys, overlap_days=days_rng[0], overlap_start=days_rng[1], overlap_end=days_rng[2], frequency_a=freq_a, frequency_b=freq_b, can_combine=ok, notes=notes))
crow('historical disease CSV', 'IMD rain/tmin/tmax NetCDF', 'district + date', ov(*hr, *rr_), 'event-based (outbreak_date)', 'daily', 'NO', f"Disease ends {hr[1]:%Y-%m-%d}; IMD files start {rr_[0]:%Y-%m-%d}. Zero overlap")
crow('historical disease CSV', 'CWPRS humidity', 'district + date', ov(*hr, *hh), 'event-based', 'sub-hourly', 'NO', 'Zero overlap and humidity is Pune only')
crow('2023-25 disease CSV', 'IMD rainfall NetCDF', 'district + report week (aggregate daily grid to weekly/district)', ov(*cr, *rr_), 'weekly report (outbreak start date often missing)', 'daily', 'YES (after processing)', f"Disease report window {cr[0]:%Y-%m-%d} to {cr[1]:%Y-%m-%d} inside IMD window; needs district->grid mapping")
crow('2023-25 disease CSV', 'IMD tmin/tmax NetCDF', 'district + report week', ov(*cr, *rr_), 'weekly report', 'daily, 1-degree grid (9x9)', 'YES (coarse spatial)', '1-degree cells are ~110 km; many districts share a cell')
crow('2023-25 disease CSV', 'CWPRS humidity', 'district + report week', ov(*cr, *hh), 'weekly report', 'sub-hourly', 'ONLY Pune district', f"Pune 2023-25 records within humidity window: {int(((cont['std']=='Pune') & (cont._ps >= hh[0]) & (cont._ps <= hh[1])).sum())}; all other districts: no humidity")
crow('historical disease CSV', '2023-25 disease CSV', 'district + disease', ov(*hr, *cr), 'event-based', 'weekly report', 'only with caution', 'Gap between 2022-06 and 2023; different schema/reporting practices, different Mumbai handling; not evidence of the same surveillance process')
crow('IMD rain', 'IMD tmin/tmax', 'lat/lon + date', ov(*rr_, *rr_), 'daily 0.25 deg', 'daily 1.0 deg', 'YES after regridding', 'Different grid resolutions; grid extents differ')
pd.DataFrame(comp).to_csv(os.path.join(OUT, "dataset_compatibility_report.csv"), index=False)

# ---------------- console diagnostics for the summary ----------------
print("HIST rows", len(hist), "CONT rows", len(cont), "RH rows", len(rh))
print("dups", dups)
print(pd.DataFrame(dr)[['disease','dataset','outbreak_records','reported_cases','reported_deaths','districts_with_records']].to_string())
print("hist districts covered", hist['std'].nunique(), "cont", cont['std'].nunique(), "either", len(set(hist['std'].dropna())|set(cont['std'].dropna())), "both", len(set(hist['std'].dropna())&set(cont['std'].dropna())))
print("missing hist", sorted(set(OFFICIAL)-set(hist['std'].dropna())), "missing cont", sorted(set(OFFICIAL)-set(cont['std'].dropna())))
print("unmapped", unm)
print(pd.DataFrame(summ)[['dataset','issues']].to_string())
print(pd.DataFrame(comp)[['dataset_a','dataset_b','overlap_days','overlap_start','overlap_end','can_combine']].to_string())
print({k:(g['t0'],g['t1'],g['res'],g['lat'].min(),g['lat'].max(),g['lon'].min(),g['lon'].max(),g['nan'],g['size']) for k,g in G.items()})
print("hist years", hist.year.value_counts().sort_index().to_dict())
print("cont rows by year", cont.report_year.value_counts().sort_index().to_dict(), "cases", int(cont.reported_cases.sum()), "deaths", int(cont.reported_deaths.sum()))
print("hist cases", int(hist.Cases.sum()), "deaths", int(hist.Deaths.sum()))
print("rh range", rh._t.min(), rh._t.max(), rh[rhcol].min(), rh[rhcol].max(), rh[rhcol].isna().sum())
print("cont freq status", cont.status.value_counts().to_dict())

