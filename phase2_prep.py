"""Phase 2 prep (non-destructive): copy raw sources into data/raw, write checksums + mapping tables, update registry.
Originals are never modified. No cleaning or imputation is performed."""
import os, shutil, hashlib, json, re
import pandas as pd
P = r"C:\Users\Janhvi Suryavanshi\.gemini\antigravity\scratch\JEEVAN-DRISHTI"
DL = r"C:\Users\Janhvi Suryavanshi\Downloads"
UP = r"C:\Users\Janhvi Suryavanshi\.gemini\antigravity\brain\77ab7f37-6ad5-4925-b8cc-b64abd77919f\.user_uploaded"
RAW = os.path.join(P, "data", "raw"); REF = os.path.join(P, "data", "reference")
os.makedirs(RAW, exist_ok=True); os.makedirs(REF, exist_ok=True)
SRC = {
 "maharashtra_epiclim_filtered.csv": DL + r"\maharashtra_epiclim_filtered.csv",
 "maharashtra_weekly_outbreak_records_2023_2025_ORIGINAL_UPLOAD.csv": UP + r"\media_1793432423584_ffb33b26.csv",
 "imdweb_5dgbAskl.nc": DL + r"\imdweb_5dgbAskl.nc",
 "imdweb_Oe9H04Bg.nc": DL + r"\imdweb_Oe9H04Bg.nc",
 "imdweb_tXexQbB6.nc": DL + r"\imdweb_tXexQbB6.nc",
 "humid_tel_hr_cwprs_mh_2021_2025.csv": DL + r"\humid_tel_hr_cwprs_mh_2021_2025.csv",
}
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
rows = []
for name, src in SRC.items():
    dst = os.path.join(RAW, name)
    if not os.path.exists(dst): shutil.copy2(src, dst)
    a, b = sha(src), sha(dst)
    rows.append(dict(raw_copy=dst, original_path=src, sha256_original=a, sha256_copy=b, identical=a == b, size_bytes=os.path.getsize(dst)))
pd.DataFrame(rows).to_csv(os.path.join(RAW, "raw_manifest_checksums.csv"), index=False)

# district alias table (user-confirmed) + disease label map (option a: combined labels stay separate categories)
OFFICIAL = [l for l in re.findall(r"'([^']+)'", open(os.path.join(P, "phase1_inspect.py"), encoding="utf-8").read().split("OFFICIAL = [")[1].split("]")[0])]
ALIAS = {'Aurangabad':'Chhatrapati Sambhajinagar','Osmanabad':'Dharashiv','Osmanabad (Maharashtra)':'Dharashiv','Nasik':'Nashik','Sholapur':'Solapur',
         'Palaghar':'Palghar','Mumbai (city)':'Mumbai City','Ahmadnagar':'Ahmednagar','Ahilyanagar':'Ahmednagar'}
d = [dict(raw_name=o, canonical=o, kind='official') for o in OFFICIAL] + [dict(raw_name=k, canonical=v, kind='alias') for k, v in ALIAS.items()]
pd.DataFrame(d).to_csv(os.path.join(REF, "district_alias_table.csv"), index=False)

hist = pd.read_csv(SRC["maharashtra_epiclim_filtered.csv"]); cont = pd.read_csv(P + r"\maharashtra_outbreaks_2023_2025.csv")
def lab(x):
    s = x.lower()
    if re.search(r'dengue', s) and re.search(r'chikungunya', s): return 'Dengue + Chikungunya (combined)'
    if 'dengue' in s and 'malaria' in s: return 'Dengue + Malaria (combined)'
    if 'dengue' in s: return 'Dengue'
    if 'chikungunya' in s: return 'Chikungunya'
    if 'malaria' in s or 'plasmodium' in s.replace(' ', ''): return 'Malaria'
    if 'cholera' in s: return 'Cholera'
    if 'leptospir' in s: return 'Leptospirosis'
    if re.search(r'diarrh|gastroenter', s.replace(' ', '')) and 'food' not in s: return 'Acute Diarrhoeal Disease'
    if re.search(r'heat|sunstroke', s): return 'Heat-related illness'
    return None
labels = sorted(set(hist.Disease) | set(cont.disease))
m = pd.DataFrame(dict(raw_label=labels)); m['proposed_category'] = m.raw_label.map(lab)
m['status'] = m.proposed_category.map(lambda v: 'mapped' if v else 'UNMAPPED - kept as-is, needs review')
m['in_historical'] = m.raw_label.isin(hist.Disease); m['in_contemporary'] = m.raw_label.isin(cont.disease)
m.to_csv(os.path.join(REF, "disease_label_map_proposed.csv"), index=False)

# registry: point to raw copies, record decisions
reg = json.load(open(P + r"\model_dataset_registry.json", encoding="utf-8"))
byname = {os.path.basename(r['raw_copy']): r for r in rows}
for ds in reg['datasets']:
    key = ds['filename'].split(' (')[0]
    for n_, r in byname.items():
        if n_ == key or (key.startswith('maharashtra_weekly') and 'ORIGINAL' in n_):
            ds['original_path'] = ds['path']; ds['path'] = r['raw_copy']; ds['sha256'] = r['sha256_copy']
reg['decisions'] = dict(
  raw_copies="Sources copied to data/raw; originals untouched; checksums in data/raw/raw_manifest_checksums.csv",
  district_aliases="User-confirmed; data/reference/district_alias_table.csv",
  combined_disease_labels="Option (a): dengue+chikungunya kept as own category; suspected/fever variants map to the base disease. Awaiting user confirmation of this choice.",
  heat_related_illness="Kept in scope; no records exist in any available dataset, so not modelable unless a file is supplied")
json.dump(reg, open(P + r"\model_dataset_registry.json", "w", encoding="utf-8"), indent=2)
print(pd.DataFrame(rows)[['raw_copy', 'identical']].to_string()); print(len(OFFICIAL)); print(m[['raw_label', 'proposed_category']].to_string())
