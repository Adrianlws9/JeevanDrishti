"""Pure standardization helpers: dates, district names, disease names."""
import re
import numpy as np
import pandas as pd
from . import config as C

_ORD = re.compile(r"(\d+)(st|nd|rd|th)\b", re.I)


def parse_ordinal_date(series):
    """'2nd January 2023' -> Timestamp. Unparseable / missing -> NaT (never guessed)."""
    s = series.astype("string").str.strip().str.replace(_ORD, r"\1", regex=True)
    return pd.to_datetime(s, format="%d %B %Y", errors="coerce")


def parse_iso_date(series):
    return pd.to_datetime(series.astype("string").str.strip(), format="%Y-%m-%d", errors="coerce")


def parse_week_number(series):
    """'45th week' -> 45 (nullable int)."""
    return pd.to_numeric(series.astype("string").str.extract(r"(\d+)", expand=False), errors="coerce").astype("Int64")


def norm_name(x):
    return re.sub(r"\s+", " ", str(x).strip().lower())


_CANON = {norm_name(d): d for d in C.OFFICIAL_DISTRICTS}


def standardize_district(raw):
    """Return (canonical or None, mapping_type)."""
    k = norm_name(raw)
    if k in _CANON:
        return _CANON[k], "exact"
    if k in C.DISTRICT_ALIASES:
        return C.DISTRICT_ALIASES[k], "alias"
    return None, "unmapped"


def repair_split_district(district, disease):
    """Repair 'Mumbai' + 'Suburban Zika Virus' (and 'Sambhajinaga r ...') produced by unquoted
    multi-word district names in the source CSV. Returns (district, disease, repaired)."""
    if not isinstance(district, str) or not isinstance(disease, str):
        return district, disease, False
    target = C.SPLIT_DISTRICT_SUFFIX.get(norm_name(district))
    if target is None:
        return district, disease, False
    toks, acc, i = disease.split(), "", 0
    while i < len(toks) and acc != target and target.startswith(acc + toks[i].lower()):
        acc += toks[i].lower()
        i += 1
    if acc != target:
        return district, disease, False
    return district.strip() + " " + target.capitalize(), " ".join(toks[i:]), True


def standardize_disease(raw):
    """Return (standard label, suspected flag, mapped flag). Combined labels stay separate categories."""
    if not isinstance(raw, str) or not raw.strip():
        return None, False, False
    c = re.sub(r"[^a-z]", "", raw.lower())   # also heals split tokens: 'Plasmodiu m', 'Gastroenterit is'
    suspected = "suspected" in c
    den, chik = "dengue" in c, "chikungunya" in c
    mal = "malaria" in c or "plasmodium" in c
    rules = [
        (den and chik, "Dengue + Chikungunya"), (den and mal, "Dengue + Malaria"),
        (den, "Dengue"), (chik, "Chikungunya"), (mal, "Malaria"), ("cholera" in c, "Cholera"),
        ("leptospir" in c, "Leptospirosis"), ("diarrh" in c or "gastroenter" in c, "Acute Diarrhoeal Disease"),
        ("heatstroke" in c or "sunstroke" in c or "heatillness" in c or "heatrelated" in c, "Heat-related illness"),
        ("dysentery" in c, "Dysentery"), ("foodpoison" in c, "Food Poisoning"),
        ("hepatitisae" in c, "Hepatitis A + E"), ("hepatitisa" in c, "Hepatitis A"),
        ("hepatitis" in c, "Acute Hepatitis (unspecified)"),
        ("japaneseencephalitis" in c, "Japanese Encephalitis"), ("encephalit" in c, "Acute Encephalitis Syndrome"),
        ("measles" in c and "rubella" in c, "Measles + Rubella"), ("measles" in c, "Measles"),
        ("mumps" in c, "Mumps"), ("pertussis" in c, "Pertussis"), ("chickenpox" in c, "Chickenpox"),
        ("rabies" in c, "Rabies (human)"), ("scrubtyphus" in c, "Scrub Typhus"), ("typhoid" in c, "Typhoid"),
        ("zika" in c, "Zika Virus"), ("jaundice" in c, "Jaundice"),
        ("feverwithrash" in c, "Fever with Rash"), ("feverofunknown" in c, "Fever of Unknown Cause"),
        (c.startswith("fever"), "Fever (unspecified)"),
    ]
    for cond, label in rules:
        if cond:
            return label, suspected, True
    return raw.strip(), suspected, False


def disease_group(std):
    if std is None or (isinstance(std, float) and np.isnan(std)):
        return "unknown"
    if std in C.TARGET_DISEASES:
        return "target"
    if " + " in std:
        return "combined"
    return "other"


def add_flag(df, mask, flag):
    """Append `flag` to the semicolon-separated df['flags'] for rows where mask is True."""
    mask = mask.fillna(False).astype(bool)
    cur = df.loc[mask, "flags"]
    df.loc[mask, "flags"] = np.where(cur == "", flag, cur + ";" + flag)
