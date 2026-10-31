"""Disease-record preprocessing: historical EpiClim-derived file + IDSP weekly outbreak file -> one cleaned table.

Rules: nothing is imputed; missing deaths stay NaN; duplicates are resolved only when byte-identical (later copy goes to
rejected_records with a pointer to the kept record); conflicting/possible duplicates stay in the data and are flagged.
"""
import numpy as np
import pandas as pd
from . import config as C
from .standardize import (parse_ordinal_date, parse_iso_date, parse_week_number, standardize_district,
                          standardize_disease, repair_split_district, disease_group, add_flag, norm_name)


def rel(p):
    return str(p.relative_to(C.PROJ)).replace("\\", "/")


def _load_hist(log):
    raw = pd.read_csv(C.HIST_RAW)
    df = pd.DataFrame(index=raw.index)
    df["source_dataset"] = "historical_epiclim"
    df["_repaired"] = False
    df["source_file"] = rel(C.HIST_RAW)
    df["source_row"] = raw.index
    df["record_id"] = "HIST-" + df.source_row.astype(str).str.zfill(5)
    df["outbreak_id"] = pd.Series(pd.NA, index=raw.index, dtype="object")
    df["district_source"] = raw.district
    df["disease_source"] = raw.Disease
    df["district_raw"] = raw.district
    df["disease_raw"] = raw.Disease
    df["event_date"] = parse_iso_date(raw.outbreak_date)
    df["event_date_kind"] = "outbreak_date"
    df["report_year"] = pd.to_numeric(raw.year, errors="coerce").astype("Int64")
    df["report_week"] = parse_week_number(raw.week_of_outbreak)
    df["report_period_start"] = pd.NaT
    df["report_period_end"] = pd.NaT
    df["report_period_source"] = "not_applicable_event_based"
    df["cases"] = pd.to_numeric(raw.Cases, errors="coerce")
    df["deaths"] = pd.to_numeric(raw.Deaths, errors="coerce")
    df["status"] = pd.NA
    df["source_pdf"] = pd.NA
    df["page"] = pd.NA
    df["lat"] = raw.Latitude
    df["lon"] = raw.Longitude
    df["hist_preci"] = raw.preci
    df["hist_LAI"] = raw.LAI
    df["hist_Temp_C"] = raw.Temp_C
    df["join_date_start"] = df.event_date
    df["join_date_end"] = df.event_date
    df["_exact_key"] = raw.astype(str).agg("|".join, axis=1)
    log.add("historical_epiclim", "load", "read raw CSV unchanged", len(raw), len(raw), 0, rel(C.HIST_RAW))
    bad = int(df.event_date.isna().sum())
    log.add("historical_epiclim", "date_standardization", "outbreak_date parsed as ISO yyyy-mm-dd", len(df), len(df), bad,
            f"{bad} unparseable dates (left NaT)")
    return raw, df


def _load_idsp(log):
    raw = pd.read_csv(C.IDSP_RAW, encoding="utf-8-sig", dtype=str)
    n = len(raw)
    dist, dis, rep = [], [], []
    for d, s in zip(raw.district, raw.disease):
        d2, s2, r = repair_split_district(d, s)
        dist.append(d2); dis.append(s2); rep.append(r)
    rep = np.array(rep)
    log.add("idsp_weekly", "structural_repair", "re-joined district names split into the disease column by the PDF parser",
            n, n, int(rep.sum()), "e.g. 'Mumbai'+'Suburban Zika Virus' -> 'Mumbai Suburban'+'Zika Virus'; source values kept in *_source columns")
    df = pd.DataFrame(index=raw.index)
    df["source_dataset"] = "idsp_weekly"
    df["source_file"] = rel(C.IDSP_RAW)
    df["source_row"] = raw.index
    df["record_id"] = "IDSP-" + df.source_row.astype(str).str.zfill(5)
    df["outbreak_id"] = raw.outbreak_id
    df["district_source"] = raw.district
    df["disease_source"] = raw.disease
    df["district_raw"] = dist
    df["disease_raw"] = dis
    df["_repaired"] = rep
    df["event_date"] = parse_iso_date(raw.outbreak_start_date)
    df["event_date_kind"] = "outbreak_start_date"
    df["report_year"] = pd.to_numeric(raw.report_year, errors="coerce").astype("Int64")
    df["report_week"] = pd.to_numeric(raw.report_week, errors="coerce").astype("Int64")
    ps, pe = parse_ordinal_date(raw.report_period_start), parse_ordinal_date(raw.report_period_end)
    df["report_period_source"] = np.where(ps.notna() & pe.notna(), "source", "missing")
    df["cases"] = pd.to_numeric(raw.reported_cases, errors="coerce")
    df["deaths"] = pd.to_numeric(raw.reported_deaths, errors="coerce")
    df["status"] = raw.status
    df["source_pdf"] = raw.source_pdf
    df["page"] = pd.to_numeric(raw.page, errors="coerce").astype("Int64")
    df["lat"] = np.nan
    df["lon"] = np.nan
    df["hist_preci"] = np.nan
    df["hist_LAI"] = np.nan
    df["hist_Temp_C"] = np.nan
    df["_last_reported"] = parse_iso_date(raw.last_reported_date)
    # --- report-week convention check, then derive missing periods only where the convention is verified
    wk = df.report_week.astype("float")
    offs = ps - pd.to_timedelta((wk - 1) * 7, unit="D")
    full = (pe - ps).dt.days == 6          # anchor verified on full 7-day weeks only; short weeks (year boundary) stay as in source
    anchors, info = {}, []
    for y in sorted(df.report_year.dropna().unique()):
        vals = offs[(df.report_year == y) & full].dropna().unique()
        anchors[int(y)] = pd.Timestamp(vals[0]) if len(vals) == 1 else None
        info.append(f"{int(y)}: {len(vals)} distinct week-1 anchors" + (f" ({pd.Timestamp(vals[0]).date()})" if len(vals) == 1 else " (convention NOT verified)"))
    len_bad = int(((pe - ps).dt.days.dropna() != 6).sum())
    log.add("idsp_weekly", "reporting_frequency_validation", "checked report_period = 7 days and week-1 anchor per year",
            n, n, len_bad, "; ".join(info) + f"; periods not 7 days long: {len_bad}")
    derived = np.zeros(n, dtype=bool)
    for y, a in anchors.items():
        if a is None:
            continue
        m = (df.report_year == y) & ps.isna() & wk.notna()
        ps[m] = a + pd.to_timedelta((wk[m] - 1) * 7, unit="D")
        pe[m] = ps[m] + pd.Timedelta(days=6)
        derived |= m.values
    df["report_period_start"], df["report_period_end"] = ps, pe
    df.loc[derived, "report_period_source"] = "derived_week_convention"
    log.add("idsp_weekly", "date_standardization", "ordinal dates parsed; missing report periods derived from verified week convention",
            n, n, int(derived.sum()), "derived rows flagged report_period_source=derived_week_convention; no other dates inferred")
    df["join_date_start"], df["join_date_end"] = ps, pe
    df["_exact_key"] = raw.drop(columns=["source_pdf"]).fillna("").astype(str).agg("|".join, axis=1)
    log.add("idsp_weekly", "load", "read raw CSV (utf-8-sig, all text) unchanged", n, n, 0, rel(C.IDSP_RAW))
    return raw, df, anchors


def _rejected(df, reason, detail, kept=None, raws=None):
    out = df[["source_dataset", "source_file", "source_row", "record_id"]].copy()
    out["reject_reason"] = reason
    out["detail"] = detail
    out["kept_record_id"] = kept if kept is not None else pd.NA
    if raws is not None:
        out["original_record"] = [raws[d].loc[r].to_json() for d, r in zip(out.source_dataset, out.source_row)]
    return out


def _dup_groups(df, keys, prefix):
    """Group ids for rows sharing `keys`; status 'conflict' if cases/deaths differ in the group else 'possible_duplicate'."""
    g = df.groupby(keys, dropna=False)
    size = g["record_id"].transform("size")
    m = size > 1
    ids = pd.Series(pd.NA, index=df.index, dtype="object")
    status = pd.Series("unique", index=df.index, dtype="object")
    if m.any():
        codes = g.ngroup()
        ids[m] = prefix + codes[m].astype(str).str.zfill(4)
        diff = (g["cases"].transform("nunique") > 1) | (df["deaths"].fillna(-1).groupby([df[k] for k in keys]).transform("nunique") > 1)
        status[m] = np.where(diff[m], "conflict", "possible_duplicate")
    return ids, status


def build_disease(log):
    raw_h, h = _load_hist(log)
    raw_i, i, anchors = _load_idsp(log)
    raws = {"historical_epiclim": raw_h, "idsp_weekly": raw_i}
    df = pd.concat([h, i], ignore_index=True)
    df["flags"] = ""
    stats = {"anchors": {k: (str(v.date()) if v is not None else None) for k, v in anchors.items()},
             "raw_rows": {"historical_epiclim": len(h), "idsp_weekly": len(i)}}

    # ---- district standardization
    dm = {raw: standardize_district(raw) for raw in df.district_raw.unique()}
    df["district_std"] = df.district_raw.map(lambda x: dm[x][0])
    df["district_mapping_type"] = df.district_raw.map(lambda x: dm[x][1])
    df.loc[df._repaired.astype(bool), "district_mapping_type"] += "+structural_repair"
    ch = int((df.district_std != df.district_raw).sum())
    log.add("all", "district_standardization", "mapped raw names to the 36 official districts", len(df), len(df), ch,
            f"{ch} rows renamed; unmapped={int(df.district_std.isna().sum())}")
    mapping = (df.groupby(["source_dataset", "district_source", "district_raw", "district_std", "district_mapping_type"], dropna=False)
                 .size().rename("n_records").reset_index())

    # ---- disease standardization
    dd = {raw: standardize_disease(raw) for raw in df.disease_raw.dropna().unique()}
    df["disease_std"] = df.disease_raw.map(lambda x: dd.get(x, (None,))[0])
    df["label_suspected"] = df.disease_raw.map(lambda x: dd.get(x, (None, False))[1])
    df["disease_mapped"] = df.disease_raw.map(lambda x: dd.get(x, (None, False, False))[2])
    df["disease_group"] = df.disease_std.map(disease_group)
    df["is_target_disease"] = df.disease_std.isin(C.TARGET_DISEASES)
    nchg = int((df.disease_std != df.disease_raw).sum())
    log.add("all", "disease_standardization", "rule-based label normalisation (combined labels kept as own categories; "
            "suspected kept as flag)", len(df), len(df), nchg,
            f"{df.disease_raw.nunique()} raw labels -> {df.disease_std.nunique()} standard labels; unmapped labels: "
            f"{sorted(df.loc[~df.disease_mapped, 'disease_raw'].dropna().unique())}")

    # ---- flags (information only)
    add_flag(df, df._repaired.astype(bool), "structural_repair_split_district")
    add_flag(df, df.district_mapping_type.str.startswith("alias"), "district_renamed_or_respelled")
    add_flag(df, df.disease_group == "combined", "combined_disease_label")
    add_flag(df, df.label_suspected, "suspected_label")
    add_flag(df, ~df.disease_mapped, "unmapped_disease_label")
    add_flag(df, df.deaths.isna(), "deaths_missing_not_zero")
    add_flag(df, df.report_period_source == "derived_week_convention", "report_period_derived_week_convention")
    isi = df.source_dataset == "idsp_weekly"
    add_flag(df, isi & df.event_date.isna(), "outbreak_start_date_missing")
    add_flag(df, isi & (df.event_date > df.report_period_end), "event_after_report_period")
    add_flag(df, isi & ((df.report_period_start - df.event_date).dt.days > 30), "event_date_gt30d_before_report")
    add_flag(df, isi & (df._last_reported < df.event_date), "last_reported_before_event")
    add_flag(df, isi & ((df.report_period_end - df.report_period_start).dt.days < 6), "report_period_shorter_than_7d_in_source")
    ish = ~isi
    yr_mis = ish & (df.event_date.dt.year != df.report_year)
    add_flag(df, yr_mis, "year_column_mismatch")
    b = C.MAHARASHTRA_BBOX
    add_flag(df, ish & ~(df.lat.between(b[0], b[1]) & df.lon.between(b[2], b[3])), "coordinates_outside_maharashtra")
    add_flag(df, ish & (df.hist_Temp_C > 50), "embedded_temp_c_gt_50_implausible_air_temp")
    pc = pd.Timestamp(C.PALGHAR_CREATED)
    df["boundary_note"] = ""
    df.loc[(df.district_std == "Thane") & (df.join_date_start < pc), "boundary_note"] = "thane_pre_palghar_split_includes_palghar_area"
    df.loc[(df.district_std == "Palghar") & (df.join_date_start < pc), "boundary_note"] = "palghar_before_district_creation_2014-08-01"
    df.loc[(df.district_std == "Chhatrapati Sambhajinagar") & df.district_raw.str.lower().eq("aurangabad"), "boundary_note"] = "renamed_from_Aurangabad_name_change_only"
    df.loc[(df.district_std == "Dharashiv") & df.district_raw.str.lower().str.startswith("osmanabad"), "boundary_note"] = "renamed_from_Osmanabad_name_change_only"
    add_flag(df, df.boundary_note != "", "boundary_or_rename_note")
    for ds, g in df.groupby("source_dataset"):
        thr = g.cases.quantile(C.HIGH_VALUE_QUANTILE)
        add_flag(df, (df.source_dataset == ds) & (df.cases > thr), "cases_above_p99_5_review")
    log.add("all", "flagging", "informational flags added (none of these remove rows)", len(df), len(df),
            int((df["flags"] != "").sum()), "see 'flags' column")

    # ---- numeric validation + invalid records -> rejected
    rej = []
    def reject(mask, reason, detail):
        nonlocal df
        mask = mask.fillna(False).astype(bool)
        if mask.any():
            rej.append(_rejected(df[mask], reason, detail, raws=raws))
        log.add("all", "invalid_record_identification", f"rejected: {reason}", len(df), len(df) - int(mask.sum()),
                int(mask.sum()), detail)
        df = df[~mask]
    reject(df.cases.isna() | (df.cases <= 0) | (df.cases % 1 != 0), "invalid_cases", "cases missing, non-integer or <= 0")
    reject((df.deaths < 0) | (df.deaths > df.cases), "invalid_deaths", "deaths negative or greater than cases")
    reject(df.district_std.isna(), "unmapped_district", "district name could not be mapped to an official district")
    reject(df.join_date_start.isna() & df.event_date.isna(), "no_usable_date", "neither event date nor report period available")
    df = df.copy()

    # ---- duplicates
    n_before = len(df)
    dup = df.duplicated("_exact_key", keep="first")
    kept = df.groupby("_exact_key")["record_id"].transform("first")
    if dup.any():
        rej.append(_rejected(df[dup], "exact_duplicate",
                             "identical to an earlier record in every source field (IDSP: except source_pdf)", kept[dup], raws=raws))
    log.add("all", "duplicate_resolution", "exact duplicates moved to rejected_records (first copy kept)", n_before,
            n_before - int(dup.sum()), int(dup.sum()), "conflicting/possible duplicates are NOT removed")
    df = df[~dup].copy()
    df["dup_group_id"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df["duplicate_status"] = "unique"
    hm = df.source_dataset == "historical_epiclim"
    gi, gs = _dup_groups(df[hm], ["event_date", "district_std", "disease_std"], "HDUP-")
    df.loc[hm, "dup_group_id"], df.loc[hm, "duplicate_status"] = gi, gs
    im = ~hm
    sub = df[im]
    gi1, gs1 = _dup_groups(sub[sub.outbreak_id.notna()], ["outbreak_id"], "IDUP-ID-")
    df.loc[gi1.index, "dup_group_id"] = gi1
    df.loc[gi1.index, "duplicate_status"] = np.where(gs1 == "unique", "unique", gs1)
    rest = sub[~sub.index.isin(gi1.index[gi1.notna()])]
    gi2, gs2 = _dup_groups(rest, ["report_year", "report_week", "district_std", "disease_std", "cases"], "IDUP-KEY-")
    df.loc[gi2.index, "dup_group_id"] = gi2
    df.loc[gi2.index, "duplicate_status"] = np.where(gs2 == "possible_duplicate", "possible_duplicate_distinct_id", gs2)
    add_flag(df, df.duplicate_status.isin(["conflict"]), "conflicting_duplicate_group")
    add_flag(df, df.duplicate_status.str.startswith("possible"), "possible_duplicate_group")
    stats["dup_status"] = df.duplicate_status.value_counts().to_dict()
    log.add("all", "duplicate_detection", "grouped conflicting / possible duplicates (kept + flagged)", len(df), len(df),
            int((df.duplicate_status != "unique").sum()), str(stats["dup_status"]))

    rejected = pd.concat(rej, ignore_index=True) if rej else pd.DataFrame()
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])
    stats["cleaned_rows"] = df.source_dataset.value_counts().to_dict()
    stats["rejected_rows"] = rejected.groupby(["source_dataset", "reject_reason"]).size().to_dict() if len(rejected) else {}
    return df.reset_index(drop=True), rejected, mapping, stats

