"""Climate preprocessing: IMD gridded rain/tmin/tmax -> district-day table (nearest grid cell), CWPRS humidity -> Pune daily.

No spatial/temporal gap filling is performed: a district whose nearest grid cell is ocean/NaN keeps NaN and is flagged.
"""
import re
import numpy as np
import pandas as pd
import netCDF4 as nc
from . import config as C
from .standardize import standardize_district
from .disease import rel


def district_points(disease_df):
    h = disease_df[disease_df.source_dataset == "historical_epiclim"]
    g = h.groupby("district_std")[["lat", "lon"]].agg(["mean", "std"])
    pts = pd.DataFrame({"lat": g[("lat", "mean")], "lon": g[("lon", "mean")],
                        "coord_spread_deg": g[[("lat", "std"), ("lon", "std")]].max(axis=1).fillna(0)})
    return pts


def load_grid(path, group, var, vmin, vmax, log, label):
    with nc.Dataset(path) as ds:
        g = ds.groups[group]
        units = g["time"].units
        m = re.match(r"days since (\d{4}-\d{2}-\d{2})", units)
        assert m, f"unexpected time units {units}"
        dates = pd.Timestamp(m.group(1)) + pd.to_timedelta(np.asarray(g["time"][:], dtype="float64"), unit="D")
        assert g[var].dimensions == ("time", "lat", "lon"), g[var].dimensions
        arr = np.ma.filled(g[var][:].astype("float64"), np.nan)
        lat, lon = np.asarray(g["lat"][:], "float64"), np.asarray(g["lon"][:], "float64")
        attrs = {k: str(getattr(ds, k)) for k in ds.ncattrs()}
        var_units = getattr(g[var], "units", "")
    n_nan = int(np.isnan(arr).sum())
    bad = (arr < vmin) | (arr > vmax)
    n_bad = int(np.nansum(bad))
    arr[bad] = np.nan
    steps = np.diff(dates.values).astype("timedelta64[D]").astype(int)
    log.add(label, "climate_validation", f"range check [{vmin},{vmax}], daily continuity, grid read",
            arr.size, arr.size - n_bad, n_bad,
            f"time {dates[0].date()}..{dates[-1].date()} n={len(dates)} non-daily steps={int((steps != 1).sum())}; "
            f"NaN cells (ocean/outside-India mask) {n_nan} ({n_nan / arr.size:.1%}); out-of-range set to NaN {n_bad}; units={var_units}")
    return dict(dates=dates, lat=lat, lon=lon, arr=arr, attrs=attrs, units=var_units, n_nan=n_nan, n_bad=n_bad,
                nonday_steps=int((steps != 1).sum()))


def _nearest(grid, lat, lon):
    i = int(np.abs(grid["lat"] - lat).argmin())
    j = int(np.abs(grid["lon"] - lon).argmin())
    return i, j


def build_humidity(log):
    rh = pd.read_csv(C.RH_RAW)
    col = [c for c in rh.columns if "Relative Humidity" in c][0]
    t = pd.to_datetime(rh["Data Acquisition Time"], format="%d-%m-%Y %H:%M", errors="coerce")
    v = pd.to_numeric(rh[col], errors="coerce")
    valid = (v > C.RH_MIN_EXCLUSIVE) & (v <= C.RH_MAX) & t.notna()
    reason = np.where(v > C.RH_MAX, "rh_above_100", np.where(v <= C.RH_MIN_EXCLUSIVE, "rh_zero_sensor_failure",
                      np.where(t.isna(), "unparseable_timestamp", "")))
    bad = ~valid
    rej = pd.DataFrame({"source_dataset": "humidity_cwprs", "source_file": rel(C.RH_RAW), "source_row": rh.index[bad],
                        "record_id": "RH-" + pd.Series(rh.index[bad]).astype(str).str.zfill(6),
                        "reject_reason": reason[bad.values], "detail": ("value=" + v[bad].astype(str)).values,
                        "kept_record_id": pd.NA, "original_record": [rh.loc[i].to_json() for i in rh.index[bad]]})
    log.add("humidity_cwprs", "climate_validation", "RH must be in (0,100]; invalid readings excluded from daily aggregates "
            "and listed in rejected_records", len(rh), int(valid.sum()), int(bad.sum()),
            f"above100={int((v > 100).sum())}; zero={int((v <= 0).sum())}; duplicate timestamps={int(t.duplicated().sum())}")
    d = pd.DataFrame({"date": t.dt.normalize(), "rh": v.where(valid)})
    g = d.groupby("date")
    daily = pd.DataFrame({"rh_mean": g.rh.mean(), "rh_min": g.rh.min(), "rh_max": g.rh.max(),
                          "rh_n_valid": g.rh.count(), "rh_n_raw": g.size()}).reset_index()
    daily["rh_n_invalid"] = daily.rh_n_raw - daily.rh_n_valid
    daily["rh_coverage_frac"] = (daily.rh_n_valid / C.RH_EXPECTED_OBS_PER_DAY).round(4)
    daily["rh_station"] = rh.Station.iloc[0]
    st_dist = standardize_district(rh.District.iloc[0])[0]
    steps = t.sort_values().diff().dt.total_seconds().div(60).value_counts().head(6).to_dict()
    info = dict(district=st_dist, station=rh.Station.iloc[0], lat=float(rh.Latitude.iloc[0]), lon=float(rh.Longitude.iloc[0]),
                n_raw=len(rh), n_invalid=int(bad.sum()), first=str(t.min()), last=str(t.max()), step_minutes=steps,
                days_with_data=int(daily.shape[0]), span_days=int((t.max().normalize() - t.min().normalize()).days + 1),
                days_low_coverage=int((daily.rh_coverage_frac < 0.5).sum()), stations=int(rh.Station.nunique()))
    return daily, rej, info


def build_climate(disease_df, log):
    pts = district_points(disease_df)
    rain = load_grid(C.RAIN_NC, "rain", "rain", 0.0, C.RAIN_MAX_MM_DAY, log, "imd_rain")
    tmin = load_grid(C.TMIN_NC, "tmin", "tmin", C.TEMP_MIN_C, C.TEMP_MAX_C, log, "imd_tmin")
    tmax = load_grid(C.TMAX_NC, "tmax", "tmax", C.TEMP_MIN_C, C.TEMP_MAX_C, log, "imd_tmax")
    assert rain["dates"].equals(tmin["dates"]) and rain["dates"].equals(tmax["dates"]), "climate time axes differ"
    dates = rain["dates"]
    daily_rh, rh_rej, rh_info = build_humidity(log)
    frames, cellinfo = [], []
    for dist in C.OFFICIAL_DISTRICTS:
        n = len(dates)
        f = pd.DataFrame({"date": dates, "district_std": dist})
        info = dict(district_std=dist)
        if dist in pts.index:
            lat, lon = float(pts.loc[dist, "lat"]), float(pts.loc[dist, "lon"])
            f["lat"], f["lon"] = lat, lon
            info.update(lat=lat, lon=lon, coord_source="historical_epiclim Latitude/Longitude (district point)")
            for name, g in (("rain", rain), ("tmin", tmin), ("tmax", tmax)):
                i, j = _nearest(g, lat, lon)
                series = g["arr"][:, i, j]
                f[{"rain": "rain_mm", "tmin": "tmin_c", "tmax": "tmax_c"}[name]] = series
                gl, gn = float(g["lat"][i]), float(g["lon"][j])
                f[f"{name}_grid_lat"], f[f"{name}_grid_lon"] = gl, gn
                info.update({f"{name}_grid_lat": gl, f"{name}_grid_lon": gn,
                             f"{name}_cell_dist_deg": round(float(np.hypot(gl - lat, gn - lon)), 3),
                             f"{name}_valid_days": int(np.isfinite(series).sum())})
        else:
            info.update(coord_source="NONE - no coordinates in any project dataset (not invented)")
            for c in ["lat", "lon", "rain_mm", "tmin_c", "tmax_c", "rain_grid_lat", "rain_grid_lon", "tmin_grid_lat",
                      "tmin_grid_lon", "tmax_grid_lat", "tmax_grid_lon"]:
                f[c] = np.nan
            info.update(rain_valid_days=0, tmin_valid_days=0, tmax_valid_days=0)
        frames.append(f)
        cellinfo.append(info)
    cl = pd.concat(frames, ignore_index=True)
    cl["tavg_derived_c"] = (cl.tmin_c + cl.tmax_c) / 2          # derived, explicitly named; NaN unless both inputs exist
    cl = cl.merge(daily_rh.assign(district_std=rh_info["district"]), on=["date", "district_std"], how="left")
    cl["flags"] = ""
    from .standardize import add_flag
    add_flag(cl, ~cl.lat.notna(), "no_district_coordinates")
    add_flag(cl, cl.lat.notna() & cl.rain_mm.isna(), "rain_cell_nan")
    add_flag(cl, cl.lat.notna() & cl.tmin_c.isna(), "tmin_cell_nan")
    add_flag(cl, cl.lat.notna() & cl.tmax_c.isna(), "tmax_cell_nan")
    add_flag(cl, cl.tmin_c > cl.tmax_c, "tmin_gt_tmax")
    add_flag(cl, cl.rh_n_valid.notna() & (cl.rh_coverage_frac < 0.5), "rh_low_coverage")
    cl["source_files"] = f"{rel(C.RAIN_NC)};{rel(C.TMIN_NC)};{rel(C.TMAX_NC)}"
    cl.loc[cl.rh_n_valid.notna(), "source_files"] += ";" + rel(C.RH_RAW)
    cell = pd.DataFrame(cellinfo)
    nan_districts = cell[(cell.rain_valid_days == 0) | (cell.tmin_valid_days == 0) | (cell.tmax_valid_days == 0)]
    log.add("climate", "district_extraction", "nearest-grid-cell extraction at district points; no gap filling",
            len(cl), len(cl), int((cl["flags"] != "").sum()),
            f"{len(pts)}/36 districts have coordinates; districts with an all-NaN nearest cell for >=1 variable: "
            f"{nan_districts.district_std.tolist()}")
    stats = dict(pts=len(pts), rain=dict(n_nan=rain["n_nan"], n_bad=rain["n_bad"], nonday=rain["nonday_steps"], shape=rain["arr"].shape),
                 tmin=dict(n_nan=tmin["n_nan"], n_bad=tmin["n_bad"], shape=tmin["arr"].shape),
                 tmax=dict(n_nan=tmax["n_nan"], n_bad=tmax["n_bad"], shape=tmax["arr"].shape),
                 date_min=str(dates.min().date()), date_max=str(dates.max().date()), n_dates=len(dates),
                 rh=rh_info, nan_districts=nan_districts.district_std.tolist(),
                 nc_attrs={"rain": rain["attrs"], "tmin": tmin["attrs"], "tmax": tmax["attrs"]},
                 grid_res={"rain": float(np.diff(rain["lat"]).mean()), "tmin": float(np.diff(tmin["lat"]).mean())})
    return cl, rh_rej, cell, daily_rh, stats
