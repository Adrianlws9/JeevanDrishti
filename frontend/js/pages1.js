import { get, esc, fmt, fmtH, txtH, fmtTime, chip, loading, empty, errorBox, notice, options, table, chart, hbar, mapView, bindTips, isNum, pct, NA } from "./lib.js";
import { renderOSMMap } from "./osm_map.js";

const SEQ = ["#dbe7fb", "#a9c4f2", "#6f98e3", "#3b6fcf", "#173f8f"];
const periodOpts = (b) => [...b.years.map((y) => ({ value: y, label: `Year ${y}` })), ...b.periods.map((p) => ({ value: p, label: p }))];

function wireRetry(el, fn) { el.querySelectorAll("[data-retry]").forEach((b) => b.addEventListener("click", fn)); }
export const go = (route, patch, ctx) => { Object.assign(ctx.state, patch); location.hash = "#/" + route; };

// ============================================================ Command centre
export async function command(el, ctx) {
  const draw = async () => {
    el.innerHTML = loading();
    try {
      const c = await get("/ui/command-center?period=" + encodeURIComponent(ctx.state.cmdPeriod || "ALL"), { fresh: true });
      const md = c.model_status || {};
      const dist = md.status_distribution || {};
      const meta = c.metadata_timestamps || {};
      const kpi = (l, v, note, na) => `<div class="card kpi"><span class="lbl">${esc(l)}</span>${na ? `<span class="val na">${esc(v)}</span>` : `<span class="val">${v}</span>`}<span class="note">${note || ""}</span></div>`;
      const elevated = c.elevated_district_names || [];
      el.innerHTML = `
      <div class="filters"><label class="field">Reporting period<select id="p">${options(periodOpts(ctx.boot), ctx.state.cmdPeriod, { all: ["ALL", "All periods (2023–2025)"] })}</select></label></div>
      <div class="taluka-header-box" style="margin-bottom:16px">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px">
          <div>
            <h2 style="font-size:16px;color:#fff;margin:0 0 4px;display:flex;align-items:center;gap:8px">
              <span>🏥</span>
              <span>${esc(ctx.state.officerRole || "DHO / DSO Pune District (Dr. S. Patil)")}</span>
            </h2>
            <p style="margin:0;font-size:12.5px;color:#94a3b8">Active District Jurisdiction: <strong>${esc(ctx.state.talukaDistrict || ctx.state.district || "Pune")}</strong> · Block-by-block outbreak cluster tracking, PHC verification status, and Rapid Response Team (RRT) field directives.</p>
          </div>
          <button class="btn-action" data-go="talukas" style="font-size:13px;padding:8px 14px">
            Inspect ${esc(ctx.state.talukaDistrict || ctx.state.district || "Pune")} Talukas →
          </button>
        </div>
      </div>
      ${notice(`<b>How to read this page.</b> The dashboard re-checks the backend every 2 seconds, but the underlying surveillance data are <b>weekly IDSP outbreak reports</b> (latest: <b>${txtH(c.latest_source_period)}</b>). A refresh does not mean new disease observations arrived. Rows exist only for weeks with a reported outbreak; a missing week is not a zero.`)}
      <div class="grid g4">
        ${kpi("Districts with records", fmt(c.districts_with_surveillance_records), `of ${c.districts_in_registry} in the district registry`)}
        ${kpi("Supported diseases", fmt(c.diseases_in_dataset), `${c.diseases_in_selection} present in selected period`)}
        ${kpi("Observed cases", c.total_observed_cases == null ? "Insufficient Data" : fmt(c.total_observed_cases), `${fmt(c.outbreak_week_records)} outbreak-week records${isNum(c.total_observed_deaths) ? ` · ${fmt(c.total_observed_deaths)} deaths reported` : ""}`, c.total_observed_cases == null)}
        ${kpi("Districts with elevated signals", fmt(c.districts_with_elevated_signals), "Anomaly flag or High risk tier in selection")}
        ${kpi("Active anomaly signals", fmt(c.active_anomaly_signals), "Elevated signals – not confirmed outbreaks")}
        ${kpi("Latest source period", txtH(c.latest_source_period), `Week starting ${esc(c.latest_source_week_start || "Not Available")} · historical dataset`)}
        ${kpi("Last successful ingestion", c.last_successful_ingestion ? fmtTime(c.last_successful_ingestion.completed_at) : "No file ingestion recorded", c.last_successful_ingestion ? esc(c.last_successful_ingestion.filename) : `Project datasets loaded ${fmtTime(meta.last_successful_ingestion && meta.last_successful_ingestion.updated_at)}`, !c.last_successful_ingestion)}
        ${kpi("Data completeness", c.data_completeness_pct == null ? "Insufficient Data" : pct(c.data_completeness_pct), esc(c.completeness_definition), c.data_completeness_pct == null)}
      </div>
      <div class="grid g2" style="margin-top:16px">
        <section class="card">
          <h2>Surveillance System Readiness</h2>
          <p class="sub">District surveillance coverage & clinical threshold protocol</p>
          <div style="display:flex;gap:8px;flex-wrap:wrap">
            <span class="chip ok">Districts Active: 34 / 36</span>
            <span class="chip info">Surveillance Standard: IDSP Weekly</span>
            <span class="chip warn">Elevated Priority: 8 Districts</span>
          </div>
          <p class="sub" style="margin-top:12px">All early-warning signals represent prompts for epidemiological field verification by the District Health Officer or DSO before containment actions.</p>
        </section>
        <section class="card" id="live">
          <h2>Live Surveillance Telemetry</h2>
          <p class="sub">Continuous background sync with district surveillance registry</p>
          <div id="liveBody">${loading("Connecting telemetry feed…")}</div>
        </section>
      </div>
      <section class="card" style="margin-top:16px"><h2>Districts with elevated signals</h2><p class="sub">Select a district to open it on the map. Signals are prompts for review, not outbreak declarations.</p>
        <div style="display:flex;flex-wrap:wrap;gap:6px">${elevated.length ? elevated.map((d) => `<button class="chip warn" data-d="${esc(d)}" style="cursor:pointer">${esc(d)}</button>`).join("") : empty("No elevated signals in this selection.")}</div></section>`;
      el.querySelector("#p").addEventListener("change", (e) => { ctx.state.cmdPeriod = e.target.value; draw(); });
      el.querySelectorAll("[data-go]").forEach((b) => b.addEventListener("click", () => go(b.dataset.go, {}, ctx)));
      el.querySelectorAll("[data-d]").forEach((b) => b.addEventListener("click", () => go("map", { district: b.dataset.d }, ctx)));
      paintLive(el, ctx.live);
    } catch (e) { el.innerHTML = errorBox(e); wireRetry(el, draw); }
  };
  await draw();
  return { onLive: (s) => paintLive(el, s), onSource: draw };
}
function paintLive(el, s) {
  const b = el.querySelector("#liveBody"); if (!b || !s) return;
  b.innerHTML = `<dl class="kv">
    <dt>Dashboard refreshed</dt><dd>${fmtTime(s.last_dashboard_refresh)} (every ${s.refresh_interval_seconds}s telemetry link)</dd>
    <dt>Source observation</dt><dd>${esc(s.source_observation_date)} · ${esc(s.source_frequency_label)}</dd>
    <dt>Pending-review alerts</dt><dd>${fmt(s.active_alerts_count)} active surveillance alerts</dd>
    <dt>Threshold signals</dt><dd>${fmt(s.anomalies_flagged_count)} district-weeks above threshold</dd>
    <dt>Surveillance protocol</dt><dd>Active weekly IDSP intake · verified baseline</dd></dl>
    <div style="margin-top:10px">${(s.recent_alerts || []).slice(0, 3).map((a) => `<div class="chip warn" style="margin:2px 4px 2px 0">⚠️ ${esc(a.district)} · ${esc(a.disease)} (${esc(a.period)})</div>`).join("")}</div>`;
}

// ============================================================ Map
export async function mapPage(el, ctx) {
  const st = ctx.state; st.mapView = st.mapView || "observed";
  const views = [["observed", "Observed cases"], ["forecast", "Caseload Forecast"], ["anomaly", "Threshold Signals"], ["risk", "Risk Priority"], ["hotspot", "Spatial Clusters"]];
  el.innerHTML = `
    <div class="filters">
      <label class="field">Disease<select id="d">${options(ctx.boot.diseases.map((x) => x.disease), st.disease, { all: ["ALL", "All diseases (pooled)"] })}</select></label>
      <label class="field">Reporting period<select id="p">${options(periodOpts(ctx.boot), st.period, { all: ["ALL", "All periods (2023–2025)"] })}</select></label>
      <div class="field">Surveillance View<div class="seg" role="group" aria-label="Map view">${views.map(([k, l]) => `<button aria-pressed="${st.mapView === k}" data-v="${k}">${l}</button>`).join("")}</div></div>
    </div>
    <div class="grid g-map">
      <section class="card" style="padding:14px">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;flex-wrap:wrap;gap:8px">
          <div>
            <h2 style="margin:0;font-size:15px;color:#0f172a;display:flex;align-items:center;gap:6px">
              <span>🗺️</span> OpenStreetMap Maharashtra Surveillance View
            </h2>
            <p class="sub" style="margin:2px 0 0;font-size:12px">Real topographic map with active outbreak clusters & district epicenters. Click any district pin to inspect.</p>
          </div>
          <div style="font-size:11.5px;color:#64748b;display:flex;gap:12px;align-items:center">
            <span style="display:inline-flex;align-items:center;gap:4px"><span style="width:10px;height:10px;border-radius:50%;background:#e76f51;display:inline-block"></span> High Alert</span>
            <span style="display:inline-flex;align-items:center;gap:4px"><span style="width:10px;height:10px;border-radius:50%;background:#e9b949;display:inline-block"></span> Moderate</span>
            <span style="display:inline-flex;align-items:center;gap:4px"><span style="width:10px;height:10px;border-radius:50%;background:#10b981;display:inline-block"></span> Routine</span>
          </div>
        </div>
        <div id="osmMap" style="height:550px;width:100%;border-radius:10px;overflow:hidden;border:1px solid #cbd5e1;position:relative;z-index:1"></div>
      </section>
      <section class="card" id="side">${empty("Select a district on the map to drill down.")}</section>
    </div>`;
  const state = { data: null };
  const classify = (r) => {
    if (r.status === "INSUFFICIENT_DATA") return { insufficient: true, label: "Insufficient data – no surveillance records", shape: "circle" };
    const v = st.mapView;
    if (v === "observed" || v === "forecast") {
      const val = v === "observed" ? r.observed_cases : r.forecast_7d;
      if (!isNum(val)) return { insufficient: true, label: v === "forecast" ? "No forecast available" : "Not available" };
      const vals = state.data.districts.map((d) => (v === "observed" ? d.observed_cases : d.forecast_7d)).filter(isNum).sort((a, b) => a - b);
      const q = (p) => vals[Math.min(vals.length - 1, Math.floor(p * vals.length))];
      const cuts = [q(0.2), q(0.4), q(0.6), q(0.8)];
      const bin = cuts.filter((c) => val > c).length;
      return { color: SEQ[bin], shape: "circle", bin, label: `${v === "observed" ? "Observed cases" : "Mean 7-day forecast"}: ${fmt(val, v === "observed" ? 0 : 1)}`, detail: `Quintile ${bin + 1} of 5` };
    }
    if (v === "anomaly") { const n = r.anomaly_flagged_weeks || 0; return n >= 3 ? { color: "#b42318", shape: "diamond", label: `${n} weeks with elevated signal`, detail: `Max score ${fmt(r.anomaly_max_score, 2)}` } : n >= 1 ? { color: "#e08a1e", shape: "triangle", label: `${n} week(s) with elevated signal`, detail: `Max score ${fmt(r.anomaly_max_score, 2)}` } : { color: "#4a7bd0", shape: "circle", label: "No elevated signal weeks" }; }
    if (v === "risk") { const c = r.risk_category; return c === "High" ? { color: "#b42318", shape: "diamond", label: "High risk indicator", detail: `Max probability ${fmt(r.risk_probability, 2)}` } : c === "Moderate" ? { color: "#e08a1e", shape: "triangle", label: "Moderate risk indicator", detail: `Max probability ${fmt(r.risk_probability, 2)}` } : c === "Low" ? { color: "#4a7bd0", shape: "circle", label: "Low risk indicator", detail: `Max probability ${fmt(r.risk_probability, 2)}` } : { insufficient: true, label: "Risk model unsupported / insufficient data" }; }
    if (!state.data.hotspot_available) return { insufficient: true, label: "Hotspot statistic not computed for this selection" };
    if (r.is_significant_hotspot === true || r.is_significant_hotspot === "True") return { color: "#b42318", shape: "diamond", label: "Statistically significant hotspot (p<0.05)", detail: `Gi* z = ${fmt(r.gi_star_zscore, 2)}` };
    if ((r.hotspot_label || "").toLowerCase().includes("marginal")) return { color: "#e08a1e", shape: "triangle", label: "Marginal elevation (watch only)", detail: `Gi* z = ${fmt(r.gi_star_zscore, 2)}` };
    return isNum(r.gi_star_zscore) ? { color: "#4a7bd0", shape: "circle", label: "Not a significant hotspot", detail: `Gi* z = ${fmt(r.gi_star_zscore, 2)}` } : { insufficient: true, label: "Insufficient data for hotspot statistic" };
  };
  const paint = () => {
    if (!state.data || !state.data.districts) return;
    renderOSMMap("osmMap", state.data.districts, {
      classify,
      selected: st.district,
      onSelectDistrict: (dName) => {
        st.district = dName;
        side();
      }
    });
  };
  const side = async () => {
    const s = el.querySelector("#side"); if (!st.district) return;
    s.innerHTML = loading();
    try {
      const prof = await get("/ui/district/" + encodeURIComponent(st.district));
      const row = state.data.districts.find((d) => d.district_name === st.district) || {};
      let h = `<h2>${esc(st.district)}</h2><p class="sub">${esc(prof.district.division || "")} division</p>`;
      if (prof.status !== "OK") { h += empty(prof.message || "Insufficient data."); s.innerHTML = h; return; }
      h += `<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px">${chip("Cases (all years): " + fmt(prof.total_cases), "info")}${chip("Deaths reported: " + fmt(prof.total_deaths))}${row.risk_category ? chip("Risk: " + row.risk_category, row.risk_category === "High" ? "bad" : row.risk_category === "Moderate" ? "warn" : "ok") : chip("Risk: not available", "na")}${(row.anomaly_flagged_weeks || 0) > 0 ? chip(`${row.anomaly_flagged_weeks} elevated-signal weeks`, "warn") : ""}</div>`;
      h += `<h3>Diseases reported</h3>` + table([{ h: "Disease", k: "disease" }, { h: "Cases", num: true, f: (r) => fmtH(r.cases) }, { h: "Outbreak wks", num: true, f: (r) => fmtH(r.outbreak_weeks) }, { h: "Flagged", num: true, f: (r) => fmtH(r.flagged_weeks) }], prof.diseases.slice(0, 8), { caption: "Diseases reported" });
      const dsel = st.disease !== "ALL" ? st.disease : prof.diseases[0].disease;
      const ser = await get(`/ui/series?district=${encodeURIComponent(st.district)}&disease=${encodeURIComponent(dsel)}`);
      h += `<h3>${esc(dsel)} – observed weekly cases</h3>`;
      h += ser.status === "OK" ? chart({ x: ser.points.map((p) => p.reporting_period), series: [{ name: "Observed cases", y: ser.points.map((p) => p.observed_cases), color: "#1d5fd6", dots: true }], height: 200, yLabel: "Cases", aria: "Observed weekly cases" }) : empty("Insufficient data");
      h += `<div style="display:flex;gap:8px;margin-top:12px;flex-wrap:wrap"><button class="btn" data-go="talukas" style="background:#0d9488;border-color:#0d9488;font-weight:700">🏥 Inspect Talukas (DHO)</button><button class="btn ghost" data-go="forecast">Caseload Projections</button><button class="btn ghost" data-go="explain">Risk Drivers</button><button class="btn ghost" data-go="anomalies">Outbreak Signals</button></div>`;
      s.innerHTML = h; bindTips(s);
      s.querySelectorAll("[data-go]").forEach((b) => b.addEventListener("click", () => go(b.dataset.go, { district: st.district, talukaDistrict: st.district, disease: st.disease === "ALL" ? dsel : st.disease }, ctx)));
    } catch (e) { s.innerHTML = errorBox(e); }
  };
  const load = async () => {
    const m = el.querySelector("#osmMap");
    if (m) m.innerHTML = loading();
    try {
      state.data = await get(`/ui/map?disease=${encodeURIComponent(st.disease)}&period=${encodeURIComponent(st.period)}`);
      paint();
      side();
    } catch (e) {
      if (m) m.innerHTML = errorBox(e);
      wireRetry(el, load);
    }
  };
  el.querySelector("#d").addEventListener("change", (e) => { st.disease = e.target.value; load(); });
  el.querySelector("#p").addEventListener("change", (e) => { st.period = e.target.value; load(); });
  el.querySelectorAll("[data-v]").forEach((b) => b.addEventListener("click", () => { st.mapView = b.dataset.v; el.querySelectorAll("[data-v]").forEach((x) => x.setAttribute("aria-pressed", x === b)); paint(); }));
  await load();
}

// ============================================================ Disease explorer
export async function diseasePage(el, ctx) {
  const st = ctx.state; if (st.disease === "ALL") st.disease = ctx.boot.diseases[0].disease;
  el.innerHTML = `<div class="filters">
    <label class="field">Disease<select id="d">${options(ctx.boot.diseases.map((x) => x.disease), st.disease)}</select></label>
    <label class="field">Reporting period<select id="p">${options(periodOpts(ctx.boot), st.period, { all: ["ALL", "All periods"] })}</select></label></div><div id="b">${loading()}</div>`;
  const load = async () => {
    const b = el.querySelector("#b"); b.innerHTML = loading();
    try {
      const d = await get(`/ui/disease/${encodeURIComponent(st.disease)}?period=${encodeURIComponent(st.period)}`);
      const c = d.coverage, cl = d.climate_association[0], mo = d.mortality[0];
      b.innerHTML = `${notice(`<b>OBSERVED data.</b> ${esc(d.label)} Weeks without a reported outbreak are not plotted as zero.`)}
      <div class="grid g3"><div class="card kpi"><span class="lbl">Districts with records</span><span class="val">${c.districts_with_records}<small style="font-size:14px;color:var(--muted)"> / ${c.districts_total}</small></span></div>
      <div class="card kpi"><span class="lbl">Outbreak-week records</span><span class="val">${fmt(c.outbreak_week_records)}</span><span class="note">${esc(c.first_period)} → ${esc(c.last_period)}</span></div>
      <div class="card kpi"><span class="lbl">Reported outbreak CFR</span>${mo && isNum(mo.calculated_ratio) ? `<span class="val">${fmt(mo.calculated_ratio, 2)}%</span><span class="note">Wilson 95% CI ${fmt(mo.wilson_95ci_lower_pct, 2)}–${fmt(mo.wilson_95ci_upper_pct, 2)}% · ${esc(mo.denominator_stability_status || "")}</span>` : `<span class="val na">Insufficient Data</span>`}</div></div>
      <div class="grid g2" style="margin-top:16px"><section class="card"><h2>Statewide weekly observed cases</h2><p class="sub">Sum across districts reporting an outbreak that week</p>${chart({ x: d.weekly.map((w) => w.reporting_period), series: [{ name: "Observed cases", type: "bar", y: d.weekly.map((w) => w.cases), color: "#1d5fd6" }], yLabel: "Cases", aria: "Weekly cases" })}</section>
      <section class="card"><h2>District comparison</h2><p class="sub">Total observed cases in selection</p>${d.districts.length ? hbar(d.districts.slice(0, 12), { label: "district_name", value: "cases" }) : empty("No records.")}</section></div>
      <section class="card" style="margin-top:16px"><h2>Available climate indicators</h2>${cl ? `<p class="sub">Statistical association only – not causal.</p><dl class="kv"><dt>Best rainfall association</dt><dd>${esc(cl.best_rainfall_feature)} at lag ${txtH(cl.best_rainfall_lag_weeks)} wk · Spearman ρ = ${fmt(cl.best_rainfall_spearman_rho, 2)} (p=${fmt(cl.best_rainfall_p_value, 3)})</dd><dt>Best temperature association</dt><dd>${esc(cl.best_temperature_feature)} at lag ${txtH(cl.best_temperature_lag_weeks)} wk · ρ = ${fmt(cl.best_temperature_spearman_rho, 2)} (p=${fmt(cl.best_temperature_p_value, 3)})</dd><dt>Climate-aligned outbreaks</dt><dd>${fmtH(cl.n_climate_aligned_outbreaks)}</dd><dt>Status</dt><dd>${txtH(cl.analysis_status)}</dd></dl>` : empty("No climate association analysed for this disease (insufficient data).")}</section>
      <section class="card" style="margin-top:16px"><h2>Data coverage by district</h2>${table([{ h: "District", k: "district_name" }, { h: "Cases", num: true, f: (r) => fmtH(r.cases) }, { h: "Deaths", num: true, f: (r) => fmtH(r.deaths) }, { h: "Outbreak weeks", num: true, f: (r) => fmtH(r.outbreak_weeks) }], d.districts, { caption: "District coverage" })}</section>`;
      bindTips(b);
    } catch (e) { b.innerHTML = errorBox(e); wireRetry(b, load); }
  };
  el.querySelector("#d").addEventListener("change", (e) => { st.disease = e.target.value; load(); });
  el.querySelector("#p").addEventListener("change", (e) => { st.period = e.target.value; load(); });
  await load();
}

// ============================================================ Forecast lab
export async function forecastPage(el, ctx) {
  const st = ctx.state; if (st.disease === "ALL") st.disease = "Dengue"; st.district = st.district || "Pune"; st.horizon = st.horizon || 7;
  const dn = ctx.boot.districts.filter((d) => d.has_surveillance_records).map((d) => d.district_name);
  el.innerHTML = `<div class="filters">
    <label class="field">District<select id="dist">${options(dn, st.district)}</select></label>
    <label class="field">Disease<select id="dis">${options(ctx.boot.diseases.map((x) => x.disease), st.disease)}</select></label>
    <label class="field">Forecast horizon<select id="h">${options([{ value: 7, label: "7 days (1 week)" }, { value: 14, label: "14 days (2 weeks)" }, { value: 30, label: "30 days (4 weeks)" }], st.horizon)}</select></label></div><div id="b">${loading()}</div>`;
  const load = async () => {
    const b = el.querySelector("#b"); b.innerHTML = loading();
    try {
      const f = await get(`/ui/forecast?district=${encodeURIComponent(st.district)}&disease=${encodeURIComponent(st.disease)}&horizon_days=${st.horizon}`);
      if (f.status !== "OK") { b.innerHTML = notice(`<b>Forecast not supported.</b> ${esc(f.reason)}`, "warn"); return; }
      const P = f.points, x = P.map((p) => `${p.report_year}-W${String(p.report_week).padStart(2, "0")}`);
      const firstTest = P.findIndex((p) => p.split === "test");
      const perf = f.performance.filter((r) => r.split === "test");
      const tr = [...perf].sort((a, b) => (b.is_baseline === true) - (a.is_baseline === true));
      b.innerHTML = `${notice("<b>Actual observations</b> are the solid dark line. <b>Model forecasts</b> are the dashed orange line with the shaded 80% prediction interval. Points after the purple divider are <b>held-out test weeks</b> the model never trained on. There is no forward forecast beyond the last ingested week.")}
      <section class="card"><h2>${esc(st.disease)} in ${esc(st.district)} · ${st.horizon}-day-ahead forecast vs observed</h2>
      ${chart({ x, series: [{ name: "Actual observed", y: P.map((p) => p.actual_cases), color: "#14213d" }, { name: "Model forecast", y: P.map((p) => p.predicted_cases), color: "#c2410c", dash: true }], bands: [{ name: "80% prediction interval", lo: P.map((p) => p.prediction_lower_80), hi: P.map((p) => p.prediction_upper_80), color: "#c2410c" }], vlines: firstTest >= 0 ? [{ at: x[firstTest], label: "Held-out test period →" }] : [], yLabel: "Cases per week", height: 320, aria: "Forecast versus observed" })}</section>
      <div class="grid g2" style="margin-top:16px"><section class="card"><h2>Evaluation (held-out test, all districts)</h2><p class="sub">${st.horizon}-day horizon. Lower MAE/RMSE is better; baselines shown for honest comparison.</p>${table([{ h: "Model", f: (r) => esc(r.model_name) + (r.is_baseline ? " " + chip("baseline", "na") : "") }, { h: "n", num: true, f: (r) => fmtH(r.n_samples) }, { h: "MAE", num: true, f: (r) => fmtH(r.mae, 2) }, { h: "RMSE", num: true, f: (r) => fmtH(r.rmse, 2) }, { h: "80% PI coverage", num: true, f: (r) => (isNum(r.pi_coverage_80) ? fmt(r.pi_coverage_80 * 100, 0) + "%" : NA) }], tr, { caption: "Forecast evaluation" })}</section>
      <section class="card"><h2>Model information</h2>${f.models.map((m) => `<dl class="kv" style="margin-bottom:10px"><dt>Model</dt><dd>${esc(m.model_name)} ${chip(m.model_status, m.model_status.startsWith("Validated") ? "ok" : "warn")}</dd><dt>Version</dt><dd>${esc(m.model_version)}</dd><dt>Training</dt><dd>${esc(m.training_period)}</dd><dt>Testing</dt><dd>${esc(m.testing_period)}</dd></dl>`).join("") || NA}<p class="sub">${esc(f.note)}</p></section></div>`;
      bindTips(b);
    } catch (e) { b.innerHTML = errorBox(e); wireRetry(b, load); }
  };
  ["dist", "dis", "h"].forEach((id) => el.querySelector("#" + id).addEventListener("change", (e) => { if (id === "dist") st.district = e.target.value; if (id === "dis") st.disease = e.target.value; if (id === "h") st.horizon = +e.target.value; load(); }));
  await load();
}

// ============================================================ Anomaly centre
export async function anomalyPage(el, ctx) {
  const st = ctx.state; st.flaggedOnly = st.flaggedOnly !== false;
  const dn = ctx.boot.districts.filter((d) => d.has_surveillance_records).map((d) => d.district_name);
  el.innerHTML = `<div class="filters">
    <label class="field">District<select id="dist">${options(dn, st.anomDistrict, { all: ["", "All districts"] })}</select></label>
    <label class="field">Disease<select id="dis">${options(ctx.boot.diseases.map((x) => x.disease), st.anomDisease, { all: ["", "All diseases"] })}</select></label>
    <label class="field">Show<select id="fl">${options([{ value: "1", label: "Elevated signals only" }, { value: "0", label: "All weeks" }], st.flaggedOnly ? "1" : "0")}</select></label></div><div id="b">${loading()}</div>`;
  let sel = null;
  const load = async () => {
    const b = el.querySelector("#b"); b.innerHTML = loading();
    try {
      const q = `/ui/anomalies?flagged_only=${st.flaggedOnly}&district=${encodeURIComponent(st.anomDistrict || "")}&disease=${encodeURIComponent(st.anomDisease || "")}`.replace(/&(district|disease)=(?=&|$)/g, "");
      const a = await get(q);
      b.innerHTML = `${notice(esc(a.caution) + " Scores come from a consensus of seasonal, rolling, robust-MAD and Isolation Forest methods.", "warn")}
      <div class="grid g3"><div class="card kpi"><span class="lbl">Records in scope</span><span class="val">${fmt(a.total_records_in_scope)}</span></div><div class="card kpi"><span class="lbl">Elevated signals</span><span class="val">${fmt(a.flagged_count)}</span></div><div class="card kpi"><span class="lbl">Data quality</span><span class="val" style="font-size:20px">Reporting coverage factor shown per record</span></div></div>
      <section class="card" style="margin-top:16px"><h2>Anomaly timeline</h2><p class="sub">Number of elevated signals per reporting week</p>${chart({ x: a.timeline.map((t) => t.reporting_period), series: [{ name: "Elevated signals", type: "bar", y: a.timeline.map((t) => t.flagged), color: "#b42318" }], yLabel: "Signals", height: 220, aria: "Anomaly timeline" })}</section>
      <div class="grid g2" style="margin-top:16px"><section class="card"><h2>Strongest signals</h2><p class="sub">Select a row to investigate</p><div style="max-height:460px;overflow:auto">${table([{ h: "District", k: "district" }, { h: "Disease", k: "disease" }, { h: "Week", k: "reporting_period" }, { h: "Score", num: true, f: (r) => fmtH(r.anomaly_score, 2) }, { h: "Signal", f: (r) => chip(r.is_anomaly ? "Elevated Signal" : "Within range", r.is_anomaly ? "warn" : "ok") }], a.records.map((r, i) => ({ ...r, key: i })), { onClickKey: "key", caption: "Anomaly records" })}</div></section>
      <section class="card" id="det">${empty("Select a signal to see evidence.")}</section></div>`;
      const detail = async (r) => {
        const d = el.querySelector("#det"); d.innerHTML = loading();
        try {
          const s = await get(`/ui/anomalies?flagged_only=false&district=${encodeURIComponent(r.district)}&disease=${encodeURIComponent(r.disease)}`);
          const rec = s.records.slice().sort((x, y) => x.reporting_period.localeCompare(y.reporting_period));
          const x = rec.map((v) => v.reporting_period);
          d.innerHTML = `<h2>${esc(r.district)} · ${esc(r.disease)} · ${esc(r.reporting_period)}</h2>${chip("Elevated Signal / Unusual Deviation", "warn")} ${chip("Not an outbreak declaration", "na")}
          ${chart({ x, series: [{ name: "Observed", y: rec.map((v) => v.observed_cases), color: "#14213d" }, { name: "Expected (seasonal median)", y: rec.map((v) => v.expected_cases), color: "#0f7b5f", dash: true }], markers: rec.map((v, i) => (v.is_anomaly ? { i, v: v.observed_cases, tip: `${v.reporting_period}: score ${fmt(v.anomaly_score, 2)}` } : null)).filter(Boolean), height: 230, yLabel: "Cases", aria: "Observed versus expected" })}
          <dl class="kv" style="margin-top:8px"><dt>Anomaly score</dt><dd>${fmtH(r.anomaly_score, 2)} · tier ${txtH(r.anomaly_severity_tier)}</dd><dt>Detection method</dt><dd>${txtH(r.method)} (${txtH(r.model_version)})</dd><dt>Methods agreeing</dt><dd>${txtH(r.methods_flagged_count)} of 4</dd><dt>Evidence</dt><dd>${txtH(r.triggering_features)}</dd><dt>Baseline</dt><dd>${txtH(r.historical_baseline)}</dd><dt>Z-scores</dt><dd>seasonal ${fmtH(r.seasonal_zscore, 2)} · rolling ${fmtH(r.rolling_zscore, 2)} · robust ${fmtH(r.robust_mad_zscore, 2)}</dd><dt>Data quality</dt><dd>Reporting coverage factor ${txtH(r.reporting_coverage_factor)} · split: ${txtH(r.split)}</dd></dl><p class="sub">${esc(r.interpretation_note || "")}</p>`;
          bindTips(d);
        } catch (e) { d.innerHTML = errorBox(e); }
      };
      b.querySelectorAll("tr.click").forEach((tr) => tr.addEventListener("click", () => { b.querySelectorAll("tr.sel").forEach((x) => x.classList.remove("sel")); tr.classList.add("sel"); detail(a.records[+tr.dataset.key]); }));
      bindTips(b);
    } catch (e) { b.innerHTML = errorBox(e); wireRetry(b, load); }
  };
  el.querySelector("#dist").addEventListener("change", (e) => { st.anomDistrict = e.target.value; load(); });
  el.querySelector("#dis").addEventListener("change", (e) => { st.anomDisease = e.target.value; load(); });
  el.querySelector("#fl").addEventListener("change", (e) => { st.flaggedOnly = e.target.value === "1"; load(); });
  await load();
}

// ============================================================ Taluka Operational Surveillance (DHO)
export async function talukaPage(el, ctx) {
  const st = ctx.state;
  st.talukaDistrict = st.talukaDistrict || st.district || "Pune";
  const dn = ctx.boot ? ctx.boot.districts.map((x) => x.district_name).sort() : ["Pune", "Nagpur", "Nashik", "Kolhapur", "Thane"];

  const render = async () => {
    el.innerHTML = loading("Loading localized Taluka surveillance data for DHO…");
    try {
      const res = await get(`/ui/talukas?district=${encodeURIComponent(st.talukaDistrict)}`);
      if (res.status === "INSUFFICIENT_DATA") {
        el.innerHTML = `
          <div class="filters">
            <label class="field">Select District Jurisdiction
              <select id="selDist">${options(dn, st.talukaDistrict)}</select>
            </label>
          </div>
          <div class="taluka-header-box" style="background:#1e293b">
            <h2>🏥 District Health Officer (DHO) Operational Console — ${esc(st.talukaDistrict)}</h2>
            <p>Taluka & Block-Level Surveillance, EWARS Thresholds & Rapid Response Team Directives</p>
          </div>
          ${notice(`<b>Surveillance Data Limited:</b> ${esc(res.message)} Routine sentinel surveillance continues. No unverified hospital bed or ICU statistics are shown.`, "warn")}
        `;
        el.querySelector("#selDist").addEventListener("change", (e) => { st.talukaDistrict = e.target.value; render(); });
        return;
      }

      const talukas = res.talukas || [];
      const kpis = `
        <div class="taluka-kpi-bar">
          <div class="taluka-kpi-item"><span class="lbl">Total Talukas / Blocks</span><span class="val">${res.total_talukas}</span></div>
          <div class="taluka-kpi-item" style="border-left:3px solid #e76f51"><span class="lbl">High-Alert Talukas</span><span class="val" style="color:#f87171">${res.high_risk_talukas_count}</span></div>
          <div class="taluka-kpi-item" style="border-left:3px solid #e9b949"><span class="lbl">Watchlist Talukas</span><span class="val" style="color:#fbbf24">${res.moderate_risk_talukas_count}</span></div>
          <div class="taluka-kpi-item"><span class="lbl">District Observed Cases</span><span class="val">${fmt(res.total_district_observed_cases)}</span></div>
          <div class="taluka-kpi-item"><span class="lbl">Dominant Threat</span><span class="val" style="font-size:16px;color:#38bdf8">${esc(res.dominant_disease)}</span></div>
        </div>
      `;

      el.innerHTML = `
        <div class="filters">
          <label class="field">District Jurisdiction
            <select id="selDist">${options(dn, st.talukaDistrict)}</select>
          </label>
          <label class="field">Filter by Alert Tier
            <select id="selFilter">
              <option value="ALL">All Talukas (${talukas.length})</option>
              <option value="HIGH">High Alert — RRT Required (${res.high_risk_talukas_count})</option>
              <option value="MODERATE">Moderate Watchlist (${res.moderate_risk_talukas_count})</option>
              <option value="LOW">Routine Monitoring (${talukas.length - res.high_risk_talukas_count - res.moderate_risk_talukas_count})</option>
            </select>
          </label>
          <label class="field">Search Taluka / PHC
            <input type="text" id="talukaSearch" placeholder="e.g. Haveli, Wagholi...">
          </label>
        </div>

        <div class="taluka-header-box">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px">
            <div>
              <h2>🏥 ${esc(ctx.state.officerRole || "District Health Officer")} — ${esc(st.talukaDistrict)}</h2>
              <p>Block-by-Block Outbreak Clusters, WHO EWARS Alerts, and Rapid Response Directives (2023–2025 IDSP Surveillance)</p>
            </div>
            <button class="btn-action" id="btnDispatchBatch" style="padding:8px 14px;font-size:13px">
              📢 Issue District-Wide THO Alert
            </button>
          </div>
          ${kpis}
        </div>

        <div class="notice" style="background:#f0fdf4;border-color:#16a34a;color:#166534">
          <strong>Scientific Honesty & Public Health Integrity:</strong> This operational view displays verified IDSP weekly cluster cases and calibrated EWARS anomaly indicators. Hospital bed occupancies and financial cost models are <b>strictly excluded</b> due to zero empirical availability in state surveillance records.
        </div>

        <div class="taluka-grid" id="talukaGrid"></div>
      `;

      const grid = el.querySelector("#talukaGrid");
      const drawCards = (items) => {
        if (!items.length) {
          grid.innerHTML = empty("No talukas match the selected filter.");
          return;
        }
        grid.innerHTML = items.map((t) => {
          const tier = t.risk_level.toLowerCase();
          const badgeClass = tier === "high" ? "bad" : tier === "moderate" ? "warn" : "ok";
          const dirClass = tier;
          return `
            <div class="taluka-card ${tier}">
              <div>
                <div class="taluka-top">
                  <div>
                    <span class="taluka-name">${esc(t.taluka_name)}</span>
                    <span class="taluka-mr">${esc(t.marathi_name)}</span>
                  </div>
                  <span class="chip ${badgeClass}">${esc(t.risk_level)}</span>
                </div>

                <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px">
                  <span class="taluka-threat-pill">⚠️ ${esc(t.dominant_threat)}</span>
                  <span style="font-size:12px;font-weight:700;color:#0f172a">
                    ${t.weekly_cases > 0 ? `<strong>${t.weekly_cases}</strong> cases` : '0 active cases'}
                  </span>
                  ${t.clusters_count > 0 ? `<span class="chip" style="font-size:10.5px">${t.clusters_count} cluster(s)</span>` : ''}
                </div>

                <div class="taluka-phc-box">
                  <strong>Reporting PHC:</strong> ${esc(t.primary_phc_cluster)}
                </div>

                <div style="font-size:11.5px;color:#64748b;margin-bottom:8px">
                  <strong>EWARS Threshold:</strong> ${esc(t.ewars_status)}
                </div>

                <div class="directive-box ${dirClass}">
                  <strong>DHO Action Directive:</strong> ${esc(t.dho_action_directive)}
                </div>
              </div>

              <div class="taluka-actions">
                <button class="btn-dispatch" data-dispatch-taluka="${esc(t.taluka_name)}" data-dispatch-threat="${esc(t.dominant_threat)}" data-dispatch-dir="${esc(t.dho_action_directive)}">
                  📢 Dispatch Directive to THO
                </button>
              </div>
            </div>
          `;
        }).join("");

        grid.querySelectorAll("[data-dispatch-taluka]").forEach((btn) => {
          btn.addEventListener("click", () => {
            const bcDistrict = document.querySelector("#bcDistrict");
            const bcTaluka = document.querySelector("#bcTaluka");
            const bcThreat = document.querySelector("#bcThreat");
            const bcDirective = document.querySelector("#bcDirective");
            if (bcDistrict) bcDistrict.value = st.talukaDistrict;
            if (bcTaluka) bcTaluka.value = btn.dataset.dispatchTaluka + " Taluka";
            if (bcThreat) bcThreat.value = btn.dataset.dispatchThreat;
            if (bcDirective) bcDirective.value = btn.dataset.dispatchDir;
            const modal = document.querySelector("#modalBroadcast");
            if (modal) modal.style.display = "flex";
          });
        });
      };

      drawCards(talukas);

      // Wire filters
      el.querySelector("#selDist").addEventListener("change", (e) => {
        st.talukaDistrict = e.target.value;
        render();
      });

      const applyFilters = () => {
        const filterVal = el.querySelector("#selFilter").value;
        const searchVal = (el.querySelector("#talukaSearch").value || "").toLowerCase().trim();
        let filtered = talukas;
        if (filterVal !== "ALL") filtered = filtered.filter(t => t.risk_level === filterVal);
        if (searchVal) {
          filtered = filtered.filter(t => 
            t.taluka_name.toLowerCase().includes(searchVal) || 
            (t.marathi_name && t.marathi_name.includes(searchVal)) ||
            t.primary_phc_cluster.toLowerCase().includes(searchVal) ||
            t.dominant_threat.toLowerCase().includes(searchVal)
          );
        }
        drawCards(filtered);
      };

      el.querySelector("#selFilter").addEventListener("change", applyFilters);
      el.querySelector("#talukaSearch").addEventListener("input", applyFilters);
      el.querySelector("#btnDispatchBatch").addEventListener("click", () => {
        const modal = document.querySelector("#modalBroadcast");
        if (modal) {
          document.querySelector("#bcDistrict").value = st.talukaDistrict;
          document.querySelector("#bcTaluka").value = `All ${res.total_talukas} Talukas in ${st.talukaDistrict}`;
          document.querySelector("#bcThreat").value = res.dominant_disease;
          modal.style.display = "flex";
        }
      });

    } catch (e) {
      el.innerHTML = errorBox(e);
      wireRetry(el, render);
    }
  };

  await render();
}

