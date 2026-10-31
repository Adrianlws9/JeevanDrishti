import { get, esc, fmt, fmtH, txtH, fmtTime, chip, loading, empty, errorBox, notice, options, table, chart, hbar, mapView, bindTips, isNum, pct, NA } from "./lib.js";
const retry = (el, fn) => el.querySelectorAll("[data-retry]").forEach((b) => b.addEventListener("click", fn));
const COLORS = ["#1d5fd6", "#c2410c", "#0f7b5f", "#7c3aed", "#475569", "#b45309"];
const activeDistricts = (ctx) => ctx.boot.districts.filter((d) => d.has_surveillance_records).map((d) => d.district_name);

// ============================================================ Climate
export async function climatePage(el, ctx) {
  const st = ctx.state; if (st.disease === "ALL") st.disease = "Dengue"; st.district = st.district || "Pune";
  el.innerHTML = `<div class="filters"><label class="field">District<select id="dist">${options(activeDistricts(ctx), st.district)}</select></label>
  <label class="field">Disease<select id="dis">${options(ctx.boot.diseases.map((x) => x.disease), st.disease)}</select></label></div><div id="b">${loading()}</div>`;
  const load = async () => {
    const b = el.querySelector("#b"); b.innerHTML = loading();
    try {
      const c = await get(`/ui/climate?district=${encodeURIComponent(st.district)}&disease=${encodeURIComponent(st.disease)}`);
      const cs = c.climate_series, x = cs.map((r) => `${r.report_year}-W${String(r.report_week).padStart(2, "0")}`);
      const rh = cs.map((r) => r.rh_mean_t), hasRH = rh.some(isNum);
      const groups = {};
      c.lags.forEach((r) => { (groups[r.climate_group] = groups[r.climate_group] || {}); const k = r.climate_variable; (groups[r.climate_group][k] = groups[r.climate_group][k] || []).push(r); });
      const lagSeries = Object.entries(groups).map(([g, vars], i) => { const best = Object.values(vars).sort((a, b) => Math.max(...b.map((r) => Math.abs(r.spearman_rho || 0))) - Math.max(...a.map((r) => Math.abs(r.spearman_rho || 0))))[0].sort((a, b) => a.lag_weeks - b.lag_weeks); return { name: `${g} (${best[0].climate_variable})`, lags: best, color: COLORS[i % COLORS.length] }; });
      const lagX = [...new Set(c.lags.map((r) => r.lag_weeks))].sort((a, b) => a - b);
      const corr = c.correlations;
      b.innerHTML = `${notice(esc(c.association_warning) + " Lagged charts show Spearman correlation between past climate and cases; they are <b>statistical associations</b>.", "warn")}
      ${cs.length ? "" : notice("No daily/weekly climate series exist for this district (insufficient data).", "warn")}
      <div class="grid g2"><section class="card"><h2>Rainfall · ${esc(st.district)}</h2><p class="sub">Weekly accumulated rainfall (mm)</p>${chart({ x, series: [{ name: "Rainfall (mm/week)", type: "bar", y: cs.map((r) => r.rain_mm_sum_t), color: "#1d5fd6" }], yLabel: "mm", height: 230 })}</section>
      <section class="card"><h2>Temperature · ${esc(st.district)}</h2><p class="sub">Weekly mean temperature (°C)</p>${chart({ x, series: [{ name: "Mean temperature (°C)", y: cs.map((r) => r.tavg_c_mean_t), color: "#c2410c", dots: false }], yLabel: "°C", height: 230, yMin: 10 })}</section>
      <section class="card"><h2>Humidity · ${esc(st.district)}</h2>${hasRH ? chart({ x, series: [{ name: "Mean relative humidity (%)", y: rh, color: "#0f7b5f", dots: false }], yLabel: "%", height: 230, yMin: 0 }) : empty("Humidity is Not Available for this district (a station dataset exists only for a few locations).")}<p class="sub">${esc(c.humidity_note)}</p></section>
      <section class="card"><h2>Disease comparison · ${esc(st.disease)}</h2><p class="sub">Observed outbreak-week cases in ${esc(st.district)} (gaps = no reported outbreak)</p>${c.case_series.length ? chart({ x: c.case_series.map((r) => r.canonical_week_start), series: [{ name: "Observed cases", type: "bar", y: c.case_series.map((r) => r.observed_cases), color: "#14213d" }], yLabel: "Cases", height: 230 }) : empty("No observed records for this district-disease pair.")}</section></div>
      <section class="card" style="margin-top:16px"><h2>Lagged relationship · ${esc(st.disease)} (all districts)</h2><p class="sub">Spearman ρ between climate at the given lag and weekly cases. Values near 0 mean weak association.</p>
      ${lagSeries.length ? chart({ x: lagX.map((l) => `${l} wk`), series: lagSeries.map((s) => ({ name: s.name, color: s.color, y: lagX.map((l) => { const r = s.lags.find((q) => q.lag_weeks === l); return r ? r.spearman_rho : null; }) })), yLabel: "Spearman ρ", height: 260, yMin: -0.6, aria: "Lagged climate association" }) : empty("No lag analysis available for this disease (insufficient aligned data).")}</section>
      <section class="card" style="margin-top:16px"><h2>Disease–climate associations</h2>${table([{ h: "Disease", k: "disease_std" }, { h: "Aligned outbreaks", num: true, f: (r) => fmtH(r.n_climate_aligned_outbreaks) }, { h: "Rainfall (lag, ρ)", f: (r) => (isNum(r.best_rainfall_spearman_rho) ? `${fmt(r.best_rainfall_lag_weeks)} wk, ${fmt(r.best_rainfall_spearman_rho, 2)}` : NA) }, { h: "Temperature (lag, ρ)", f: (r) => (isNum(r.best_temperature_spearman_rho) ? `${fmt(r.best_temperature_lag_weeks)} wk, ${fmt(r.best_temperature_spearman_rho, 2)}` : NA) }, { h: "Status", k: "analysis_status" }], corr, { caption: "Climate associations" })}</section>`;
      bindTips(b);
    } catch (e) { b.innerHTML = errorBox(e); retry(b, load); }
  };
  ["dist", "dis"].forEach((id) => el.querySelector("#" + id).addEventListener("change", (e) => { st[id === "dist" ? "district" : "disease"] = e.target.value; load(); }));
  await load();
}

// ============================================================ Hotspots
export async function hotspotPage(el, ctx) {
  const st = ctx.state; st.hsDisease = st.hsDisease || "ALL_DISEASES"; st.hsPeriod = st.hsPeriod || "2023-2025_POOL";
  const first = await get(`/ui/hotspots?disease=${encodeURIComponent(st.hsDisease)}&period=${encodeURIComponent(st.hsPeriod)}`).catch((e) => { el.innerHTML = errorBox(e); return null; });
  if (!first) return;
  el.innerHTML = `<div class="filters"><label class="field">Disease<select id="d">${options(first.available_diseases, st.hsDisease)}</select></label>
  <label class="field">Period<select id="p">${options(first.available_periods, st.hsPeriod)}</select></label></div><div id="b"></div>`;
  const load = async () => {
    const b = el.querySelector("#b"); b.innerHTML = loading();
    try {
      const h = await get(`/ui/hotspots?disease=${encodeURIComponent(st.hsDisease)}&period=${encodeURIComponent(st.hsPeriod)}`);
      const rows = ctx.boot.districts.map((d) => ({ ...d, h: h.districts.find((x) => x.district_name === d.district_name) }));
      const classify = (r) => { const q = r.h; if (!q || !isNum(q.gi_star_zscore)) return { insufficient: true, label: "Insufficient data" }; if (q.is_statistically_significant_hotspot === true) return { color: "#b42318", shape: "diamond", label: "Significant hotspot (p<0.05)", detail: `Gi* z=${fmt(q.gi_star_zscore, 2)}` }; if ((q.hotspot_consensus_label || "").toLowerCase().includes("marginal")) return { color: "#e08a1e", shape: "triangle", label: "Marginal (watch only)", detail: `Gi* z=${fmt(q.gi_star_zscore, 2)}` }; return { color: "#4a7bd0", shape: "circle", label: "Not significant", detail: `Gi* z=${fmt(q.gi_star_zscore, 2)}` }; };
      const cov = h.districts.filter((q) => isNum(q.gi_star_zscore)).length;
      b.innerHTML = `${notice(esc(h.caution))}
      <div class="grid g-map"><section class="card"><h2>${esc(st.hsDisease.replace("_", " "))} · ${esc(st.hsPeriod)}</h2><p class="sub">Getis-Ord Gi* local statistics (permutation p-values)</p>${mapView(rows, { classify, labels: true, legend: [{ color: "#4a7bd0", label: "Not significant" }, { color: "#e08a1e", shape: "triangle", label: "Marginal (watch)" }, { color: "#b42318", shape: "diamond", label: "Significant (p<0.05)" }, { insufficient: true, label: "Insufficient data" }] })}</section>
      <section class="card"><h2>Global spatial autocorrelation</h2><p class="sub">Geographic coverage: ${cov} of ${ctx.boot.districts.length} districts have a statistic.</p>${table([{ h: "Indicator", k: "indicator" }, { h: "Moran's I", num: true, f: (r) => fmtH(r.morans_i, 3) }, { h: "p (perm.)", num: true, f: (r) => fmtH(r.p_value_permutation, 3) }, { h: "Pattern", f: (r) => chip(r.is_significant_p05 ? "Significant" : "Not significant", r.is_significant_p05 ? "warn" : "") }], h.global_statistics, { caption: "Global statistics" })}
      <h3>Spatial clusters</h3>${h.clusters.length ? h.clusters.map((c) => `<div class="chip info" style="margin:2px">${esc(c.member_districts)}</div>`).join("") : empty("No clusters computed.")}</section></div>
      <section class="card" style="margin-top:16px"><h2>District-level comparison</h2><div style="max-height:420px;overflow:auto">${table([{ h: "District", k: "district_name" }, { h: "Cases", num: true, f: (r) => fmtH(r.observed_cases) }, { h: "Gi* z", num: true, f: (r) => fmtH(r.gi_star_zscore, 2) }, { h: "p (perm.)", num: true, f: (r) => fmtH(r.gi_star_p_perm, 3) }, { h: "Local Moran", f: (r) => txtH(r.lisa_quadrant) }, { h: "Classification", f: (r) => txtH(r.hotspot_consensus_label) }], h.districts, { caption: "District hotspot statistics" })}</div></section>`;
      bindTips(b);
    } catch (e) { b.innerHTML = errorBox(e); retry(b, load); }
  };
  el.querySelector("#d").addEventListener("change", (e) => { st.hsDisease = e.target.value; load(); });
  el.querySelector("#p").addEventListener("change", (e) => { st.hsPeriod = e.target.value; load(); });
  await load();
}

// ============================================================ Scenario lab
export async function scenarioPage(el, ctx) {
  const st = ctx.state; if (st.disease === "ALL") st.disease = "Dengue"; st.district = st.district || "Pune";
  const p = st.sc || (st.sc = { weeks: 12, g: 1.05, damp: 0.25, start: 3, ramp: 2 });
  el.innerHTML = `${notice("<b>SIMULATION.</b> Everything on this page is a hypothetical what-if. Growth and effect sizes are <b>your assumptions</b>; no validated intervention-effectiveness data exist in the project. Results are never observations.", "sim")}
  <section class="card"><h2>Interactive scenario calculator</h2><p class="sub">Transparent formula: baseline cases grow by the weekly multiplier from the mean of the last observed weeks; the intervention reduces growth above 1.0 by the dampening share after the start week (linear ramp).</p>
  <div class="filters"><label class="field">District<select id="dist">${options(activeDistricts(ctx), st.district)}</select></label><label class="field">Disease<select id="dis">${options(ctx.boot.diseases.map((x) => x.disease), st.disease)}</select></label></div>
  <div class="grid g3">${[["g", "Weekly growth multiplier", 0.9, 1.4, 0.01], ["damp", "Assumed transmission dampening", 0, 0.9, 0.05], ["start", "Intervention start week", 1, 20, 1], ["ramp", "Ramp-up weeks", 0, 8, 1], ["weeks", "Horizon (weeks)", 4, 26, 1]].map(([k, l, mn, mx, s]) => `<label class="field">${l}: <b id="v_${k}">${p[k]}</b><input type="range" id="r_${k}" min="${mn}" max="${mx}" step="${s}" value="${p[k]}"></label>`).join("")}</div>
  <div id="calc" style="margin-top:14px">${loading()}</div></section>
  <section class="card" style="margin-top:16px"><h2>Pre-computed scenario catalogue</h2><p class="sub">Scenarios generated by the Phase 14 simulator. Select one to compare its baseline and hypothetical trajectory.</p><div id="cat">${loading()}</div><div id="traj" style="margin-top:14px"></div></section>`;
  let timer;
  const calc = async () => {
    const c = el.querySelector("#calc");
    try {
      const q = new URLSearchParams({ district: st.district, disease: st.disease, weeks: p.weeks, weekly_growth: p.g, dampening: p.damp, start_week: p.start, ramp_weeks: p.ramp });
      const r = await get("/ui/scenario-calc?" + q, { fresh: true });
      if (r.status !== "OK") { c.innerHTML = empty(r.reason); return; }
      c.innerHTML = `<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px">${chip("SIMULATED – not observed", "bad")}${chip("Anchor: " + fmt(r.anchor_mean_weekly_cases, 1) + " cases/week", "info")}${chip("Hypothetical cases averted: " + fmt(r.hypothetical_difference, 0), "ok")}</div>
      ${chart({ x: r.weeks.map((w) => "Wk " + w.week), series: [{ name: "Baseline (no intervention, simulated)", y: r.weeks.map((w) => w.baseline), color: "#475569", dash: true }, { name: "With intervention (simulated)", y: r.weeks.map((w) => w.scenario), color: "#7c3aed" }], yLabel: "Cases per week (simulated)", height: 260, aria: "Scenario comparison" })}
      <p class="sub">${esc(r.anchor_definition)} ${esc(r.disclaimer)}</p>`;
      bindTips(c);
    } catch (e) { c.innerHTML = errorBox(e); }
  };
  const sched = () => { clearTimeout(timer); timer = setTimeout(calc, 250); };
  ["g", "damp", "start", "ramp", "weeks"].forEach((k) => el.querySelector("#r_" + k).addEventListener("input", (e) => { p[k] = +e.target.value; el.querySelector("#v_" + k).textContent = p[k]; sched(); }));
  el.querySelector("#dist").addEventListener("change", (e) => { st.district = e.target.value; calc(); });
  el.querySelector("#dis").addEventListener("change", (e) => { st.disease = e.target.value; calc(); });
  calc();
  try {
    const s = await get("/ui/scenarios");
    el.querySelector("#cat").innerHTML = table([{ h: "Scenario", f: (r) => esc(r.scenario_name) }, { h: "Disease", k: "selected_disease" }, { h: "Weeks", num: true, f: (r) => fmtH(r.scenario_duration_weeks) }, { h: "Baseline cum.", num: true, f: (r) => fmtH(r.unmitigated_baseline_cumulative_cases) }, { h: "Scenario cum.", num: true, f: (r) => fmtH(r.scenario_hypothetical_cumulative_cases) }, { h: "Δ %", num: true, f: (r) => fmtH(r.hypothetical_relative_difference_vs_baseline_pct, 1) }], s.catalog, { onClickKey: "scenario_id", caption: "Scenario catalogue" });
    el.querySelectorAll("#cat tr.click").forEach((tr) => tr.addEventListener("click", async () => {
      el.querySelectorAll("#cat tr.sel").forEach((x) => x.classList.remove("sel")); tr.classList.add("sel");
      const t = el.querySelector("#traj"); t.innerHTML = loading();
      try { const j = await get(`/ui/scenarios/${encodeURIComponent(tr.dataset.key)}/trajectory`); t.innerHTML = `${chip("SIMULATED", "bad")}${chart({ x: j.weeks.map((w) => "Wk " + w.simulation_week), series: [{ name: "Unmitigated baseline (simulated)", y: j.weeks.map((w) => w.baseline), color: "#475569", dash: true }, { name: "Scenario (simulated)", y: j.weeks.map((w) => w.scenario), color: "#7c3aed" }], yLabel: "Cases (simulated)", height: 240 })}`; bindTips(t); } catch (e) { t.innerHTML = errorBox(e); }
    }));
  } catch (e) { el.querySelector("#cat").innerHTML = errorBox(e); }
}

// ============================================================ Resources
export async function resourcePage(el) {
  el.innerHTML = loading();
  try {
    const r = await get("/ui/resources");
    el.innerHTML = `${notice(`<b>${r.empirical_resource_data_available ? "Resource data available." : "Resource capacity data are not available."}</b> ${esc(r.message)}`, r.empirical_resource_data_available ? "" : "warn")}
    <section class="card"><h2>Resource data completeness audit</h2><p class="sub">Every capacity variable we looked for and whether it exists in the project data.</p>${table([{ h: "Variable", k: "variable_label" }, { h: "Present?", f: (x) => chip(String(x.present_in_workspace) === "True" || x.present_in_workspace === true ? "Available" : "Not Available", String(x.present_in_workspace).toLowerCase() === "true" ? "ok" : "na") }, { h: "District coverage", k: "district_coverage" }, { h: "Missing", num: true, f: (x) => (isNum(x.missingness_pct) ? fmt(x.missingness_pct, 0) + "%" : NA) }], r.audit, { caption: "Resource audit" })}</section>
    <section class="card" style="margin-top:16px"><h2>Epidemiological demand indicators</h2><p class="sub">These are case-based indicators, not hospital capacity. Beds, ICU and medicine columns are Not Available.</p><div style="max-height:460px;overflow:auto">${table([{ h: "District", k: "district_name" }, { h: "Observed cases", num: true, f: (x) => fmtH(x.observed_2023_2025_total_outbreak_cases) }, { h: "Peak weekly", num: true, f: (x) => fmtH(x.observed_2023_2025_peak_weekly_cases) }, { h: "7-day forecast (mean)", num: true, f: (x) => fmtH(x.upstream_phase6_forecast_7d_mean_cases, 1) }, { h: "Beds", f: () => NA }, { h: "ICU", f: () => NA }, { h: "Medicine stock", f: () => NA }], r.district_demand, { caption: "District demand" })}</div></section>
    <section class="card" style="margin-top:16px"><h2>Hypothetical capacity planning</h2><p class="sub">For labelled what-if bed-day calculations use the <a href="#/scenarios">Scenario laboratory</a>. No hospital capacity is invented.</p></section>`;
  } catch (e) { el.innerHTML = errorBox(e); retry(el, () => resourcePage(el)); }
}

// ============================================================ Explainability
export async function explainPage(el, ctx) {
  const st = ctx.state; st.district = st.district || "Pune";
  el.innerHTML = `<div class="filters"><label class="field">District<select id="dist">${options(ctx.boot.districts.map((d) => d.district_name), st.district)}</select></label></div><div id="b">${loading()}</div>`;
  const load = async () => {
    const b = el.querySelector("#b"); b.innerHTML = loading();
    try {
      const r = await get("/ui/explain?district=" + encodeURIComponent(st.district));
      if (r.status !== "OK") { b.innerHTML = empty("Insufficient data – no explanation panel exists for this district."); return; }
      const p = r.panel, shap = [["Historical disease level", p.historical_disease_mean_shap], ["Recent case growth", p.case_growth_mean_shap], ["Climate", p.climate_mean_shap], ["Seasonality", p.seasonal_mean_shap]].filter((x) => isNum(x[1]));
      const mx = Math.max(...shap.map((s) => Math.abs(s[1])), 0.001);
      b.innerHTML = `<section class="card"><h2>Why was ${esc(st.district)} ${p.is_flagged ? "flagged" : "not flagged"}?</h2><div style="display:flex;gap:8px;flex-wrap:wrap;margin:6px 0 12px">${chip(p.is_flagged ? "Flagged for review" : "Not flagged", p.is_flagged ? "warn" : "ok")}${chip(p.flag_priority_tier || "Priority not available", "info")}${chip("Not an outbreak declaration", "na")}</div>
      <h3>Main supporting evidence</h3><p>${txtH(p.main_evidence)}</p>
      <h3>Relevant historical observations</h3><p>${txtH(p.historical_comparison)}</p>
      <div class="split"><div><h3>Epidemiological Risk Drivers (Relative Influence)</h3>${shap.length ? shap.map((s) => `<div style="margin:6px 0;font-size:13px">${esc(s[0])}<div class="bar"><i style="width:${(Math.abs(s[1]) / mx) * 100}%"></i></div></div>`).join("") : empty("Feature contributions Not Available.")}<p class="sub">Key contributing factors: ${txtH(p.important_contributing_features)}</p></div>
      <div><h3>Surveillance & Risk Profile</h3><dl class="kv"><dt>Surveillance status</dt><dd>Active IDSP Monitoring</dd><dt>Peak risk</dt><dd>${isNum(p.peak_risk_probability) ? fmt(p.peak_risk_probability, 2) + " (" + esc(p.peak_risk_category) + ")" : NA}</dd><dt>Flagged anomaly weeks</dt><dd>${fmtH(p.flagged_anomaly_weeks_count)}</dd><dt>Hotspot class</dt><dd>${txtH(p.geospatial_hotspot_class)}</dd><dt>7-day forecast (mean)</dt><dd>${fmtH(p.forecast_7d_mean_cases, 1)}</dd><dt>Data quality</dt><dd>${txtH(p.data_quality)}</dd></dl></div></div>
      <div class="notice warn" style="margin-top:12px"><b>Epidemiological Guidance.</b> ${txtH(p.limitations)}</div></section>`;
    } catch (e) { b.innerHTML = errorBox(e); retry(b, load); }
  };
  el.querySelector("#dist").addEventListener("change", (e) => { st.district = e.target.value; load(); });
  await load();
}

// ============================================================ Fusion
const pill = (avail, flagged, on, off) => (!avail ? chip("n/a", "na") : flagged ? chip(on, "warn") : chip(off, "ok"));
export async function fusionPage(el, ctx) {
  const st = ctx.state;
  el.innerHTML = loading();
  try {
    const f = await get("/ui/fusion");
    const rows = f.districts;
    el.innerHTML = `${notice("Cross-district epidemiological synthesis for Maharashtra. Surveillance signals, climate triggers, and spatial clustering indicators are cross-evaluated. The system never declares an outbreak automatically (" + f.automated_outbreak_declarations + " automated declarations).")}
    <div class="grid g3" style="margin-bottom:16px"><div class="card kpi"><span class="lbl">Districts monitored</span><span class="val">${rows.length}</span></div><div class="card kpi"><span class="lbl">Districts with mixed signals</span><span class="val">${rows.filter((r) => r.disagreement_flag === true).length}</span></div><div class="card kpi"><span class="lbl">Surveillance review queue</span><span class="val">${fmt(f.disagreement_queue_size)}</span></div></div>
    <div class="grid g-map"><section class="card"><h2>Pooled district intelligence (2023–2025)</h2><div style="max-height:620px;overflow:auto">${table([{ h: "District", k: "district_name" }, { h: "Risk", f: (r) => pill(r.risk_available, r.risk_flagged_elevated, "Elevated", "Not elevated") }, { h: "Forecast", f: (r) => (r.forecast_available ? chip(fmt(r.forecast_7d_cases, 1) + "/wk", "info") : chip("n/a", "na")) }, { h: "Anomaly", f: (r) => pill(r.anomaly_available, r.anomaly_flagged, "Elevated Signal", "Normal") }, { h: "Geo", f: (r) => pill(r.geospatial_available, r.is_statistically_significant_hotspot === true, "Hotspot", "No hotspot") }, { h: "Conflict", f: (r) => (r.disagreement_flag === true ? chip("Disagree", "bad") : chip("Aligned", "ok")) }], rows, { onClickKey: "district_name", selected: st.district, caption: "Fusion monitor" })}</div></section>
    <section class="card" id="det">${empty("Select a district to see the combined evidence.")}</section></div>`;
    const show = (name) => {
      const r = rows.find((x) => x.district_name === name); if (!r) return; st.district = name;
      let mv = {}; try { mv = JSON.parse(r.model_versions_json || "{}"); } catch (_) {}
      el.querySelector("#det").innerHTML = `<h2>${esc(name)}</h2><p class="sub">${esc(r.administrative_division)} · ${esc(r.overall_signal_status)}</p>
      <div class="engine" style="margin-bottom:10px">${chip("Priority: " + (r.recommended_review_priority || "n/a").replace(/_/g, " "), "info")}${chip(r.disagreement_type ? r.disagreement_type.replace(/_/g, " ") : "n/a", r.disagreement_flag ? "bad" : "ok")}</div>
      <dl class="kv"><dt>Risk classification</dt><dd>${r.risk_available ? `${esc(r.risk_category)} · probability ${fmt(r.risk_calibrated_probability, 2)}` : NA}</dd><dt>Forecast</dt><dd>${r.forecast_available ? `7d ${fmt(r.forecast_7d_cases, 1)} · 14d ${fmt(r.forecast_14d_cases, 1)} · 30d ${fmt(r.forecast_30d_cases, 1)}` : NA}</dd><dt>Anomaly</dt><dd>${r.anomaly_available ? `${r.anomaly_flagged ? "Elevated Signal" : "Normal"} · score ${fmt(r.anomaly_composite_score, 2)}` : NA}</dd><dt>Climate</dt><dd>${r.climate_available ? `Rain–case ρ(4w) ${fmt(r.district_rain_4w_spearman_rho, 2)} (association only)` : NA}</dd><dt>Geospatial</dt><dd>${r.geospatial_available ? `Gi* z ${fmt(r.gi_star_zscore, 2)} · ${esc(r.gi_star_class)}` : NA}</dd><dt>Resources</dt><dd>${r.empirical_resource_data_available ? "Available" : "Not Available (no capacity dataset)"}</dd><dt>Severity</dt><dd>${r.severity_available ? `${fmt(r.verified_outbreak_deaths)} deaths / ${fmt(r.verified_outbreak_cases)} cases · CFR ${fmt(r.outbreak_cluster_cfr_pct, 2)}%` : NA}</dd></dl>
      <h3>Supporting evidence</h3>${(r.supporting_analytical_evidence || "").split(" || ").filter(Boolean).map((t) => `<p style="margin:4px 0;font-size:13px">• ${esc(t)}</p>`).join("") || NA}
      <h3>Conflicting evidence</h3>${r.conflicting_evidence ? `<p>${esc(r.conflicting_evidence)}</p>` : '<p class="sub">No conflicting evidence recorded.</p>'}
      <h3>Overall summary</h3><p style="font-size:13px">${txtH(r.why_this_result_explanation)}</p>
      <p class="sub">Surveillance protocol: IDSP Standardized Epidemiological Registry</p><div class="notice warn">${txtH(r.uncertainty_and_limitations)}</div>`;
      el.querySelectorAll("tr.click").forEach((t) => t.classList.toggle("sel", t.dataset.key === name));
    };
    el.querySelectorAll("tr.click").forEach((t) => t.addEventListener("click", () => show(t.dataset.key)));
    if (st.district && rows.some((r) => r.district_name === st.district)) show(st.district);
  } catch (e) { el.innerHTML = errorBox(e); retry(el, () => fusionPage(el, ctx)); }
}

// ============================================================ Models
export async function modelsPage(el) {
  el.innerHTML = loading();
  try {
    const m = await get("/ui/models");
    const kind = (s) => (s === "Validated prototype" ? "ok" : s === "Unsupported" ? "bad" : "warn");
    const metrics = (em) => { if (!em || typeof em !== "object") return NA; return Object.entries(em).map(([split, v]) => `<div style="margin:4px 0"><b>${esc(split)}</b>: ${typeof v === "object" && v ? Object.entries(v).slice(0, 6).map(([k, x]) => `${esc(k)} ${isNum(x) ? fmt(x, 3) : esc(x)}`).join(" · ") : esc(v)}</div>`).join(""); };
    el.innerHTML = `<div class="notice">Registry ${esc(m.registry_version)} · generated ${fmtTime(m.generated)}. ${Object.entries(m.status_distribution).map(([k, v]) => chip(`${k}: ${v}`, kind(k))).join(" ")}</div>
    ${m.models.map((x) => `<details><summary>${esc(x.model_name)} ${chip(x.model_status, kind(x.model_status))} <span class="sub">${esc(x.model_type)} · ${esc(x.model_version)}</span></summary>
      <div class="split" style="margin-top:10px"><dl class="kv"><dt>Engine</dt><dd>${txtH(x.engine_name)}</dd><dt>Diseases</dt><dd>${txtH(x.disease)}</dd><dt>Supported districts</dt><dd>${fmtH(x.supported_districts_count)}${x.unsupported_districts && x.unsupported_districts.length ? " · unsupported: " + esc([].concat(x.unsupported_districts).join(", ")) : ""}</dd><dt>Training</dt><dd>${txtH(x.training_period)}</dd><dt>Validation</dt><dd>${txtH(x.validation_period)}</dd><dt>Testing</dt><dd>${txtH(x.testing_period)}</dd><dt>Dataset / features</dt><dd>${txtH(x.dataset_version)} / ${txtH(x.feature_version)}</dd><dt>Artifact</dt><dd>${txtH(x.artifact_path)} ${x.artifact_exists ? chip("exists", "ok") : chip("not present", "na")}</dd></dl>
      <div><h3 style="margin-top:0">Evaluation metrics</h3>${metrics(x.evaluation_metrics)}<h3>Status justification</h3><p style="font-size:13px">${txtH(x.status_justification)}</p><div class="notice warn"><b>Known limitations.</b> ${txtH(x.production_readiness_disclaimer)}</div></div></div></details>`).join("")}`;
  } catch (e) { el.innerHTML = errorBox(e); retry(el, () => modelsPage(el)); }
}

// ============================================================ Data quality & transparency
export async function qualityPage(el) {
  el.innerHTML = loading();
  try {
    const q = await get("/ui/quality", { fresh: true });
    const md = q.system_metadata || {};
    el.innerHTML = `<div class="grid g2"><section class="card"><h2>Processing status</h2><dl class="kv"><dt>Source observation</dt><dd>${txtH(md.source_observation_date && md.source_observation_date.value)} · ${txtH(md.source_frequency_label && md.source_frequency_label.value)}</dd><dt>Last ingestion / load</dt><dd>${fmtTime(md.last_successful_ingestion && md.last_successful_ingestion.value)}</dd><dt>Last processing</dt><dd>${fmtTime(md.last_processing_completion && md.last_processing_completion.value)}</dd><dt>Last prediction run</dt><dd>${fmtTime(md.last_prediction_generation && md.last_prediction_generation.value)}</dd><dt>Model version</dt><dd>${txtH(md.model_version && md.model_version.value)}</dd></dl>
      <p class="sub">${q.ingestion_history.length ? "Ingestion history below." : "No file ingestions recorded since the database was loaded from the project datasets."}</p></section>
      <section class="card"><h2>Registered sources</h2>${table([{ h: "Source", k: "source_name" }, { h: "File", k: "source_file" }, { h: "Records", num: true, f: (r) => fmtH(r.records_count) }, { h: "Span", k: "time_span" }, { h: "Status", f: (r) => chip(r.status, "ok") }], q.sources_registered, { caption: "Sources" })}</section></div>
    ${q.ingestion_history.length ? `<section class="card" style="margin-top:16px"><h2>Ingestion history</h2>${table([{ h: "File", k: "filename" }, { h: "Status", k: "status" }, { h: "Stage", k: "stage_completed" }, { h: "New rows", num: true, f: (r) => fmtH(r.new_records_inserted) }, { h: "Completed", f: (r) => fmtTime(r.completed_at) }], q.ingestion_history)}</section>` : ""}
    <section class="card" style="margin-top:16px"><h2>Dataset inventory</h2><div style="max-height:320px;overflow:auto">${table([{ h: "File", k: "filename" }, { h: "Source", k: "original_source" }, { h: "Rows", num: true, f: (r) => fmtH(r.rows) }, { h: "Role", k: "role" }, { h: "Notes", k: "notes" }], q.dataset_inventory, { caption: "Dataset inventory" })}</div></section>
    <div class="grid g2" style="margin-top:16px"><section class="card"><h2>District coverage</h2><div style="max-height:360px;overflow:auto">${table([{ h: "District", k: "district" }, { h: "2023–25 records", num: true, f: (r) => fmtH(r.cont_records) }, { h: "First week", k: "cont_first_report_week" }, { h: "Humidity station", k: "cwprs_humidity_station" }], q.district_coverage, { caption: "District coverage" })}</div></section>
    <section class="card"><h2>Missing-data information</h2><p class="sub">Columns with missing values (missing is never treated as zero).</p><div style="max-height:360px;overflow:auto">${table([{ h: "Dataset", k: "dataset" }, { h: "Column", k: "column" }, { h: "Missing", num: true, f: (r) => fmtH(r.n_missing) }, { h: "%", num: true, f: (r) => fmtH(r.pct_missing, 1) }], q.missing_data, { caption: "Missing data" })}</div></section></div>
    <section class="card" style="margin-top:16px"><h2>Outcome-variable availability</h2><div style="max-height:300px;overflow:auto">${table(q.outcome_variable_availability.length ? Object.keys(q.outcome_variable_availability[0]).slice(0, 5).map((k) => ({ h: k.replace(/_/g, " "), k })) : [], q.outcome_variable_availability)}</div></section>
    <section class="card" style="margin-top:16px"><h2>Model lineage</h2><div style="max-height:300px;overflow:auto">${table([{ h: "Model", k: "model_name" }, { h: "Dataset version", k: "dataset_version" }, { h: "Datasets", f: (r) => esc([].concat(r.datasets || []).join(", ")) }], q.model_lineage, { caption: "Model lineage" })}</div></section>
    <section class="card" style="margin-top:16px"><h2>Preprocessing steps</h2><div style="max-height:300px;overflow:auto">${table([{ h: "#", k: "step_id" }, { h: "Dataset", k: "dataset" }, { h: "Step", k: "step" }, { h: "Action", k: "action" }, { h: "Rows before→after", f: (r) => `${fmt(r.rows_before)} → ${fmt(r.rows_after)}` }], q.preprocessing_steps, { caption: "Preprocessing" })}</div></section>`;
  } catch (e) { el.innerHTML = errorBox(e); retry(el, () => qualityPage(el)); }
}
