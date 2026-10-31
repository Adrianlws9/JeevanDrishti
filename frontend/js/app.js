import { get, esc, fmtTime, errorBox, loading, clearCache } from "./lib.js";
import * as A from "./pages1.js";
import * as B from "./pages2.js";

const ICON = (d) => `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${d}</svg>`;
const ROUTES = [
  { grp: "Command & Taluka Operations" },
  { id: "", title: "Intelligence Command Centre", nav: "Command Centre", icon: '<rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/>', fn: A.command },
  { id: "talukas", title: "Taluka Operational Surveillance Console (DHO)", nav: "Taluka Surveillance", icon: '<path d="M3 21h18M5 21V7l8-4v18M19 21V11l-6-4"/>', fn: A.talukaPage },
  { id: "map", title: "Maharashtra Disease Intelligence Map", nav: "Disease Map", icon: '<path d="M12 21s7-6.2 7-11a7 7 0 0 0-14 0c0 4.8 7 11 7 11z"/><circle cx="12" cy="10" r="2.5"/>', fn: A.mapPage },
  { grp: "Surveillance & Forecasting" },
  { id: "disease", title: "Disease Explorer", nav: "Disease Explorer", icon: '<path d="M4 19V5M4 19h16M8 15l3-4 3 2 4-6"/>', fn: A.diseasePage },
  { id: "forecast", title: "Epidemic Caseload Projections", nav: "Caseload Projections", icon: '<path d="M3 17l5-6 4 3 8-9"/><path d="M15 5h5v5"/>', fn: A.forecastPage },
  { id: "anomalies", title: "Outbreak Alerts & Threshold Signals", nav: "Outbreak Alerts", icon: '<path d="M12 3l9 16H3z"/><path d="M12 10v4M12 17h.01"/>', fn: A.anomalyPage },
  { id: "climate", title: "Environmental Risk Triggers", nav: "Environmental Triggers", icon: '<path d="M7 18a4 4 0 0 1 0-8 5 5 0 0 1 9.6-1A4.5 4.5 0 0 1 17 18z"/>', fn: B.climatePage },
  { id: "hotspots", title: "Geospatial Outbreak Clusters", nav: "Outbreak Clusters", icon: '<circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="8"/>', fn: B.hotspotPage },
  { grp: "Decision Support & Governance" },
  { id: "fusion", title: "Consolidated Threat Assessment", nav: "Threat Assessment", icon: '<circle cx="8" cy="12" r="5"/><circle cx="16" cy="12" r="5"/>', fn: B.fusionPage },
  { id: "explain", title: "Epidemiological Risk Drivers", nav: "Risk Drivers", icon: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.5 2.5 0 1 1 3.5 2.3c-.7.4-1 1-1 1.7M12 17h.01"/>', fn: B.explainPage },
  { id: "scenarios", title: "Intervention Planning Sandbox", nav: "Intervention Sandbox", icon: '<path d="M9 3h6M10 3v6l-5 9a2 2 0 0 0 2 3h10a2 2 0 0 0 2-3l-5-9V3"/>', fn: B.scenarioPage },
  { id: "quality", title: "Surveillance Registry & Data Audit", nav: "Surveillance Audit", icon: '<ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>', fn: B.qualityPage },
];

const ctx = {
  boot: null,
  state: { 
    disease: "ALL", 
    period: "ALL", 
    district: "Pune", 
    talukaDistrict: "Pune", 
    cmdPeriod: "ALL",
    officerRole: "DHO / DSO Pune District (Dr. S. Patil)",
    isMainDSO: false
  },
  live: null
};
let current = null;

window.setTalukaDistrict = (name) => {
  ctx.state.talukaDistrict = name;
  ctx.state.district = name;
  const sel = document.getElementById("talukaDistrictSelect");
  if (sel) {
    sel.value = name;
    sel.dispatchEvent(new Event("change"));
  }
};

const $ = (s) => document.querySelector(s);
const nav = $("#nav");
nav.innerHTML = ROUTES.map((r) => (r.grp ? `<div class="grp">${r.grp}</div>` : `<a href="#/${r.id}" data-id="${r.id}">${ICON(r.icon)}<span>${r.nav}</span></a>`)).join("");
$("#menuBtn").addEventListener("click", () => { const o = $("#side").classList.toggle("open"); $("#menuBtn").setAttribute("aria-expanded", o); });

async function route() {
  const id = (location.hash.replace(/^#\//, "") || "").split("?")[0];
  const r = ROUTES.find((x) => !x.grp && x.id === id) || ROUTES[1];
  nav.querySelectorAll("a").forEach((a) => (a.getAttribute("data-id") === r.id ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
  $("#pageTitle").textContent = r.title; document.title = r.title + " · JEEVAN-DRISHTI";
  $("#side").classList.remove("open");
  const main = $("#main"); main.innerHTML = loading();
  if (!ctx.boot) {
    try { ctx.boot = await get("/ui/bootstrap"); } catch (e) { main.innerHTML = errorBox(e); main.querySelector("[data-retry]").addEventListener("click", route); return; }
  }
  main.innerHTML = ""; current = null;
  try { current = (await r.fn(main, ctx)) || null; } catch (e) { main.innerHTML = errorBox(e); }
  main.focus({ preventScroll: true }); window.scrollTo(0, 0);
}
window.addEventListener("hashchange", route);

// Modal event wiring
document.querySelectorAll("[data-close]").forEach(b => {
  b.addEventListener("click", () => {
    const m = document.getElementById(b.dataset.close);
    if (m) m.style.display = "none";
  });
});

// ============================================================ Dynamic Alert Audit Trail
const AUDIT_LOGS = [
  {
    id: "ALT-PUN-W44",
    district: "Pune",
    taluka: "Haveli",
    disease: "Dengue",
    status: "UNDER_REVIEW",
    actionBy: "DHO Dr. S. Patil",
    timestamp: "2026-10-31 18:40",
    notes: "NS1 positivity spike at Wagholi PHC. RRT dispatched for vector source reduction."
  },
  {
    id: "ALT-NAG-W44",
    district: "Nagpur",
    taluka: "Hingna",
    disease: "Chikungunya",
    status: "RRT_DISPATCHED",
    actionBy: "DHO Dr. R. Deshmukh",
    timestamp: "2026-10-31 16:15",
    notes: "Fever cluster reported in MIDC industrial block. Rapid response team mobilized."
  },
  {
    id: "ALT-MUM-W44",
    district: "Mumbai Suburban",
    taluka: "Kurla",
    disease: "Acute Diarrheal",
    status: "CONFIRMED",
    actionBy: "Main DSO Dr. R. Kadam",
    timestamp: "2026-10-31 14:20",
    notes: "Pipeline contamination verified by municipal lab. Emergency super-chlorination ongoing."
  },
  {
    id: "ALT-NSK-W43",
    district: "Nashik",
    taluka: "Malegaon",
    disease: "Cholera",
    status: "RESOLVED",
    actionBy: "DHO Dr. A. Kulkarni",
    timestamp: "2026-10-30 11:05",
    notes: "Community well chlorination complete. Zero new admissions reported across 7 days."
  },
  {
    id: "ALT-KOL-W43",
    district: "Kolhapur",
    taluka: "Karvir",
    disease: "Leptospirosis",
    status: "UNDER_REVIEW",
    actionBy: "DHO Dr. M. Shinde",
    timestamp: "2026-10-29 09:30",
    notes: "Post-flood surveillance active in agricultural belt. Prophylactic Doxycycline distributed."
  },
  {
    id: "ALT-THN-W42",
    district: "Thane",
    taluka: "Kalyan",
    disease: "Malaria",
    status: "RESOLVED",
    actionBy: "DHO Dr. P. More",
    timestamp: "2026-10-28 15:45",
    notes: "Urban construction site breeding check complete. Active blood smear screening normal."
  }
];

function statusChip(s) {
  if (s === "UNDER_REVIEW") return `<span class="chip warn">UNDER_REVIEW</span>`;
  if (s === "RRT_DISPATCHED") return `<span class="chip info">RRT_DISPATCHED</span>`;
  if (s === "CONFIRMED") return `<span class="chip bad">CONFIRMED</span>`;
  if (s === "RESOLVED") return `<span class="chip ok">RESOLVED</span>`;
  return `<span class="chip na">${esc(s)}</span>`;
}

function renderAuditTable(filterDistrict = "ALL") {
  const tbody = $("#auditTableBody");
  if (!tbody) return;
  const filtered = filterDistrict === "ALL" 
    ? AUDIT_LOGS 
    : AUDIT_LOGS.filter(a => a.district.toLowerCase() === filterDistrict.toLowerCase());

  if (!filtered.length) {
    tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;padding:24px;color:#64748b">No alerts recorded for ${esc(filterDistrict)} district.</td></tr>`;
    return;
  }

  tbody.innerHTML = filtered.map(a => `
    <tr>
      <td><code style="font-weight:700">${esc(a.id)}</code></td>
      <td><strong>${esc(a.district)}</strong> / ${esc(a.taluka)}</td>
      <td>${esc(a.disease)}</td>
      <td id="chip_${esc(a.id)}">${statusChip(a.status)}</td>
      <td>
        <select class="audit-status-select" data-id="${esc(a.id)}" style="padding:4px 6px;border-radius:6px;border:1px solid #cbd5e1;font-size:11.5px;font-weight:600">
          <option value="UNDER_REVIEW" ${a.status === "UNDER_REVIEW" ? "selected" : ""}>Under Review</option>
          <option value="RRT_DISPATCHED" ${a.status === "RRT_DISPATCHED" ? "selected" : ""}>RRT Dispatched</option>
          <option value="CONFIRMED" ${a.status === "CONFIRMED" ? "selected" : ""}>Confirmed Outbreak</option>
          <option value="RESOLVED" ${a.status === "RESOLVED" ? "selected" : ""}>Resolved / Normal</option>
          <option value="FALSE_ALERT" ${a.status === "FALSE_ALERT" ? "selected" : ""}>False Positive</option>
        </select>
      </td>
      <td id="by_${esc(a.id)}" style="font-size:12px;color:#334155">${esc(a.actionBy)}</td>
      <td style="font-size:12px">
        <span style="color:#64748b;display:block;font-size:11px">${esc(a.timestamp)}</span>
        <span>${esc(a.notes)}</span>
      </td>
    </tr>
  `).join("");

  // Wire status change dropdowns
  tbody.querySelectorAll(".audit-status-select").forEach(sel => {
    sel.addEventListener("change", (e) => {
      const alertId = e.target.dataset.id;
      const newStatus = e.target.value;
      const targetAlert = AUDIT_LOGS.find(x => x.id === alertId);
      if (targetAlert) {
        targetAlert.status = newStatus;
        const actingOfficer = ctx.state.officerRole || "DHO / DSO";
        targetAlert.actionBy = actingOfficer;
        targetAlert.timestamp = new Date().toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" }) + " (Status Updated)";
        
        // Update DOM row elements immediately
        const chipEl = document.getElementById(`chip_${alertId}`);
        if (chipEl) chipEl.innerHTML = statusChip(newStatus);
        const byEl = document.getElementById(`by_${alertId}`);
        if (byEl) byEl.textContent = actingOfficer;

        // Show confirmation banner
        const banner = $("#auditStatusBanner");
        if (banner) {
          banner.style.display = "block";
          banner.innerHTML = `✅ Alert <strong>${esc(alertId)}</strong> status updated to <strong>${esc(newStatus)}</strong> by ${esc(actingOfficer)}.`;
          setTimeout(() => { banner.style.display = "none"; }, 3500);
        }
      }
    });
  });
}

$("#auditDistrictFilter")?.addEventListener("change", (e) => {
  renderAuditTable(e.target.value);
});

$("#btnThoAdvisory")?.addEventListener("click", () => {
  const m = $("#modalBroadcast");
  if (m) {
    const curDist = ctx.state.talukaDistrict || ctx.state.district || "Pune";
    $("#bcDistrict").value = curDist;
    if (curDist === "Nagpur") {
      $("#bcTaluka").value = "Hingna Taluka (Wanadongri & MIDC PHCs)";
      $("#bcThreat").value = "Chikungunya / Thermal Stress Cluster";
    } else if (curDist === "Nashik") {
      $("#bcTaluka").value = "Malegaon Taluka (Camp & Dyane PHCs)";
      $("#bcThreat").value = "Cholera / Water-Borne Surge";
    } else if (curDist === "Kolhapur") {
      $("#bcTaluka").value = "Karvir Taluka (Uchgaon & Shiroli PHCs)";
      $("#bcThreat").value = "Leptospirosis / Post-Flood Watch";
    } else if (curDist === "Chhatrapati Sambhajinagar") {
      $("#bcTaluka").value = "Paithan Taluka (Bidkin & Pachod PHCs)";
      $("#bcThreat").value = "Acute Diarrheal Outbreak Cluster";
    } else if (curDist === "Thane") {
      $("#bcTaluka").value = "Kalyan Taluka (Titwala & Dombivli PHCs)";
      $("#bcThreat").value = "Malaria / Construction Vector Breeding";
    } else {
      $("#bcTaluka").value = "Haveli Taluka (Wagholi & Uruli Kanchan PHCs)";
      $("#bcThreat").value = "Dengue (NS1 Antigen Cluster Surge)";
    }
    m.style.display = "flex";
  }
});

$("#btnAlertAudit")?.addEventListener("click", () => {
  const m = $("#modalAudit");
  if (m) {
    const f = $("#auditDistrictFilter");
    if (f) {
      f.value = ctx.state.isMainDSO ? "ALL" : (ctx.state.talukaDistrict || "Pune");
    }
    renderAuditTable(f ? f.value : "ALL");
    m.style.display = "flex";
  }
});

$("#btnPolicySandbox")?.addEventListener("click", () => {
  const m = $("#modalSimulation");
  if (m) m.style.display = "flex";
});

$("#roleBtn")?.addEventListener("click", () => {
  const m = $("#modalRole");
  if (m) m.style.display = "flex";
});

// Role switching: updates chips from 'Change' to 'Active', changes active district & header immediately
document.querySelectorAll("#roleList button").forEach(btn => {
  btn.addEventListener("click", () => {
    const d = btn.dataset.district;
    const r = btn.dataset.role;

    // 1. Toggle chips in modal: all to 'Change', clicked to 'Active'
    document.querySelectorAll("#roleList button").forEach(b => {
      const chip = b.querySelector(".role-chip");
      if (chip) {
        chip.className = "chip info role-chip";
        chip.textContent = "Change";
      }
    });
    const clickedChip = btn.querySelector(".role-chip");
    if (clickedChip) {
      clickedChip.className = "chip ok role-chip";
      clickedChip.textContent = "Active";
    }

    // 2. Update context state
    ctx.state.officerRole = r;
    if (d === "All Districts") {
      ctx.state.isMainDSO = true;
      ctx.state.talukaDistrict = "Pune";
      ctx.state.district = "Pune";
    } else {
      ctx.state.isMainDSO = false;
      ctx.state.talukaDistrict = d;
      ctx.state.district = d;
    }

    // 3. Update top micro-bar label
    const roleLabel = $("#roleLabel");
    if (roleLabel) roleLabel.textContent = r;

    // 4. Update broadcast target district
    if ($("#bcDistrict")) $("#bcDistrict").value = ctx.state.talukaDistrict;

    // 5. Update audit filter
    if ($("#auditDistrictFilter")) {
      $("#auditDistrictFilter").value = d === "All Districts" ? "ALL" : d;
    }

    // 6. Close modal & re-route immediately
    $("#modalRole").style.display = "none";
    route();
  });
});

// Simulation slider
function updateSimulation() {
  const f = parseFloat($("#rangeFogging")?.value || 65);
  const s = parseFloat($("#rangeScreening")?.value || 50);
  $("#valFogging").textContent = `${f}%`;
  $("#valScreening").textContent = `${s}%`;
  const base = 152;
  const damp = (f * 0.4 + s * 0.3) / 100;
  const net = Math.round(base * (1 - damp));
  const averted = base - net;
  const pct = Math.round((averted / base) * 100);
  if ($("#simCasesAverted")) $("#simCasesAverted").textContent = `+${averted} cases`;
  if ($("#simNetCaseload")) $("#simNetCaseload").textContent = `${net} cases`;
}
$("#rangeFogging")?.addEventListener("input", updateSimulation);
$("#rangeScreening")?.addEventListener("input", updateSimulation);

$("#btnSendBroadcast")?.addEventListener("click", () => {
  const dist = $("#bcDistrict")?.value || ctx.state.talukaDistrict || "Pune";
  const taluka = $("#bcTaluka")?.value || "Field Block";
  const threat = $("#bcThreat")?.value || "Outbreak Cluster";
  const dir = $("#bcDirective")?.value || "Deploy Rapid Response Team immediately.";
  const urgency = $("#bcUrgency")?.value || "HIGH";
  const officer = ctx.state.officerRole || "DHO / DSO";

  // Prepend to live AUDIT_LOGS
  const newId = `ALT-${dist.substring(0, 3).toUpperCase()}-W44-${Math.floor(100 + Math.random() * 900)}`;
  AUDIT_LOGS.unshift({
    id: newId,
    district: dist,
    taluka: taluka.split(" ")[0],
    disease: threat.split(" ")[0],
    status: urgency === "CRITICAL" ? "CONFIRMED" : "RRT_DISPATCHED",
    actionBy: officer,
    timestamp: "Just now",
    notes: `${urgency} Advisory: ${dir.substring(0, 75)}...`
  });

  renderAuditTable($("#auditDistrictFilter")?.value || "ALL");

  alert(`✅ Emergency Public Health Directive dispatched to Taluka Health Officer (${taluka}) by ${officer}.\n\nRegistered under Alert ID: ${newId} in the live Alert Audit Trail.`);
  $("#modalBroadcast").style.display = "none";
});

// ---------------- 2-second dashboard polling
let inflight = false, last = null;
async function poll() {
  if (document.hidden || inflight) return;
  inflight = true;
  const t0 = performance.now();
  const ac = new AbortController(); const to = setTimeout(() => ac.abort(), 1800);
  try {
    const s = await get("/refresh/summary", { fresh: true, signal: ac.signal });
    const latency = Math.round(performance.now() - t0);
    ctx.live = s;
    const beat = $("#beat"); beat.className = "dot dot-ok"; void beat.offsetWidth; beat.classList.add("pulse");
    $("#statusText").textContent = `Dashboard refreshed ${new Date(s.last_dashboard_refresh).toLocaleTimeString("en-IN")} · every ${s.refresh_interval_seconds}s`;
    if ($("#liveLatency")) $("#liveLatency").textContent = `(${latency}ms)`;
    $("#timeline").innerHTML = `<span>Dashboard refresh: <b>${fmtTime(s.last_dashboard_refresh)}</b></span><span class="sep">|</span><span>Source observation date: <b>${esc(s.source_observation_date)}</b> <i>(${esc(s.source_frequency_label)})</i></span><span class="sep">|</span><span>Last successful ingestion: <b>${fmtTime(s.last_successful_ingestion)}</b></span><span class="sep">|</span><span>Last prediction run: <b>${fmtTime(s.last_prediction_generation)}</b></span><span class="sep">|</span><span>Models: <b>${esc(s.model_version)}</b></span><span class="sep">|</span><span><b>IDSP Weekly Surveillance</b></span>`;
    $("#banner").innerHTML = "";
    if (current && current.onLive) current.onLive(s);
    const sig = s.source_observation_date + "|" + s.last_successful_ingestion + "|" + s.last_prediction_generation;
    if (last !== null && sig !== last) { clearCache(); ctx.boot = null; const m = $("#main"); if (current && current.onSource) { ctx.boot = await get("/ui/bootstrap"); current.onSource(); } }
    last = sig;
  } catch (e) {
    $("#beat").className = "dot dot-bad";
    $("#statusText").textContent = "Backend unreachable – showing last loaded data";
    $("#banner").innerHTML = `<div class="banner" role="alert">Cannot reach the JEEVAN-DRISHTI API (${esc(e.name === "AbortError" ? "timeout" : e.message)}). Displayed values may be out of date. Retrying every 2 seconds.</div>`;
  } finally { clearTimeout(to); inflight = false; }
}
setInterval(poll, 2000); poll();
document.addEventListener("visibilitychange", () => { if (!document.hidden) poll(); });
route();
