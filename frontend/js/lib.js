// API client + formatting + reusable UI components (no external dependencies).
export const API = "/api/v1";
const cache = new Map();

export async function get(path, { fresh = false, signal } = {}) {
  if (!fresh && cache.has(path)) return cache.get(path);
  const r = await fetch(API + path, { signal });
  if (!r.ok) {
    let d = ""; try { d = (await r.json()).detail || ""; } catch (_) {}
    throw new Error(`${r.status} ${r.statusText}${d ? " – " + d : ""}`);
  }
  const j = await r.json();
  cache.set(path, j);
  return j;
}
export const clearCache = () => cache.clear();

export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
export const isNum = (v) => typeof v === "number" && isFinite(v);
export const NA = '<span class="na-txt">Not Available</span>';
export const fmt = (v, d = 0) => (isNum(v) ? v.toLocaleString("en-IN", { minimumFractionDigits: d, maximumFractionDigits: d }) : "Not Available");
export const fmtH = (v, d = 0) => (isNum(v) ? esc(fmt(v, d)) : NA);
export const txtH = (v) => (v === null || v === undefined || v === "" ? NA : esc(v));
export const pct = (v, d = 1) => (isNum(v) ? v.toFixed(d) + "%" : "Not Available");
export function fmtTime(iso) {
  if (!iso) return "Not Available";
  const d = new Date(iso); if (isNaN(d)) return esc(iso);
  return d.toLocaleString("en-IN", { day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" });
}
export const chip = (t, kind = "") => `<span class="chip ${kind}">${esc(t)}</span>`;
export const loading = (m = "Loading from backend…") => `<div class="loading"><span class="spin"></span>${esc(m)}</div>`;
export const empty = (m) => `<div class="empty">${esc(m)}</div>`;
export const errorBox = (e) => `<div class="notice bad"><b>Could not load data.</b> ${esc(e.message || e)} <button class="btn ghost" data-retry>Retry</button></div>`;
export const notice = (m, k = "") => `<div class="notice ${k}">${m}</div>`;

export function options(list, sel, { all } = {}) {
  let h = all ? `<option value="${esc(all[0])}">${esc(all[1])}</option>` : "";
  h += list.map((o) => { const v = typeof o === "object" ? o.value : o; const l = typeof o === "object" ? o.label : o; return `<option value="${esc(v)}"${v === sel ? " selected" : ""}>${esc(l)}</option>`; }).join("");
  return h;
}
export function table(cols, rows, { caption = "", onClickKey, selected } = {}) {
  if (!rows || !rows.length) return empty("No records for this selection.");
  const head = cols.map((c) => `<th class="${c.num ? "num" : ""}">${esc(c.h)}</th>`).join("");
  const body = rows.map((r, i) => `<tr class="${onClickKey ? "click" : ""}${selected !== undefined && r[onClickKey] === selected ? " sel" : ""}" ${onClickKey ? `data-key="${esc(r[onClickKey])}" tabindex="0"` : ""}>` +
    cols.map((c) => `<td class="${c.num ? "num" : ""}">${c.f ? c.f(r, i) : txtH(r[c.k])}</td>`).join("") + "</tr>").join("");
  return `<div class="tbl-wrap"><table>${caption ? `<caption class="sr-only" style="position:absolute;left:-999px">${esc(caption)}</caption>` : ""}<thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

// ---- tooltip
const tip = () => document.getElementById("tip");
export function bindTips(root) {
  root.querySelectorAll("[data-tip]").forEach((el) => {
    const show = (e) => { const t = tip(); t.innerHTML = el.getAttribute("data-tip"); t.hidden = false; const x = (e.clientX ?? el.getBoundingClientRect().left) + 14, y = (e.clientY ?? el.getBoundingClientRect().top) + 14; t.style.left = Math.min(x, innerWidth - 270) + "px"; t.style.top = y + "px"; };
    el.addEventListener("mousemove", show); el.addEventListener("focus", show);
    el.addEventListener("mouseleave", () => (tip().hidden = true)); el.addEventListener("blur", () => (tip().hidden = true));
  });
}

// ---- SVG chart: categorical x, optional bands, lines (solid/dashed), bars, markers; gaps for nulls
export function chart({ x, series, bands = [], yLabel = "", height = 280, markers = [], vlines = [], yMin = null, aria = "" }) {
  const W = 760, H = height, m = { l: 52, r: 14, t: 14, b: 46 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const vals = [];
  series.forEach((s) => s.y.forEach((v) => isNum(v) && vals.push(v)));
  bands.forEach((b) => { b.lo.forEach((v) => isNum(v) && vals.push(v)); b.hi.forEach((v) => isNum(v) && vals.push(v)); });
  if (!vals.length || !x.length) return empty("Nothing to plot – insufficient data for this selection.");
  let lo = yMin !== null ? yMin : Math.min(0, ...vals), hi = Math.max(...vals);
  if (hi === lo) hi = lo + 1;
  const pad = (hi - lo) * 0.06; hi += pad;
  const n = x.length, step = iw / Math.max(1, n);
  const X = (i) => m.l + step * (i + 0.5), Y = (v) => m.t + ih - ((v - lo) / (hi - lo)) * ih;
  let g = "";
  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const v = lo + ((hi - lo) * i) / ticks, y = Y(v);
    g += `<line x1="${m.l}" x2="${W - m.r}" y1="${y}" y2="${y}" stroke="#e6eaf2"/><text x="${m.l - 8}" y="${y + 4}" text-anchor="end" font-size="11" fill="#5b6780">${v >= 100 ? Math.round(v).toLocaleString("en-IN") : v.toFixed(v < 10 ? 1 : 0)}</text>`;
  }
  const every = Math.ceil(n / 10);
  x.forEach((l, i) => { if (i % every === 0 || i === n - 1) g += `<text x="${X(i)}" y="${H - m.b + 16}" text-anchor="end" font-size="10.5" fill="#5b6780" transform="rotate(-35 ${X(i)} ${H - m.b + 16})">${esc(l)}</text>`; });
  vlines.forEach((v) => { const i = x.indexOf(v.at); if (i >= 0) g += `<line x1="${X(i) - step / 2}" x2="${X(i) - step / 2}" y1="${m.t}" y2="${m.t + ih}" stroke="#7c3aed" stroke-dasharray="4 3"/><text x="${X(i) - step / 2 + 4}" y="${m.t + 11}" font-size="10.5" fill="#7c3aed">${esc(v.label)}</text>`; });
  bands.forEach((b) => {
    const up = [], dn = [];
    b.hi.forEach((v, i) => { if (isNum(v) && isNum(b.lo[i])) { up.push([X(i), Y(v)]); dn.push([X(i), Y(b.lo[i])]); } });
    if (up.length > 1) g += `<path d="M${up.map((p) => p.join(",")).join("L")}L${dn.reverse().map((p) => p.join(",")).join("L")}Z" fill="${b.color}" opacity=".16"><title>${esc(b.name)}</title></path>`;
  });
  series.forEach((s) => {
    if (s.type === "bar") {
      const bw = Math.max(2, Math.min(18, step * 0.7));
      s.y.forEach((v, i) => { if (isNum(v)) g += `<rect x="${X(i) - bw / 2}" y="${Y(Math.max(v, 0))}" width="${bw}" height="${Math.max(1, Math.abs(Y(0) - Y(v)))}" fill="${s.color}" rx="1.5" data-tip="${esc(x[i])}<br>${esc(s.name)}: <b>${fmt(v, 1)}</b>" tabindex="0"/>`; });
      return;
    }
    let d = "", pen = false;
    s.y.forEach((v, i) => { if (isNum(v)) { d += `${pen ? "L" : "M"}${X(i)},${Y(v)}`; pen = true; } else pen = false; });
    g += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2.2" ${s.dash ? 'stroke-dasharray="6 4"' : ""} stroke-linejoin="round"/>`;
    s.y.forEach((v, i) => { if (isNum(v)) g += `<circle cx="${X(i)}" cy="${Y(v)}" r="${s.dots === false ? 0 : 3}" fill="${s.color}" stroke="#fff" stroke-width="1" data-tip="${esc(x[i])}<br>${esc(s.name)}: <b>${fmt(v, 1)}</b>" tabindex="0"/>`; });
  });
  markers.forEach((mk) => { const i = mk.i; if (isNum(mk.v)) g += `<g data-tip="${esc(mk.tip)}" tabindex="0"><path d="M${X(i)},${Y(mk.v) - 11} l7,12 h-14 z" fill="#b42318" stroke="#fff"/></g>`; });
  const leg = series.map((s) => `<span><svg width="26" height="10"><line x1="0" x2="26" y1="5" y2="5" stroke="${s.color}" stroke-width="3" ${s.dash ? 'stroke-dasharray="6 4"' : ""}/></svg>${esc(s.name)}</span>`).join("") +
    bands.map((b) => `<span><svg width="18" height="10"><rect width="18" height="10" fill="${b.color}" opacity=".3"/></svg>${esc(b.name)}</span>`).join("") +
    (markers.length ? '<span><svg width="14" height="12"><path d="M7,0 l7,12 h-14z" fill="#b42318"/></svg>Elevated signal</span>' : "");
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(aria || yLabel)}"><text x="12" y="${m.t + ih / 2}" transform="rotate(-90 12 ${m.t + ih / 2})" text-anchor="middle" font-size="11" fill="#5b6780">${esc(yLabel)}</text>${g}</svg><div class="legend">${leg}</div>`;
}

export function hbar(rows, { label, value, color = "#1d5fd6", max, fmtv = (v) => fmt(v, 0) }) {
  const m = max ?? Math.max(...rows.map((r) => r[value] || 0), 1);
  return rows.map((r) => `<div style="display:grid;grid-template-columns:150px 1fr 70px;gap:8px;align-items:center;margin:5px 0;font-size:13px"><span>${esc(r[label])}</span><div class="bar"><i style="width:${Math.max(1, ((r[value] || 0) / m) * 100)}%;background:${color}"></i></div><span style="text-align:right">${fmtv(r[value])}</span></div>`).join("");
}

import { MAHARASHTRA_DISTRICT_GEOS, findDistrictGeo } from "./maharashtra_geo.js";

// ---- Map (Vector Polygons & Hotspot Epicenters for Maharashtra) ----
const SHAPES = {
  circle: (x, y, r) => `<circle cx="${x}" cy="${y}" r="${r}"/>`,
  triangle: (x, y, r) => `<path d="M${x},${y - r * 1.2} L${x + r * 1.1},${y + r * 0.9} L${x - r * 1.1},${y + r * 0.9}Z"/>`,
  diamond: (x, y, r) => `<path d="M${x},${y - r * 1.3} L${x + r * 1.2},${y} L${x},${y + r * 1.3} L${x - r * 1.2},${y}Z"/>`,
  square: (x, y, r) => `<rect x="${x - r}" y="${y - r}" width="${r * 2}" height="${r * 2}" rx="2"/>`,
};
export const SHAPE_PATH = SHAPES;

export function mapView(rows, { classify, legend, selected, labels = true }) {
  let gPolys = "";
  let gHotspots = "";
  let gLabels = "";

  MAHARASHTRA_DISTRICT_GEOS.forEach((geo) => {
    const row = rows.find(r => {
      const rn = (r.district_name || "").toLowerCase().replace(/[^a-z0-9]/g, "");
      const gn = geo.name.toLowerCase().replace(/[^a-z0-9]/g, "");
      return rn === gn || geo.id === rn;
    });

    const c = row ? classify(row) : { insufficient: true, label: "Context district" };
    const isSel = selected && (selected.toLowerCase() === geo.name.toLowerCase() || selected.toLowerCase() === geo.id);
    const fillColor = c.insufficient ? "#cbd5e1" : (c.color || "#34a853");
    const strokeColor = isSel ? "#0d9488" : "#ffffff";
    const strokeWidth = isSel ? "3" : "1.2";

    const tip = `<b>${esc(geo.name)} (${esc(geo.marathiName || "")})</b><br>${esc(c.label)}${c.detail ? "<br>" + esc(c.detail) : ""}${geo.hotspotLabel ? "<br><b>Epicenter:</b> " + esc(geo.hotspotLabel) : ""}`;

    gPolys += `
      <path d="${geo.svgPath}"
        fill="${fillColor}"
        stroke="${strokeColor}"
        stroke-width="${strokeWidth}"
        class="mappt"
        role="button"
        tabindex="0"
        data-d="${esc(geo.name)}"
        data-tip="${tip}"
        style="cursor:pointer;transition:fill .2s,stroke-width .2s"
      />
    `;

    // Hotspot beacon if marked or elevated
    if (geo.hotspotX && geo.hotspotY && (c.shape === "diamond" || c.color === "#b42318" || c.color === "#e76f51")) {
      gHotspots += `
        <g class="mappt" data-d="${esc(geo.name)}" data-tip="${tip}">
          <circle cx="${geo.hotspotX}" cy="${geo.hotspotY}" r="4.5" fill="#e76f51" stroke="#fff" stroke-width="1.2"/>
          <circle cx="${geo.hotspotX}" cy="${geo.hotspotY}" r="9" fill="none" stroke="#e76f51" stroke-width="1.5" opacity="0.8">
            <animate attributeName="r" values="4;12;4" dur="2s" repeatCount="indefinite"/>
            <animate attributeName="opacity" values="0.9;0.1;0.9" dur="2s" repeatCount="indefinite"/>
          </circle>
        </g>
      `;
    }

    if (labels && geo.centerX && geo.centerY) {
      gLabels += `
        <text x="${geo.centerX}" y="${geo.centerY}"
          font-size="9"
          font-weight="700"
          fill="${isSel ? '#0f172a' : '#1e293b'}"
          text-anchor="middle"
          pointer-events="none"
          paint-order="stroke"
          stroke="#ffffff"
          stroke-width="2.5"
          opacity="0.9"
        >${esc(geo.name)}</text>
      `;
    }
  });

  const leg = legend.map((l) => `<span><svg width="18" height="18" viewBox="-9 -9 18 18" fill="${l.insufficient ? "#cbd5e1" : l.color}" stroke="${l.insufficient ? "#5b6780" : "#fff"}" ${l.insufficient ? 'stroke-dasharray="2 2"' : ""}>${SHAPES[l.shape || "circle"](0, 0, 6)}</svg>${esc(l.label)}</span>`).join("");

  return `
    <svg class="chart" viewBox="100 60 640 500" role="group" aria-label="Maharashtra district vector choropleth map" style="background:linear-gradient(to bottom, #f8fafc, #f1f5f9);border-radius:12px;border:1px solid #e2e8f0;max-height:480px">
      <g id="mapPolygons">${gPolys}</g>
      <g id="mapHotspots">${gHotspots}</g>
      <g id="mapLabels">${gLabels}</g>
    </svg>
    <div class="legend" style="margin-top:10px">${leg}</div>
    <p class="sub" style="margin-top:8px">Maharashtra State 36-district vector choropleth with active taluka epicenters. Click any district to inspect localized taluka outbreak clusters.</p>
  `;
}
