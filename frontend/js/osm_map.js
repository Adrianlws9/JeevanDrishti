/**
 * OpenStreetMap (Leaflet) Disease Surveillance Map for Maharashtra
 * =================================================================
 * Renders real OpenStreetMap tiles (roads, rivers, topography, taluka centres)
 * with color-coded epidemiological risk beacons, active outbreak clusters,
 * and localized DHO drilldown popups.
 */

export const MAHARASHTRA_COORDS = {
  "Pune": [18.5204, 73.8567],
  "Nagpur": [21.1458, 79.0882],
  "Nashik": [19.9975, 73.7898],
  "Kolhapur": [16.7050, 74.2433],
  "Chhatrapati Sambhajinagar": [19.8762, 75.3433],
  "Thane": [19.2183, 72.9781],
  "Mumbai Suburban": [19.0760, 72.8777],
  "Mumbai City": [18.9388, 72.8354],
  "Amravati": [20.9374, 77.7796],
  "Solapur": [17.6599, 75.9064],
  "Satara": [17.6805, 73.9997],
  "Ahmednagar": [19.0952, 74.7496],
  "Jalgaon": [21.0077, 75.5626],
  "Dhule": [20.9042, 74.7749],
  "Nandurbar": [21.3732, 74.2407],
  "Yavatmal": [20.3888, 78.1204],
  "Akola": [20.7002, 77.0082],
  "Buldhana": [20.5293, 76.1843],
  "Wardha": [20.7453, 78.6022],
  "Chandrapur": [19.9615, 79.2961],
  "Gadchiroli": [20.1809, 80.0031],
  "Bhandara": [21.1714, 79.6548],
  "Gondia": [21.4598, 80.1961],
  "Nanded": [19.1383, 77.3210],
  "Latur": [18.4088, 76.5604],
  "Dharashiv": [18.1856, 76.0419],
  "Beed": [18.9891, 75.7601],
  "Parbhani": [19.2704, 76.7749],
  "Jalna": [19.8347, 75.8816],
  "Hingoli": [19.7180, 77.1477],
  "Washim": [20.1098, 77.1352],
  "Sangli": [16.8524, 74.5815],
  "Raigad": [18.5158, 73.1822],
  "Palghar": [19.6967, 72.7655],
  "Ratnagiri": [16.9902, 73.3120],
  "Sindhudurg": [16.1111, 73.6939]
};

let activeMap = null;

export function renderOSMMap(containerId, rows, { classify, selected, onSelectDistrict }) {
  const container = document.getElementById(containerId);
  if (!container) return;

  // Cleanup existing map instance if any
  if (activeMap) {
    activeMap.remove();
    activeMap = null;
  }

  // Fallback check if Leaflet is loaded
  if (typeof L === 'undefined') {
    container.innerHTML = `<div class="notice warn">Leaflet map library loading... please refresh or ensure script is loaded.</div>`;
    return;
  }

  // Create Leaflet Map centered on Maharashtra state
  const map = L.map(containerId, {
    center: [19.25, 75.85],
    zoom: 7,
    minZoom: 6,
    maxZoom: 14,
    scrollWheelZoom: true
  });
  activeMap = map;

  setTimeout(() => {
    if (map) map.invalidateSize();
  }, 150);

  // Add OpenStreetMap real cartographic tile layer
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors | IDSP Maharashtra',
    maxZoom: 18
  }).addTo(map);

  // Group for all district markers
  const markersGroup = L.featureGroup().addTo(map);

  rows.forEach((r) => {
    const dName = r.district_name;
    const coords = MAHARASHTRA_COORDS[dName] || (r.lat && r.lon ? [r.lat, r.lon] : null);
    if (!coords) return;

    const c = classify(r);
    const isSelected = selected && selected.toLowerCase() === dName.toLowerCase();
    const cases = r.observed_cases || 0;
    
    // Determine marker radius based on cases
    const baseRadius = 10;
    const caseRadius = cases > 0 ? Math.min(18, baseRadius + Math.sqrt(cases) / 3.5) : baseRadius;
    
    // Choose clean accessible color
    let fillColor = "#10b981"; // Low / Normal Green
    if (c.color === "#b42318" || c.color === "#e76f51" || r.risk_category === "High") {
      fillColor = "#e76f51"; // Coral Red
    } else if (c.color === "#e08a1e" || c.color === "#e9b949" || r.risk_category === "Moderate") {
      fillColor = "#e9b949"; // Amber Gold
    } else if (c.insufficient) {
      fillColor = "#94a3b8"; // Slate
    }

    // Outer beacon for High Alert districts
    if (fillColor === "#e76f51") {
      L.circleMarker(coords, {
        radius: caseRadius + 9,
        color: "#e76f51",
        weight: 1.5,
        opacity: 0.6,
        dashArray: "3 3",
        fillColor: "transparent"
      }).addTo(markersGroup);
    }

    // Main district circle marker
    const marker = L.circleMarker(coords, {
      radius: caseRadius,
      fillColor: fillColor,
      color: isSelected ? "#0a192f" : "#ffffff",
      weight: isSelected ? 3.5 : 1.8,
      fillOpacity: 0.9
    }).addTo(markersGroup);

    // Tooltip on hover
    marker.bindTooltip(`
      <div style="font-family:system-ui,-apple-system,sans-serif;font-size:12px;padding:2px">
        <strong style="color:#0f172a">${dName}</strong><br>
        <span style="color:#64748b">${c.label || 'Surveillance Status'}</span>
        ${cases > 0 ? `<br><b>${cases}</b> reported cases` : ''}
      </div>
    `, { direction: 'top', offset: [0, -caseRadius] });

    // Rich popup on click
    marker.bindPopup(`
      <div style="font-family:system-ui,-apple-system,sans-serif;font-size:13px;padding:4px;min-width:200px">
        <div style="display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #e2e8f0;padding-bottom:6px;margin-bottom:8px">
          <strong style="font-size:14px;color:#0f172a">${dName} District</strong>
          <span style="font-size:11px;font-weight:700;padding:2px 6px;border-radius:4px;background:${fillColor}22;color:${fillColor}">
            ${r.risk_category || 'Routine'}
          </span>
        </div>
        <div style="font-size:12px;color:#475569;margin-bottom:4px">
          <strong>Observed Cases:</strong> ${cases > 0 ? cases : '0 active outbreak records'}
        </div>
        ${r.dominant_disease ? `<div style="font-size:12px;color:#475569;margin-bottom:4px"><strong>Primary Threat:</strong> ${r.dominant_disease}</div>` : ''}
        ${r.anomaly_flagged_weeks ? `<div style="font-size:12px;color:#c2410c;margin-bottom:6px">⚠️ ${r.anomaly_flagged_weeks} week(s) above threshold</div>` : ''}
        <button id="btnPopTaluka_${dName.replace(/\s+/g, '_')}" style="width:100%;margin-top:8px;background:#0d9488;color:#fff;border:none;border-radius:6px;padding:6px 10px;font-size:12px;font-weight:700;cursor:pointer">
          🏥 Inspect Talukas (DHO) →
        </button>
      </div>
    `);

    marker.on('click', () => {
      if (onSelectDistrict) onSelectDistrict(dName);
    });

    marker.on('popupopen', () => {
      const btn = document.getElementById(`btnPopTaluka_${dName.replace(/\s+/g, '_')}`);
      if (btn) {
        btn.addEventListener('click', () => {
          location.hash = '#/talukas';
          if (window.setTalukaDistrict) window.setTalukaDistrict(dName);
        });
      }
    });
  });

  return map;
}
