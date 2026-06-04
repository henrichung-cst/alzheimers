"use strict";

function _ctPayload() {
  return (PAYLOAD && PAYLOAD.celltype_assignment) || {};
}

function _ctContextId() {
  return (Store.state.selection && Store.state.selection.context) || META.default_context || "donor1";
}

function _ctContextLabel(ctx) {
  const contexts = (META && META.contexts) || [];
  const hit = contexts.find(c => c.id === ctx);
  return hit ? hit.label : ctx;
}

function _ctPct(n, denom) {
  if (!denom) return "0.0%";
  return (100 * Number(n || 0) / denom).toFixed(1) + "%";
}

function _ctFormatInt(n) {
  return Number(n || 0).toLocaleString();
}

const _CT_COLORS = [
  "#1b5e20", "#0b6e69", "#1565c0", "#6a1b9a", "#ad1457",
  "#c62828", "#ef6c00", "#9e9d24", "#2e7d32", "#00838f",
  "#283593", "#7b1fa2", "#bf360c", "#455a64", "#5d4037",
];

function _ctColorForState(state) {
  const states = (_ctPayload().states || []).slice().sort();
  const idx = Math.max(0, states.indexOf(state));
  return _CT_COLORS[idx % _CT_COLORS.length];
}

function _ctBarRows(rows, opts) {
  const max = Math.max(1, ...rows.map(r => Number(r.value || 0)));
  const denom = opts && opts.denom ? opts.denom : rows.reduce((a, r) => a + Number(r.value || 0), 0);
  const cls = opts && opts.className ? opts.className : "";
  if (!rows.length) return `<div class="muted">No assignment rows available.</div>`;
  return rows.map(r => {
    const value = Number(r.value || 0);
    const pct = _ctPct(value, denom);
    const width = Math.max(1.5, 100 * value / max);
    const extra = r.detail ? `<span class="ct-bar-detail">${_escapeHtml(r.detail)}</span>` : "";
    return `<div class="ct-bar-row ${cls}">` +
      `<div class="ct-bar-label" title="${_escapeHtml(r.label)}">${_escapeHtml(r.label)}</div>` +
      `<div class="ct-bar-track"><div class="ct-bar-fill" style="width:${width.toFixed(2)}%;"></div></div>` +
      `<div class="ct-bar-value">${_ctFormatInt(value)} <span class="muted">${pct}</span>${extra}</div>` +
      `</div>`;
  }).join("");
}

function _renderCtStates(block) {
  const totals = block.state_totals || {};
  const conf = block.confidence_by_state || {};
  const denom = Object.values(totals).reduce((a, v) => a + Number(v || 0), 0);
  const rows = Object.keys(totals)
    .sort((a, b) => Number(totals[b] || 0) - Number(totals[a] || 0) || a.localeCompare(b))
    .map(k => ({
      label: k,
      value: totals[k],
      detail: conf[k] == null ? "" : `median conf ${Number(conf[k]).toFixed(2)}`,
    }));
  const el = document.getElementById("ct-state-bars");
  if (el) el.innerHTML = _ctBarRows(rows, {denom, className: "ct-state-row"});
}

function _ctSetOptions(select, values, fallbackLabel) {
  if (!select) return "";
  const prev = select.value;
  const vals = values && values.length ? values : [""];
  select.innerHTML = vals.map(v => {
    const label = v || fallbackLabel || "none";
    return `<option value="${_escapeHtml(v)}">${_escapeHtml(label)}</option>`;
  }).join("");
  select.value = vals.includes(prev) ? prev : vals[0];
  return select.value;
}

function _renderCtEmbedding(block) {
  const emb = block.embedding || {};
  const status = document.getElementById("ct-embedding-status");
  const canvas = document.getElementById("ct-embedding-canvas");
  const legend = document.getElementById("ct-embedding-legend");
  const refSel = document.getElementById("ct-embedding-ref");
  const redSel = document.getElementById("ct-embedding-reduction");
  if (!status || !canvas || !legend) return;
  if (!emb.available || !emb.points || !(emb.points.x || []).length) {
    status.textContent = "No ProjecTILs embedding coordinates are packaged for this donor. Rerun tcells-projectils-map to generate projectils_embeddings.csv.";
    canvas.hidden = true;
    legend.innerHTML = "";
    if (refSel) refSel.innerHTML = `<option value="">none</option>`;
    if (redSel) redSel.innerHTML = `<option value="">none</option>`;
    return;
  }

  canvas.hidden = false;
  const ref = _ctSetOptions(refSel, emb.projection_references || [], "reference");
  const reductionsForRef = Array.from(new Set((emb.points.reduction || []).filter((r, i) =>
    (emb.points.projection_reference || [])[i] === ref
  ))).sort();
  const reduction = _ctSetOptions(redSel, reductionsForRef.length ? reductionsForRef : emb.reductions || [], "reduction");

  const pts = [];
  const P = emb.points;
  for (let i = 0; i < P.x.length; i++) {
    if (P.projection_reference[i] !== ref || P.reduction[i] !== reduction) continue;
    const x = Number(P.x[i]);
    const y = Number(P.y[i]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    pts.push({x, y, state: P.state[i] || "unknown", day: P.day[i] || ""});
  }
  if (!pts.length) {
    status.textContent = "No coordinates match the selected projection reference and reduction.";
    legend.innerHTML = "";
    return;
  }
  status.textContent = `${_ctFormatInt(pts.length)} cells shown. CD4 and CD8 projections are separate reference spaces.`;

  const rect = canvas.getBoundingClientRect();
  const cssW = Math.max(360, Math.floor(rect.width || canvas.clientWidth || 920));
  const cssH = Math.max(260, Math.floor(rect.height || 420));
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(cssW * dpr);
  canvas.height = Math.floor(cssH * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, cssW, cssH);

  const pad = {l: 28, r: 14, t: 12, b: 24};
  const xs = pts.map(p => p.x);
  const ys = pts.map(p => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const sx = (cssW - pad.l - pad.r) / Math.max(1e-9, maxX - minX);
  const sy = (cssH - pad.t - pad.b) / Math.max(1e-9, maxY - minY);

  ctx.strokeStyle = "#d7dee2";
  ctx.lineWidth = 1;
  ctx.strokeRect(pad.l, pad.t, cssW - pad.l - pad.r, cssH - pad.t - pad.b);
  ctx.globalAlpha = 0.72;
  for (const p of pts) {
    const px = pad.l + (p.x - minX) * sx;
    const py = cssH - pad.b - (p.y - minY) * sy;
    ctx.fillStyle = _ctColorForState(p.state);
    ctx.beginPath();
    ctx.arc(px, py, 2.2, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
  ctx.fillStyle = "#607d8b";
  ctx.font = "11px IBM Plex Mono, ui-monospace, monospace";
  ctx.fillText(reduction, pad.l, cssH - 7);

  const counts = new Map();
  pts.forEach(p => counts.set(p.state, (counts.get(p.state) || 0) + 1));
  const items = Array.from(counts.entries())
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
  legend.innerHTML = items.map(([state, n]) => (
    `<span class="ct-legend-item"><i style="background:${_ctColorForState(state)}"></i>` +
    `${_escapeHtml(state)} <em>${_ctFormatInt(n)}</em></span>`
  )).join("");
}

function _renderCtDayGrid(block) {
  const rows = block.state_by_day || [];
  const states = Array.from(new Set(rows.map(r => r.state))).sort();
  const days = Array.from(new Set(rows.map(r => r.day))).sort((a, b) => {
    const na = Number(String(a).replace(/^d/, ""));
    const nb = Number(String(b).replace(/^d/, ""));
    return na - nb;
  });
  const byKey = new Map(rows.map(r => [`${r.state}|${r.day}`, Number(r.n_cells || 0)]));
  const max = Math.max(1, ...rows.map(r => Number(r.n_cells || 0)));
  const el = document.getElementById("ct-day-grid");
  if (!el) return;
  if (!states.length || !days.length) {
    el.innerHTML = `<div class="muted">No state-by-day counts available.</div>`;
    return;
  }
  const head = [`<div class="ct-day-corner"></div>`]
    .concat(days.map(d => `<div class="ct-day-head">${_escapeHtml(d)}</div>`))
    .join("");
  const body = states.map(state => {
    const cells = days.map(day => {
      const n = byKey.get(`${state}|${day}`) || 0;
      const alpha = Math.max(0.08, Math.min(0.95, n / max));
      return `<div class="ct-day-cell" title="${_escapeHtml(state)} ${_escapeHtml(day)}: ${_ctFormatInt(n)} cells">` +
        `<span style="opacity:${alpha.toFixed(2)}"></span>${_ctFormatInt(n)}` +
        `</div>`;
    }).join("");
    return `<div class="ct-day-state" title="${_escapeHtml(state)}">${_escapeHtml(state)}</div>${cells}`;
  }).join("");
  el.style.gridTemplateColumns = `minmax(130px, 1.4fr) repeat(${days.length}, minmax(54px, 0.65fr))`;
  el.innerHTML = head + body;
}

function renderCellTypeAssignment() {
  const ctx = _ctContextId();
  const payload = _ctPayload();
  const block = (payload.by_context && payload.by_context[ctx]) || {};
  const label = document.getElementById("ct-context-label");
  if (label) label.textContent = _ctContextLabel(ctx);
  _renderCtEmbedding(block);
  _renderCtStates(block);
  _renderCtDayGrid(block);
}

function wireCellTypeAssignment() {
  ["ct-embedding-ref", "ct-embedding-reduction"].forEach(id => {
    const el = document.getElementById(id);
    if (!el || el.dataset.wired === "1") return;
    el.dataset.wired = "1";
    el.addEventListener("change", renderCellTypeAssignment);
  });
  window.addEventListener("resize", () => {
    if (Store.state.view.activeTab === "celltype") renderCellTypeAssignment();
  });
}
