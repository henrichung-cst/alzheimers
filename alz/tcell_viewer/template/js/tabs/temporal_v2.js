// ---------------------------------------------------------------------------
// T-cell Temporal tab
// ---------------------------------------------------------------------------
let _temporalState = null;

function _temporalActiveCapabilities() {
  return ViewerPayload.contextCapabilities
    ? ViewerPayload.contextCapabilities()
    : {};
}

function _temporalDefaultSeries(layer) {
  const caps = _temporalActiveCapabilities();
  const resolvedLayer = layer || (caps.kinases ? "pathway" : "pathway");
  return {
    layer: resolvedLayer,
    sign: "signed",
    fdr: 0.25,
    pvalue: null,
    absPds: 0.01,
  };
}

function _temporalEnsureState() {
  if (_temporalState) return;
  _temporalState = {
    series: [_temporalDefaultSeries("pathway")],
    shareY: false,
  };
}

function _temporalDayNumber(day) {
  const m = String(day || "").match(/^d(\d+)$/);
  return m ? Number(m[1]) : Number.POSITIVE_INFINITY;
}

function _temporalSortDays(days) {
  return Array.from(new Set(days.filter(Boolean))).sort((a, b) => {
    const da = _temporalDayNumber(a);
    const db = _temporalDayNumber(b);
    if (da !== db) return da - db;
    return String(a).localeCompare(String(b));
  });
}

function _temporalPathwayBlock() {
  const block = ViewerPayload.incytr();
  if (!block) return null;
  if (window.IncytrCelltypeQc
      && IncytrCelltypeQc.enabled(block)
      && block.pathway_counts_low_signal_excluded) {
    return block.pathway_counts_low_signal_excluded;
  }
  return block.pathway_counts || null;
}

function _temporalActiveIncytrBlock() {
  return ViewerPayload.incytr ? ViewerPayload.incytr() : null;
}

function _temporalBaseline() {
  const axis = ViewerPayload.contrastAxis ? ViewerPayload.contrastAxis() : {};
  const block = _temporalActiveIncytrBlock();
  if (axis.baseline) return axis.baseline;
  if (block && block.timepoints && block.timepoints.length) return block.timepoints[0];
  const tps = axis.timepoints || [];
  return tps.length ? tps[0] : "d2";
}

function _temporalPathwayDays() {
  const axis = ViewerPayload.contrastAxis ? ViewerPayload.contrastAxis() : {};
  const block = _temporalActiveIncytrBlock();
  if (block && block.diseases && block.diseases.length) return _temporalSortDays(block.diseases);
  if (axis.groups && axis.groups.length) return _temporalSortDays(axis.groups);
  if (block && block.contrasts && block.contrasts.length) {
    return _temporalSortDays(block.contrasts.map(c => String(c).split("_", 1)[0]));
  }
  return [];
}

function _temporalKinaseDays() {
  const K = ViewerPayload.kinases ? ViewerPayload.kinases() : {};
  const out = [];
  for (const key of Object.keys(K || {})) {
    const m = key.match(/^NES_(d\d+)$/);
    if (m && K["FDR_" + m[1]]) out.push(m[1]);
  }
  return _temporalSortDays(out);
}

function _temporalAxisDays(series) {
  const days = [];
  const wantKinase = series.some(s => s.layer === "kinase");
  const wantPathway = series.some(s => s.layer === "pathway");
  if (wantPathway) days.push(..._temporalPathwayDays());
  if (wantKinase) days.push(..._temporalKinaseDays());
  if (!days.length) days.push(..._temporalPathwayDays(), ..._temporalKinaseDays());
  return _temporalSortDays(days);
}

function _temporalLayerAvailable(layer) {
  if (layer === "pathway") return !!_temporalPathwayBlock();
  if (layer === "kinase") {
    const caps = _temporalActiveCapabilities();
    return !!caps.kinases && _temporalKinaseDays().length > 0;
  }
  return false;
}

function _temporalSnapPathwayPvalue(p) {
  const block = _temporalPathwayBlock();
  if (!block || !block.thresholds || !block.thresholds.length) return null;
  const thr = block.thresholds;
  if (p == null || !isFinite(p)) return { value: null, index: thr.length - 1, open: true };
  let idx = -1;
  for (let i = 0; i < thr.length; i++) if (thr[i] <= p) idx = i;
  if (idx < 0) idx = 0;
  return { value: thr[idx], index: idx, open: false };
}

function _temporalSnapPathwayAbsPds(ap) {
  const block = _temporalPathwayBlock();
  if (!block || !block.abs_pds_thresholds || !block.abs_pds_thresholds.length) return null;
  const thr = block.abs_pds_thresholds;
  const v = (ap == null || !isFinite(ap)) ? 0 : ap;
  let idx = 0;
  for (let i = 0; i < thr.length; i++) if (thr[i] <= v) idx = i;
  return { value: thr[idx], index: idx };
}

function _temporalPathwayCountAt(cIdx, signBucket, pThrIdx, apThrIdx) {
  const block = _temporalPathwayBlock();
  if (!block) return 0;
  const shape = block.shape || [];
  if (shape.length === 3) {
    const [nC, nS, nT] = shape;
    if (cIdx < 0 || cIdx >= nC || signBucket < 0 || signBucket >= nS
        || pThrIdx < 0 || pThrIdx >= nT) return 0;
    return block.counts[cIdx * nS * nT + signBucket * nT + pThrIdx] || 0;
  }
  const [nC, nS, nT, nAP] = shape;
  const ap = (apThrIdx == null) ? 0 : apThrIdx;
  if (cIdx < 0 || cIdx >= nC || signBucket < 0 || signBucket >= nS
      || pThrIdx < 0 || pThrIdx >= nT || ap < 0 || ap >= nAP) return 0;
  return block.counts[cIdx * nS * nT * nAP
                    + signBucket * nT * nAP
                    + pThrIdx * nAP
                    + ap] || 0;
}

function _temporalEmptyCounts(days) {
  const out = {};
  for (const d of days) out[d] = { up: 0, down: 0, total: 0, upIds: [], downIds: [], totalIds: [] };
  return out;
}

function _temporalPathwayCounts(series, days) {
  const counts = _temporalEmptyCounts(days);
  const block = _temporalPathwayBlock();
  if (!block) return counts;
  const snap = _temporalSnapPathwayPvalue(series.pvalue);
  if (!snap) return counts;
  const apSnap = _temporalSnapPathwayAbsPds(series.absPds);
  const pThrIdx = snap.index;
  const apThrIdx = apSnap ? apSnap.index : 0;
  const baseline = _temporalBaseline();
  const cIdxOf = new Map();
  (block.contrasts || []).forEach((c, i) => cIdxOf.set(c, i));
  for (const day of days) {
    const cIdx = cIdxOf.get(day + "_" + baseline);
    if (cIdx == null) continue;
    const down = _temporalPathwayCountAt(cIdx, 0, pThrIdx, apThrIdx);
    const zero = _temporalPathwayCountAt(cIdx, 1, pThrIdx, apThrIdx);
    const up = _temporalPathwayCountAt(cIdx, 2, pThrIdx, apThrIdx);
    counts[day].up = up;
    counts[day].down = down;
    counts[day].total = (series.sign === "up") ? up
      : (series.sign === "down" ? down : up + zero + down);
  }
  return counts;
}

function _temporalKinaseCounts(series, days) {
  const counts = _temporalEmptyCounts(days);
  const K = ViewerPayload.kinases ? ViewerPayload.kinases() : {};
  const ids = K.id || [];
  const fdrGate = (series.fdr == null || !isFinite(series.fdr)) ? 0.25 : series.fdr;
  for (const day of days) {
    const nesCol = K["NES_" + day];
    const fdrCol = K["FDR_" + day];
    if (!nesCol || !fdrCol) continue;
    for (let i = 0; i < ids.length; i++) {
      const fdr = fdrCol[i];
      const nes = nesCol[i];
      if (fdr == null || !isFinite(fdr) || fdr >= fdrGate) continue;
      if (nes == null || !isFinite(nes) || nes === 0) continue;
      const sign = nes > 0 ? 1 : -1;
      if (series.sign === "up" && sign < 0) continue;
      if (series.sign === "down" && sign > 0) continue;
      counts[day].total++;
      counts[day].totalIds.push(ids[i]);
      if (sign > 0) {
        counts[day].up++;
        counts[day].upIds.push(ids[i]);
      } else {
        counts[day].down++;
        counts[day].downIds.push(ids[i]);
      }
    }
  }
  return counts;
}

function _temporalCounts(series, days) {
  if (series.layer === "kinase") return _temporalKinaseCounts(series, days);
  return _temporalPathwayCounts(series, days);
}

function _temporalSeriesLabel(series) {
  if (series.layer === "kinase") {
    const parts = ["Kinase MEA", `FDR<${series.fdr}`];
    if (series.sign !== "signed") parts.push(series.sign);
    return parts.join(" · ");
  }
  const parts = ["Pathway (Incytr)"];
  const snap = _temporalSnapPathwayPvalue(series.pvalue);
  if (snap && !snap.open) parts.push(`pvalue<${snap.value}`);
  const apSnap = _temporalSnapPathwayAbsPds(series.absPds);
  if (apSnap && apSnap.value > 0) parts.push(`|PDS|>=${apSnap.value}`);
  if (window.IncytrCelltypeQc) {
    const block = _temporalActiveIncytrBlock();
    const txt = IncytrCelltypeQc.enabled(block) ? IncytrCelltypeQc.controlText(block) : "";
    if (txt) parts.push(txt);
  }
  if (series.sign !== "signed") parts.push(series.sign);
  return parts.join(" · ");
}

function _temporalLayerOptions(activeLayer) {
  const kinaseAvailable = _temporalLayerAvailable("kinase");
  const opts = [
    { value: "pathway", label: "pathway (Incytr)", disabled: false },
    {
      value: "kinase",
      label: kinaseAvailable ? "kinase MEA" : "kinase MEA (donor1 only)",
      disabled: !kinaseAvailable && activeLayer !== "kinase",
    },
  ];
  return opts.map(o => {
    const dis = o.disabled ? " disabled" : "";
    return `<option value="${_escapeHtml(o.value)}"${dis}>${_escapeHtml(o.label)}</option>`;
  }).join("");
}

function _temporalRenderSeriesRow(series, idx) {
  const isPathway = series.layer === "pathway";
  const signOpts = [
    ["signed", "signed (up/down)"],
    ["up", "up only"],
    ["down", "down only"],
    ["either", "either (total)"],
  ].map(([v, l]) => `<option value="${_escapeHtml(v)}">${_escapeHtml(l)}</option>`).join("");
  const sparse = isPathway && window.IncytrCelltypeQc
    && IncytrCelltypeQc.hasLowSignal(_temporalActiveIncytrBlock());
  const unavailable = !_temporalLayerAvailable(series.layer);
  return `<div class="temporal-row${unavailable ? " temporal-row-unavailable" : ""}" data-idx="${idx}">
    <span class="temporal-label">Series ${idx + 1}</span>
    <label>Layer <select class="temporal-layer">${_temporalLayerOptions(series.layer)}</select></label>
    <label>Sign <select class="temporal-sign">${signOpts}</select></label>
    ${series.layer === "kinase" ? `<label title="Kinase MEA false-discovery-rate gate.">FDR <input class="temporal-fdr" type="number" min="0" max="1" step="0.01" style="width:54px;"></label>` : ""}
    ${isPathway ? `<label title="Incytr pathway pvalue gate. Blank = no pvalue gate.">pvalue <input class="temporal-pvalue" type="number" min="0" max="1" step="0.005" style="width:64px;"></label>` : ""}
    ${isPathway ? `<label title="Minimum absolute composite Pathway Disturbance Score.">|PDS| >= <input class="temporal-abs-pds" type="number" min="0" step="0.01" style="width:64px;"></label>` : ""}
    ${sparse ? `<label title="Sensitivity mode: remove Incytr pathways where sender or receiver has median n_cells <= 3.">Sparse <select class="temporal-low-signal"><option value="include">Include all</option><option value="exclude">Exclude median n<=3</option></select></label>` : ""}
    <button class="temporal-rm" title="Remove this series">x</button>
  </div>`;
}

function _temporalWireSeriesRow(rowEl, idx) {
  const s = _temporalState.series[idx];
  const layerSel = rowEl.querySelector(".temporal-layer");
  const signSel = rowEl.querySelector(".temporal-sign");
  const fdrEl = rowEl.querySelector(".temporal-fdr");
  const pvalueEl = rowEl.querySelector(".temporal-pvalue");
  const absPdsEl = rowEl.querySelector(".temporal-abs-pds");
  const lowSignalEl = rowEl.querySelector(".temporal-low-signal");
  const rmBtn = rowEl.querySelector(".temporal-rm");
  layerSel.value = s.layer;
  signSel.value = s.sign;
  if (fdrEl) fdrEl.value = s.fdr;
  if (pvalueEl) pvalueEl.value = (s.pvalue == null ? "" : s.pvalue);
  if (absPdsEl) absPdsEl.value = (s.absPds == null ? "" : s.absPds);
  if (lowSignalEl) lowSignalEl.value = IncytrFilter.get("excludeLowSignalCelltypes") ? "exclude" : "include";
  layerSel.addEventListener("change", () => {
    s.layer = layerSel.value;
    if (s.layer === "pathway" && s.absPds == null) s.absPds = 0.01;
    if (s.layer === "kinase" && s.fdr == null) s.fdr = 0.25;
    _temporalRenderUI();
    renderTemporal();
  });
  signSel.addEventListener("change", () => { s.sign = signSel.value; renderTemporal(); });
  if (fdrEl) fdrEl.addEventListener("change", () => {
    const v = parseFloat(fdrEl.value);
    if (isFinite(v) && v > 0 && v <= 1) {
      s.fdr = v;
      renderTemporal();
    } else {
      fdrEl.value = s.fdr;
    }
  });
  if (pvalueEl) pvalueEl.addEventListener("change", () => {
    if (pvalueEl.value === "") {
      s.pvalue = null;
      renderTemporal();
      return;
    }
    const v = parseFloat(pvalueEl.value);
    if (isFinite(v) && v > 0 && v <= 1) {
      s.pvalue = v;
      renderTemporal();
    } else {
      pvalueEl.value = (s.pvalue == null ? "" : s.pvalue);
    }
  });
  if (absPdsEl) absPdsEl.addEventListener("change", () => {
    const v = absPdsEl.value === "" ? 0 : parseFloat(absPdsEl.value);
    if (isFinite(v) && v >= 0) {
      s.absPds = v;
      renderTemporal();
    } else {
      absPdsEl.value = (s.absPds == null ? "" : s.absPds);
    }
  });
  if (lowSignalEl) lowSignalEl.addEventListener("change", () => {
    IncytrFilter.set({ excludeLowSignalCelltypes: lowSignalEl.value === "exclude" });
    _temporalRenderUI();
    renderTemporal();
  });
  rmBtn.addEventListener("click", () => {
    _temporalState.series.splice(idx, 1);
    if (!_temporalState.series.length) _temporalState.series.push(_temporalDefaultSeries("pathway"));
    _temporalRenderUI();
    renderTemporal();
  });
}

function _temporalRenderUI() {
  const list = document.getElementById("temporal-series-list");
  if (!list) return;
  _temporalEnsureState();
  list.innerHTML = _temporalState.series.map((s, i) => _temporalRenderSeriesRow(s, i)).join("");
  list.querySelectorAll(".temporal-row").forEach((row, i) => _temporalWireSeriesRow(row, i));
}

function _temporalResetSeries() {
  _temporalEnsureState();
  _temporalState.series = [_temporalDefaultSeries("pathway")];
  _temporalRenderUI();
  renderTemporal();
}

function wireTemporal() {
  _temporalEnsureState();
  _temporalRenderUI();
  const addBtn = document.getElementById("temporal-add-series");
  if (addBtn) addBtn.addEventListener("click", () => {
    _temporalState.series.push(_temporalDefaultSeries("pathway"));
    _temporalRenderUI();
    renderTemporal();
  });
  const clearBtn = document.getElementById("temporal-clear");
  if (clearBtn) clearBtn.addEventListener("click", () => _temporalResetSeries());
  const shareCb = document.getElementById("temporal-share-y");
  if (shareCb) {
    shareCb.checked = !!_temporalState.shareY;
    shareCb.addEventListener("change", () => {
      _temporalState.shareY = shareCb.checked;
      renderTemporal();
    });
  }
}

function renderTemporal() {
  const el = document.getElementById("temporal-plot");
  const sub = document.getElementById("temporal-subtitle");
  if (!el) return;
  _temporalEnsureState();
  const series = (_temporalState.series || []).slice();
  const days = _temporalAxisDays(series);
  if (!series.length || !days.length) {
    Plotly.purge(el);
    if (sub) sub.textContent = "No temporal data for the active donor.";
    return;
  }
  _temporalRenderUI();
  const allCounts = series.map(s => _temporalCounts(s, days));
  const traces = [];
  const layout = {
    grid: { rows: series.length, columns: 1, pattern: "independent" },
    margin: { l: 70, r: 20, t: 24, b: 54 },
    height: Math.max(240, 190 * series.length + 48),
    barmode: "group",
    bargap: 0.28,
    showlegend: false,
    annotations: [],
  };
  let sharedRange = null;
  if (_temporalState.shareY) {
    let lo = 0, hi = 0;
    for (let s = 0; s < series.length; s++) {
      const ser = series[s];
      const counts = allCounts[s];
      for (const d of days) {
        const c = counts[d];
        if (ser.sign === "signed") {
          if (c.up > hi) hi = c.up;
          if (-c.down < lo) lo = -c.down;
        } else if (c.total > hi) {
          hi = c.total;
        }
      }
    }
    const pad = Math.max(1, Math.ceil(Math.max(hi, -lo) * 0.05));
    sharedRange = [lo - (lo < 0 ? pad : 0), hi + pad];
  }
  const colors = {
    pathway: "#4f7fbf",
    kinase: "#7a8b2f",
  };
  const opacity = {
    pathway: 0.82,
    kinase: 0.9,
  };
  for (let si = 0; si < series.length; si++) {
    const ser = series[si];
    const counts = allCounts[si];
    const sfx = si === 0 ? "" : String(si + 1);
    const xAxis = "x" + sfx;
    const yAxis = "y" + sfx;
    const color = colors[ser.layer] || "#607d8b";
    const offsetGroup = "temporal-" + si;
    const clickHint = ser.layer === "kinase" ? "click to open Kinase tab" : "click to open Incytr Pathways";
    if (ser.sign === "signed") {
      traces.push({
        type: "bar",
        name: _temporalSeriesLabel(ser) + " up",
        x: days,
        y: days.map(d => counts[d].up),
        marker: { color, opacity: opacity[ser.layer] || 0.85 },
        offsetgroup: offsetGroup,
        alignmentgroup: "temporal-series-" + si,
        xaxis: xAxis,
        yaxis: yAxis,
        showlegend: false,
        customdata: days.map(d => [d, counts[d].up, counts[d].upIds]),
        meta: { series: si, layer: ser.layer, sign: "up" },
        hovertemplate: `[S${si + 1}] %{x} up: %{customdata[1]} · ${clickHint}<extra></extra>`,
      });
      traces.push({
        type: "bar",
        name: _temporalSeriesLabel(ser) + " down",
        x: days,
        y: days.map(d => -counts[d].down),
        marker: { color, opacity: 0.45 },
        offsetgroup: offsetGroup,
        alignmentgroup: "temporal-series-" + si,
        xaxis: xAxis,
        yaxis: yAxis,
        showlegend: false,
        customdata: days.map(d => [d, counts[d].down, counts[d].downIds]),
        meta: { series: si, layer: ser.layer, sign: "down" },
        hovertemplate: `[S${si + 1}] %{x} down: %{customdata[1]} · ${clickHint}<extra></extra>`,
      });
    } else {
      traces.push({
        type: "bar",
        name: _temporalSeriesLabel(ser),
        x: days,
        y: days.map(d => counts[d].total),
        marker: { color, opacity: opacity[ser.layer] || 0.85 },
        offsetgroup: offsetGroup,
        alignmentgroup: "temporal-series-" + si,
        xaxis: xAxis,
        yaxis: yAxis,
        showlegend: false,
        customdata: days.map(d => [d, counts[d].total, counts[d].totalIds]),
        meta: { series: si, layer: ser.layer, sign: "total" },
        hovertemplate: `[S${si + 1}] %{x}: %{customdata[1]} · ${clickHint}<extra></extra>`,
      });
    }
    layout["xaxis" + sfx] = {
      title: si === series.length - 1 ? "Day" : "",
      anchor: yAxis,
    };
    layout["yaxis" + sfx] = {
      title: ser.layer === "kinase" ? "n kinases" : "n pathways",
      zeroline: true,
      anchor: xAxis,
    };
    if (ser.sign === "signed") {
      layout["yaxis" + sfx].zerolinecolor = "#000";
      layout["yaxis" + sfx].zerolinewidth = 1;
    }
    if (sharedRange) layout["yaxis" + sfx].range = sharedRange;
    layout.annotations.push({
      xref: "paper",
      yref: "paper",
      x: 0,
      xanchor: "left",
      y: 1 - (si / series.length) - 0.02 / series.length,
      yanchor: "top",
      text: `<b>S${si + 1}</b> · ${_escapeHtml(_temporalSeriesLabel(ser))}`,
      showarrow: false,
      font: { size: 11, color: "#37474f" },
    });
  }
  Plotly.react(el, traces, layout, { displaylogo: false, responsive: true });
  el.removeAllListeners && el.removeAllListeners("plotly_click");
  el.on && el.on("plotly_click", ev => {
    if (!ev.points || !ev.points.length) return;
    const p = ev.points[0];
    const meta = (p.data && p.data.meta) || {};
    const ser = series[meta.series] || {};
    const cd = p.customdata || [];
    const day = cd[0] || p.x;
    if (ser.layer === "kinase") {
      _temporalOpenKinases(day, cd[2] || [], meta.sign, ser);
    } else {
      _temporalOpenPathways(day, ser);
    }
  });
  if (sub) {
    const ctx = ViewerPayload.contextRecord ? ViewerPayload.contextRecord() : {};
    const baseline = _temporalBaseline();
    const pieces = [`${ctx.label || ViewerPayload.activeContext()}`, `${series.length} series`, `pathway baseline ${baseline}`];
    if (!_temporalLayerAvailable("kinase")) pieces.push("MEA unavailable for this donor");
    sub.textContent = pieces.join(" · ");
  }
}

function _temporalOpenPathways(day, series) {
  if (typeof IncytrFilter === "undefined") return;
  const snap = _temporalSnapPathwayPvalue(series.pvalue);
  const apSnap = _temporalSnapPathwayAbsPds(series.absPds);
  IncytrFilter.set({
    pair: null,
    senderIn: [],
    receiverIn: [],
    disease: day ? [day] : [],
    timepoint: [_temporalBaseline()],
    sliderP: (snap && !snap.open) ? snap.value : null,
    sliderPds: apSnap && apSnap.value > 0 ? apSnap.value : null,
  });
  if (typeof _incytrPane !== "undefined") _incytrPane = "table";
  Store.dispatch({ type: "SET_VIEW", key: "activeTab", value: "incytr" });
}

function _temporalOpenKinases(day, kinaseIds, sign, series) {
  if (typeof KinaseFilter === "undefined" || !KinaseFilter.setWhitelist) return;
  KinaseFilter.set({
    day: day ? [day] : [],
    sign: sign === "total" ? "" : sign,
    fdr: series.fdr || 0.25,
  });
  KinaseFilter.setWhitelist(
    Array.isArray(kinaseIds) ? kinaseIds.slice() : [],
    `Temporal · Kinase MEA · ${day}${sign === "total" ? "" : " · " + sign}`
  );
  KinaseFilter.setWhitelistStack(false);
  Store.dispatch({ type: "SET_VIEW", key: "activeTab", value: "kinase" });
  if (typeof _syncKinaseFilterUI === "function") setTimeout(_syncKinaseFilterUI, 0);
}

// ---------------------------------------------------------------------------
// Kinase Explorer tab
// ---------------------------------------------------------------------------

// Single filter-state object replacing scattered module-level vars and
// window._keFilters. Backed by a T-cell-specific localStorage key.
// The Song/AD viewer also has a KinaseFilter; keep the keys separate because
// both viewers are often served from the same localhost origin.
window.KinaseFilter = (function() {
  const _KEY = "tcell.kinaseFilter.v8";
  const _defaults = {
    search: "",
    day: [],          // [] = any; otherwise MEA contrast labels such as "d13".
    disease: [],      // Legacy/shared handoff slot; not exposed in T-cell MEA.
    timepoint: [],    // Legacy/shared handoff slot; not exposed in T-cell MEA.
    celltype: [],     // [] = any; within-cohort attributed cell-type states.
    confidence: "",   // Legacy/shared handoff slot; not exposed in T-cell MEA.
    nSigMin: 0,       // minimum n_sig (count of significant contrasts in scope)
    tcellMin: 0,      // 0 = Any (off, default); 1.5/2/3 = opt-in minimum state
                      // enrichment (fold over the kinase's baseline mean state).
                      // Enrichment only — concordance is never gated
                      // (docs/tcell_exhaustion_analysis_summary.md).
    sign: "",         // "" | "up" | "down" over selected day contrasts.
    pattern: "",      // TrendFilter value over ordered donor-day NES contrasts.
    fdr: 0.25, sortCol: "nes_profile", sortAsc: false,
  };
  const _arrKeys = new Set(["day","disease","timepoint","celltype"]);
  let _state = Object.assign({}, _defaults);
  try {
    const saved = JSON.parse(localStorage.getItem(_KEY) || "null");
    if (saved && typeof saved === "object") {
      for (const k of Object.keys(_defaults)) {
        if (k in saved) {
          if (_arrKeys.has(k)) _state[k] = Array.isArray(saved[k]) ? saved[k].slice() : [];
          else _state[k] = saved[k];
        }
      }
    }
  } catch(e) {}
  const _subs = [];
  // Whitelist is in-memory only (NOT persisted) — survives tab switches but not
  // page reloads. Set by cross-tab handoffs (e.g. Temporal bar click); when
  // active, the kinase explorer bypasses attribution / n_sig / confidence gates
  // and shows exactly the listed kinase IDs.
  let _whitelist = null;       // null | Set<number>
  let _whitelistLabel = "";    // human-readable source description
  let _whitelistStack = false; // false = whitelist bypass; true = AND with dropdowns
  function _save() {
    try { localStorage.setItem(_KEY, JSON.stringify(_state)); } catch(e) {}
  }
  return {
    get: function(k) { return k ? _state[k] : Object.assign({}, _state); },
    getWhitelist: function() {
      return _whitelist ? { ids: _whitelist, label: _whitelistLabel,
                            stack: _whitelistStack } : null;
    },
    setWhitelist: function(ids, label) {
      _whitelist = new Set(ids);
      _whitelistLabel = label || "";
      for (const fn of _subs) fn();
    },
    clearWhitelist: function() {
      if (_whitelist === null) return;
      _whitelist = null; _whitelistLabel = ""; _whitelistStack = false;
      for (const fn of _subs) fn();
    },
    setWhitelistStack: function(on) {
      const v = !!on;
      if (_whitelistStack === v) return;
      _whitelistStack = v;
      for (const fn of _subs) fn();
    },
    set: function(patch) {
      let changed = false;
      for (const k of Object.keys(patch)) {
        const nv = patch[k];
        if (_arrKeys.has(k)) {
          const cur = _state[k] || [];
          const a = Array.isArray(nv) ? nv.slice() : [];
          if (cur.length !== a.length || cur.some((v,i) => v !== a[i])) {
            _state[k] = a; changed = true;
          }
        } else if (_state[k] !== nv) {
          _state[k] = nv; changed = true;
        }
      }
      if (changed) { _save(); for (const fn of _subs) fn(); }
    },
    reset: function() {
      _state = JSON.parse(JSON.stringify(_defaults));
      _state.fdr = Store.state.filters.fdr || 0.25;
      _save();
      for (const fn of _subs) fn();
    },
    subscribe: function(fn) { _subs.push(fn); },
  };
})();

// Back-compat shim so any code reading window._keFilters still works.
// Multiselect: collapse to single selection if exactly one chosen, else "".
Object.defineProperty(window, "_keFilters", {
  get: function() {
    const f = KinaseFilter.get();
    const one = a => (Array.isArray(a) && a.length === 1) ? a[0] : "";
    return { disease: one(f.disease), tp: one(f.day) || one(f.timepoint),
             celltype: one(f.celltype), trajectory: f.pattern || "" };
  },
  configurable: true,
});

let _keRows = null;
let _keSigFdr = null;
let _kinaseIdxById = null;
let _decompByKey = null;
let _decompByKinCtx = null;
let _agreementByKey = null;
const _AGREEMENT_STATE_NAMES = ["neither_sig","agree","mixed","disagree","bulk_only","decomp_only"];

// ---------------------------------------------------------------------------
// Scoped attribution helpers (single source of truth: PAYLOAD.attribution_index)
// ---------------------------------------------------------------------------

// Coerce a filter dimension value to a Set of selected values.
// Accepts: undefined/null/"" → empty (any), string (single) → {string},
// array → set of array entries.
