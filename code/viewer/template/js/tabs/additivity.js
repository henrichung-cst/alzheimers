
const _axSuf = (k) => (k === 0) ? "" : String(k + 1);
const _newAcc = () => ({ n:0, sx:0, sy:0, sxx:0, syy:0, sxy:0 });
function _accAdd(a, x, y) {
  a.n++; a.sx += x; a.sy += y; a.sxx += x*x; a.syy += y*y; a.sxy += x*y;
}
function _pearson(a) {
  if (a.n < 3) return { r: null, n: a.n };
  const num = a.n*a.sxy - a.sx*a.sy;
  const den = Math.sqrt((a.n*a.sxx - a.sx*a.sx) * (a.n*a.syy - a.sy*a.sy));
  return { r: den > 0 ? num/den : null, n: a.n };
}

function _addTimepointsInScope() {
  const tp = Store.state.view.additivityTimepoint;
  const TPS = META.timepoints;
  return (tp === "ALL") ? TPS.slice() : [tp];
}

function _addDiagonalShapes(tps, xRange) {
  const shapes = [];
  const annotations = [];
  for (let k = 0; k < tps.length; k++) {
    const s = _axSuf(k);
    shapes.push({
      type: "line", xref: "x" + s, yref: "y" + s,
      x0: xRange[0], x1: xRange[1], y0: xRange[0], y1: xRange[1],
      line: { color: "#888", width: 1, dash: "dash" },
    });
    annotations.push({
      xref: "x" + s + " domain", yref: "y" + s + " domain",
      x: 0.03, y: 0.97, xanchor: "left", yanchor: "top", showarrow: false,
      text: "Synergistic", font: { size: 10, color: "#888" },
    });
    annotations.push({
      xref: "x" + s + " domain", yref: "y" + s + " domain",
      x: 0.97, y: 0.03, xanchor: "right", yanchor: "bottom", showarrow: false,
      text: "Sub-additive", font: { size: 10, color: "#888" },
    });
    annotations.push({
      xref: "x" + s + " domain", yref: "y" + s + " domain",
      x: 0.5, y: 1.08, xanchor: "center", yanchor: "bottom", showarrow: false,
      text: "<b>" + tps[k] + "</b>", font: { size: 13 },
    });
  }
  return { shapes, annotations };
}

function _addAxesLayout(tps, axRange, xTitle, yTitle) {
  const layout = {
    margin: { l: 60, r: 20, t: 40, b: 50 },
    grid: { rows: 1, columns: tps.length, pattern: "independent" },
    height: 520, hovermode: "closest",
  };
  for (let k = 0; k < tps.length; k++) {
    const s = _axSuf(k);
    layout["xaxis" + s] = { title: xTitle, range: axRange, zeroline: true };
    layout["yaxis" + s] = { title: (k === 0) ? yTitle : "",
                             range: axRange, zeroline: true,
                             scaleanchor: "x" + s, scaleratio: 1 };
  }
  return layout;
}

function _addCategory(fApp, fTau, fApTt, thresh) {
  const sApp = (fApp != null && fApp < thresh);
  const sTau = (fTau != null && fTau < thresh);
  const sAp  = (fApTt != null && fApTt < thresh);
  const n = (sApp ? 1 : 0) + (sTau ? 1 : 0) + (sAp ? 1 : 0);
  if (n === 0) return null;
  if (n >= 2) return "Multi";
  if (sApp) return "App only";
  if (sTau) return "Tau only";
  return "ApTt only";
}

function _writeStats(stats, tps, accs) {
  if (!stats) return;
  stats.textContent = tps.map((t, k) => {
    const r = _pearson(accs[k]);
    return `${t}: n=${r.n}, Pearson r=${r.r == null ? "–" : r.r.toFixed(3)}`;
  }).join("  ·  ");
}

function renderAdditivityKinase() {
  const el = document.getElementById("add-plot");
  const sub = document.getElementById("add-subtitle");
  const stats = document.getElementById("add-stats");
  const K = PAYLOAD.kinases;
  const fdr = Store.state.filters.fdr;
  const recv = Store.state.filters.receiver;
  const n = K.id.length;
  const tps = _addTimepointsInScope();

  const buckets = tps.map(() => {
    const b = {};
    for (const c of _ADD_CATEGORIES) b[c] = { x: [], y: [], text: [], customdata: [] };
    return b;
  });
  const accs = tps.map(_newAcc);
  let xMin = -0.1, xMax = 0.1, yMin = -0.1, yMax = 0.1;

  for (let k = 0; k < tps.length; k++) {
    const t = tps[k];
    const nAppCol = K["NES_App_" + t],  nTauCol = K["NES_Tau_" + t],  nApCol = K["NES_ApTt_" + t];
    const fAppCol = K["FDR_App_" + t],  fTauCol = K["FDR_Tau_" + t],  fApCol = K["FDR_ApTt_" + t];
    if (!nAppCol || !nTauCol || !nApCol) continue;
    const bucket = buckets[k];
    const acc = accs[k];
    for (let i = 0; i < n; i++) {
      if (recv !== "ALL" && K.top_celltype_1 && K.top_celltype_1[i] !== recv) continue;
      const nApp = nAppCol[i], nTau = nTauCol[i], nAp = nApCol[i];
      if (nApp == null || nTau == null || nAp == null) continue;
      const fApp = fAppCol[i], fTau = fTauCol[i], fAp = fApCol[i];
      const x = nApp + nTau, y = nAp;
      const cat = _addCategory(fApp, fTau, fAp, fdr);
      if (cat == null) continue;
      const b = bucket[cat];
      b.x.push(x); b.y.push(y);
      b.text.push(K.name[i]);
      b.customdata.push([nApp, nTau, nAp, fApp, fTau, fAp]);
      _accAdd(acc, x, y);
      if (x < xMin) xMin = x; if (x > xMax) xMax = x;
      if (y < yMin) yMin = y; if (y > yMax) yMax = y;
    }
  }

  const axRange = [Math.min(xMin, yMin) - 0.2, Math.max(xMax, yMax) + 0.2];
  const traces = [];
  for (let k = 0; k < tps.length; k++) {
    const s = _axSuf(k);
    for (const cat of _ADD_CATEGORIES) {
      const b = buckets[k][cat];
      if (!b.x.length) continue;
      traces.push({
        type: "scattergl", mode: "markers", name: cat,
        legendgroup: cat, showlegend: (k === 0),
        x: b.x, y: b.y, text: b.text, customdata: b.customdata,
        xaxis: "x" + s, yaxis: "y" + s,
        marker: { color: _ADD_COLORS[cat], size: 7, opacity: 0.75,
                  line: { width: 0.5, color: "#fff" } },
        hovertemplate:
          "<b>%{text}</b><br>App NES: %{customdata[0]:.2f} (q=%{customdata[3]:.2g})" +
          "<br>Tau NES: %{customdata[1]:.2f} (q=%{customdata[4]:.2g})" +
          "<br>ApTt NES: %{customdata[2]:.2f} (q=%{customdata[5]:.2g})" +
          "<br>Pred (App+Tau): %{x:.2f}<br>Obs (ApTt): %{y:.2f}<extra></extra>",
      });
    }
  }
  const { shapes, annotations } = _addDiagonalShapes(tps, axRange);
  const layout = _addAxesLayout(tps, axRange, "App + Tau NES", "ApTt NES (observed)");
  layout.showlegend = true;
  layout.legend = { orientation: "h", y: -0.18 };
  layout.shapes = shapes;
  layout.annotations = annotations;
  Plotly.react(el, traces, layout, { displaylogo: false, responsive: true });

  _writeStats(stats, tps, accs);
  if (sub) sub.textContent =
    `Kinase level · predicted = App NES + Tau NES · observed = ApTt NES · FDR < ${fdr}` +
    (recv !== "ALL" ? ` · receiver=${recv}` : "");
}

function renderAdditivityBackbone() {
  const el = document.getElementById("add-plot");
  const sub = document.getElementById("add-subtitle");
  const stats = document.getElementById("add-stats");
  const BB = PAYLOAD.backbones;
  const tps = _addTimepointsInScope();
  const idx = getFilteredIndices();

  let sampleIdx = idx;
  let thinned = false;
  if (idx.length > _ADD_BACKBONE_MAX_POINTS) {
    const stride = idx.length / _ADD_BACKBONE_MAX_POINTS;
    sampleIdx = new Int32Array(_ADD_BACKBONE_MAX_POINTS);
    for (let j = 0; j < _ADD_BACKBONE_MAX_POINTS; j++)
      sampleIdx[j] = idx[Math.floor(j * stride)];
    thinned = true;
  }

  const perTp = tps.map(() => ({ x: [], y: [] }));
  const accs = tps.map(_newAcc);
  let xMin = 0, xMax = 0, yMin = 0, yMax = 0;
  for (let k = 0; k < tps.length; k++) {
    const t = tps[k];
    const oApp = BB["observed_score_App_" + t];
    const oTau = BB["observed_score_Tau_" + t];
    const oAp  = BB["observed_score_ApTt_" + t];
    if (!oApp || !oTau || !oAp) continue;
    const dst = perTp[k];
    const acc = accs[k];
    const sMin = Math.max(0, Number(Store.state.view.additivityScoreMin) || 0);
    for (let j = 0; j < sampleIdx.length; j++) {
      const i = sampleIdx[j];
      const a = oApp[i], tv = oTau[i], av = oAp[i];
      if (a == null || tv == null || av == null) continue;
      if (sMin > 0 && (a < sMin && tv < sMin)) continue;
      const x = a + tv;
      dst.x.push(x); dst.y.push(av);
      _accAdd(acc, x, av);
      if (x < xMin) xMin = x; if (x > xMax) xMax = x;
      if (av < yMin) yMin = av; if (av > yMax) yMax = av;
    }
  }
  const axRange = [Math.min(xMin, yMin) * 1.05 - 0.1,
                   Math.max(xMax, yMax) * 1.05 + 0.1];

  const traces = [];
  for (let k = 0; k < tps.length; k++) {
    const s = _axSuf(k);
    const p = perTp[k];
    const npts = p.x.length;
    const mSize = npts > 10000 ? 3 : npts > 2000 ? 5 : 8;
    const mOpacity = npts > 10000 ? 0.35 : npts > 2000 ? 0.5 : 0.75;
    traces.push({
      type: "scattergl", mode: "markers", name: tps[k], showlegend: false,
      x: p.x, y: p.y, xaxis: "x" + s, yaxis: "y" + s,
      marker: { color: "#2e86ab", size: mSize, opacity: mOpacity },
      hovertemplate: "Pred: %{x:.3f}<br>Obs: %{y:.3f}<extra></extra>",
    });
  }
  const { shapes, annotations } = _addDiagonalShapes(tps, axRange);
  const layout = _addAxesLayout(tps, axRange, "App + Tau score", "ApTt score (observed)");
  layout.showlegend = false;
  layout.shapes = shapes;
  layout.annotations = annotations;
  Plotly.react(el, traces, layout, { displaylogo: false, responsive: true });

  _writeStats(stats, tps, accs);
  if (sub) sub.textContent =
    `Backbone level · ${idx.length.toLocaleString()} in current filter` +
    (thinned ? ` (showing ${_ADD_BACKBONE_MAX_POINTS.toLocaleString()} sampled)` : "") +
    ` · predicted = App + Tau observed_score · observed = ApTt observed_score.`;
}

function renderAdditivity() {
  const el = document.getElementById("add-plot");
  if (!el) return;
  const level = Store.state.view.additivityLevel;
  if (level === "kinase") renderAdditivityKinase();
  else renderAdditivityBackbone();
}

function wireAdditivityControls() {
  const levelSel = document.getElementById("add-level");
  const tpSel = document.getElementById("add-tp");
  if (!levelSel || !tpSel) return;
  levelSel.value = Store.state.view.additivityLevel;
  tpSel.value = Store.state.view.additivityTimepoint;
  levelSel.addEventListener("change", ev =>
    Store.dispatch({type:"SET_VIEW", key:"additivityLevel", value: ev.target.value}));
  tpSel.addEventListener("change", ev =>
    Store.dispatch({type:"SET_VIEW", key:"additivityTimepoint", value: ev.target.value}));
  const scoreInp = document.getElementById("add-score-min");
  if (scoreInp) {
    scoreInp.value = Store.state.view.additivityScoreMin || 0;
    scoreInp.addEventListener("change", ev =>
      Store.dispatch({type:"SET_VIEW", key:"additivityScoreMin",
                      value: Math.max(0, parseFloat(ev.target.value) || 0)}));
  }
}

// ---------------------------------------------------------------------------
// Temporal v2 — series builder (draft)
// ---------------------------------------------------------------------------
// Each series is a predicate over (kinase, contrast). Bar height = unique
// kinases passing per (genotype, timepoint), split by NES sign when
// requested. Series stack as small multiples (one row per series) so y-scales
// stay independent.
let _tv2State = null;
let _tv2DecompCellsCache = null;
let _tv2AttrTierByKinCtx = null;  // Map<`${kid}|${cidx}`, Array<{cell, rank}>>

function _tv2EnsureAttrIndex() {
  if (_tv2AttrTierByKinCtx) return;
  _ensureKinaseIndexes();
  const AI = PAYLOAD.attribution_index || {kinase_id:[]};
  const m = new Map();
  for (let j = 0; j < AI.kinase_id.length; j++) {
    const kid = AI.kinase_id[j];
    const cidx = AI.contrast_id[j];
    const cell = AI.cell_type[j];
    const tier = _combinedTierFor(kid, cidx, cell, AI.combined_confidence[j]);
    const rank = _CONF_RANK[tier] || 0;
    const key = kid + "|" + cidx;
    let arr = m.get(key);
    if (!arr) { arr = []; m.set(key, arr); }
    arr.push({ cell, rank });
  }
  _tv2AttrTierByKinCtx = m;
}

function _tv2AttrPasses(ctxKey, cellsScope, threshold) {
  // Returns true if at least one attribution row at (kid, contrastIdx) within
  // the cell-type scope reaches the requested tier rank. threshold "" → pass.
  if (!threshold) return true;
  const wantRank = _CONF_RANK[threshold] || 0;
  if (wantRank <= 0) return true;
  const arr = _tv2AttrTierByKinCtx.get(ctxKey);
  if (!arr) return false;
  for (const r of arr) {
    if (cellsScope !== "ALL" && r.cell !== cellsScope) continue;
    if (r.rank >= wantRank) return true;
  }
  return false;
}

function _tv2DecompCellTypes() {
  if (_tv2DecompCellsCache) return _tv2DecompCellsCache;
  const D = PAYLOAD.decomposition_index || {cell_type:[]};
  const s = new Set();
  for (const c of D.cell_type) s.add(c);
  _tv2DecompCellsCache = Array.from(s).sort();
  return _tv2DecompCellsCache;
}

function _tv2DefaultSeries(layer) {
  return {
    layer: layer || "bulk",
    cells: "ALL",
    sign: "signed",
    fdrBulk: 0.25,
    fdrDecomp: 0.25,
    agree: true,
    attrTier: "",   // "" any | "low" | "moderate" | "high" | "very_high"
  };
}

function _tv2InitState() {
  if (_tv2State) return;
  _tv2State = { series: [_tv2DefaultSeries("bulk")], shareY: false };
}

function _tv2Eval(series, kid, contrastIdx) {
  // Returns null if the kinase fails the predicate at this contrast,
  // else { sign: -1|0|+1 } based on bulk NES (or decomp NES when bulk absent).
  const K = PAYLOAD.kinases;
  const cName = CONTRASTS[contrastIdx];
  const bulkNesCol = K["NES_" + cName];
  const bulkFdrCol = K["FDR_" + cName];
  const bulkNes = bulkNesCol ? bulkNesCol[kid] : null;
  const bulkFdr = bulkFdrCol ? bulkFdrCol[kid] : null;
  const bulkSig = bulkFdr != null && isFinite(bulkFdr) && bulkFdr < series.fdrBulk;

  const ctxKey = kid + "|" + contrastIdx;
  let dRows = (_decompByKinCtx && _decompByKinCtx.get(ctxKey)) || [];
  if (series.cells !== "ALL") dRows = dRows.filter(r => r.cell_type === series.cells);
  // For each decomp row at this kinase × contrast: sig + sign-vs-bulk.
  let decompAnyPass = false;        // any decomp row sig at fdrDecomp (sign-agnostic)
  let decompAgreePass = false;      // ≥1 decomp row sig AND sign-matches bulk
  let decompDisagreePass = false;   // ≥1 decomp row sig AND sign-disagrees with bulk
  let decompSignNes = null;
  for (const r of dRows) {
    if (r.fdr == null || !isFinite(r.fdr) || r.fdr >= series.fdrDecomp) continue;
    if (r.nes == null || !isFinite(r.nes) || r.nes === 0) continue;
    decompAnyPass = true;
    if (decompSignNes == null || Math.abs(r.nes) > Math.abs(decompSignNes)) {
      decompSignNes = r.nes;
    }
    if (bulkNes != null && bulkNes !== 0) {
      if ((r.nes > 0) === (bulkNes > 0)) decompAgreePass = true;
      else decompDisagreePass = true;
    }
  }

  let pass, refNes;
  if (series.layer === "bulk") { pass = bulkSig; refNes = bulkNes; }
  else if (series.layer === "decomp") { pass = decompAnyPass; refNes = decompSignNes; }
  else if (series.layer === "intersect") {
    pass = bulkSig && (series.agree ? decompAgreePass : decompAnyPass);
    refNes = bulkNes;
  }
  else if (series.layer === "contested") { pass = bulkSig && decompDisagreePass; refNes = bulkNes; }
  else if (series.layer === "diff") { pass = bulkSig && !decompAnyPass; refNes = bulkNes; }
  else { pass = false; refNes = null; }
  if (!pass) return null;
  // Attribution-tier gate: applies to any series, scoped to the same cells set.
  if (series.attrTier && !_tv2AttrPasses(ctxKey, series.cells, series.attrTier)) {
    return null;
