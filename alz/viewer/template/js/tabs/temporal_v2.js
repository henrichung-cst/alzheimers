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
    pvalue: null,   // only used when layer === "pathway"; null = no pvalue gate
    absPds: 0.01,   // only used when layer === "pathway"; default = "real composite effect"
  };
}

function _tv2PathwayBlock() {
  const block = ViewerPayload.incytr();
  if (!block) return null;
  if (window.IncytrCelltypeQc
      && IncytrCelltypeQc.enabled(block)
      && block.pathway_counts_low_signal_excluded) {
    return block.pathway_counts_low_signal_excluded;
  }
  return block.pathway_counts || null;
}

function _tv2HasLowSignalPathwayFilter() {
  const block = ViewerPayload.incytr();
  return !!(window.IncytrCelltypeQc && IncytrCelltypeQc.hasLowSignal(block));
}

function _tv2LowSignalPathwayLabel() {
  const block = ViewerPayload.incytr();
  return (window.IncytrCelltypeQc && IncytrCelltypeQc.enabled(block))
    ? IncytrCelltypeQc.controlText(block) : "";
}

function _tv2SnapPathwayPvalue(p) {
  // Snaps user pvalue down to the nearest precomputed threshold. Null /
  // non-finite opens the gate (largest threshold in grid) — pvalue is opt-in
  // because per-animal SigProb Wald-t is unreliable in this cohort, so
  // |PDS| is the recommended primary filter.
  const block = _tv2PathwayBlock();
  if (!block || !block.thresholds || !block.thresholds.length) return null;
  const thr = block.thresholds;
  if (p == null || !isFinite(p)) {
    return { value: null, index: thr.length - 1, open: true };
  }
  let idx = -1;
  for (let i = 0; i < thr.length; i++) if (thr[i] <= p) idx = i;
  if (idx < 0) idx = 0;
  return { value: thr[idx], index: idx, open: false };
}

function _tv2SnapPathwayAbsPds(ap) {
  // Snaps user |PDS| DOWN to the nearest precomputed threshold. The grid is
  // a ≥-gate: threshold 0 → keep everything, 0.5 → only the strongest paths.
  // 4D payloads carry abs_pds_thresholds; older 3D payloads (no |PDS| axis)
  // return null and the count lookup falls back to "no effect-size filter".
  const block = _tv2PathwayBlock();
  if (!block || !block.abs_pds_thresholds || !block.abs_pds_thresholds.length) {
    return null;
  }
  const thr = block.abs_pds_thresholds;
  const v = (ap == null || !isFinite(ap)) ? 0 : ap;
  let idx = 0;
  for (let i = 0; i < thr.length; i++) {
    if (thr[i] <= v) idx = i;
  }
  return { value: thr[idx], index: idx };
}

function _tv2PathwayCountAt(cIdx, signBucket, pThrIdx, apThrIdx) {
  // Returns count at (contrast, sign, pvalue-threshold, |PDS|-threshold).
  // Back-compat: 3D payloads (no |PDS| axis) ignore apThrIdx — equivalent
  // to apThrIdx=0 (|PDS|>=0, no filter).
  const block = _tv2PathwayBlock();
  if (!block) return 0;
  const shape = block.shape;
  if (shape.length === 3) {
    const [nC, nS, nT] = shape;
    if (cIdx < 0 || cIdx >= nC || signBucket < 0 || signBucket >= nS
        || pThrIdx < 0 || pThrIdx >= nT) return 0;
    return block.counts[cIdx * nS * nT + signBucket * nT + pThrIdx];
  }
  const [nC, nS, nT, nAP] = shape;
  const ap = (apThrIdx == null) ? 0 : apThrIdx;
  if (cIdx < 0 || cIdx >= nC || signBucket < 0 || signBucket >= nS
      || pThrIdx < 0 || pThrIdx >= nT || ap < 0 || ap >= nAP) return 0;
  return block.counts[cIdx * nS * nT * nAP
                    + signBucket * nT * nAP
                    + pThrIdx * nAP
                    + ap];
}

function _tv2YUnit(series) {
  return series.layer === "pathway" ? "n pathways" : "n kinases";
}

function _tv2Axis() {
  const axis = ViewerPayload.contrastAxis();
  return {
    groups: axis.groups.length ? axis.groups : (META.diseaseGroups || []),
    timepoints: axis.timepoints.length ? axis.timepoints : (META.timepoints || []),
  };
}

function _tv2InitState() {
  if (_tv2State) return;
  _tv2State = { series: [_tv2DefaultSeries("bulk")], shareY: false };
}

function _tv2Eval(series, kid, contrastIdx) {
  // Returns null if the kinase fails the predicate at this contrast,
  // else { sign: -1|0|+1 } based on bulk NES (or decomp NES when bulk absent).
  const K = ViewerPayload.kinases();
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
  }

  const sign = (refNes == null || refNes === 0) ? 0 : (refNes > 0 ? 1 : -1);
  if (series.sign === "up" && sign < 0) return null;
  if (series.sign === "down" && sign > 0) return null;
  return { sign };
}
function _tv2Counts(series) {
  // Returns counts[g][t] = { up, down, total, upIds, downIds, totalIds } of unique kinases.
  // Pathway layer: same shape, but units are n_pathways and Ids are empty
  // (pathways have no kinase-ID payload; clicks route to the Incytr tab).
  if (series.layer === "pathway") return _tv2PathwayCounts(series);
  _tv2EnsureAttrIndex();
  const K = ViewerPayload.kinases();
  const axis = _tv2Axis();
  const DG = axis.groups;
  const TPS = axis.timepoints;
  const counts = {};
  // Hoist (g, t) → contrast-index lookup out of the per-kinase loop.
  const gtPairs = [];
  for (const g of DG) {
    counts[g] = {};
    for (const t of TPS) {
      counts[g][t] = {
        up: 0, down: 0, total: 0,
        upIds: [], downIds: [], totalIds: [],
      };
      const cIdx = CONTRASTS.indexOf(g + "_" + t);
      if (cIdx >= 0) gtPairs.push({ g, t, cIdx });
    }
  }
  const n = K.id.length;
  for (let i = 0; i < n; i++) {
    const kid = K.id[i];
    for (const p of gtPairs) {
      const r = _tv2Eval(series, kid, p.cIdx);
      if (!r) continue;
      const cell = counts[p.g][p.t];
      cell.total++;
      cell.totalIds.push(kid);
      if (r.sign > 0) { cell.up++; cell.upIds.push(kid); }
      else if (r.sign < 0) { cell.down++; cell.downIds.push(kid); }
    }
  }
  return counts;
}

function _tv2PathwayCounts(series) {
  // Pathway counts per (g, t) using the precomputed (contrast, sign,
  // pvalue-threshold, |PDS|-threshold) grid. up/down come from sign(PDS) —
  // the composite Pathway Disturbance Score, which aggregates the factorial
  // OLS β across all available omics layers. Rows with PDS == 0 or
  // PDS IS NULL go to "total" but neither up nor down.
  const axis = _tv2Axis();
  const DG = axis.groups;
  const TPS = axis.timepoints;
  const counts = {};
  for (const g of DG) {
    counts[g] = {};
    for (const t of TPS) {
      counts[g][t] = { up: 0, down: 0, total: 0,
                       upIds: [], downIds: [], totalIds: [] };
    }
  }
  const block = _tv2PathwayBlock();
  if (!block) return counts;
  const snap = _tv2SnapPathwayPvalue(series.pvalue);
  if (!snap) return counts;
  const pThrIdx = snap.index;
  const apSnap = _tv2SnapPathwayAbsPds(series.absPds);
  const apThrIdx = apSnap ? apSnap.index : 0;
  const block_contrasts = block.contrasts;
  const cIdxOf = new Map();
  for (let i = 0; i < block_contrasts.length; i++) cIdxOf.set(block_contrasts[i], i);
  for (const g of DG) {
    for (const t of TPS) {
      const cIdx = cIdxOf.get(g + "_" + t);
      if (cIdx == null) continue;
      const down = _tv2PathwayCountAt(cIdx, 0, pThrIdx, apThrIdx);
      const zero = _tv2PathwayCountAt(cIdx, 1, pThrIdx, apThrIdx);
      const up   = _tv2PathwayCountAt(cIdx, 2, pThrIdx, apThrIdx);
      const cell = counts[g][t];
      if (series.sign === "up")        cell.total = up;
      else if (series.sign === "down") cell.total = down;
      else                              cell.total = up + zero + down;
      cell.up = up;
      cell.down = down;
    }
  }
  return counts;
}

function _tv2SeriesLabel(series) {
  const layerLabels = { bulk: "Bulk", decomp: "Decomp",
                         intersect: "Bulk ∩ Decomp", contested: "Bulk vs Decomp (contested)",
                         diff: "Bulk \\ Decomp",
                         pathway: "Pathway (Incytr)" };
  const parts = [layerLabels[series.layer] || series.layer];
  if (series.layer === "pathway") {
    const snap = _tv2SnapPathwayPvalue(series.pvalue);
    if (snap && !snap.open) parts.push(`pvalue<${snap.value}`);
    const apSnap = _tv2SnapPathwayAbsPds(series.absPds);
    if (apSnap && apSnap.value > 0) parts.push(`|PDS|≥${apSnap.value}`);
    const lowTxt = _tv2LowSignalPathwayLabel();
    if (lowTxt) parts.push(lowTxt);
    if (series.sign !== "signed") parts.push(series.sign);
    return parts.join(" · ");
  }
  if (series.layer !== "bulk") {
    parts.push(series.cells === "ALL" ? "any cell type" : series.cells);
  }
  if (series.layer !== "decomp") parts.push(`bulk FDR<${series.fdrBulk}`);
  if (series.layer !== "bulk") parts.push(`decomp FDR<${series.fdrDecomp}`);
  if (series.layer === "intersect") parts.push(series.agree ? "sign agree" : "any sign");
  if (series.attrTier) {
    const lbl = { very_high: "attr=very_high", high: "attr≥high",
                  moderate: "attr≥moderate", low: "attr≥low" };
    parts.push(lbl[series.attrTier] || ("attr≥" + series.attrTier));
  }
  if (series.sign !== "signed") parts.push(series.sign);
  return parts.join(" · ");
}

function _tv2RenderSeriesRow(series, idx) {
  const cells = _tv2DecompCellTypes();
  const cellOpts = ['<option value="ALL">any (OR)</option>']
    .concat(cells.map(c => `<option value="${c}">${c}</option>`)).join("");
  const layerOpts = [
    ['bulk', 'bulk'], ['decomp', 'decomp'],
    ['intersect', 'bulk ∩ decomp (corroborated)'],
    ['contested', 'bulk ∩ decomp (contested)'],
    ['diff', 'bulk \\ decomp (bulk-only)'],
    ['pathway', 'pathway (Incytr)'],
  ].map(([v, l]) => `<option value="${v}">${l}</option>`).join("");
  const signOpts = [
    ['signed', 'signed (up/down)'], ['up', 'up only'],
    ['down', 'down only'], ['either', 'either (total)'],
  ].map(([v, l]) => `<option value="${v}">${l}</option>`).join("");
  const attrOpts = [
    ['', 'Any'], ['very_high', 'very high (only)'], ['high', 'high+'],
    ['moderate', 'moderate+'], ['low', 'low+'],
  ].map(([v, l]) => `<option value="${v}">${l}</option>`).join("");
  const isPathway = (series.layer === "pathway");
  const cellsDisabled = (series.layer === "bulk") || isPathway;
  const agreeDisabled = (series.layer !== "intersect");
  const showBulkFdr = !isPathway && (series.layer !== "decomp");
  const showDecompFdr = !isPathway && (series.layer !== "bulk");
  const showPathwayPvalue = isPathway;
  const showPathwayLowSignal = isPathway && _tv2HasLowSignalPathwayFilter();
  const showAttr = !isPathway;
  const showAgree = !isPathway;
  const disParts = [];
  if (cellsDisabled) disParts.push("cells");
  if (agreeDisabled) disParts.push("agree");
  const disAttr = disParts.length ? ` data-disabled="${disParts.join(' ')}"` : '';
  return `<div class="tv2-row" data-idx="${idx}"${disAttr}>
    <span class="tv2-label">Series ${idx + 1}</span>
    <label>Layer <select class="tv2-layer">${layerOpts}</select></label>
    <label class="tv2-cells">Cells <select class="tv2-cells-sel">${cellOpts}</select></label>
    <label>Sign <select class="tv2-sign">${signOpts}</select></label>
    ${showBulkFdr ? `<label>Bulk FDR<input class="tv2-fdr-bulk" type="number" min="0" max="1" step="0.01" style="width:54px;"></label>` : ''}
    ${showDecompFdr ? `<label>Decomp FDR<input class="tv2-fdr-decomp" type="number" min="0" max="1" step="0.01" style="width:54px;"></label>` : ''}
    ${showPathwayPvalue ? `<label title="Incytr pathway pvalue (Wald t-test from factorial OLS on per-animal SigProb). Blank = no pvalue gate (default). Snaps down to: 0.001 / 0.005 / 0.01 / 0.05 / 0.1 / 0.25 / 0.5 / 1.0.">pvalue<input class="tv2-pvalue" type="number" min="0" max="1" step="0.005" style="width:64px;"></label>` : ''}
    ${showPathwayPvalue ? `<label title="Minimum |PDS| — magnitude of the composite Pathway Disturbance Score (multimodel β across all omics layers). ≥0.01 = real composite signal; ≥0.1 = strong. Snaps down to the precomputed grid: 0 / 0.001 / 0.01 / 0.05 / 0.1 / 0.25 / 0.5 / 1.0.">|PDS|≥<input class="tv2-abs-pds" type="number" min="0" step="0.01" style="width:64px;"></label>` : ''}
    ${showPathwayLowSignal ? `<label title="Sensitivity mode: remove Incytr pathways where sender or receiver has median n_cells <= 3.">Sparse <select class="tv2-low-signal"><option value="include">Include all</option><option value="exclude">Exclude median n≤3</option></select></label>` : ''}
    ${showAttr ? `<label title="Require ≥1 attribution row in scope reaching this confidence tier (very_high = high+decomp agree, high = WMB+concordance, etc.).">Attr <select class="tv2-attr">${attrOpts}</select></label>` : ''}
    ${showAgree ? `<label class="tv2-agree" title="When on, decomp row must match bulk NES sign to count as corroboration."><input class="tv2-agree-cb" type="checkbox"> sign agree</label>` : ''}
    <button class="tv2-rm" title="Remove this series">×</button>
  </div>`;
}

function _tv2WireSeriesRow(rowEl, idx) {
  const s = _tv2State.series[idx];
  const layerSel = rowEl.querySelector(".tv2-layer");
  const cellsSel = rowEl.querySelector(".tv2-cells-sel");
  const signSel = rowEl.querySelector(".tv2-sign");
  const fdrB = rowEl.querySelector(".tv2-fdr-bulk");
  const fdrD = rowEl.querySelector(".tv2-fdr-decomp");
  const pvalueEl = rowEl.querySelector(".tv2-pvalue");
  const absPdsEl = rowEl.querySelector(".tv2-abs-pds");
  const lowSignalEl = rowEl.querySelector(".tv2-low-signal");
  const agreeCb = rowEl.querySelector(".tv2-agree-cb");
  const attrSel = rowEl.querySelector(".tv2-attr");
  const rmBtn = rowEl.querySelector(".tv2-rm");
  layerSel.value = s.layer;
  cellsSel.value = s.cells;
  signSel.value = s.sign;
  if (fdrB) fdrB.value = s.fdrBulk;
  if (fdrD) fdrD.value = s.fdrDecomp;
  if (pvalueEl) pvalueEl.value = (s.pvalue == null ? "" : s.pvalue);
  if (absPdsEl) absPdsEl.value = (s.absPds == null ? "" : s.absPds);
  if (lowSignalEl) lowSignalEl.value = IncytrFilter.get("excludeLowSignalCelltypes") ? "exclude" : "include";
  if (agreeCb) agreeCb.checked = !!s.agree;
  if (attrSel) attrSel.value = s.attrTier || "";
  layerSel.addEventListener("change", () => {
    s.layer = layerSel.value;
    if (s.layer === "bulk" || s.layer === "pathway") s.cells = "ALL";
    _tv2RenderUI(); renderTemporalV2();
  });
  cellsSel.addEventListener("change", () => { s.cells = cellsSel.value; renderTemporalV2(); });
  signSel.addEventListener("change", () => { s.sign = signSel.value; renderTemporalV2(); });
  if (fdrB) fdrB.addEventListener("change", () => {
    const v = parseFloat(fdrB.value); if (isFinite(v) && v > 0 && v <= 1) {
      s.fdrBulk = v; renderTemporalV2();
    } else { fdrB.value = s.fdrBulk; }
  });
  if (fdrD) fdrD.addEventListener("change", () => {
    const v = parseFloat(fdrD.value); if (isFinite(v) && v > 0 && v <= 1) {
      s.fdrDecomp = v; renderTemporalV2();
    } else { fdrD.value = s.fdrDecomp; }
  });
  if (pvalueEl) pvalueEl.addEventListener("change", () => {
    if (pvalueEl.value === "") { s.pvalue = null; renderTemporalV2(); return; }
    const v = parseFloat(pvalueEl.value);
    if (isFinite(v) && v > 0 && v <= 1) {
      s.pvalue = v;
      renderTemporalV2();
    } else { pvalueEl.value = (s.pvalue == null ? "" : s.pvalue); }
  });
  if (absPdsEl) absPdsEl.addEventListener("change", () => {
    const raw = absPdsEl.value === "" ? 0 : parseFloat(absPdsEl.value);
    if (isFinite(raw) && raw >= 0) {
      s.absPds = raw;
      renderTemporalV2();
    } else { absPdsEl.value = (s.absPds == null ? "" : s.absPds); }
  });
  if (lowSignalEl) lowSignalEl.addEventListener("change", () => {
    IncytrFilter.set({ excludeLowSignalCelltypes: lowSignalEl.value === "exclude" });
    renderTemporalV2();
  });
  if (agreeCb) agreeCb.addEventListener("change", () => { s.agree = agreeCb.checked; renderTemporalV2(); });
  if (attrSel) attrSel.addEventListener("change", () => {
    s.attrTier = attrSel.value; renderTemporalV2();
  });
  rmBtn.addEventListener("click", () => {
    _tv2State.series.splice(idx, 1);
    if (_tv2State.series.length === 0) _tv2State.series.push(_tv2DefaultSeries("bulk"));
    _tv2RenderUI(); renderTemporalV2();
  });
}

function _tv2RenderUI() {
  const list = document.getElementById("tv2-series-list");
  if (!list) return;
  list.innerHTML = _tv2State.series.map((s, i) => _tv2RenderSeriesRow(s, i)).join("");
  list.querySelectorAll(".tv2-row").forEach((row, i) => _tv2WireSeriesRow(row, i));
}

function _tv2ApplyPreset(name) {
  const cells = _tv2DecompCellTypes();
  if (name === "bulk_only") {
    _tv2State.series = [_tv2DefaultSeries("bulk")];
  } else if (name === "bulk_corrob_contest") {
    const corrob = _tv2DefaultSeries("intersect"); corrob.agree = true;
    const contest = _tv2DefaultSeries("contested");
    _tv2State.series = [_tv2DefaultSeries("bulk"), corrob, contest];
  } else if (name === "bulk_vs_decomp") {
    _tv2State.series = [_tv2DefaultSeries("bulk"), _tv2DefaultSeries("decomp")];
  } else if (name === "bulk_attr_vs_decomp") {
    const bulkAttr = _tv2DefaultSeries("bulk"); bulkAttr.attrTier = "high";
    _tv2State.series = [
      _tv2DefaultSeries("bulk"),
      bulkAttr,
      _tv2DefaultSeries("decomp"),
    ];
  } else if (name === "celltype_sweep") {
    _tv2State.series = cells.slice(0, Math.min(4, cells.length)).map(c => {
      const s = _tv2DefaultSeries("decomp"); s.cells = c; return s;
    });
    if (_tv2State.series.length === 0) _tv2State.series = [_tv2DefaultSeries("decomp")];
  }
  _tv2RenderUI();
  renderTemporalV2();
}

function renderTemporalV2() {
  const el = document.getElementById("tv2-plot");
  const sub = document.getElementById("tv2-subtitle");
  if (!el) return;
  _ensureKinaseIndexes();
  const series = _tv2State ? _tv2State.series : [];
  if (!series.length) {
    Plotly.purge(el);
    if (sub) sub.textContent = "No series defined. Click + Add series or pick a preset.";
    return;
  }
  const axis = _tv2Axis();
  const DG = axis.groups;
  const TPS = axis.timepoints;
  const traces = [];
  const layout = {
    grid: { rows: series.length, columns: 1, pattern: "independent" },
    margin: { l: 70, r: 20, t: 20, b: 50 },
    height: Math.max(220, 200 * series.length + 40),
    barmode: "group", bargap: 0.25,
    legend: { orientation: "h", y: -0.1 / series.length },
    annotations: [],
  };
  // First pass: compute counts per series and (if shared y) the global range.
  const allCounts = series.map(ser => _tv2Counts(ser));
  let sharedRange = null;
  if (_tv2State.shareY) {
    let lo = 0, hi = 0;
    for (let s = 0; s < series.length; s++) {
      const ser = series[s];
      const counts = allCounts[s];
      for (const g of DG) for (const t of TPS) {
        const cell = counts[g][t];
        if (ser.sign === "signed") {
          if (cell.up > hi) hi = cell.up;
          if (-cell.down < lo) lo = -cell.down;
        } else {
          if (cell.total > hi) hi = cell.total;
        }
      }
    }
    const pad = Math.max(1, Math.ceil(Math.max(hi, -lo) * 0.05));
    sharedRange = [lo - (lo < 0 ? pad : 0), hi + pad];
  }
  for (let s = 0; s < series.length; s++) {
    const ser = series[s];
    const counts = allCounts[s];
    const sfx = (s === 0) ? "" : String(s + 1);
    const xAxis = "x" + sfx, yAxis = "y" + sfx;
    const showLegend = (s === 0);
    const clickHint = (ser.layer === "pathway")
      ? "click to open in Incytr Pathways tab"
      : "click to open in Kinase tab";
    for (const g of DG) {
      const color = (META.diseaseColors || {})[g] || "#555";
      if (ser.sign === "signed") {
        traces.push({
          type: "bar", name: g + " up",
          x: TPS, y: TPS.map(t => counts[g][t].up),
          marker: { color }, legendgroup: g + "-up",
          offsetgroup: g, alignmentgroup: "v" + s,
          xaxis: xAxis, yaxis: yAxis, showlegend: showLegend,
          customdata: TPS.map(t => [counts[g][t].up, counts[g][t].upIds]),
          meta: { s, g, sign: "up" },
          hovertemplate: `[S${s+1}] ${g} up @ %{x}: %{customdata[0]} · ${clickHint}<extra></extra>`,
        });
        traces.push({
          type: "bar", name: g + " down",
          x: TPS, y: TPS.map(t => -counts[g][t].down),
          marker: { color, opacity: 0.55 }, legendgroup: g + "-down",
          offsetgroup: g, alignmentgroup: "v" + s,
          xaxis: xAxis, yaxis: yAxis, showlegend: showLegend,
          customdata: TPS.map(t => [counts[g][t].down, counts[g][t].downIds]),
          meta: { s, g, sign: "down" },
          hovertemplate: `[S${s+1}] ${g} down @ %{x}: %{customdata[0]} · ${clickHint}<extra></extra>`,
        });
      } else {
        traces.push({
          type: "bar", name: g,
          x: TPS, y: TPS.map(t => counts[g][t].total),
          marker: { color }, legendgroup: g,
          offsetgroup: g, alignmentgroup: "v" + s,
          xaxis: xAxis, yaxis: yAxis, showlegend: showLegend,
          customdata: TPS.map(t => [counts[g][t].total, counts[g][t].totalIds]),
          meta: { s, g, sign: "total" },
          hovertemplate: `[S${s+1}] ${g} @ %{x}: %{customdata[0]} · ${clickHint}<extra></extra>`,
        });
      }
    }
    layout["xaxis" + sfx] = {
      title: (s === series.length - 1) ? "Timepoint" : "",
      anchor: "y" + sfx,
    };
    layout["yaxis" + sfx] = {
      title: _tv2YUnit(ser),
      zeroline: true,
      anchor: "x" + sfx,
    };
    if (ser.sign === "signed") {
      layout["yaxis" + sfx].zerolinecolor = "#000";
      layout["yaxis" + sfx].zerolinewidth = 1;
    }
    if (sharedRange) layout["yaxis" + sfx].range = sharedRange;
    layout.annotations.push({
      xref: "paper", yref: "paper",
      x: 0, xanchor: "left",
      y: 1 - (s / series.length) - 0.02 / series.length,
      yanchor: "top",
      text: `<b>S${s + 1}</b> · ${_tv2SeriesLabel(ser)}`,
      showarrow: false, font: { size: 11, color: "#37474f" },
    });
  }
  Plotly.react(el, traces, layout, { displaylogo: false, responsive: true });
  el.removeAllListeners && el.removeAllListeners("plotly_click");
  el.on && el.on("plotly_click", ev => {
    if (!ev.points || !ev.points.length) return;
    const p = ev.points[0];
    const cd = p.customdata;
    const meta = (p.data && p.data.meta) || {};
    const ser = (_tv2State.series || [])[meta.s] || {};
    if (ser.layer === "pathway") {
      _openIncytrPathwaysFromBar(ser, meta.g, p.x);
      return;
    }
    if (!cd || !Array.isArray(cd) || !cd[1] || !cd[1].length) return;
    const ids = cd[1];
    const label = `Temporal v2 · ${_tv2SeriesLabel(ser)} · ${meta.g}_${p.x}` +
                  (meta.sign === "total" ? "" : ` · ${meta.sign}`);
    _openKinaseDeepDiveWithWhitelist(ids, label, {
      genotype: meta.g, timepoint: p.x,
      cells: ser.cells, attrTier: ser.attrTier,
    });
  });
  if (sub) {
    sub.textContent = `${series.length} series · y depends on layer (kinase / pathway) · `
      + `signed series split up at +y, down at −y · scales independent across rows · `
      + `click kinase bars → Kinase deep dive · click pathway bars → Incytr Pathways tab.`;
  }
}

function _openIncytrPathwaysFromBar(series, genotype, timepoint) {
  if (typeof IncytrFilter === "undefined") return;
  const snap = _tv2SnapPathwayPvalue(series.pvalue);
  const apSnap = _tv2SnapPathwayAbsPds(series.absPds);
  IncytrFilter.set({
    pair:       null,
    senderIn:   [],
    receiverIn: [],
    disease:    genotype  ? [genotype]  : [],
    timepoint:  timepoint ? [timepoint] : [],
    sliderP:    (snap && !snap.open) ? snap.value : null,
    sliderPds:  apSnap && apSnap.value > 0 ? apSnap.value : null,
  });
  Store.dispatch({type:"SET_VIEW", key:"activeTab", value:"incytrpathways"});
}

function _openKinaseDeepDiveWithWhitelist(kinaseIds, sourceLabel, ctx) {
  if (typeof KinaseFilter === "undefined" || !KinaseFilter.setWhitelist) {
    console.warn("KinaseFilter whitelist not available");
    return;
  }
  // Prefill the filter dropdowns from the bar's context so the user can see
  // and edit the implied scope. The whitelist is stored separately and ANDs
  // with these filters when "Stack with filters" is toggled on; otherwise
  // the dropdowns are visible-but-inactive (the whitelist takes precedence).
  if (ctx) {
    const patch = {
      disease:    ctx.genotype  ? [ctx.genotype]  : [],
      timepoint:  ctx.timepoint ? [ctx.timepoint] : [],
      celltype:   (ctx.cells && ctx.cells !== "ALL") ? [ctx.cells] : [],
      confidence: ctx.attrTier || "",
      // n_sig isn't part of the bar context; leave whatever the user had.
    };
    KinaseFilter.set(patch);
  }
  // New whitelists default to bypass mode (stack=false) so the user sees the
  // full clicked set first, then opts into stacking with the toggle.
  KinaseFilter.setWhitelist(kinaseIds.slice(), sourceLabel);
  KinaseFilter.setWhitelistStack(false);
  Store.dispatch({type:"SET_VIEW", key:"activeTab", value:"kinase"});
  // Push the prefilled state into the visible toolbar inputs after the tab
  // has been swapped in. Defer to next frame because syncTabsFromStore runs
  // synchronously inside the dispatch handler and may unhide the panel.
  if (typeof _syncKinaseFilterUI === "function") {
    setTimeout(_syncKinaseFilterUI, 0);
  }
}

function wireTemporalV2() {
  _tv2InitState();
  _tv2RenderUI();
  document.querySelectorAll("#tv2-presets [data-tv2-preset]").forEach(btn => {
    btn.addEventListener("click", () => _tv2ApplyPreset(btn.dataset.tv2Preset));
  });
  const addBtn = document.getElementById("tv2-add-series");
  if (addBtn) addBtn.addEventListener("click", () => {
    _tv2State.series.push(_tv2DefaultSeries("bulk"));
    _tv2RenderUI(); renderTemporalV2();
  });
  const clrBtn = document.getElementById("tv2-clear");
  if (clrBtn) clrBtn.addEventListener("click", () => {
    _tv2State.series = [_tv2DefaultSeries("bulk")];
    _tv2RenderUI(); renderTemporalV2();
  });
  const shareCb = document.getElementById("tv2-share-y");
  if (shareCb) {
    shareCb.checked = !!_tv2State.shareY;
    shareCb.addEventListener("change", () => {
      _tv2State.shareY = shareCb.checked;
      renderTemporalV2();
    });
  }
}

// ---------------------------------------------------------------------------
// Kinase Explorer tab
// ---------------------------------------------------------------------------

// Single filter-state object replacing scattered module-level vars and
// window._keFilters. Backed by localStorage key kinaseFilter.v5.
// (v1/v2/v3 keys are intentionally ignored — schema changed.)
window.KinaseFilter = (function() {
  const _KEY = "kinaseFilter.v5";
  const _defaults = {
    search: "",
    disease: [],      // [] = any; otherwise array of "App"|"Tau"|"ApTt"
    timepoint: [],    // [] = any; otherwise array of "2mo"|"4mo"|"6mo"
    celltype: [],     // [] = any; otherwise array of subclass strings
    confidence: "",   // "" | "high" | "moderate" | "low" — ordinal threshold (≥)
    nSigMin: 0,       // minimum n_sig (count of significant contrasts in scope)
    wmbMin: 0,        // 0 = any; 1/2/5/10 = minimum WMB specificity tier (× uniform)
    pattern: "",      // TrendFilter value over disease-specific NES time courses.
    fdr: 0.25, sortCol: "nes_profile", sortAsc: false,
  };
  const _arrKeys = new Set(["disease","timepoint","celltype"]);
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
  // page reloads. Set by cross-tab handoffs (e.g. Temporal v2 bar click); when
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
    return { disease: one(f.disease), tp: one(f.timepoint),
             celltype: one(f.celltype), trajectory: f.pattern || "" };
  },
  configurable: true,
});

let _keRows = null;
let _keSigFdr = null;
let _kinaseIdxById = null;
let _evidenceByKinase = null;
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
