function _filterSet(v) {
  const s = new Set();
  if (v == null || v === "") return s;
  if (Array.isArray(v)) { for (const x of v) if (x !== "" && x != null) s.add(x); return s; }
  s.add(v);
  return s;
}

function getScopedContrastIds(filter) {
  // Returns Set of contrast indices matching the filter's disease × timepoint
  // selection sets. Empty set on a dimension = any.
  const ds = _filterSet(filter.disease);
  const ts = _filterSet(filter.timepoint);
  const ids = new Set();
  for (let ci = 0; ci < CONTRASTS.length; ci++) {
    const c = CONTRASTS[ci];
    const d = c.split("_")[0];
    const m = c.match(/_(\d+mo)$/);
    const t = m ? m[1] : "";
    if (ds.size && !ds.has(d)) continue;
    if (ts.size && !ts.has(t)) continue;
    ids.add(ci);
  }
  return ids;
}

// Confidence threshold: a row passes if its tier rank ≥ requested rank.
// "" / undefined → no constraint.
const _CONF_RANK = {very_high: 4, high: 3, moderate: 2, low: 1, none: 0};
function _confPass(rowConf, threshold) {
  if (!threshold) return true;
  return (_CONF_RANK[rowConf] || 0) >= (_CONF_RANK[threshold] || 0);
}

// WMB specificity tiers as multiples of the even-split baseline. WMB specificity
// is a share normalized over the retained WMB classes the spine maps onto (~11),
// so the honest uniform is 1/N_retained, read canonically from meta.wmb_uniform
// (matches the crosstable; see docs/plans/specificity_validation_2026-06-05.md §6).
const _WMB_UNIFORM = (typeof PAYLOAD !== "undefined" && PAYLOAD.meta && PAYLOAD.meta.wmb_uniform) || (1 / 11);
const _WMB_TIER_VALUES = [10, 5, 2, 1];
function _wmbTier(s) {
  if (s == null || !isFinite(s)) return 0;
  for (const t of _WMB_TIER_VALUES) {
    if (s >= t * _WMB_UNIFORM) return t;
  }
  return 0;
}
function _wmbTierLabel(t) { return t > 0 ? "≥" + t + "×" : ""; }
function _wmbTierBadge(t) {
  if (!t) return '<span class="muted">—</span>';
  const cls = t >= 10 ? "vhi" : (t >= 5 ? "hi" : (t >= 2 ? "mid" : "lo"));
  return '<span class="badge ' + cls + '" title="WMB specificity ≥ ' + t +
         '× uniform (' + (t * _WMB_UNIFORM).toFixed(3) + ')">' + _wmbTierLabel(t) + '</span>';
}

// Decomp step ordinal vs bulk MEA direction:
//   3 strong-agree (FDR<0.10), 2 sig-agree (FDR<0.25), 1 nominal,
//   0 absent, -2 sig-disagree (FDR<0.25, sign opposes bulk).
// bulkNes may be passed in pre-fetched; otherwise read from the active context.
function _decompStep(decompNes, decompFdr, bulkNes) {
  if (decompNes == null || !isFinite(decompNes) || decompNes === 0) return 0;
  if (bulkNes == null || !isFinite(bulkNes) || bulkNes === 0) return 1;
  const agree = (decompNes > 0) === (bulkNes > 0);
  const sig = decompFdr != null && isFinite(decompFdr) && decompFdr < 0.25;
  const strong = decompFdr != null && isFinite(decompFdr) && decompFdr < 0.10;
  if (agree) return strong ? 3 : (sig ? 2 : 1);
  return sig ? -2 : 1;
}

function _decompStepFor(kid, contrastId, cellType) {
  if (!_decompByKey) return 0;
  const d = _decompByKey.get(`${kid}|${contrastId}|${cellType}`);
  if (!d) return 0;
  const cName = CONTRASTS[contrastId];
  const _K = ViewerPayload.kinases();
  const bulkNes = (_K && cName && _K["NES_" + cName]) ? _K["NES_" + cName][kid] : null;
  return _decompStep(d.nes, d.fdr, bulkNes);
}

// Apply the very_high upgrade rule: a "high" attribution row whose decomp
// significantly agrees with the bulk direction is promoted.
function _upgradeTier(attrConf, decompStep) {
  if (attrConf === "high" && decompStep >= 2) return "very_high";
  return attrConf || "none";
}

function _combinedTierFor(kid, contrastId, cellType, attrConf) {
  if (attrConf !== "high") return attrConf || "none";
  return _upgradeTier(attrConf, _decompStepFor(kid, contrastId, cellType));
}

function getScopedAttribution(kinaseId, filter) {
  // Returns filtered rows from PAYLOAD.attribution_index for one kinase.
  // filter: { disease, timepoint, celltype, confidence } where dimension values
  // may be string ("" = any) or array ([] = any).
  const AI = PAYLOAD.attribution_index || {};
  if (!AI.kinase_id) return [];
  const scopedCtx = getScopedContrastIds(filter);
  const ctSet = _filterSet(filter.celltype);
  const confidence = filter.confidence || "";
  const out = [];
  for (let j = 0; j < AI.kinase_id.length; j++) {
    if (AI.kinase_id[j] !== kinaseId) continue;
    if (scopedCtx.size > 0 && !scopedCtx.has(AI.contrast_id[j])) continue;
    if (ctSet.size && !ctSet.has(AI.cell_type[j]))                continue;
    const _attrConf = AI.combined_confidence[j];
    const _tier = _combinedTierFor(kinaseId, AI.contrast_id[j], AI.cell_type[j], _attrConf);
    // Confidence threshold tests the upgraded tier so "very_high" filters work.
    if (!_confPass(_tier, confidence))                            continue;
    out.push({
      contrast_id:               AI.contrast_id[j],
      cell_type:                 AI.cell_type[j],
      combined_confidence:       _attrConf,
      combined_tier:             _tier,
      combined_score:            AI.combined_score[j],
      wmb_specificity:           AI.wmb_specificity            ? AI.wmb_specificity[j]            : null,
      wmb_mean_log2_expression:  AI.wmb_mean_log2_expression   ? AI.wmb_mean_log2_expression[j]   : null,
      wmb_fraction_cells_expressing: AI.wmb_fraction_cells_expressing ? AI.wmb_fraction_cells_expressing[j] : null,
      wmb_binary_expressed:      AI.wmb_binary_expressed       ? AI.wmb_binary_expressed[j]       : false,
      sea_ad_lfc:                AI.sea_ad_lfc                 ? AI.sea_ad_lfc[j]                 : null,
      song_lfc:                  AI.song_lfc                   ? AI.song_lfc[j]                   : null,
      song_pval:                 AI.song_pval                  ? AI.song_pval[j]                  : null,
      song_fdr:                  AI.song_fdr                   ? AI.song_fdr[j]                   : null,
      concordance_source:        AI.concordance_source         ? AI.concordance_source[j]         : "",
      nes:                       AI.nes                        ? AI.nes[j]                        : null,
      fdr:                       AI.fdr                        ? AI.fdr[j]                        : null,
    });
  }
  return out;
}

// Cross-grid AND coverage: kinase passes iff for every cell of the selected
// sub-grid (selected diseases × timepoints × cell types) ≥1 attribution row
// exists at the requested confidence threshold. Empty selection on a dimension
// = wildcard for that axis. Trajectory is per-kinase scalar (OR), checked
// upstream. Search is also handled upstream.
function kinaseQualifies(kinaseId, filter) {
  const rows = getScopedAttribution(kinaseId, filter);
  if (rows.length === 0) return false;
  const dSet = _filterSet(filter.disease);
  const tSet = _filterSet(filter.timepoint);
  const cSet = _filterSet(filter.celltype);
  if (!dSet.size && !tSet.size && !cSet.size) return true;
  const Ds = dSet.size ? Array.from(dSet) : [null];
  const Ts = tSet.size ? Array.from(tSet) : [null];
  const Cs = cSet.size ? Array.from(cSet) : [null];
  // Pre-decode contrast → (disease, timepoint) once.
  const decoded = new Array(rows.length);
  for (let i = 0; i < rows.length; i++) {
    const ctx = CONTRASTS[rows[i].contrast_id] || "";
    const d = ctx.split("_")[0];
    const m = ctx.match(/_(\d+mo)$/);
    decoded[i] = { d, t: m ? m[1] : "", c: rows[i].cell_type };
  }
  for (const d of Ds) {
    for (const t of Ts) {
      for (const c of Cs) {
        let ok = false;
        for (let i = 0; i < decoded.length; i++) {
          const e = decoded[i];
          if (d != null && e.d !== d) continue;
          if (t != null && e.t !== t) continue;
          if (c != null && e.c !== c) continue;
          ok = true; break;
        }
        if (!ok) return false;
      }
    }
  }
  return true;
}

function _buildKinaseRowModel() {
  const K = ViewerPayload.kinases();
  const famMap = META.familyMap || {};
  const idxById = new Map();
  const out = [];
  for (let i = 0; i < K.id.length; i++) {
    idxById.set(K.id[i], i);
    out.push({
      id: K.id[i],
      name: K.name[i],
      gene_symbol: K.gene_symbol[i] || "",
      family: famMap[K.name[i]] || "",
      residue_type: (K.residue_type && K.residue_type[i]) || "ST",
      trajectory: K.trajectory[i] || "",
      peak_contrast: K.peak_contrast[i] || "",
      peak_NES: K.peak_NES[i],
      top_celltype_1: K.top_celltype_1[i] || "",
      _fdr: CONTRASTS.map(c => K["FDR_" + c][i]),
      _nes: CONTRASTS.map(c => K["NES_" + c][i]),
    });
  }
  _kinaseIdxById = idxById;
  return out;
}

function _ensureKinaseIdx() {
  if (_kinaseIdxById !== null) return;
  const K = ViewerPayload.kinases();
  const m = new Map();
  for (let i = 0; i < K.id.length; i++) m.set(K.id[i], i);
  _kinaseIdxById = m;
}

function resetKinaseContextCaches() {
  _keRows = null;
  _kinaseIdxById = null;
  _evidenceByKinase = null;
  _decompByKey = null;
  _decompByKinCtx = null;
  _agreementByKey = null;
  _highlightKinaseIds = null;
  _highlightForBid = null;
  _tv2State = null;
  _tv2DecompCellsCache = null;
  _tv2AttrTierByKinCtx = null;
}

function _ensureKinaseIndexes() {
  if (_keRows === null) _keRows = _buildKinaseRowModel();
  if (_evidenceByKinase === null) {
    const EV = PAYLOAD.kinase_celltype_evidence || {kinase_id:[]};
    const m = new Map();
    for (let k = 0; k < EV.kinase_id.length; k++) {
      const kid = EV.kinase_id[k];
      let arr = m.get(kid);
      if (!arr) { arr = []; m.set(kid, arr); }
      arr.push(k);
    }
    _evidenceByKinase = m;
  }
  if (_decompByKey === null) {
    const D = PAYLOAD.decomposition_index || {kinase_id:[]};
    const m = new Map();
    const m2 = new Map();
    for (let k = 0; k < D.kinase_id.length; k++) {
      const key = `${D.kinase_id[k]}|${D.contrast_id[k]}|${D.cell_type[k]}`;
      m.set(key, {nes: D.decomp_nes[k], fdr: D.decomp_fdr[k]});
      const k2 = `${D.kinase_id[k]}|${D.contrast_id[k]}`;
      let arr = m2.get(k2);
      if (!arr) { arr = []; m2.set(k2, arr); }
      arr.push({cell_type: D.cell_type[k], nes: D.decomp_nes[k], fdr: D.decomp_fdr[k]});
    }
    _decompByKey = m;
    _decompByKinCtx = m2;
  }
  if (_agreementByKey === null) {
    const A = PAYLOAD.agreement_index || {kinase_id:[]};
    const m = new Map();
    for (let k = 0; k < A.kinase_id.length; k++) {
      const key = `${A.kinase_id[k]}|${A.contrast_id[k]}`;
      m.set(key, {
        state: A.state[k],
        bulk_nes: A.bulk_nes[k],
        bulk_fdr: A.bulk_fdr[k],
        top_cell: A.top_cell[k],
        top_cell_nes: A.top_cell_nes[k],
        top_cell_fdr: A.top_cell_fdr[k],
        n_match: A.n_cells_match[k],
        n_oppose: A.n_cells_oppose[k],
      });
    }
    _agreementByKey = m;
  }
}

function _refreshSigCounts(fdr) {
  if (_keSigFdr === fdr) return;
  for (const r of _keRows) {
    let n = 0;
    for (const v of r._fdr) if (v != null && v < fdr) n++;
    r._sigCount = n;
  }
  _keSigFdr = fdr;
}

function _kineMaxAbsNesScoped(r, scopedCtxIds) {
  // Returns max |NES| among contrast indices in scopedCtxIds (all if empty Set).
  let best = null;
  for (let ci = 0; ci < CONTRASTS.length; ci++) {
    if (scopedCtxIds.size > 0 && !scopedCtxIds.has(ci)) continue;
    const v = r._nes[ci];
    if (v == null) continue;
    const a = Math.abs(v);
    if (best == null || a > best) best = a;
  }
  return best;
}

// Max WMB specificity tier across attribution rows for this kinase under the
// active filter scope. Returns 0 when no qualifying rows have wmb_specificity.
function _kineMaxWmbTierScoped(kinaseId, filter) {
  let best = 0;
  for (const e of getScopedAttribution(kinaseId, filter)) {
    const t = _wmbTier(Number(e.wmb_specificity));
    if (t > best) best = t;
  }
  return best;
}

function _kineSigCountScoped(r, fdr, scopedCtxIds) {
  let n = 0;
  for (let ci = 0; ci < CONTRASTS.length; ci++) {
    if (scopedCtxIds.size > 0 && !scopedCtxIds.has(ci)) continue;
    const q = r._fdr[ci];
    if (q != null && q < fdr) n++;
  }
  return n;
}

function _kineTrendDiseaseOrder(filter) {
  const selected = _filterSet(filter.disease);
  if (selected.size) return Array.from(selected);
  const axis = ViewerPayload.contrastAxis ? ViewerPayload.contrastAxis() : {};
  if (axis.groups && axis.groups.length) return axis.groups.slice();
  const out = [];
  const seen = new Set();
  for (const c of CONTRASTS) {
    const d = String(c).split("_")[0];
    if (!seen.has(d)) { seen.add(d); out.push(d); }
  }
  return out;
}

function _kineTrendTimeOrder() {
  const axis = ViewerPayload.contrastAxis ? ViewerPayload.contrastAxis() : {};
  if (axis.timepoints && axis.timepoints.length) return axis.timepoints.slice();
  const out = [];
  const seen = new Set();
  for (const c of CONTRASTS) {
    const m = String(c).match(/_(.+)$/);
    const t = m ? m[1] : "";
    if (t && !seen.has(t)) { seen.add(t); out.push(t); }
  }
  return out;
}

function _kineTrendMatches(r, filter) {
  const pattern = TrendFilter.normalize(filter.pattern || "");
  if (!pattern) return true;
  const diseases = _kineTrendDiseaseOrder(filter);
  const selected = _filterSet(filter.disease);
  const timepoints = _kineTrendTimeOrder();
  let any = false;
  for (const d of diseases) {
    const vals = [];
    for (const t of timepoints) {
      const ci = CONTRASTS.indexOf(`${d}_${t}`);
      if (ci >= 0) vals.push(r._nes[ci]);
    }
    const ok = TrendFilter.vectorMatches(vals, pattern);
    if (selected.size && !ok) return false;
    if (ok) any = true;
  }
  return selected.size ? true : any;
}

// Legacy lens array kept for any remaining references; new code uses
// getScopedContrastIds via KinaseFilter.
const NES_PROFILE_LENSES = [
  {key:"any",  label:"any disease"},
  {key:"App",  label:"App"},
  {key:"Tau",  label:"Tau"},
  {key:"ApTt", label:"ApTt"},
  {key:"nsig", label:"# sig contrasts"},
];

// Legacy helpers — kept for any surviving call sites (e.g. _renderMeaTrajectory
// still uses the ctx.contrast which uses _selectedAuditContrast).
function _kineMaxAbsNesIn(r, diseasePrefix, tpFilter) {
  let best = null;
  for (let ci = 0; ci < CONTRASTS.length; ci++) {
    const c = CONTRASTS[ci];
    if (diseasePrefix && c.indexOf(diseasePrefix) !== 0) continue;
    if (tpFilter && c.indexOf(tpFilter) < 0) continue;
    const v = r._nes[ci];
    if (v == null) continue;
    const a = Math.abs(v);
    if (best == null || a > best) best = a;
  }
  return best;
}

function _kineSigCountIn(r, fdr, diseasePrefix, tpFilter) {
  let n = 0;
  for (let ci = 0; ci < CONTRASTS.length; ci++) {
    const c = CONTRASTS[ci];
    if (diseasePrefix && c.indexOf(diseasePrefix) !== 0) continue;
    if (tpFilter && c.indexOf(tpFilter) < 0) continue;
    const q = r._fdr[ci];
    if (q != null && q < fdr) n++;
  }
  return n;
}

// _keCompare is called with a pre-computed scopedCtxIds set injected via closure.
// We wrap it in a factory called from renderKinaseExplorer.
function _makeKeCompare(scopedCtxIds) {
  const kf = KinaseFilter.get();
  const col = kf.sortCol || "nes_profile";
  const asc = !!kf.sortAsc;
  const fdr = kf.fdr || Store.state.filters.fdr || 0.25;
  return function(a, b) {
    let va, vb;
    if (col === "nes_profile") {
      va = _kineMaxAbsNesScoped(a, scopedCtxIds);
      vb = _kineMaxAbsNesScoped(b, scopedCtxIds);
      if (va == null) va = -Infinity;
      if (vb == null) vb = -Infinity;
    }
    else if (col === "n_attributed_celltypes") {
      // Match what the Cell types pill column displays: dedup by cell_type
      // keeping best tier, then count rows at moderate-or-better.
      const _bestTierByCT = (kid) => {
        const m = new Map();
        for (const e of getScopedAttribution(kid, kf)) {
          const r = _CONF_RANK[e.combined_tier] || 0;
          if (r > (m.get(e.cell_type) || 0)) m.set(e.cell_type, r);
        }
        let n = 0;
        for (const r of m.values()) if (r >= 2) n++;
        return n;
      };
      va = _bestTierByCT(a.id);
      vb = _bestTierByCT(b.id);
    }
    else if (col === "conf") {
      // Sort by max tier reached in scope: very_high(4) > high(3) > moderate(2) > low(1) > none(0).
      const _maxTier = (kid) => {
        let m = 0;
        for (const e of getScopedAttribution(kid, kf)) {
          const r = _CONF_RANK[e.combined_tier] || 0;
          if (r > m) m = r;
        }
        return m;
      };
      va = _maxTier(a.id);
      vb = _maxTier(b.id);
    }
    else if (col === "n_sig") {
      va = _kineSigCountScoped(a, fdr, scopedCtxIds);
      vb = _kineSigCountScoped(b, fdr, scopedCtxIds);
    }
    else if (col === "wmb_max_tier") {
      va = _kineMaxWmbTierScoped(a.id, kf);
      vb = _kineMaxWmbTierScoped(b.id, kf);
    }
    else if (col === "agreement_profile") {
      va = _kineDisagreeCountScoped(a, scopedCtxIds);
      vb = _kineDisagreeCountScoped(b, scopedCtxIds);
    }
    else if (col === "peak_NES") {
      // Scope-aware to match the column's displayed value.
      va = _kineMaxAbsNesScoped(a, scopedCtxIds);
      vb = _kineMaxAbsNesScoped(b, scopedCtxIds);
      if (va == null) va = -Infinity;
      if (vb == null) vb = -Infinity;
    }
    else { va = a[col]; vb = b[col]; }
    if (va == null && vb == null) return 0;
    if (va == null) return 1;
    if (vb == null) return -1;
    if (typeof va === "string") return asc
      ? va.localeCompare(vb) : vb.localeCompare(va);
    return asc ? (va - vb) : (vb - va);
  };
}

// Render the NES profile mini-heatmap (3 diseases × 3 timepoints) for one row.
// Always shows all 9 cells — this glyph IS the cross-contrast comparison.
function _renderNesProfile(r, fdrThresh, maxAbs) {
  const axis = ViewerPayload.contrastAxis();
  const DG = axis.groups.length ? axis.groups : ["App","Tau","ApTt"];
  const TPS = axis.timepoints.length ? axis.timepoints : ["2mo","4mo","6mo"];
  const cells = [];
  for (const d of DG) {
    for (const t of TPS) {
      const c = `${d}_${t}`;
      const ci = CONTRASTS.indexOf(c);
      const nes = ci >= 0 ? r._nes[ci] : null;
      const fdrV = ci >= 0 ? r._fdr[ci] : null;
      const sig = fdrV != null && fdrV < fdrThresh;
      let bg = "#fff";
      if (nes != null && isFinite(nes) && maxAbs > 0) {
        const a = Math.min(1, Math.abs(nes) / maxAbs);
        const rgb = nes >= 0 ? [197,48,48] : [43,108,176];
        bg = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${(0.15 + 0.85 * a).toFixed(3)})`;
      }
      const tip = nes == null ? `${c}: n/a`
        : `${c}: NES ${nes.toFixed(2)}${fdrV != null ? `, FDR ${fdrV.toExponential(1)}` : ""}${sig ? " (sig)" : ""}`;
      cells.push(`<div class="npc${sig ? " sig" : ""}" style="background:${bg};" title="${_escapeHtml(tip)}"></div>`);
    }
  }
  // Layout: rows = diseases (App/Tau/ApTt), cols = timepoints.
  const rowLabels = DG.map(d => `<span>${_escapeHtml(d)}</span>`).join("");
  return `<div class="nes-profile-wrap">` +
    `<div class="nes-profile-row-labels">${rowLabels}</div>` +
    `<div class="nes-profile-cell">${cells.join("")}</div>` +
    `</div>`;
}

function _agreementStateFor(kid, ci) {
  if (!_agreementByKey) return null;
  return _agreementByKey.get(`${kid}|${ci}`) || null;
}

function _kineDisagreeCountScoped(r, scopedCtxIds) {
  let n = 0;
  for (let ci = 0; ci < CONTRASTS.length; ci++) {
    if (scopedCtxIds && scopedCtxIds.size > 0 && !scopedCtxIds.has(ci)) continue;
    const a = _agreementStateFor(r.id, ci);
    if (a && a.state >= 2) n++;
  }
  return n;
}

function _renderAgreementProfile(r) {
  const axis = ViewerPayload.contrastAxis();
  const DG = axis.groups.length ? axis.groups : ["App","Tau","ApTt"];
  const TPS = axis.timepoints.length ? axis.timepoints : ["2mo","4mo","6mo"];
  const cells = [];
  for (const d of DG) {
    for (const t of TPS) {
      const c = `${d}_${t}`;
      const ci = CONTRASTS.indexOf(c);
      const a = ci >= 0 ? _agreementStateFor(r.id, ci) : null;
      let cls = "";
      let tip;
      if (!a) {
        tip = `${c}: neither pipeline significant`;
      } else {
        const stateName = _AGREEMENT_STATE_NAMES[a.state] || "?";
        if (a.state === 1) {
          cls = " agree";
          tip = `${c}: agree — bulk and decomp both significant, same direction`;
        } else {
          cls = " disagree";
          let detail;
          if (stateName === "decomp_only") detail = "bulk null, ≥1 decomp class significant";
          else if (stateName === "bulk_only") detail = "bulk significant, no decomp class significant";
          else if (stateName === "mixed") detail = "bulk significant, decomp classes split (some match, some oppose)";
          else if (stateName === "disagree") detail = "bulk significant, all sig decomp classes oppose bulk sign";
          else detail = stateName;
          tip = `${c}: ${stateName} — ${detail}`;
          if (a.top_cell) tip += ` · top decomp ${a.top_cell} NES ${Number(a.top_cell_nes).toFixed(2)}`;
          if (a.bulk_nes != null && isFinite(a.bulk_nes)) tip += ` · bulk NES ${Number(a.bulk_nes).toFixed(2)}`;
        }
      }
      cells.push(`<div class="apc${cls}" title="${_escapeHtml(tip)}"></div>`);
    }
  }
  const rowLabels = DG.map(d => `<span>${_escapeHtml(d)}</span>`).join("");
  return `<div class="agreement-profile-wrap">` +
    `<div class="nes-profile-row-labels">${rowLabels}</div>` +
    `<div class="agreement-profile-cell">${cells.join("")}</div>` +
    `</div>`;
}

function _renderCellTypesCell(r, filter) {
  // Reflect rows in the active filter scope; if filter is empty, scope = all 9 contrasts.
  const rows = getScopedAttribution(r.id, filter || {});
  const byCell = new Map();
  for (const e of rows) {
    const prev = byCell.get(e.cell_type);
    if (!prev || e.combined_score > prev.combined_score) byCell.set(e.cell_type, e);
  }
  const displayRows = Array.from(byCell.values()).filter(e =>
    e.combined_tier === "very_high" || e.combined_tier === "high" || e.combined_tier === "moderate");
  // Sort: tier first (very_high → high → moderate), then score desc within tier.
  displayRows.sort((a, b) => {
    const dt = (_CONF_RANK[b.combined_tier] || 0) - (_CONF_RANK[a.combined_tier] || 0);
    if (dt !== 0) return dt;
    return b.combined_score - a.combined_score;
  });
  const n = displayRows.length;
  if (n === 0) return `<span class="muted">—</span>`;
  const top = displayRows.slice(0, 3);
  const tip = displayRows.map(e => `${e.cell_type} (${(e.combined_tier || '').replace('_', ' ')}, ${e.combined_score.toFixed(2)})`).join("\n");
  const topNames = top.map(e => {
    const cls = e.combined_tier === "very_high" ? "vhi"
              : e.combined_tier === "high"      ? "hi"
              : "mid";
    return `<span class="badge ${cls}">${_escapeHtml(e.cell_type)}</span>`;
  }).join(" ");
  return `<span title="${_escapeHtml(tip)}"><strong>${n}</strong> ${topNames}${displayRows.length > 3 ? ` <span class="muted">+${displayRows.length - 3}</span>` : ""}</span>`;
}

function _renderKinaseWhitelistBanner(wl) {
  const wrap = document.querySelector(".ke-table-wrap");
  if (!wrap) return;
  let banner = document.getElementById("ke-whitelist-banner");
  if (!wl) {
    if (banner) banner.remove();
    return;
  }
  if (!banner) {
    banner = document.createElement("div");
    banner.id = "ke-whitelist-banner";
    banner.style.cssText = "background:#fff3cd; border:1px solid #f0ad4e; "
      + "color:#8a6d3b; padding:6px 10px; font-size:11px; border-radius:3px; "
      + "margin-bottom:6px; display:flex; align-items:center; gap:10px; "
      + "flex-wrap:wrap;";
    wrap.parentNode.insertBefore(banner, wrap);
  }
  const n = wl.ids.size;
  const lbl = wl.label || "external whitelist";
  const stackHint = wl.stack
    ? "Dropdowns AND with this set — turning them off broadens the result."
    : "Dropdowns are pre-filled with the click context but inactive. Toggle stack to apply them.";
  banner.innerHTML =
    `<span><b>Filtered to ${n} kinases</b> from ${_escapeHtml(lbl)}.</span>`
    + `<label style="display:flex; gap:4px; align-items:center;">`
    +   `<input type="checkbox" id="ke-whitelist-stack"${wl.stack ? " checked" : ""}> stack with filters`
    + `</label>`
    + `<span class="muted" style="flex:1; min-width:240px;">${stackHint}</span>`
    + `<button id="ke-whitelist-clear" class="chip">Clear filter</button>`;
  const stackCb = document.getElementById("ke-whitelist-stack");
  if (stackCb) stackCb.onchange = () => {
    KinaseFilter.setWhitelistStack(stackCb.checked);
    renderKinaseExplorer();
  };
  const btn = document.getElementById("ke-whitelist-clear");
  if (btn) btn.onclick = () => {
    KinaseFilter.clearWhitelist();
    renderKinaseExplorer();
  };
}

function renderKinaseExplorer() {
  const tbody = document.querySelector("#ke-table tbody");
  if (!tbody) return;
  _ensureKinaseIndexes();
  const kf = KinaseFilter.get();
  const fdr = kf.fdr || Store.state.filters.fdr || 0.25;
  const selKid = Store.state.selection.kinase;
  const q = (kf.search || "").trim().toLowerCase();
  const wl = KinaseFilter.getWhitelist();
  _renderKinaseWhitelistBanner(wl);

  _refreshSigCounts(fdr);

  // Scoped contrast IDs from the list filter (disease + timepoint) — used for
  // row inclusion (require ≥1 sig contrast in scope) and sort keys, NOT for
  // visualization scoping inside a row.
  const scopedCtxIds = getScopedContrastIds(kf);

  // Whether any attribution-grid filter is active (drives full qualification).
  const dSet = _filterSet(kf.disease);
  const tSet = _filterSet(kf.timepoint);
  const cSet = _filterSet(kf.celltype);
  const gridActive = dSet.size > 0 || tSet.size > 0 || cSet.size > 0 || !!kf.confidence;
  const nSigMin = Math.max(0, parseInt(kf.nSigMin, 10) || 0);
  const wmbMin = Math.max(0, parseInt(kf.wmbMin, 10) || 0);
  const wmbMinScore = wmbMin > 0 ? wmbMin * _WMB_UNIFORM : 0;

  // Whitelist mode (cross-tab handoff) has two sub-modes:
  //   stack=false (default): whitelist bypasses every other gate. Decomp-only
  //     kinases that would normally fail the attribution grid still appear.
  //   stack=true: whitelist ANDs with the normal filter chain. Useful for
  //     narrowing within a click-through set, but the attribution grid will
  //     drop kinases that lack attribution rows (interpretable empties).
  const visible = [];
  for (const r of _keRows) {
    if (wl) {
      if (!wl.ids.has(r.id)) continue;
      if (!wl.stack) { visible.push(r); continue; }
      // Stack mode: fall through to the normal predicate chain below.
    }
    // Text search
    if (q && !(r.name.toLowerCase().includes(q) ||
               r.gene_symbol.toLowerCase().includes(q))) continue;
    const scopedSig = _kineSigCountScoped(r, fdr, scopedCtxIds);
    if (!q) {
      // n_sig minimum (numeric filter).
      if (scopedSig < nSigMin) continue;
      // Disease/timepoint scope: require ≥1 sig contrast in scope.
      if (scopedCtxIds.size > 0 && scopedSig === 0) continue;
    }
    // Attribution grid: cross-product AND coverage on disease × timepoint × celltype,
    // with confidence as ordinal threshold (≥). Skipped when text search is
    // active so a targeted lookup (e.g. "EGFR") still surfaces the kinase even
    // if persisted localStorage filters would otherwise disqualify it.
    if (!q && gridActive && !kinaseQualifies(r.id, kf)) continue;
    // WMB tier minimum: kinase passes if any attribution row in scope has
    // wmb_specificity ≥ threshold. Independent of grid filters — uses the same
    // disease/timepoint/celltype scope getScopedAttribution honors.
    if (!q && wmbMin > 0) {
      const _rows = getScopedAttribution(r.id, kf);
      let _ok = false;
      for (const e of _rows) {
        const s = Number(e.wmb_specificity);
        if (isFinite(s) && s >= wmbMinScore) { _ok = true; break; }
      }
      if (!_ok) continue;
    }
    if (!q && kf.pattern && !_kineTrendMatches(r, kf)) continue;
    visible.push(r);
  }

  // maxAbsNes computed across all 9 contrasts on visible kinases — color is
  // a global comparison, not scope-restricted.
  let maxAbsNes = 0;
  for (const r of visible) {
    for (let ci = 0; ci < CONTRASTS.length; ci++) {
      const v = r._nes[ci];
      if (v != null && isFinite(v)) {
        const a = Math.abs(v);
        if (a > maxAbsNes) maxAbsNes = a;
      }
    }
  }
  if (maxAbsNes <= 0) maxAbsNes = 1;

  visible.sort(_makeKeCompare(scopedCtxIds));

  // Header arrows: show sort col + direction.
  document.querySelectorAll("#ke-table thead th").forEach(th => {
    const c = th.dataset.col;
    const sortCol = kf.sortCol || "nes_profile";
    const sortAsc = !!kf.sortAsc;
    th.textContent = th.textContent.replace(/[ ▲▼]+$/, "");
    th.textContent = th.textContent.replace(/\s*\(.*\)\s*$/, "");
    if (c === sortCol) th.textContent += sortAsc ? " ▲" : " ▼";
  });

  // Filter scope passed to per-row column renderers. If no grid filter is
  // active, fall back to all-9-contrasts scope for those columns.
  const colFilter = gridActive ? kf
    : {disease:[], timepoint:[], celltype:[], confidence:""};
  const shortContrast = c => c.replace(/_(\d+)mo$/, "·$1").replace(/^ApTt/, "AT");

  const parts = [];
  const drvSet = _highlightKinaseIds;
  const sigDenom = scopedCtxIds.size > 0 ? scopedCtxIds.size : CONTRASTS.length;
  for (const r of visible) {
    const selCls = r.id === selKid ? " selected" : "";
    // sub-thresh: 0 sig contrasts in the scoped set.
    const scopedSig = _kineSigCountScoped(r, fdr, scopedCtxIds);
    const peakAbsNes = _kineMaxAbsNesScoped(r, scopedCtxIds);
    const subCls = scopedSig === 0 ? " sub-thresh" : "";
    const drvCls = (drvSet && drvSet.has(r.id)) ? " driver" : "";

    // Conf pill: highest tier present in scope, with contributing contrasts as chips.
    const scopedRows = getScopedAttribution(r.id, colFilter);
    const ctxByTier = {very_high: new Set(), high: new Set(), moderate: new Set()};
    for (const e of scopedRows) {
      if (ctxByTier[e.combined_tier]) ctxByTier[e.combined_tier].add(e.contrast_id);
    }
    const tierSpec = [
      {tier:"very_high", cls:"vhi", label:"VERY HIGH", suffix:" (attribution + decomp agreement)"},
      {tier:"high",      cls:"hi",  label:"HIGH",      suffix:""},
      {tier:"moderate",  cls:"mid", label:"MOD",       suffix:""},
    ];
    let confBadge;
    const hit = tierSpec.find(s => ctxByTier[s.tier].size > 0);
    if (hit) {
      const ctxs = Array.from(ctxByTier[hit.tier]).map(ci => CONTRASTS[ci]);
      const shown = ctxs.slice(0, 3).map(c => `<span class="ctx-chip ${hit.cls}">${shortContrast(c)}</span>`).join("");
      const overflow = ctxs.length > 3 ? `<span class="ctx-overflow">+${ctxs.length - 3}</span>` : "";
      const tip = `${hit.label} in ${ctxs.length} contrast${ctxs.length===1?"":"s"}${hit.suffix}: ${ctxs.join(", ")}`;
      confBadge = `<span class="badge ${hit.cls}" title="${_escapeHtml(tip)}">${hit.label}</span>${shown}${overflow}`;
    } else {
      const tipScope = gridActive ? "in active filter scope" : "across all 9 contrasts";
      confBadge = `<span class="badge lo" title="No HIGH or MODERATE attribution ${tipScope}.">low</span>`;
    }

    const residueBadge = r.residue_type === "Y"
      ? ' <span class="track-badge track-y" title="Tyrosine kinase (pY track)">pY</span>'
      : "";
    const profile = _renderNesProfile(r, fdr, maxAbsNes);
    const agreementProfile = _renderAgreementProfile(r);
    parts.push(
      `<tr class="ke-row${selCls}${subCls}${drvCls}" data-kid="${r.id}" ` +
      `tabindex="0" aria-label="Kinase ${r.name}; ${scopedSig} sig contrasts in scope">` +
      `<td>${r.name}${residueBadge}</td>` +
      `<td>${r.gene_symbol}</td>` +
      `<td>${_escapeHtml(r.family || "")}</td>` +
      `<td>${profile}</td>` +
      `<td>${agreementProfile}</td>` +
      `<td class="attr-num">${peakAbsNes != null ? peakAbsNes.toFixed(2) : '<span class="muted">—</span>'}</td>` +
      `<td class="attr-num">${scopedSig}<span class="muted" style="font-size:10px;"> / ${sigDenom}</span></td>` +
      `<td>${_renderCellTypesCell(r, colFilter)}</td>` +
      `<td>${_wmbTierBadge(_kineMaxWmbTierScoped(r.id, colFilter))}</td>` +
      `<td>${confBadge}</td>` +
      `</tr>`
    );
  }
  tbody.innerHTML = parts.join("");
  const countEl = document.getElementById("ke-count");
  if (countEl) countEl.textContent = `${visible.length} / ${_keRows.length} kinases`;
}

function _updateRowSelection(tableSel, rowCls, dataAttr, value) {
  const tbody = document.querySelector(`${tableSel} tbody`);
  if (!tbody) return;
  const prev = tbody.querySelector(`tr.${rowCls}.selected`);
  if (prev) prev.classList.remove("selected");
  if (value == null) return;
  const row = tbody.querySelector(`tr.${rowCls}[${dataAttr}="${value}"]`);
  if (row) row.classList.add("selected");
}

function _updateKinaseRowSelection(kid) {
  _updateRowSelection("#ke-table", "ke-row", "data-kid", kid);
}

function _diseaseColorFor(contrast) {
  for (const d of ["App","Tau","ApTt"])
    if (contrast.indexOf(d) === 0) return DISEASE_COLORS[d];
  return "#90a4ae";
}

let _kinaseAuditSeq = 0;
let _sourceCatalogKey = "mea_stoichiometry";

function _normMotif(s) {
  return String(s || "").replace(/^_+|_+$/g, "").toUpperCase();
}
