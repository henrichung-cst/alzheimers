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

let _attributionRowsByKinase = null;
function _ensureAttributionRowsByKinase() {
  if (_attributionRowsByKinase) return _attributionRowsByKinase;
  const m = new Map();
  const AI = PAYLOAD.attribution_index || {};
  const ids = AI.kinase_id || [];
  for (let i = 0; i < ids.length; i++) {
    const kid = Number(ids[i]);
    if (!m.has(kid)) m.set(kid, []);
    m.get(kid).push(i);
  }
  _attributionRowsByKinase = m;
  return m;
}

// WMB location tiers as multiples of the even-split baseline. WMB specificity
// is a share normalized over the retained WMB classes that carry atlas cells (~9),
// so the honest uniform is 1/N_retained, read canonically from meta.wmb_uniform
// (matches the crosstable; see docs/foundation/concordance.md §6).
const _WMB_UNIFORM = (typeof PAYLOAD !== "undefined" && PAYLOAD && PAYLOAD.meta && PAYLOAD.meta.wmb_uniform) || (1 / 9);
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
  return '<span class="badge ' + cls + '" title="WMB location cross-check: ≥ ' + t +
         '× uniform (' + (t * _WMB_UNIFORM).toFixed(3) + ')">' + _wmbTierLabel(t) + '</span>';
}

// Song detection evidence. Shown as concentration tier (≥2×/5×/10× over
// 1/n_detected) at the kinase's top cell type, plus effective # cell types
// (breadth indicator). Replaces the former share-fold (song_top_share / even-split)
// which was inversely predictive of presence: ~83% of high-tier share pairs were
// undetected in the reference.
function _songConcTier(conc) {
  if (conc == null || !isFinite(conc)) return 0;
  if (conc >= 10) return 10;
  if (conc >= 5) return 5;
  if (conc >= 2) return 2;
  if (conc >= 1) return 1;
  return 0;
}
function _keSongBadge(song) {
  if (!song || song.topConcentration == null) return '<span class="muted">—</span>';
  const tier = _songConcTier(song.topConcentration);
  const effN = song.effectiveN;
  const top = song.topCelltype || "";
  if (!tier && effN == null) return '<span class="muted">—</span>';
  const tierBadge = tier > 0
    ? `<span class="badge ${tier >= 10 ? 'vhi' : tier >= 5 ? 'hi' : tier >= 2 ? 'mid' : 'lo'}" title="Concentration ≥${tier}× over detected-cell-type uniform in ${_escapeHtml(top)}">≥${tier}×</span>`
    : `<span class="muted" title="Top cell type: ${_escapeHtml(top)}">&lt;1×</span>`;
  const nTxt = effN != null && isFinite(effN) ? ` <span class="muted" title="Effective # cell types: ${effN.toFixed(1)} (lower = more specific)">${effN.toFixed(1)}ct</span>` : "";
  return tierBadge + nTxt;
}

function _rowSongLocationRank(r) {
  const conc = r && r.song ? Number(r.song.topConcentration) : null;
  if (conc == null || !isFinite(conc)) return -1;
  return conc;
}

function _songSpecificityRank(e) {
  const conc = e ? Number(e.song_concentration) : null;
  if (conc == null || !isFinite(conc)) return -1;
  return conc;
}

function _cmpCanonicalAttribution(a, b) {
  const ta = _CONF_RANK[a && a.confidence_tier] || 0;
  const tb = _CONF_RANK[b && b.confidence_tier] || 0;
  if (ta !== tb) return tb - ta;
  const sa = _songSpecificityRank(a);
  const sb = _songSpecificityRank(b);
  if (sa !== sb) return sb - sa;
  const da = a && a.decomp_agrees_bulk ? 1 : 0;
  const db = b && b.decomp_agrees_bulk ? 1 : 0;
  return db - da;
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

function getScopedAttribution(kinaseId, filter) {
  // Returns filtered rows from PAYLOAD.attribution_index for one kinase.
  // filter: { disease, timepoint, celltype, confidence } where dimension values
  // may be string ("" = any) or array ([] = any).
  const AI = PAYLOAD.attribution_index || {};
  if (!AI.kinase_id) return [];
  const rowIdxs = _ensureAttributionRowsByKinase().get(Number(kinaseId)) || [];
  if (rowIdxs.length === 0) return [];
  const scopedCtx = getScopedContrastIds(filter);
  const ctSet = _filterSet(filter.celltype);
  const confidence = filter.confidence || "";
  const out = [];
  for (const j of rowIdxs) {
    if (scopedCtx.size > 0 && !scopedCtx.has(AI.contrast_id[j])) continue;
    if (ctSet.size && !ctSet.has(AI.cell_type[j]))                continue;
    const _tier = AI.confidence_tier ? AI.confidence_tier[j] : "none";
    if (!_confPass(_tier, confidence))                            continue;
    out.push({
      contrast_id:               AI.contrast_id[j],
      cell_type:                 AI.cell_type[j],
      confidence_tier:           _tier,
      confidence_basis:          AI.confidence_basis ? AI.confidence_basis[j] : "",
      song_direction_support:    AI.song_direction_support ? AI.song_direction_support[j] : false,
      human_location_tier:       AI.human_location_tier ? AI.human_location_tier[j] : "none",
      decomp_agrees_bulk:        AI.decomp_agrees_bulk ? AI.decomp_agrees_bulk[j] : false,
      song_detected:             AI.song_detected              ? AI.song_detected[j]              : false,
      song_concentration:        AI.song_concentration         ? AI.song_concentration[j]         : null,
      song_concentration_tier:   AI.song_concentration_tier    ? AI.song_concentration_tier[j]    : 0,
      song_fraction_cells_expressing: AI.song_fraction_cells_expressing ? AI.song_fraction_cells_expressing[j] : null,
      song_effective_n:          AI.song_effective_n           ? AI.song_effective_n[j]           : null,
      song_top_celltype:         AI.song_top_celltype          ? AI.song_top_celltype[j]          : "",
      song_top_concentration:    AI.song_top_concentration     ? AI.song_top_concentration[j]     : null,
      wmb_detected:              AI.wmb_detected               ? AI.wmb_detected[j]               : false,
      wmb_concentration:         AI.wmb_concentration          ? AI.wmb_concentration[j]          : null,
      wmb_concentration_tier:    AI.wmb_concentration_tier     ? AI.wmb_concentration_tier[j]     : 0,
      wmb_mean_log2_expression:  AI.wmb_mean_log2_expression   ? AI.wmb_mean_log2_expression[j]   : null,
      wmb_fraction_cells_expressing: AI.wmb_fraction_cells_expressing ? AI.wmb_fraction_cells_expressing[j] : null,
      wmb_binary_expressed:      AI.wmb_binary_expressed       ? AI.wmb_binary_expressed[j]       : false,
      sea_ad_lfc:                AI.sea_ad_lfc                 ? AI.sea_ad_lfc[j]                 : null,
      seaad_location_score:      AI.seaad_location_score       ? AI.seaad_location_score[j]       : null,
      hbca_location_score:       AI.hbca_location_score        ? AI.hbca_location_score[j]        : null,
      human_location_score:      AI.human_location_score       ? AI.human_location_score[j]       : null,
      decomp_nes_python:         AI.decomp_nes                 ? AI.decomp_nes[j]                 : null,
      decomp_fdr_python:         AI.decomp_fdr                 ? AI.decomp_fdr[j]                 : null,
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
  // Per-kinase Song detection evidence (per-gene fields, constant across a kid's
  // attribution rows) — stamped once so the Song column, filter, and sort can read it.
  const songByKid = new Map();
  const A = PAYLOAD.attribution_index;
  if (A && A.kinase_id && A.song_effective_n) {
    for (let i = 0; i < A.kinase_id.length; i++) {
      const kid = A.kinase_id[i];
      if (songByKid.has(kid)) continue;
      const effN = A.song_effective_n[i];
      if (effN != null && isFinite(effN)) {
        songByKid.set(kid, {
          effectiveN: effN,
          topCelltype: (A.song_top_celltype && A.song_top_celltype[i]) || "",
          topConcentration: (A.song_top_concentration && isFinite(A.song_top_concentration[i])) ? A.song_top_concentration[i] : null,
        });
      }
    }
  }
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
      song: songByKid.get(K.id[i]) || null,
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

// Max WMB concentration tier across attribution rows for this kinase under the
// active filter scope. Returns 0 when no qualifying rows have wmb_concentration_tier.
function _kineMaxWmbTierScoped(kinaseId, filter) {
  let best = 0;
  for (const e of getScopedAttribution(kinaseId, filter)) {
    const t = Number(e.wmb_concentration_tier) || 0;
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
      // keeping best tier, then count rows at moderate-or-better. Song location
      // breaks count ties so the own-cohort cell-specific signal is favored.
      const _bestTierByCT = (kid) => {
        const m = new Map();
        for (const e of getScopedAttribution(kid, kf)) {
          const r = _CONF_RANK[e.confidence_tier] || 0;
          if (r > (m.get(e.cell_type) || 0)) m.set(e.cell_type, r);
        }
        let n = 0;
        for (const r of m.values()) if (r >= 2) n++;
        return n;
      };
      const ca = _bestTierByCT(a.id);
      const cb = _bestTierByCT(b.id);
      if (ca !== cb) return asc ? (ca - cb) : (cb - ca);
      va = _rowSongLocationRank(a);
      vb = _rowSongLocationRank(b);
    }
    else if (col === "conf") {
      // Sort by max tier reached in scope, with Song location as the tie-breaker.
      const _maxTier = (kid) => {
        let m = 0;
        for (const e of getScopedAttribution(kid, kf)) {
          const r = _CONF_RANK[e.confidence_tier] || 0;
          if (r > m) m = r;
        }
        return m;
      };
      const ca = _maxTier(a.id);
      const cb = _maxTier(b.id);
      if (ca !== cb) return asc ? (ca - cb) : (cb - ca);
      va = _rowSongLocationRank(a);
      vb = _rowSongLocationRank(b);
    }
    else if (col === "n_sig") {
      va = _kineSigCountScoped(a, fdr, scopedCtxIds);
      vb = _kineSigCountScoped(b, fdr, scopedCtxIds);
    }
    else if (col === "song_spec") {
      // Rank by the displayed Song tier, then the fold driver (peak share), not τ.
      va = _rowSongLocationRank(a);
      vb = _rowSongLocationRank(b);
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
  const rows = getScopedAttribution(r.id, filter || {}).filter(e =>
    e.confidence_tier === "very_high" || e.confidence_tier === "high" || e.confidence_tier === "moderate");
  const byCell = new Map();
  for (const e of rows) {
    const prev = byCell.get(e.cell_type);
    if (!prev || _cmpCanonicalAttribution(e, prev) < 0) byCell.set(e.cell_type, e);
  }
  const displayRows = Array.from(byCell.values());
  displayRows.sort(_cmpCanonicalAttribution);
  const n = displayRows.length;
  if (n === 0) return `<span class="muted">—</span>`;
  const top = displayRows.slice(0, 3);
  const tip = displayRows.map(e => {
    const songDet = e.song_detected ? `✓ ${e.song_concentration_tier > 0 ? '≥' + e.song_concentration_tier + '×' : 'low conc'}` : "✗ not detected";
    return `${e.cell_type} (Song ${songDet}, ${(e.confidence_tier || '').replace('_', ' ')})`;
  }).join("\n");
  const topNames = top.map(e => {
    const cls = e.confidence_tier === "very_high" ? "vhi"
              : e.confidence_tier === "high"      ? "hi"
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
  // Song concentration tier floor: 0 = any, 2/5/10 = ≥2×/≥5×/≥10× over detected-set uniform.
  const songMin = Math.max(0, parseFloat(kf.songMin) || 0);

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
    // Song concentration tier floor. Per-kinase, pivot-independent.
    if (!q && songMin > 0) {
      if (!r.song || r.song.topConcentration == null || _songConcTier(r.song.topConcentration) < songMin) continue;
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
      if (ctxByTier[e.confidence_tier]) ctxByTier[e.confidence_tier].add(e.contrast_id);
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

    // Stamp export-friendly computed values onto the row for csvSerialize.
    r._exportScopedSig = scopedSig;
    r._exportPeakAbsNes = peakAbsNes;
    r._exportSongEffectiveN = r.song ? r.song.effectiveN : null;
    r._exportSongTopCelltype = r.song ? r.song.topCelltype : null;
    r._exportWmbMaxTier = _kineMaxWmbTierScoped(r.id, colFilter);
    r._exportConf = hit ? hit.tier : "low";

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
      `<td class="attr-num">${r.peak_NES != null && isFinite(r.peak_NES) ? (r.peak_NES > 0 ? "+" : "") + r.peak_NES.toFixed(2) : '<span class="muted">—</span>'}</td>` +
      `<td class="attr-num">${scopedSig}<span class="muted" style="font-size:10px;"> / ${sigDenom}</span></td>` +
      `<td>${_renderCellTypesCell(r, colFilter)}</td>` +
      `<td style="text-align:center;">${_keSongBadge(r.song)}</td>` +
      `<td style="text-align:center;">${_wmbTierBadge(r._exportWmbMaxTier)}</td>` +
      `<td>${confBadge}</td>` +
      `</tr>`
    );
  }
  _keVisible = visible.slice();
  tbody.innerHTML = parts.join("");
  const countEl = document.getElementById("ke-count");
  if (countEl) countEl.textContent = `${visible.length} / ${_keRows.length} kinases`;
}

function exportKinaseCsv() {
  const stamp = new Date().toISOString().slice(0, 10);
  const headers = ["Kinase","Gene","Family","Residue","n_sig","peak_NES","song_effectiveN","song_topCelltype","wmb_max_tier","conf"];
  const keys    = ["name","gene_symbol","family","residue_type","_exportScopedSig","_exportPeakAbsNes","_exportSongEffectiveN","_exportSongTopCelltype","_exportWmbMaxTier","_exportConf"];
  csvDownload(csvSerialize(headers, keys, _keVisible), `kinase_${stamp}.csv`);
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
