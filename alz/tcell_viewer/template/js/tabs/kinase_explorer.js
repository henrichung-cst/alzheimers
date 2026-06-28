function _filterSet(v) {
  const s = new Set();
  if (v == null || v === "") return s;
  if (Array.isArray(v)) { for (const x of v) if (x !== "" && x != null) s.add(x); return s; }
  s.add(v);
  return s;
}

function getScopedContrastIds(filter) {
  // T-cell kinase MEA scopes directly by day contrast (d13, d15, ...).
  // Legacy `timepoint` survives for cross-tab compatibility; if it contains
  // direct contrast names, treat it as a day selection too.
  const days = _filterSet(filter.day);
  if (!days.size) {
    for (const v of _filterSet(filter.timepoint)) {
      if (CONTRASTS.indexOf(v) >= 0) days.add(v);
    }
  }
  const ids = new Set();
  for (let ci = 0; ci < CONTRASTS.length; ci++) {
    const c = CONTRASTS[ci];
    if (days.size && !days.has(c)) continue;
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

// Within-cohort T-cell state enrichment:
// detection evidence (tcell_fraction_expressing >= 0.10) is shown separately.
// STATE ENRICHMENT = each state's linear expression as a fold over the
// kinase's BASELINE mean across all adequately-sampled ProjecTILs states (gene-agnostic;
// precomputed in Python), so a kinase concentrated in one state scores a high fold. The
// concentration-of-total tier is retired here — it saturates at ≥2× because the
// ~14 ProjecTILs states are transcriptionally homogeneous, so it cannot localize
// a kinase along the activation continuum. Enrichment uses the full state set and
// discriminates. The location-confidence pill still uses the per-gene effective # of states.
function _tcellEnrichBadge(fold) {
  const v = Number(fold);
  if (fold == null || !isFinite(v)) return '<span class="muted">—</span>';
  const lbl = v.toFixed(1) + "×";
  const cls = v >= 3 ? "hi" : (v >= 2 ? "mid" : (v >= 1.5 ? "lo" : ""));
  if (!cls) return `<span class="muted" title="Within ~1.5× of the kinase's baseline (mean) T-cell state — broadly expressed, not state-enriched">${lbl}</span>`;
  return `<span class="badge ${cls}" title="${lbl} the kinase's baseline (mean-state) expression — enriched in this state vs a typical T-cell state">${lbl}</span>`;
}

function getScopedAttribution(kinaseId, filter) {
  // Returns filtered within-cohort attribution rows from
  // PAYLOAD.attribution_index for one kinase. The T-cell attribution carries the
  // expression evidence (tcell_detected) + two within-cohort axes — cell-TYPE
  // confidence (tcell_top_celltype = CD8/CD4/Treg, which drives confidence_tier)
  // and the activation-state axis (tcell_state_enrichment, gated on detection) —
  // plus pseudobulk concordance vs bulk NES + confidence_tier/confidence_basis.
  // specificity_celltype (the shared key the unified engine reads) = _pill_anchor;
  // it anchors the cell-type pill on the kinase's headline state row.
  // filter dimensions may be string ("" = any) or array ([] = any);
  // celltype scopes the cell_type axis.
  const AI = PAYLOAD.attribution_index || {};
  if (!AI.kinase_id) return [];
  const scopedCtx = getScopedContrastIds(filter);
  const ctSet = _filterSet(filter.celltype);
  const out = [];
  for (let j = 0; j < AI.kinase_id.length; j++) {
    if (AI.kinase_id[j] !== kinaseId) continue;
    if (scopedCtx.size > 0 && !scopedCtx.has(AI.contrast_id[j])) continue;
    if (ctSet.size && !ctSet.has(AI.cell_type[j]))                continue;
    const topCt = AI.tcell_top_celltype ? AI.tcell_top_celltype[j] : "";
    const pillAnchor = AI._pill_anchor ? AI._pill_anchor[j] : "";
    out.push({
      contrast_id:        AI.contrast_id[j],
      cell_type:          AI.cell_type[j],
      tcell_detected:           AI.tcell_detected           ? AI.tcell_detected[j]           : false,
      tcell_fraction_expressing: AI.tcell_fraction_expressing ? AI.tcell_fraction_expressing[j] : null,
      tcell_effective_n:        AI.tcell_effective_n        ? AI.tcell_effective_n[j]        : null,
      tcell_top_celltype:       topCt,
      tcell_state_enrichment:   AI.tcell_state_enrichment   ? AI.tcell_state_enrichment[j]   : null,
      tcell_lfc:          AI.tcell_lfc          ? AI.tcell_lfc[j]          : null,
      tcell_concordance:  AI.tcell_concordance  ? AI.tcell_concordance[j]  : null,
      tcell_concordant:   AI.tcell_concordant   ? AI.tcell_concordant[j]   : null,
      tcell_consistency:  AI.tcell_consistency  ? AI.tcell_consistency[j]  : 0,
      nes:                AI.nes                ? AI.nes[j]                : null,
      fdr:                AI.fdr                ? AI.fdr[j]                : null,
      nsclc_frac:         AI.nsclc_frac         ? AI.nsclc_frac[j]         : null,
      nsclc_detected:     AI.nsclc_detected     ? AI.nsclc_detected[j]     : null,
      confidence_tier:    AI.confidence_tier    ? AI.confidence_tier[j]    : "none",
      confidence_basis:   AI.confidence_basis   ? AI.confidence_basis[j]   : "",
      // The cell-type confidence pill (a kinase-level property) is rendered once,
      // on the kinase's headline STATE row — _attrVerdictConfCell shows it where
      // cell_type === specificity_celltype.
      specificity_celltype: pillAnchor,
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
      tcell_celltype: (K.tcell_celltype && K.tcell_celltype[i]) || "",
      tcell_celltype_tier: (K.tcell_celltype_tier && K.tcell_celltype_tier[i] != null)
        ? K.tcell_celltype_tier[i] : 0,
      nsclc_specificity_count: (K.nsclc_specificity_count && K.nsclc_specificity_count[i] != null)
        ? K.nsclc_specificity_count[i] : null,
      nsclc_groups_total: (K.nsclc_groups_total && K.nsclc_groups_total[i] != null)
        ? K.nsclc_groups_total[i] : null,
      nsclc_expressing_groups: (K.nsclc_expressing_groups && K.nsclc_expressing_groups[i]) || null,
      nsclc_top_group: (K.nsclc_top_group && K.nsclc_top_group[i]) || "",
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

let _keVisible = [];

function resetKinaseContextCaches() {
  _keRows = null;
  _keVisible = [];
  _kinaseIdxById = null;
  _decompByKey = null;
  _decompByKinCtx = null;
  _agreementByKey = null;
  _highlightKinaseIds = null;
  _highlightForBid = null;
}

function _ensureKinaseIndexes() {
  if (_keRows === null) _keRows = _buildKinaseRowModel();
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

function _kineSignedPeakNesScoped(r, scopedCtxIds) {
  // Returns the signed NES at the max-|NES| contrast in scopedCtxIds (all if empty Set).
  // Selects the peak by magnitude; ranks by sign. Returns null when no values.
  let bestAbs = null, bestSigned = null;
  for (let ci = 0; ci < CONTRASTS.length; ci++) {
    if (scopedCtxIds.size > 0 && !scopedCtxIds.has(ci)) continue;
    const v = r._nes[ci];
    if (v == null) continue;
    const a = Math.abs(v);
    if (bestAbs == null || a > bestAbs) { bestAbs = a; bestSigned = v; }
  }
  return bestSigned;
}

// Peak within-cohort state enrichment across attribution rows for this kinase
  // under the active filter scope (fold over the baseline mean state). Returns 0
// when no qualifying rows. This is the T-cell state-enrichment signal.
function _kineMaxTcellEnrichScoped(kinaseId, filter) {
  let best = 0;
  for (const e of getScopedAttribution(kinaseId, filter)) {
    const v = e.tcell_state_enrichment;
    if (v != null && isFinite(v) && v > best) best = v;
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

function _kineSignPassScoped(r, sign, scopedCtxIds) {
  if (!sign) return true;
  for (let ci = 0; ci < CONTRASTS.length; ci++) {
    if (scopedCtxIds.size > 0 && !scopedCtxIds.has(ci)) continue;
    const v = r._nes[ci];
    if (v == null || !isFinite(v)) continue;
    if (sign === "up" && v > 0) return true;
    if (sign === "down" && v < 0) return true;
  }
  return false;
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
      return numCmp(_kineSignedPeakNesScoped(a, scopedCtxIds),
                    _kineSignedPeakNesScoped(b, scopedCtxIds),
                    asc ? -1 : 1);
    }
    else if (col === "n_attributed_celltypes") {
      // Match the Cell states column: count distinct T-cell states the kinase is
      // DETECTED in (≥10% of cells) in scope.
      const _nDetectedStates = (kid) => {
        const s = new Set();
        for (const e of getScopedAttribution(kid, kf)) {
          if (e.tcell_detected === true || e.tcell_detected === "True" || e.tcell_detected === "true")
            s.add(e.cell_type);
        }
        return s.size;
      };
      va = _nDetectedStates(a.id);
      vb = _nDetectedStates(b.id);
    }
    else if (col === "n_sig") {
      va = _kineSigCountScoped(a, fdr, scopedCtxIds);
      vb = _kineSigCountScoped(b, fdr, scopedCtxIds);
    }
    else if (col === "tcell_max_enrich") {
      va = _kineMaxTcellEnrichScoped(a.id, kf);
      vb = _kineMaxTcellEnrichScoped(b.id, kf);
    }
    else if (col === "tcell_celltype") {
      // Sort by cell-type concentration tier (concentrated kinases on top).
      va = a.tcell_celltype_tier || 0;
      vb = b.tcell_celltype_tier || 0;
    }
    else if (col === "agreement_profile") {
      va = _kineDisagreeCountScoped(a, scopedCtxIds);
      vb = _kineDisagreeCountScoped(b, scopedCtxIds);
    }
    else if (col === "peak_NES") {
      // Scope-aware signed sort — select peak by magnitude, rank by sign.
      return numCmp(_kineSignedPeakNesScoped(a, scopedCtxIds),
                    _kineSignedPeakNesScoped(b, scopedCtxIds),
                    asc ? -1 : 1);
    }
    else { va = a[col]; vb = b[col]; }
    if (typeof va === "string" || typeof vb === "string") {
      const sa = va == null ? "" : String(va), sb = vb == null ? "" : String(vb);
      return asc ? sa.localeCompare(sb) : sb.localeCompare(sa);
    }
    return numCmp(va, vb, asc ? -1 : 1);
  };
}

// Render the NES profile mini-strip for one row.
function _renderNesProfile(r, fdrThresh, maxAbs) {
  // T-cell: one cell per day-vs-baseline contrast, in payload order.
  const cells = [];
  const colLabels = [];
  for (let ci = 0; ci < CONTRASTS.length; ci++) {
    const c = CONTRASTS[ci];
    const nes = r._nes[ci];
    const fdrV = r._fdr[ci];
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
    const dayLabel = c.replace(/_d2$/, "");
    colLabels.push(`<span title="${_escapeHtml(c)}">${_escapeHtml(dayLabel)}</span>`);
  }
  const countStyle = `--nes-profile-count:${CONTRASTS.length};`;
  return `<div class="nes-profile-wrap tcell-nes-profile" style="${countStyle}">` +
    `<div class="nes-profile-col-labels">${colLabels.join("")}</div>` +
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
  // T-cell states this kinase is DETECTED in (fraction expressing ≥10%) in the
  // active filter scope — breadth of detection within the cohort, parallel to the
  // NSCLC spec column. Enrichment lives in the State
  // enrichment column; this counts presence, so it stays coherent with the
  // per-state detection shown in the Attribution verdict tab.
  const rows = getScopedAttribution(r.id, filter || {});
  const detectedCells = new Set();
  for (const e of rows) {
    if (e.tcell_detected === true || e.tcell_detected === "True" || e.tcell_detected === "true")
      detectedCells.add(e.cell_type);
  }
  const n = detectedCells.size;
  if (n === 0) return `<span class="muted" title="Not detected (≥10% of cells) in any T-cell state in scope">0</span>`;
  const names = Array.from(detectedCells).sort();
  const tip = `Detected (≥10% of cells) in ${n} T-cell state(s): ${names.join(", ")}`;
  return `<span title="${_escapeHtml(tip)}"><strong>${n}</strong><span class="muted" style="font-size:10px;"> / 14</span></span>`;
}

// Within-cohort cell-TYPE confidence (CD8 / CD4 / Treg) from the donor's own
// scRNA: the dominant cell type + whether the kinase concentrates there (tier ≥2
// = ≥2× the even 1/3 share) or is broad across the three types. Distinct from the
// NSCLC spec column (beyond-T cell-type breadth). Precomputed in Python
// (tcell_celltype / tcell_celltype_tier slice columns).
function _renderCellTypeCell(r) {
  const ct = r.tcell_celltype || "";
  if (!ct) return '<span class="muted" title="No measurable cell-type expression distribution">—</span>';
  const tier = r.tcell_celltype_tier || 0;
  if (tier >= 2) {
    return `<span class="badge hi" title="Concentrated in ${_escapeHtml(ct)} — ≥2× the even 1/3 cell-type share (concentrated in one cell type)">${_escapeHtml(ct)} ✓</span>`;
  }
  // Broad: the kinase is spread across CD8/CD4/Treg. Naming the bare argmax (a
  // near-tie) misreads as "it's a <type> kinase", so show only "broad".
  return `<span class="muted" title="Broad across CD8 / CD4 / Treg — not concentrated in any single T-cell type (≤2× the even 1/3 share)">broad</span>`;
}

// NSCLC cell-type specificity — the SINGLE independent-reference breadth metric:
// N of the coarse cell-type groups (T_NK + non-T: B_plasma, Myeloid, Epithelial,
// Endothelial, Fibroblast, Mast) whose cell-weighted prevalence ≥ the shared 10%
// detection floor. One definition, read straight from the canonical
// nsclc_kinase_expression.csv columns (specificity_count / expressing_groups /
// top_coarse_group) — no native-max variant. Fewer = more cell-type-restricted.
function _renderNSCLCSpecificityCountCell(r) {
  const cnt = r.nsclc_specificity_count;
  if (cnt == null) {
    return `<span class="muted" title="Outside the NSCLC Flex probe panel — the independent reference cannot speak to this kinase.">n/a</span>`;
  }
  const total = r.nsclc_groups_total != null ? r.nsclc_groups_total : 7;
  const groups = Array.isArray(r.nsclc_expressing_groups) ? r.nsclc_expressing_groups : [];
  const groupStr = groups.length ? groups.join(", ") : "none";
  const top = r.nsclc_top_group || "";
  // cnt 0 = expressed in NO cell type above the 10% bar (least specific, not most):
  // it must not read green. 1 = restricted to a single cell type (most specific).
  const cls = cnt === 0 ? "lo" : (cnt === 1 ? "hi" : (cnt <= 2 ? "mid" : "lo"));
  const tipBase = `Expressed in ≥10% of cells (cell-weighted prevalence) in ${cnt} of ${total} coarse cell types (NSCLC reference, shared 10% floor): ${groupStr}.`;
  if (cnt === 0) {
    // Nothing clears the floor. The "dominant" group here is a ratio of
    // sub-threshold expression (noise over noise) — it must NOT render at all,
    // or it reads as a specificity call the data does not support.
    const tip = `${tipBase} Not robustly detected in any cell type.`;
    return `<span class="badge lo" title="${_escapeHtml(tip)}">${cnt} / ${total}</span>`;
  }
  const tip = `${tipBase}` +
    (top ? ` Most prevalent in ${top}.` : "") +
    " Fewer = more cell-type-restricted (more specific).";
  const badge = top ? ` <span class="badge ${cls}">${_escapeHtml(top)}</span>` : "";
  const memberInline = groups.length
    ? ` <span class="muted" style="font-size:10px;">${_escapeHtml(groupStr)}</span>`
    : "";
  return `<span title="${_escapeHtml(tip)}"><strong>${cnt}</strong><span class="muted" style="font-size:10px;"> / ${total}</span>${badge}${memberInline}</span>`;
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
  const donor = ViewerPayload.activeContext();
  const meaDonor = (PAYLOAD.meta && PAYLOAD.meta.mea_kinase_donor) || "donor1";
  if (donor !== meaDonor) {
    tbody.innerHTML = '<tr><td colspan="8" class="muted" style="padding:24px;text-align:center;">'
      + 'Kinase MEA is ' + _escapeHtml(meaDonor) + '-only — ' + _escapeHtml(donor)
      + ' has no IMAC. Switch to ' + _escapeHtml(meaDonor) + ' to see kinase activity.'
      + '</td></tr>';
    const countEl = document.getElementById("ke-count");
    if (countEl) countEl.textContent = donor + ": no kinase MEA";
    const detail = document.getElementById("ke-detail");
    if (detail) detail.innerHTML = '<div class="muted">No kinase MEA for ' + _escapeHtml(donor) + '.</div>';
    return;
  }
  _ensureKinaseIndexes();
  const kf = KinaseFilter.get();
  const fdr = kf.fdr || Store.state.filters.fdr || 0.25;
  const selKid = Store.state.selection.kinase;
  const q = (kf.search || "").trim().toLowerCase();
  const wl = KinaseFilter.getWhitelist();
  _renderKinaseWhitelistBanner(wl);

  _refreshSigCounts(fdr);

  // Scoped contrast IDs from the day filter — used for row inclusion, n_sig,
  // |NES|, sign filtering, and sort keys.
  const scopedCtxIds = getScopedContrastIds(kf);
  const scopedDenom = scopedCtxIds.size > 0 ? scopedCtxIds.size : CONTRASTS.length;

  // Whether any attribution-grid filter is active. The current T-cell payload
  // does not ship attribution rows, so legacy disease/timepoint/celltype values
  // should not be allowed to zero the kinase table.
  const hasAttribution = !!(PAYLOAD.attribution_index
    && PAYLOAD.attribution_index.kinase_id
    && PAYLOAD.attribution_index.kinase_id.length);
  const dSet = _filterSet(kf.disease);
  const tSet = _filterSet(kf.timepoint);
  const cSet = _filterSet(kf.celltype);
  const gridActive = hasAttribution
    && (dSet.size > 0 || tSet.size > 0 || cSet.size > 0 || !!kf.confidence);
  const nSigMin = Math.min(scopedDenom, Math.max(0, parseInt(kf.nSigMin, 10) || 0));
  // Opt-in state-enrichment narrowing. 0 = Any (off) — the default; hides nothing.
  // Threshold is a state-enrichment fold (e.g. 1.5, 2, 3).
  const tcellMin = Math.max(0, parseFloat(kf.tcellMin) || 0);

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
      // Day scope: require ≥1 significant selected contrast.
      if (scopedCtxIds.size > 0 && scopedSig === 0) continue;
    }
    if (!q && !_kineSignPassScoped(r, kf.sign || "", scopedCtxIds)) continue;
    // Attribution grid: cross-product AND coverage on disease × timepoint × celltype,
    // with confidence as ordinal threshold (≥). Skipped when text search is
    // active so a targeted lookup (e.g. "EGFR") still surfaces the kinase even
    // if persisted localStorage filters would otherwise disqualify it.
    if (!q && gridActive && !kinaseQualifies(r.id, kf)) continue;
    // Opt-in state-enrichment narrowing (tcellMin > 0, off by default): kinase passes
    // if any state in scope reaches the requested enrichment fold (≥ tcellMin ×
    // the kinase's baseline mean state). Enrichment only — concordance is
    // never used to filter (de-gate
    // directive, docs/tcell_exhaustion_analysis_summary.md). Skipped
    // under text search so a targeted lookup always surfaces the kinase.
    if (!q && tcellMin > 0 && _kineMaxTcellEnrichScoped(r.id, kf) < tcellMin) continue;
    // Trajectory-shape pattern. Skipped under text search so a targeted lookup
    // still surfaces the kinase regardless of its NES shape.
    if (!q && kf.pattern && !TrendFilter.vectorMatches(r._nes, kf.pattern)) continue;
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
  const sigDenom = scopedDenom;
  for (const r of visible) {
    const selCls = r.id === selKid ? " selected" : "";
    // sub-thresh: 0 sig contrasts in the scoped set.
    const scopedSig = _kineSigCountScoped(r, fdr, scopedCtxIds);
    const peakAbsNes = _kineMaxAbsNesScoped(r, scopedCtxIds);
    const subCls = scopedSig === 0 ? " sub-thresh" : "";

    // Stamp export-friendly computed values onto the row for csvSerialize.
    r._exportScopedSig = scopedSig;
    r._exportPeakNes = _kineSignedPeakNesScoped(r, scopedCtxIds);
    r._exportTopCelltype = r.tcell_celltype || r.top_celltype_1 || "";
    const drvCls = (drvSet && drvSet.has(r.id)) ? " driver" : "";

    // Within-cohort cell-type badge + state-enrichment badge + cell-states pill +
    // NSCLC specificity (single N/7 metric at the shared ≥10% prevalence floor).
    const cellTypeCell = _renderCellTypeCell(r);
    const specBadge = _tcellEnrichBadge(_kineMaxTcellEnrichScoped(r.id, colFilter));
    const cellStatesCell = _renderCellTypesCell(r, colFilter);
    const nsclcSpecCell = _renderNSCLCSpecificityCountCell(r);

    const residueBadge = r.residue_type === "Y"
      ? ' <span class="track-badge track-y" title="Tyrosine kinase (pY track)">pY</span>'
      : "";
    const profile = _renderNesProfile(r, fdr, maxAbsNes);
    parts.push(
      `<tr class="ke-row${selCls}${subCls}${drvCls}" data-kid="${r.id}" ` +
      `tabindex="0" aria-label="Kinase ${r.name}; ${scopedSig} sig contrasts in scope">` +
      `<td>${r.name}${residueBadge}</td>` +
      `<td>${r.gene_symbol}</td>` +
      `<td>${_escapeHtml(r.family || "")}</td>` +
      `<td>${profile}</td>` +
      `<td class="attr-num">${r.peak_NES != null && isFinite(r.peak_NES) ? (r.peak_NES > 0 ? "+" : "") + r.peak_NES.toFixed(2) : '<span class="muted">—</span>'}</td>` +
      `<td class="attr-num">${scopedSig}<span class="muted" style="font-size:10px;"> / ${sigDenom}</span></td>` +
      `<td>${cellTypeCell}</td>` +
      `<td>${specBadge}</td>` +
      `<td>${cellStatesCell}</td>` +
      `<td>${nsclcSpecCell}</td>` +
      `</tr>`
    );
  }
  _keVisible = visible.slice();
  tbody.innerHTML = parts.join("");
  const countEl = document.getElementById("ke-count");
  if (countEl) countEl.textContent = `${visible.length} / ${_keRows.length} kinases`;
}

function exportKinaseCsv() {
  const donor = (ViewerPayload.activeContext && ViewerPayload.activeContext()) || "donor1";
  const headers = ["Kinase","Gene","Family","Residue","n_sig","peak_NES","tcell_top_celltype"];
  const keys    = ["name","gene_symbol","family","residue_type","_exportScopedSig","_exportPeakNes","_exportTopCelltype"];
  csvDownload(csvSerialize(headers, keys, _keVisible), exportFilename(donor, "kinase"));
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
