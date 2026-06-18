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

// Within-cohort specificity tiers expressed as multiples of the uniform
// baseline (1/N_states; donor1 = 1/14 ≈ 0.0714). 10× / 5× / 2× / 1×. The tier
// integer is precomputed in Python (alz/cross_reference/tcell_within_cohort.py)
// because N_states is donor-dependent; here we only need the uniform value for
// the badge tooltip.
function _tcellUniform() {
  return (typeof PAYLOAD !== "undefined" && PAYLOAD && PAYLOAD.meta &&
          PAYLOAD.meta.tcell_attribution_uniform) || (1 / 14);
}
function _tcellTierLabel(t) { return t > 0 ? "≥" + t + "×" : ""; }
function _tcellTierBadge(t) {
  if (!t) return '<span class="muted">—</span>';
  const cls = t >= 10 ? "vhi" : (t >= 5 ? "hi" : (t >= 2 ? "mid" : "lo"));
  return '<span class="badge ' + cls + '" title="Within-cohort specificity ≥ ' + t +
         '× uniform (' + (t * _tcellUniform()).toFixed(3) + ')">' + _tcellTierLabel(t) + '</span>';
}

function getScopedAttribution(kinaseId, filter) {
  // Returns filtered within-cohort attribution rows from
  // PAYLOAD.attribution_index for one kinase. The T-cell attribution carries
  // binned specificity (tcell_tier ∈ {0,1,2,5,10}) + pseudobulk concordance vs
  // bulk NES (no confidence string, no FDR — see
  // docs/tcell_exhaustion_analysis_summary.md). filter dimensions may be
  // string ("" = any) or array ([] = any); celltype scopes the cell_type axis.
  const AI = PAYLOAD.attribution_index || {};
  if (!AI.kinase_id) return [];
  const scopedCtx = getScopedContrastIds(filter);
  const ctSet = _filterSet(filter.celltype);
  const out = [];
  for (let j = 0; j < AI.kinase_id.length; j++) {
    if (AI.kinase_id[j] !== kinaseId) continue;
    if (scopedCtx.size > 0 && !scopedCtx.has(AI.contrast_id[j])) continue;
    if (ctSet.size && !ctSet.has(AI.cell_type[j]))                continue;
    out.push({
      contrast_id:        AI.contrast_id[j],
      cell_type:          AI.cell_type[j],
      tcell_specificity:  AI.tcell_specificity  ? AI.tcell_specificity[j]  : null,
      tcell_tier:         AI.tcell_tier         ? AI.tcell_tier[j]         : 0,
      tcell_lfc:          AI.tcell_lfc          ? AI.tcell_lfc[j]          : null,
      tcell_concordance:  AI.tcell_concordance  ? AI.tcell_concordance[j]  : null,
      tcell_concordant:   AI.tcell_concordant   ? AI.tcell_concordant[j]   : null,
      tcell_consistency:  AI.tcell_consistency  ? AI.tcell_consistency[j]  : 0,
      nes:                AI.nes                ? AI.nes[j]                : null,
      fdr:                AI.fdr                ? AI.fdr[j]                : null,
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

// Max within-cohort specificity tier across attribution rows for this kinase
// under the active filter scope. Returns 0 when no qualifying rows. The tier is
// precomputed (binned in Python), so we read it directly.
function _kineMaxTcellTierScoped(kinaseId, filter) {
  let best = 0;
  for (const e of getScopedAttribution(kinaseId, filter)) {
    const t = e.tcell_tier || 0;
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
      va = _kineMaxAbsNesScoped(a, scopedCtxIds);
      vb = _kineMaxAbsNesScoped(b, scopedCtxIds);
      if (va == null) va = -Infinity;
      if (vb == null) vb = -Infinity;
    }
    else if (col === "n_attributed_celltypes") {
      // Match what the Cell types pill column displays: dedup by cell_type
      // keeping best tier, then count cell types specific at ≥ 1× uniform.
      const _bestTierByCT = (kid) => {
        const m = new Map();
        for (const e of getScopedAttribution(kid, kf)) {
          const t = e.tcell_tier || 0;
          if (t > (m.get(e.cell_type) || 0)) m.set(e.cell_type, t);
        }
        let n = 0;
        for (const t of m.values()) if (t >= 1) n++;
        return n;
      };
      va = _bestTierByCT(a.id);
      vb = _bestTierByCT(b.id);
    }
    else if (col === "n_sig") {
      va = _kineSigCountScoped(a, fdr, scopedCtxIds);
      vb = _kineSigCountScoped(b, fdr, scopedCtxIds);
    }
    else if (col === "tcell_max_tier") {
      va = _kineMaxTcellTierScoped(a.id, kf);
      vb = _kineMaxTcellTierScoped(b.id, kf);
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
  // Cell types this kinase is specific to (≥1× uniform) in the active filter
  // scope — a compact specificity summary (full per-state rows live in the
  // Attribution verdict tab). Dedup by cell_type keeping the best tier.
  const rows = getScopedAttribution(r.id, filter || {});
  const byCell = new Map();
  for (const e of rows) {
    const prev = byCell.get(e.cell_type);
    if (!prev || (e.tcell_tier || 0) > (prev.tcell_tier || 0) ||
        ((e.tcell_tier || 0) === (prev.tcell_tier || 0)
         && (e.tcell_concordance || 0) > (prev.tcell_concordance || 0)))
      byCell.set(e.cell_type, e);
  }
  const displayRows = Array.from(byCell.values()).filter(e => (e.tcell_tier || 0) >= 1);
  // Sort: tier first (10 → 5 → 2), then concordance desc within tier.
  displayRows.sort((a, b) => {
    const dt = (b.tcell_tier || 0) - (a.tcell_tier || 0);
    if (dt !== 0) return dt;
    return (b.tcell_concordance || 0) - (a.tcell_concordance || 0);
  });
  const n = displayRows.length;
  if (n === 0) return `<span class="muted">—</span>`;
  const top = displayRows.slice(0, 3);
  const tip = displayRows.map(e =>
    `${e.cell_type} (≥${e.tcell_tier}× uniform, concordance ${Number(e.tcell_concordance).toFixed(3)})`).join("\n");
  const topNames = top.map(e => {
    const cls = e.tcell_tier >= 10 ? "vhi" : e.tcell_tier >= 5 ? "hi"
      : e.tcell_tier >= 2 ? "mid" : "lo";
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
  // Opt-in specificity narrowing. 0 = Any (off) — the default; hides nothing.
  const tcellMin = Math.max(0, parseInt(kf.tcellMin, 10) || 0);

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
    // Opt-in specificity narrowing (tcellMin > 0, off by default): kinase passes
    // if any attribution row in scope reaches the requested tier (≥ tcellMin ×
    // uniform). Specificity only — concordance is never used to filter (de-gate
    // directive, docs/tcell_exhaustion_analysis_summary.md). Skipped
    // under text search so a targeted lookup always surfaces the kinase.
    if (!q && tcellMin > 0 && _kineMaxTcellTierScoped(r.id, kf) < tcellMin) continue;
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
    const drvCls = (drvSet && drvSet.has(r.id)) ? " driver" : "";

    // Specificity tier badge (max within-cohort tier in scope) + cell-types pill.
    const specBadge = _tcellTierBadge(_kineMaxTcellTierScoped(r.id, colFilter));
    const cellTypesCell = _renderCellTypesCell(r, colFilter);

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
      `<td>${specBadge}</td>` +
      `<td>${cellTypesCell}</td>` +
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
