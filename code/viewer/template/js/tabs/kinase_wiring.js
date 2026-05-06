
// Module-level handle so _syncKinaseFilterUI can re-render multi-selects from
// outside wireKinaseTable's closure (e.g. after a cross-tab handoff).
let _kineRenderMultiselect = null;

function wireKinaseTable() {
  const tbl = document.getElementById("ke-table");
  if (!tbl) return;

  tbl.querySelectorAll("thead th").forEach(th => {
    th.addEventListener("click", () => {
      const col = th.dataset.col;
      const cur = KinaseFilter.get();
      if (col === "nes_profile") {
        // Sort by NES profile: first click selects it descending.
        if (cur.sortCol !== col) KinaseFilter.set({sortCol: col, sortAsc: false});
        else KinaseFilter.set({sortAsc: !cur.sortAsc});
        renderKinaseExplorer();
        return;
      }
      if (cur.sortCol === col) KinaseFilter.set({sortAsc: !cur.sortAsc});
      else KinaseFilter.set({sortCol: col, sortAsc: false});
      renderKinaseExplorer();
    });
  });
  tbl.querySelector("tbody").addEventListener("click", ev => {
    const tr = ev.target.closest("tr.ke-row");
    if (!tr) return;
    const kid = parseInt(tr.dataset.kid, 10);
    Store.dispatch({type:"SET_SELECTION", key:"kinase", value: kid});
  });
  tbl.querySelector("tbody").addEventListener("keydown", ev =>
    _activateRowOnKey(ev, "tr.ke-row", tr => {
      const kid = parseInt(tr.dataset.kid, 10);
      Store.dispatch({type:"SET_SELECTION", key:"kinase", value: kid});
    }));
  const search = document.getElementById("ke-search");
  if (search) {
    search.value = KinaseFilter.get("search");
    search.addEventListener("input", ev => {
      KinaseFilter.set({search: ev.target.value});
      renderKinaseExplorer();
    });
  }

  // Multiselect option sources.
  const AI = PAYLOAD.attribution_index || {};
  const allCells = Array.from(new Set(AI.cell_type || [])).sort();
  const MS_OPTS = {
    disease:    ["App","Tau","ApTt"],
    timepoint:  ["2mo","4mo","6mo"],
    celltype:   allCells,
  };

  // Render a multiselect into its placeholder span. Idempotent.
  function _renderMultiselect(key) {
    const host = document.getElementById("ke-ms-" + key);
    if (!host) return;
    const label = host.dataset.label || key;
    const opts = MS_OPTS[key] || [];
    const cur = (KinaseFilter.get(key) || []).slice();
    const curSet = new Set(cur);
    const summary = cur.length === 0 ? "Any"
      : cur.length <= 2 ? cur.join(", ")
      : `${cur.length} selected`;
    const optsHtml = opts.map(v => {
      const checked = curSet.has(v) ? " checked" : "";
      return `<label class="ms-opt"><input type="checkbox" data-val="${_escapeHtml(v)}"${checked}/>${_escapeHtml(v)}</label>`;
    }).join("");
    host.innerHTML =
      `<span style="margin-right:4px;">${_escapeHtml(label)}</span>` +
      `<span class="ms-wrap">` +
        `<button type="button" class="ms-button" data-active="${cur.length ? 1 : 0}" ` +
          `aria-haspopup="true" aria-expanded="false">${_escapeHtml(summary)}</button>` +
        `<div class="ms-panel" role="listbox" aria-multiselectable="true">` +
          `<div class="ms-action" data-action="clear">Clear</div>` +
          `<div class="ms-divider"></div>` +
          optsHtml +
        `</div>` +
      `</span>`;
    const wrap = host.querySelector(".ms-wrap");
    const btn  = wrap.querySelector(".ms-button");
    const panel = wrap.querySelector(".ms-panel");
    btn.addEventListener("click", ev => {
      ev.stopPropagation();
      const open = panel.classList.toggle("open");
      btn.setAttribute("aria-expanded", open ? "true" : "false");
      // Close other open panels.
      document.querySelectorAll(".ms-panel.open").forEach(p => {
        if (p !== panel) {
          p.classList.remove("open");
          const b = p.parentElement && p.parentElement.querySelector(".ms-button");
          if (b) b.setAttribute("aria-expanded", "false");
        }
      });
    });
    panel.addEventListener("click", ev => ev.stopPropagation());
    panel.querySelectorAll('input[type="checkbox"]').forEach(cb => {
      cb.addEventListener("change", () => {
        const next = (KinaseFilter.get(key) || []).slice();
        const v = cb.dataset.val;
        const i = next.indexOf(v);
        if (cb.checked && i < 0) next.push(v);
        else if (!cb.checked && i >= 0) next.splice(i, 1);
        KinaseFilter.set({[key]: next});
        _renderMultiselect(key);
        renderKinaseExplorer();
      });
    });
    const clearBtn = panel.querySelector('[data-action="clear"]');
    if (clearBtn) clearBtn.addEventListener("click", () => {
      KinaseFilter.set({[key]: []});
      _renderMultiselect(key);
      renderKinaseExplorer();
    });
  }
  // Close panels on outside click (one-time wiring).
  if (!window._msOutsideWired) {
    document.addEventListener("click", () => {
      document.querySelectorAll(".ms-panel.open").forEach(p => {
        p.classList.remove("open");
        const b = p.parentElement && p.parentElement.querySelector(".ms-button");
        if (b) b.setAttribute("aria-expanded", "false");
      });
    });
    window._msOutsideWired = true;
  }

  ["disease","timepoint","celltype"].forEach(_renderMultiselect);
  _kineRenderMultiselect = _renderMultiselect;

  // Confidence (single, ordinal threshold).
  const confSel = document.getElementById("ke-filter-confidence");
  if (confSel) {
    confSel.value = KinaseFilter.get("confidence") || "";
    confSel.addEventListener("change", () => {
      KinaseFilter.set({confidence: confSel.value});
      renderKinaseExplorer();
    });
  }

  // WMB specificity tier minimum (single, ordinal threshold).
  const wmbSel = document.getElementById("ke-filter-wmb");
  if (wmbSel) {
    wmbSel.value = String(KinaseFilter.get("wmbMin") || 0);
    wmbSel.addEventListener("change", () => {
      KinaseFilter.set({wmbMin: parseInt(wmbSel.value, 10) || 0});
      renderKinaseExplorer();
    });
  }

  // n_sig minimum (numeric input).
  const nsigInp = document.getElementById("ke-filter-nsig-min");
  if (nsigInp) {
    nsigInp.value = String(KinaseFilter.get("nSigMin") || 0);
    nsigInp.addEventListener("change", () => {
      const v = Math.max(0, Math.min(9, parseInt(nsigInp.value, 10) || 0));
      nsigInp.value = String(v);
      KinaseFilter.set({nSigMin: v});
      renderKinaseExplorer();
    });
  }

  const resetBtn = document.getElementById("ke-filter-reset");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      KinaseFilter.reset();
      _syncKinaseFilterUI();
      renderKinaseExplorer();
    });
  }
}

// Re-pushes the persisted KinaseFilter state into all the toolbar inputs.
// Used after programmatic mutations (e.g. cross-tab handoff prefilling
// disease/timepoint from a Temporal v2 click) so the dropdowns reflect the
// new state without a full page rebuild.
function _syncKinaseFilterUI() {
  const inp = document.getElementById("ke-search");
  if (inp) inp.value = KinaseFilter.get("search") || "";
  if (_kineRenderMultiselect) {
    ["disease","timepoint","celltype"].forEach(k => _kineRenderMultiselect(k));
  }
  const confSel = document.getElementById("ke-filter-confidence");
  if (confSel) confSel.value = KinaseFilter.get("confidence") || "";
  const wmbSel = document.getElementById("ke-filter-wmb");
  if (wmbSel) wmbSel.value = String(KinaseFilter.get("wmbMin") || 0);
  const nsigInp = document.getElementById("ke-filter-nsig-min");
  if (nsigInp) nsigInp.value = String(KinaseFilter.get("nSigMin") || 0);
}

// ---------------------------------------------------------------------------
// Pathway Explorer tab
// ---------------------------------------------------------------------------
let peSortCol = "tpds";
let peSortAsc = false;
let peSearch = "";
let peTrajectory = "all";          // "all" | "App" | "Tau" | "ApTt" | "2mo" | "4mo" | "6mo"
let _peRows = null;
let _peSearchTimer = null;
let _peTrajMaskCache = null;       // bitmask cached for the active trajectory

const PE_TRAJECTORIES = {
  all:  { label: "All contrasts", contrasts: [] },
  App:  { label: "App trajectory",   contrasts: ["App_2mo","App_4mo","App_6mo"] },
  Tau:  { label: "Tau trajectory",   contrasts: ["Tau_2mo","Tau_4mo","Tau_6mo"] },
  ApTt: { label: "ApTt trajectory",  contrasts: ["ApTt_2mo","ApTt_4mo","ApTt_6mo"] },
  "2mo": { label: "2mo cross-section", contrasts: ["App_2mo","Tau_2mo","ApTt_2mo"] },
  "4mo": { label: "4mo cross-section", contrasts: ["App_4mo","Tau_4mo","ApTt_4mo"] },
  "6mo": { label: "6mo cross-section", contrasts: ["App_6mo","Tau_6mo","ApTt_6mo"] },
};

function _peTrajMask() {
  if (_peTrajMaskCache != null) return _peTrajMaskCache;
  let m = 0;
  for (const c of (PE_TRAJECTORIES[peTrajectory] || PE_TRAJECTORIES.all).contrasts) {
    const idx = CONTRASTS.indexOf(c);
    if (idx >= 0) m |= (1 << idx);
  }
  _peTrajMaskCache = m;
  return m;
}

function _peCsetMatch(rowMask) {
  if (peTrajectory === "all") return true;
  const sel = _peTrajMask();
  // Implicit "any": the backbone passes both nulls in at least one of the
  // contrasts named by the active trajectory.
  return (rowMask & sel) !== 0;
}

function _peContrastChips(mask) {
  // Render passing contrasts as small inline chips. Up to 3 visible, +N overflow tooltip.
  const passing = [];
  for (let i = 0; i < CONTRASTS.length; i++) {
    if (mask & (1 << i)) passing.push(CONTRASTS[i]);
  }
  if (passing.length === 0) return '<span class="muted">—</span>';
  const SHOW = 3;
  const head = passing.slice(0, SHOW)
    .map(c => `<span class="pe-cchip">${c}</span>`).join("");
  const tail = passing.length > SHOW
    ? `<span class="pe-cchip pe-cchip-more" title="${passing.join(", ")}">+${passing.length - SHOW}</span>`
    : "";
  return head + tail;
}

function _popcount(m) {
  m = m - ((m >> 1) & 0x55555555);
  m = (m & 0x33333333) + ((m >> 2) & 0x33333333);
  return (((m + (m >> 4)) & 0x0f0f0f0f) * 0x01010101) >> 24;
}

function _pathwayEvidenceLabel(v) {
  return PATHWAY_EVIDENCE_LABELS[v] || "Unknown";
}

function _pathwayEvidenceClass(v) {
  if (v === "expression-confirmed") return "expr";
  if (v === "kinase-imputed") return "imp";
  if (v === "mixed") return "mix";
  return "lo";
}

function _pathwayEvidenceRank(v) {
  if (v === "expression-confirmed") return 0;
  if (v === "kinase-imputed") return 1;
  if (v === "mixed") return 2;
  return 3;
}

function _pathwayEvidenceBadge(v) {
  return `<span class="badge ${_pathwayEvidenceClass(v)}">${_pathwayEvidenceLabel(v)}</span>`;
}

function _pathwayEvidenceChip(v, label) {
  return `<span class="pe-chip ${_pathwayEvidenceClass(v)}">${label}</span>`;
}

function _buildPathwayRowModel() {
  const BB = PAYLOAD.backbones;
  const n = BB.id.length;
  const tpdsCols = CONTRASTS.map(c => BB["mean_tpds_" + c]);
  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const tpds = new Array(CONTRASTS.length);
    for (let c = 0; c < CONTRASTS.length; c++) tpds[c] = tpdsCols[c][i];
    out[i] = {
      idx: i,
      id: BB.id[i],
      receiver_id: BB.receiver_id[i],
      receiver: RECEIVERS[BB.receiver_id[i]],
      Receptor: BB.Receptor[i] || "",
      EM: BB.EM[i] || "",
      Target: BB.Target[i] || "",
      sender_mask: BB.sender_mask[i],
      n_senders: BB.n_senders[i],
      n_senders_sig: BB.n_senders_significant[i],
      max_abs_tpds: BB.max_abs_tpds[i],
      sig_mask: BB.significant_both_mask[i],
      sig_count: _popcount(BB.significant_both_mask[i]),
      pathway_evidence_all: BB.all_contrasts_pathway_evidence[i] || "expression-confirmed",
      all_imputed_nodes_union: BB.all_imputed_nodes_union[i] || "",
      all_n_expression_confirmed: BB.all_n_expression_confirmed[i] || 0,
      all_n_kinase_imputed: BB.all_n_kinase_imputed[i] || 0,
      _tpds: tpds,
    };
  }
  return out;
}

function _ensurePathwayIndexes() {
  if (_peRows === null) _peRows = _buildPathwayRowModel();
  _ensureBackboneIdx();
}

function _peCompare(a, b, cIdx) {
  const col = peSortCol;
  let va, vb;
  if (col === "tpds") {
    va = cIdx >= 0 ? a._tpds[cIdx] : a.max_abs_tpds;
    vb = cIdx >= 0 ? b._tpds[cIdx] : b.max_abs_tpds;
    if (va == null) va = -Infinity;
    if (vb == null) vb = -Infinity;
  }
  else if (col === "passing_contrasts") { va = a.sig_count; vb = b.sig_count; }
  else if (col === "receiver") { va = a.receiver; vb = b.receiver; }
  else if (col === "pathway_evidence") {
    va = _pathwayEvidenceRank(cIdx >= 0
      ? (PAYLOAD.backbones["pathway_evidence_backbone_" + CONTRASTS[cIdx]][a.idx] || "expression-confirmed")
      : a.pathway_evidence_all);
    vb = _pathwayEvidenceRank(cIdx >= 0
      ? (PAYLOAD.backbones["pathway_evidence_backbone_" + CONTRASTS[cIdx]][b.idx] || "expression-confirmed")
      : b.pathway_evidence_all);
  }
  else { va = a[col]; vb = b[col]; }
  if (va == null && vb == null) return 0;
  if (va == null) return 1;
  if (vb == null) return -1;
  if (typeof va === "string") return peSortAsc
    ? va.localeCompare(vb) : vb.localeCompare(va);
  return peSortAsc ? (va - vb) : (vb - va);
}

