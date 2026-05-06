function _checkRequirement(req) {
  const f = Store.state.filters;
  const sel = Store.state.selection;
  if (req.type === "filter") {
    if (req.notEqual !== undefined) return f[req.key] !== req.notEqual;
    if (req.equal !== undefined) return f[req.key] === req.equal;
    return f[req.key] != null;
  }
  if (req.type === "selection") return sel[req.key] != null;
  return true;
}

function renderUnmetPrerequisite(panelEl, tab) {
  const manifest = TAB_MANIFEST[tab];
  if (!manifest || !manifest.requires || manifest.requires.length === 0)
    return false;
  const unmet = manifest.requires.find(r => !_checkRequirement(r));
  if (!unmet) return false;
  const card = document.createElement("div");
  card.className = "prereq-card";
  card.innerHTML =
    '<span class="prereq-icon" aria-hidden="true">&#9888;</span>' +
    '<div class="prereq-msg"></div>' +
    '<button type="button" class="prereq-action"></button>';
  card.querySelector(".prereq-msg").textContent = unmet.message;
  const btn = card.querySelector(".prereq-action");
  btn.textContent = unmet.cta;
  btn.addEventListener("click", () => {
    if (unmet.goTo) {
      Store.dispatch({type:"SET_VIEW", key:"activeTab", value:unmet.goTo});
    } else if (unmet.focus) {
      const el = document.getElementById(unmet.focus);
      if (el) { el.focus(); if (el.click) try { el.click(); } catch(_){} }
    }
  });
  panelEl.innerHTML = "";
  panelEl.appendChild(card);
  return true;
}

// ---------------------------------------------------------------------------
// URL-hash sync — serialize tab/filters/selection into location.hash so
// reload and back/forward restore state. Only non-default keys are emitted
// to keep the URL short. Suppresses re-broadcast while applying inbound.
// ---------------------------------------------------------------------------
const _HASH_DEFAULTS = {
  t: "signal", r: "ALL", s: "ALL",
  fdr: 0.25,
  k: null, b: null, ct: null,
};
let _hashApplying = false;

function _serializeHash() {
  const v = Store.state.view, f = Store.state.filters, s = Store.state.selection;
  const cur = {
    t: v.activeTab, r: f.receiver,
    s: f.pathwayEvidence, fdr: f.fdr,
    k: s.kinase, b: s.backbone, ct: s.celltype,
  };
  const parts = [];
  for (const k in cur) {
    const val = cur[k];
    if (val == null) continue;
    if (val === _HASH_DEFAULTS[k]) continue;
    parts.push(encodeURIComponent(k) + "=" + encodeURIComponent(String(val)));
  }
  return parts.length ? "#" + parts.join("&") : "";
}

function pushHash() {
  if (_hashApplying) return;
  const h = _serializeHash();
  if (h === window.location.hash) return;
  // Use replaceState so each filter twiddle doesn't pollute history; only
  // tab changes create a new history entry.
  history.replaceState(null, "", h || window.location.pathname + window.location.search);
}

function applyHash() {
  const raw = (window.location.hash || "").replace(/^#/, "");
  if (!raw) return;
  const parts = raw.split("&");
  const map = {};
  parts.forEach(p => {
    const [k, v] = p.split("=").map(decodeURIComponent);
    if (k) map[k] = v;
  });
  _hashApplying = true;
  try {
    if (map.r != null) Store.dispatch({type:"SET_FILTER", key:"receiver", value:map.r});
    if (map.s != null) Store.dispatch({type:"SET_FILTER", key:"pathwayEvidence", value:map.s});
    if (map.fdr != null) Store.dispatch({type:"SET_FILTER", key:"fdr", value:parseFloat(map.fdr)});
    if (map.k != null) Store.dispatch({type:"SET_SELECTION", key:"kinase", value:parseInt(map.k,10)});
    if (map.b != null) Store.dispatch({type:"SET_SELECTION", key:"backbone", value:parseInt(map.b,10)});
    if (map.ct != null) Store.dispatch({type:"SET_SELECTION", key:"celltype", value:parseInt(map.ct,10)});
    if (map.t != null) Store.dispatch({type:"SET_VIEW", key:"activeTab", value:map.t});
  } finally {
    _hashApplying = false;
  }
}

// ---------------------------------------------------------------------------
// Derived-array memoization — keyed on JSON signature of filters slice
// ---------------------------------------------------------------------------
let _filteredCache = { key:null, gnRef:null, indices:null };

function _computeFilteredIndices() {
  const f = Store.state.filters;
  const sel = Store.state.selection;
  const BB = PAYLOAD.backbones;
  const n = BB.id.length;
  const cIdx = CONTRASTS.indexOf(f.contrast);
  const tpdsCol = cIdx >= 0 ? BB["mean_tpds_" + f.contrast] : null;
  const sigCol = BB.significant_both_mask;
  const evidenceCol = cIdx >= 0
    ? BB["pathway_evidence_backbone_" + f.contrast]
    : BB.all_contrasts_pathway_evidence;
  const rIdx = (f.receiver === "ALL") ? -1 : RECEIVERS.indexOf(f.receiver);
  // graphNodeIds is a transient filter applied after a Pathway Graph node
  // click. Stored as a Set of backbone_id for O(1) membership.
  const gnSet = (f.graphNodeIds && f.graphNodeIds.length)
    ? new Set(f.graphNodeIds) : null;
  const senderBit = (f.sender == null) ? 0 : (1 << f.sender);
  const senderMaskCol = BB.sender_mask;
  // selection.kinase mask is null until the slice loads — the SET_SELECTION
  // subscriber re-renders on resolve so the unconstrained pass is transient.
  const kSet = (sel.kinase != null)
    ? SliceCache.kinaseBackboneSetSync(sel.kinase) : null;
  const ctIdx = (sel.celltype != null) ? sel.celltype : -1;
  // TPDS-magnitude significance: gates on whether the chain's TPDS is
  // distinguishable from zero, distinct from the kinase chain test gated
  // via significant_both_mask. Threshold via UI dropdown (off / 0.10 /
  // 0.05 / 0.01).
  const tpdsSigCol = (f.tpdsSig === "0.01") ? BB.tpds_sig_001_mask
                   : (f.tpdsSig === "0.05") ? BB.tpds_sig_005_mask
                   : (f.tpdsSig === "0.10") ? BB.tpds_sig_010_mask
                   : null;
  const out = [];
  for (let i = 0; i < n; i++) {
    if (rIdx >= 0 && BB.receiver_id[i] !== rIdx) continue;
    if (ctIdx >= 0 && BB.receiver_id[i] !== ctIdx) continue;
    if (senderBit && !(senderMaskCol[i] & senderBit)) continue;
    if (f.pathwayEvidence !== "ALL") {
      const ev = evidenceCol ? evidenceCol[i] : null;
      if (ev !== f.pathwayEvidence) continue;
    }
    if (cIdx >= 0) {
      if (!((sigCol[i] >> cIdx) & 1)) continue;
      const t = tpdsCol[i];
      if (t == null) continue;
    }
    if (tpdsSigCol !== null) {
      // contrast=ALL ⇒ require TPDS significance in any contrast.
      // contrast=specific ⇒ require it in that contrast.
      if (cIdx >= 0) {
        if (!((tpdsSigCol[i] >> cIdx) & 1)) continue;
      } else {
        if (tpdsSigCol[i] === 0) continue;
      }
    }
    if (gnSet !== null && !gnSet.has(BB.id[i])) continue;
    if (kSet !== null && !kSet.has(BB.id[i])) continue;
    out.push(i);
  }
  return out;
}

function getFilteredIndices() {
  const f = Store.state.filters;
  const sel = Store.state.selection;
  // graphNodeIds array identity changes on each SET_FILTER dispatch (reducer
  // deep-clones state) — use identity, not stringify, to avoid scanning the
  // full array on every read.
  const gnKey = f.graphNodeIds ? ("gn:" + f.graphNodeIds.length) : "gn:null";
  const gnRef = f.graphNodeIds;  // also compare by identity
  // kLoaded distinguishes pre-load (no mask) from post-load (mask applied).
  const kLoaded = (sel.kinase != null
    && SliceCache.kinaseBackboneSetSync(sel.kinase) !== null) ? "1" : "0";
  const key = f.contrast + "|" + f.receiver + "|"
            + f.pathwayEvidence + "|" + f.fdr + "|" + gnKey + "|s:" + (f.sender ?? "")
            + "|k:" + (sel.kinase ?? "") + "/" + kLoaded
            + "|c:" + (sel.celltype ?? "")
            + "|t:" + (f.tpdsSig ?? "OFF");
  if (key !== _filteredCache.key || gnRef !== _filteredCache.gnRef) {
    _filteredCache = {
      key, gnRef, indices: _computeFilteredIndices(),
    };
  }
  return _filteredCache.indices;
}
function invalidateFilterCache(){ _filteredCache.key = null; }
