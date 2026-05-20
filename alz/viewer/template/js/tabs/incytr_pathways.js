// ---------------------------------------------------------------------------
// Incytr Pathways tab — table over a single (sender, receiver) shard or the
// union of multiple selected pairs. Filter UI mirrors the Kinase tab:
// multiselect popovers for Sender / Receiver / Disease / Timepoint;
// 3 numeric inputs for metric thresholds; Reset button + live count.
//
// Filter state lives in IncytrFilter (shared with the heatmap tab). The
// heatmap click handler writes pair + senderIn + receiverIn + disease +
// timepoint into IncytrFilter and switches tabs.
// ---------------------------------------------------------------------------

const _IP_DISEASES = ["App", "Tau", "ApTt"];
const _IP_TIMEPOINTS = ["2mo", "4mo", "6mo"];
// Per-shard pagination: the table renders at most one (sender, receiver)
// shard at a time, paginated client-side. Page size bounds DOM render cost;
// the loaded shard itself bounds JS heap residency. Previous "load all
// shards and concat" architecture peaked at ~10M JS objects and OOM'd the
// browser process.
const _IP_PAGE_SIZE = 100;

// Coarse trajectory labels (CR-04). Ordered for chip display.
// Sign vectors are triples of 'u' (up), 'd' (down), 'f' (flat) at 2/4/6 mo.
const _IP_TRAJ_LABELS = [
  "always-up", "always-down", "monotonic-up", "monotonic-down",
  "early-only", "late-onset", "mixed", "flat",
];
// Color palette for trajectory chips — distinct hues per label.
const _IP_TRAJ_COLORS = {
  "always-up":      { bg: "#ffe0e0", fg: "#a3203c", border: "#e8a0a0" },
  "always-down":    { bg: "#dde8f8", fg: "#1f4ea3", border: "#a0bee8" },
  "monotonic-up":   { bg: "#fff0d0", fg: "#9a5000", border: "#e8c080" },
  "monotonic-down": { bg: "#e0f0ff", fg: "#005090", border: "#80c0e0" },
  "early-only":     { bg: "#e8f8e0", fg: "#206020", border: "#90d080" },
  "late-onset":     { bg: "#f0e8f8", fg: "#602080", border: "#c090e0" },
  "mixed":          { bg: "#f8f0e0", fg: "#705020", border: "#d0b080" },
  "flat":           { bg: "#f0f0f0", fg: "#606060", border: "#c0c0c0" },
};
// Human-readable tooltip for each label (sign vector semantics).
const _IP_TRAJ_TIPS = {
  "always-up":      "Sign vector uuu: up at 2, 4, and 6 months in this disease contrast.",
  "always-down":    "Sign vector ddd: down at 2, 4, and 6 months.",
  "monotonic-up":   "Sign monotonically up: starts flat or up and rises (e.g., fuu, uuu).",
  "monotonic-down": "Sign monotonically down: starts flat or down and falls (e.g., fdd, ddd).",
  "early-only":     "Signal at early timepoints only (uff, udf, dff — dies by 6 mo).",
  "late-onset":     "Signal appears late (ffu, ffd — absent at 2 mo, present at 6 mo).",
  "mixed":          "Mixed direction across timepoints (not monotonic; e.g., udu, dud).",
  "flat":           "No signal at any timepoint (|PDS| < 0.01 AND pvalue ≥ 0.05 at all three).",
};

// Score and per-node FC columns are advertised on the payload block but kept
// in module-local fallbacks so the JS stays usable against an older payload.
const _IP_SCORE_COLS_FALLBACK = ["TPDS", "PPDS", "PhPDS_ps", "PhPDS_py", "SiK_score"];
const _IP_FC_NODES_FALLBACK = ["Ligand", "Receptor", "EM", "Target"];
const _IP_FC_METRICS_FALLBACK = [
  "sclog2FC",
  "pr_log2FC",
  "ps_log2FC",
  "py_log2FC",
];
const _IP_FC_METRIC_LABELS = {
  "sclog2FC":  "sc",
  "pr_log2FC": "pr",
  "ps_log2FC": "ps",
  "py_log2FC": "py",
};
const _IP_FC_METRIC_TIPS = {
  "sclog2FC":  "Single-cell expression log₂ fold-change for this gene.",
  "pr_log2FC": "Bulk-proteomics log₂ fold-change for this gene.",
  "ps_log2FC": "Phosphoserine log₂ fold-change for this site/gene.",
  "py_log2FC": "Phosphotyrosine log₂ fold-change for this site/gene.",
};
const _IP_FC_NODE_TIPS = {
  "Ligand":   "Per-omics fold-changes for the ligand gene.",
  "Receptor": "Per-omics fold-changes for the receptor gene.",
  "EM":       "Per-omics fold-changes for the effector molecule.",
  "Target":   "Per-omics fold-changes for the target gene.",
};

// Per-node evidence-source labels. Each of Ligand/Receptor/EM/Target carries a
// {DEG, prG} tag indicating which seed list admitted the gene:
//   DEG → single-cell differentially-expressed gene (transcript evidence)
//   prG → proteomics-significant gene (bulk-protein evidence)
// Rendered inline next to the gene name as a small colored badge.
const _IP_LABEL_NODES_FALLBACK = ["Ligand", "Receptor", "EM", "Target"];
const _IP_LABEL_COLORS = {
  "DEG": { bg: "#e8eefc", fg: "#1f4ea3" },   // blue — single-cell
  "prG": { bg: "#e7f4ec", fg: "#1f7a3a" },   // green — proteomics
};
function _ipLabelNodes() {
  const block = _ipBlock();
  return (block && block.label_nodes) || _IP_LABEL_NODES_FALLBACK;
}

// Column-header tooltips. Native browser title= popups — kept short so they
// fit on one line in most viewers. The four node columns reference the
// evidence-source badge legend rendered above the table.
const _IP_SCORE_TIPS = {
  "TPDS":             "Transcriptomic PDS — aggregated from factorial OLS β across the 4 nodes on single-cell expression. Range ≈ [-1, +1] (sign indicates direction vs WT).",
  "PPDS":             "Proteomic PDS — aggregated from factorial OLS β across the 4 nodes on bulk proteomics.",
  "PhPDS_ps":         "Phosphoserine PDS — aggregated from factorial OLS β on pS site intensities.",
  "PhPDS_py":         "Phosphotyrosine PDS — aggregated from factorial OLS β on pY site intensities.",
  "SiK_score":        "Signaling-kinase composite — aggregates per-condition SiK_<X>_of_<Y> kinase-substrate evidence into a single per-path score (treatment condition only; WT side dropped during reshape). Pair-mode reshape only; NULL when source is the legacy factorial cache.",
};
function _ipNodeCell(name, label) {
  const safeName = _escapeHtml(name == null ? "" : name);
  if (!label) return safeName;
  const c = _IP_LABEL_COLORS[label] || { bg: "#eee", fg: "#444" };
  const badge = `<span class="ip-evidence" style="`
    + `display:inline-block;margin-left:4px;padding:0 4px;`
    + `font-size:10px;font-family:ui-monospace,monospace;`
    + `border-radius:2px;background:${c.bg};color:${c.fg};vertical-align:middle;"`
    + ` title="evidence source: ${_escapeHtml(label)}">${_escapeHtml(label)}</span>`;
  return safeName + badge;
}

const _ipRuntime = {
  rows:           null,         // rows from the currently-loaded single shard
  loadedKey:      null,         // "<sender>||<receiver>" of shard currently loaded
  loading:        false,
  loadError:      null,
  openKeys:       new Set(),    // keys of rows whose per-node FC detail is expanded
  detailTab:      {},           // rk → "fc" | "trajectory" (which sub-tab is active)
  page:           0,            // 0-indexed current page (post-filter, post-sort)
};

// ---------------------------------------------------------------------------
// Trajectory helpers (CR-04)
// ---------------------------------------------------------------------------
// traj_label and sign_vec are shard columns — read directly from each row.
// recur_index is in the payload block (small enough to inline).

function _ipRecurIndex() {
  const block = _ipBlock();
  return (block && block.recur_index) || null;
}

// Derive a path string from a shard row — matches the Python build key:
// sender||receiver||Path.
function _ipPathStr(r) {
  return `${r._sender}||${r._receiver}||${r.Path}`;
}

// Return the trajectory entry for this row: { traj_label, sign_vec }.
// Reads from shard columns traj_label / sign_vec (set by the Python build).
// Returns null when the payload pre-dates CR-04 (version < 2).
function _ipTrajEntry(r) {
  if (r.traj_label == null) return null;
  return { traj_label: r.traj_label, sign_vec: r.sign_vec || "???" };
}

// Return all distinct (disease, traj_label, sign_vec) tuples for a path across
// the currently-loaded rows — used by the trajectory chart to show all 3 diseases.
function _ipTrajEntriesForPath(pathStr, allRows) {
  const seen = new Set();
  const out = [];
  for (const r of (allRows || [])) {
    if (_ipPathStr(r) !== pathStr) continue;
    if (r.traj_label == null) continue;
    const [dis] = (r.contrast || "").split("_");
    const key = `${dis}|${r.traj_label}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ contrast: dis, traj_label: r.traj_label, sign_vec: r.sign_vec || "" });
  }
  return out;
}

// Return the set of disease contrasts in which this path is significant,
// using recur_index when available, else falling back to loaded rows.
function _ipRecurSetForPath(pathStr) {
  const ri = _ipRecurIndex();
  if (ri && ri[pathStr]) return new Set(ri[pathStr]);
  return new Set();
}

// Check whether trajectory data is available (payload version >= 2).
function _ipHasTraj() {
  const block = _ipBlock();
  return !!(block && block.version >= 2);
}

function _ipScoreCols() {
  const block = _ipBlock();
  return (block && block.score_columns) || _IP_SCORE_COLS_FALLBACK;
}
function _ipFcNodes() {
  const block = _ipBlock();
  return (block && block.fc_nodes) || _IP_FC_NODES_FALLBACK;
}
function _ipFcMetrics() {
  const block = _ipBlock();
  return (block && block.fc_metrics) || _IP_FC_METRICS_FALLBACK;
}
function _ipRowKey(r) {
  return `${r._sender}||${r._receiver}||${r.Path}||${r.contrast}`;
}

function _ipBlock() {
  return (typeof PAYLOAD !== "undefined" && PAYLOAD.incytr_pathways) || null;
}

function _ipPairsInScope(block) {
  // Per-shard pagination: scope is always 0 or 1 (sender, receiver) pair.
  // The table renders exactly one shard at a time. Returns [{sender,
  // receiver}] (length 0 or 1).
  //
  // Resolution order:
  //   1. f.pair (pinned from heatmap click) — always wins
  //   2. Intersection of senderIn × receiverIn × present_pairs — if exactly
  //      one pair, use it
  //   3. Otherwise: empty (UI will prompt user to narrow)
  const f = IncytrFilter.get();
  if (f.pair) return [f.pair];
  const sIn = new Set(f.senderIn || []);
  const rIn = new Set(f.receiverIn || []);
  const matches = [];
  for (const [s, r] of block.slice_index.present) {
    if (sIn.size && !sIn.has(s)) continue;
    if (rIn.size && !rIn.has(r)) continue;
    matches.push({ sender: s, receiver: r });
    if (matches.length > 1) return matches; // caller only cares about >1 vs ==1
  }
  return matches;
}

function _ipScopeSig(pairs) {
  return pairs.map(p => p.sender + "||" + p.receiver).sort().join(";");
}

// ---- toolbar builders ----

function _ipMountMultiselect(hostId, label, options, key) {
  const host = document.getElementById(hostId);
  if (!host) return;
  mountMultiselect(host, {
    label, options,
    current: IncytrFilter.get(key) || [],
    onChange: (next) => {
      // Picking sender/receiver clears any pinned pair from the heatmap.
      const patch = { [key]: next };
      if (key === "senderIn" || key === "receiverIn") patch.pair = null;
      IncytrFilter.set(patch);
      _ipMountMultiselect(hostId, label, options, key);   // re-render badge
      _ipInvalidateScope();
      _ipEnsureShards();
    },
  });
}

function _ipSyncControls(block) {
  const f = IncytrFilter.get();

  _ipMountMultiselect("ip-ms-sender",   "Sender",    block.senders,   "senderIn");
  _ipMountMultiselect("ip-ms-receiver", "Receiver",  block.receivers, "receiverIn");
  _ipMountMultiselect("ip-ms-disease",  "Disease",   _IP_DISEASES,    "disease");
  _ipMountMultiselect("ip-ms-time",     "Timepoint", _IP_TIMEPOINTS,  "timepoint");

  // CR-04: "Recur in" multiselect — AND gate across selected disease contrasts.
  _ipMountMultiselect("ip-ms-recur", "Recur in", _IP_DISEASES, "recurContrasts");

  // Numeric sliders.
  const set = (id, v) => {
    const el = document.getElementById(id);
    if (el) el.value = (v == null || !isFinite(v)) ? "" : v;
  };
  set("ip-slider-p",   f.sliderP);
  set("ip-slider-pds", f.sliderPds);

  // CR-04: trajectory chips.
  _ipRenderTrajChips();
}

// Render the trajectory chip bar. Chips toggle trajLabels (OR within the set).
function _ipRenderTrajChips() {
  const host = document.getElementById("ip-traj-chips");
  if (!host) return;
  if (!_ipHasTraj()) {
    // Payload pre-dates CR-04 — hide the chip bar gracefully.
    host.style.display = "none";
    return;
  }
  host.style.display = "flex";
  const selected = new Set(IncytrFilter.get("trajLabels") || []);
  host.innerHTML = _IP_TRAJ_LABELS.map(label => {
    const on = selected.has(label);
    const c = _IP_TRAJ_COLORS[label] || { bg: "#eee", fg: "#444", border: "#bbb" };
    const tip = _IP_TRAJ_TIPS[label] || label;
    return `<button type="button" data-ip-traj="${_escapeHtml(label)}"
      title="${_escapeHtml(tip)}"
      style="padding:3px 10px;border-radius:12px;font-size:11px;cursor:pointer;
             border:1.5px solid ${on ? c.fg : c.border};
             background:${on ? c.fg : c.bg};
             color:${on ? "#fff" : c.fg};
             font-weight:${on ? "600" : "400"};"
    >${_escapeHtml(label)}</button>`;
  }).join("");
  host.addEventListener("click", ev => {
    const btn = ev.target.closest("button[data-ip-traj]");
    if (!btn) return;
    const label = btn.dataset.ipTraj;
    const cur = new Set(IncytrFilter.get("trajLabels") || []);
    if (cur.has(label)) cur.delete(label);
    else cur.add(label);
    IncytrFilter.set({ trajLabels: [...cur] });
    _ipRenderTrajChips();
    _ipResetPage();
    _ipRenderTable();
  });
}

function _ipInvalidateScope() {
  _ipRuntime.rows = null;
  _ipRuntime.loadedKey = null;
  _ipRuntime.openKeys = new Set();
  _ipRuntime.page = 0;
}

function _ipResetPage() {
  _ipRuntime.page = 0;
}

// ---- shard loading ----

async function _ipEnsureShards() {
  const block = _ipBlock();
  if (!block) return;
  const pairs = _ipPairsInScope(block);
  if (pairs.length !== 1) {
    // 0 pairs: prompt user. >1 pairs: also a prompt (per-shard pagination
    // means the table is always single-pair). Either way, no shard load.
    _ipRuntime.rows = null;
    _ipRuntime.loadedKey = null;
    _ipRuntime.loading = false;
    _ipRuntime.loadError = null;
    _ipRenderTable();
    return;
  }
  const p = pairs[0];
  const sig = _ipScopeSig(pairs);
  if (_ipRuntime.loadedKey === sig) {
    _ipRenderTable();
    return;
  }
  _ipRuntime.loading = true;
  _ipRuntime.loadError = null;
  _ipRuntime.rows = null;
  _ipResetPage();
  _ipRenderTable();
  try {
    const rows = await SliceCache.loadIncytrShard(p.sender, p.receiver);
    // Stamp sender/receiver so per-row UI helpers can reach the originating
    // pair without consulting the scope state.
    for (const r of rows) { r._sender = p.sender; r._receiver = p.receiver; }
    // Resolve race: only commit if the scope hasn't changed mid-fetch.
    const newSig = _ipScopeSig(_ipPairsInScope(block));
    if (newSig !== sig) return;
    _ipRuntime.rows = rows;
    _ipRuntime.loadedKey = sig;
  } catch (e) {
    _ipRuntime.loadError = String(e.message || e);
    console.error("incytr shard load failed", e);
  } finally {
    _ipRuntime.loading = false;
    _ipRenderTable();
  }
}

// ---- row filtering + sort ----

function _ipFilterRows() {
  if (!_ipRuntime.rows) return [];
  const f = IncytrFilter.get();
  const diseaseSet    = new Set(f.disease        || []);
  const timeSet       = new Set(f.timepoint      || []);
  const trajSet       = new Set(f.trajLabels     || []);
  const recurSet      = new Set(f.recurContrasts || []);
  const hasRecur      = recurSet.size > 0;
  const hasTraj       = trajSet.size > 0 && _ipHasTraj();
  const recurIdx      = hasRecur ? _ipRecurIndex() : null;

  // For recur filtering: if recur_index is absent, build a fast path_id →
  // disease-set map from the already-loaded rows (applying the active
  // pvalue/|PDS| gates) so we don't have to re-scan for every row.
  let recurPathMap = null;
  if (hasRecur && !recurIdx) {
    recurPathMap = new Map();
    for (const r of _ipRuntime.rows) {
      const pSig = f.sliderP   == null || (r.pvalue != null && r.pvalue < f.sliderP);
      const pdsSig = f.sliderPds == null || Math.abs(r.PDS || 0) >= f.sliderPds;
      if (!pSig || !pdsSig) continue;
      const [d] = (r.contrast || "").split("_");
      const pid = _ipPathStr(r);
      if (!recurPathMap.has(pid)) recurPathMap.set(pid, new Set());
      recurPathMap.get(pid).add(d);
    }
  }

  const out = [];
  for (const r of _ipRuntime.rows) {
    if (diseaseSet.size || timeSet.size) {
      const [d, t] = (r.contrast || "").split("_");
      if (diseaseSet.size && !diseaseSet.has(d)) continue;
      if (timeSet.size    && !timeSet.has(t))    continue;
    }
    if (f.sliderP   != null && !(r.pvalue       <  f.sliderP))   continue;
    if (f.sliderPds != null && !(Math.abs(r.PDS || 0) >= f.sliderPds)) continue;

    // CR-04: trajectory label chip filter (OR across selected labels).
    // traj_label is a shard column — read directly from the row.
    if (hasTraj) {
      if (!trajSet.has(r.traj_label)) continue;
    }

    // CR-04: recur_in filter (AND across selected disease contrasts).
    if (hasRecur) {
      const pid = _ipPathStr(r);
      let diseases;
      if (recurIdx) {
        diseases = new Set(recurIdx[pid] || []);
      } else if (recurPathMap) {
        diseases = recurPathMap.get(pid) || new Set();
      } else {
        diseases = new Set();
      }
      let passes = true;
      for (const d of recurSet) {
        if (!diseases.has(d)) { passes = false; break; }
      }
      if (!passes) continue;
    }

    out.push(r);
  }
  const key = f.sortKey, dir = f.sortDir;
  const numericKeys = new Set([
    "pvalue", "PDS", ..._ipScoreCols(),
  ]);
  out.sort((a, b) => {
    const av = a[key], bv = b[key];
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    if (numericKeys.has(key)) return dir * (av - bv);
    return dir * (String(av).localeCompare(String(bv)));
  });
  return out;
}

function _ipFmtNum(v, digits) {
  if (v == null || !isFinite(v)) return "—";
  if (digits === "sci" && Math.abs(v) < 0.01 && v !== 0) return v.toExponential(2);
  return Number(v).toFixed(digits == null ? 3 : digits);
}

function _ipRenderTable() {
  const countEl = document.getElementById("ip-count");
  const wrap = document.getElementById("ip-table-wrap");
  const block = _ipBlock();
  if (!wrap || !countEl || !block) return;

  const pairs = _ipPairsInScope(block);
  if (pairs.length === 0) {
    countEl.textContent = "No (sender, receiver) pair matches the current selection.";
    wrap.innerHTML = '<div class="muted" style="padding:16px;">'
      + 'Pick one sender and one receiver from the filters, or click a cell in the heatmap, to load that pair\'s shard.'
      + '</div>';
    return;
  }
  if (pairs.length > 1) {
    // Per-shard pagination: the table renders one (sender, receiver) shard
    // at a time. Show the matching pairs so the user can pick.
    countEl.textContent = `${pairs.length} matching (sender, receiver) pairs — narrow to one to load.`;
    const items = pairs.map(p =>
      `<li><button type="button" class="ip-pair-pick" `
      + `data-ip-sender="${_escapeHtml(p.sender)}" `
      + `data-ip-receiver="${_escapeHtml(p.receiver)}">`
      + `${_escapeHtml(p.sender)} → ${_escapeHtml(p.receiver)}</button></li>`
    ).join("");
    wrap.innerHTML = '<div style="padding:16px;">'
      + '<div class="muted" style="margin-bottom:8px;">'
      + 'Select a single (sender, receiver) pair to view its pathways:'
      + '</div><ul style="list-style:none;padding:0;margin:0;">' + items + '</ul></div>';
    wrap.querySelectorAll(".ip-pair-pick").forEach(btn => {
      btn.addEventListener("click", () => {
        IncytrFilter.set({
          senderIn: [btn.dataset.ipSender],
          receiverIn: [btn.dataset.ipReceiver],
        });
        _ipInvalidateScope();
        _ipEnsureShards();
      });
    });
    return;
  }
  if (_ipRuntime.loading) {
    const p = pairs[0];
    countEl.textContent = `Loading shard for ${p.sender} → ${p.receiver}…`;
    wrap.innerHTML = '<div class="muted" style="padding:16px;">Fetching shard…</div>';
    return;
  }
  if (_ipRuntime.loadError) {
    countEl.textContent = "Shard load failed.";
    wrap.innerHTML = `<div class="muted" style="padding:16px;">${_escapeHtml(_ipRuntime.loadError)}</div>`;
    return;
  }
  if (!_ipRuntime.rows) {
    countEl.textContent = "";
    wrap.innerHTML = "";
    return;
  }
  if (!_ipRuntime.rows.length) {
    countEl.textContent = "No rows in the selected shard.";
    wrap.innerHTML = '<div class="muted" style="padding:16px;">Empty (likely an empty-DEG cell type).</div>';
    return;
  }
  const filtered = _ipFilterRows();
  const total = _ipRuntime.rows.length;
  const f = IncytrFilter.get();
  // Pagination math — clamp page in case filter shrinks the row set.
  const nPages = Math.max(1, Math.ceil(filtered.length / _IP_PAGE_SIZE));
  if (_ipRuntime.page >= nPages) _ipRuntime.page = nPages - 1;
  const page = _ipRuntime.page;
  const startIdx = page * _IP_PAGE_SIZE;
  const endIdx = Math.min(filtered.length, startIdx + _IP_PAGE_SIZE);
  const p0 = pairs[0];
  countEl.textContent =
    `${p0.sender} → ${p0.receiver}: `
    + `${filtered.length.toLocaleString()} rows pass filters `
    + `(of ${total.toLocaleString()} in shard). `
    + `Page ${page + 1} / ${nPages.toLocaleString()} `
    + `(rows ${(startIdx + 1).toLocaleString()}–${endIdx.toLocaleString()}).`;

  const hasTrajIdx = _ipHasTraj();
  const scoreCols = _ipScoreCols().map(k => ({
    key: k, label: k, numeric: true, digits: 3,
    tip: _IP_SCORE_TIPS[k] || `${k} score column from Incytr factorial scoring.`,
  }));
  const cols = [
    { key: "_sender",       label: "Sender",
      tip: "WMB cell-type class emitting the ligand." },
    { key: "_receiver",     label: "Receiver",
      tip: "WMB cell-type class receiving the signal." },
    { key: "Path",          label: "Path",
      tip: "4-node signaling path: Ligand → Receptor → EM → Target." },
    { key: "Ligand",        label: "Ligand",   labelKey: "Ligand_label",
      tip: "Secreted/membrane ligand gene at the start of the path. Badge marks the evidence source (DEG/prG)." },
    { key: "Receptor",      label: "Receptor", labelKey: "Receptor_label",
      tip: "Receptor gene on the receiver cell. Badge marks the evidence source (DEG/prG)." },
    { key: "EM",            label: "EM",       labelKey: "EM_label",
      tip: "Effector molecule — intracellular signaling node between Receptor and Target. Badge marks the evidence source (DEG/prG)." },
    { key: "Target",        label: "Target",   labelKey: "Target_label",
      tip: "Terminal gene the path is predicted to regulate. Badge marks the evidence source (DEG/prG)." },
    { key: "contrast",      label: "contrast",
      tip: "Disease × timepoint contrast vs WT (e.g., App_4mo = APP/PS1 vs WT at 4 mo)." },
    { key: "pvalue",        label: "pvalue", numeric: true, digits: "sci",
      tip: "Wald t-test pvalue on the contrast coefficient from Incytr's factorial OLS (pvalue_method=t_test, n_perm=0 in this run). Lower = more confident change vs WT." },
    { key: "PDS",           label: "PDS",    numeric: true, digits: 3,
      tip: "Pathway Disturbance Score — composite per-path effect-size (multimodel)." },
    ...scoreCols,
    // CR-04: trajectory column — rendered only when the payload carries trajectory_index.
    ...(hasTrajIdx ? [{
      key: "_trajectory", label: "trajectory",
      tip: "Coarse temporal trajectory label for this pathway in this disease contrast. Sign vector (u/f/d at 2/4/6 mo) shown on hover.",
      isTraj: true,
    }] : []),
  ];
  // Leading expander column header is non-sortable.
  const thead =
    `<th style="width:24px;" title="Toggle detail panel (fold-change matrix + trajectory chart)."></th>`
    + cols.map(c => {
        if (c.isTraj) {
          const tip = c.tip ? ` title="${_escapeHtml(c.tip)}"` : "";
          return `<th${tip}>${_escapeHtml(c.label)}</th>`;
        }
        const on = (f.sortKey === c.key);
        const arrow = on ? (f.sortDir > 0 ? " ▲" : " ▼") : "";
        const tip = c.tip ? ` title="${_escapeHtml(c.tip)}"` : "";
        return `<th data-ip-sort="${c.key}"${tip}>${_escapeHtml(c.label)}${arrow}</th>`;
      }).join("");
  const visible = filtered.slice(startIdx, endIdx);
  const totalCols = cols.length + 1;
  const tbody = visible.map((r, idx) => {
    const rk = _ipRowKey(r);
    const isOpen = _ipRuntime.openKeys.has(rk);
    const toggle = `<td style="text-align:center;cursor:pointer;" `
      + `data-ip-toggle="${idx}" title="${isOpen ? "Hide" : "Show"} detail panel">`
      + `${isOpen ? "▾" : "▸"}</td>`;
    const cells = cols.map(c => {
      if (c.isTraj) {
        const label = r.traj_label || null;
        if (!label) return `<td class="muted">—</td>`;
        const sv = r.sign_vec || "";
        const cpal = _IP_TRAJ_COLORS[label] || { bg: "#eee", fg: "#444" };
        const tip = (_IP_TRAJ_TIPS[label] || label) + (sv ? ` Sign vector: ${sv}` : "");
        return `<td title="${_escapeHtml(tip)}">`
          + `<span style="padding:1px 7px;border-radius:10px;font-size:10px;`
          + `background:${cpal.bg};color:${cpal.fg};white-space:nowrap;">`
          + `${_escapeHtml(label)}</span></td>`;
      }
      const v = r[c.key];
      if (c.numeric) return `<td style="text-align:right;">${_ipFmtNum(v, c.digits)}</td>`;
      if (c.labelKey) return `<td>${_ipNodeCell(v, r[c.labelKey])}</td>`;
      return `<td>${_escapeHtml(v == null ? "" : v)}</td>`;
    }).join("");
    let html = `<tr data-ip-row="${idx}">${toggle}${cells}</tr>`;
    if (isOpen) {
      const activeTab = _ipRuntime.detailTab[rk] || "fc";
      html += `<tr class="ip-detail-row" data-ip-detail="${idx}"><td></td>`
        + `<td colspan="${cols.length}" style="padding:8px 12px;background:#fafafa;">`
        + _ipRenderDetailPanel(r, rk, activeTab)
        + `</td></tr>`;
    }
    return html;
  }).join("");
  // Pager (rendered above and below the table for long pages).
  const canPrev = page > 0;
  const canNext = page < nPages - 1;
  const pagerHtml =
    `<div class="ip-pager" style="display:flex;gap:8px;align-items:center;padding:6px 0;">`
    + `<button type="button" data-ip-page="first" ${canPrev ? "" : "disabled"}>&laquo; First</button>`
    + `<button type="button" data-ip-page="prev"  ${canPrev ? "" : "disabled"}>&lsaquo; Prev</button>`
    + `<span class="muted" style="margin:0 8px;">Page `
    + `<input type="number" min="1" max="${nPages}" value="${page + 1}" `
    + `data-ip-page="jump" style="width:64px;text-align:right;"/> / ${nPages.toLocaleString()}</span>`
    + `<button type="button" data-ip-page="next"  ${canNext ? "" : "disabled"}>Next &rsaquo;</button>`
    + `<button type="button" data-ip-page="last"  ${canNext ? "" : "disabled"}>Last &raquo;</button>`
    + `</div>`;
  wrap.innerHTML = pagerHtml
    + `<div class="ke-table-wrap"><table class="data-table" id="ip-table">`
    + `<thead><tr>${thead}</tr></thead><tbody>${tbody}</tbody></table></div>`
    + pagerHtml;
  wrap.querySelectorAll("[data-ip-page]").forEach(el => {
    const action = el.dataset.ipPage;
    if (action === "jump") {
      el.addEventListener("change", () => {
        const n = parseInt(el.value, 10);
        if (!isFinite(n)) return;
        const clamped = Math.max(1, Math.min(nPages, n)) - 1;
        if (clamped !== _ipRuntime.page) {
          _ipRuntime.page = clamped;
          _ipRuntime.openKeys = new Set();
          _ipRenderTable();
        }
      });
    } else {
      el.addEventListener("click", () => {
        let next = _ipRuntime.page;
        if (action === "first") next = 0;
        else if (action === "prev")  next = Math.max(0, _ipRuntime.page - 1);
        else if (action === "next")  next = Math.min(nPages - 1, _ipRuntime.page + 1);
        else if (action === "last")  next = nPages - 1;
        if (next !== _ipRuntime.page) {
          _ipRuntime.page = next;
          _ipRuntime.openKeys = new Set();
          _ipRenderTable();
        }
      });
    }
  });
  const head = wrap.querySelector("#ip-table thead");
  if (head) head.addEventListener("click", ev => {
    const th = ev.target.closest("th[data-ip-sort]");
    if (!th) return;
    const k = th.dataset.ipSort;
    if (f.sortKey === k) IncytrFilter.set({ sortDir: -1 * f.sortDir });
    else IncytrFilter.set({ sortKey: k, sortDir: (k === "pvalue" ? 1 : -1) });
    _ipResetPage();
    _ipRenderTable();
  });
  const body = wrap.querySelector("#ip-table tbody");
  if (body) body.addEventListener("click", ev => {
    // Row expander toggle.
    const cell = ev.target.closest("td[data-ip-toggle]");
    if (cell) {
      const idx = +cell.dataset.ipToggle;
      const r = visible[idx];
      if (!r) return;
      const rk = _ipRowKey(r);
      if (_ipRuntime.openKeys.has(rk)) {
        _ipRuntime.openKeys.delete(rk);
        delete _ipRuntime.detailTab[rk];
      } else {
        _ipRuntime.openKeys.add(rk);
        if (!_ipRuntime.detailTab[rk]) _ipRuntime.detailTab[rk] = "fc";
      }
      _ipRenderTable();
      return;
    }
    // Detail sub-tab switcher.
    const tabBtn = ev.target.closest("button[data-ip-detail-tab]");
    if (tabBtn) {
      const rk = tabBtn.dataset.ipDetailRk;
      const tab = tabBtn.dataset.ipDetailTab;
      _ipRuntime.detailTab[rk] = tab;
      // Re-render only the detail panel in-place to avoid full re-render.
      const detailIdx = +tabBtn.closest("tr[data-ip-detail]").dataset.ipDetail;
      const r = visible[detailIdx];
      if (!r) { _ipRenderTable(); return; }
      const tdEl = tabBtn.closest("td");
      if (tdEl) tdEl.innerHTML = _ipRenderDetailPanel(r, rk, tab);
      // Re-wire Plotly if we switched to trajectory.
      if (tab === "trajectory") _ipRenderTrajChart(rk, r);
      if (tab === "trace") _ipRenderTranscriptTrace(rk, r);
      return;
    }
  });
  // After render, draw Plotly charts for any open trajectory panels, and
  // populate any open transcript-trace panels.
  for (const rk of _ipRuntime.openKeys) {
    const tab = _ipRuntime.detailTab[rk] || "fc";
    if (tab === "trajectory") {
      const idx = visible.findIndex(r => _ipRowKey(r) === rk);
      if (idx >= 0) _ipRenderTrajChart(rk, visible[idx]);
    } else if (tab === "trace") {
      const idx = visible.findIndex(r => _ipRowKey(r) === rk);
      if (idx >= 0) _ipRenderTranscriptTrace(rk, visible[idx]);
    }
  }
}

// ---------------------------------------------------------------------------
// Detail panel: two sub-tabs — "FC matrix" and "Trajectory" (CR-04).
// ---------------------------------------------------------------------------

function _ipRenderDetailPanel(r, rk, activeTab) {
  const hasTrajIdx = _ipHasTraj();
  const hasTT = (typeof TranscriptTraceStore !== "undefined")
    && TranscriptTraceStore.isAvailable();
  const btn = (tab, label) =>
    `<button type="button" data-ip-detail-tab="${tab}" data-ip-detail-rk="${_escapeHtml(rk)}"
       style="padding:2px 12px;border-radius:4px;font-size:12px;cursor:pointer;
              border:1px solid #c0c0c0;
              background:${activeTab === tab ? "#1f4ea3" : "#f4f4f4"};
              color:${activeTab === tab ? "#fff" : "#444"};"
     >${label}</button>`;
  const tabBar = `<div style="display:flex;gap:6px;margin-bottom:8px;">`
    + btn("fc", "Fold-change")
    + (hasTrajIdx ? btn("trajectory", "Trajectory") : "")
    + (hasTT ? btn("trace", "Measurement trace") : "")
    + `</div>`;
  if (activeTab === "trajectory" && hasTrajIdx) {
    const chartId = `ip-traj-${rk.replace(/[^a-zA-Z0-9]/g, "_")}`;
    return tabBar
      + `<div id="${_escapeHtml(chartId)}" style="width:100%;min-height:220px;"></div>`;
  }
  if (activeTab === "trace" && hasTT) {
    return tabBar + _ipRenderTranscriptTraceHost(r, rk);
  }
  return tabBar + _ipRenderFcMatrix(r);
}

// Render the empty host for the transcript-trace panel; populated
// asynchronously by _ipRenderTranscriptTrace once the cluster shards load.
function _ipRenderTranscriptTraceHost(r, rk) {
  const hostId = `ip-tt-${rk.replace(/[^a-zA-Z0-9]/g, "_")}`;
  const headerNote = `Transcript pseudobulk · 1 library per arm · males-only`;
  return `<div class="audit-measurement-trace tt-panel-host" id="${_escapeHtml(hostId)}">`
       + `<div class="tt-panel-note muted" style="font-size:11px;margin-bottom:6px;">`
       + _escapeHtml(headerNote) + `</div>`
       + `<div class="tt-panel-loading muted">Loading transcript trace…</div>`
       + `</div>`;
}

// Populate the 4 element-panels (L, R, EM, T) for a row's transcript trace.
// Sender cluster routes L/EM; receiver cluster routes R/T.
async function _ipRenderTranscriptTrace(rk, r) {
  const hostId = `ip-tt-${rk.replace(/[^a-zA-Z0-9]/g, "_")}`;
  const host = document.getElementById(hostId);
  if (!host) return;
  if (typeof TranscriptTraceStore === "undefined"
      || !TranscriptTraceStore.isAvailable()) {
    host.innerHTML = `<div class="muted">Transcript trace not available in this build.</div>`;
    return;
  }
  const contrast = r.contrast;
  const sender = r._sender, receiver = r._receiver;
  // 4 element panels in path order.
  const elements = [
    { node: "Ligand",   gene: r.Ligand,   cluster: sender,   side: "sender" },
    { node: "Receptor", gene: r.Receptor, cluster: receiver, side: "receiver" },
    { node: "EM",       gene: r.EM,       cluster: sender,   side: "sender" },
    { node: "Target",   gene: r.Target,   cluster: receiver, side: "receiver" },
  ];
  const arms = TranscriptTraceStore.contrastToArms(contrast);
  const note = `Transcript pseudobulk · 1 library per arm · males-only`
    + (arms ? ` · ${arms[0].arm} vs ${arms[1].arm} @ ${contrast.split("_")[1] || ""}` : "");
  // Build placeholder grid first so it's responsive even before async fetches.
  const cellHtml = elements.map((el, i) => {
    const slotId = `${hostId}-${i}`;
    const headerSub = `${el.node} · ${el.cluster || "—"}`;
    return `<div class="tt-cell" id="${_escapeHtml(slotId)}">`
         + `<div class="tt-cell-head muted" style="font-size:10px;">${_escapeHtml(headerSub)}</div>`
         + `<div class="tt-cell-body muted">…</div>`
         + `</div>`;
  }).join("");
  host.innerHTML =
    `<div class="tt-panel-note muted" style="font-size:11px;margin-bottom:6px;">`
    + _escapeHtml(note) + `</div>`
    + `<div class="tt-grid">${cellHtml}</div>`;

  await Promise.all(elements.map(async (el, i) => {
    const slot = document.getElementById(`${hostId}-${i}`);
    if (!slot) return;
    const body = slot.querySelector(".tt-cell-body");
    if (!el.cluster) {
      if (body) body.innerHTML = `<div class="muted">no cluster on row</div>`;
      return;
    }
    if (!TranscriptTraceStore.hasCluster(el.cluster)) {
      if (body) body.innerHTML =
        `<div class="muted">no transcript trace for this cluster in this build.</div>`;
      return;
    }
    if (!el.gene) {
      if (body) body.innerHTML = `<div class="muted">no gene on this slot</div>`;
      return;
    }
    try {
      const armVals = await TranscriptTraceStore.values(
        el.cluster, el.gene, contrast);
      if (!armVals) {
        body.innerHTML = `<div class="muted">no contrast mapping for ${_escapeHtml(contrast)}</div>`;
        return;
      }
      TranscriptTraceStore.renderTwoBarPanel(body, el.gene, armVals);
    } catch (err) {
      body.innerHTML = `<div class="muted">load error: ${_escapeHtml(err.message || err)}</div>`;
    }
  }));
}

function _ipRenderFcMatrix(r) {
  // 4×N matrix of per-node fold-changes (single-cell + 3 omics layers × 2
  // metrics). Cells where the layer has no value render as "—".
  const nodes   = _ipFcNodes();
  const metrics = _ipFcMetrics();
  const head = `<tr><th></th>${metrics.map(m => {
    const tip = _IP_FC_METRIC_TIPS[m]
      ? ` title="${_escapeHtml(_IP_FC_METRIC_TIPS[m])}"` : "";
    return `<th style="text-align:right;font-weight:500;"${tip}>`
      + `${_escapeHtml(_IP_FC_METRIC_LABELS[m] || m)}</th>`;
  }).join("")}</tr>`;
  const rows = nodes.map(n => {
    const cells = metrics.map(m => {
      const v = r[`${n}_${m}`];
      const color = (v == null || !isFinite(v) || v === 0) ? ""
        : (v > 0 ? "color:#a3203c;" : "color:#1f4ea3;");
      return `<td style="text-align:right;${color}">${_ipFmtNum(v, 3)}</td>`;
    }).join("");
    const rowTip = _IP_FC_NODE_TIPS[n]
      ? ` title="${_escapeHtml(_IP_FC_NODE_TIPS[n])}"` : "";
    return `<tr><th style="text-align:left;font-weight:500;"${rowTip}>${_escapeHtml(n)}</th>${cells}</tr>`;
  }).join("");
  return `<div class="muted" style="margin-bottom:4px;font-size:11px;">`
    + `Per-node log₂ fold-changes: single-cell (sc), proteomics (pr), `
    + `phosphoserine (ps), phosphotyrosine (py). `
    + `Red = up in disease, blue = down.</div>`
    + `<table class="data-table" style="font-size:12px;">`
    + `<thead>${head}</thead><tbody>${rows}</tbody></table>`;
}

// Render the temporal PDS bar chart for a path in the trajectory sub-tab.
// Reuses the grouped-bar pattern from kinase_audit.js _renderMeaTrajectory.
// x = 2/4/6 mo timepoints, grouped by App / Tau / ApTt disease contrasts.
// Bar color: red (positive PDS), blue (negative), grey (flat/missing).
function _ipRenderTrajChart(rk, r) {
  const chartId = `ip-traj-${rk.replace(/[^a-zA-Z0-9]/g, "_")}`;
  const host = document.getElementById(chartId);
  if (!host) return;
  if (typeof Plotly === "undefined") {
    host.innerHTML = `<div class="muted">Plotly not available for trajectory chart.</div>`;
    return;
  }
  const pathStr = _ipPathStr(r);
  const entries = _ipTrajEntriesForPath(pathStr, _ipRuntime.rows);

  // Load PDS values directly from the loaded rows (all contrasts present in
  // the shard — one row per disease×timepoint). Keyed by contrast string.
  const pdsByContrast = new Map();
  const pvalByContrast = new Map();
  for (const row of (_ipRuntime.rows || [])) {
    if (_ipPathStr(row) === pathStr) {
      const k = row.contrast || "";
      pdsByContrast.set(k, row.PDS != null ? Number(row.PDS) : null);
      pvalByContrast.set(k, row.pvalue != null ? Number(row.pvalue) : null);
    }
  }

  const timepoints = ["2mo", "4mo", "6mo"];
  const diseaseOrder = ["App", "Tau", "ApTt"];
  const diseaseColors = { App: "#c8261c", Tau: "#1f5fa6", ApTt: "#5c2d91" };

  // Flat threshold from spec: |PDS| < 0.01 AND pvalue >= 0.05.
  const FLAT_PDS = 0.01, FLAT_P = 0.05;
  const isFlat = (pds, pv) =>
    (pds == null || Math.abs(pds) < FLAT_PDS) &&
    (pv  == null || pv >= FLAT_P);

  const traces = diseaseOrder.map(dis => {
    const xs = [], ys = [], colors = [], hovers = [];
    for (const tp of timepoints) {
      const contrast = `${dis}_${tp}`;
      const pds = pdsByContrast.get(contrast);
      const pv  = pvalByContrast.get(contrast);
      const flat = isFlat(pds, pv);
      const y = pds != null ? pds : 0;
      const base = diseaseColors[dis] || "#888";
      const color = flat ? "rgba(160,160,160,0.35)"
        : (y >= 0 ? `${base}` : `${base}`);
      // Slightly desaturate negative bars by blending toward blue.
      const fillColor = flat ? "rgba(160,160,160,0.35)"
        : (y >= 0 ? _ipHexAlpha(base, 0.85) : _ipHexAlpha(base, 0.55));
      xs.push(tp);
      ys.push(pds != null ? pds : 0);
      colors.push(fillColor);
      const pvStr = pv != null ? pv.toExponential(2) : "—";
      hovers.push(`${dis} ${tp}<br>PDS ${y.toFixed(3)}<br>p ${pvStr}${flat ? " (flat)" : ""}`);
    }
    return {
      type: "bar",
      name: dis,
      x: xs, y: ys,
      marker: { color: colors, line: { color: "rgba(0,0,0,0.15)", width: 0.5 } },
      hovertemplate: "%{customdata}<extra></extra>",
      customdata: hovers,
    };
  });

  const [curDis] = (r.contrast || "").split("_");
  const traj = entries.find(e => e.contrast === curDis) || entries[0];
  const titleText = traj
    ? `Trajectory: <b>${traj.traj_label}</b> (sign vec ${traj.sign_vec || "—"}) · ${r.Path || ""}`
    : `PDS over time · ${r.Path || ""}`;

  Plotly.react(chartId, traces, {
    barmode: "group",
    margin: { l: 48, r: 12, t: 32, b: 48 },
    height: 240,
    title: { text: titleText, font: { size: 12 } },
    yaxis: { zeroline: true, zerolinecolor: "#bbb", title: "PDS" },
    xaxis: { title: "timepoint" },
    showlegend: true,
    legend: { orientation: "h", y: -0.25 },
  }, { displaylogo: false, responsive: true });
}

// Hex color to rgba with alpha (for the trajectory chart color generation).
function _ipHexAlpha(hex, alpha) {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex || "");
  if (!m) return hex;
  return `rgba(${parseInt(m[1],16)},${parseInt(m[2],16)},${parseInt(m[3],16)},${alpha})`;
}

function wireIncytrPathways() {
  // Numeric sliders.
  const wireSlider = (id, key) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener("input", () => {
      const raw = el.value === "" ? null : parseFloat(el.value);
      IncytrFilter.set({ [key]: (raw != null && isFinite(raw)) ? raw : null });
      _ipResetPage();
      _ipRenderTable();
    });
  };
  wireSlider("ip-slider-p",   "sliderP");
  wireSlider("ip-slider-pds", "sliderPds");

  // Reset.
  const resetBtn = document.getElementById("ip-reset");
  if (resetBtn) resetBtn.addEventListener("click", () => {
    IncytrFilter.reset();
    const block = _ipBlock();
    if (block) _ipSyncControls(block);
    _ipRenderTrajChips();
    _ipInvalidateScope();
    _ipEnsureShards();
  });
}

function renderIncytrPathways() {
  const block = _ipBlock();
  const countEl = document.getElementById("ip-count");
  if (!block) {
    if (countEl) countEl.textContent = "No incytr_pathways block in payload.";
    return;
  }
  _ipSyncControls(block);
  _ipEnsureShards();
}
