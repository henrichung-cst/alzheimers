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

// Trajectory labels — fixed at build time, derived per (path, disease) from
// raw PDS at 2/4/6 mo. Non-exclusive: a single (path, disease) tuple can
// carry multiple labels (e.g. always-up AND monotonic-up). Shard rows carry
// a semicolon-joined string in `traj_labels`. Incomplete paths (any
// timepoint missing in the Incytr output) get an empty string and render
// "—" in the table.
const _IP_TRAJ_LABELS = [
  "always-up", "always-down", "monotonic-up", "monotonic-down", "mixed",
];
const _IP_TRAJ_COLORS = {
  "always-up":      { bg: "#ffe0e0", fg: "#a3203c", border: "#e8a0a0" },
  "always-down":    { bg: "#dde8f8", fg: "#1f4ea3", border: "#a0bee8" },
  "monotonic-up":   { bg: "#fff0d0", fg: "#9a5000", border: "#e8c080" },
  "monotonic-down": { bg: "#e0f0ff", fg: "#005090", border: "#80c0e0" },
  "mixed":          { bg: "#f8f0e0", fg: "#705020", border: "#d0b080" },
};
const _IP_TRAJ_TIPS = {
  "always-up":      "PDS > 0 at all three timepoints (uuu).",
  "always-down":    "PDS < 0 at all three timepoints (ddd).",
  "monotonic-up":   "PDS strictly increasing: PDS(2mo) < PDS(4mo) < PDS(6mo).",
  "monotonic-down": "PDS strictly decreasing: PDS(2mo) > PDS(4mo) > PDS(6mo).",
  "mixed":          "Sign of PDS changes across timepoints (e.g. udu, ddu).",
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
  openKeys:       new Set(),    // keys of rows whose Evidence detail is expanded
  detailTab:      {},           // rk → "evidence" | "trajectory" (which sub-tab is active)
  recurFallback:  null,         // { sig, map } cache for fallback recurPathMap
  pathIndex:      null,         // Map<pathStr, rows[]> — built at shard-load
  pathLabels:     null,         // Map<pathStr, Map<disease, Set<label>>> — for chip filter
  trajCounts:     null,         // Map<disease, Map<label, count>> — built on demand
  page:           0,            // 0-indexed current page (post-filter, post-sort)
};

// ---------------------------------------------------------------------------
// Trajectory helpers (build-time labels, per-disease multi-label)
// ---------------------------------------------------------------------------
// Shard rows carry `traj_labels` (semicolon-joined string) and `sign_vec`.
// A precomputed `pathLabels: Map<pathStr, Map<disease, Set<label>>>` is
// built once at shard-load and reused for filter, chip counts, and chart.

function _ipRecurIndex() {
  const block = _ipBlock();
  return (block && block.recur_index) || null;
}

function _ipPathStr(r) {
  return r._pathStr || (`${r._sender}||${r._receiver}||${r.Path}`);
}

// Decode the semicolon-joined traj_labels string into an array. Empty
// strings → []. Returns the raw array; callers wrap in Set when needed.
function _ipDecodeLabels(s) {
  if (!s) return [];
  return s.split(";");
}

// Return distinct (disease, traj_labels, sign_vec) tuples for a path. Uses
// the precomputed pathIndex (O(1) lookup).
function _ipTrajEntriesForPath(pathStr) {
  const idx = _ipRuntime.pathIndex;
  const rows = idx ? idx.get(pathStr) : null;
  if (!rows) return [];
  const seen = new Set();
  const out = [];
  for (const r of rows) {
    if (!r.traj_labels) continue;
    const c = r.contrast || "";
    const ui = c.indexOf("_");
    const dis = ui < 0 ? c : c.substring(0, ui);
    if (seen.has(dis)) continue;
    seen.add(dis);
    out.push({
      contrast: dis,
      traj_labels: _ipDecodeLabels(r.traj_labels),
      sign_vec: r.sign_vec || "",
    });
  }
  return out;
}

// Trajectory is build-time; v3+ payload carries multi-label traj_labels.
function _ipHasTraj() {
  const block = _ipBlock();
  return !!(block && block.version >= 3);
}

// Per-(disease, label) counts over the loaded shards — for chip UI.
// Returns Map<disease, Map<label, count>>.
function _ipTrajCounts() {
  if (_ipRuntime.trajCounts) return _ipRuntime.trajCounts;
  const counts = new Map();
  if (!_ipRuntime.rows) return counts;
  const seen = new Set();
  for (const r of _ipRuntime.rows) {
    if (!r.traj_labels) continue;
    const c = r.contrast || "";
    const ui = c.indexOf("_");
    const dis = ui < 0 ? c : c.substring(0, ui);
    const k = _ipPathStr(r) + "|" + dis;
    if (seen.has(k)) continue;
    seen.add(k);
    let byLbl = counts.get(dis);
    if (!byLbl) { byLbl = new Map(); counts.set(dis, byLbl); }
    for (const lbl of _ipDecodeLabels(r.traj_labels)) {
      byLbl.set(lbl, (byLbl.get(lbl) || 0) + 1);
    }
  }
  _ipRuntime.trajCounts = counts;
  return counts;
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
  // The table renders exactly one shard at a time.
  //
  // Resolution order:
  //   1. f.pair (pinned from heatmap click) — always wins
  //   2. Intersection of senderIn × receiverIn × present_pairs — if exactly
  //      one pair, use it
  //   3. Otherwise: return all matches so the renderer can prompt the user
  //      to narrow (caller distinguishes length 0 / 1 / >1).
  const f = IncytrFilter.get();
  if (f.pair) return [f.pair];
  const sIn = new Set(f.senderIn || []);
  const rIn = new Set(f.receiverIn || []);
  const matches = [];
  for (const [s, r] of block.slice_index.present) {
    if (sIn.size && !sIn.has(s)) continue;
    if (rIn.size && !rIn.has(r)) continue;
    matches.push({ sender: s, receiver: r });
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
      if (key === "senderIn" || key === "receiverIn") {
        patch.pair = null;
      }
      IncytrFilter.set(patch);
      _ipMountMultiselect(hostId, label, options, key);   // re-render badge
      _ipInvalidateScope();
      _ipResetPage();
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
  const searchEl = document.getElementById("ip-search");
  if (searchEl) searchEl.value = f.searchText || "";

  // CR-04: trajectory chips.
  _ipRenderTrajChips();
}

// Render trajectory chips — one row per disease (App / Tau / ApTt), each
// row offering the 5 labels. OR within a row, AND across rows. Counts
// reflect the loaded shards.
function _ipRenderTrajChips() {
  const host = document.getElementById("ip-traj-chips");
  if (!host) return;
  if (!_ipHasTraj()) {
    host.style.display = "none";
    return;
  }
  host.style.display = "flex";
  host.style.flexDirection = "column";
  host.style.gap = "4px";
  const trajLabels = IncytrFilter.get("trajLabels") || {};
  const counts = _ipTrajCounts();
  const rowHtml = (disease) => {
    const sel = new Set(trajLabels[disease] || []);
    const byLbl = counts.get(disease) || new Map();
    const chips = _IP_TRAJ_LABELS.map(label => {
      const on = sel.has(label);
      const c = _IP_TRAJ_COLORS[label] || { bg: "#eee", fg: "#444", border: "#bbb" };
      const tip = _IP_TRAJ_TIPS[label] || label;
      const n = byLbl.get(label) || 0;
      const check = on ? "✓ " : "";
      return `<button type="button" data-ip-traj-dis="${disease}" data-ip-traj-lbl="${_escapeHtml(label)}"
        title="${_escapeHtml(disease + ": " + tip)}"
        style="padding:3px 10px;border-radius:14px;font-size:11px;cursor:pointer;
               border:${on ? "2px" : "1px"} solid ${on ? c.fg : c.border};
               background:${on ? c.fg : c.bg};
               color:${on ? "#fff" : c.fg};
               font-weight:${on ? "700" : "500"};
               box-shadow:${on ? "0 0 0 2px " + c.bg : "none"};"
      >${check}${_escapeHtml(label)} <span style="opacity:0.75;font-weight:400;">(${n.toLocaleString()})</span></button>`;
    }).join("");
    return `<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
        <span style="min-width:42px;font-size:11px;font-weight:600;color:#444;">${disease}:</span>
        ${chips}
      </div>`;
  };
  host.innerHTML = _IP_DISEASES.map(rowHtml).join("");
  host.onclick = ev => {
    const btn = ev.target.closest("button[data-ip-traj-dis]");
    if (!btn) return;
    const dis = btn.dataset.ipTrajDis;
    const lbl = btn.dataset.ipTrajLbl;
    const cur = Object.assign({}, IncytrFilter.get("trajLabels") || {});
    const set = new Set(cur[dis] || []);
    if (set.has(lbl)) set.delete(lbl);
    else set.add(lbl);
    cur[dis] = [...set];
    IncytrFilter.set({ trajLabels: cur });
    _ipResetPage();
    _ipRenderTrajChips();
    _ipRenderTable();
  };
}

function _ipResetPage() {
  _ipRuntime.page = 0;
}

// Debounce helper. Returns a wrapped function that fires `fn` only after
// `delay` ms have passed without another call. Used for the search box and
// numeric sliders so each keystroke/drag doesn't re-filter the full row set.
function _ipDebounce(fn, delay) {
  let h = null;
  return function() {
    if (h) clearTimeout(h);
    h = setTimeout(() => { h = null; fn(); }, delay);
  };
}
const _ipRenderTableDebounced = _ipDebounce(() => _ipRenderTable(), 180);

function _ipInvalidateScope() {
  _ipRuntime.rows = null;
  _ipRuntime.loadedKey = null;
  _ipRuntime.openKeys = new Set();
  _ipRuntime.recurFallback = null;
  _ipRuntime.pathIndex = null;
  _ipRuntime.pathLabels = null;
  _ipRuntime.trajCounts = null;
  _ipRuntime._didDebugLog = false;
}

// ---- shard loading ----

async function _ipEnsureShards() {
  const block = _ipBlock();
  if (!block) return;
  const pairs = _ipPairsInScope(block);
  if (pairs.length !== 1) {
    // 0 pairs: prompt to widen. >1 pairs: prompt to narrow (per-shard
    // pagination means the table is always single-pair). Either way, no
    // shard load.
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
    // Stamp sender/receiver and reconstruct Path (dropped from the shard
    // — it's just L|R|E|T concatenated) so the table column renders. Also
    // precompute _pathStr and _hay so hot filter loops don't allocate.
    const sPipe = p.sender + "||" + p.receiver + "||";
    for (const r of rows) {
      r._sender = p.sender;
      r._receiver = p.receiver;
      if (r.Path == null)
        r.Path = (r.Ligand || "") + "|" + (r.Receptor || "") + "|"
               + (r.EM || "") + "|" + (r.Target || "");
      r._pathStr = sPipe + r.Path;
      r._hay = (
        p.sender + "\n" + p.receiver + "\n" +
        (r.Ligand || "") + "\n" + (r.Receptor || "") + "\n" +
        (r.EM || "") + "\n" + (r.Target || "") + "\n" +
        r.Path + "\n" + (r.contrast || "")
      ).toLowerCase();
    }
    // Resolve race: only commit if the scope hasn't changed mid-fetch.
    const newSig = _ipScopeSig(_ipPairsInScope(block));
    if (newSig !== sig) return;
    _ipRuntime.rows = rows;
    _ipRuntime.loadedKey = sig;
    // Build two indexes once so per-render filtering is O(1) lookups:
    //   pathIndex  : pathStr → rows[]   (chart, debug)
    //   pathLabels : pathStr → Map<disease, Set<label>>  (chip filter)
    const pathIndex  = new Map();
    const pathLabels = new Map();
    for (const r of rows) {
      const pid = _ipPathStr(r);
      let bucket = pathIndex.get(pid);
      if (!bucket) { bucket = []; pathIndex.set(pid, bucket); }
      bucket.push(r);
      if (r.traj_labels) {
        const c = r.contrast || "";
        const ui = c.indexOf("_");
        const dis = ui < 0 ? c : c.substring(0, ui);
        let byDis = pathLabels.get(pid);
        if (!byDis) { byDis = new Map(); pathLabels.set(pid, byDis); }
        if (!byDis.has(dis)) {
          byDis.set(dis, new Set(_ipDecodeLabels(r.traj_labels)));
        }
      }
    }
    _ipRuntime.pathIndex  = pathIndex;
    _ipRuntime.pathLabels = pathLabels;
    _ipRuntime.trajCounts = null;
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
  const recurSet      = new Set(f.recurContrasts || []);
  const hasRecur      = recurSet.size > 0;
  // Per-disease trajectory chip selection: { App: [...], Tau: [...], ApTt: [...] }.
  // Active when at least one disease has a non-empty array. OR within disease,
  // AND across diseases.
  const trajByDis = f.trajLabels || {};
  const trajGates = [];   // [{ disease, labels: Set<string> }]
  for (const d of _IP_DISEASES) {
    const arr = trajByDis[d] || [];
    if (arr.length) trajGates.push({ disease: d, labels: new Set(arr) });
  }
  const hasTraj  = trajGates.length > 0 && _ipHasTraj();
  const pathLbl  = hasTraj ? _ipRuntime.pathLabels : null;
  const recurIdx = hasRecur ? _ipRecurIndex() : null;
  const searchTokens  = (f.searchText || "")
    .toLowerCase().split(/\s+/).filter(Boolean);

  // For recur filtering: if recur_index is absent, build a fast pathStr →
  // disease-set map from the already-loaded rows (applying the active
  // pvalue/|PDS| gates) so we don't have to re-scan for every row. Cache it
  // by (loadedKey, sliderP, sliderPds) — invalidated on shard reload.
  let recurPathMap = null;
  if (hasRecur && !recurIdx) {
    const sig = `${_ipRuntime.loadedKey}|${f.sliderP}|${f.sliderPds}`;
    const cached = _ipRuntime.recurFallback;
    if (cached && cached.sig === sig) {
      recurPathMap = cached.map;
    } else {
      recurPathMap = new Map();
      const sP = f.sliderP, sPds = f.sliderPds;
      for (const r of _ipRuntime.rows) {
        if (sP   != null && !(r.pvalue != null && r.pvalue < sP)) continue;
        if (sPds != null && !(Math.abs(r.PDS || 0) >= sPds))      continue;
        const c = r.contrast || "";
        const ui = c.indexOf("_");
        const d = ui < 0 ? c : c.substring(0, ui);
        const pid = _ipPathStr(r);
        let s = recurPathMap.get(pid);
        if (!s) { s = new Set(); recurPathMap.set(pid, s); }
        s.add(d);
      }
      _ipRuntime.recurFallback = { sig, map: recurPathMap };
    }
  }

  const sP = f.sliderP, sPds = f.sliderPds;
  const hasSearch = searchTokens.length > 0;
  const hasDis = diseaseSet.size > 0, hasTime = timeSet.size > 0;
  const out = [];
  for (const r of _ipRuntime.rows) {
    if (hasDis || hasTime) {
      const c = r.contrast || "";
      const ui = c.indexOf("_");
      const d = ui < 0 ? c : c.substring(0, ui);
      const t = ui < 0 ? "" : c.substring(ui + 1);
      if (hasDis  && !diseaseSet.has(d)) continue;
      if (hasTime && !timeSet.has(t))    continue;
    }
    if (sP   != null && !(r.pvalue < sP))          continue;
    if (sPds != null && !(Math.abs(r.PDS || 0) >= sPds)) continue;

    if (hasSearch) {
      const hay = r._hay || "";
      let ok = true;
      for (const t of searchTokens) { if (hay.indexOf(t) < 0) { ok = false; break; } }
      if (!ok) continue;
    }

    if (hasTraj) {
      const byDis = pathLbl ? pathLbl.get(_ipPathStr(r)) : null;
      if (!byDis) continue;
      let passes = true;
      // AND within disease (path must satisfy every selected label) and
      // AND across diseases. Picking incompatible labels (always-up +
      // always-down) is meant to yield zero results.
      for (const gate of trajGates) {
        const lbls = byDis.get(gate.disease);
        if (!lbls) { passes = false; break; }
        for (const lbl of gate.labels) {
          if (!lbls.has(lbl)) { passes = false; break; }
        }
        if (!passes) break;
      }
      if (!passes) continue;
    }
    // One-time diagnostic per filter call: log gate setup + pathLabels stats.
    if (hasTraj && !_ipRuntime._didDebugLog) {
      _ipRuntime._didDebugLog = true;
      const sample = [];
      let n = 0;
      for (const [pid, byDis] of pathLbl || []) {
        if (n++ >= 3) break;
        const dump = {};
        for (const [d, s] of byDis) dump[d] = [...s];
        sample.push({ pid, labels: dump });
      }
      console.log("[ip-filter] trajGates=",
        trajGates.map(g => ({ disease: g.disease, labels: [...g.labels] })));
      console.log("[ip-filter] pathLabels size=", pathLbl ? pathLbl.size : 0,
        " sample=", sample);
    }

    if (hasRecur) {
      const pid = _ipPathStr(r);
      const diseases = recurIdx
        ? recurIdx[pid]
        : (recurPathMap ? recurPathMap.get(pid) : null);
      if (!diseases) continue;
      // diseases is an Array (recur_index) or a Set (fallback map).
      const isArr = Array.isArray(diseases);
      let passes = true;
      for (const d of recurSet) {
        const has = isArr ? (diseases.indexOf(d) >= 0) : diseases.has(d);
        if (!has) { passes = false; break; }
      }
      if (!passes) continue;
    }

    out.push(r);
  }
  const key = f.sortKey, dir = f.sortDir;
  const numericKeys = new Set([
    "pvalue", "PDS", ..._ipScoreCols(),
  ]);
  const isNumeric = numericKeys.has(key);
  const cmp = isNumeric
    ? (a, b) => {
        const av = a[key], bv = b[key];
        if (av == null && bv == null) return 0;
        if (av == null) return 1;
        if (bv == null) return -1;
        return dir * (av - bv);
      }
    : (a, b) => {
        const av = a[key], bv = b[key];
        if (av == null && bv == null) return 0;
        if (av == null) return 1;
        if (bv == null) return -1;
        if (av < bv) return -dir;
        if (av > bv) return  dir;
        return 0;
      };
  // Pagination slices in _ipRenderTable; sort the full filtered set so paging
  // through is stable.
  out.sort(cmp);
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
  const f = IncytrFilter.get();
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
      `<li style="margin:2px 0;"><button type="button" class="ip-pair-pick" `
      + `data-ip-sender="${_escapeHtml(p.sender)}" `
      + `data-ip-receiver="${_escapeHtml(p.receiver)}" `
      + `style="padding:2px 8px;font-size:12px;cursor:pointer;">`
      + `${_escapeHtml(p.sender)} → ${_escapeHtml(p.receiver)}</button></li>`
    ).join("");
    wrap.innerHTML = '<div style="padding:16px;">'
      + '<div class="muted" style="margin-bottom:8px;">'
      + 'Select a single (sender, receiver) pair to view its pathways:'
      + '</div><ul style="list-style:none;padding:0;margin:0;max-height:400px;overflow:auto;">'
      + items + '</ul></div>';
    wrap.querySelectorAll(".ip-pair-pick").forEach(btn => {
      btn.addEventListener("click", () => {
        IncytrFilter.set({
          senderIn: [btn.dataset.ipSender],
          receiverIn: [btn.dataset.ipReceiver],
        });
        _ipInvalidateScope();
        _ipResetPage();
        const b = _ipBlock();
        if (b) _ipSyncControls(b);
        _ipEnsureShards();
      });
    });
    return;
  }
  if (_ipRuntime.loading) {
    countEl.textContent = `Loading ${pairs[0].sender} → ${pairs[0].receiver}…`;
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
  // Pagination math — clamp page in case filter shrinks the row set.
  const nPages = Math.max(1, Math.ceil(filtered.length / _IP_PAGE_SIZE));
  if (_ipRuntime.page >= nPages) _ipRuntime.page = nPages - 1;
  if (_ipRuntime.page < 0) _ipRuntime.page = 0;
  const page = _ipRuntime.page;
  const startIdx = page * _IP_PAGE_SIZE;
  const endIdx = Math.min(filtered.length, startIdx + _IP_PAGE_SIZE);
  const p0 = pairs[0];
  countEl.textContent =
    `${p0.sender} → ${p0.receiver}: `
    + `${filtered.length.toLocaleString()} rows pass filters `
    + `(of ${total.toLocaleString()} in shard). `
    + (filtered.length
        ? `Page ${page + 1} / ${nPages.toLocaleString()} `
          + `(rows ${(startIdx + 1).toLocaleString()}–${endIdx.toLocaleString()}).`
        : "");

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
      tip: "Trajectory labels for this (pathway, disease): any combination of always-up (uuu), always-down (ddd), monotonic-up (PDS strictly increasing), monotonic-down (strictly decreasing), or mixed (sign changes). Only populated when all 3 timepoints have rows in the Incytr output. Hover for sign vector.",
      isTraj: true,
    }] : []),
  ];
  // Leading expander column header is non-sortable.
  const thead =
    `<th style="width:24px;" title="Toggle detail panel (Evidence: 4 nodes × 4 layers, + Trajectory chart)."></th>`
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
        const labels = _ipDecodeLabels(r.traj_labels);
        if (!labels.length) return `<td class="muted">—</td>`;
        const sv = r.sign_vec || "";
        const tipParts = labels.map(l => _IP_TRAJ_TIPS[l] || l);
        const tip = tipParts.join(" • ") + (sv ? ` · Sign vector: ${sv}` : "");
        const badges = labels.map(label => {
          const cpal = _IP_TRAJ_COLORS[label] || { bg: "#eee", fg: "#444" };
          return `<span style="padding:1px 6px;margin-right:2px;border-radius:10px;`
            + `font-size:10px;background:${cpal.bg};color:${cpal.fg};white-space:nowrap;">`
            + `${_escapeHtml(label)}</span>`;
        }).join("");
        return `<td title="${_escapeHtml(tip)}">${badges}</td>`;
      }
      const v = r[c.key];
      if (c.numeric) return `<td style="text-align:right;">${_ipFmtNum(v, c.digits)}</td>`;
      if (c.labelKey) return `<td>${_ipNodeCell(v, r[c.labelKey])}</td>`;
      return `<td>${_escapeHtml(v == null ? "" : v)}</td>`;
    }).join("");
    let html = `<tr data-ip-row="${idx}">${toggle}${cells}</tr>`;
    if (isOpen) {
      const activeTab = _ipRuntime.detailTab[rk] || "evidence";
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
    `<div class="ip-pager" style="display:flex;gap:8px;align-items:center;padding:6px 0;font-size:12px;">`
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
        if (el.disabled) return;
        let next = _ipRuntime.page;
        if (action === "first") next = 0;
        else if (action === "prev") next = Math.max(0, _ipRuntime.page - 1);
        else if (action === "next") next = Math.min(nPages - 1, _ipRuntime.page + 1);
        else if (action === "last") next = nPages - 1;
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
        if (!_ipRuntime.detailTab[rk]) _ipRuntime.detailTab[rk] = "evidence";
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
      if (tab === "evidence") _ipRenderEvidencePanel(rk, r);
      return;
    }
  });
  // After render, draw Plotly charts for open trajectory panels, and
  // populate open Evidence panels.
  for (const rk of _ipRuntime.openKeys) {
    const tab = _ipRuntime.detailTab[rk] || "evidence";
    if (tab === "trajectory") {
      const idx = visible.findIndex(r => _ipRowKey(r) === rk);
      if (idx >= 0) _ipRenderTrajChart(rk, visible[idx]);
    } else if (tab === "evidence") {
      const idx = visible.findIndex(r => _ipRowKey(r) === rk);
      if (idx >= 0) _ipRenderEvidencePanel(rk, visible[idx]);
    }
  }
}

// ---------------------------------------------------------------------------
// Detail panel: "Evidence" tab (default) + "Trajectory" sub-tab (when
// trajectory_index is present in the payload).
//
// The old "Fold-change" tab (_ipRenderFcMatrix) and "Measurement trace" tab
// (_ipRenderTranscriptTrace) were removed in Item 3.3 and replaced by the
// Evidence tab, which renders all 4 nodes × 4 layers from the per-cluster
// omics + transcript shards. Cluster routing per evaluation.R:227-230 is
// enforced in EvidencePanel.render().
// ---------------------------------------------------------------------------

function _ipRenderDetailPanel(r, rk, activeTab) {
  const hasTrajIdx = _ipHasTraj();
  const btn = (tab, label) =>
    `<button type="button" data-ip-detail-tab="${tab}" data-ip-detail-rk="${_escapeHtml(rk)}"
       style="padding:2px 12px;border-radius:4px;font-size:12px;cursor:pointer;
              border:1px solid #c0c0c0;
              background:${activeTab === tab ? "#1f4ea3" : "#f4f4f4"};
              color:${activeTab === tab ? "#fff" : "#444"};"
     >${label}</button>`;
  const tabBar = `<div style="display:flex;gap:6px;margin-bottom:8px;">`
    + btn("evidence", "Evidence")
    + (hasTrajIdx ? btn("trajectory", "Trajectory") : "")
    + `</div>`;
  if (activeTab === "trajectory" && hasTrajIdx) {
    const chartId = `ip-traj-${rk.replace(/[^a-zA-Z0-9]/g, "_")}`;
    return tabBar
      + `<div id="${_escapeHtml(chartId)}" style="width:100%;min-height:220px;"></div>`;
  }
  // Default: Evidence tab.
  const hostId = `ip-ev-${rk.replace(/[^a-zA-Z0-9]/g, "_")}`;
  return tabBar
    + `<div class="ev-panel-host" id="${_escapeHtml(hostId)}">`
    + `<div class="ev-col-loading muted" style="font-size:11px;">Loading evidence…</div>`
    + `</div>`;
}

// Populate the Evidence panel for a row. Called after DOM is updated.
// Routes all 4 nodes through EvidencePanel.render() (cluster routing is
// internal to EvidencePanel per evaluation.R:227-230).
function _ipRenderEvidencePanel(rk, r) {
  const hostId = `ip-ev-${rk.replace(/[^a-zA-Z0-9]/g, "_")}`;
  const host = document.getElementById(hostId);
  if (!host) return;
  if (typeof EvidencePanel === "undefined") {
    host.innerHTML = `<div class="muted">EvidencePanel widget not loaded.</div>`;
    return;
  }
  EvidencePanel.render(host, r, rk).catch(err => {
    if (host) host.innerHTML =
      `<div class="muted">Evidence load error: ${_escapeHtml(err.message || err)}</div>`;
  });
}

// Render the temporal PDS bar chart for a path in the trajectory sub-tab.
// Reuses the grouped-bar pattern from kinase_audit.js _renderMeaTrajectory.
// x = 2/4/6 mo timepoints, grouped by App / Tau / ApTt disease contrasts.
// Bar color: red (positive PDS), blue (negative), grey (missing).
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

  // No flat threshold — color by raw PDS sign. Missing rows are grey.
  const traces = diseaseOrder.map(dis => {
    const xs = [], ys = [], colors = [], hovers = [];
    for (const tp of timepoints) {
      const contrast = `${dis}_${tp}`;
      const pds = pdsByContrast.get(contrast);
      const pv  = pvalByContrast.get(contrast);
      const missing = (pds == null);
      const y = missing ? 0 : pds;
      const base = diseaseColors[dis] || "#888";
      const fillColor = missing ? "rgba(160,160,160,0.35)"
        : (y >= 0 ? _ipHexAlpha(base, 0.85) : _ipHexAlpha(base, 0.55));
      xs.push(tp);
      ys.push(y);
      colors.push(fillColor);
      const pvStr = pv != null ? pv.toExponential(2) : "—";
      hovers.push(`${dis} ${tp}<br>PDS ${y.toFixed(3)}<br>p ${pvStr}${missing ? " (missing)" : ""}`);
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

  // Show all three diseases' labels in the title — trajectory is per-disease,
  // and the chart plots all three groups, so the title must reflect each.
  const titleParts = entries.map(e => {
    const lbls = (e.traj_labels && e.traj_labels.length)
      ? e.traj_labels.join("+") : "—";
    return `<b>${e.contrast}</b>: ${lbls} (${e.sign_vec || "—"})`;
  });
  const titleText = titleParts.length
    ? `${titleParts.join(" · ")} — ${r.Path || ""}`
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
  // Numeric sliders. State updates synchronously (so a Reset/sync read sees
  // the latest values); the heavy re-render is debounced so dragging a slider
  // doesn't refilter ~700K rows per `input` event.
  const wireSlider = (id, key) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener("input", () => {
      const raw = el.value === "" ? null : parseFloat(el.value);
      IncytrFilter.set({ [key]: (raw != null && isFinite(raw)) ? raw : null });
      // Slider change invalidates the cached recur-fallback map (it depends
      // on the active pvalue/|PDS| gates).
      _ipRuntime.recurFallback = null;
      _ipResetPage();
      _ipRenderTableDebounced();
    });
  };
  wireSlider("ip-slider-p",   "sliderP");
  wireSlider("ip-slider-pds", "sliderPds");

  // Search box — substring AND across Path/nodes/sender/receiver/contrast.
  // Debounced for the same reason as sliders.
  const searchEl = document.getElementById("ip-search");
  if (searchEl) searchEl.addEventListener("input", () => {
    IncytrFilter.set({ searchText: searchEl.value || "" });
    _ipResetPage();
    _ipRenderTableDebounced();
  });

  // Reset.
  const resetBtn = document.getElementById("ip-reset");
  if (resetBtn) resetBtn.addEventListener("click", () => {
    IncytrFilter.reset();
    const block = _ipBlock();
    if (block) _ipSyncControls(block);
    _ipRenderTrajChips();
    _ipInvalidateScope();
    _ipResetPage();
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
