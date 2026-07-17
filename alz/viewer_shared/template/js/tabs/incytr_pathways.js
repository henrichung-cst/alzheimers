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

// Context payloads can supply block.diseases / block.timepoints. Otherwise
// fall back to the context contrast axis, then to parsing block.contrasts.
const _IP_DISEASES_FALLBACK = ["App", "Tau", "ApTt"];
const _IP_TIMEPOINTS_FALLBACK = ["2mo", "4mo", "6mo"];

function _ipAxisParts() {
  const block = _ipBlock();
  const axis = ViewerPayload.contrastAxis();
  const groups = (block && block.diseases && block.diseases.length)
    ? block.diseases
    : (axis.groups || []);
  const timepoints = (block && block.timepoints && block.timepoints.length)
    ? block.timepoints
    : (axis.timepoints || []);
  if ((groups && groups.length) || (timepoints && timepoints.length) || !block) {
    return { groups: groups || [], timepoints: timepoints || [] };
  }
  const g = [], t = [];
  const seenG = new Set(), seenT = new Set();
  for (const c of block.contrasts || []) {
    const i = String(c).indexOf("_");
    const a = i < 0 ? String(c) : String(c).slice(0, i);
    const b = i < 0 ? "" : String(c).slice(i + 1);
    if (!seenG.has(a)) { seenG.add(a); g.push(a); }
    if (b && !seenT.has(b)) { seenT.add(b); t.push(b); }
  }
  return { groups: g, timepoints: t };
}

function _ipDiseases() {
  const v = _ipAxisParts().groups;
  return v.length ? v : _IP_DISEASES_FALLBACK;
}
function _ipTimepoints() {
  const v = _ipAxisParts().timepoints;
  return v.length ? v : _IP_TIMEPOINTS_FALLBACK;
}
// Per-shard pagination: the table renders at most one (sender, receiver)
// shard at a time, paginated client-side. Page size bounds DOM render cost;
// the loaded shard itself bounds JS heap residency. Previous "load all
// shards and concat" architecture peaked at ~10M JS objects and OOM'd the
// browser process.
const _IP_PAGE_SIZE = 100;

// Trajectory labels — fixed at build time, derived per (path, disease) from
// raw PDS across the active context's ordered timepoints. Non-exclusive: a single (path, group) tuple can
// carry multiple labels (e.g. always-up AND monotonic-up). Shard rows carry
// a semicolon-joined string in `traj_labels`. Incomplete paths (any
// timepoint missing in the Incytr output) get an empty string and render
// "—" in the table.
const _IP_TRAJ_COLORS = {
  "always-up":      { bg: "#ffe0e0", fg: "#a3203c", border: "#e8a0a0" },
  "always-down":    { bg: "#dde8f8", fg: "#1f4ea3", border: "#a0bee8" },
  "monotonic-up":   { bg: "#fff0d0", fg: "#9a5000", border: "#e8c080" },
  "monotonic-down": { bg: "#e0f0ff", fg: "#005090", border: "#80c0e0" },
  "mixed":          { bg: "#f8f0e0", fg: "#705020", border: "#d0b080" },
};
const _IP_TRAJ_TIPS = {
  "always-up":      "PDS > 0 at every available timepoint.",
  "always-down":    "PDS < 0 at every available timepoint.",
  "monotonic-up":   "PDS strictly increasing across the ordered timepoints.",
  "monotonic-down": "PDS strictly decreasing across the ordered timepoints.",
  "mixed":          "Sign of PDS changes across timepoints.",
};
const _IP_TRAJ_ALL_GROUPS = "__all__";
const _IP_TRAJ_METRIC_COLORS = {
  "PDS": "#263238",
  "TPDS": "#c8261c",
  "PPDS": "#1f7a3a",
  "PhPDS_ps": "#7c3aed",
  "PhPDS_py": "#d97706",
  "SiK_score": "#0f766e",
  "Ack_score": "#b5179e",   // acetylation (5xFAD PTM track)
  "KGG_score": "#4361ee",   // ubiquitination (5xFAD PTM track)
  "Rme1_score": "#3a5a40",  // methylation (reserved; no assay data today)
};
const _IP_TRAJ_GROUP_COLORS = {
  "App": "#c8261c",
  "Tau": "#1f5fa6",
  "ApTt": "#5c2d91",
  "TG": "#b45309",
};
// Score columns are advertised on the payload block but kept in a module-local
// fallback so the JS stays usable against an older payload.
const _IP_SCORE_COLS_FALLBACK = ["TPDS", "PPDS", "PhPDS_ps", "PhPDS_py", "SiK_score"];

// Per-node evidence-source labels. Each of Ligand/Receptor/EM/Target carries a
// {DEG, prG, KsG} tag indicating which seed list admitted the gene:
//   DEG → single-cell differentially-expressed gene (transcript evidence)
//   prG → proteomics-significant gene (bulk-protein evidence)
//   KsG → kinase-substrate gene (admitted via kinase→cell-type attribution)
// Rendered inline next to the gene name as a small colored badge.
const _IP_LABEL_NODES_FALLBACK = ["Ligand", "Receptor", "EM", "Target"];
const _IP_LABEL_COLORS = {
  "DEG": { bg: "#e8eefc", fg: "#1f4ea3" },   // blue — single-cell
  "prG": { bg: "#e7f4ec", fg: "#1f7a3a" },   // green — proteomics
  "KsG": { bg: "#fdeede", fg: "#9a5413" },   // amber — kinase-substrate
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
  "Ack_score":        "Acetylation PDS — aggregated from factorial OLS β on deconvoluted acetyl-lysine (AcK) site intensities. 5xFAD-only (surfaced only where the acetylation assay was run).",
  "KGG_score":        "Ubiquitination PDS — aggregated from factorial OLS β on deconvoluted di-glycyl-lysine (KGG) site intensities. 5xFAD-only (surfaced only where the ubiquitination assay was run).",
  "Rme1_score":       "Methylation PDS — reserved channel; no methylation assay feeds it today, so it is gated out of the surfaced columns.",
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
  detailTab:      {},           // rk → active row-detail sub-tab
  trajSettings:   {},           // rk → { group: "<axis group>"|"__all__", metrics: [...], view: "both"|"line"|"bar" }
  trajRows:       {},           // rk → currently loaded sibling rows for drawer controls
  trajPromises:   {},           // rk → in-flight sibling-row load promises
  recurFallback:  null,         // { sig, map } cache for fallback recurPathMap
  pathIndex:      null,         // Map<pathStr, rows[]> — built at shard-load
  pathLabels:     null,         // Map<pathStr, Map<disease, Set<label>>> — for trend filter
  geneIndexBlock: null,
  geneIndexMap:   null,         // upper-case gene symbol -> [gene ids]
  geneIndex:      null,         // { url, data, error, promise } — lazy gene_node_index sidecar (audit P5)
  page:           0,            // 0-indexed current page (post-filter, post-sort)
  indexLoading:   false,        // top-mode global filter-index fetch in flight
  indexError:     null,         // top-mode global filter-index load error
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
  const sender = r._sender || r.sender || "";
  const receiver = r._receiver || r.receiver || "";
  const path = r.Path || [r.Ligand, r.Receptor, r.EM, r.Target].join("|");
  return r._pathStr || (`${sender}||${receiver}||${path}`);
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

function _ipScoreTrajectoryAxis() {
  const groups = _ipDiseases();
  const timepoints = _ipTimepoints();
  if (timepoints.length >= 2) {
    return {
      xKind: "timepoint",
      xLabel: "timepoint",
      xValues: timepoints,
      seriesKind: "group",
      seriesLabel: "Group",
      seriesValues: groups,
    };
  }
  if (groups.length >= 2) {
    return {
      xKind: "group",
      xLabel: "condition",
      xValues: groups,
      seriesKind: "timepoint",
      seriesLabel: "Baseline",
      seriesValues: timepoints.length ? timepoints : [""],
    };
  }
  return {
    xKind: "timepoint",
    xLabel: "timepoint",
    xValues: timepoints,
    seriesKind: "group",
    seriesLabel: "Group",
    seriesValues: groups,
  };
}

function _ipHasScoreTrajectory() {
  return _ipScoreTrajectoryAxis().xValues.length >= 2;
}

function _ipContrastParts(contrast) {
  const c = String(contrast || "");
  const groups = _ipDiseases();
  const timepoints = _ipTimepoints();
  for (const g of groups) {
    for (const t of timepoints) {
      if (c === `${g}_${t}`) return { group: g, timepoint: t };
    }
  }
  const i = c.indexOf("_");
  return {
    group: i < 0 ? c : c.substring(0, i),
    timepoint: i < 0 ? "" : c.substring(i + 1),
  };
}

// Friendly column/series labels for the score columns. PTM tracks (5xFAD only)
// render as "AcK"/"KGG" rather than the raw "Ack_score"/"KGG_score".
const _IP_SCORE_SHORT = {
  "SiK_score": "SiK",
  "Ack_score": "AcK",
  "KGG_score": "KGG",
  "Rme1_score": "Rme1",
};
function _ipMetricLabel(metric) {
  return _IP_SCORE_SHORT[metric] || metric;
}

function _ipScoreCols() {
  const block = _ipBlock();
  return (block && block.score_columns) || _IP_SCORE_COLS_FALLBACK;
}
function _ipScoreMinAbs() {
  const f = IncytrFilter.get();
  const raw = (f.scoreMinAbs && typeof f.scoreMinAbs === "object") ? f.scoreMinAbs : {};
  const allowed = new Set(_ipScoreCols());
  const out = {};
  for (const [key, value] of Object.entries(raw)) {
    if (!allowed.has(key)) continue;
    if (value == null || value === "") continue;
    const n = Number(value);
    if (isFinite(n) && n >= 0) out[key] = n;
  }
  return out;
}
function _ipScoreGatesPass(row, gates) {
  const entries = Object.entries(gates || {});
  if (!entries.length) return true;
  for (const [key, minAbs] of entries) {
    const v = Number(row[key]);
    if (!(isFinite(v) && Math.abs(v) >= minAbs)) return false;
  }
  return true;
}
function _ipPdsSignPass(pds, sign) {
  if (sign === "up") return Number(pds) > 0;
  if (sign === "down") return Number(pds) < 0;
  return true;
}
function _ipRowKey(r) {
  return `${r._sender || r.sender || ""}||${r._receiver || r.receiver || ""}||${r.Path || ""}||${r.contrast}`;
}

function _ipNormalizeTopRow(r) {
  if (!r) return r;
  if (r._sender == null) r._sender = r.sender || "";
  if (r._receiver == null) r._receiver = r.receiver || "";
  if (r.Path == null)
    r.Path = (r.Ligand || "") + "|" + (r.Receptor || "") + "|"
           + (r.EM || "") + "|" + (r.Target || "");
  if (r._pathStr == null) r._pathStr = `${r._sender}||${r._receiver}||${r.Path}`;
  return r;
}

// J-3: Grain-aware block accessor.
// "Full" → returns the parent block unchanged (today's behavior).
// Backbone grain → returns a shallow-merged block:
//   - Parent block supplies all metadata (contrasts, diseases, timepoints,
//     senders, receivers, label_nodes, label_states, traj_label_vocab,
//     recur_index, low_signal_celltypes, …).
//   - Grain block overrides: global_index, heatmap_counts,
//     heatmap_counts_signed, score_columns; slice_index if grain is sharded.
//   - Adds: _grain, _grainMode, nodes, node_id_columns.
// Falls back to parent when the grain is not present in the payload
// (pair hasn't run backbone scoring yet → graceful no-op).
// window._ipGrainBlock is registered here so IncytrGlobalIndex._block()
// picks up the same grain-merged block for its own index lookups.
function _ipActiveBlock() {
  const parent = ViewerPayload.incytr ? ViewerPayload.incytr() : null;
  if (!parent) return parent;
  const grain = (window.IncytrFilter && IncytrFilter.get("grain")) || "Full";
  if (grain === "Full") return parent;
  const g = (parent.backbone_grains || {})[grain];
  if (!g) return parent;   // grain absent → graceful fallback
  const nodeIdCols = (g.global_index && g.global_index.node_id_columns) || [];
  return Object.assign({}, parent, {
    global_index:          g.global_index,
    heatmap_counts:        g.heatmap_counts,
    heatmap_counts_signed: g.heatmap_counts_signed,
    score_columns:         g.score_columns,
    nodes:                 g.nodes || [],
    node_id_columns:       nodeIdCols,
    slice_index:           g.slice_index || null,   // null for inline grains
    _grain:                grain,
    _grainMode:            g.mode,   // "inline" | "sharded"
  });
}
// Register so IncytrGlobalIndex._block() uses the grain overlay.
window._ipGrainBlock = _ipActiveBlock;

function _ipBlock() {
  return _ipActiveBlock();
}

// True when the active grain is a backbone grain (not Full).
function _ipIsBackboneGrain() {
  const block = _ipBlock();
  return !!(block && block._grain && block._grain !== "Full");
}

// Nodes that are dropped in the active grain (used to show "—").
// Returns a Set<"Ligand"|"Receptor"|"EM"|"Target">.
function _ipDroppedNodes() {
  const block = _ipBlock();
  if (!block || !block._grain || block._grain === "Full") return new Set();
  const allNodes = ["Ligand", "Receptor", "EM", "Target"];
  const activeNodes = new Set(block.nodes || allNodes);
  return new Set(allNodes.filter(n => !activeNodes.has(n)));
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
  if (f.pair) {
    return IncytrCelltypeQc.pairExcluded(f.pair.sender, f.pair.receiver, block)
      ? [] : [f.pair];
  }
  const sIn = new Set(f.senderIn || []);
  const rIn = new Set(f.receiverIn || []);
  const matches = [];
  // Inline backbone grains (R-EM, L-R-EM) carry no slice_index — their rows come
  // from the global-index scan, not per-pair shards. The present (sender,receiver)
  // pairs are the same as the parent (Full) block, so fall back to the parent's
  // present rather than reporting zero pairs (which would strand the table on the
  // "pick a pair" placeholder for every backbone grain).
  let si = block.slice_index || {};
  if (!block.slice_index && block._grain && block._grain !== "Full") {
    const parent = (window.ViewerPayload && ViewerPayload.incytr) ? ViewerPayload.incytr() : null;
    si = (parent && parent.slice_index) || {};
  }
  const present = si.present || [];
  for (const [s, r] of (present || [])) {
    if (IncytrCelltypeQc.pairExcluded(s, r, block)) continue;
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
      // J-3: timepointCombine selection toggles the all/any mode wrapper
      // visibility — call _ipSyncControls so it updates without a full render.
      if (key === "timepointCombine") {
        const b = _ipBlock();
        if (b) _ipSyncControls(b);
      }
      _ipInvalidateScope();
      _ipResetPage();
      _ipEnsureShards();
    },
  });
}

function _ipRenderScoreFilterControls() {
  const host = document.getElementById("ip-score-filters");
  if (!host) return;
  const gates = _ipScoreMinAbs();
  const cols = _ipScoreCols();
  host.innerHTML = cols.map(col => {
    const label = _ipMetricLabel(col);
    const v = gates[col];
    return `<label class="ke-filter-label" title="Keep rows with |${_escapeHtml(col)}| greater than or equal to this value. Blank = no constraint.">${_escapeHtml(label)} |≥`
      + `<input data-ip-score-filter="${_escapeHtml(col)}" type="number" step="0.05" min="0" `
      + `value="${v == null ? "" : _escapeHtml(v)}" style="width:58px;"/>`
      + `</label>`;
  }).join("");
}

function _ipSyncControls(block) {
  const f = IncytrFilter.get();

  _ipMountMultiselect("ip-ms-sender",   "Sender",    block.senders,   "senderIn");
  _ipMountMultiselect("ip-ms-receiver", "Receiver",  block.receivers, "receiverIn");
  _ipMountMultiselect("ip-ms-disease",  "Disease",   _ipDiseases(),    "disease");
  _ipMountMultiselect("ip-ms-time",     "Timepoint", _ipTimepoints(),  "timepoint");

  // CR-04: "Recur in" multiselect — AND gate across selected disease contrasts.
  _ipMountMultiselect("ip-ms-recur", "Recur in", _ipDiseases(), "recurContrasts");

  // J-3: Grain selector — show only when backbone_grains exist in the parent block.
  const parentBlock = ViewerPayload.incytr ? ViewerPayload.incytr() : null;
  const hasGrains = !!(parentBlock && parentBlock.backbone_grains
    && Object.keys(parentBlock.backbone_grains).length);
  const grainWrap = document.getElementById("ip-grain-wrap");
  if (grainWrap) grainWrap.hidden = !hasGrains;
  const grainSel = document.getElementById("ip-grain");
  if (grainSel) grainSel.value = f.grain || "Full";

  // J-3: Timepoint-combination filter multiselect + all/any toggle.
  const tps = _ipTimepoints();
  const hasTps = tps.length >= 2;
  _ipMountMultiselect("ip-ms-tp-combine", "Cover tps", hasTps ? tps : [], "timepointCombine");
  const tpCombineEl = document.getElementById("ip-ms-tp-combine");
  if (tpCombineEl) tpCombineEl.style.display = hasTps ? "" : "none";
  const tpCombineActive = hasTps && (f.timepointCombine || []).length > 0;
  const tpCombineModeWrap = document.getElementById("ip-tp-combine-mode-wrap");
  if (tpCombineModeWrap) tpCombineModeWrap.hidden = !tpCombineActive;
  const tpCombineModeSel = document.getElementById("ip-tp-combine-mode");
  if (tpCombineModeSel) tpCombineModeSel.value = f.timepointCombineMode || "all";

  // Numeric sliders.
  const set = (id, v) => {
    const el = document.getElementById(id);
    if (el) el.value = (v == null || !isFinite(v)) ? "" : v;
  };
  set("ip-slider-p",   f.sliderP);
  set("ip-slider-pds", f.sliderPds);
  const signSel = document.getElementById("ip-pds-sign");
  if (signSel) signSel.value = (f.pdsSign === "up" || f.pdsSign === "down") ? f.pdsSign : "both";
  _ipRenderScoreFilterControls();
  const searchEl = document.getElementById("ip-search");
  if (searchEl) searchEl.value = f.searchText || "";
  const trendSel = document.getElementById("ip-trend");
  if (trendSel) {
    trendSel.value = TrendFilter.normalize(f.trend || "");
    if (trendSel.parentElement) trendSel.parentElement.style.display = _ipHasTraj() ? "" : "none";
  }
  const lowSel = document.getElementById("if-low-signal");
  if (lowSel) {
    const hasLow = IncytrCelltypeQc.hasLowSignal(block);
    lowSel.value = (f.excludeLowSignalCelltypes && hasLow) ? "exclude" : "include";
    if (lowSel.parentElement) lowSel.parentElement.style.display = hasLow ? "" : "none";
  }
  const mode = f.ipMode || "top";
  const modeSel = document.getElementById("ip-mode");
  if (modeSel) modeSel.value = mode;
  const limitSel = document.getElementById("ip-top-limit");
  if (limitSel) {
    const topLimit = [500, 1000, 5000].includes(Number(f.topLimit))
      ? Number(f.topLimit) : 500;
    limitSel.value = String(topLimit);
    if (limitSel.parentElement) limitSel.parentElement.style.display = (mode === "top") ? "" : "none";
  }
  // Sender/Receiver stay visible in both modes — filterRank honors senderIn/
  // receiverIn in Top mode too, so hiding them there left an active filter with
  // no control to discover or clear it (e.g. a stale persisted Microglia sender
  // silently emptying Top overall). Recur-in is gated only in the pair-mode row
  // filter (filterRank has no recur predicate), so it stays Cell-Type-only.
  for (const id of ["ip-ms-sender", "ip-ms-receiver"]) {
    const el = document.getElementById(id);
    if (el) el.style.display = "";
  }
  const recurEl = document.getElementById("ip-ms-recur");
  if (recurEl) recurEl.style.display = (mode === "top") ? "none" : "";
}

function _ipSameArray(a, b) {
  const aa = Array.isArray(a) ? a : [];
  const bb = Array.isArray(b) ? b : [];
  return aa.length === bb.length && aa.every((v, i) => v === bb[i]);
}

function _ipKeepValid(values, options) {
  const allowed = new Set(options || []);
  return (Array.isArray(values) ? values : []).filter(v => allowed.has(v));
}

function _ipPairPresent(block, pair) {
  if (!block || !pair) return false;
  const present = ((block.slice_index || {}).present || block.present_pairs || []);
  return present.some(p => p && p[0] === pair.sender && p[1] === pair.receiver);
}

function _ipSanitizeFilterState(block) {
  if (!block) return false;
  const f = IncytrFilter.get();
  const patch = {};
  const senders = block.senders || [];
  const receivers = block.receivers || [];
  const disease = _ipKeepValid(f.disease, _ipDiseases());
  const timepoint = _ipKeepValid(f.timepoint, _ipTimepoints());
  const senderIn = _ipKeepValid(f.senderIn, senders);
  const receiverIn = _ipKeepValid(f.receiverIn, receivers);
  const recurContrasts = _ipKeepValid(f.recurContrasts, _ipDiseases());
  if (!_ipSameArray(f.disease, disease)) patch.disease = disease;
  if (!_ipSameArray(f.timepoint, timepoint)) patch.timepoint = timepoint;
  if (!_ipSameArray(f.senderIn, senderIn)) patch.senderIn = senderIn;
  if (!_ipSameArray(f.receiverIn, receiverIn)) patch.receiverIn = receiverIn;
  if (!_ipSameArray(f.recurContrasts, recurContrasts)) patch.recurContrasts = recurContrasts;
  if (!_ipHasTraj() && f.trend) patch.trend = "";
  if (!["top", "pair"].includes(f.ipMode || "top")) patch.ipMode = "top";
  if (![500, 1000, 5000].includes(Number(f.topLimit))) patch.topLimit = 500;
  if (!["both", "up", "down"].includes(f.pdsSign || "both")) patch.pdsSign = "both";
  const cleanScores = _ipScoreMinAbs();
  const currentScores = (f.scoreMinAbs && typeof f.scoreMinAbs === "object") ? f.scoreMinAbs : {};
  if (JSON.stringify(cleanScores) !== JSON.stringify(currentScores)) patch.scoreMinAbs = cleanScores;
  if (f.pair && (!_ipPairPresent(block, f.pair)
      || IncytrCelltypeQc.pairExcluded(f.pair.sender, f.pair.receiver, block))) {
    patch.pair = null;
  }

  // J-3: Validate grain against available grains in the parent block.
  const parentBlock = ViewerPayload.incytr ? ViewerPayload.incytr() : null;
  const availableGrainSet = new Set(["Full",
    ...Object.keys((parentBlock && parentBlock.backbone_grains) || {})]);
  const grain = availableGrainSet.has(f.grain || "Full") ? (f.grain || "Full") : "Full";
  if (grain !== (f.grain || "Full")) patch.grain = grain;

  // J-3: Validate timepointCombine and timepointCombineMode.
  const timepointCombine = _ipKeepValid(f.timepointCombine, _ipTimepoints());
  if (!_ipSameArray(f.timepointCombine, timepointCombine)) patch.timepointCombine = timepointCombine;
  if (!["all", "any"].includes(f.timepointCombineMode || "all")) patch.timepointCombineMode = "all";

  const keys = Object.keys(patch);
  if (!keys.length) return false;
  IncytrFilter.set(patch);
  return true;
}

function _ipActiveFilterSummary(f) {
  const parts = [];
  if ((f.disease || []).length) parts.push(`Disease: ${(f.disease || []).join(", ")}`);
  if ((f.timepoint || []).length) parts.push(`Timepoint: ${(f.timepoint || []).join(", ")}`);
  if ((f.senderIn || []).length) parts.push(`Sender: ${(f.senderIn || []).join(", ")}`);
  if ((f.receiverIn || []).length) parts.push(`Receiver: ${(f.receiverIn || []).join(", ")}`);
  if (f.sliderP != null) parts.push(`pvalue < ${f.sliderP}`);
  if (f.sliderPds != null) parts.push(`|PDS| >= ${f.sliderPds}`);
  if (f.pdsSign === "up") parts.push("PDS > 0");
  if (f.pdsSign === "down") parts.push("PDS < 0");
  for (const [key, value] of Object.entries(_ipScoreMinAbs())) {
    parts.push(`|${key}| >= ${value}`);
  }
  if (f.searchText) parts.push(`search: ${f.searchText}`);
  if (f.trend) parts.push(`Trend: ${TrendFilter.label(f.trend)}`);
  if ((f.recurContrasts || []).length) parts.push(`Recur in: ${(f.recurContrasts || []).join(", ")}`);
  return parts.join("; ");
}

function _ipGeneSearchTerms(text) {
  return [...new Set(String(text || "")
    .split(/[\s,;|>*]+/)
    .map(s => s.trim())
    .filter(Boolean)
    .filter(s => /^[A-Za-z0-9_.-]+$/.test(s))
    .map(s => s.toUpperCase()))];
}

// Audit P5: gene_node_index is a ~15 MB sidecar fetched on demand (pair-mode gene
// search only), not inlined in the payload. _ipResolveGeneIndex returns the loaded
// dict (or null until _ipEnsureGeneIndex resolves); the renderer gates on it.
function _ipResolveGeneIndex(block) {
  const url = block && block.gene_node_index_shard;
  if (!url) return null;
  const gi = _ipRuntime.geneIndex;
  return (gi && gi.url === url && gi.data) ? gi.data : null;
}

function _ipEnsureGeneIndex(block) {
  const url = block && block.gene_node_index_shard;
  if (!url) return Promise.resolve(null);
  let gi = _ipRuntime.geneIndex;
  if (gi && gi.url === url) {
    if (gi.data) return Promise.resolve(gi.data);
    if (gi.promise) return gi.promise;
  }
  gi = _ipRuntime.geneIndex = { url, data: null, error: null, promise: null };
  // Transport-tolerant gunzip (mirrors 01_state.js:_decodeGzipBuffer): sniff the
  // gzip magic (0x1f 0x8b) and only decompress when the bytes are actually
  // gzip. Some hosting layers (bioplat voila-gateway) auto-decompress and hand
  // back plain bytes; a stale cache entry stored while the object carried
  // Content-Encoding: gzip lands here too.
  gi.promise = (async () => {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`gene index fetch ${url} -> ${resp.status}`);
    const raw = await resp.arrayBuffer();
    const magic = new Uint8Array(raw);
    let text;
    if (magic.length >= 2 && magic[0] === 0x1f && magic[1] === 0x8b) {
      const stream = new Response(raw).body.pipeThrough(new DecompressionStream("gzip"));
      text = await new Response(stream).text();
    } else {
      text = new TextDecoder("utf-8").decode(raw);
    }
    gi.data = JSON.parse(text);
    return gi.data;
  })();
  gi.promise.catch(e => { gi.error = String(e && e.message ? e.message : e); });
  return gi.promise;
}

function _ipGeneIndexMap(block) {
  const idx = _ipResolveGeneIndex(block);
  if (!idx || !Array.isArray(idx.genes) || !Array.isArray(idx.gene_id))
    return null;
  if (_ipRuntime.geneIndexBlock === block && _ipRuntime.geneIndexMap)
    return _ipRuntime.geneIndexMap;
  const map = new Map();
  idx.genes.forEach((g, id) => {
    const key = String(g || "").toUpperCase();
    if (!key) return;
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(id);
  });
  _ipRuntime.geneIndexBlock = block;
  _ipRuntime.geneIndexMap = map;
  return map;
}

function _ipGeneIndexMatches(block, pairs) {
  const idx = _ipResolveGeneIndex(block);
  const terms = _ipGeneSearchTerms(IncytrFilter.get("searchText"));
  if (!idx || !terms.length) return null;
  if (Object.keys(_ipScoreMinAbs()).length) return null;
  const map = _ipGeneIndexMap(block);
  if (!map) return null;
  const targetGeneIds = new Set();
  const canonicalById = {};
  for (const term of terms) {
    for (const gid of (map.get(term) || [])) {
      targetGeneIds.add(gid);
      canonicalById[gid] = idx.genes[gid];
    }
  }
  const f = IncytrFilter.get();
  const pairSet = new Set((pairs || []).map(p => `${p.sender}||${p.receiver}`));
  const senderSet = new Set(f.senderIn || []);
  const receiverSet = new Set(f.receiverIn || []);
  const out = [];
  const n = (idx.gene_id || []).length;
  for (let i = 0; i < n; i++) {
    const gid = idx.gene_id[i];
    if (!targetGeneIds.has(gid)) continue;
    const sender = idx.senders[idx.sender_id[i]];
    const receiver = idx.receivers[idx.receiver_id[i]];
    if (pairSet.size && !pairSet.has(`${sender}||${receiver}`)) continue;
    if (senderSet.size && !senderSet.has(sender)) continue;
    if (receiverSet.size && !receiverSet.has(receiver)) continue;
    if (IncytrCelltypeQc.pairExcluded(sender, receiver, block)) continue;
    const bestAbs = idx.best_abs_pds[i];
    const bestP = idx.best_pvalue[i];
    const bestPds = idx.best_pds[i];
    if (f.sliderPds != null && !(bestAbs != null && Number(bestAbs) >= f.sliderPds))
      continue;
    if (f.sliderP != null && !(bestP != null && Number(bestP) < f.sliderP))
      continue;
    if (!_ipPdsSignPass(bestPds, f.pdsSign || "both")) continue;
    out.push({
      gene: canonicalById[gid] || idx.genes[gid],
      role: idx.roles[idx.role_id[i]],
      sender,
      receiver,
      n_rows: idx.n_rows[i] || 0,
      best_abs_pds: bestAbs,
      best_pds: bestPds,
      best_pvalue: bestP,
    });
  }
  out.sort((a, b) => {
    const ga = String(a.gene), gb = String(b.gene);
    if (ga !== gb) return ga.localeCompare(gb);
    const aa = Number(a.best_abs_pds || 0);
    const bb = Number(b.best_abs_pds || 0);
    if (aa !== bb) return bb - aa;
    if (a.n_rows !== b.n_rows) return b.n_rows - a.n_rows;
    const sa = `${a.sender}||${a.receiver}||${a.role}`;
    const sb = `${b.sender}||${b.receiver}||${b.role}`;
    return sa.localeCompare(sb);
  });
  return { terms, matches: out };
}

function _ipRenderGeneIndexSearch(block, pairs) {
  const countEl = document.getElementById("ip-count");
  const wrap = document.getElementById("ip-table-wrap");
  if (!countEl || !wrap) return false;
  // Audit P5: the gene_node_index is fetched on demand. If the user is running a
  // gene search but the sidecar isn't mapped yet, show a loading state, kick off
  // the one-time fetch, and re-render when it resolves.
  const terms = _ipGeneSearchTerms(IncytrFilter.get("searchText"));
  const wantsGeneSearch = !!(block && block.gene_node_index_shard)
    && terms.length > 0
    && Object.keys(_ipScoreMinAbs()).length === 0;
  if (wantsGeneSearch && !_ipResolveGeneIndex(block)) {
    const gi = _ipRuntime.geneIndex;
    if (gi && gi.url === block.gene_node_index_shard && gi.error) {
      countEl.textContent = "Gene index load failed.";
      wrap.innerHTML = `<div class="muted" style="padding:16px;">${_escapeHtml(gi.error)}</div>`;
      return true;
    }
    countEl.textContent = `Loading gene index for ${terms.join(", ")}…`;
    wrap.innerHTML = '<div class="muted" style="padding:16px;">'
      + 'Fetching the cross-pair gene index (one-time download)…</div>';
    _ipEnsureGeneIndex(block).then(() => _ipRenderTable(), () => _ipRenderTable());
    return true;
  }
  const res = _ipGeneIndexMatches(block, pairs);
  if (!res) return false;
  const termsText = res.terms.join(", ");
  if (!res.matches.length) {
    countEl.textContent = `No indexed Ligand/Receptor/EM/Target matches for ${termsText}.`;
    wrap.innerHTML = '<div class="muted" style="padding:16px;">'
      + 'The cross-pair index uses exact gene-symbol matches over Ligand, Receptor, EM, and Target. '
      + 'Use Reset or loosen sender/receiver, pvalue, or |PDS| filters.'
      + '</div>';
    return true;
  }
  const limit = 500;
  const visible = res.matches.slice(0, limit);
  countEl.textContent =
    `Indexed gene search: ${res.matches.length.toLocaleString()} `
    + `gene-role-pair match${res.matches.length === 1 ? "" : "es"} for ${termsText}`
    + (res.matches.length > limit ? `; showing first ${limit.toLocaleString()}` : "")
    + `.`;
  const rows = visible.map((r, i) => {
    const pds = r.best_pds == null ? "—" : _ipFmtNum(r.best_pds, 3);
    const absPds = r.best_abs_pds == null ? "—" : _ipFmtNum(r.best_abs_pds, 3);
    const pval = r.best_pvalue == null ? "—" : _ipFmtNum(r.best_pvalue, "sci");
    return `<tr>`
      + `<td>${_escapeHtml(r.gene)}</td>`
      + `<td>${_escapeHtml(r.role)}</td>`
      + `<td>${_escapeHtml(r.sender)}</td>`
      + `<td>${_escapeHtml(r.receiver)}</td>`
      + `<td style="text-align:right;">${Number(r.n_rows || 0).toLocaleString()}</td>`
      + `<td style="text-align:right;">${absPds}</td>`
      + `<td style="text-align:right;">${pds}</td>`
      + `<td style="text-align:right;">${pval}</td>`
      + `<td><button type="button" class="ip-pair-pick" data-ip-sender="${_escapeHtml(r.sender)}" `
      + `data-ip-receiver="${_escapeHtml(r.receiver)}" data-ip-gene-hit="${i}" `
      + `style="padding:2px 8px;font-size:12px;cursor:pointer;">Load pair</button></td>`
      + `</tr>`;
  }).join("");
  wrap.innerHTML = '<div class="muted" style="padding:8px 0;">'
    + 'Exact gene-symbol matches across all currently eligible cell-type pairs. '
    + 'Load a pair to inspect full pathway rows and omics evidence.'
    + '</div>'
    + '<div class="ke-table-wrap"><table class="data-table" id="ip-gene-index-table">'
    + '<thead><tr>'
    + '<th>Gene</th><th>Node</th><th>Sender</th><th>Receiver</th>'
    + '<th>Rows</th><th>best |PDS|</th><th>best PDS</th><th>best pvalue</th><th></th>'
    + '</tr></thead><tbody>' + rows + '</tbody></table></div>';
  wrap.querySelectorAll(".ip-pair-pick").forEach(btn => {
    btn.addEventListener("click", () => {
      IncytrFilter.set({
        senderIn: [btn.dataset.ipSender],
        receiverIn: [btn.dataset.ipReceiver],
        pair: null,
      });
      _ipInvalidateScope();
      _ipResetPage();
      const b = _ipBlock();
      if (b) _ipSyncControls(b);
      _ipEnsureShards();
    });
  });
  return true;
}


function _ipRenderTopTable() {
  const countEl = document.getElementById("ip-count");
  const wrap = document.getElementById("ip-table-wrap");
  const block = _ipBlock();
  if (!wrap || !countEl || !block) return;
  if (!IncytrGlobalIndex.available()) {
    countEl.textContent = "No global Incytr pathway index is packaged in this payload.";
    wrap.innerHTML = '<div class="muted" style="padding:16px;">Switch to Cell Type mode to inspect one sender/receiver pair.</div>';
    return;
  }
  if (_ipRuntime.indexError) {
    countEl.textContent = "Global pathway index load failed.";
    wrap.innerHTML = `<div class="muted" style="padding:16px;">${_escapeHtml(_ipRuntime.indexError)}</div>`;
    return;
  }
  // The complete-universe index is fetched once on first entry to Top mode.
  // filterRank/materialize require it mapped; show a loading state until then.
  if (!IncytrGlobalIndex.loaded()) {
    countEl.textContent = "Loading the full pathway index…";
    wrap.innerHTML = '<div class="muted" style="padding:16px;">'
      + 'Fetching the complete pathway universe (one-time download)…</div>';
    if (!_ipRuntime.indexLoading) {
      _ipRuntime.indexLoading = true;
      IncytrGlobalIndex.ensureLoaded()
        .then(() => { _ipRuntime.indexLoading = false; _ipRenderTopTable(); })
        .catch(e => {
          _ipRuntime.indexLoading = false;
          _ipRuntime.indexError = String(e && e.message ? e.message : e);
          _ipRenderTopTable();
        });
    }
    return;
  }
  const gi = IncytrGlobalIndex.manifest();
  const f = IncytrFilter.get();
  const topLimit = [500, 1000, 5000].includes(Number(f.topLimit))
    ? Number(f.topLimit) : 500;
  // filter -> rank -> cap over the WHOLE universe. `total` is the true count of
  // matching pathways; `indices` is the capped, sorted row-id list (<= cap).
  const { indices, total } = IncytrGlobalIndex.filterRank(f);
  if (!total) {
    countEl.textContent =
      `Top overall: 0 of ${gi.nrows.toLocaleString()} pathways match the active filters.`;
    const summary = _ipActiveFilterSummary(f);
    wrap.innerHTML = '<div class="muted" style="padding:16px;">'
      + (summary
          ? `No rows match: ${_escapeHtml(summary)}. Use Reset or loosen the filters.`
          : 'No rows match the active filters. Use Reset or loosen the filters.')
      + '</div>';
    return;
  }
  const nPages = Math.max(1, Math.ceil(indices.length / _IP_PAGE_SIZE));
  if (_ipRuntime.page >= nPages) _ipRuntime.page = nPages - 1;
  if (_ipRuntime.page < 0) _ipRuntime.page = 0;
  const page = _ipRuntime.page;
  const startIdx = page * _IP_PAGE_SIZE;
  const endIdx = Math.min(indices.length, startIdx + _IP_PAGE_SIZE);
  // Hydrate only the visible page from the columns.
  const visible = indices.slice(startIdx, endIdx).map(i => IncytrGlobalIndex.materialize(i));
  const fmt = (v, d) => (v == null || !isFinite(Number(v))) ? "—" : Number(v).toFixed(d);
  const trajBadges = (r) => {
    const labels = _ipDecodeLabels(r.traj_labels || "");
    if (!labels.length) return `<span class="muted">—</span>`;
    return labels.map(label => {
      const c = _IP_TRAJ_COLORS[label] || { bg: "#eee", fg: "#444" };
      return `<span style="padding:1px 6px;margin-right:2px;border-radius:10px;`
        + `font-size:10px;background:${c.bg};color:${c.fg};white-space:nowrap;">`
        + `${_escapeHtml(label)}</span>`;
    }).join("");
  };
  // Cap-vs-universe is explicit: a collapse reads as "only N exist", never as a
  // silent truncation. The cap is now a render budget, not a data gate.
  const capped = total > indices.length;
  countEl.textContent =
    (capped
      ? `Top overall: top ${indices.length.toLocaleString()} of ${total.toLocaleString()} matching pathways `
        + `(raise the cap or narrow filters to see more)`
      : `Top overall: all ${total.toLocaleString()} matching pathways shown`)
    + `, ranked by ${gi.rank_by || "abs(PDS)"}. `
    + `Page ${page + 1} / ${nPages.toLocaleString()} `
    + `(rows ${(startIdx + 1).toLocaleString()}–${endIdx.toLocaleString()}).`
    + (IncytrCelltypeQc.enabled(block) ? ` ${IncytrCelltypeQc.controlText(block)}.` : "");

  const scoreCols = _ipScoreCols().map(k => ({
    key: k, label: _ipMetricLabel(k), numeric: true, digits: 3,
    tip: _IP_SCORE_TIPS[k] || `${k} score column from Incytr factorial scoring.`,
  }));
  // J-3: pvalue absent for backbone grain indexes.
  const hasPvalue = !!(block && (!block._grain || block._grain === "Full"));
  // J-3: dropped nodes in the active grain show "—" in their column.
  const droppedNodes = _ipDroppedNodes();
  const cols = [
    { key: "_sender", label: "Sender" },
    { key: "_receiver", label: "Receiver" },
    { key: "Path", label: "Path" },
    { key: "Ligand", label: "Ligand", labelKey: "Ligand_label" },
    { key: "Receptor", label: "Receptor", labelKey: "Receptor_label" },
    { key: "EM", label: "EM", labelKey: "EM_label" },
    { key: "Target", label: "Target", labelKey: "Target_label" },
    { key: "contrast", label: "contrast" },
    ...(hasPvalue ? [{ key: "pvalue", label: "pvalue", numeric: true, digits: "sci" }] : []),
    { key: "PDS", label: "PDS", numeric: true, digits: 3 },
    ...scoreCols,
    ...(_ipHasTraj() ? [{ key: "_trajectory", label: "trajectory", isTraj: true }] : []),
  ];
  const thead = `<th style="width:24px;" title="Toggle evidence detail panel."></th>` + cols.map(c => {
    if (c.isTraj) return `<th>${_escapeHtml(c.label)}</th>`;
    const on = (f.sortKey === c.key);
    const arrow = on ? (f.sortDir > 0 ? " ▲" : " ▼") : "";
    return `<th data-ip-sort="${c.key}">${_escapeHtml(c.label)}${arrow}</th>`;
  }).join("");
  const body = visible.map((r, idx) => {
    _ipNormalizeTopRow(r);
    const rk = _ipRowKey(r);
    const isOpen = _ipRuntime.openKeys.has(rk);
    const toggle = `<td style="text-align:center;cursor:pointer;" `
      + `data-ip-toggle="${idx}" title="${isOpen ? "Hide" : "Show"} evidence detail">`
      + `${isOpen ? "▾" : "▸"}</td>`;
    const path = r.Path || [r.Ligand, r.Receptor, r.EM, r.Target].join("|");
    const cells = cols.map(c => {
      if (c.isTraj) return `<td>${trajBadges(r)}</td>`;
      const v = c.key === "Path" ? path : r[c.key];
      if (c.key === "PDS") {
        const color = Number(v) >= 0 ? "#b91c1c" : "#1d4ed8";
        const dir = Number(v) >= 0 ? "↑" : "↓";
        return `<td style="text-align:right;color:${color};font-weight:600;">${dir} ${fmt(v, 3)}</td>`;
      }
      if (c.numeric) return `<td style="text-align:right;">${_ipFmtNum(v, c.digits)}</td>`;
      // J-3: dropped node → "—" (null from materialize; no label badge).
      if (c.labelKey) {
        if (droppedNodes.has(c.key)) return `<td style="color:#999;">—</td>`;
        return `<td>${_ipNodeCell(v, r[c.labelKey])}</td>`;
      }
      return `<td>${_escapeHtml(v == null ? "" : v)}</td>`;
    }).join("");
    let html = `<tr data-ip-row="${idx}">${toggle}${cells}</tr>`;
    if (isOpen) {
      html += `<tr class="ip-detail-row" data-ip-detail="${idx}"><td></td>`
        + `<td colspan="${cols.length}" style="padding:8px 12px;background:#fafafa;">`
        + _ipRenderDetailPanel(r, rk, "evidence")
        + `</td></tr>`;
    }
    return html;
  }).join("");

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
    + `<thead><tr>${thead}</tr></thead><tbody>${body}</tbody></table></div>`
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
          _ipCloseOpenPanels();
          _ipRenderTopTable();
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
          _ipCloseOpenPanels();
          _ipRenderTopTable();
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
    else IncytrFilter.set({ sortKey: k, sortDir: (k === "rank" || k === "pvalue" ? 1 : -1) });
    _ipResetPage();
    _ipCloseOpenPanels();
    _ipRenderTopTable();
  });
  const bodyEl = wrap.querySelector("#ip-table tbody");
  if (bodyEl) bodyEl.addEventListener("click", ev => {
    const cell = ev.target.closest("td[data-ip-toggle]");
    if (cell) {
      const idx = +cell.dataset.ipToggle;
      const r = visible[idx];
      if (!r) return;
      _ipNormalizeTopRow(r);
      const rk = _ipRowKey(r);
      if (_ipRuntime.openKeys.has(rk)) {
        _ipRuntime.openKeys.delete(rk);
        delete _ipRuntime.detailTab[rk];
        delete _ipRuntime.trajSettings[rk];
        delete _ipRuntime.trajRows[rk];
        delete _ipRuntime.trajPromises[rk];
      } else {
        _ipRuntime.openKeys.add(rk);
        _ipRuntime.detailTab[rk] = "evidence";
      }
      _ipRenderTopTable();
      return;
    }
    const tabBtn = ev.target.closest("button[data-ip-detail-tab]");
    if (tabBtn) {
      const rk = tabBtn.dataset.ipDetailRk;
      const tab = tabBtn.dataset.ipDetailTab;
      _ipRuntime.detailTab[rk] = tab;
      const detailIdx = +tabBtn.closest("tr[data-ip-detail]").dataset.ipDetail;
      const r = visible[detailIdx];
      if (!r) { _ipRenderTopTable(); return; }
      _ipNormalizeTopRow(r);
      const tdEl = tabBtn.closest("td");
      if (tdEl) tdEl.innerHTML = _ipRenderDetailPanel(r, rk, tab);
      if (tab === "trajectory") _ipRenderTrajChart(rk, r);
      if (tab === "evidence") _ipRenderEvidencePanel(rk, r);
      // Related-pairs lookup is async; populate when the tab opens.
      if (tab === "expands-to") _ipRenderRelatedPairs(rk, r);
      if (tab === "sidechains") _isRenderSidechains(rk, r);
      return;
    }
  });
  for (const rk of _ipRuntime.openKeys) {
    const idx = visible.findIndex(r => _ipRowKey(_ipNormalizeTopRow(r)) === rk);
    const tab = _ipRuntime.detailTab[rk] || "evidence";
    if (idx >= 0 && tab === "trajectory") _ipRenderTrajChart(rk, visible[idx]);
    else if (idx >= 0 && tab === "expands-to") _ipRenderRelatedPairs(rk, visible[idx]);
    else if (idx >= 0 && tab === "sidechains") _isRenderSidechains(rk, visible[idx]);
    else if (idx >= 0) _ipRenderEvidencePanel(rk, visible[idx]);
  }
}

function _ipResetPage() {
  _ipRuntime.page = 0;
}

function _ipCloseOpenPanels() {
  _ipRuntime.openKeys = new Set();
  _ipRuntime.detailTab = {};
  _ipRuntime.trajSettings = {};
  _ipRuntime.trajRows = {};
  _ipRuntime.trajPromises = {};
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
  _ipCloseOpenPanels();
  _ipRuntime.recurFallback = null;
  _ipRuntime.pathIndex = null;
  _ipRuntime.pathLabels = null;
  _ipRuntime._didDebugLog = false;
}

function _ipStampRowsForPair(rows, sender, receiver) {
  const sPipe = sender + "||" + receiver + "||";
  for (const r of (rows || [])) {
    r._sender = sender;
    r._receiver = receiver;
    if (r.Path == null)
      r.Path = (r.Ligand || "") + "|" + (r.Receptor || "") + "|"
             + (r.EM || "") + "|" + (r.Target || "");
    r._pathStr = sPipe + r.Path;
  }
  return rows || [];
}

function _ipRowHasExactGeneSearchValue(r, token) {
  return String(r.Ligand || "").toLowerCase() === token
    || String(r.Receptor || "").toLowerCase() === token
    || String(r.EM || "").toLowerCase() === token
    || String(r.Target || "").toLowerCase() === token;
}

// Split a cell-state / contrast label into lowercased search segments on
// camelCase humps, digit boundaries, and non-alphanumeric delimiters:
// "CD8CytotoxicEffector" -> ["cd8", "cytotoxic", "effector"], "d20_d2" -> ["d20", "d2"].
// Descriptive search matches a token only when it is a PREFIX of a segment, so
// a gene query like "tox" is not swallowed mid-word by "cytotoxic" while
// "exhaust" still matches "CD8Exhausted".
function _ipSearchSegments(value) {
  return String(value == null ? "" : value)
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter(Boolean);
}

function _ipRowHasDescriptiveSearchValue(r, token) {
  const hit = (v) => _ipSearchSegments(v).some(seg => seg.startsWith(token));
  return hit(r._sender) || hit(r._receiver) || hit(r.contrast);
}

function _ipBuildPathIndexes(rows) {
  const pathIndex  = new Map();
  const pathLabels = new Map();
  for (const r of (rows || [])) {
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
  return { pathIndex, pathLabels };
}

// ---- shard loading ----

// J-3: For an inline backbone grain (no slice_index), materialize the
// pair's rows directly from the loaded global_index. This keeps pair-mode
// working for R-EM and L-R-EM without requiring shard parquet files.
async function _ipInlineGrainRowsForPair(sender, receiver) {
  const d = await IncytrGlobalIndex.ensureLoaded();
  if (!d) return [];
  const { cols, gi } = d;
  const sidWant = gi.sender_vocab.indexOf(sender);
  const ridWant = gi.receiver_vocab.indexOf(receiver);
  if (sidWant < 0 || ridWant < 0) return [];
  const sidCol = cols.senderId, ridCol = cols.receiverId;
  const rows = [];
  for (let i = 0; i < d.nrows; i++) {
    if (sidCol[i] !== sidWant || ridCol[i] !== ridWant) continue;
    rows.push(IncytrGlobalIndex.materialize(i));
  }
  return rows.filter(Boolean);
}

async function _ipEnsureShards() {
  const block = _ipBlock();
  if (!block) return;
  if ((IncytrFilter.get("ipMode") || "top") === "top") {
    _ipRuntime.loading = false;
    _ipRuntime.loadError = null;
    _ipRenderTable();
    return;
  }
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
  // J-3: include active grain in the shard key so grain switching clears
  // the cached shard (block._grain is undefined for Full).
  const grainKey = block._grain || "Full";
  const sig = _ipScopeSig(pairs) + "||grain=" + grainKey;
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
    let rows;
    // J-3: inline backbone grains have no slice_index; use global_index scan.
    if (block._grainMode === "inline" && !block.slice_index) {
      rows = await _ipInlineGrainRowsForPair(p.sender, p.receiver);
      rows = _ipStampRowsForPair(rows, p.sender, p.receiver);
    } else {
      rows = _ipStampRowsForPair(
        await SliceCache.loadIncytrShard(p.sender, p.receiver),
        p.sender, p.receiver,
      );
    }
    // Resolve race: only commit if the scope hasn't changed mid-fetch.
    const newSig = _ipScopeSig(_ipPairsInScope(block)) + "||grain=" + grainKey;
    if (newSig !== sig) return;
    _ipRuntime.rows = rows;
    _ipRuntime.loadedKey = sig;
    // Build two indexes once so per-render filtering is O(1) lookups:
    //   pathIndex  : pathStr → rows[]   (chart, debug)
    //   pathLabels : pathStr → Map<disease, Set<label>>  (trend filter)
    const { pathIndex, pathLabels } = _ipBuildPathIndexes(rows);
    _ipRuntime.pathIndex  = pathIndex;
    _ipRuntime.pathLabels = pathLabels;
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
  const trend = TrendFilter.normalize(f.trend || "");
  const trendLabel = trend ? TrendFilter.payloadLabel(trend) : "";
  const hasTraj  = !!trendLabel && _ipHasTraj();
  const pathLbl  = hasTraj ? _ipRuntime.pathLabels : null;
  const recurIdx = hasRecur ? _ipRecurIndex() : null;
  const searchTokens  = (f.searchText || "")
    .toLowerCase().split(/\s+/).filter(Boolean);
  const pdsSign = (f.pdsSign === "up" || f.pdsSign === "down") ? f.pdsSign : "both";
  const scoreGates = _ipScoreMinAbs();

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
    if (!_ipPdsSignPass(r.PDS, pdsSign)) continue;
    if (!_ipScoreGatesPass(r, scoreGates)) continue;

    if (hasSearch) {
      let ok = true;
      for (const t of searchTokens) {
        const matched = _ipRowHasExactGeneSearchValue(r, t)
          || _ipRowHasDescriptiveSearchValue(r, t);
        if (!matched) { ok = false; break; }
      }
      if (!ok) continue;
    }

    if (hasTraj) {
      const byDis = pathLbl ? pathLbl.get(_ipPathStr(r)) : null;
      if (!byDis) continue;
      const c = r.contrast || "";
      const ui = c.indexOf("_");
      const d = ui < 0 ? c : c.substring(0, ui);
      const lbls = byDis.get(d);
      if (!lbls || !lbls.has(trendLabel)) continue;
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
      console.log("[ip-filter] trend=", trendLabel);
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
  // J-3: Timepoint-combination filter (entity-level, evaluated per-disease).
  // Mirrors _filterTimepointCombine in incytr_global_index.js for pair mode.
  const tpCombine = f.timepointCombine || [];
  if (tpCombine.length && out.length) {
    const tpMode = f.timepointCombineMode || "all";
    const req = new Set(tpCombine);
    const qualifies = (tpSet) => tpMode === "all"
      ? [...req].every(tp => tpSet.has(tp))
      : [...req].some(tp => tpSet.has(tp));
    // Group entity key (pathStr) → Map<disease, Set<timepoint>> from filtered rows.
    const entityDisTp = new Map();
    for (const r of out) {
      const pk = _ipPathStr(r);
      const c = r.contrast || "";
      const ui = c.indexOf("_");
      const dis = ui < 0 ? c : c.substring(0, ui);
      const tp  = ui < 0 ? "" : c.substring(ui + 1);
      let dm = entityDisTp.get(pk);
      if (!dm) { dm = new Map(); entityDisTp.set(pk, dm); }
      let ts = dm.get(dis);
      if (!ts) { ts = new Set(); dm.set(dis, ts); }
      ts.add(tp);
    }
    const passing = new Set();
    for (const [pk, dm] of entityDisTp) {
      for (const tpSet of dm.values()) {
        if (qualifies(tpSet)) { passing.add(pk); break; }
      }
    }
    const prevLen = out.length;
    for (let i = prevLen - 1; i >= 0; i--) {
      if (!passing.has(_ipPathStr(out[i]))) out.splice(i, 1);
    }
  }

  const key = f.sortKey === "rank" ? "PDS" : f.sortKey;
  const dir = f.sortKey === "rank" ? -1 : f.sortDir;
  const numericKeys = new Set([
    "pvalue", "PDS", ..._ipScoreCols(),
  ]);
  const isNumeric = numericKeys.has(key);
  const cmp = isNumeric
    ? (a, b) => numCmp(a[key], b[key], dir)
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
  if ((IncytrFilter.get("ipMode") || "top") === "top") {
    _ipRenderTopTable();
    return;
  }

  const pairs = _ipPairsInScope(block);
  const f = IncytrFilter.get();
  if (pairs.length === 0) {
    if (_ipRenderGeneIndexSearch(block, pairs)) return;
    const lowTxt = IncytrCelltypeQc.enabled(block)
      ? ` after ${IncytrCelltypeQc.controlText(block)}` : "";
    countEl.textContent = `No (sender, receiver) pair matches the current selection${lowTxt}.`;
    wrap.innerHTML = '<div class="muted" style="padding:16px;">'
      + 'Pick one sender and one receiver from the filters, or click a cell in the heatmap, to load that pair\'s shard.'
      + '</div>';
    return;
  }
  if (pairs.length > 1) {
    if (_ipRenderGeneIndexSearch(block, pairs)) return;
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
  if (!filtered.length) {
    const p0 = pairs[0];
    countEl.textContent =
      `${p0.sender} → ${p0.receiver}: 0 rows pass filters `
      + `(of ${total.toLocaleString()} in shard).`
      + (IncytrCelltypeQc.enabled(block) ? ` ${IncytrCelltypeQc.controlText(block)}.` : "");
    const summary = _ipActiveFilterSummary(f);
    wrap.innerHTML = '<div class="muted" style="padding:16px;">'
      + (summary
          ? `No rows match: ${_escapeHtml(summary)}. Use Reset or loosen the filters.`
          : 'No rows match the active filters. Use Reset or loosen the filters.')
      + '</div>';
    return;
  }
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
        : "")
    + (IncytrCelltypeQc.enabled(block) ? ` ${IncytrCelltypeQc.controlText(block)}.` : "");

  const hasTrajIdx = _ipHasTraj();
  const scoreCols = _ipScoreCols().map(k => ({
    key: k, label: _ipMetricLabel(k), numeric: true, digits: 3,
    tip: _IP_SCORE_TIPS[k] || `${k} score column from Incytr factorial scoring.`,
  }));
  // J-3: dropped nodes in the active grain show "—" in their column.
  const droppedNodesPair = _ipDroppedNodes();
  // J-3: pvalue absent for backbone grain row data (shards omit pvalue col in backbone mode).
  const pairHasPvalue = !_ipIsBackboneGrain();
  const cols = [
    { key: "_sender",       label: "Sender",
      tip: "WMB cell-type class emitting the ligand." },
    { key: "_receiver",     label: "Receiver",
      tip: "WMB cell-type class receiving the signal." },
    { key: "Path",          label: "Path",
      tip: droppedNodesPair.size ? `Node path (active grain: ${_ipBlock()._grain || "Full"}; dropped nodes show —).` : "4-node signaling path: Ligand → Receptor → EM → Target." },
    { key: "Ligand",        label: "Ligand",   labelKey: "Ligand_label",
      tip: droppedNodesPair.has("Ligand") ? "Ligand dropped in active grain." : "Secreted/membrane ligand gene at the start of the path. Badge marks the evidence source (DEG/prG)." },
    { key: "Receptor",      label: "Receptor", labelKey: "Receptor_label",
      tip: "Receptor gene on the receiver cell. Badge marks the evidence source (DEG/prG)." },
    { key: "EM",            label: "EM",       labelKey: "EM_label",
      tip: "Effector molecule — intracellular signaling node between Receptor and Target. Badge marks the evidence source (DEG/prG)." },
    { key: "Target",        label: "Target",   labelKey: "Target_label",
      tip: droppedNodesPair.has("Target") ? "Target dropped in active grain." : "Terminal gene the path is predicted to regulate. Badge marks the evidence source (DEG/prG)." },
    { key: "contrast",      label: "contrast",
      tip: "Disease × timepoint contrast vs WT (e.g., App_4mo = APP/PS1 vs WT at 4 mo)." },
    ...(pairHasPvalue ? [{ key: "pvalue", label: "pvalue", numeric: true, digits: "sci",
      tip: "Wald t-test pvalue on the contrast coefficient from Incytr's factorial OLS (pvalue_method=t_test, n_perm=0 in this run). Lower = more confident change vs WT." }] : []),
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
    `<th style="width:24px;" title="Toggle detail panel (Evidence: 4 nodes × 4 layers, + score trajectories)."></th>`
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
      // J-3: dropped node → "—" (null value from materialize/inline-scan).
      if (c.labelKey) {
        if (droppedNodesPair.has(c.key)) return `<td style="color:#999;">—</td>`;
        return `<td>${_ipNodeCell(v, r[c.labelKey])}</td>`;
      }
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
          _ipCloseOpenPanels();
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
          _ipCloseOpenPanels();
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
        delete _ipRuntime.trajSettings[rk];
        delete _ipRuntime.trajRows[rk];
        delete _ipRuntime.trajPromises[rk];
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
      // Related-pairs lookup is async; populate when the tab opens.
      if (tab === "expands-to") _ipRenderRelatedPairs(rk, r);
      if (tab === "sidechains") _isRenderSidechains(rk, r);
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
    } else if (tab === "expands-to") {
      const idx = visible.findIndex(r => _ipRowKey(r) === rk);
      if (idx >= 0) _ipRenderRelatedPairs(rk, visible[idx]);
    } else if (tab === "sidechains") {
      const idx = visible.findIndex(r => _ipRowKey(r) === rk);
      if (idx >= 0) _isRenderSidechains(rk, visible[idx]);
    }
  }
}

// ---------------------------------------------------------------------------
// Detail panel: "Evidence" tab (default) + "Trajectory" sub-tab (when
// trajectory_index is present in the payload). The Evidence tab renders all
// 4 nodes × 4 omics layers from the per-cluster omics + transcript shards.
// Cluster routing per evaluation.R:227-230 is enforced in EvidencePanel.render().
// ---------------------------------------------------------------------------

function _ipRenderDetailPanel(r, rk, activeTab, allowTrajectory = true) {
  const hasTrajIdx = allowTrajectory && _ipHasScoreTrajectory();
  // B-6: show "Expands to" tab when in a backbone grain OR when backbone grains
  // are available (Full rows can collapse to them).
  const block = _ipBlock();
  const grain = block && block._grain || "Full";
  const parentBlock = ViewerPayload.incytr ? ViewerPayload.incytr() : null;
  const hasBackboneGrains = !!(parentBlock && parentBlock.backbone_grains
    && Object.keys(parentBlock.backbone_grains).length);
  const showExpandsTo = grain !== "Full" || hasBackboneGrains;
  const showSidechains = typeof IncytrSidechains !== "undefined"
    && IncytrSidechains.hasActiveSlice();
  if (activeTab === "sidechains" && !showSidechains) {
    _ipRuntime.detailTab[rk] = "evidence";
    activeTab = "evidence";
  }
  const btn = (tab, label) =>
    `<button type="button" data-ip-detail-tab="${tab}" data-ip-detail-rk="${_escapeHtml(rk)}"
       style="padding:2px 12px;border-radius:4px;font-size:12px;cursor:pointer;
              border:1px solid #c0c0c0;
              background:${activeTab === tab ? "#1f4ea3" : "#f4f4f4"};
              color:${activeTab === tab ? "#fff" : "#444"};"
     >${label}</button>`;
  const tabBar = allowTrajectory
    ? (`<div style="display:flex;gap:6px;margin-bottom:8px;">`
      + btn("evidence", "Evidence")
      + (hasTrajIdx ? btn("trajectory", "Scores") : "")
      + (showExpandsTo ? btn("expands-to", "Related pathways") : "")
      + (showSidechains ? btn("sidechains", "Sidechains") : "")
      + `</div>`)
    : "";
  if (activeTab === "trajectory" && hasTrajIdx) {
    const safe = rk.replace(/[^a-zA-Z0-9]/g, "_");
    const chartId = `ip-traj-${safe}`;
    const controlsId = `ip-traj-controls-${safe}`;
    return tabBar
      + `<div id="${_escapeHtml(controlsId)}"></div>`
      + `<div id="${_escapeHtml(chartId)}" style="width:100%;min-height:260px;"></div>`;
  }
  // B-6: "Expands to" tab — grain drill / expansion navigation peek.
  if (activeTab === "expands-to" && showExpandsTo) {
    return tabBar + _ipRenderExpandsToPanel(r, rk);
  }
  if (activeTab === "sidechains" && showSidechains) {
    return tabBar + `<div id="${_escapeHtml(_isHostId(rk))}">`
      + `<div class="muted" style="font-size:11px;padding:8px 0;">Loading kinase sidechains…</div>`
      + `</div>`;
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

// ---------------------------------------------------------------------------
// "Related pathways" drawer tab — read-only. Reports two facts about the row:
//   • backbone fan-out: how many Full pathways collapse into this backbone row
//     (r.n_paths; Full rows are a single pathway and omit this);
//   • the other (sender, receiver) cell-type pairs carrying this same pathway,
//     enumerated from the per-grain global index (no sidecar, no navigation).
// ---------------------------------------------------------------------------

function _ipRenderExpandsToPanel(r, rk) {
  const block = _ipBlock();
  if (!r || !block) return '<div class="muted">No pathway data.</div>';

  const grain  = block._grain || "Full";
  const esc     = _escapeHtml;
  const safeRk  = rk.replace(/[^a-zA-Z0-9]/g, "_");

  let html = `<div class="ip-related">`;

  // Backbone rows aggregate multiple Full pathways (distinct Target/Ligand
  // nodes); surface that count. Full rows are already a single pathway.
  const nPaths = Number(r.n_paths);
  if (grain !== "Full" && isFinite(nPaths) && nPaths > 0) {
    html += `<div class="ip-related__count"><strong>${nPaths.toLocaleString()}</strong> `
      + `full pathway${nPaths === 1 ? "" : "s"} collapse into this backbone</div>`;
  }

  // Every (sender, receiver) pair carrying this pathway/spine at the active grain.
  html += `<section class="ip-related-pairs">`;
  html += `<div class="ip-related__head">Related cell-type pairs</div>`;
  html += `<div id="ip-related-${esc(safeRk)}" class="ip-related-pairs__results">`;
  html += `<span class="muted">Loading…</span></div>`;
  html += `</section></div>`;
  return html;
}

// Populate the read-only "Related cell-type pairs" panel: the distinct
// (sender, receiver) pairs whose pathway matches this row at the active grain,
// grouped by receiver. Backed by the per-grain global index — available in both
// Top and Cell Type modes (the binary ships per grain; ensureLoaded fetches it
// on demand). Re-queries the host after the await since the panel may re-render.
async function _ipRenderRelatedPairs(rk, r) {
  if (!r) return;
  const hostId = `ip-related-${rk.replace(/[^a-zA-Z0-9]/g, "_")}`;
  if (typeof IncytrGlobalIndex === "undefined" || !IncytrGlobalIndex.available()) {
    const host = document.getElementById(hostId);
    if (host) host.innerHTML = `<div class="muted" style="font-size:11px;">Pair lookup unavailable in this view.</div>`;
    return;
  }
  try {
    if (!IncytrGlobalIndex.loaded()) await IncytrGlobalIndex.ensureLoaded();
    const host = document.getElementById(hostId);
    if (!host) return;   // panel closed/re-rendered while loading
    const ident = _ipPathIdentity(r);
    const pairs = IncytrGlobalIndex.pairsForPath(ident);
    if (!pairs.length) {
      host.innerHTML = `<div class="muted" style="font-size:11px;">No cell-type pairs carry this pathway.</div>`;
      return;
    }
    // One row per pair (count == #rows). Heaviest pairs first; the parent pair is
    // marked so the user can locate the row they opened the panel from.
    pairs.sort((a, b) => (b[2] - a[2]) || a[1].localeCompare(b[1]) || a[0].localeCompare(b[0]));
    let rows = "";
    for (const [s, rec, cnt] of pairs) {
      const self = s === ident.sender && rec === ident.receiver;
      const tdBase = "padding:2px 6px;border-bottom:1px solid #eee;";
      rows += `<tr style="${self ? "background:#eef4fb;font-weight:600;" : ""}">`
        + `<td style="${tdBase}">${_escapeHtml(s)}</td>`
        + `<td style="${tdBase}color:#555;">${_escapeHtml(rec)}</td>`
        + `<td style="${tdBase}text-align:right;">${cnt.toLocaleString()}</td></tr>`;
    }
    host.innerHTML = `<div style="font-size:11px;color:#444;">
      <div style="margin-bottom:4px;font-weight:600;">${pairs.length} cell-type pair(s) carry this pathway:</div>
      <table style="border-collapse:collapse;width:100%;">
        <thead><tr>
          <th style="text-align:left;padding:2px 6px;border-bottom:2px solid #ddd;">Sender</th>
          <th style="text-align:left;padding:2px 6px;border-bottom:2px solid #ddd;">Receiver</th>
          <th style="text-align:right;padding:2px 6px;border-bottom:2px solid #ddd;">Pathways</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table></div>`;
  } catch (e) {
    const host = document.getElementById(hostId);
    if (host) host.innerHTML = `<div class="muted" style="font-size:11px;">Pair lookup error: ${_escapeHtml(String(e && e.message ? e.message : e))}</div>`;
  }
}

function _ipPathIdentity(r) {
  const row = _ipNormalizeTopRow(Object.assign({}, r));
  return {
    sender: row._sender || row.sender || "",
    receiver: row._receiver || row.receiver || "",
    path: row.Path || [row.Ligand, row.Receptor, row.EM, row.Target].join("|"),
  };
}

function _ipSamePathIdentity(row, ident) {
  const ri = _ipPathIdentity(row);
  return ri.sender === ident.sender && ri.receiver === ident.receiver && ri.path === ident.path;
}

async function _ipRowsForTrajectory(r, rk) {
  if (rk && _ipRuntime.trajRows[rk]) return _ipRuntime.trajRows[rk];
  if (rk && _ipRuntime.trajPromises[rk]) return _ipRuntime.trajPromises[rk];
  const ident = _ipPathIdentity(r);
  if (!ident.sender || !ident.receiver || !ident.path) return [];
  const load = (async () => {
    if ((IncytrFilter.get("ipMode") || "top") === "top"
        && typeof IncytrGlobalIndex !== "undefined"
        && IncytrGlobalIndex.available()) {
      if (!IncytrGlobalIndex.loaded()) await IncytrGlobalIndex.ensureLoaded();
      if (IncytrGlobalIndex.loaded() && typeof IncytrGlobalIndex.pathRows === "function") {
        return IncytrGlobalIndex.pathRows(ident);
      }
    }
    const pairSig = `${ident.sender}||${ident.receiver}`;
    if (_ipRuntime.rows && _ipRuntime.loadedKey === pairSig) {
      return _ipRuntime.rows.filter(row => _ipSamePathIdentity(row, ident));
    }
    const rows = _ipStampRowsForPair(
      await SliceCache.loadIncytrShard(ident.sender, ident.receiver),
      ident.sender, ident.receiver,
    );
    return rows.filter(row => _ipSamePathIdentity(row, ident));
  })();
  if (rk) {
    _ipRuntime.trajPromises[rk] = load;
    load.then(rows => {
      if (_ipRuntime.openKeys.has(rk)) _ipRuntime.trajRows[rk] = rows;
      return rows;
    })
      .finally(() => { delete _ipRuntime.trajPromises[rk]; });
  }
  return load;
}

function _ipAvailableTrajectoryMetrics(rows) {
  const metrics = ["PDS", ..._ipScoreCols()];
  return metrics.filter(m => (rows || []).some(r => {
    const v = Number(r[m]);
    return isFinite(v);
  }));
}

function _ipTrajectorySettings(rk, r, metrics) {
  const axis = _ipScoreTrajectoryAxis();
  const clicked = _ipContrastParts(r.contrast || "");
  const clickedSeries = axis.seriesKind === "group" ? clicked.group : clicked.timepoint;
  const fallbackSeries = axis.seriesValues.indexOf(clickedSeries) >= 0
    ? clickedSeries
    : (axis.seriesValues[0] || clickedSeries || _IP_TRAJ_ALL_GROUPS);
  const available = metrics.length ? metrics : ["PDS"];
  let st = _ipRuntime.trajSettings[rk];
  if (!st) st = _ipRuntime.trajSettings[rk] = { series: fallbackSeries, metrics: ["PDS"], view: "both" };
  if (st.series == null && st.group != null) st.series = st.group;
  if (st.series !== _IP_TRAJ_ALL_GROUPS && axis.seriesValues.indexOf(st.series) < 0) st.series = fallbackSeries;
  st.metrics = (Array.isArray(st.metrics) ? st.metrics : ["PDS"]).filter(m => available.indexOf(m) >= 0);
  if (!st.metrics.length) st.metrics = [available.indexOf("PDS") >= 0 ? "PDS" : available[0]];
  if (st.series === _IP_TRAJ_ALL_GROUPS && st.metrics.length > 1) st.metrics = [st.metrics[0]];
  if (!["both", "line", "bar"].includes(st.view || "")) st.view = "both";
  return st;
}

function _ipBestRowByContrast(rows) {
  const byContrast = new Map();
  for (const row of (rows || [])) {
    const c = row.contrast || "";
    const cur = byContrast.get(c);
    if (!cur || Math.abs(Number(row.PDS) || 0) > Math.abs(Number(cur.PDS) || 0)) {
      byContrast.set(c, row);
    }
  }
  return byContrast;
}

function _ipTrajectoryControlHtml(rk, state, metrics) {
  const axis = _ipScoreTrajectoryAxis();
  const viewButtons = [
    { key: "both", label: "Both" },
    { key: "line", label: "Line" },
    { key: "bar", label: "Bars" },
  ].map(v => {
    const active = state.view === v.key;
    return `<button type="button" data-ip-traj-rk="${_escapeHtml(rk)}" `
      + `data-ip-traj-view="${_escapeHtml(v.key)}" `
      + `style="padding:2px 9px;border-radius:4px;border:1px solid ${active ? "#1f4ea3" : "#c0c0c0"};`
      + `background:${active ? "#1f4ea3" : "#f7f7f7"};color:${active ? "#fff" : "#37474f"};`
      + `font-size:12px;cursor:pointer;">${_escapeHtml(v.label)}</button>`;
  }).join("");
  const seriesChoices = axis.seriesValues.length > 1
    ? [...axis.seriesValues, _IP_TRAJ_ALL_GROUPS]
    : [];
  const seriesButtons = seriesChoices.map(s => {
    const label = s === _IP_TRAJ_ALL_GROUPS ? "All" : s;
    const active = state.series === s;
    return `<button type="button" data-ip-traj-rk="${_escapeHtml(rk)}" `
      + `data-ip-traj-series="${_escapeHtml(s)}" `
      + `style="padding:2px 9px;border-radius:4px;border:1px solid ${active ? "#1f4ea3" : "#c0c0c0"};`
      + `background:${active ? "#1f4ea3" : "#f7f7f7"};color:${active ? "#fff" : "#37474f"};`
      + `font-size:12px;cursor:pointer;">${_escapeHtml(label)}</button>`;
  }).join("");
  const seriesControl = seriesButtons
    ? `<span class="muted" style="font-size:11px;">${_escapeHtml(axis.seriesLabel)}</span>`
      + `<span style="display:flex;gap:4px;">${seriesButtons}</span>`
    : "";
  const metricButtons = metrics.map(m => {
    const active = state.metrics.indexOf(m) >= 0;
    const color = _IP_TRAJ_METRIC_COLORS[m] || "#455a64";
    return `<button type="button" data-ip-traj-rk="${_escapeHtml(rk)}" `
      + `data-ip-traj-metric="${_escapeHtml(m)}" `
      + `style="padding:2px 9px;border-radius:4px;border:1px solid ${active ? color : "#c0c0c0"};`
      + `background:${active ? _ipHexAlpha(color, 0.12) : "#f7f7f7"};color:${active ? color : "#37474f"};`
      + `font-size:12px;cursor:pointer;">${_escapeHtml(_ipMetricLabel(m))}</button>`;
  }).join("");
  const metricNote = state.series === _IP_TRAJ_ALL_GROUPS
    ? `<span class="muted" style="font-size:11px;">All mode shows one metric across ${_escapeHtml(axis.seriesLabel.toLowerCase())}s.</span>`
    : "";
  return `<div style="display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;margin-bottom:6px;">`
    + `<span class="muted" style="font-size:11px;">View</span><span style="display:flex;gap:4px;">${viewButtons}</span>`
    + seriesControl
    + `<span class="muted" style="font-size:11px;">Score</span><span style="display:flex;gap:4px;flex-wrap:wrap;">${metricButtons}</span>`
    + metricNote
    + `</div>`;
}

function _ipWireTrajectoryControls(rk, r, controlEl) {
  if (!controlEl) return;
  controlEl.querySelectorAll("[data-ip-traj-series]").forEach(btn => {
    btn.addEventListener("click", () => {
      const metrics = _ipAvailableTrajectoryMetrics(_ipRuntime.trajRows[rk] || []);
      const st = _ipTrajectorySettings(rk, r, metrics);
      st.series = btn.dataset.ipTrajSeries || st.series;
      if (st.series === _IP_TRAJ_ALL_GROUPS && st.metrics.length > 1) st.metrics = [st.metrics[0]];
      _ipRenderTrajChart(rk, r);
    });
  });
  controlEl.querySelectorAll("[data-ip-traj-metric]").forEach(btn => {
    btn.addEventListener("click", () => {
      const metrics = _ipAvailableTrajectoryMetrics(_ipRuntime.trajRows[rk] || []);
      const metric = btn.dataset.ipTrajMetric;
      const st = _ipTrajectorySettings(rk, r, metrics);
      if (st.series === _IP_TRAJ_ALL_GROUPS) {
        st.metrics = [metric];
      } else if (st.metrics.indexOf(metric) >= 0) {
        const next = st.metrics.filter(m => m !== metric);
        st.metrics = next.length ? next : [metric];
      } else {
        st.metrics = [...st.metrics, metric];
      }
      _ipRenderTrajChart(rk, r);
    });
  });
  controlEl.querySelectorAll("[data-ip-traj-view]").forEach(btn => {
    btn.addEventListener("click", () => {
      const metrics = _ipAvailableTrajectoryMetrics(_ipRuntime.trajRows[rk] || []);
      const st = _ipTrajectorySettings(rk, r, metrics);
      st.view = btn.dataset.ipTrajView || "both";
      _ipRenderTrajChart(rk, r);
    });
  });
}

function _ipPointText(group, timepoint, metric, value, row) {
  const pv = row && row.pvalue != null && isFinite(row.pvalue)
    ? Number(row.pvalue).toExponential(2) : "—";
  const v = value == null || !isFinite(value) ? "missing" : Number(value).toFixed(3);
  return `${group} ${timepoint}<br>${_ipMetricLabel(metric)} ${v}<br>p ${pv}`;
}

function _ipTrajectoryContrast(axis, series, x) {
  return axis.xKind === "timepoint" ? `${series}_${x}` : `${x}_${series}`;
}

function _ipTrajectoryPointParts(axis, series, x) {
  return axis.xKind === "timepoint"
    ? { group: series, timepoint: x }
    : { group: x, timepoint: series };
}

function _ipTrajectorySeriesColor(axis, series) {
  if (axis.seriesKind === "group") return _IP_TRAJ_GROUP_COLORS[series] || "#455a64";
  return "#455a64";
}

function _ipTrajectoryTrace(kind, name, color, x, y, text, showLegend) {
  if (kind === "bar") {
    return {
      type: "bar", name, x, y,
      marker: { color: _ipHexAlpha(color, 0.42), line: { color, width: 1 } },
      customdata: text,
      hovertemplate: "%{customdata}<extra></extra>",
      showlegend: !!showLegend,
    };
  }
  return {
    type: "scatter", mode: "lines+markers", name,
    x, y, connectgaps: false,
    line: { color, width: name === "PDS" ? 2.5 : 1.9 },
    marker: { color, size: 6 },
    customdata: text,
    hovertemplate: "%{customdata}<extra></extra>",
    showlegend: !!showLegend,
  };
}

function _ipRenderTrajChartRows(rk, r, rows) {
  const safe = rk.replace(/[^a-zA-Z0-9]/g, "_");
  const chartId = `ip-traj-${safe}`;
  const controlsId = `ip-traj-controls-${safe}`;
  const host = document.getElementById(chartId);
  const controlEl = document.getElementById(controlsId);
  if (!host) return;
  const metrics = _ipAvailableTrajectoryMetrics(rows);
  if (!rows.length || !metrics.length) {
    if (controlEl) controlEl.innerHTML = "";
    host.innerHTML = `<div class="muted">No sibling time-course rows found for this pathway.</div>`;
    return;
  }
  const state = _ipTrajectorySettings(rk, r, metrics);
  if (controlEl) {
    controlEl.innerHTML = _ipTrajectoryControlHtml(rk, state, metrics);
    _ipWireTrajectoryControls(rk, r, controlEl);
  }
  const byContrast = _ipBestRowByContrast(rows);
  const axis = _ipScoreTrajectoryAxis();
  const seriesValues = state.series === _IP_TRAJ_ALL_GROUPS ? axis.seriesValues : [state.series];
  const traces = [];
  const addSeries = (label, color, series, metric) => {
    const ys = [], text = [];
    for (const x of axis.xValues) {
      const row = byContrast.get(_ipTrajectoryContrast(axis, series, x));
      const v = row && isFinite(Number(row[metric])) ? Number(row[metric]) : null;
      const parts = _ipTrajectoryPointParts(axis, series, x);
      ys.push(v);
      text.push(_ipPointText(parts.group, parts.timepoint, metric, v, row));
    }
    if (state.view === "bar" || state.view === "both") {
      traces.push(_ipTrajectoryTrace("bar", label, color, axis.xValues, ys, text, state.view === "bar"));
    }
    if (state.view === "line" || state.view === "both") {
      traces.push(_ipTrajectoryTrace("line", label, color, axis.xValues, ys, text, true));
    }
  };
  if (state.series === _IP_TRAJ_ALL_GROUPS) {
    const metric = state.metrics[0];
    for (const series of seriesValues) {
      addSeries(series || axis.seriesLabel, _ipTrajectorySeriesColor(axis, series), series, metric);
    }
  } else {
    for (const metric of state.metrics) {
      addSeries(_ipMetricLabel(metric), _IP_TRAJ_METRIC_COLORS[metric] || "#455a64", state.series, metric);
    }
  }
  const anyValue = traces.some(t => (t.y || []).some(v => v != null && isFinite(v)));
  if (!anyValue) {
    host.innerHTML = `<div class="muted">No finite score values for the selected group and metric.</div>`;
    return;
  }
  const ident = _ipPathIdentity(r);
  const titleSeries = state.series === _IP_TRAJ_ALL_GROUPS
    ? `all ${axis.seriesLabel.toLowerCase()}s · ${_ipMetricLabel(state.metrics[0])}`
    : `${state.series || axis.seriesLabel} · ${state.metrics.map(_ipMetricLabel).join(", ")}`;
  Plotly.react(chartId, traces, {
    margin: { l: 54, r: 18, t: 36, b: 48 },
    height: 320,
    title: { text: `${titleSeries} · ${ident.path}`, font: { size: 12 } },
    barmode: "group",
    bargap: 0.28,
    yaxis: {
      title: "score",
      zeroline: true,
      zerolinecolor: "#111",
      zerolinewidth: 1,
    },
    xaxis: { title: axis.xLabel, type: "category", categoryorder: "array", categoryarray: axis.xValues },
    showlegend: true,
    legend: { orientation: "h", y: -0.25 },
    plot_bgcolor: "#fff",
    paper_bgcolor: "#fff",
  }, { displaylogo: false, responsive: true });
}

// Render score trajectories for the selected pathway. In Cell Type mode this
// uses the already-loaded pair shard; in Top mode it resolves sibling rows from
// the global typed-array index and avoids a full pair-shard fetch.
function _ipRenderTrajChart(rk, r) {
  const safe = rk.replace(/[^a-zA-Z0-9]/g, "_");
  const chartId = `ip-traj-${safe}`;
  const host = document.getElementById(chartId);
  if (!host) return;
  if (typeof Plotly === "undefined") {
    host.innerHTML = `<div class="muted">Plotly not available for score trajectories.</div>`;
    return;
  }
  if (!_ipHasScoreTrajectory()) {
    host.innerHTML = `<div class="muted">Score trajectory unavailable for this context.</div>`;
    return;
  }
  if (_ipRuntime.trajRows[rk]) {
    _ipRenderTrajChartRows(rk, r, _ipRuntime.trajRows[rk]);
    return;
  }
  if (!_ipRuntime.trajPromises[rk]) {
    host.innerHTML = `<div class="muted" style="font-size:11px;padding:8px 0;">Loading score trajectory…</div>`;
  }
  _ipRowsForTrajectory(r, rk).then(rows => {
    if (!document.getElementById(chartId)) return;
    _ipRuntime.trajRows[rk] = rows;
    _ipRenderTrajChartRows(rk, r, rows);
  }).catch(err => {
    host.innerHTML = `<div class="muted">Score trajectory load error: ${_escapeHtml(err.message || err)}</div>`;
  });
}

// Hex color to rgba with alpha (for score-trajectory control styling).
function _ipHexAlpha(hex, alpha) {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex || "");
  if (!m) return hex;
  return `rgba(${parseInt(m[1],16)},${parseInt(m[2],16)},${parseInt(m[3],16)},${alpha})`;
}

function _ipCsvColumns(rows) {
  const scoreCols = _ipScoreCols();
  const cols = [
    "sender", "receiver", "Path", "Ligand", "Ligand_label",
    "Receptor", "Receptor_label", "EM", "EM_label", "Target", "Target_label",
    "contrast", "pvalue", "PDS", ...scoreCols,
  ];
  if (rows.some(r => r.rank != null)) cols.unshift("rank");
  if (rows.some(r => r.traj_labels != null)) cols.push("traj_labels");
  if (rows.some(r => r.sign_vec != null)) cols.push("sign_vec");
  if (rows.some(r => r.low_signal_endpoint != null)) cols.push("low_signal_endpoint");
  return cols;
}

function _ipDownloadCsv(rows, filename) {
  const normalized = rows.map(r => {
    const out = Object.assign({}, r);
    if (out.sender == null) out.sender = out._sender || "";
    if (out.receiver == null) out.receiver = out._receiver || "";
    if (out.Path == null)
      out.Path = [out.Ligand, out.Receptor, out.EM, out.Target].join("|");
    return out;
  });
  const cols = _ipCsvColumns(normalized);
  // Column names equal data keys for the Incytr pathway schema.
  csvDownload(csvSerialize(cols, cols, normalized), filename);
}

async function _ipExportCurrentView() {
  const block = _ipBlock();
  if (!block) {
    alert("No Incytr pathways block is available for export.");
    return;
  }
  const mode = IncytrFilter.get("ipMode") || "top";
  let rows = [];
  if (mode === "top") {
    if (!IncytrGlobalIndex.available()) {
      alert("This payload does not package a complete global Incytr pathway index. Switch to Cell Type mode and select one pair to export its filtered rows.");
      return;
    }
    await IncytrGlobalIndex.ensureLoaded();
    // Export the ranked Top-N the table shows (caps at f.topLimit), not the
    // whole matching universe. The display cap IS a filter from the user's view.
    const { indices } = IncytrGlobalIndex.filterRank(IncytrFilter.get());
    rows = indices.map(i => IncytrGlobalIndex.materialize(i)).filter(Boolean);
  } else {
    const pairs = _ipPairsInScope(block);
    if (pairs.length !== 1) {
      alert("Select exactly one sender/receiver pair in Cell Type mode before exporting.");
      return;
    }
    const sig = _ipScopeSig(pairs);
    if (_ipRuntime.loadedKey !== sig || !_ipRuntime.rows) {
      await _ipEnsureShards();
    }
    if (_ipRuntime.loadError) {
      alert(`InCytr pathway shard could not be loaded: ${_ipRuntime.loadError}`);
      return;
    }
    rows = _ipFilterRows();
  }
  const ctx = (ViewerPayload.activeContext && ViewerPayload.activeContext()) || "context";
  // Map internal context id to a display name when a COHORT_LABELS map is available
  // (unified viewer). Tcell donor ids (donor1/donor2) pass through as-is.
  const ctxDisplay = (typeof COHORT_LABELS !== "undefined" && COHORT_LABELS[ctx])
    ? COHORT_LABELS[ctx]
    : String(ctx).replace(/[^A-Za-z0-9_.-]+/g, "_");
  _ipDownloadCsv(rows, exportFilename(ctxDisplay, "incytr_pathways"));
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

  const signSel = document.getElementById("ip-pds-sign");
  if (signSel) signSel.addEventListener("change", () => {
    IncytrFilter.set({ pdsSign: signSel.value || "both" });
    _ipResetPage();
    _ipRenderTable();
  });

  const scoreHost = document.getElementById("ip-score-filters");
  if (scoreHost) scoreHost.addEventListener("input", ev => {
    const input = ev.target.closest("input[data-ip-score-filter]");
    if (!input) return;
    const next = Object.assign({}, _ipScoreMinAbs());
    const key = input.dataset.ipScoreFilter;
    const raw = input.value === "" ? null : parseFloat(input.value);
    if (raw == null || !isFinite(raw)) delete next[key];
    else next[key] = raw;
    IncytrFilter.set({ scoreMinAbs: next });
    _ipResetPage();
    _ipRenderTableDebounced();
  });

  // Search box — substring AND across Path/nodes/sender/receiver/contrast.
  // Debounced for the same reason as sliders.
  const searchEl = document.getElementById("ip-search");
  if (searchEl) searchEl.addEventListener("input", () => {
    IncytrFilter.set({ searchText: searchEl.value || "" });
    _ipResetPage();
    _ipRenderTableDebounced();
  });

  const trendSel = document.getElementById("ip-trend");
  if (trendSel) trendSel.addEventListener("change", () => {
    IncytrFilter.set({ trend: TrendFilter.normalize(trendSel.value || "") });
    _ipResetPage();
    _ipRenderTable();
  });

  const modeSel = document.getElementById("ip-mode");
  if (modeSel) modeSel.addEventListener("change", () => {
    IncytrFilter.set({
      ipMode: modeSel.value || "top",
      pair: null,
    });
    const block = _ipBlock();
    if (block) _ipSyncControls(block);
    _ipInvalidateScope();
    _ipResetPage();
    _ipEnsureShards();
  });

  const limitSel = document.getElementById("ip-top-limit");
  if (limitSel) limitSel.addEventListener("change", () => {
    const n = Number(limitSel.value || 500);
    IncytrFilter.set({ topLimit: [500, 1000, 5000].includes(n) ? n : 500 });
    _ipResetPage();
    _ipCloseOpenPanels();
    _ipRenderTable();
  });

  // J-3: Grain selector. Resets the global binary index (each grain has its
  // own binary) and invalidates the shard cache (pair-mode key includes grain).
  const grainSel = document.getElementById("ip-grain");
  if (grainSel && !grainSel._wired) {
    grainSel._wired = true;
    grainSel.addEventListener("change", () => {
      IncytrGlobalIndex.reset();
      _ipRuntime.loadedKey = null;   // invalidate pair-mode cache
      IncytrFilter.set({ grain: grainSel.value || "Full", pairPage: 0 });
      // The grain selector is shared between the table, heatmap, and chord panes.
      // Refresh whichever is active: the heatmap/chord re-read the grain's count
      // tensors; the table re-scopes its shards/global index.
      if (_ihVisPaneActive()) {
        _ihRefreshActive();
        return;
      }
      _ipInvalidateScope();
      _ipResetPage();
      _ipEnsureShards();
    });
  }

  // J-3: Timepoint-combination mode toggle (all/any). Visibility managed by
  // _ipSyncControls when timepointCombine selection changes.
  const tpCombineModeSel = document.getElementById("ip-tp-combine-mode");
  if (tpCombineModeSel && !tpCombineModeSel._wired) {
    tpCombineModeSel._wired = true;
    tpCombineModeSel.addEventListener("change", () => {
      IncytrFilter.set({ timepointCombineMode: tpCombineModeSel.value || "all" });
      _ipResetPage();
      _ipRenderTable();
    });
  }

  // if-low-signal is wired once in wireIncytrPanel() to avoid double-wiring.

  // Reset.
  const resetBtn = document.getElementById("ip-reset");
  if (resetBtn) resetBtn.addEventListener("click", () => {
    IncytrFilter.reset();
    const block = _ipBlock();
    if (block) _ipSyncControls(block);
    _ipInvalidateScope();
    _ipResetPage();
    _ipEnsureShards();
  });

  const exportBtn = document.getElementById("ip-export");
  if (exportBtn) exportBtn.addEventListener("click", async () => {
    const prev = exportBtn.textContent;
    exportBtn.disabled = true;
    exportBtn.textContent = "Exporting...";
    try {
      await _ipExportCurrentView();
    } finally {
      exportBtn.disabled = false;
      exportBtn.textContent = prev || "Export CSV";
    }
  });
}

function renderIncytrPathways() {
  const block = _ipBlock();
  const countEl = document.getElementById("ip-count");
  if (!block) {
    if (countEl) countEl.textContent = "No incytr_pathways block in payload.";
    return;
  }
  if (_ipSanitizeFilterState(block)) {
    _ipInvalidateScope();
    _ipResetPage();
  }
  _ipSyncControls(block);
  if (typeof IncytrSidechains !== "undefined") {
    IncytrSidechains.ensureIndex().then(() => _ipRenderTable(), () => _ipRenderTable());
  }
  _ipEnsureShards();
}
