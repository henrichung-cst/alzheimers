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
const _IP_ROW_CAP = 1000;

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
  rows:        null,         // concatenated rows from currently-loaded shards
  loadedKey:   null,         // sig string of pairs currently loaded
  loading:     false,
  loadError:   null,
  openKeys:    new Set(),    // keys of rows whose per-node FC detail is expanded
};

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
  // Returns [{sender, receiver}] to load. Honors the multiselect filters; if
  // both are empty, defaults to ALL present pairs ("show all pathways").
  const f = IncytrFilter.get();
  if (f.pair) return [f.pair];
  const sIn = new Set(f.senderIn || []);
  const rIn = new Set(f.receiverIn || []);
  const out = [];
  for (const [s, r] of block.slice_index.present) {
    if (sIn.size && !sIn.has(s)) continue;
    if (rIn.size && !rIn.has(r)) continue;
    out.push({ sender: s, receiver: r });
  }
  return out;
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

  // Numeric sliders.
  const set = (id, v) => {
    const el = document.getElementById(id);
    if (el) el.value = (v == null || !isFinite(v)) ? "" : v;
  };
  set("ip-slider-p",   f.sliderP);
  set("ip-slider-pds", f.sliderPds);
}

function _ipInvalidateScope() {
  _ipRuntime.rows = null;
  _ipRuntime.loadedKey = null;
  _ipRuntime.openKeys = new Set();
}

// ---- shard loading ----

async function _ipEnsureShards() {
  const block = _ipBlock();
  if (!block) return;
  const pairs = _ipPairsInScope(block);
  const sig = _ipScopeSig(pairs);
  if (_ipRuntime.loadedKey === sig) {
    _ipRenderTable();
    return;
  }
  _ipRuntime.loading = true;
  _ipRuntime.loadError = null;
  _ipRuntime.rows = null;
  _ipRenderTable();
  try {
    // Parallel fetch — for "show all" the scope is 349 pairs, so sequential
    // awaits would be O(seconds). Browser concurrency limits already gate this.
    const perPair = await Promise.all(pairs.map(p =>
      SliceCache.loadIncytrShard(p.sender, p.receiver).then(rows => {
        // Stamp sender/receiver so multi-pair queries still identify the
        // originating cell-type pair in the table.
        for (const r of rows) { r._sender = p.sender; r._receiver = p.receiver; }
        return rows;
      })
    ));
    // Resolve race: only commit if the scope hasn't changed mid-fetch.
    const newSig = _ipScopeSig(_ipPairsInScope(block));
    if (newSig !== sig) return;
    const all = [];
    for (const arr of perPair) all.push(...arr);
    _ipRuntime.rows = all;
    _ipRuntime.loadedKey = sig;
  } catch (e) {
    _ipRuntime.loadError = String(e.message || e);
    console.error("incytr shards load failed", e);
  } finally {
    _ipRuntime.loading = false;
    _ipRenderTable();
  }
}

// ---- row filtering + sort ----

function _ipFilterRows() {
  if (!_ipRuntime.rows) return [];
  const f = IncytrFilter.get();
  const diseaseSet   = new Set(f.disease   || []);
  const timeSet      = new Set(f.timepoint || []);
  const out = [];
  for (const r of _ipRuntime.rows) {
    if (diseaseSet.size || timeSet.size) {
      const [d, t] = (r.contrast || "").split("_");
      if (diseaseSet.size && !diseaseSet.has(d)) continue;
      if (timeSet.size    && !timeSet.has(t))    continue;
    }
    if (f.sliderP   != null && !(r.pvalue       <  f.sliderP))   continue;
    if (f.sliderPds != null && !(Math.abs(r.PDS) >= f.sliderPds)) continue;
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
  if (!pairs.length) {
    countEl.textContent = "No (sender, receiver) pairs match the current selection.";
    wrap.innerHTML = '<div class="muted" style="padding:16px;">Try clearing or widening the sender / receiver filters.</div>';
    return;
  }
  if (_ipRuntime.loading) {
    countEl.textContent = `Loading ${pairs.length} shard${pairs.length === 1 ? "" : "s"} in parallel…`;
    wrap.innerHTML = '<div class="muted" style="padding:16px;">Fetching shards…</div>';
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
    countEl.textContent = "No rows in the selected shard(s).";
    wrap.innerHTML = '<div class="muted" style="padding:16px;">Empty (likely an empty-DEG cell type).</div>';
    return;
  }
  const filtered = _ipFilterRows();
  const total = _ipRuntime.rows.length;
  const shown = Math.min(filtered.length, _IP_ROW_CAP);
  const f = IncytrFilter.get();
  countEl.textContent =
    `${filtered.length.toLocaleString()} rows pass filters `
    + `(of ${total.toLocaleString()} loaded from ${pairs.length} pair${pairs.length === 1 ? "" : "s"}).`
    + (filtered.length > _IP_ROW_CAP
        ? ` Showing top ${shown.toLocaleString()} by ${f.sortKey}.`
        : "");

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
  ];
  // Leading expander column header is non-sortable.
  const thead =
    `<th style="width:24px;" title="Toggle per-node log₂ fold-change detail (sc / pr / ps / py)."></th>`
    + cols.map(c => {
        const on = (f.sortKey === c.key);
        const arrow = on ? (f.sortDir > 0 ? " ▲" : " ▼") : "";
        const tip = c.tip ? ` title="${_escapeHtml(c.tip)}"` : "";
        return `<th data-ip-sort="${c.key}"${tip}>${_escapeHtml(c.label)}${arrow}</th>`;
      }).join("");
  const visible = filtered.slice(0, _IP_ROW_CAP);
  const totalCols = cols.length + 1;
  const tbody = visible.map((r, idx) => {
    const rk = _ipRowKey(r);
    const isOpen = _ipRuntime.openKeys.has(rk);
    const toggle = `<td style="text-align:center;cursor:pointer;" `
      + `data-ip-toggle="${idx}" title="${isOpen ? "Hide" : "Show"} per-node fold-change detail">`
      + `${isOpen ? "▾" : "▸"}</td>`;
    const cells = cols.map(c => {
      const v = r[c.key];
      if (c.numeric) return `<td style="text-align:right;">${_ipFmtNum(v, c.digits)}</td>`;
      if (c.labelKey) return `<td>${_ipNodeCell(v, r[c.labelKey])}</td>`;
      return `<td>${_escapeHtml(v == null ? "" : v)}</td>`;
    }).join("");
    let html = `<tr data-ip-row="${idx}">${toggle}${cells}</tr>`;
    if (isOpen) {
      html += `<tr class="ip-detail-row"><td></td><td colspan="${cols.length}" `
        + `style="padding:8px 12px;background:#fafafa;">`
        + _ipRenderDetail(r) + `</td></tr>`;
    }
    return html;
  }).join("");
  wrap.innerHTML = `<div class="ke-table-wrap"><table class="data-table" id="ip-table">`
    + `<thead><tr>${thead}</tr></thead><tbody>${tbody}</tbody></table></div>`;
  const head = wrap.querySelector("#ip-table thead");
  if (head) head.addEventListener("click", ev => {
    const th = ev.target.closest("th[data-ip-sort]");
    if (!th) return;
    const k = th.dataset.ipSort;
    if (f.sortKey === k) IncytrFilter.set({ sortDir: -1 * f.sortDir });
    else IncytrFilter.set({ sortKey: k, sortDir: (k === "pvalue" ? 1 : -1) });
    _ipRenderTable();
  });
  const body = wrap.querySelector("#ip-table tbody");
  if (body) body.addEventListener("click", ev => {
    const cell = ev.target.closest("td[data-ip-toggle]");
    if (!cell) return;
    const idx = +cell.dataset.ipToggle;
    const r = visible[idx];
    if (!r) return;
    const rk = _ipRowKey(r);
    if (_ipRuntime.openKeys.has(rk)) _ipRuntime.openKeys.delete(rk);
    else _ipRuntime.openKeys.add(rk);
    _ipRenderTable();
  });
}

function _ipRenderDetail(r) {
  // 4×7 matrix of per-node fold-changes (single-cell + 3 omics layers × 2
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

function wireIncytrPathways() {
  // Numeric sliders.
  const wireSlider = (id, key) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener("input", () => {
      const raw = el.value === "" ? null : parseFloat(el.value);
      IncytrFilter.set({ [key]: (raw != null && isFinite(raw)) ? raw : null });
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
