function renderHowToDrawer(tab) {
  const drawer = document.getElementById("howto-drawer");
  if (!drawer) return;
  const guide = TAB_GUIDE[tab];
  const manifest = TAB_MANIFEST[tab];
  const label = (manifest && manifest.label) || tab;
  if (!guide) {
    drawer.innerHTML = `<h3>${_escapeHtml(label)}</h3><p class="muted">No guide for this tab.</p>`;
    return;
  }
  const parts = [`<h3>${_escapeHtml(label)}</h3>`];
  if (guide.preamble) {
    parts.push(`<p class="ht-preamble">${_escapeHtml(guide.preamble)}</p>`);
  }
  const isNewSchema = guide.method || guide.shows || guide.howTo || guide.toggles;
  if (isNewSchema) {
    if (guide.method) {
      const paras = Array.isArray(guide.method) ? guide.method : [guide.method];
      parts.push(`<h4>How it was generated</h4>`);
      paras.forEach(p => parts.push(`<p>${_escapeHtml(p)}</p>`));
    }
    if (guide.shows) {
      parts.push(`<h4>What it shows</h4>`);
      if (typeof guide.shows === "string") {
        parts.push(`<p>${_escapeHtml(guide.shows)}</p>`);
      } else {
        if (guide.shows.lead) {
          const leads = Array.isArray(guide.shows.lead) ? guide.shows.lead : [guide.shows.lead];
          leads.forEach(l => parts.push(`<p>${_escapeHtml(l)}</p>`));
        }
        if (guide.shows.bullets && guide.shows.bullets.length) {
          const lis = guide.shows.bullets.map(b => `<li>${_escapeHtml(b)}</li>`).join("");
          parts.push(`<ul>${lis}</ul>`);
        }
      }
    }
    if (guide.howTo) {
      parts.push(`<h4>How to read it</h4><p>${_escapeHtml(guide.howTo)}</p>`);
    }
    if (guide.conclusions && guide.conclusions.length) {
      const cl = guide.conclusions.map(c => `<div class="ht-conclusion">${_escapeHtml(c)}</div>`).join("");
      parts.push(`<h4>Conclusions</h4>${cl}`);
    }
    if (guide.toggles && guide.toggles.length) {
      const lis = guide.toggles.map(t => {
        const name = t.name ? `<strong>${_escapeHtml(t.name)}</strong>` : "";
        const desc = t.desc ? ` — ${_escapeHtml(t.desc)}` : "";
        return `<li>${name}${desc}</li>`;
      }).join("");
      parts.push(`<h4>Adjustable toggles</h4><ul class="ht-toggles">${lis}</ul>`);
    }
  } else {
    if (guide.purpose) {
      parts.push(`<h4>What this tab answers</h4><p>${_escapeHtml(guide.purpose)}</p>`);
    }
    if (guide.primary) {
      parts.push(`<h4>How to read it</h4><p>${_escapeHtml(guide.primary)}</p>`);
    }
    if (guide.cues && guide.cues.length) {
      const cueRows = guide.cues.map(cue => {
        const m = METRIC_DEFS[cue.metric];
        const name = m ? m.label : cue.metric;
        const hr = m && m.howToRead ? m.howToRead : (m ? m.short : "");
        const when = cue.when ? ` <span class="ht-when">— ${_escapeHtml(cue.when)}</span>` : "";
        return `<div class="ht-cue"><span class="ht-metric">${_escapeHtml(name)}</span>${when}<br><span class="ht-when">${_escapeHtml(hr)}</span></div>`;
      }).join("");
      parts.push(`<h4>Metrics to watch</h4>${cueRows}`);
    }
    if (guide.conclusions && guide.conclusions.length) {
      const cl = guide.conclusions.map(c => `<div class="ht-conclusion">${_escapeHtml(c)}</div>`).join("");
      parts.push(`<h4>Conclusions</h4>${cl}`);
    }
  }
  drawer.innerHTML = parts.join("");
}

// ---------------------------------------------------------------------------
// View export — copy the current on-screen view (filters, methods preamble,
// visible rows) as Markdown for pasting into an AI chatbot. Scope: kinase,
// pathway, temporal, additivity, senders, graph. Reads DOM and Store state
// directly so the export tracks exactly what is rendered.
// ---------------------------------------------------------------------------
const EXPORT_TABS = ["kinase","pathway","temporal","additivity","senders","graph"];

function _exportFilterMap(tab) {
  // Returns alphabetized {Label: "value"} for the filters this tab consumes,
  // plus any tab-local thresholds.
  const m = TAB_MANIFEST[tab] || { filters: [] };
  const f = Store.state.filters;
  const out = {};
  if (m.filters.includes("receiver")) out["Receiver"] = f.receiver || "ALL";
  if (m.filters.includes("pathwayEvidence")) out["Support"] = f.pathwayEvidence || "any";
  if (m.filters.includes("fdr")) out["FDR"] = "< " + f.fdr;
  if (f.tpdsSig && f.tpdsSig !== "OFF") out["TPDS p"] = "< " + f.tpdsSig;
  // Tab-local thresholds (read from DOM so they reflect what the user set).
  const grab = (id) => { const e = document.getElementById(id); return e ? e.value : null; };
  if (tab === "temporal") {
    out["Mode"] = Store.state.view.temporalLevel;
    const v = grab("tm-score-min"); if (v != null && Number(v) > 0) out["|TPDS| min"] = v;
    const tiss = grab("tm-tissue"); if (tiss && tiss !== "ALL") out["Tissue"] = tiss;
    const met = grab("tm-metric"); if (met && Store.state.view.temporalLevel === "backbone") out["Metric"] = met;
  }
  if (tab === "additivity") {
    out["Mode"] = grab("add-level") || "kinase";
    out["Timepoint"] = grab("add-tp") || "ALL";
    const v = grab("add-score-min"); if (v != null && Number(v) > 0) out["Score min"] = v;
  }
  if (tab === "pathway") {
    const v = grab("pe-tpds-min"); if (v != null && Number(v) > 0) out["|TPDS| min"] = v;
    out["Trajectory"] = (typeof peTrajectory !== "undefined" && peTrajectory) ? peTrajectory : "all";
  }
  if (tab === "senders") {
    out["Compare"] = grab("sm-axis") || "timepoint";
    out["Anchor"] = grab("sm-anchor") || "";
    out["Mode"] = grab("sm-mode") || "count";
  }
  if (tab === "graph") {
    out["Genotype"] = grab("graph-genotype") || "";
    out["Timepoint"] = grab("graph-timepoint") || "";
    out["Layout"] = grab("graph-layout") || "";
    out["Min degree"] = grab("graph-min-degree") || "1";
    const v = grab("graph-tpds-min"); if (v != null && Number(v) > 0) out["|TPDS| min"] = v;
    const top = grab("graph-top-n"); if (top) out["Max edges"] = top;
  }
  // Sort alphabetically by key.
  const sorted = {};
  Object.keys(out).sort().forEach(k => { sorted[k] = out[k]; });
  return sorted;
}

function _exportSelectionChips() {
  const sel = Store.state.selection;
  const chips = [];
  if (sel.kinase != null) {
    _ensureKinaseIdx();
    const ki = _kinaseIdxById.get(sel.kinase);
    chips.push("kinase=" + (ki != null ? PAYLOAD.kinases.name[ki] : ("kid:" + sel.kinase)));
  }
  if (sel.backbone != null) chips.push("backbone=BB_" + sel.backbone);
  if (sel.celltype != null) chips.push("celltype=" + (RECEIVERS[sel.celltype] || ("cid:" + sel.celltype)));
  return chips;
}

function _exportDenominator(tab) {
  // Pull whichever subtitle/count element the tab already maintains.
  const ids = {
    kinase: "ke-count",
    pathway: "pe-count",
    temporal: "tm-subtitle",
    additivity: "add-subtitle",
    senders: "sm-subtitle",
    graph: "graph-stats",
  };
  const el = document.getElementById(ids[tab]);
  return el ? el.textContent.trim() : "";
}

function _exportMethods(tab) {
  const g = TAB_GUIDE[tab];
  if (!g) return "";
  const lines = [];
  if (Array.isArray(g.method)) g.method.forEach(p => lines.push(p));
  if (g.shows && g.shows.lead) {
    if (Array.isArray(g.shows.lead)) g.shows.lead.forEach(p => lines.push(p));
    else lines.push(g.shows.lead);
  }
  return lines.join("\n\n");
}

function _exportTableFromDom(tableId) {
  const tbl = document.getElementById(tableId);
  if (!tbl) return null;
  const headers = Array.from(tbl.querySelectorAll("thead th"))
    .map(th => th.textContent.replace(/[ ▲▼]+$/, "").trim());
  const rows = Array.from(tbl.querySelectorAll("tbody tr"))
    .map(tr => Array.from(tr.children).map(td => td.textContent.trim()));
  return { headers, rows };
}

function _exportTableFromPlotly(elId) {
  const el = document.getElementById(elId);
  if (!el || !el.data || !el.data.length) return null;
  // Generic flattening: for each trace, emit (trace_name, x, y).
  const headers = ["Series", "X", "Y"];
  const rows = [];
  for (const tr of el.data) {
    const name = tr.name || "";
    const xs = tr.x || [];
    const ys = tr.y || [];
    const n = Math.max(xs.length, ys.length);
    for (let i = 0; i < n; i++) {
      const x = xs[i] != null ? String(xs[i]) : "";
      const y = ys[i] != null ? (typeof ys[i] === "number" ? ys[i].toFixed(3) : String(ys[i])) : "";
      rows.push([name, x, y]);
    }
  }
  return { headers, rows };
}

function _exportTableFromHeatmaps(elIds) {
  // Three Plotly heatmaps for senders. Each carries z (matrix), x (receivers),
  // y (senders), name (panel label). Flatten cells with any non-null value.
  const headers = ["Panel", "Sender", "Receiver", "Value"];
  const rows = [];
  for (const elId of elIds) {
    const el = document.getElementById(elId);
    if (!el || !el.data || !el.data.length) continue;
    const tr = el.data[0];
    const z = tr.z || [];
    const xs = tr.x || [];
    const ys = tr.y || [];
    const panel = (el.layout && el.layout.title && el.layout.title.text) || tr.name || elId;
    for (let i = 0; i < z.length; i++) {
      for (let j = 0; j < (z[i] || []).length; j++) {
        const v = z[i][j];
        if (v == null) continue;
        rows.push([String(panel), String(ys[i] || i), String(xs[j] || j), typeof v === "number" ? v.toFixed(3) : String(v)]);
      }
    }
  }
  return { headers, rows };
}

function _exportTableFromGraph() {
  if (!_cyInstance) return null;
  const headers = ["Type", "Id", "Label", "Degree/Weight", "Extra"];
  const rows = [];
  _cyInstance.nodes(":visible").forEach(n => {
    rows.push(["node", n.id(), n.data("label") || "", String(n.degree(false)), n.data("kind") || ""]);
  });
  _cyInstance.edges(":visible").forEach(e => {
    const w = e.data("weight");
    rows.push(["edge", e.id(),
      (e.source().data("label") || e.source().id()) + " → " + (e.target().data("label") || e.target().id()),
      w == null ? "" : (typeof w === "number" ? w.toFixed(3) : String(w)),
      e.data("genotype") || ""]);
  });
  return { headers, rows };
}

function _exportTable(tab) {
  if (tab === "kinase")     return _exportTableFromDom("ke-table");
  if (tab === "pathway")    return _exportTableFromDom("pe-table");
  if (tab === "temporal")   return _exportTableFromPlotly("temporal-plot");
  if (tab === "additivity") return _exportTableFromPlotly("add-plot");
  if (tab === "senders")    return _exportTableFromHeatmaps(["sender-matrix-plot-0","sender-matrix-plot-1","sender-matrix-plot-2"]);
  if (tab === "graph")      return _exportTableFromGraph();
  return null;
}

function _exportEscapeMd(s) {
  return String(s).replace(/\|/g, "\\|").replace(/\n/g, " ");
}

function _exportRenderTable(table) {
  if (!table || !table.headers || !table.headers.length) return "_(no table data captured for this view)_";
  const head = "| " + table.headers.map(_exportEscapeMd).join(" | ") + " |";
  const sep  = "| " + table.headers.map(() => "---").join(" | ") + " |";
  const body = table.rows.map(r => "| " + r.map(_exportEscapeMd).join(" | ") + " |").join("\n");
  return [head, sep, body].join("\n");
}

function _exportAssemble(tab) {
  const label = (TAB_MANIFEST[tab] && TAB_MANIFEST[tab].label) || tab;
  const filters = _exportFilterMap(tab);
  const sels = _exportSelectionChips();
  const denom = _exportDenominator(tab);
  const methods = _exportMethods(tab);
  const table = _exportTable(tab);

  const lines = [];
  lines.push("# Unified Viewer export — " + label + " tab");
  lines.push("");
  lines.push("## Active view");
  Object.entries(filters).forEach(([k, v]) => lines.push("- **" + k + ":** " + v));
  if (sels.length) lines.push("- **Selection:** " + sels.join(", "));
  if (denom) {
    lines.push("");
    lines.push("**Denominator:** " + denom);
  }
  lines.push("");
  if (methods) {
    lines.push("## How this view was generated");
    lines.push("");
    lines.push(methods);
    lines.push("");
  }
  lines.push("## Visible rows");
  lines.push("");
  lines.push(_exportRenderTable(table));
  lines.push("");
  lines.push("_Generated by build_unified_viewer.py · view-scoped export. For full underlying data see outputs/reports/._");
  return lines.join("\n");
}

function _exportFlash(btn, msg) {
  const orig = btn.textContent;
  btn.textContent = msg;
  btn.disabled = true;
  setTimeout(() => { btn.textContent = orig; btn.disabled = false; }, 1400);
}

function _exportDownload(md, tab) {
  const blob = new Blob([md], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "viewer_" + tab + "_view.md";
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function exportTab(tab, mode, btn) {
  try {
    const md = _exportAssemble(tab);
    if (mode === "download") {
      _exportDownload(md, tab);
      if (btn) _exportFlash(btn, "Downloaded ✓");
      return;
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(md);
      if (btn) _exportFlash(btn, "Copied ✓");
    } else {
      _exportDownload(md, tab);
      if (btn) _exportFlash(btn, "Downloaded ✓");
    }
  } catch (e) {
    console.error("export failed", e);
    if (btn) _exportFlash(btn, "Failed — see console");
  }
}

function _exportInjectButton(tab, hostSelector) {
  const host = document.querySelector(hostSelector);
  if (!host || host.querySelector(".export-view-btn")) return;
  const wrap = document.createElement("span");
  wrap.className = "export-view-wrap";
  wrap.style.marginLeft = "auto";
  wrap.innerHTML = `<button type="button" class="chip export-view-btn" title="Copy this view as Markdown for an AI chatbot">⧉ Export view</button>` +
                   `<button type="button" class="chip export-view-dl" title="Download this view as a .md file" style="margin-left:4px;">⤓</button>`;
  host.appendChild(wrap);
  wrap.querySelector(".export-view-btn").addEventListener("click", e => exportTab(tab, "clipboard", e.currentTarget));
  wrap.querySelector(".export-view-dl").addEventListener("click", e => exportTab(tab, "download", e.currentTarget));
}

function wireExportButtons() {
  _exportInjectButton("kinase",     "#tab-kinase .ke-toolbar");
  _exportInjectButton("pathway",    "#tab-pathway .ke-toolbar");
  _exportInjectButton("temporal",   "#tab-temporal .detail-chips");
  _exportInjectButton("additivity", "#tab-additivity .detail-chips");
  _exportInjectButton("senders",    "#tab-senders .detail-chips");
  _exportInjectButton("graph",      "#graph-controls");
}

function wireDrawerResizer() {
  const resizer = document.getElementById("drawer-resizer");
  const drawer  = document.getElementById("howto-drawer");
  const shell   = document.getElementById("content-shell");
  if (!resizer || !drawer || !shell) return;

  // Restore saved width.
  try {
    const saved = localStorage.getItem("howtoDrawer.width");
    if (saved) { const w = parseInt(saved, 10); if (w >= 180 && w <= 800) drawer.style.width = w + "px"; }
  } catch (_) {}

  let startX = 0, startW = 0;
  resizer.addEventListener("mousedown", e => {
    startX = e.clientX;
    startW = drawer.getBoundingClientRect().width;
    resizer.classList.add("dragging");
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";

    function onMove(ev) {
      const delta = startX - ev.clientX;   // dragging left = narrower main, wider drawer
      const newW = Math.min(600, Math.max(180, startW + delta));
      drawer.style.width = newW + "px";
    }
    function onUp() {
      resizer.classList.remove("dragging");
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
      try { localStorage.setItem("howtoDrawer.width", parseInt(drawer.style.width, 10)); } catch (_) {}
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    }
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    e.preventDefault();
  });

  // Calibrate --shell-top so content-shell height fills the viewport below the header.
  function calibrateShellHeight() {
    const header = document.querySelector("header#app-header");
    const nav    = document.querySelector("nav#tab-bar");
    let top = 0;
    if (header) top += header.getBoundingClientRect().height;
    if (nav)    top += nav.getBoundingClientRect().height;
    shell.style.setProperty("--shell-top", top + "px");
    shell.style.height = "calc(100vh - " + top + "px)";
  }
  calibrateShellHeight();
  window.addEventListener("resize", calibrateShellHeight);

  // Drawer collapse toggle. Default: collapsed. Persisted in localStorage.
  const toggleBtn = document.getElementById("howto-drawer-toggle");
  if (toggleBtn) {
    let expanded = false;
    try { expanded = localStorage.getItem("howtoDrawer.expanded") === "1"; } catch (_) {}
    function applyExpanded(e) {
      shell.classList.toggle("drawer-collapsed", !e);
      toggleBtn.classList.toggle("expanded", e);
      toggleBtn.setAttribute("aria-expanded", e ? "true" : "false");
    }
    applyExpanded(expanded);
    toggleBtn.addEventListener("click", () => {
      expanded = !expanded;
      applyExpanded(expanded);
      try { localStorage.setItem("howtoDrawer.expanded", expanded ? "1" : "0"); } catch (_) {}
    });
  }

  // Kinase-tab splitter between ranked-kinase table and audit detail.
  const kaSplitter = document.getElementById("ka-splitter");
  if (kaSplitter) {
    const leftPanel = kaSplitter.previousElementSibling;
    if (leftPanel) {
      try {
        const savedW = localStorage.getItem("kinaseTab.leftWidth");
        if (savedW) {
          const w = parseInt(savedW, 10);
          if (w >= 280 && w <= 1200) leftPanel.style.width = w + "px";
        }
      } catch (_) {}
      let kStartX = 0, kStartW = 0;
      kaSplitter.addEventListener("mousedown", e => {
        kStartX = e.clientX;
        kStartW = leftPanel.getBoundingClientRect().width;
        kaSplitter.classList.add("dragging");
        document.body.style.cursor = "col-resize";
        document.body.style.userSelect = "none";
        function onMove(ev) {
          const newW = Math.min(1200, Math.max(280, kStartW + (ev.clientX - kStartX)));
          leftPanel.style.width = newW + "px";
        }
        function onUp() {
          kaSplitter.classList.remove("dragging");
          document.body.style.cursor = "";
          document.body.style.userSelect = "";
          try { localStorage.setItem("kinaseTab.leftWidth", parseInt(leftPanel.style.width, 10)); } catch (_) {}
          document.removeEventListener("mousemove", onMove);
          document.removeEventListener("mouseup", onUp);
        }
        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);
        e.preventDefault();
      });
    }
  }
}

// ---------------------------------------------------------------------------
// Per-tab manifest — declares which filters each tab consumes and what
// prerequisites must be met before content can render. Single source of
// truth for the filter-bar dim/hide logic and prerequisite empty states.
// ---------------------------------------------------------------------------
const TAB_MANIFEST = {
  signal:     { group:"landscape", label:"Signal Map",
                filters:[], requires:[] },
  senders:    { group:"landscape", label:"Sender×Receiver",
                filters:[], requires:[] },
  temporal:   { group:"landscape", label:"Temporal",
                filters:["fdr","pathwayEvidence","receiver"],
                requires:[] },
  additivity: { group:"landscape", label:"Additivity",
                filters:["fdr","receiver","pathwayEvidence"],
                requires:[] },
  kinase:     { group:"drilldown", label:"Kinase",
                filters:["fdr"], requires:[] },
  pathway:    { group:"drilldown", label:"Pathway",
                filters:["receiver","pathwayEvidence","fdr"],
                requires:[] },
  graph:      { group:"drilldown", label:"Graph",
                filters:["receiver","pathwayEvidence"],
                requires:[] },
  methods:    { group:"reference", label:"Methods",
                filters:[], requires:[] },
};

function syncFilterBarToTab(tab) {
  const manifest = TAB_MANIFEST[tab];
  const consumed = new Set(manifest ? manifest.filters : []);
  document.querySelectorAll(".filter-label").forEach(lab => {
    const key = lab.dataset.filter;
    lab.hidden = !consumed.has(key);
  });
}

// ---------------------------------------------------------------------------
// Unified prerequisite empty state — replaces bespoke placeholders for
// tabs that need a prior selection or filter to be set. Reads the active
// tab's manifest.requires[] and renders an actionable card if any
// requirement is unmet. Returns true if a prerequisite was rendered (caller
// should bail).
// ---------------------------------------------------------------------------
