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
// visible rows) as Markdown for pasting into an AI chatbot. Reads DOM and
// Store state directly so the export tracks exactly what is rendered.
// ---------------------------------------------------------------------------
function _exportFilterMap(tab) {
  // Returns alphabetized {Label: "value"} for the filters this tab consumes.
  const m = TAB_MANIFEST[tab] || { filters: [] };
  const f = Store.state.filters;
  const out = {};
  if (m.filters.includes("fdr")) out["FDR"] = "< " + f.fdr;
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
    const K = ViewerPayload.kinases();
    chips.push("kinase=" + (ki != null ? K.name[ki] : ("kid:" + sel.kinase)));
  }
  if (sel.backbone != null) chips.push("backbone=BB_" + sel.backbone);
  if (sel.celltype != null) chips.push("celltype=" + (RECEIVERS[sel.celltype] || ("cid:" + sel.celltype)));
  return chips;
}

function _exportDenominator(tab) {
  // Pull whichever subtitle/count element the tab already maintains.
  const ids = { kinase: "ke-count" };
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

function _exportTable(tab) {
  if (tab === "kinase") return _exportTableFromDom("ke-table");
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
  _exportInjectButton("kinase", "#tab-kinase .ke-toolbar");
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

  // Master/detail splitters between the ranked table and the detail panel.
  // One helper drives all table/detail tabs; each persists
  // its left-panel width under its own localStorage key.
  function _wireSplitter(splitterId, storageKey, minW, maxW) {
    const sp = document.getElementById(splitterId);
    if (!sp) return;
    const leftPanel = sp.previousElementSibling;
    if (!leftPanel) return;
    minW = minW || 420; maxW = maxW || 1600;
    try {
      const savedW = localStorage.getItem(storageKey);
      if (savedW) {
        const w = parseInt(savedW, 10);
        if (w >= minW && w <= maxW) leftPanel.style.width = w + "px";
      }
    } catch (_) {}
    sp.addEventListener("mousedown", e => {
      const startX = e.clientX;
      const startW = leftPanel.getBoundingClientRect().width;
      sp.classList.add("dragging");
      document.body.style.cursor = "col-resize";
      document.body.style.userSelect = "none";
      function onMove(ev) {
        const newW = Math.min(maxW, Math.max(minW, startW + (ev.clientX - startX)));
        leftPanel.style.width = newW + "px";
      }
      function onUp() {
        sp.classList.remove("dragging");
        document.body.style.cursor = "";
        document.body.style.userSelect = "";
        try { localStorage.setItem(storageKey, parseInt(leftPanel.style.width, 10)); } catch (_) {}
        document.removeEventListener("mousemove", onMove);
        document.removeEventListener("mouseup", onUp);
      }
      document.addEventListener("mousemove", onMove);
      document.addEventListener("mouseup", onUp);
      e.preventDefault();
    });
  }
  _wireSplitter("ka-splitter", "kinaseTab.leftWidth");
  _wireSplitter("kh-splitter", "humanTab.leftWidth");
  _wireSplitter("kx-splitter", "crosstableTab.leftWidth");
  _wireSplitter("f5-splitter", "fivexfadTab.leftWidth");
}

// ---------------------------------------------------------------------------
// Per-tab manifest — single source of truth for tab group, label, consumed
// header filters, prerequisites, and lifecycle hooks (wire, render,
// rerenderOn, onChange). See MANIFEST.md "Adding a new tab" for the contract.
// ---------------------------------------------------------------------------
const TAB_GROUP_ORDER = ["landscape", "drilldown", "reference"];
const TAB_GROUP_LABELS = {
  landscape: "Landscape",
  drilldown: "Drill-down",
  reference: "Reference",
};

const TAB_MANIFEST = {
  temporalv2: {
    group: "landscape", label: "Temporal v2",
    filters: [], requires: [], modes: ["mouse"],
    wire: () => wireTemporalV2(),
    render: () => renderTemporalV2(),
    rerenderOn: { filters: true, selection: [] },
  },
  crosstable: {
    group: "landscape", label: "Crosstable",
    filters: ["fdr"], requires: [], modes: ["mouse", "human"],
    wire: () => { if (typeof wireKinaseCrosstable === "function") wireKinaseCrosstable(); },
    render: () => { if (typeof renderKinaseCrosstable === "function") renderKinaseCrosstable(); },
    rerenderOn: { filters: true, selection: [] },
  },
  kinasehuman: {
    group: "drilldown", label: "Kinase",
    filters: ["fdr"], requires: [], modes: ["human"],
    wire: () => { if (typeof wireKinaseHuman === "function") wireKinaseHuman(); },
    render: () => { if (typeof renderKinaseHuman === "function") renderKinaseHuman(); },
    rerenderOn: { filters: true, selection: [] },
    onChange: ({ tabChanged, kinaseHumanSelChanged, khid }) => {
      if (!kinaseHumanSelChanged || tabChanged) return false;
      if (typeof updateKinaseHumanSelection === "function")
        updateKinaseHumanSelection(khid);
      return true;
    },
  },
  kinase: {
    group: "drilldown", label: "Kinase",
    modes: ["mouse"],
    filters: ["fdr"], requires: [],
    wire: () => wireKinaseTable(),
    render: () => {
      renderKinaseExplorer();
      const kid = Store.state.selection.kinase;
      if (kid != null) renderKinaseDetail(kid);
    },
    rerenderOn: { filters: true, selection: ["celltype"] },
    // Kinase-selection change skips the full table re-render: only the row
    // highlight and detail panel update here.
    onChange: ({ tabChanged, kinaseSelChanged, kid }) => {
      if (!kinaseSelChanged || tabChanged) return false;
      _updateKinaseRowSelection(kid);
      if (kid != null) renderKinaseDetail(kid);
      return true;
    },
  },
  fivexfadkinase: {
    group: "drilldown", label: "Kinase",
    modes: ["fivexfad"],
    filters: ["fdr"], requires: [{type:"payload", key:"supporting_5xfad",
      message:"5xFAD payload data are not available in this viewer build.",
      cta:"Use another tab"}],
    wire: () => { if (typeof wireFiveXFADKinase === "function") wireFiveXFADKinase(); },
    render: () => { if (typeof renderFiveXFADKinase === "function") renderFiveXFADKinase(); },
    rerenderOn: { filters: true, selection: [] },
  },
  incytrheatmap: {
    group: "landscape", label: "Incytr Heatmap",
    filters: [], requires: [], modes: ["mouse"],
    wire: () => wireIncytrHeatmap(),
    render: () => renderIncytrHeatmap(),
    rerenderOn: { filters: false, selection: [] },
  },
  incytrpathways: {
    group: "drilldown", label: "Incytr Pathways",
    filters: [], requires: [], modes: ["mouse"],
    wire: () => wireIncytrPathways(),
    render: () => renderIncytrPathways(),
    rerenderOn: { filters: false, selection: [] },
  },
  methods: {
    group: "reference", label: "Methods",
    filters: [], requires: [], modes: ["mouse"],
    wire: () => {},
    render: () => {},
    rerenderOn: {},
  },
};

function _activeModeTabs() {
  const mode = (Store.state.view && Store.state.view.mode) || "mouse";
  const out = {};
  for (const id of Object.keys(TAB_MANIFEST)) {
    const m = TAB_MANIFEST[id];
    const modes = m.modes || ["mouse"];
    if (modes.includes(mode)) out[id] = m;
  }
  return out;
}

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
