function wireViewerSplitters() {
  wireTableDetailSplitter("ka-splitter", "kinaseTab.leftWidth");
  wireTableDetailSplitter("kh-splitter", "humanTab.leftWidth");
  wireTableDetailSplitter("kx-splitter", "crosstableTab.leftWidth");
  wireTableDetailSplitter("f5-splitter", "fivexfadTab.leftWidth");
}

// ---------------------------------------------------------------------------
// Per-tab manifest — single source of truth for tab group, label, consumed
// header filters, prerequisites, and lifecycle hooks (wire, render,
// rerenderOn, onChange). See MANIFEST.md "Adding a new tab" for the contract.
// ---------------------------------------------------------------------------
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
    rerenderOn: { filters: true, selection: ["kinaseFiveXFAD"] },
    onChange: ({ tabChanged, kinaseFiveXFADSelChanged, f5key }) => {
      if (!kinaseFiveXFADSelChanged || tabChanged) return false;
      if (typeof updateFiveXFADKinaseSelection === "function")
        updateFiveXFADKinaseSelection(f5key);
      return true;
    },
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

// ---------------------------------------------------------------------------
// 5xFAD incytr surfacing. The cortex/hippocampus incytr lives under
// incytr_pathways.by_context (context ids fivexfad_cortex / fivexfad_hippocampus)
// and is reached inside the 5xFAD mode: the Incytr Heatmap + Incytr Pathways
// tabs are mode-shared (mouse → Song AD, fivexfad → 5xFAD), and a Cortex/
// Hippocampus toggle picks the tissue context. See body.html
// #fivexfad-tissue-toggle and boot.js (mode→context wiring).
// ---------------------------------------------------------------------------
const _INCYTR_TAB_IDS = new Set(["incytrheatmap", "incytrpathways"]);
let _fivexfadIncytrContextPref = null;   // remembers the last tissue choice

function _fivexfadIncytrContexts() {
  if (typeof ViewerPayload === "undefined") return [];
  return ViewerPayload.contexts().filter(
    c => c.cohort === "fivexfad" && c.capabilities && c.capabilities.incytr);
}

function _defaultFivexfadIncytrContext() {
  const list = _fivexfadIncytrContexts();
  if (!list.length) return null;
  if (_fivexfadIncytrContextPref
      && list.some(c => c.id === _fivexfadIncytrContextPref))
    return _fivexfadIncytrContextPref;
  return list[0].id;
}

// Add the incytr tabs to the 5xFAD mode once, after payload load reveals that
// 5xFAD incytr data exists. Mutates the shared TAB_MANIFEST in place.
function _enableFivexfadIncytrTabs() {
  if (!HAS_FIVEXFAD_INCYTR) return;
  _INCYTR_TAB_IDS.forEach(id => {
    const m = TAB_MANIFEST[id];
    if (m && Array.isArray(m.modes) && !m.modes.includes("fivexfad"))
      m.modes = m.modes.concat("fivexfad");
  });
}

// Mode → incytr context: 5xFAD mode points incytr at its tissue context; any
// other mode restores the cohort default (Song AD). Called from boot.js on
// mode change.
function _applyModeContext(mode) {
  if (typeof ViewerPayload === "undefined") return;
  const want = (mode === "fivexfad")
    ? _defaultFivexfadIncytrContext()
    : ViewerPayload.defaultContext();
  if (want && Store.state.selection.context !== want)
    Store.dispatch({ type: "SET_SELECTION", key: "context", value: want });
}

function _tissueLabel(ctx) {
  // "5xFAD Cortex" → "Cortex"; fall back to the raw label.
  const lbl = String(ctx.label || ctx.id);
  return lbl.replace(/^5xFAD\s+/i, "") || lbl;
}

// Build the Cortex/Hippocampus buttons from the available 5xFAD incytr
// contexts and wire click → SET_SELECTION context. Idempotent.
function wireFivexfadTissueToggle() {
  const wrap = document.getElementById("fivexfad-tissue-toggle");
  if (!wrap) return;
  if (!wrap._built) {
    const list = _fivexfadIncytrContexts();
    wrap.innerHTML = list.map((c, i) =>
      '<button class="mode-btn" role="tab" data-context="' + c.id + '"'
      + ' aria-selected="' + (i === 0 ? "true" : "false") + '">'
      + _escapeHtml(_tissueLabel(c)) + '</button>'
    ).join("");
    wrap._built = true;
  }
  wrap.querySelectorAll("button.mode-btn").forEach(btn => {
    if (btn._wired) return;
    btn._wired = true;
    btn.addEventListener("click", () => {
      const ctx = btn.dataset.context;
      _fivexfadIncytrContextPref = ctx;
      if (ctx !== ViewerPayload.activeContext())
        Store.dispatch({ type: "SET_SELECTION", key: "context", value: ctx });
    });
  });
}

// Visible only in 5xFAD mode while an incytr tab is active (tissue is an
// incytr-only axis; the 5xFAD kinase tab is tissue-pooled). Highlights the
// button matching the active context.
function _syncFivexfadTissueToggle() {
  const wrap = document.getElementById("fivexfad-tissue-toggle");
  if (!wrap) return;
  const mode = (Store.state.view && Store.state.view.mode) || "mouse";
  const tab = Store.state.view && Store.state.view.activeTab;
  const show = HAS_FIVEXFAD_INCYTR && mode === "fivexfad"
    && _INCYTR_TAB_IDS.has(tab);
  wrap.hidden = !show;
  if (!show) return;
  const ctx = ViewerPayload.activeContext();
  wrap.querySelectorAll("button.mode-btn").forEach(btn => {
    const on = btn.dataset.context === ctx;
    btn.classList.toggle("active", on);
    btn.setAttribute("aria-selected", on ? "true" : "false");
  });
}
