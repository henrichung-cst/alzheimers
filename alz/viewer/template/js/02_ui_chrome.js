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

// Config switch: surface the 5xFAD Incytr results (Heatmap + Pathways) in the
// 5xFAD view. Set false to drop both Incytr tabs from 5xFAD (they stay visible
// in mouse/Song mode).
const SHOW_FIVEXFAD_INCYTR = true;
const _INCYTR_MODES = SHOW_FIVEXFAD_INCYTR ? ["mouse", "fivexfad"] : ["mouse"];

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
    filters: [], requires: [], modes: _INCYTR_MODES,
    wire: () => wireIncytrHeatmap(),
    render: () => renderIncytrHeatmap(),
    rerenderOn: { filters: false, selection: [] },
  },
  incytrpathways: {
    group: "drilldown", label: "Incytr Pathways",
    filters: [], requires: [], modes: _INCYTR_MODES,
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
// tabs declare modes:_INCYTR_MODES and _modeAvailable("fivexfad") controls
// visibility without any post-hoc mutation. While SHOW_FIVEXFAD_INCYTR is
// false the modes list excludes "fivexfad", so this machinery (tissue selector,
// context switching) is dormant in 5xFAD until the toggle is flipped on.
// A Tissue <select> in each incytr tab's toolbar switches the active context.
// ---------------------------------------------------------------------------

// Static after payload load — cached to avoid filtering on every Store sync.
let _f5IncytrCtxCache = null;
function _fivexfadIncytrContexts() {
  if (_f5IncytrCtxCache) return _f5IncytrCtxCache;
  if (typeof ViewerPayload === "undefined") return [];
  _f5IncytrCtxCache = ViewerPayload.contexts().filter(
    c => c.cohort === "fivexfad" && c.capabilities && c.capabilities.incytr);
  return _f5IncytrCtxCache;
}

function _defaultFivexfadIncytrContext() {
  const list = _fivexfadIncytrContexts();
  if (!list.length) return null;
  // If the current context is already a valid 5xFAD incytr context (user
  // previously chose a tissue), stay on it.
  const cur = Store.state.selection.context;
  if (cur && list.some(c => c.id === cur)) return cur;
  return list[0].id;
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
  const lbl = String(ctx.label || ctx.id);
  return lbl.replace(/^5xFAD\s+/i, "") || lbl;
}

// The 5xFAD tissue selector is an in-tab filter: a Tissue <select> at the head
// of each incytr toolbar (#ih-tissue / #ip-tissue). Populated from the 5xFAD
// incytr contexts; dispatches SET_SELECTION context on change. Idempotent.
// Caches the {sel, wrap} pairs so _syncFivexfadTissueToggle avoids DOM lookups.
const _FIVEXFAD_TISSUE_SELECT_IDS = ["ih-tissue", "ip-tissue"];
let _f5TissueSels = null;   // [{sel, wrap}] cached after first wire

function wireFivexfadTissueToggle() {
  const list = _fivexfadIncytrContexts();
  _FIVEXFAD_TISSUE_SELECT_IDS.forEach(id => {
    const sel = document.getElementById(id);
    if (!sel) return;
    if (!sel._built) {
      sel.innerHTML = list.map(c =>
        '<option value="' + c.id + '">' + _escapeHtml(_tissueLabel(c)) + '</option>'
      ).join("");
      sel._built = true;
    }
    if (!sel._wired) {
      sel._wired = true;
      sel.addEventListener("change", () => {
        if (sel.value !== ViewerPayload.activeContext())
          Store.dispatch({ type: "SET_SELECTION", key: "context", value: sel.value });
      });
    }
  });
  _f5TissueSels = _FIVEXFAD_TISSUE_SELECT_IDS
    .map(id => ({ sel: document.getElementById(id), wrap: document.getElementById(id + "-wrap") }))
    .filter(p => p.sel);
}

// Shown only when 5xFAD Incytr is explicitly surfaced. The payload can package
// 5xFAD cortex/hippocampus sidecars while SHOW_FIVEXFAD_INCYTR is false, so
// visibility must follow the UI feature flag, not payload presence alone.
function _syncFivexfadTissueToggle() {
  const show = !!(SHOW_FIVEXFAD_INCYTR && HAS_FIVEXFAD_INCYTR
    && ((Store.state.view && Store.state.view.mode) || "mouse") === "fivexfad");
  const ctx = ViewerPayload.activeContext();
  (_f5TissueSels || []).forEach(({ sel, wrap }) => {
    if (wrap) wrap.hidden = !show;
    if (show && sel.value !== ctx) sel.value = ctx;
  });
}
