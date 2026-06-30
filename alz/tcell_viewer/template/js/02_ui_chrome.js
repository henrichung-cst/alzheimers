function wireViewerSplitters() {
  wireTableDetailSplitter("ka-splitter", "kinaseTab.leftWidth");
}

// ---------------------------------------------------------------------------
// Per-tab manifest — single source of truth for tab group, label, consumed
// header filters, prerequisites, and lifecycle hooks (wire, render,
// rerenderOn, onChange). See MANIFEST.md "Adding a new tab" for the contract.
// ---------------------------------------------------------------------------
// Kinase tabs gate on context:donor1 — donor2 has no IMAC/MEA outputs.
// Incytr tabs render for both contexts; the context selection picks contrast
// vocab and shard prefix.
const _KINASE_REQ_DONOR1 = [{
  type:"selection", key:"context", equal:"donor1",
  message:"Kinase MEA is donor1-only — donor2 has no IMAC.",
  cta:"Switch to donor 1",
  setSelection:{ key:"context", value:"donor1" },
}];

const TAB_MANIFEST = {
  temporal: {
    group: "landscape", label: "Temporal",
    filters: [], requires: [], modes: ["mouse"],
    wire: () => wireTemporal(),
    render: () => renderTemporal(),
    rerenderOn: { filters: false, selection: [] },
  },
  kinase: {
    group: "drilldown", label: "Kinase",
    modes: ["mouse"],
    filters: ["fdr"], requires: _KINASE_REQ_DONOR1,
    wire: () => wireKinaseTable(),
    render: () => {
      renderKinaseExplorer();
      const kid = Store.state.selection.kinase;
      if (kid != null) renderKinaseDetail(kid);
    },
    rerenderOn: { filters: true, selection: ["celltype"] },
    onChange: ({ tabChanged, kinaseSelChanged, kid }) => {
      if (!kinaseSelChanged || tabChanged) return false;
      _updateKinaseRowSelection(kid);
      if (kid != null) renderKinaseDetail(kid);
      return true;
    },
  },
  incytr: {
    group: "landscape", label: "Incytr",
    filters: [], requires: [], modes: ["mouse"],
    wire: () => { wireIncytrHeatmap(); wireIncytrPathways(); wireIncytrPanel(); },
    render: () => _renderIncytrTab(),
    rerenderOn: { filters: false, selection: [] },
  },
};
