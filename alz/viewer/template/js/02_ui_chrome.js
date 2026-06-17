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
