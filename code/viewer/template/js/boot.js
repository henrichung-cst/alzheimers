// ---------------------------------------------------------------------------
function syncGlossary() {
  const open = Store.state.view.glossaryOpen;
  const panel = document.getElementById("glossary-panel");
  const toggle = document.getElementById("glossary-toggle");
  if (panel) {
    panel.classList.toggle("open", open);
    panel.setAttribute("aria-hidden", open ? "false" : "true");
  }
  if (toggle) toggle.setAttribute("aria-expanded", open ? "true" : "false");
}

function _kinaseRerenderForFilter(activeTab){
  if (activeTab === "pathway") renderPathwayExplorer();
  if (activeTab === "graph") renderGraph();
  if (activeTab === "temporal" && Store.state.view.temporalLevel === "backbone")
    renderTemporal();
  if (activeTab === "additivity" && Store.state.view.additivityLevel === "backbone")
    renderAdditivity();
  if (activeTab === "kinase") renderKinaseExplorer();
}

// Backbone selection → highlight kinases that drive it. Loaded async from the
// per-backbone edge slice; updates a Set used by the table renderer.
let _highlightKinaseIds = null;
let _highlightForBid = null;
async function _refreshHighlightForBackbone(bid){
  if (bid == null) {
    _highlightKinaseIds = null; _highlightForBid = null;
    return;
  }
  if (bid === _highlightForBid) return;
  _highlightForBid = bid;
  try {
    const rows = await SliceCache.backboneEdges(bid);
    if (Store.state.selection.backbone !== bid) return;
    const s = new Set();
    for (const r of rows) s.add(r.kinase_id);
    _highlightKinaseIds = s;
    if (Store.state.view.activeTab === "kinase") renderKinaseExplorer();
  } catch (e) { console.warn("highlight fetch failed", e); }
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
function boot() {
  populateHeader();
  wireTabs();
  wireKinaseTable();
  wirePathwayTable();
  wireGraphControls();
  wireGraphKeyboard();
  wireSenderMatrix();
  wireSenderMatrixKeyboard();
  wireTemporalControls();
  wireAdditivityControls();
  wireTemporalV2();
  syncHeaderFromStore();
  syncTabsFromStore();
  applyMetricTooltips();
  syncFilterBarToTab(Store.state.view.activeTab);
  syncGlossary();
  wireDrawerResizer();
  wireExportButtons();
  renderHowToDrawer(Store.state.view.activeTab);
  applyHash();
  window.addEventListener("popstate", applyHash);
  window.addEventListener("hashchange", () => {
    if (_serializeHash() !== window.location.hash) applyHash();
  });

  Store.subscribe((next, prev) => {
    const activeTab = next.view.activeTab;
    pushHash();
    if (next.filters !== prev.filters) {
      syncHeaderFromStore();
      if (activeTab === "signal") renderOverview();
      if (activeTab === "kinase") {
        renderKinaseExplorer();
        if (next.selection.kinase != null)
          renderKinaseDetail(next.selection.kinase);
      }
      if (activeTab === "pathway") {
        renderPathwayExplorer();
        if (next.selection.backbone != null)
          renderPathwayDetail(next.selection.backbone);
      }
      if (activeTab === "graph") renderGraph();
      if (activeTab === "senders") renderSenderMatrix();
      if (activeTab === "temporal") renderTemporal();
      if (activeTab === "additivity") renderAdditivity();
      if (activeTab === "temporalv2") renderTemporalV2();
    }
    if (next.selection.kinase !== prev.selection.kinase) {
      syncHeaderFromStore();
      const kid = next.selection.kinase;
      if (kid != null && SliceCache.kinaseBackboneSetSync(kid) === null) {
        SliceCache.loadKinase(kid).then(() => {
          if (Store.state.selection.kinase !== kid) return;
          invalidateFilterCache();
          _kinaseRerenderForFilter(Store.state.view.activeTab);
        }).catch(e => console.warn("kinase slice load failed", e));
      } else {
        invalidateFilterCache();
      }
      if (activeTab === "kinase") {
        _updateKinaseRowSelection(next.selection.kinase);
        renderKinaseDetail(next.selection.kinase);
      }
      if (activeTab !== "kinase") _kinaseRerenderForFilter(activeTab);
    }
    if (next.selection.celltype !== prev.selection.celltype) {
      syncHeaderFromStore();
      invalidateFilterCache();
      _kinaseRerenderForFilter(activeTab);
    }
    if (next.selection.backbone !== prev.selection.backbone) {
      if (activeTab === "pathway") {
        _updatePathwayRowSelection(next.selection.backbone);
        renderPathwayDetail(next.selection.backbone);
      }
      _refreshHighlightForBackbone(next.selection.backbone);
    }
    if (next.view !== prev.view) {
      if (next.view.activeTab !== prev.view.activeTab) {
        syncTabsFromStore();
        syncFilterBarToTab(activeTab);
        renderHowToDrawer(activeTab);
        if (activeTab === "kinase") {
          renderKinaseExplorer();
          if (next.selection.kinase != null)
            renderKinaseDetail(next.selection.kinase);
        }
        if (activeTab === "pathway") {
          renderPathwayExplorer();
          if (next.selection.backbone != null)
            renderPathwayDetail(next.selection.backbone);
        }
        if (activeTab === "graph") renderGraph();
        if (activeTab === "signal") renderOverview();
        if (activeTab === "senders") renderSenderMatrix();
        if (activeTab === "temporal") renderTemporal();
        if (activeTab === "additivity") renderAdditivity();
        if (activeTab === "temporalv2") renderTemporalV2();
        if (prev.view.activeTab === "graph" && activeTab !== "graph")
          _destroyCy();
      }
      if (next.view.glossaryOpen !== prev.view.glossaryOpen) syncGlossary();
      if (next.view.overviewMode !== prev.view.overviewMode &&
          activeTab === "signal") renderOverview();
      if ((next.view.graphLayout !== prev.view.graphLayout ||
           next.view.graphMinDegree !== prev.view.graphMinDegree ||
           next.view.graphGenotype !== prev.view.graphGenotype ||
           next.view.graphTimepoint !== prev.view.graphTimepoint ||
           next.view.graphTpdsMin !== prev.view.graphTpdsMin ||
           next.view.graphTopN !== prev.view.graphTopN) &&
          activeTab === "graph") {
        const genoSel = document.getElementById("graph-genotype");
        if (genoSel && genoSel.value !== next.view.graphGenotype)
          genoSel.value = next.view.graphGenotype;
        const tpSel = document.getElementById("graph-timepoint");
        if (tpSel && tpSel.value !== next.view.graphTimepoint)
          tpSel.value = next.view.graphTimepoint;
        renderGraph();
      }
      if (activeTab === "senders" &&
          (next.view.senderMatrixMode !== prev.view.senderMatrixMode ||
           next.view.senderMatrixAxis !== prev.view.senderMatrixAxis ||
           next.view.senderMatrixAnchor !== prev.view.senderMatrixAnchor)) {
        if (next.view.senderMatrixAxis !== prev.view.senderMatrixAxis) {
          _populateSenderAnchorSelect();
        } else {
          const anchorSel = document.getElementById("sm-anchor");
          if (anchorSel && anchorSel.value !== next.view.senderMatrixAnchor) {
            anchorSel.value = next.view.senderMatrixAnchor;
          }
        }
        renderSenderMatrix();
      }
      if ((next.view.temporalLevel !== prev.view.temporalLevel ||
           next.view.temporalMetric !== prev.view.temporalMetric ||
           next.view.temporalTissue !== prev.view.temporalTissue ||
           next.view.temporalScoreMin !== prev.view.temporalScoreMin) &&
          activeTab === "temporal") renderTemporal();
      if ((next.view.additivityLevel !== prev.view.additivityLevel ||
           next.view.additivityTimepoint !== prev.view.additivityTimepoint ||
           next.view.additivityScoreMin !== prev.view.additivityScoreMin) &&
          activeTab === "additivity") renderAdditivity();
      if (next.view.kinaseAuditTab !== prev.view.kinaseAuditTab &&
          activeTab === "kinase" && next.selection.kinase != null)
        renderActiveKinaseAuditTab(next.selection.kinase);
      if (next.view.pathwayScoreMin !== prev.view.pathwayScoreMin &&
          activeTab === "pathway") renderPathwayExplorer();
    }
  });
}

if (document.readyState === "loading")
  document.addEventListener("DOMContentLoaded", boot);
else boot();
