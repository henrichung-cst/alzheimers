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

function _activeTabRender() {
  const tab = Store.state.view.activeTab;
  const m = TAB_MANIFEST[tab];
  if (m && m.render) m.render();
}

// Defensive: warn loudly if the manifest and the body.html panels drift apart.
function _validateTabManifest() {
  for (const id of Object.keys(TAB_MANIFEST)) {
    if (!document.getElementById("tab-" + id))
      console.warn(`TAB_MANIFEST tab "${id}" has no <div id="tab-${id}"> panel.`);
  }
  document.querySelectorAll(".tab-panel").forEach(p => {
    const id = p.id.replace(/^tab-/, "");
    if (!TAB_MANIFEST[id])
      console.warn(`Panel #${p.id} has no TAB_MANIFEST entry.`);
  });
  if (!TAB_MANIFEST[Store.state.view.activeTab])
    console.warn(`Initial activeTab "${Store.state.view.activeTab}" missing from TAB_MANIFEST.`);
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
function boot() {
  populateHeader();
  wireTabs();
  _validateTabManifest();
  // One-time per-tab wiring. Order is insertion order of TAB_MANIFEST.
  Object.values(TAB_MANIFEST).forEach(m => {
    try { m.wire && m.wire(); }
    catch (e) { console.error("tab wire failed", e); }
  });
  syncHeaderFromStore();
  syncTabsFromStore();
  applyMetricTooltips();
  syncFilterBarToTab(Store.state.view.activeTab);
  syncGlossary();
  wireDrawerResizer();
  wireExportButtons();
  renderHowToDrawer(Store.state.view.activeTab);
  applyHash();
  // Initial render — Store.subscribe doesn't fire on the seed state, and
  // applyHash() only dispatches when the URL hash carries values.
  _activeTabRender();
  window.addEventListener("popstate", applyHash);
  window.addEventListener("hashchange", () => {
    if (_serializeHash() !== window.location.hash) applyHash();
  });

  Store.subscribe((next, prev) => {
    const activeTab = next.view.activeTab;
    pushHash();

    const filtersChanged     = next.filters             !== prev.filters;
    const kinaseSelChanged   = next.selection.kinase    !== prev.selection.kinase;
    const celltypeSelChanged = next.selection.celltype  !== prev.selection.celltype;
    const backboneSelChanged = next.selection.backbone  !== prev.selection.backbone;
    const kinaseHumanSelChanged = next.selection.kinaseHuman !== prev.selection.kinaseHuman;
    const kinaseFiveXFADSelChanged = next.selection.kinaseFiveXFAD !== prev.selection.kinaseFiveXFAD;
    const contextChanged     = next.selection.context   !== prev.selection.context;
    const modeChanged        = next.view.mode           !== prev.view.mode;
    const tabChanged         = next.view.activeTab      !== prev.view.activeTab;
    const viewChanged        = next.view                !== prev.view;

    if (!filtersChanged && !kinaseSelChanged && !celltypeSelChanged
        && !backboneSelChanged && !kinaseHumanSelChanged
        && !kinaseFiveXFADSelChanged && !contextChanged && !modeChanged
        && !tabChanged && !viewChanged) return;

    if (modeChanged) {
      // Clear cross-mode selections so a mouse kinase doesn't haunt the human panel.
      Store.dispatch({type:"SET_SELECTION", key:"kinase", value:null});
      Store.dispatch({type:"SET_SELECTION", key:"kinaseHuman", value:null});
      Store.dispatch({type:"SET_SELECTION", key:"kinaseFiveXFAD", value:null});
      // Mode drives the incytr context: 5xFAD mode points incytr at its tissue
      // context, every other mode restores Song AD. (No-op in viewers without
      // the hook, e.g. the t-cell viewer.) Dispatching here cascades into the
      // contextChanged branch, which re-renders the active tab.
      if (typeof _applyModeContext === "function") _applyModeContext(next.view.mode);
      syncTabsFromStore();
      return;
    }

    if (contextChanged) {
      // Context switch: clear context-scoped selections, refresh prereq cards,
      // and re-render the active tab against the new context's vocab/shards.
      if (typeof resetKinaseContextCaches === "function") resetKinaseContextCaches();
      // The incytr binary filter-index is a per-context artifact; drop the
      // cached one so it refetches against the new context's manifest.
      if (typeof IncytrGlobalIndex !== "undefined" && IncytrGlobalIndex.reset)
        IncytrGlobalIndex.reset();
      RECEIVERS = ViewerPayload.celltypes().name || [];
      if (next.selection.kinase != null)
        Store.dispatch({type:"SET_SELECTION", key:"kinase", value:null});
      if (next.selection.backbone != null)
        Store.dispatch({type:"SET_SELECTION", key:"backbone", value:null});
      if (next.selection.celltype != null)
        Store.dispatch({type:"SET_SELECTION", key:"celltype", value:null});
      pushHash();
      syncTabsFromStore();
      syncHeaderFromStore();
      _activeTabRender();
      return;
    }

    if (filtersChanged || kinaseSelChanged || kinaseFiveXFADSelChanged || celltypeSelChanged) syncHeaderFromStore();

    if (backboneSelChanged) _refreshHighlightForBackbone(next.selection.backbone);

    if (tabChanged) {
      syncTabsFromStore();
      syncFilterBarToTab(activeTab);
      renderHowToDrawer(activeTab);
    }

    // Render decision — manifest-driven. A tab can override with onChange();
    // returning truthy suppresses the default rerenderOn dispatch.
    const m = TAB_MANIFEST[activeTab];
    if (m) {
      const handled = m.onChange && m.onChange({
        tabChanged, kinaseSelChanged, kid: next.selection.kinase,
        kinaseHumanSelChanged, khid: next.selection.kinaseHuman,
        kinaseFiveXFADSelChanged, f5key: next.selection.kinaseFiveXFAD,
      });
      if (!handled) {
        const re = m.rerenderOn || {};
        let need = tabChanged;
        if (filtersChanged && re.filters) need = true;
        if (re.selection) {
          if (re.selection.includes("kinase")   && kinaseSelChanged)   need = true;
          if (re.selection.includes("celltype") && celltypeSelChanged) need = true;
          if (re.selection.includes("backbone") && backboneSelChanged) need = true;
          if (re.selection.includes("kinaseFiveXFAD") && kinaseFiveXFADSelChanged) need = true;
        }
        if (need) _activeTabRender();
      }
    }

    // Global view-state subscribers (not tab-specific).
    if (next.view !== prev.view) {
      if (next.view.glossaryOpen !== prev.view.glossaryOpen) syncGlossary();
      if (next.view.kinaseAuditTab !== prev.view.kinaseAuditTab &&
          activeTab === "kinase" && next.selection.kinase != null)
        renderActiveKinaseAuditTab(next.selection.kinase);
    }
  });
}

async function _bootWithPayload() {
  try {
    await _loadPayload();
  } catch (e) {
    document.body.innerHTML =
      '<div style="padding:24px;font-family:system-ui;color:#a23;">'
      + '<h3>Failed to load payload</h3>'
      + '<pre style="white-space:pre-wrap;">' + (e.message || e) + '</pre>'
      + '<p>The viewer must be served over http (e.g. '
      + '<code>npx http-server -g -p 8000</code>) — file:// is no longer '
      + 'supported because PAYLOAD is now fetched asynchronously.</p></div>';
    return;
  }
  boot();
}
if (document.readyState === "loading")
  document.addEventListener("DOMContentLoaded", _bootWithPayload);
else _bootWithPayload();
