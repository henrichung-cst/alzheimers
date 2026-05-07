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
  const m = TAB_MANIFEST[Store.state.view.activeTab];
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
    const tabChanged         = next.view.activeTab      !== prev.view.activeTab;
    const viewChanged        = next.view                !== prev.view;

    if (!filtersChanged && !kinaseSelChanged && !celltypeSelChanged
        && !backboneSelChanged && !tabChanged && !viewChanged) return;

    if (filtersChanged || kinaseSelChanged || celltypeSelChanged) syncHeaderFromStore();

    // Pre-render side effects that run regardless of which tab is active.
    if (kinaseSelChanged) {
      const kid = next.selection.kinase;
      if (kid != null && SliceCache.kinaseBackboneSetSync(kid) === null) {
        SliceCache.loadKinase(kid).then(() => {
          if (Store.state.selection.kinase !== kid) return;
          invalidateFilterCache();
          _activeTabRender();
        }).catch(e => console.warn("kinase slice load failed", e));
      } else {
        invalidateFilterCache();
      }
    }
    if (celltypeSelChanged) invalidateFilterCache();
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
      });
      if (!handled) {
        const re = m.rerenderOn || {};
        let need = tabChanged;
        if (filtersChanged && re.filters) need = true;
        if (re.selection) {
          if (re.selection.includes("kinase")   && kinaseSelChanged)   need = true;
          if (re.selection.includes("celltype") && celltypeSelChanged) need = true;
          if (re.selection.includes("backbone") && backboneSelChanged) need = true;
        }
        if (need && m.render) m.render();
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

if (document.readyState === "loading")
  document.addEventListener("DOMContentLoaded", boot);
else boot();
