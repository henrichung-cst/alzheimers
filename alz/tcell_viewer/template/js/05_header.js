function populateHeader() {
  const fileNotice = document.getElementById("file-mode-notice");
  if (fileNotice && window.location.protocol === "file:")
    fileNotice.classList.add("show");
  document.getElementById("f-fdr").addEventListener("change", e =>
    Store.dispatch({type:"SET_FILTER", key:"fdr", value:parseFloat(e.target.value)}));
  document.getElementById("glossary-toggle").addEventListener("click", () =>
    Store.dispatch({type:"SET_VIEW", key:"glossaryOpen",
      value:!Store.state.view.glossaryOpen}));
  const skClear = document.getElementById("f-selection-kinase-clear");
  if (skClear) skClear.addEventListener("click", () =>
    Store.dispatch({type:"SET_SELECTION", key:"kinase", value:null}));
  const scClear = document.getElementById("f-selection-celltype-clear");
  if (scClear) scClear.addEventListener("click", () =>
    Store.dispatch({type:"SET_SELECTION", key:"celltype", value:null}));
}

function _activateRowOnKey(ev, selector, handler) {
  if (ev.key !== "Enter" && ev.key !== " ") return;
  const tr = ev.target.closest(selector);
  if (!tr) return;
  ev.preventDefault();
  handler(tr);
}

function syncHeaderFromStore() {
  const f = Store.state.filters;
  document.getElementById("f-fdr").value = f.fdr;
  const sel = Store.state.selection;
  const skClear = document.getElementById("f-selection-kinase-clear");
  if (skClear) {
    const on = sel.kinase != null;
    skClear.hidden = !on;
    if (on) {
      _ensureKinaseIdx();
      const K = PAYLOAD.kinases;
      const ki = _kinaseIdxById.get(sel.kinase);
      const name = ki != null ? K.name[ki] : ("kid:" + sel.kinase);
      skClear.textContent = "Clear kinase selection (" + name + ")";
    }
  }
  const scClear = document.getElementById("f-selection-celltype-clear");
  if (scClear) {
    const on = sel.celltype != null;
    scClear.hidden = !on;
    if (on) {
      const name = RECEIVERS[sel.celltype] || ("cid:" + sel.celltype);
      scClear.textContent = "Clear cell-type selection (" + name + ")";
    }
  }
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

// Build the <nav#tab-bar> children from TAB_MANIFEST. Group order is fixed
// by TAB_GROUP_ORDER; unknown groups (defensive) emit at the end. Emits bare
// structure only — active state is owned by syncTabsFromStore().
function _buildTabBar() {
  const nav = document.getElementById("tab-bar");
  if (!nav) return;
  const visible = _activeModeTabs();
  const byGroup = new Map();
  for (const id of Object.keys(visible)) {
    const m = visible[id];
    if (!byGroup.has(m.group)) byGroup.set(m.group, []);
    byGroup.get(m.group).push([id, m]);
  }
  const orderedGroups = TAB_GROUP_ORDER
    .filter(g => byGroup.has(g))
    .concat(Array.from(byGroup.keys()).filter(g => !TAB_GROUP_ORDER.includes(g)));
  const parts = [];
  orderedGroups.forEach((g, gi) => {
    if (gi > 0) parts.push('<span class="tab-group-divider" aria-hidden="true"></span>');
    parts.push('<span class="tab-group-label">'
      + _escapeHtml(TAB_GROUP_LABELS[g] || g) + '</span>');
    byGroup.get(g).forEach(([id, m]) => {
      parts.push(
        '<button id="tabbtn-' + id + '" role="tab" aria-selected="false"'
        + ' aria-controls="tab-' + id + '"'
        + ' data-tab="' + id + '" data-tab-group="' + g + '">'
        + _escapeHtml(m.label) + '</button>'
      );
    });
  });
  nav.innerHTML = parts.join("");
}

function _wireTabHandlers() {
  const tabs = Array.from(document.querySelectorAll("nav#tab-bar button"));
  tabs.forEach((btn, idx) => {
    if (btn._wired) return;
    btn._wired = true;
    btn.addEventListener("click", () => {
      Store.dispatch({type:"SET_VIEW", key:"activeTab", value:btn.dataset.tab});
    });
    btn.addEventListener("keydown", ev => {
      const key = ev.key;
      if (!["ArrowRight", "ArrowLeft", "Home", "End"].includes(key)) return;
      ev.preventDefault();
      const live = Array.from(document.querySelectorAll("nav#tab-bar button"));
      const myIdx = live.indexOf(btn);
      let nextIdx = myIdx;
      if (key === "ArrowRight") nextIdx = (myIdx + 1) % live.length;
      else if (key === "ArrowLeft") nextIdx = (myIdx - 1 + live.length) % live.length;
      else if (key === "Home") nextIdx = 0;
      else if (key === "End") nextIdx = live.length - 1;
      live[nextIdx].focus();
      Store.dispatch({type:"SET_VIEW", key:"activeTab", value:live[nextIdx].dataset.tab});
    });
  });
}

function wireTabs() {
  _buildTabBar();
  _wireTabHandlers();
  // Mode toggle wiring (visible only when PAYLOAD.human exists).
  const wrap = document.getElementById("mode-toggle");
  if (wrap && HAS_HUMAN) {
    wrap.querySelectorAll("button.mode-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const m = btn.dataset.mode;
        if (m === Store.state.view.mode) return;
        Store.dispatch({type:"SET_VIEW", key:"mode", value:m});
      });
    });
  }
  // Donor toggle wiring (T-cell viewer only).
  const dwrap = document.getElementById("donor-toggle");
  if (dwrap) {
    dwrap.querySelectorAll("button.mode-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const d = btn.dataset.donor;
        if (d === Store.state.selection.donor) return;
        Store.dispatch({type:"SET_SELECTION", key:"donor", value:d});
      });
    });
  }
}

function _syncModeToggle() {
  const wrap = document.getElementById("mode-toggle");
  if (!wrap) return;
  if (!HAS_HUMAN) { wrap.hidden = true; return; }
  wrap.hidden = false;
  const mode = Store.state.view.mode || "mouse";
  wrap.querySelectorAll("button.mode-btn").forEach(btn => {
    const on = btn.dataset.mode === mode;
    btn.classList.toggle("active", on);
    btn.setAttribute("aria-selected", on ? "true" : "false");
  });
}

function _syncDonorToggle() {
  const wrap = document.getElementById("donor-toggle");
  if (!wrap) return;
  const donor = (Store.state.selection && Store.state.selection.donor) || "donor1";
  wrap.querySelectorAll("button.mode-btn").forEach(btn => {
    const on = btn.dataset.donor === donor;
    btn.classList.toggle("active", on);
    btn.setAttribute("aria-selected", on ? "true" : "false");
  });
}

function _defaultTabForMode(mode) {
  return mode === "human" ? "kinasehuman" : "kinase";
}

function syncTabsFromStore() {
  // Mode-gate: rebuild tab bar if the current set of visible tabs doesn't
  // match the current mode (covers SET_MODE transitions).
  _buildTabBar();
  _wireTabHandlers();
  _syncModeToggle();
  _syncDonorToggle();
  // If the active tab is not visible in the current mode, snap to the mode's default.
  const visible = _activeModeTabs();
  let active = Store.state.view.activeTab;
  if (!visible[active]) {
    active = _defaultTabForMode(Store.state.view.mode || "mouse");
    Store.dispatch({type:"SET_VIEW", key:"activeTab", value:active});
    return;  // store change will retrigger syncTabsFromStore
  }
  document.querySelectorAll("nav#tab-bar button").forEach(btn => {
    const on = btn.dataset.tab === active;
    btn.classList.toggle("active", on);
    btn.setAttribute("aria-selected", on ? "true" : "false");
    btn.tabIndex = on ? 0 : -1;
  });
  document.querySelectorAll(".tab-panel").forEach(p => {
    const on = p.id === "tab-" + active;
    p.classList.toggle("active", on);
    p.hidden = !on;
  });
}

// ---------------------------------------------------------------------------
// Signal Map tab — receiver × contrast heatmap
// ---------------------------------------------------------------------------
