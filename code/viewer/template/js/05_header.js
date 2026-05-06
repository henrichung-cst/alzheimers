function populateHeader() {
  const fileNotice = document.getElementById("file-mode-notice");
  if (fileNotice && window.location.protocol === "file:")
    fileNotice.classList.add("show");
  const fr = document.getElementById("f-receiver");
  fr.innerHTML = ['<option value="ALL">All</option>']
    .concat(RECEIVERS.map(r => `<option value="${r}">${r}</option>`)).join("");
  fr.addEventListener("change", e => Store.dispatch({
    type:"SET_FILTER", key:"receiver", value:e.target.value}));
  document.getElementById("f-pathway-evidence").addEventListener("change", e =>
    Store.dispatch({type:"SET_FILTER", key:"pathwayEvidence", value:e.target.value}));
  document.getElementById("f-fdr").addEventListener("change", e =>
    Store.dispatch({type:"SET_FILTER", key:"fdr", value:parseFloat(e.target.value)}));
  const tps = document.getElementById("f-tpds-sig");
  if (tps) tps.addEventListener("change", e =>
    Store.dispatch({type:"SET_FILTER", key:"tpdsSig", value:e.target.value}));
  document.getElementById("glossary-toggle").addEventListener("click", () =>
    Store.dispatch({type:"SET_VIEW", key:"glossaryOpen",
      value:!Store.state.view.glossaryOpen}));
  const gnClear = document.getElementById("f-graph-nodes-clear");
  if (gnClear) gnClear.addEventListener("click", () =>
    Store.dispatch({type:"SET_FILTER", key:"graphNodeIds", value:null}));
  const sClear = document.getElementById("f-sender-clear");
  if (sClear) sClear.addEventListener("click", () =>
    Store.dispatch({type:"SET_FILTER", key:"sender", value:null}));
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
  const ids = ["f-receiver","f-pathway-evidence"];
  const vals = [f.receiver, f.pathwayEvidence];
  for (let i = 0; i < ids.length; i++) {
    const el = document.getElementById(ids[i]);
    if (el && el.value !== String(vals[i])) el.value = vals[i];
  }
  document.getElementById("f-fdr").value = f.fdr;
  const tps = document.getElementById("f-tpds-sig");
  if (tps && tps.value !== String(f.tpdsSig || "OFF")) tps.value = f.tpdsSig || "OFF";
  const gnClear = document.getElementById("f-graph-nodes-clear");
  if (gnClear) {
    const on = !!(f.graphNodeIds && f.graphNodeIds.length);
    gnClear.hidden = !on;
    if (on) gnClear.textContent = "Clear graph-node filter ("
      + f.graphNodeIds.length + " backbones)";
  }
  const sClear = document.getElementById("f-sender-clear");
  if (sClear) {
    const on = f.sender != null;
    sClear.hidden = !on;
    if (on) {
      const SENDERS = META.senderOrder || [];
      sClear.textContent = "Clear sender filter (" +
        (SENDERS[f.sender] || ("sid:" + f.sender)) + ")";
    }
  }
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
function wireTabs() {
  const tabs = Array.from(document.querySelectorAll("nav#tab-bar button"));
  tabs.forEach((btn, idx) => {
    btn.addEventListener("click", () => {
      Store.dispatch({type:"SET_VIEW", key:"activeTab", value:btn.dataset.tab});
    });
    btn.addEventListener("keydown", ev => {
      const key = ev.key;
      if (!["ArrowRight", "ArrowLeft", "Home", "End"].includes(key)) return;
      ev.preventDefault();
      let nextIdx = idx;
      if (key === "ArrowRight") nextIdx = (idx + 1) % tabs.length;
      else if (key === "ArrowLeft") nextIdx = (idx - 1 + tabs.length) % tabs.length;
      else if (key === "Home") nextIdx = 0;
      else if (key === "End") nextIdx = tabs.length - 1;
      tabs[nextIdx].focus();
      Store.dispatch({type:"SET_VIEW", key:"activeTab", value:tabs[nextIdx].dataset.tab});
    });
  });
}

function syncTabsFromStore() {
  const active = Store.state.view.activeTab;
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
