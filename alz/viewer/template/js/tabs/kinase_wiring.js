
// Module-level handle so _syncKinaseFilterUI can re-render multi-selects from
// outside wireKinaseTable's closure (e.g. after a cross-tab handoff).
let _kineRenderMultiselect = null;

function wireKinaseTable() {
  const tbl = document.getElementById("ke-table");
  if (!tbl) return;

  tbl.querySelectorAll("thead th").forEach(th => {
    th.addEventListener("click", () => {
      const col = th.dataset.col;
      const cur = KinaseFilter.get();
      if (col === "nes_profile") {
        // Sort by NES profile: first click selects it descending.
        if (cur.sortCol !== col) KinaseFilter.set({sortCol: col, sortAsc: false});
        else KinaseFilter.set({sortAsc: !cur.sortAsc});
        renderKinaseExplorer();
        return;
      }
      if (cur.sortCol === col) KinaseFilter.set({sortAsc: !cur.sortAsc});
      else KinaseFilter.set({sortCol: col, sortAsc: false});
      renderKinaseExplorer();
    });
  });
  tbl.querySelector("tbody").addEventListener("click", ev => {
    const tr = ev.target.closest("tr.ke-row");
    if (!tr) return;
    const kid = parseInt(tr.dataset.kid, 10);
    Store.dispatch({type:"SET_SELECTION", key:"kinase", value: kid});
  });
  tbl.querySelector("tbody").addEventListener("keydown", ev =>
    _activateRowOnKey(ev, "tr.ke-row", tr => {
      const kid = parseInt(tr.dataset.kid, 10);
      Store.dispatch({type:"SET_SELECTION", key:"kinase", value: kid});
    }));
  const search = document.getElementById("ke-search");
  if (search) {
    search.value = KinaseFilter.get("search");
    search.addEventListener("input", ev => {
      KinaseFilter.set({search: ev.target.value});
      renderKinaseExplorer();
    });
  }

  // Multiselect option sources.
  const AI = PAYLOAD.attribution_index || {};
  const allCells = Array.from(new Set(AI.cell_type || [])).sort();
  const MS_OPTS = {
    disease:    ["App","Tau","ApTt"],
    timepoint:  ["2mo","4mo","6mo"],
    celltype:   allCells,
  };

  function _renderMultiselect(key) {
    const host = document.getElementById("ke-ms-" + key);
    if (!host) return;
    mountMultiselect(host, {
      label:    host.dataset.label || key,
      options:  MS_OPTS[key] || [],
      current:  KinaseFilter.get(key) || [],
      onChange: (next) => {
        KinaseFilter.set({[key]: next});
        _renderMultiselect(key);
        renderKinaseExplorer();
      },
    });
  }

  ["disease","timepoint","celltype"].forEach(_renderMultiselect);
  _kineRenderMultiselect = _renderMultiselect;

  // Confidence (single, ordinal threshold).
  const confSel = document.getElementById("ke-filter-confidence");
  if (confSel) {
    confSel.value = KinaseFilter.get("confidence") || "";
    confSel.addEventListener("change", () => {
      KinaseFilter.set({confidence: confSel.value});
      renderKinaseExplorer();
    });
  }

  // WMB specificity tier minimum (single, ordinal threshold).
  const wmbSel = document.getElementById("ke-filter-wmb");
  if (wmbSel) {
    wmbSel.value = String(KinaseFilter.get("wmbMin") || 0);
    wmbSel.addEventListener("change", () => {
      KinaseFilter.set({wmbMin: parseInt(wmbSel.value, 10) || 0});
      renderKinaseExplorer();
    });
  }

  // n_sig minimum (numeric input).
  const nsigInp = document.getElementById("ke-filter-nsig-min");
  if (nsigInp) {
    nsigInp.value = String(KinaseFilter.get("nSigMin") || 0);
    nsigInp.addEventListener("change", () => {
      const v = Math.max(0, Math.min(9, parseInt(nsigInp.value, 10) || 0));
      nsigInp.value = String(v);
      KinaseFilter.set({nSigMin: v});
      renderKinaseExplorer();
    });
  }

  const resetBtn = document.getElementById("ke-filter-reset");
  if (resetBtn) {
    resetBtn.addEventListener("click", () => {
      KinaseFilter.reset();
      _syncKinaseFilterUI();
      renderKinaseExplorer();
    });
  }
}

// Re-pushes the persisted KinaseFilter state into all the toolbar inputs.
// Used after programmatic mutations (e.g. cross-tab handoff prefilling
// disease/timepoint from a Temporal v2 click) so the dropdowns reflect the
// new state without a full page rebuild.
function _syncKinaseFilterUI() {
  const inp = document.getElementById("ke-search");
  if (inp) inp.value = KinaseFilter.get("search") || "";
  if (_kineRenderMultiselect) {
    ["disease","timepoint","celltype"].forEach(k => _kineRenderMultiselect(k));
  }
  const confSel = document.getElementById("ke-filter-confidence");
  if (confSel) confSel.value = KinaseFilter.get("confidence") || "";
  const wmbSel = document.getElementById("ke-filter-wmb");
  if (wmbSel) wmbSel.value = String(KinaseFilter.get("wmbMin") || 0);
  const nsigInp = document.getElementById("ke-filter-nsig-min");
  if (nsigInp) nsigInp.value = String(KinaseFilter.get("nSigMin") || 0);
}


