
// Module-level handle so _syncKinaseFilterUI can re-render multiselects after
// reset or cross-tab handoff.
let _kineRenderMultiselect = null;

function wireKinaseTable() {
  const tbl = document.getElementById("ke-table");
  if (!tbl) return;

  tbl.querySelectorAll("thead th").forEach(th => {
    th.addEventListener("click", () => {
      const col = th.dataset.col;
      const cur = KinaseFilter.get();
      if (col === "nes_profile") {
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

  const MS_OPTS = { day: (CONTRASTS || []).slice() };
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
        _syncKinaseFilterUI();
        renderKinaseExplorer();
      },
    });
  }
  _renderMultiselect("day");
  _kineRenderMultiselect = _renderMultiselect;

  const nsigInp = document.getElementById("ke-filter-nsig-min");
  if (nsigInp) {
    nsigInp.max = String((CONTRASTS || []).length || 0);
    nsigInp.value = String(KinaseFilter.get("nSigMin") || 0);
    nsigInp.addEventListener("change", () => {
      const maxN = (getScopedContrastIds(KinaseFilter.get()).size || (CONTRASTS || []).length || 0);
      const v = Math.max(0, Math.min(maxN, parseInt(nsigInp.value, 10) || 0));
      nsigInp.value = String(v);
      KinaseFilter.set({nSigMin: v});
      renderKinaseExplorer();
    });
  }

  const signSel = document.getElementById("ke-filter-sign");
  if (signSel) {
    signSel.value = KinaseFilter.get("sign") || "";
    signSel.addEventListener("change", () => {
      KinaseFilter.set({sign: signSel.value || ""});
      renderKinaseExplorer();
    });
  }

  const patSel = document.getElementById("ke-filter-pattern");
  if (patSel) {
    patSel.value = KinaseFilter.get("pattern") || "";
    patSel.addEventListener("change", () => {
      KinaseFilter.set({pattern: TrendFilter.normalize(patSel.value || "")});
      renderKinaseExplorer();
    });
  }

  // Specificity narrowing tool — OPT-IN only (defaults to Any/0, hides nothing
  // by default; the ke-count indicator reflects when it's active). Specificity
  // is the informative localizer; concordance is never filtered (de-gate dir.).
  const tcellSel = document.getElementById("ke-filter-tcell");
  if (tcellSel) {
    tcellSel.value = String(KinaseFilter.get("tcellMin") || 0);
    tcellSel.addEventListener("change", () => {
      KinaseFilter.set({tcellMin: parseInt(tcellSel.value, 10) || 0});
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

function _syncKinaseFilterUI() {
  const inp = document.getElementById("ke-search");
  if (inp) inp.value = KinaseFilter.get("search") || "";
  if (_kineRenderMultiselect) _kineRenderMultiselect("day");
  const nsigInp = document.getElementById("ke-filter-nsig-min");
  if (nsigInp) {
    const maxN = (getScopedContrastIds(KinaseFilter.get()).size || (CONTRASTS || []).length || 0);
    nsigInp.max = String(maxN);
    const v = Math.max(0, Math.min(maxN, parseInt(KinaseFilter.get("nSigMin"), 10) || 0));
    nsigInp.value = String(v);
  }
  const signSel = document.getElementById("ke-filter-sign");
  if (signSel) signSel.value = KinaseFilter.get("sign") || "";
  const patSel = document.getElementById("ke-filter-pattern");
  if (patSel) patSel.value = KinaseFilter.get("pattern") || "";
  const tcellSel = document.getElementById("ke-filter-tcell");
  if (tcellSel) tcellSel.value = String(KinaseFilter.get("tcellMin") || 0);
}
