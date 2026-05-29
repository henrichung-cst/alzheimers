function _hasAuditWorkbench() {
  const at = (typeof PAYLOAD !== "undefined" && PAYLOAD.audit_tables) || {};
  const tables = at.tables || {};
  return Object.prototype.hasOwnProperty.call(tables, "mea_stoichiometry");
}

function _renderKinaseDetailSummary(el, K, i) {
  const name = K.name[i];
  const gene = K.gene_symbol[i] || "";
  const rows = CONTRASTS.map(c => {
    const nes = K["NES_" + c][i];
    const fdr = K["FDR_" + c][i];
    const nf = Number.isFinite(nes) ? nes.toFixed(2) : "—";
    const ff = Number.isFinite(fdr) ? fdr.toExponential(1) : "—";
    return `<tr><td>${c}</td><td style="text-align:right;">${nf}</td><td style="text-align:right;">${ff}</td></tr>`;
  }).join("");
  el.innerHTML =
    `<div class="kinase-workbench-header">` +
    `<div class="kinase-workbench-title"><h3>${name}</h3>` +
    `<div class="muted">${gene}</div></div></div>` +
    `<table class="data-table" style="margin-top:8px;">` +
    `<thead><tr><th>Contrast</th><th style="text-align:right;">NES</th><th style="text-align:right;">FDR</th></tr></thead>` +
    `<tbody>${rows}</tbody></table>` +
    `<div class="muted" style="margin-top:8px;font-size:11px;">` +
    `T-cell MEA is bulk-only; no per-site OLS audit substrate.</div>`;
}

function renderKinaseDetail(kinase_id) {
  const el = document.getElementById("ke-detail");
  if (!el) return;
  if (kinase_id == null) {
    el.innerHTML = '<div class="muted">Select a kinase to view NES / FDR per contrast.</div>';
    return;
  }
  _ensureKinaseIndexes();
  const K = PAYLOAD.kinases;
  const i = _kinaseIdxById.get(kinase_id);
  if (i == null) {
    el.innerHTML = '<div class="muted">Kinase not found.</div>';
    return;
  }
  if (!_hasAuditWorkbench()) {
    _renderKinaseDetailSummary(el, K, i);
    return;
  }
  const name = K.name[i];
  ++_kinaseAuditSeq;
  const tabButtons = KINASE_AUDIT_TABS.map(t =>
    `<button type="button" data-audit-tab="${t.id}" class="${t.id === _activeKinaseAuditTab() ? "active" : ""}">${t.label}</button>`
  ).join("");

  el.innerHTML =
    `<div class="kinase-workbench-header">` +
    `<div class="kinase-workbench-title"><h3>${name}</h3></div>` +
    `<div class="kinase-workbench-controls">` +
    `<label>Contrast <select id="audit-contrast-select"><option value="ALL">Auto peak/global</option>${CONTRASTS.map(c => `<option value="${c}">${c}</option>`).join("")}</select></label>` +
    `<label>Sample <select id="audit-sample-select"></select></label>` +
    `</div></div>` +
    `<div class="kinase-audit-tabs" role="tablist" aria-label="Kinase audit walkthrough">${tabButtons}</div>` +
    `<div class="kinase-audit-tab-body" id="kinase-audit-body"></div>`;

  const contrastSelect = document.getElementById("audit-contrast-select");
  if (contrastSelect) {
    contrastSelect.value = Store.state.filters.contrast || "ALL";
    contrastSelect.onchange = ev => Store.dispatch({type:"SET_FILTER", key:"contrast", value:ev.target.value});
  }
  document.querySelectorAll(".kinase-audit-tabs button").forEach(btn => {
    btn.addEventListener("click", () => {
      Store.dispatch({type:"SET_VIEW", key:"kinaseAuditTab", value:btn.dataset.auditTab});
    });
  });
  renderActiveKinaseAuditTab(kinase_id);
}
