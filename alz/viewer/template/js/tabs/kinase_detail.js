function renderKinaseDetail(kinase_id) {
  const el = document.getElementById("ke-detail");
  if (!el) return;
  if (kinase_id == null) {
    el.innerHTML = '<div class="muted">Select a kinase to open the audit workbench.</div>';
    return;
  }
  _ensureKinaseIndexes();
  const K = ViewerPayload.kinases();
  const i = _kinaseIdxById.get(kinase_id);
  if (i == null) {
    el.innerHTML = '<div class="muted">Kinase not found.</div>';
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
