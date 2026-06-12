// 5xFAD mouse kinase explorer.
//
// Keeps 5xFAD as a first-class mode while using the same master/detail kinase
// workbench grammar as the Song and Mukesh kinase tabs.

const _F5_AGES = [3, 6, 9, 12];
const _F5_AUDIT_TABS = [
  {id: "measurement-trace", label: "Measurement Trace"},
  {id: "ols-details", label: "OLS Details"},
  {id: "mea-input", label: "MEA Preparation"},
  {id: "mea-score", label: "MEA Score"},
];

let _F5Groups = null;
let _F5QcByKey = null;
let _F5Wired = false;

const _F5State = {
  search: "",
  tissue: "",
  assay: "",
  analysis: "",
  age: "",
  nsigMin: 0,
  sortCol: "peakAbsNes",
  sortAsc: false,
  selectedKey: null,
  auditTab: "mea-score",
  auditAge: null,
};

function _f5Block() {
  return (PAYLOAD && PAYLOAD.supporting_5xfad) || null;
}

function _f5Esc(s) {
  return (typeof _escapeHtml === "function") ? _escapeHtml(s == null ? "" : s) : String(s == null ? "" : s);
}

function _f5Num(v) {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}

function _f5Fmt(v, digits) {
  const n = _f5Num(v);
  return n == null ? "—" : n.toFixed(digits == null ? 3 : digits);
}

function _f5TrackLabel(v) {
  if (v === "stoichiometry") return "stoichiometry";
  if (v === "raw_phospho") return "raw phospho";
  return String(v || "");
}

function _f5SliceLabel(group) {
  return `${group.tissue} · ${group.assay} · ${_f5TrackLabel(group.analysis_track)}`;
}

function _f5RowKey(row) {
  return [
    row.kinase || "",
    row.tissue || "",
    row.assay || "",
    row.analysis_track || "",
  ].join("|");
}

function _f5QcKey(tissue, track, age) {
  return [tissue || "", track || "", String(age || "")].join("|");
}

function _f5Family(name) {
  const famMap = (typeof META !== "undefined" && META && META.familyMap) || {};
  return famMap[name] || "";
}

function _f5EnsureIndexes() {
  if (_F5Groups && _F5QcByKey) return;
  const block = _f5Block();
  _F5Groups = [];
  _F5QcByKey = new Map();
  if (!block) return;

  for (const q of (block.contrast_qc || [])) {
    _F5QcByKey.set(_f5QcKey(q.tissue, q.track, q.age_months), q);
  }

  const byKey = new Map();
  for (const row of (block.rows || [])) {
    const key = _f5RowKey(row);
    let rec = byKey.get(key);
    if (!rec) {
      rec = {
        key,
        kinase: row.kinase || "",
        gene_symbol: row.gene_symbol || row.kinase || "",
        family: _f5Family(row.kinase || ""),
        tissue: row.tissue || "",
        track: row.track || "",
        assay: row.assay || "",
        residue_type: row.residue_type || "",
        analysis_track: row.analysis_track || "",
        rows: new Map(),
      };
      byKey.set(key, rec);
    }
    rec.rows.set(Number(row.age_months), row);
  }
  _F5Groups = Array.from(byKey.values());
}

function _f5AgeScope() {
  return _F5State.age ? [Number(_F5State.age)] : _F5_AGES.slice();
}

function _f5StatusFor(group, age) {
  const row = group.rows.get(Number(age));
  if (row && row.contrast_status) return row.contrast_status;
  const qc = _F5QcByKey && _F5QcByKey.get(_f5QcKey(group.tissue, group.track, age));
  return qc ? (qc.contrast_status || "") : "";
}

function _f5Metric(group, ages) {
  let peakAbsNes = null;
  let sigCount = 0;
  let substrateHits = null;
  let substrateUniverse = null;
  const fdr = Store.state.filters.fdr;
  for (const age of ages) {
    const row = group.rows.get(Number(age));
    if (!row) continue;
    const nes = _f5Num(row.NES);
    if (nes != null) {
      const abs = Math.abs(nes);
      if (peakAbsNes == null || abs > peakAbsNes) peakAbsNes = abs;
    }
    const q = _f5Num(row.FDR);
    if (q != null && q < fdr) sigCount += 1;
    const hits = _f5Num(row.substrate_hits);
    if (hits != null && (substrateHits == null || hits > substrateHits)) {
      substrateHits = hits;
      substrateUniverse = _f5Num(row.substrate_universe);
    }
  }
  return {peakAbsNes, sigCount, substrateHits, substrateUniverse};
}

function _f5GroupPasses(group, ages, metric) {
  const q = _F5State.search.trim().toLowerCase();
  if (q && !(String(group.kinase).toLowerCase().includes(q)
          || String(group.gene_symbol).toLowerCase().includes(q))) return false;
  if (_F5State.tissue && group.tissue !== _F5State.tissue) return false;
  if (_F5State.assay && group.assay !== _F5State.assay) return false;
  if (_F5State.analysis && group.analysis_track !== _F5State.analysis) return false;
  if (metric.sigCount < _F5State.nsigMin) return false;
  if (_F5State.age) {
    const age = Number(_F5State.age);
    if (!group.rows.has(age) && !_f5StatusFor(group, age)) return false;
  }
  return true;
}

function _f5FilteredRows() {
  _f5EnsureIndexes();
  const ages = _f5AgeScope();
  const out = [];
  for (const group of _F5Groups) {
    const metric = _f5Metric(group, ages);
    if (_f5GroupPasses(group, ages, metric)) {
      out.push({...group, ...metric, slice: _f5SliceLabel(group)});
    }
  }
  const col = _F5State.sortCol;
  const asc = _F5State.sortAsc;
  out.sort((a, b) => {
    let va = col === "profile" ? a.peakAbsNes : a[col];
    let vb = col === "profile" ? b.peakAbsNes : b[col];
    if (va == null && vb == null) return String(a.kinase).localeCompare(String(b.kinase));
    if (va == null) return 1;
    if (vb == null) return -1;
    if (typeof va === "string") return asc ? va.localeCompare(vb) : vb.localeCompare(va);
    return asc ? (va - vb) : (vb - va);
  });
  return out;
}

function _f5MaxAbsVisible(rows) {
  let maxAbs = 0;
  for (const group of rows) {
    for (const age of _F5_AGES) {
      const row = group.rows.get(age);
      const nes = row ? _f5Num(row.NES) : null;
      if (nes != null) maxAbs = Math.max(maxAbs, Math.abs(nes));
    }
  }
  return maxAbs > 0 ? maxAbs : 1;
}

function _f5ProfileCell(group, age, maxAbs) {
  const row = group.rows.get(Number(age));
  const status = _f5StatusFor(group, age);
  if (!row) {
    const title = `${age}mo: ${status || "no MEA row"}`;
    return `<div class="npc" title="${_f5Esc(title)}"></div>`;
  }
  const nes = _f5Num(row.NES);
  const q = _f5Num(row.FDR);
  const sig = q != null && q < Store.state.filters.fdr;
  let bg = "#fff";
  if (nes != null && maxAbs > 0) {
    const a = Math.min(1, Math.abs(nes) / maxAbs);
    const rgb = nes >= 0 ? [197, 48, 48] : [43, 108, 176];
    bg = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${(0.15 + 0.85 * a).toFixed(3)})`;
  }
  const qc = status ? `, status ${status}` : "";
  const title = `${age}mo: NES ${_f5Fmt(nes, 2)}, FDR ${q == null ? "—" : q.toExponential(1)}${sig ? " (sig)" : ""}${qc}`;
  return `<div class="npc${sig ? " sig" : ""}" style="background:${bg};" title="${_f5Esc(title)}"></div>`;
}

function _f5Profile(group, maxAbs) {
  const cells = _F5_AGES.map(age => _f5ProfileCell(group, age, maxAbs)).join("");
  return `<div class="nes-profile-wrap"><div class="nes-profile-cell" style="grid-template-columns:repeat(${_F5_AGES.length},1fr);">${cells}</div></div>`;
}

function _f5SetOptions(id, values, labels) {
  const el = document.getElementById(id);
  if (!el) return;
  const cur = el.value || "";
  const opts = ['<option value="">Any</option>'];
  values.forEach(v => {
    const lab = labels && labels[v] ? labels[v] : v;
    opts.push(`<option value="${_f5Esc(v)}">${_f5Esc(lab)}</option>`);
  });
  el.innerHTML = opts.join("");
  el.value = values.includes(cur) ? cur : "";
}

function _f5PopulateControls() {
  const block = _f5Block();
  if (!block) return;
  const filters = block.filters || {};
  _f5SetOptions("f5-filter-tissue", filters.tissue || []);
  _f5SetOptions("f5-filter-assay", filters.assay || []);
  _f5SetOptions("f5-filter-analysis", filters.analysis_track || [], {
    stoichiometry: "stoichiometry",
    raw_phospho: "raw phospho",
  });
  const ageVals = (filters.age_months || _F5_AGES).map(v => String(v));
  _f5SetOptions("f5-filter-age", ageVals, {3: "3mo", 6: "6mo", 9: "9mo", 12: "12mo"});
}

function _f5SyncControls() {
  const byId = {
    "f5-search": "search",
    "f5-filter-tissue": "tissue",
    "f5-filter-assay": "assay",
    "f5-filter-analysis": "analysis",
    "f5-filter-age": "age",
    "f5-filter-nsig": "nsigMin",
  };
  Object.entries(byId).forEach(([id, key]) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.value = key === "nsigMin" ? String(_F5State.nsigMin) : (_F5State[key] || "");
  });
}

function wireFiveXFADKinase() {
  if (_F5Wired) return;
  _F5Wired = true;
  const bind = (id, key, numeric) => {
    const el = document.getElementById(id);
    if (!el) return;
    const apply = () => {
      _F5State[key] = numeric ? Math.max(0, parseInt(el.value || "0", 10)) : el.value;
      renderFiveXFADKinase();
    };
    el.addEventListener("input", apply);
    el.addEventListener("change", apply);
  };
  bind("f5-search", "search");
  bind("f5-filter-tissue", "tissue");
  bind("f5-filter-assay", "assay");
  bind("f5-filter-analysis", "analysis");
  bind("f5-filter-age", "age");
  bind("f5-filter-nsig", "nsigMin", true);
  const reset = document.getElementById("f5-filter-reset");
  if (reset) reset.addEventListener("click", () => {
    Object.assign(_F5State, {
      search: "", tissue: "", assay: "", analysis: "", age: "",
      nsigMin: 0, sortCol: "peakAbsNes", sortAsc: false,
      selectedKey: null, auditTab: "mea-score", auditAge: null,
    });
    renderFiveXFADKinase();
  });
  const table = document.getElementById("f5-table");
  if (table) {
    table.querySelectorAll("thead th[data-col]").forEach(th => {
      th.addEventListener("click", () => {
        const col = th.dataset.col;
        if (_F5State.sortCol === col) _F5State.sortAsc = !_F5State.sortAsc;
        else { _F5State.sortCol = col; _F5State.sortAsc = false; }
        renderFiveXFADKinase();
      });
    });
    const tbody = table.querySelector("tbody");
    tbody.addEventListener("click", ev => {
      const tr = ev.target.closest("tr[data-f5-key]");
      if (!tr) return;
      _F5State.selectedKey = tr.dataset.f5Key;
      _F5State.auditAge = null;
      renderFiveXFADKinase();
    });
    tbody.addEventListener("keydown", ev => {
      if (ev.key !== "Enter" && ev.key !== " ") return;
      const tr = ev.target.closest("tr[data-f5-key]");
      if (!tr) return;
      ev.preventDefault();
      _F5State.selectedKey = tr.dataset.f5Key;
      _F5State.auditAge = null;
      renderFiveXFADKinase();
    });
  }
}

function renderFiveXFADKinase() {
  const panel = document.getElementById("tab-fivexfadkinase");
  if (!panel) return;
  if (typeof renderUnmetPrerequisite === "function" && renderUnmetPrerequisite(panel, "fivexfadkinase")) return;
  const block = _f5Block();
  if (!block || !Array.isArray(block.rows) || block.rows.length === 0) {
    const detail = document.getElementById("f5-detail");
    const count = document.getElementById("f5-count");
    if (count) count.textContent = "5xFAD payload unavailable";
    if (detail) detail.innerHTML = '<div class="muted">5xFAD payload data are not available in this viewer build.</div>';
    return;
  }
  _f5PopulateControls();
  _f5SyncControls();
  const rows = _f5FilteredRows();
  if (_F5State.selectedKey && !rows.some(r => r.key === _F5State.selectedKey)) {
    _F5State.selectedKey = null;
    _F5State.auditAge = null;
  }
  const count = document.getElementById("f5-count");
  if (count) count.textContent = `${rows.length.toLocaleString()} / ${_F5Groups.length.toLocaleString()} kinase slices`;
  const tbody = document.querySelector("#f5-table tbody");
  if (!tbody) return;
  const maxAbs = _f5MaxAbsVisible(rows);
  const ages = _f5AgeScope();
  const denom = ages.length;
  const html = rows.map(r => {
    const subs = r.substrateHits == null ? "—" : `${r.substrateHits}/${r.substrateUniverse == null ? "?" : r.substrateUniverse}`;
    const sel = r.key === _F5State.selectedKey ? " selected" : "";
    const sub = r.sigCount === 0 ? " sub-thresh" : "";
    const residueBadge = r.residue_type === "Y"
      ? ' <span class="track-badge track-y" title="Tyrosine kinase (pY track)">pY</span>'
      : "";
    return `<tr data-f5-key="${_f5Esc(r.key)}" tabindex="0" class="ke-row${sel}${sub}" aria-label="5xFAD kinase ${_f5Esc(r.kinase)}; ${r.sigCount} significant ages">
      <td>${_f5Esc(r.kinase)}${residueBadge}</td>
      <td>${_f5Esc(r.gene_symbol || "")}</td>
      <td>${_f5Esc(r.family || "")}</td>
      <td>${_f5Esc(r.residue_type || "")}</td>
      <td>${_f5Esc(r.slice)}</td>
      <td>${_f5Profile(r, maxAbs)}</td>
      <td class="attr-num">${r.peakAbsNes == null ? '<span class="muted">—</span>' : r.peakAbsNes.toFixed(2)}</td>
      <td class="attr-num">${r.sigCount}<span class="muted" style="font-size:10px;"> / ${denom}</span></td>
      <td>${subs}</td>
    </tr>`;
  }).join("");
  tbody.innerHTML = html || '<tr><td colspan="9" class="muted">No 5xFAD rows match the active filters.</td></tr>';
  _f5SyncSortIndicators();
  renderFiveXFADKinaseDetail();
}

function _f5SyncSortIndicators() {
  document.querySelectorAll("#f5-table thead th").forEach(th => {
    const c = th.dataset.col;
    th.textContent = th.textContent.replace(/[ ▲▼]+$/, "");
    if (c === _F5State.sortCol) th.textContent += _F5State.sortAsc ? " ▲" : " ▼";
  });
}

function _f5SelectedGroup() {
  _f5EnsureIndexes();
  return _F5Groups.find(g => g.key === _F5State.selectedKey) || null;
}

function _f5GroupsForKinase(name) {
  _f5EnsureIndexes();
  return _F5Groups.filter(g => g.kinase === name);
}

function _f5SelectedAge(group) {
  if (_F5State.auditAge && group.rows.has(Number(_F5State.auditAge))) return Number(_F5State.auditAge);
  if (_F5State.age && group.rows.has(Number(_F5State.age))) return Number(_F5State.age);
  let bestAge = null;
  let bestAbs = -1;
  for (const age of _F5_AGES) {
    const row = group.rows.get(age);
    const nes = row ? _f5Num(row.NES) : null;
    if (nes != null && Math.abs(nes) > bestAbs) {
      bestAbs = Math.abs(nes);
      bestAge = age;
    }
  }
  if (bestAge == null) bestAge = _F5_AGES.find(age => group.rows.has(age)) || _F5_AGES[0];
  _F5State.auditAge = bestAge;
  return bestAge;
}

function renderFiveXFADKinaseDetail() {
  const host = document.getElementById("f5-detail");
  if (!host) return;
  const group = _f5SelectedGroup();
  if (!group) {
    host.innerHTML = '<div class="muted">Select a kinase to see details.</div>';
    return;
  }
  const age = _f5SelectedAge(group);
  const sliceOptions = _f5GroupsForKinase(group.kinase).map(g =>
    `<option value="${_f5Esc(g.key)}"${g.key === group.key ? " selected" : ""}>${_f5Esc(_f5SliceLabel(g))}</option>`
  ).join("");
  const ageOptions = _F5_AGES.map(a => {
    const disabled = group.rows.has(a) ? "" : " disabled";
    return `<option value="${a}"${a === age ? " selected" : ""}${disabled}>${a}mo</option>`;
  }).join("");
  const tabButtons = _F5_AUDIT_TABS.map(t =>
    `<button type="button" data-f5-audit-tab="${t.id}" class="${t.id === _F5State.auditTab ? "active" : ""}">${_f5Esc(t.label)}</button>`
  ).join("");

  host.innerHTML = `
    <div class="kinase-workbench-header">
      <div class="kinase-workbench-title">
        <h3>${_f5Esc(group.kinase)}</h3>
        <div class="muted">${_f5Esc(group.gene_symbol || "")}${group.residue_type ? " · " + _f5Esc(group.residue_type) : ""}</div>
      </div>
      <div class="kinase-workbench-controls">
        <label>Slice <select id="f5-audit-slice">${sliceOptions}</select></label>
        <label>Age <select id="f5-audit-age">${ageOptions}</select></label>
      </div>
    </div>
    <div class="kinase-audit-tabs" role="tablist" aria-label="5xFAD kinase audit walkthrough">${tabButtons}</div>
    <div class="kinase-audit-tab-body" id="f5-audit-body"></div>
  `;

  const sliceSel = document.getElementById("f5-audit-slice");
  if (sliceSel) sliceSel.addEventListener("change", ev => {
    _F5State.selectedKey = ev.target.value;
    _F5State.auditAge = null;
    renderFiveXFADKinase();
  });
  const ageSel = document.getElementById("f5-audit-age");
  if (ageSel) ageSel.addEventListener("change", ev => {
    _F5State.auditAge = Number(ev.target.value);
    _f5RenderAuditBody(group);
  });
  host.querySelectorAll("[data-f5-audit-tab]").forEach(btn => {
    btn.addEventListener("click", () => {
      _F5State.auditTab = btn.dataset.f5AuditTab;
      renderFiveXFADKinaseDetail();
    });
  });
  _f5RenderAuditBody(group);
}

function _f5RenderAuditBody(group) {
  const body = document.getElementById("f5-audit-body");
  if (!body) return;
  if (_F5State.auditTab === "measurement-trace") return _f5RenderTrace(body, group);
  if (_F5State.auditTab === "ols-details") return _f5RenderOls(body, group);
  if (_F5State.auditTab === "mea-input") return _f5RenderPrep(body, group);
  return _f5RenderScore(body, group);
}

function _f5SelectedRow(group) {
  const age = _f5SelectedAge(group);
  return group.rows.get(age) || null;
}

function _f5ScoreTier(row) {
  const f = _f5Num(row && row.FDR);
  const gate = Store.state.filters.fdr;
  if (f == null) return {label: "no FDR", cls: "muted"};
  if (f < gate) return {label: `FDR ${f.toFixed(3)} · passes ${gate}`, cls: "chip-pass"};
  if (f < gate * 2) return {label: `FDR ${f.toFixed(3)} · borderline`, cls: "chip-borderline"};
  return {label: `FDR ${f.toFixed(3)} · fails ${gate}`, cls: "chip-fail"};
}

function _f5RenderScore(body, group) {
  const row = _f5SelectedRow(group);
  const age = _f5SelectedAge(group);
  const tier = _f5ScoreTier(row);
  const nes = _f5Num(row && row.NES);
  const nesColor = nes == null ? "#666" : (nes > 0 ? "#c53030" : "#2b6cb0");
  const allRows = _F5_AGES.map(a => {
    const r = group.rows.get(a);
    const status = _f5StatusFor(group, a);
    return `<tr>
      <td>${a}mo</td>
      <td>${_f5Fmt(r && r.NES, 2)}</td>
      <td>${_f5Fmt(r && r.FDR, 3)}</td>
      <td>${_f5Fmt(r && r.ES, 3)}</td>
      <td>${_f5Fmt(r && r.p_value, 4)}</td>
      <td>${r && r.substrate_hits != null ? `${r.substrate_hits}/${r.substrate_universe}` : "—"}</td>
      <td>${r && r.n_wt != null ? r.n_wt : "—"}</td>
      <td>${r && r.n_tg != null ? r.n_tg : "—"}</td>
      <td>${_f5Esc(status || "—")}</td>
    </tr>`;
  }).join("");
  body.innerHTML = `
    <section class="audit-panel">
      <h4>MEA Score <span class="muted">(${age}mo TG vs WT)</span></h4>
      <div class="mea-scorecard">
        <div class="mea-score-nes" style="color:${nesColor}">
          <div class="mea-score-label">NES</div>
          <div class="mea-score-value">${nes == null ? "—" : nes.toFixed(2)}</div>
          <div class="mea-score-chip ${tier.cls}">${_f5Esc(tier.label)}</div>
        </div>
        <dl class="mea-score-stats">
          <dt>ES</dt><dd>${_f5Fmt(row && row.ES, 3)}</dd>
          <dt>p-value</dt><dd>${_f5Fmt(row && row.p_value, 4)}</dd>
          <dt>Substrates tested</dt><dd>${row && row.substrate_hits != null ? `${row.substrate_hits}/${row.substrate_universe}` : "—"}</dd>
          <dt>Samples</dt><dd>WT ${row && row.n_wt != null ? row.n_wt : "—"} · TG ${row && row.n_tg != null ? row.n_tg : "—"}</dd>
          <dt>QC status</dt><dd>${_f5Esc(_f5StatusFor(group, age) || "—")}</dd>
        </dl>
      </div>
    </section>
    <section class="audit-panel" style="margin-top:10px;">
      <h4>Age profile</h4>
      <div class="kh-audit-tablewrap">
        <table class="data-table">
          <thead><tr><th>Age</th><th>NES</th><th>FDR</th><th>ES</th><th>p-value</th><th>Subs</th><th>WT</th><th>TG</th><th>QC</th></tr></thead>
          <tbody>${allRows}</tbody>
        </table>
      </div>
    </section>`;
}

function _f5SmallTable(rows, cols) {
  if (!rows || !rows.length) return '<div class="muted">No rows available.</div>';
  const head = cols.map(c => `<th>${_f5Esc(c.label)}</th>`).join("");
  const body = rows.map(r => `<tr>${cols.map(c => `<td>${_f5Esc(r[c.key] == null ? "" : r[c.key])}</td>`).join("")}</tr>`).join("");
  return `<div class="kh-audit-tablewrap"><table class="data-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
}

function _f5SourceList() {
  const files = (_f5Block() && _f5Block().source_files) || [];
  if (!files.length) return '<div class="muted">No source files listed in the payload.</div>';
  return `<ul style="margin:4px 0 0 18px;padding:0;">${files.map(f => `<li><code>${_f5Esc(f)}</code></li>`).join("")}</ul>`;
}

function _f5RenderPrep(body, group) {
  const age = _f5SelectedAge(group);
  const block = _f5Block() || {};
  const qcRows = (block.contrast_qc || []).filter(r =>
    r.tissue === group.tissue && r.assay === group.assay && r.age_months === age
  );
  const sampleRows = (block.sample_counts || []).filter(r =>
    r.tissue === group.tissue && r.assay === group.assay && Number(r.age) === age
  );
  body.innerHTML = `
    <section class="audit-panel">
      <h4>Contrast QC <span class="muted">(${_f5Esc(_f5SliceLabel(group))}, ${age}mo)</span></h4>
      <p class="kinase-stage-note">Categorical contrast status and biological sample counts are shown as QC context only; they are not converted into viewer-facing analysis scores.</p>
      ${_f5SmallTable(qcRows, [
        {key: "contrast", label: "Contrast"},
        {key: "residue_type", label: "Res"},
        {key: "n_wt", label: "WT"},
        {key: "n_tg", label: "TG"},
        {key: "contrast_status", label: "Status"},
      ])}
    </section>
    <section class="audit-panel" style="margin-top:10px;">
      <h4>Sample counts</h4>
      ${_f5SmallTable(sampleRows, [
        {key: "tissue", label: "Tissue"},
        {key: "assay", label: "Assay"},
        {key: "age", label: "Age"},
        {key: "genotype", label: "Genotype"},
        {key: "n_biological_samples", label: "n"},
      ])}
    </section>
    <section class="audit-panel" style="margin-top:10px;">
      <h4>Packaged source files</h4>
      ${_f5SourceList()}
    </section>`;
}

function _f5RenderOls(body, group) {
  const age = _f5SelectedAge(group);
  const row = _f5SelectedRow(group);
  body.innerHTML = `
    <section class="audit-panel">
      <h4>OLS Details <span class="muted">(${age}mo TG vs WT)</span></h4>
      <p class="kinase-stage-note">The current 5xFAD viewer payload packages summary MEA rows and contrast QC for this cohort. Full site-level OLS rows are generated by the 5xFAD analysis workflow but are not embedded in this viewer payload.</p>
      ${_f5SmallTable(row ? [row] : [], [
        {key: "contrast", label: "Contrast"},
        {key: "NES", label: "NES"},
        {key: "FDR", label: "FDR"},
        {key: "ES", label: "ES"},
        {key: "p_value", label: "p-value"},
        {key: "substrate_hits", label: "Substrate hits"},
        {key: "substrate_universe", label: "Substrate universe"},
      ])}
    </section>`;
}

function _f5RenderTrace(body, group) {
  const age = _f5SelectedAge(group);
  body.innerHTML = `
    <section class="audit-panel">
      <h4>Measurement Trace <span class="muted">(${age}mo TG vs WT)</span></h4>
      <p class="kinase-stage-note">Normalized phosphosite, matched protein, and stoichiometry matrices are retained as 5xFAD analysis artifacts, but the unified viewer payload does not currently embed per-site measurement matrices for this cohort. The table remains in the standard audit workbench position so the absence is explicit rather than hidden.</p>
      <div class="muted">No per-site measurement trace is packaged for ${_f5Esc(group.kinase)} in ${_f5Esc(_f5SliceLabel(group))}.</div>
    </section>`;
}
