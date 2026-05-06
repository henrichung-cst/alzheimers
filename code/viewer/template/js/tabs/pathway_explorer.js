function renderPathwayExplorer() {
  const tbody = document.querySelector("#pe-table tbody");
  if (!tbody) return;
  _ensurePathwayIndexes();
  const f = Store.state.filters;
  const cIdx = CONTRASTS.indexOf(f.contrast);
  const selBid = Store.state.selection.backbone;
  const q = peSearch.trim().toLowerCase();
  const baseIdx = getFilteredIndices();

  const tpdsMin = Math.max(0, Number(Store.state.view.pathwayScoreMin) || 0);
  const visible = [];
  for (const i of baseIdx) {
    const r = _peRows[i];
    if (r.sig_count === 0) continue;  // hard sig-both gate (formerly checkbox)
    if (!_peCsetMatch(r.sig_mask)) continue;
    if (tpdsMin > 0) {
      const mt = r.max_abs_tpds;
      if (mt == null || mt < tpdsMin) continue;
    }
    if (q && !(r.Receptor.toLowerCase().includes(q) ||
               r.EM.toLowerCase().includes(q) ||
               r.Target.toLowerCase().includes(q))) continue;
    visible.push(r);
  }
  visible.sort((a, b) => _peCompare(a, b, cIdx));

  document.querySelectorAll("#pe-table thead th").forEach(th => {
    const c = th.dataset.col;
    th.textContent = th.textContent.replace(/[ ▲▼]+$/, "");
    if (c === peSortCol) th.textContent += peSortAsc ? " ▲" : " ▼";
  });

  const CAP = 2000;
  const shown = visible.slice(0, CAP);
  const parts = [];
  for (const r of shown) {
    const selCls = r.id === selBid ? " selected" : "";
    const t = cIdx >= 0 ? r._tpds[cIdx] : r.max_abs_tpds;
    const tStr = (t == null) ? "—" : t.toFixed(3);
    const evidence = cIdx >= 0
      ? (PAYLOAD.backbones["pathway_evidence_backbone_" + f.contrast][r.idx] || "expression-confirmed")
      : r.pathway_evidence_all;
    const evidenceLabel = _pathwayEvidenceLabel(evidence);
    r.pathway_evidence_label = evidenceLabel;
    parts.push(
      `<tr class="pe-row${selCls}" data-bid="${r.id}" tabindex="0" ` +
      `aria-label="Backbone ${r.Receptor} to ${r.EM} to ${r.Target}; receiver ${r.receiver}; support ${evidenceLabel}; TPDS ${tStr}; ${r.sig_count} passing-null contrasts">` +
      `<td>${r.receiver}</td>` +
      `<td>${r.Receptor}</td>` +
      `<td>${r.EM}</td>` +
      `<td>${r.Target}</td>` +
      `<td>${_pathwayEvidenceBadge(evidence)}</td>` +
      `<td>${tStr}</td>` +
      `<td class="pe-cchip-cell">${_peContrastChips(r.sig_mask)}</td>` +
      `<td>${r.n_senders_sig}/${r.n_senders}</td>` +
      `<td>${r.max_abs_tpds == null ? "—" : r.max_abs_tpds.toFixed(3)}</td>` +
      `</tr>`
    );
  }
  tbody.innerHTML = parts.join("");
  const countEl = document.getElementById("pe-count");
  if (countEl) {
    const cap = visible.length > CAP ? ` (first ${CAP} shown)` : "";
    countEl.textContent = `${visible.length.toLocaleString()} / ${_peRows.length.toLocaleString()} backbones${cap}`;
  }
}

function _updatePathwayRowSelection(bid) {
  _updateRowSelection("#pe-table", "pe-row", "data-bid", bid);
}

function renderPathwayDetail(backbone_id) {
  const el = document.getElementById("pe-detail");
  if (!el) return;
  if (backbone_id == null) {
    el.innerHTML = '<div class="muted">Select a backbone to see details.</div>';
    return;
  }
  _ensurePathwayIndexes();
  const BB = PAYLOAD.backbones;
  const i = _backboneIdxById.get(backbone_id);
  if (i == null) {
    el.innerHTML = '<div class="muted">Backbone not found.</div>';
    return;
  }
  const receiver = RECEIVERS[BB.receiver_id[i]];
  const rcp = BB.Receptor[i] || "—";
  const em = BB.EM[i] || "—";
  const tgt = BB.Target[i] || "—";
  const sigMask = BB.significant_both_mask[i];
  const evidenceSummary = BB.all_contrasts_pathway_evidence[i] || "expression-confirmed";
  const allImputedNodes = BB.all_imputed_nodes_union[i] || "";
  const nExpr = BB.all_n_expression_confirmed[i] || 0;
  const nImp = BB.all_n_kinase_imputed[i] || 0;
  const chips = CONTRASTS.map((c, ci) => {
    const on = ((sigMask >> ci) & 1) ? " on" : "";
    return `<span class="pe-chip${on}">${c}</span>`;
  }).join("");
  const evidenceChips = CONTRASTS.map((c) => {
    const ev = BB["pathway_evidence_backbone_" + c][i];
    if (!ev) return "";
    return _pathwayEvidenceChip(ev, `${c}: ${_pathwayEvidenceLabel(ev)}`);
  }).join("");
  const nSendSig = BB.n_senders_significant[i];
  const nSend = BB.n_senders[i];

  el.innerHTML =
    `<h3>${rcp} → ${em} → ${tgt}</h3>` +
    `<div class="meta">Receiver: ${receiver} · Senders: ${nSendSig}/${nSend} sig · Support: ${_pathwayEvidenceLabel(evidenceSummary)} · Expression-confirmed paths: ${nExpr} · Kinase-imputed paths: ${nImp} · Imputed positions observed: ${allImputedNodes || "none"}</div>` +
    `<div class="detail-chips">${_pathwayEvidenceBadge(evidenceSummary)}</div>` +
    `<h4>Passed both nulls by contrast <span class="metric-help" tabindex="0" data-metric="passedNulls" title="${_metricShort('passedNulls')}">i</span></h4><div>${chips}</div>` +
    `<h4>Pathway support by contrast <span class="metric-help" tabindex="0" data-metric="pathwaySupportH" title="${_metricShort('pathwaySupportH')}">i</span></h4><div>${evidenceChips || '<span class="muted">No support provenance available.</span>'}</div>` +
    `<h4>TPDS across contrasts <span class="metric-help" tabindex="0" data-metric="tpdsCross" title="${_metricShort('tpdsCross')}">i</span></h4><div id="pe-detail-cross"></div>` +
    `<h4>Driving kinases <span class="metric-help" tabindex="0" data-metric="drivingKinasesH" title="${_metricShort('drivingKinasesH')}">i</span></h4><div id="pe-detail-kinases" class="muted">loading…</div>`;

  const tpds = CONTRASTS.map(c => BB["mean_tpds_" + c][i]);
  const barColors = tpds.map(v => {
    if (v == null || v === 0) return "#cfd8dc";
    return v > 0 ? "var(--up-red)" : "var(--down-blue)";
  });
  const outlines = CONTRASTS.map((_, ci) =>
    ((sigMask >> ci) & 1) ? "#000" : "rgba(0,0,0,0)");
  Plotly.react("pe-detail-cross", [{
    type: "bar", x: CONTRASTS, y: tpds.map(v => v == null ? 0 : v),
    marker: { color: barColors, line: { color: outlines, width: 1.5 } },
    hovertemplate: "%{x}<br>TPDS %{y:.3f}<extra></extra>",
  }], {
    margin:{l:40,r:10,t:6,b:60}, height:180,
    yaxis:{zeroline:true, zerolinecolor:"#bbb"},
    xaxis:{tickangle:-35},
  }, {displaylogo:false, responsive:true});

  renderPathwayKinases(backbone_id);
}

async function renderPathwayKinases(backbone_id) {
  const container = document.getElementById("pe-detail-kinases");
  if (!container) return;
  _ensurePathwayIndexes();
  const bi = _backboneIdxById.get(backbone_id);
  if (bi == null) {
    container.innerHTML = '<div class="muted">Backbone not found.</div>';
    return;
  }
  if (PAYLOAD.backbones.significant_both_mask[bi] === 0) {
    container.innerHTML = '<div class="muted">No significant kinase edges.</div>';
    return;
  }
  let rows;
  try {
    rows = await SliceCache.backboneEdges(backbone_id);
  } catch (e) {
    if (Store.state.selection.backbone !== backbone_id) return;
    container.innerHTML = `<div class="muted">Failed to load: ${e.message}</div>`;
    return;
  }
  if (Store.state.selection.backbone !== backbone_id) return;

  const f = Store.state.filters;
  const cIdx = CONTRASTS.indexOf(f.contrast);
  const filtered = (cIdx >= 0)
    ? rows.filter(r => r.contrast_id === cIdx)
    : rows;

  const byK = new Map();
  for (const r of filtered) {
    let g = byK.get(r.kinase_id);
    if (!g) { g = { sum_abs:0, net:0, up:0, down:0, n:0 }; byK.set(r.kinase_id, g); }
    const s = Math.abs(r.support_contribution);
    const sign = (r.concordance > 0) ? 1 : (r.concordance < 0 ? -1 : 0);
    g.sum_abs += s;
    g.net += sign * s;
    if (sign > 0) g.up++;
    else if (sign < 0) g.down++;
    g.n++;
  }
  const groups = Array.from(byK.entries()).map(([kid, g]) => ({ kid, ...g }));
  groups.sort((a, b) => b.sum_abs - a.sum_abs);

  _ensureKinaseIdx();
  const K = PAYLOAD.kinases;
  const famMap = META.familyMap || {};

  const TOP = 200;
  const shown = groups.slice(0, TOP);
  const header = cIdx >= 0
    ? `Showing ${shown.length} of ${groups.length} kinases (contrast ${f.contrast}).`
    : `Showing ${shown.length} of ${groups.length} kinases (all contrasts).`;
  const parts = [
    `<div class="muted">${header}</div>`,
    '<table class="data-table"><thead><tr>',
    `<th data-metric="kinaseName" title="${_metricShort('kinaseName')}">Kinase</th>`,
    `<th data-metric="kinaseFamily" title="${_metricShort('kinaseFamily')}">Family</th>`,
    `<th data-metric="support" title="${_metricShort('support')}">Support</th>`,
    `<th data-metric="drivingDirection" title="${_metricShort('drivingDirection')}">Direction</th>`,
    `<th data-metric="trend" title="${_metricShort('trend')}">Trend</th>`,
    '</tr></thead><tbody>',
  ];
  for (const g of shown) {
    const kIdx = _kinaseIdxById.get(g.kid);
    const name = kIdx != null ? K.name[kIdx] : `kid:${g.kid}`;
    const fam = famMap[name] || "";
    const conc = (g.up > g.down) ? "↑" : (g.down > g.up ? "↓" : "—");
    parts.push(
      `<tr><td>${name}</td><td>${fam}</td>` +
      `<td>${g.sum_abs.toFixed(3)}</td>` +
      `<td>${g.net.toFixed(3)}</td>` +
      `<td>${conc} (${g.up}/${g.down})</td></tr>`
    );
  }
  parts.push("</tbody></table>");
  parts.push('<div class="muted" style="margin-top:6px;">Top 200 kinases shown. Open the How-to-read drawer for column meanings.</div>');
  container.innerHTML = parts.join("");
}

function wirePathwayTable() {
  const tbl = document.getElementById("pe-table");
  if (!tbl) return;
  tbl.querySelectorAll("thead th").forEach(th => {
    th.addEventListener("click", () => {
      const col = th.dataset.col;
      if (peSortCol === col) peSortAsc = !peSortAsc;
      else { peSortCol = col; peSortAsc = false; }
      renderPathwayExplorer();
    });
  });
  tbl.querySelector("tbody").addEventListener("click", ev => {
    const tr = ev.target.closest("tr.pe-row");
    if (!tr) return;
    const bid = parseInt(tr.dataset.bid, 10);
    Store.dispatch({type:"SET_SELECTION", key:"backbone", value: bid});
  });
  tbl.querySelector("tbody").addEventListener("keydown", ev =>
    _activateRowOnKey(ev, "tr.pe-row", tr => {
      const bid = parseInt(tr.dataset.bid, 10);
      Store.dispatch({type:"SET_SELECTION", key:"backbone", value: bid});
    }));
  const search = document.getElementById("pe-search");
  if (search) search.addEventListener("input", ev => {
    const val = ev.target.value;
    if (_peSearchTimer) clearTimeout(_peSearchTimer);
    _peSearchTimer = setTimeout(() => {
      peSearch = val;
      renderPathwayExplorer();
    }, 250);
  });
  _renderPeTrajectoryButtons();
  const tpdsInp = document.getElementById("pe-tpds-min");
  if (tpdsInp) {
    tpdsInp.value = Store.state.view.pathwayScoreMin || 0;
    tpdsInp.addEventListener("change", ev => {
      const v = Math.max(0, parseFloat(ev.target.value) || 0);
      Store.dispatch({type:"SET_VIEW", key:"pathwayScoreMin", value: v});
    });
  }
}

function _renderPeTrajectoryButtons() {
  const host = document.getElementById("pe-traj-buttons");
  if (!host) return;
  const order = ["all", "App", "Tau", "ApTt", "2mo", "4mo", "6mo"];
  host.innerHTML = order.map(k => {
    const t = PE_TRAJECTORIES[k];
    const on = peTrajectory === k;
    const tip = t.contrasts.length
      ? `Show backbones passing in any of ${t.contrasts.join(", ")}.`
      : "Show all passing backbones (no trajectory filter).";
    return `<button type="button" class="pe-cset-chip${on ? " on" : ""}" data-k="${k}" aria-pressed="${on}" title="${tip}">${t.label}</button>`;
  }).join("");
  host.querySelectorAll(".pe-cset-chip").forEach(btn => {
    btn.addEventListener("click", () => {
      peTrajectory = btn.dataset.k;
      _peTrajMaskCache = null;
      _renderPeTrajectoryButtons();
      renderPathwayExplorer();
    });
  });
}

// ---------------------------------------------------------------------------
// Pathway Graph (Cytoscape) — aggregates filtered backbones into an
// R → EM → T node DAG where each node is a unique gene across many backbones.
// ---------------------------------------------------------------------------
const GRAPH_MAX_NODES = 600;
const GRAPH_COLORS = { "Receptor":"#43a047", "EM":"#fb8c00", "Target":"#5c6bc0" };

let _cyInstance = null;
let _nodeInfo = null;  // Map<nodeId, {bbs:number[], scoreSum, scoreN, nUp, nDown}>

