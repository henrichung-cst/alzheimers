
function _selectedAuditContrast(K, ki) {
  // Audit panel's Contrast picker drives this. Falls back to peak_NES when
  // picker = "ALL". Independent of the left-list KinaseFilter.
  const f = Store.state && Store.state.filters && Store.state.filters.contrast;
  if (f && f !== "ALL" && CONTRASTS.indexOf(f) >= 0) return f;
  return K.peak_contrast[ki] || CONTRASTS[0];
}

function _siteOlsColumns(contrast) {
  return ["site_id", "gene_symbol", "motif", "stoich_lfc_" + contrast,
          "stoich_fdr_" + contrast, "raw_lfc_" + contrast, "raw_fdr_" + contrast,
          "n_obs_stoich", "matched_protein"];
}

async function _renderAuditTable(hostId, tableKey, rows, columns, sourceKey) {
  const t = new AuditTable(hostId, {tableKey, rows, columns, fullSourceKey: sourceKey === false ? false : (sourceKey || tableKey)});
  t.render();
  return t;
}

const KINASE_AUDIT_TABS = [
  {id:"measurement-trace", label:"Measurement Trace"},
  {id:"site-stats", label:"OLS Details"},
  {id:"mea-input", label:"MEA Preparation"},
  {id:"mea-score", label:"MEA Score"},
  {id:"attribution", label:"Attribution"},
];

function _activeKinaseAuditTab() {
  const id = Store.state.view.kinaseAuditTab || KINASE_AUDIT_TABS[0].id;
  return KINASE_AUDIT_TABS.some(t => t.id === id) ? id : KINASE_AUDIT_TABS[0].id;
}

function _selectedAuditSample() {
  return document.getElementById("audit-sample-select")?.value || "plex2_130c_sn_mean";
}

function _selectedAuditSite() {
  return document.getElementById("audit-site-select")?.value || "2488";
}

function _existingCols(rows, cols) {
  return cols.filter(c => rows.some(r => Object.prototype.hasOwnProperty.call(r, c)));
}

function _selectedSiteRows(rows, siteIds, limit) {
  const sid = _selectedAuditSite();
  let out = rows.filter(r => String(r.site_id) === String(sid));
  if (!out.length && siteIds && siteIds.size) out = rows.filter(r => siteIds.has(String(r.site_id)));
  return out.slice(0, limit || 200);
}

function _substrateSiteRows(rows, siteIds, limit) {
  const sid = _selectedAuditSite();
  let out = (siteIds && siteIds.size) ? rows.filter(r => siteIds.has(String(r.site_id))) : [];
  if (!out.length) out = rows.filter(r => String(r.site_id) === String(sid));
  out.sort((a, b) => {
    const as = String(a.site_id) === String(sid) ? 1 : 0;
    const bs = String(b.site_id) === String(sid) ? 1 : 0;
    return bs - as;
  });
  return out.slice(0, limit || 500);
}

function _leadingSubstrateRows(leadRow) {
  return String(leadRow["Leading substrates"] || "")
    .split(";").map(_normMotif).filter(Boolean)
    .map((motif, i) => ({rank:i + 1, substrate_motif:motif}));
}

function _shiftFor(globalRows, contrast) {
  const r = (globalRows || []).find(x => x.contrast === contrast);
  if (!r) return null;
  const v = Number(r.median_shift);
  return Number.isFinite(v) ? v : null;
}

function _winsorBoundsFor(winsorRows, contrast) {
  const r = (winsorRows || []).find(x => x.contrast === contrast);
  if (!r) return null;
  const lo = Number(r.lower_bound), hi = Number(r.upper_bound);
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return null;
  return [lo, hi];
}

function _winsorContrastSummary(winsorRows, contrast) {
  const filt = (winsorRows || []).filter(x => x.contrast === contrast);
  return {n: filt.length, rows: filt};
}

const _preRankCache = new Map();
function _ensurePreRank(contrast, olsRows, stoichMatrix, shift, bounds) {
  const key = contrast;
  if (_preRankCache.has(key)) return _preRankCache.get(key);
  if (shift == null || !bounds || !Array.isArray(olsRows) || !Array.isArray(stoichMatrix)) {
    return null;
  }
  const motifBySite = new Map();
  for (const r of stoichMatrix) {
    const m = _normMotif(r.motif);
    if (m) motifBySite.set(String(r.site_id), m);
  }
  const lfcCol = "stoich_lfc_" + contrast;
  const [lo, hi] = bounds;
  const ranked = [];
  for (const r of olsRows) {
    const sid = String(r.site_id);
    const motif = motifBySite.get(sid);
    if (!motif) continue;
    const v = r[lfcCol];
    if (v == null || v === "") continue;
    const raw = Number(v);
    if (!Number.isFinite(raw)) continue;
    const centered = raw - shift;
    const clipped = Math.min(Math.max(centered, lo), hi);
    ranked.push({sid, motif, clipped, gene_symbol: r.gene_symbol || ""});
  }
  ranked.sort((a, b) => b.clipped - a.clipped);
  const rankMap = new Map();
  for (let i = 0; i < ranked.length; i++) rankMap.set(ranked[i].sid, i + 1);
  const out = {rankMap, ranked, total: ranked.length};
  _preRankCache.set(key, out);
  return out;
}

function _computeRunningES(ranked, substrateMotifs) {
  const N = ranked.length;
  if (!N || !substrateMotifs || !substrateMotifs.size) return null;
  const hits = new Array(N);
  let Nh = 0, sumHitWeights = 0;
  for (let i = 0; i < N; i++) {
    const isHit = substrateMotifs.has(_normMotif(ranked[i].motif));
    hits[i] = isHit;
    if (isHit) { Nh += 1; sumHitWeights += Math.abs(ranked[i].clipped); }
  }
  if (Nh === 0 || Nh === N) return null;
  const missStep = 1 / (N - Nh);
  const running = new Array(N);
  const hitIndices = [];
  let es = 0, peakES = 0, peakIdx = 0;
  for (let i = 0; i < N; i++) {
    if (hits[i]) {
      es += sumHitWeights > 0 ? Math.abs(ranked[i].clipped) / sumHitWeights : 0;
      hitIndices.push(i);
    } else {
      es -= missStep;
    }
    running[i] = es;
    if (Math.abs(es) > Math.abs(peakES)) { peakES = es; peakIdx = i; }
  }
  const leadingEdge = peakES >= 0
    ? hitIndices.filter(i => i <= peakIdx)
    : hitIndices.filter(i => i >= peakIdx);
  return {running, hitIndices, peakES, peakIdx, leadingEdge, N, Nh};
}

function _buildMeaComparisonRows(leadRow, rawRow) {
  const num = (v) => {
    if (v == null || v === "") return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  };
  const fmt = (v, d=3) => v == null ? "—" : v.toFixed(d);
  const fmtSigned = (v, d=3) => v == null ? "—" : (v > 0 ? "+" : "") + v.toFixed(d);
  const stoichVals = {
    ES: num(leadRow && leadRow.ES),
    NES: num(leadRow && leadRow.NES),
    p: num(leadRow && leadRow["p-value"]),
    FDR: num(leadRow && leadRow.FDR),
    subs: leadRow && leadRow["Subs fraction"] || "",
  };
  const rawVals = {
    ES: num(rawRow && rawRow.ES),
    NES: num(rawRow && rawRow.NES),
    p: num(rawRow && rawRow["p-value"]),
    FDR: num(rawRow && rawRow.FDR),
    subs: rawRow && rawRow["Subs fraction"] || "",
  };
  const delta = (a, b) => (a == null || b == null) ? null : a - b;
  return [
    {metric:"ES",                  stoich: fmt(stoichVals.ES),  raw: fmt(rawVals.ES),  delta: fmtSigned(delta(stoichVals.ES,  rawVals.ES))},
    {metric:"NES",                 stoich: fmt(stoichVals.NES, 2), raw: fmt(rawVals.NES, 2), delta: fmtSigned(delta(stoichVals.NES, rawVals.NES), 2)},
    {metric:"p-value",             stoich: fmt(stoichVals.p, 4), raw: fmt(rawVals.p, 4), delta: fmtSigned(delta(stoichVals.p,   rawVals.p), 4)},
    {metric:"FDR",                 stoich: fmt(stoichVals.FDR, 3), raw: fmt(rawVals.FDR, 3), delta: fmtSigned(delta(stoichVals.FDR, rawVals.FDR), 3)},
    {metric:"Substrates tested",   stoich: stoichVals.subs || "—", raw: rawVals.subs || "—", delta: "—"},
  ];
}

function _diagnoseRawAbsence(ctx, rawRow) {
  if (rawRow && rawRow.contrast) return null;
  const meaRaw = ctx.meaRaw || [];
  if (!meaRaw.length) {
    return {kind:"file_missing", note:"No raw-phospho MEA loaded for this kinase. Run <code>pixi run python code/kinase_attribution.py --mechanism-annotation</code> to generate <code>mea_raw_phospho.csv</code> (and <code>mea_raw_phospho_pY.csv</code> for tyrosine kinases)."};
  }
  const contrasts = new Set(meaRaw.map(r => r.contrast));
  if (!contrasts.has(ctx.contrast)) {
    return {kind:"contrast_missing", note:`Raw-phospho MEA exists for ${meaRaw.length} other contrast(s) of this kinase but not <strong>${_escapeHtml(ctx.contrast)}</strong>. The raw-phospho file may have been generated under an older contrast set; rerun <code>--mechanism-annotation</code> to refresh.`};
  }
  return {kind:"unknown", note:"Raw-phospho row not found for the selected kinase × contrast."};
}

function _renderMeaScorecard(hostId, leadRow, rawRow, ctx) {
  const host = document.getElementById(hostId);
  if (!host) return;
  const fdrThresh = (Store.state.filters && Store.state.filters.fdr) || 0.25;
  const fmt = (v, d=3) => {
    if (v == null || v === "") return "—";
    const n = Number(v);
    return Number.isFinite(n) ? n.toFixed(d) : String(v);
  };
  const tier = (() => {
    const f = Number(leadRow && leadRow.FDR);
    if (!Number.isFinite(f)) return {label:"no FDR", cls:"muted"};
    if (f < fdrThresh) return {label:`FDR ${f.toFixed(3)} · passes ${fdrThresh}`, cls:"chip-pass"};
    if (f < fdrThresh * 2) return {label:`FDR ${f.toFixed(3)} · borderline`, cls:"chip-borderline"};
    return {label:`FDR ${f.toFixed(3)} · fails ${fdrThresh}`, cls:"chip-fail"};
  })();
  const nesVal = leadRow ? Number(leadRow.NES) : null;
  const nesColor = (nesVal == null || !Number.isFinite(nesVal)) ? "#666"
    : (nesVal > 0 ? "#1f77b4" : "#d62728");
  const subsFrac = leadRow ? leadRow["Subs fraction"] : "";
  const rawNes = rawRow ? rawRow.NES : null;
  const rawFdr = rawRow ? rawRow.FDR : null;
  host.innerHTML = `
    <div class="mea-scorecard">
      <div class="mea-score-nes" style="color:${nesColor}">
        <div class="mea-score-label">NES</div>
        <div class="mea-score-value">${nesVal == null || !Number.isFinite(nesVal) ? "—" : nesVal.toFixed(2)}</div>
        <div class="mea-score-chip ${tier.cls}">${_escapeHtml(tier.label)}</div>
      </div>
      <dl class="mea-score-stats">
        <dt>ES</dt><dd>${fmt(leadRow && leadRow.ES)}</dd>
        <dt>p-value</dt><dd>${fmt(leadRow && leadRow["p-value"], 4)}</dd>
        <dt>Substrates tested</dt><dd>${_escapeHtml(subsFrac || "—")}<span class="muted"> (kinase substrates &cap; contrast prerank)</span></dd>
        <dt>Raw phospho NES</dt><dd>${fmt(rawNes)}<span class="muted"> · FDR ${fmt(rawFdr, 3)}</span></dd>
      </dl>
    </div>`;
}

function _renderRunningEnrichmentPlot(hostId, ctx) {
  const host = document.getElementById(hostId);
  if (!host) return;
  const shift = _shiftFor(ctx.globalRows, ctx.contrast);
  const bounds = _winsorBoundsFor(ctx.winsorRows, ctx.contrast);
  const prerank = _ensurePreRank(ctx.contrast, ctx.olsRows, ctx.stoichMatrix, shift, bounds);
  if (!prerank || !prerank.ranked || !prerank.ranked.length) {
    host.innerHTML = `<div class="muted" style="padding:1em">Running enrichment requires the full prerank list (site_level_ols + mea_global_shift + winsorized_sites). Under file:// the audit tables are preview-only — serve the viewer directory over HTTP to view this plot.</div>`;
    return;
  }
  if (!ctx.substrateMotifs || !ctx.substrateMotifs.size) {
    host.innerHTML = `<div class="muted" style="padding:1em">No substrate-set motifs found for ${_escapeHtml(ctx.name)} on ${_escapeHtml(ctx.contrast)} (mea_substrate_sets.csv).</div>`;
    return;
  }
  const r = _computeRunningES(prerank.ranked, ctx.substrateMotifs);
  if (!r) {
    host.innerHTML = `<div class="muted" style="padding:1em">Running enrichment unavailable: kinase has no substrate hits in the contrast prerank.</div>`;
    return;
  }
  const ranks = new Array(r.N);
  for (let i = 0; i < r.N; i++) ranks[i] = i + 1;
  const hitX = r.hitIndices.map(i => i + 1);
  const hitY = r.hitIndices.map(i => r.running[i]);
  const hitText = r.hitIndices.map(i => {
    const e = prerank.ranked[i];
    return `rank ${i + 1}<br>${_escapeHtml(e.gene_symbol || "")} · ${_escapeHtml(e.motif)}<br>clipped LFC ${e.clipped.toFixed(3)}<br>running ES ${r.running[i].toFixed(3)}`;
  });
  const peakX = r.peakIdx + 1;
  const peakY = r.peakES;
  const leShape = r.peakES >= 0
    ? {x0: 1, x1: peakX, y0: 0, y1: 1}
    : {x0: peakX, x1: r.N, y0: 0, y1: 1};
  Plotly.react(hostId, [
    {type:"scatter", mode:"lines", x: ranks, y: r.running,
     line:{color:"#1f77b4", width:1.5}, name:"running ES", hoverinfo:"skip"},
    {type:"scatter", mode:"markers", x: hitX, y: hitY,
     marker:{color:"#1f77b4", size:5, opacity:0.9}, name:"substrate hit",
     text: hitText, hovertemplate:"%{text}<extra></extra>"},
    {type:"scatter", mode:"markers", x:[peakX], y:[peakY],
     marker:{color:"#000", size:9, symbol:"diamond"}, name:"peak ES",
     hovertemplate:`peak ES ${peakES_safe(peakY)} at rank ${peakX}<extra></extra>`},
  ], {
    margin:{l:50, r:10, t:30, b:40}, height:300,
    showlegend:false,
    annotations:[{
      x: peakX, y: peakY, xref:"x", yref:"y",
      text: `peak ES ${peakY.toFixed(3)} at rank ${peakX}<br>leading edge: ${r.leadingEdge.length} of ${r.Nh} hits`,
      showarrow:true, arrowhead:2, ax: 30, ay: peakY >= 0 ? -40 : 40,
      font:{size:11},
    }],
    shapes:[{
      type:"rect", xref:"x", yref:"paper",
      x0: leShape.x0, x1: leShape.x1, y0: 0, y1: 1,
      fillcolor:"#1f77b4", opacity:0.08, line:{width:0},
    }, {
      type:"line", xref:"x", yref:"y",
      x0: 1, x1: r.N, y0: 0, y1: 0,
      line:{color:"#999", width:1, dash:"dot"},
    }],
    xaxis:{title:"prerank rank (1 = most up-shifted)", range:[1, r.N]},
    yaxis:{title:"running ES", zeroline:false},
  }, {displaylogo:false, responsive:true});
}
function peakES_safe(v) { return Number.isFinite(v) ? v.toFixed(3) : "—"; }

function _renderMeaTrajectory(hostId, kinase_id, ctx) {
  const host = document.getElementById(hostId);
  if (!host) return;
  _ensureKinaseIndexes();
  const K = PAYLOAD.kinases;
  const i = _kinaseIdxById.get(kinase_id);
  if (i == null) return;
  const fdrThresh = (Store.state.filters && Store.state.filters.fdr) || 0.25;
  const stoichNes = CONTRASTS.map(c => K["NES_" + c][i]);
  const stoichFdr = CONTRASTS.map(c => K["FDR_" + c][i]);
  const rawByContrast = new Map((ctx.meaRaw || []).map(r => [r.contrast, r]));
  const rawNes = CONTRASTS.map(c => {
    const r = rawByContrast.get(c);
    if (!r) return null;
    const v = Number(r.NES);
    return Number.isFinite(v) ? v : null;
  });
  const _hexToRgba = (hex, alpha) => {
    const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex || "");
    if (!m) return hex;
    return `rgba(${parseInt(m[1],16)},${parseInt(m[2],16)},${parseInt(m[3],16)},${alpha})`;
  };
  const colors = CONTRASTS.map((c, ci) => {
    const base = _diseaseColorFor(c);
    const sig = stoichFdr[ci] != null && stoichFdr[ci] < fdrThresh;
    return sig ? base : _hexToRgba(base, 0.28);
  });
  const selectedIdx = CONTRASTS.indexOf(ctx.contrast);
  const outlines = CONTRASTS.map((_, i) => i === selectedIdx ? "#000" : "rgba(0,0,0,0)");
  const barLineWidth = CONTRASTS.map((_, i) => i === selectedIdx ? 2.5 : 0);
  Plotly.react(hostId, [
    {type:"bar", x: CONTRASTS, y: stoichNes,
     marker:{color: colors, line:{color: outlines, width: barLineWidth}},
     name:"stoichiometry NES",
     hovertemplate:"%{x}<br>stoich NES %{y:.2f}<extra></extra>"},
    {type:"scatter", mode:"markers", x: CONTRASTS, y: rawNes,
     marker:{color:"#000", size:9, symbol:"diamond-open", line:{width:1.5, color:"#000"}},
     name:"raw phospho NES",
     hovertemplate:"%{x}<br>raw NES %{y:.2f}<extra></extra>"},
  ], {
    margin:{l:40, r:10, t:10, b:60}, height:220,
    yaxis:{zeroline:true, zerolinecolor:"#bbb", title:"NES"},
    xaxis:{tickangle:-35},
    showlegend:false,
  }, {displaylogo:false, responsive:true}).then(() => {
    if (host.on && !host.__meaTrajWired) {
      host.__meaTrajWired = true;
      host.on("plotly_click", (ev) => {
        const pts = ev && ev.points ? ev.points : null;
        if (!pts || !pts[0]) return;
        const target = pts[0].x;
        const select = document.getElementById("audit-contrast-select");
        if (select && Array.from(select.options).some(o => o.value === target)) {
          select.value = target;
          select.dispatchEvent(new Event("change"));
        }
      });
    }
  });
}

function _renderDecompPanel(hostId, kinase_id, ctx, leadRow) {
  const host = document.getElementById(hostId);
  if (!host) return;
  _ensureKinaseIndexes();
  const cid = CONTRASTS.indexOf(ctx.contrast);
  if (cid < 0) {
    host.innerHTML = `<div class="muted">No decomposition data for this contrast.</div>`;
    return;
  }
  const rows = (_decompByKinCtx && _decompByKinCtx.get(`${kinase_id}|${cid}`)) || [];
  if (!rows.length) {
    host.innerHTML = `<div class="muted">No decomposition rows for this kinase &times; contrast.</div>`;
    return;
  }
  const sorted = rows.slice().sort((a, b) => (a.nes ?? 0) - (b.nes ?? 0));
  const fdrThresh = (Store.state.filters && Store.state.filters.fdr) || 0.25;
  const bulkNes = leadRow && Number.isFinite(Number(leadRow.NES)) ? Number(leadRow.NES) : null;
  const bulkFdr = leadRow && Number.isFinite(Number(leadRow.FDR)) ? Number(leadRow.FDR) : null;
  const bulkSig = bulkFdr != null && bulkFdr < fdrThresh;
  const cellTypes = sorted.map(r => r.cell_type);
  const nes = sorted.map(r => (r.nes == null || !isFinite(r.nes)) ? 0 : r.nes);
  const fdrs = sorted.map(r => r.fdr);
  const sigMask = fdrs.map(v => v != null && isFinite(v) && v < fdrThresh);
  const colors = nes.map((v, i) => {
    const base = v >= 0 ? "#c8261c" : "#1f5fa6";
    if (sigMask[i]) return base;
    return v >= 0 ? "rgba(200,38,28,0.22)" : "rgba(31,95,166,0.22)";
  });
  const outlines = sigMask.map(s => s ? "#000" : "rgba(0,0,0,0)");
  const lineWidths = sigMask.map(s => s ? 1.2 : 0);
  const hovers = sorted.map((r, i) =>
    `${r.cell_type}<br>decomp NES ${nes[i].toFixed(2)}` +
    (r.fdr != null && isFinite(r.fdr) ? `<br>FDR ${Number(r.fdr).toExponential(2)}${sigMask[i] ? " (sig)" : ""}` : "")
  );
  const traces = [{
    type: "bar", orientation: "h",
    x: nes, y: cellTypes,
    marker: {color: colors, line: {color: outlines, width: lineWidths}},
    hovertemplate: "%{customdata}<extra></extra>",
    customdata: hovers,
    name: "decomp NES",
  }];
  const shapes = [];
  const annotations = [];
  if (bulkNes != null) {
    shapes.push({
      type: "line", xref: "x", yref: "paper",
      x0: bulkNes, x1: bulkNes, y0: 0, y1: 1,
      line: {color: bulkSig ? "#000" : "#888", width: 2, dash: bulkSig ? "solid" : "dash"},
    });
    annotations.push({
      xref: "x", yref: "paper", x: bulkNes, y: 1.04,
      text: `bulk NES ${bulkNes.toFixed(2)}${bulkSig ? "" : " (ns)"}`,
      showarrow: false, font: {size: 11, color: "#000"},
      xanchor: bulkNes >= 0 ? "left" : "right",
    });
  }
  const height = Math.max(220, 22 * cellTypes.length + 60);
  Plotly.react(hostId, traces, {
    margin: {l: 180, r: 30, t: 30, b: 40},
    height,
    xaxis: {title: "NES", zeroline: true, zerolinecolor: "#bbb"},
    yaxis: {automargin: true, tickfont: {size: 11}},
    shapes, annotations,
    showlegend: false,
  }, {displaylogo: false, responsive: true});
}

function _buildPreparedMeaInput(ctx) {
  const shift = _shiftFor(ctx.globalRows, ctx.contrast);
  const bounds = _winsorBoundsFor(ctx.winsorRows, ctx.contrast);
  const winsorSitesForContrast = new Set(
    (ctx.winsorRows || []).filter(r => r.contrast === ctx.contrast)
      .map(r => String(r.site_id)));
  const leadingEdgeSiteIds = ctx.siteIds || new Set();
  const lfcCol = "stoich_lfc_" + ctx.contrast;
  const prerank = _ensurePreRank(ctx.contrast, ctx.olsRows, ctx.stoichMatrix, shift, bounds);
  const sourceRows = (ctx.substrateSiteRows && ctx.substrateSiteRows.length)
    ? ctx.substrateSiteRows : (ctx.siteRows || []);
  const sourceMode = (ctx.substrateSiteRows && ctx.substrateSiteRows.length) ? "substrate" : "leading_edge_fallback";
  const rows = [];
  for (const sr of sourceRows) {
    const sid = String(sr.site_id);
    const v = sr[lfcCol];
    const raw = (v == null || v === "") ? null : Number(v);
    const rawNum = Number.isFinite(raw) ? raw : null;
    const centered = (rawNum == null || shift == null) ? null : rawNum - shift;
    let clipped = centered;
    if (centered != null && bounds) clipped = Math.min(Math.max(centered, bounds[0]), bounds[1]);
    const wasWin = winsorSitesForContrast.has(sid) ||
      (centered != null && clipped != null && Math.abs(centered - clipped) > 1e-12);
    rows.push({
      site_id: sr.site_id,
      gene_symbol: sr.gene_symbol || "",
      motif: sr.motif || "",
      n_obs_stoich: sr.n_obs_stoich,
      raw_lfc: rawNum,
      centered_lfc: centered,
      clipped_lfc: clipped,
      was_winsorized: wasWin ? "yes" : "no",
      rank_in_contrast: prerank ? (prerank.rankMap.get(sid) ?? null) : null,
      in_leading_edge: leadingEdgeSiteIds.has(sid) ? "yes" : "no",
    });
  }
  rows.sort((a, b) => {
    const ar = a.rank_in_contrast == null ? Infinity : a.rank_in_contrast;
    const br = b.rank_in_contrast == null ? Infinity : b.rank_in_contrast;
    return ar - br;
  });
  return {rows, shift, bounds, prerank, sourceMode};
}

function _renderKinaseNesPlot(hostId, kinase_id) {
  const host = document.getElementById(hostId);
  if (!host) return;
  _ensureKinaseIndexes();
  const K = PAYLOAD.kinases;
  const i = _kinaseIdxById.get(kinase_id);
  if (i == null) return;
  const fdr = Store.state.filters.fdr;
  const nes = CONTRASTS.map(c => K["NES_" + c][i]);
  const fdrs = CONTRASTS.map(c => K["FDR_" + c][i]);
  const colors = CONTRASTS.map(_diseaseColorFor);
  const outlines = fdrs.map(v => (v != null && v < fdr) ? "#000" : "rgba(0,0,0,0)");
  Plotly.react(hostId, [{
    type: "bar", x: CONTRASTS, y: nes,
    marker: { color: colors, line: { color: outlines, width: 1.5 } },
    hovertemplate: "%{x}<br>NES %{y:.2f}<extra></extra>",
  }], {
    margin:{l:40,r:10,t:6,b:60}, height:180,
    yaxis:{zeroline:true, zerolinecolor:"#bbb"},
    xaxis:{tickangle:-35},
  }, {displaylogo:false, responsive:true});
}

function _renderKinaseCelltypeEvidence(hostId, kinase_id) {
  const host = document.getElementById(hostId);
  if (!host) return;
  const EV = PAYLOAD.kinase_celltype_evidence || {kinase_id:[]};
  const evIdx = _evidenceByKinase.get(kinase_id) || [];
  const rows = evIdx.map(k => ({
    cell_type: EV.cell_type[k],
    wmb_fold: EV.wmb_fold[k],
    sea_ad_lfc: EV.sea_ad_lfc[k],
    song_lfc: EV.song_lfc[k],
    wmb_tier: EV.wmb_tier[k],
    evidence_basis: EV.evidence_basis ? EV.evidence_basis[k] : "",
    concordance_direction: EV.concordance_direction ? EV.concordance_direction[k] : "",
  }));
  rows.sort((a, b) => {
    const av = a.wmb_fold == null ? -Infinity : a.wmb_fold;
    const bv = b.wmb_fold == null ? -Infinity : b.wmb_fold;
    return bv - av;
  });
  _renderAuditTable(hostId, "celltype_evidence_table", rows,
    ["cell_type","wmb_fold","sea_ad_lfc","song_lfc","wmb_tier","evidence_basis","concordance_direction"],
    "celltype_evidence_table");
}

// ---- Attribution drawer helpers ----------------------------------------
// The Attribution subtab uses three audit sources read directly so the
// reviewer sees the underlying evidence in idiomatic single-cell-biology
// shapes (Seurat dot plot for WMB; supertype LFC heatmap for SEA-AD; OLS
// coefficient table for Song) rather than a synthesized score.

function _attrPathwayFromContrast(contrast) {
  return String(contrast || "").split("_")[0] || "";
}

function _attrLfcColor(lfc) {
  if (lfc == null || !isFinite(lfc) || lfc === 0) return "#f3f4f6";
  const m = Math.min(Math.abs(lfc), 1.0);
  const alpha = 0.08 + 0.32 * m;
  return lfc > 0
    ? `rgba(197, 48, 48, ${alpha.toFixed(3)})`
    : `rgba(43, 108, 176, ${alpha.toFixed(3)})`;
}

function _attrConfidenceClass(conf) {
  if (conf === "very_high") return "attr-conf attr-conf-very-high";
  if (conf === "high") return "attr-conf attr-conf-high";
  if (conf === "moderate") return "attr-conf attr-conf-moderate";
  if (conf === "low") return "attr-conf attr-conf-low";
  return "attr-conf attr-conf-none";
}

function _allenABALink(gene) {
  if (!gene) return "";
  const abc = "https://knowledge.brain-map.org/abcatlas";
  const ctxHpf = `https://celltypes.brain-map.org/rnaseq/mouse_ctx-hpf_10x?selectedVisualization=Scatter+Plot&colorByFeature=Gene+Expression&colorByFeatureValue=${encodeURIComponent(gene)}`;
  return (
    `<a href="${abc}" target="_blank" rel="noopener" class="attr-allen-link" ` +
    `title="ABC Atlas (whole brain) — same Allen WMB 10Xv3 dataset our specificity score is computed on. Search '${_escapeHtml(gene)}' to verify against the same data we used.">` +
    `Verify in ABC Atlas (whole brain) →</a>` +
    ` <a href="${ctxHpf}" target="_blank" rel="noopener" class="attr-allen-link attr-allen-link-secondary" ` +
    `title="Allen Cortex+HPF Transcriptomics Explorer — different dataset (cortex + hippocampal formation only, ~1.1M cells). Useful for high-resolution per-cell intensity in cortical/HPF cell types, but does not contain striatum, olfactory bulb, thalamus, or cerebellum.">` +
    `ctx+HPF (partial tissue)</a>`
  );
}

const ATTR_VERDICT_COLS = [
  {key:"cell_type",                    label:"Cell type",   type:"str", group:"id",
   title:""},
  {key:"cross_rank",                   label:"Conf",        type:"num", group:"attr",
   title:"Combined confidence tier. Starts from the attribution-only tier (high / moderate / low / none). Upgraded to 'very high' when the decomposition layer significantly agrees (Decomp FDR < 0.25 with sign matching bulk MEA). Sort uses cross_rank: tier first, decomposition step as tie-breaker."},
  {key:"wmb_specificity",              label:"WMB enrich",  type:"num", group:"attr",
   title:"WMB enrichment: cell type's share of total log2 expression across 34 WMB classes (uniform = 1/34 ≈ 0.029). Higher = more concentrated in this cell type."},
  {key:"wmb_tier",                     label:"WMB tier",    type:"num", group:"attr",
   title:"WMB specificity expressed as a multiple of uniform (1/34 ≈ 0.029): ≥10× / ≥5× / ≥2× / ≥1×. Empty = below 1× uniform."},
  {key:"wmb_mean_log2_expression",     label:"log2 expr",   type:"num", group:"attr",
   title:"WMB mean log2 expression in this cell type (Allen Whole Mouse Brain 10Xv3, pooled across 13 regions). Absolute level — low values flag the score as potentially noise-driven."},
  {key:"wmb_fraction_cells_expressing",label:"% cells",     type:"num", group:"attr",
   title:"WMB fraction of cells of this cell type with non-zero counts for this gene."},
  {key:"sea_ad_lfc",                   label:"SEA-AD LFC",  type:"num", group:"attr",
   title:"SEA-AD log2 fold change in human AD vs control, median across SEA-AD supertypes mapped to this subclass. Stratum (early / late / full CPS) is selected from the contrast pathway. Color: red = up in AD, blue = down."},
  {key:"song_lfc",                     label:"Song LFC",    type:"num", group:"attr",
   title:"Song log2 fold change from within-cohort snRNA-seq factorial OLS (β at this contrast — 10-param design, time-resolved). Color: red = up in disease genotype, blue = down."},
  {key:"combined_score",               label:"Score",       type:"num", group:"attr",
   title:"Combined attribution score: effective concordance × (0.5 + WMB specificity). The unified attribution uses this for confidence tiers."},
  {key:"decomp_nes",                   label:"Decomp NES",  type:"num", group:"decomp",
   title:"Decomposition NES from the CTM-native proportional decomposition (per-cell-type kinase MEA on bulk phospho ranking weighted by snRNA share for the kinase's substrate set). Same join key as Song LFC. Hypothesis-strength signal — see Methods."},
  {key:"decomp_fdr",                   label:"Decomp FDR",  type:"num", group:"decomp",
   title:"Decomposition MEA FDR for this (kinase, contrast, cell type) row. < 0.25 is the standard MEA gate."},
  {key:"bulk_match",                   label:"vs Bulk",     type:"num", group:"decomp",
   title:"Sign agreement between Decomp NES and the bulk MEA NES for this kinase × contrast. Bold ✓/✗ when Decomp FDR < 0.25; muted when not. Hover any cell for the underlying values."},
];
function _attrVerdictCmp(a, b, key, type, asc) {
  let va, vb;
  if (type === "num") {
    va = a[key]; vb = b[key];
    va = (va == null || !isFinite(va)) ? null : Number(va);
    vb = (vb == null || !isFinite(vb)) ? null : Number(vb);
  } else if (type === "conf") {
    va = _CONF_RANK[a[key]] ?? -1;
    vb = _CONF_RANK[b[key]] ?? -1;
  } else {
    va = (a[key] || "").toString();
    vb = (b[key] || "").toString();
  }
  if (va == null && vb == null) return 0;
  if (va == null) return 1;
  if (vb == null) return -1;
  if (typeof va === "string") return asc ? va.localeCompare(vb) : vb.localeCompare(va);
  return asc ? (va - vb) : (vb - va);
}

function _renderAttributionVerdict(hostId, ctx) {
  const host = document.getElementById(hostId);
  if (!host) return;

  // Verdict reads attribution_index for the audit picker's contrast. Independent
  // of the left-list KinaseFilter — the detail panel's job is to inspect this kinase
  // at the contrast the user picked.
  const verdictFilter = {
    disease:   (ctx.contrast || "").split("_")[0] || "",
    timepoint: ((ctx.contrast || "").match(/_(\d+mo)$/) || ["",""])[1] || "",
    celltype: "", confidence: "",
  };
  const allRows = ctx.kinase_id != null
    ? getScopedAttribution(ctx.kinase_id, verdictFilter)
    : [];

  if (allRows.length === 0) {
    host.innerHTML = `<div class="muted">No attribution rows in ${_escapeHtml(ctx.contrast || "")}.</div>`;
    return;
  }

  // Deduplicate by (contrast_id, cell_type), keeping best-score row.
  const rowKey = r => `${r.contrast_id}|${r.cell_type}`;
  const deduped = new Map();
  for (const r of allRows) {
    const k = rowKey(r);
    const prev = deduped.get(k);
    if (!prev || r.combined_score > prev.combined_score) deduped.set(k, r);
  }
  const rows = Array.from(deduped.values());

  // Attach decomposition NES/FDR for sorting + render. Also compute bulk-NES
  // sign agreement once (same bulk NES/FDR for every row in this kinase × contrast).
  const _K = PAYLOAD.kinases;
  const _bulkNes = (_K && _K["NES_" + ctx.contrast]) ? _K["NES_" + ctx.contrast][ctx.kinase_id] : null;
  const _bulkFdr = (_K && _K["FDR_" + ctx.contrast]) ? _K["FDR_" + ctx.contrast][ctx.kinase_id] : null;
  for (const r of rows) {
    const dk = `${ctx.kinase_id}|${r.contrast_id}|${r.cell_type}`;
    const d = _decompByKey ? _decompByKey.get(dk) : null;
    r.decomp_nes = d ? d.nes : null;
    r.decomp_fdr = d ? d.fdr : null;
    // bulk_match: +2 sig-agree, +1 nonsig-agree, -1 nonsig-disagree, -2 sig-disagree,
    // null when either side is missing. "Sig" here = Decomp FDR < 0.25.
    if (r.decomp_nes == null || !isFinite(r.decomp_nes) || r.decomp_nes === 0
        || _bulkNes == null || !isFinite(_bulkNes) || _bulkNes === 0) {
      r.bulk_match = null;
    } else {
      const agree = (r.decomp_nes > 0) === (_bulkNes > 0);
      const sig = r.decomp_fdr != null && isFinite(r.decomp_fdr) && r.decomp_fdr < 0.25;
      r.bulk_match = agree ? (sig ? 2 : 1) : (sig ? -2 : -1);
    }
    const decompStep = _decompStep(r.decomp_nes, r.decomp_fdr, _bulkNes);
    r.decomp_step = decompStep;
    r.combined_tier = _upgradeTier(r.combined_confidence, decompStep);
    r.wmb_tier = _wmbTier(Number(r.wmb_specificity));
    // cross_rank: combine combined_tier (0..4) and decomp step (-2..3) so
    // reinforcing rows sort first, conflicts demoted, single-layer in between.
    r.cross_rank = (_CONF_RANK[r.combined_tier] || 0) * 6 + decompStep;
  }

  const sortKey = host.dataset.sortKey || "combined_score";
  const sortAsc = host.dataset.sortAsc === "1";
  const sortCol = ATTR_VERDICT_COLS.find(c => c.key === sortKey)
    || ATTR_VERDICT_COLS.find(c => c.key === "combined_score")
    || ATTR_VERDICT_COLS[ATTR_VERDICT_COLS.length - 1];
  rows.sort((a, b) => _attrVerdictCmp(a, b, sortCol.key, sortCol.type, sortAsc));
  const showAllId = `${hostId}-show-all`;
  const showAll = !!(host.dataset.showAll === "1");
  const visibleRows = showAll
    ? rows
    : rows.filter(r => r.combined_tier === "very_high"
                    || r.combined_tier === "high"
                    || r.combined_tier === "moderate");
  const hiddenCount = rows.length - visibleRows.length;
  const num = (v, d=3) => (v == null || !isFinite(v)) ? "" : Number(v).toFixed(d);
  const tbody = visibleRows.map((r, i) => {
    const seaCell = r.sea_ad_lfc == null || !isFinite(r.sea_ad_lfc)
      ? `<td class="attr-num attr-empty">—</td>`
      : `<td class="attr-num attr-num-lfc" style="background:${_attrLfcColor(r.sea_ad_lfc)}">${num(r.sea_ad_lfc, 3)}</td>`;
    const songCell = r.song_lfc == null || !isFinite(r.song_lfc)
      ? `<td class="attr-num attr-empty">—</td>`
      : `<td class="attr-num attr-num-lfc" style="background:${_attrLfcColor(r.song_lfc)}">${num(r.song_lfc, 3)}</td>`;
    const decompNesCell = r.decomp_nes == null || !isFinite(r.decomp_nes)
      ? `<td class="attr-num attr-empty">—</td>`
      : `<td class="attr-num attr-num-lfc" style="background:${_attrLfcColor(r.decomp_nes)}">${num(r.decomp_nes, 2)}</td>`;
    const decompFdrSig = r.decomp_fdr != null && isFinite(r.decomp_fdr) && r.decomp_fdr < 0.25;
    const decompFdrCell = r.decomp_fdr == null || !isFinite(r.decomp_fdr)
      ? `<td class="attr-num attr-empty">—</td>`
      : `<td class="attr-num"${decompFdrSig ? ' style="font-weight:600"' : ''}>${num(r.decomp_fdr, 3)}</td>`;
    let bulkMatchCell;
    if (r.bulk_match == null) {
      bulkMatchCell = `<td class="attr-num attr-empty">—</td>`;
    } else {
      const agree = r.bulk_match > 0;
      const sig = Math.abs(r.bulk_match) === 2;
      const glyph = agree ? "✓" : "✗";
      const color = agree ? "#15803d" : "#b91c1c";
      const style = sig
        ? `color:${color};font-weight:700`
        : `color:#94a3b8;font-weight:500`;
      const tip = `Bulk NES = ${num(_bulkNes, 2)}` +
        (_bulkFdr != null && isFinite(_bulkFdr) ? ` (FDR ${num(_bulkFdr, 3)})` : "") +
        ` · Decomp NES = ${num(r.decomp_nes, 2)}` +
        (r.decomp_fdr != null && isFinite(r.decomp_fdr) ? ` (FDR ${num(r.decomp_fdr, 3)})` : "") +
        (sig ? "" : " · Decomp not significant (FDR ≥ 0.25)");
      bulkMatchCell = `<td class="attr-num" style="${style};text-align:center" title="${_escapeHtml(tip)}">${glyph}</td>`;
    }
    const binFlag = r.wmb_binary_expressed === true || String(r.wmb_binary_expressed).toLowerCase() === "true";
    const expBadge = binFlag
      ? ""
      : `<span class="attr-badge attr-badge-warn" title="Mean log2 expression < 1 OR fewer than 10% of cells detect the gene in this cell type. The enrichment score may be elevated because the gene is barely expressed anywhere.">low expr</span>`;
    const _sbk = (PAYLOAD.subclass_breakdown || {})[String(ctx.kinase_id)] || {};
    const _sbTip = _sbk[r.cell_type] || "";
    const _sbAttr = _sbTip ? ` title="WMB subclass breakdown: ${_escapeHtml(_sbTip)}"` : "";
    const scoreCell = `<td class="attr-num">${num(r.combined_score, 3)}</td>`;
    return `<tr data-cell-type="${_escapeHtml(r.cell_type)}" class="attr-verdict-row${i === 0 ? ' attr-verdict-selected' : ''}">` +
      `<td class="attr-celltype"${_sbAttr}>${_escapeHtml(r.cell_type)}${_sbTip ? ' <span class="attr-subclass-marker" aria-hidden="true">ⓘ</span>' : ''} ${expBadge}</td>` +
      `<td><span class="${_attrConfidenceClass(r.combined_tier)}" title="${_escapeHtml('Attribution: ' + (r.combined_confidence || 'none') + (r.combined_tier === 'very_high' ? ' · upgraded to very_high by significant decomp agreement' : ''))}">${_escapeHtml((r.combined_tier || '').replace('_', ' '))}</span></td>` +
      `<td class="attr-num">${num(r.wmb_specificity, 3)}</td>` +
      `<td class="attr-num">${_wmbTierBadge(_wmbTier(Number(r.wmb_specificity)))}</td>` +
      `<td class="attr-num">${num(r.wmb_mean_log2_expression, 2)}</td>` +
      `<td class="attr-num">${num(r.wmb_fraction_cells_expressing, 2)}</td>` +
      seaCell +
      songCell +
      scoreCell +
      decompNesCell +
      decompFdrCell +
      bulkMatchCell +
      `</tr>`;
  }).join("");
  const headCells = ATTR_VERDICT_COLS.map(c => {
    const arrow = (c.key === sortCol.key) ? (sortAsc ? " ▲" : " ▼") : "";
    const title = c.title ? ` title="${_escapeHtml(c.title)}"` : "";
    return `<th class="attr-verdict-th" data-sort-key="${c.key}"${title}>${c.label}${arrow}</th>`;
  }).join("");
  // Super-header groups the columns into Layer-1 (attribution) and Layer-2 (decomp).
  const _grpCounts = ATTR_VERDICT_COLS.reduce((acc, c) => { acc[c.group] = (acc[c.group]||0)+1; return acc; }, {});
  const superHead =
    `<tr class="attr-verdict-supergroup">` +
      `<th class="attr-supergroup-spacer" colspan="${_grpCounts.id || 0}"></th>` +
      `<th class="attr-supergroup-attr" colspan="${_grpCounts.attr || 0}" title="Cell-type attribution evidence. Each component is compared against the bulk MEA direction at this contrast.">Attribution (vs bulk direction)</th>` +
      `<th class="attr-supergroup-decomp" colspan="${_grpCounts.decomp || 0}" title="Per-cell-type pseudo-deconvolution MEA. A second look at the bulk phospho ranking re-projected by snRNA share.">Decomposition cross-check</th>` +
    `</tr>`;
  // Bulk anchor — both layers compare against this kinase's bulk MEA at this contrast.
  const _bulkSig = _bulkFdr != null && isFinite(_bulkFdr) && _bulkFdr < 0.25;
  const _bulkDir = (_bulkNes != null && isFinite(_bulkNes))
    ? (_bulkNes > 0 ? `<span class="attr-bulk-up">↑ NES = +${num(_bulkNes, 2)}</span>`
                    : `<span class="attr-bulk-down">↓ NES = ${num(_bulkNes, 2)}</span>`)
    : `<span class="attr-bulk-ns">NES n/a</span>`;
  const _bulkFdrTxt = (_bulkFdr != null && isFinite(_bulkFdr))
    ? `FDR = ${num(_bulkFdr, 3)}${_bulkSig ? "" : " (n.s.)"}` : "FDR n/a";
  const bulkAnchor =
    `<div class="attr-bulk-anchor">Bulk MEA anchor for ${_escapeHtml(ctx.contrast || "")}: ` +
    `<span class="attr-bulk-pill">${_bulkDir} · ${_bulkFdrTxt}</span> ` +
    `<span class="muted">— sign of the bulk NES is the reference direction every column below is checked against.</span></div>`;
  host.innerHTML =
    bulkAnchor +
    `<table class="attr-verdict-table">` +
      `<thead>${superHead}<tr>${headCells}</tr></thead><tbody>${tbody}</tbody>` +
    `</table>` +
    (hiddenCount > 0
      ? `<div class="attr-verdict-toggle"><label><input type="checkbox" id="${showAllId}"${showAll ? " checked" : ""}> Show all 34 WMB classes <span class="muted">(${hiddenCount} hidden — low/none confidence)</span></label></div>`
      : (showAll && rows.length > 0
        ? `<div class="attr-verdict-toggle"><label><input type="checkbox" id="${showAllId}" checked> Showing all cell types</label></div>`
        : "")) +
    `<details class="attr-explainer"><summary>How to read <em>Score</em> vs. <em>Confidence</em> in this table</summary>` +
      `<div class="attr-explainer-body">` +
      `<p>Score and tier come from the same three evidence sources but answer different questions:</p>` +
      `<table class="attr-explainer-table" style="margin-bottom:8px;">` +
        `<thead><tr><th>Source</th><th>What it tells you</th></tr></thead><tbody>` +
        `<tr><td><strong>Song</strong></td><td>Does this gene go up or down in our own mice? (within-cohort snRNA-seq from the same animals)</td></tr>` +
        `<tr><td><strong>SEA-AD</strong></td><td>Does this gene go up or down in human Alzheimer's brains? (human postmortem reference)</td></tr>` +
        `<tr><td><strong>WMB</strong></td><td>Is this gene normally on in this cell type, in a healthy mouse? (used as a sanity check, not a direction)</td></tr>` +
        `<tr><td><strong>Decomp NES / FDR</strong></td><td>A per-cell-type version of the bulk phospho signal, reweighted toward each cell type using the snRNA data. Uses the same snRNA data as Song, so treat it as a second look, not independent evidence.</td></tr>` +
        `</tbody></table>` +
      `<p><strong>Confidence tier</strong> grades <em>which sources agree</em>, not how strong any one signal is:</p>` +
      `<ul>` +
        `<li><strong><span class="attr-conf attr-conf-very-high">very high</span></strong> — a <em>high</em> attribution row that is also corroborated by the decomposition layer: Decomp FDR < 0.25 with the same sign as the bulk MEA NES. Both evidence streams reinforce one another.</li>` +
        `<li><strong><span class="badge hi">high</span></strong> — all three of these hold: <em>(a)</em> within-cohort Song supports the direction, <em>(b)</em> the gene is clearly cell-type-specific in WMB (specificity ≥ 2× uniform, i.e. ≈ 0.059 for 34 WMB classes), and <em>(c)</em> at least one reference shows real movement (|Song LFC| or |SEA-AD LFC| > 0.1).</li>` +
        `<li><strong><span class="badge mid">moderate</span></strong> — meaningful evidence but missing one strict gate. Two ways to land here: Song-supported but WMB specificity falls below the high threshold, <em>or</em> only SEA-AD reached concordance (no Song). SEA-AD-only is <strong>always</strong> capped at moderate — we won't promote a cross-species call to high.</li>` +
        `<li><strong><span class="badge lo">low</span></strong> — concordance is positive but the gene isn't expression-specific in WMB and no reference LFC clears the magnitude bar.</li>` +
        `<li><strong>none</strong> — concordance ≤ 0 (signs disagree). Row is excluded from <code>unified_attribution.csv</code> entirely.</li>` +
      `</ul>` +
      `<p><strong>Higher score does not imply higher tier.</strong> A row with strong magnitudes but no within-cohort Song evidence stays at moderate regardless of score; a row with a modest score but Song support + WMB specificity ≥ 0.059 reaches high. Read tier as evidence <em>type</em>, score as evidence <em>weight</em>.</p>` +
      `<p><strong>Combined score</strong> = <code>effective_concordance × (0.5 + wmb_specificity)</code> where <code>effective_concordance = sign(NES) × (3·song_lfc + 1·sea_ad_lfc) / 4</code>. Continuous; used to rank cell types within a kinase (tie-break within tier) and to weight kinase support in the Incytr cell–cell integration.</p>` +
      `</div></details>`;
  host.querySelectorAll("tr.attr-verdict-row").forEach(tr => tr.addEventListener("click", () => {
    host.querySelectorAll("tr.attr-verdict-row").forEach(r => r.classList.remove("attr-verdict-selected"));
    tr.classList.add("attr-verdict-selected");
    _renderAttributionDrawer("attr-drawer", ctx, tr.dataset.cellType);
  }));
  host.querySelectorAll("th.attr-verdict-th").forEach(th => th.addEventListener("click", () => {
    const k = th.dataset.sortKey;
    if (host.dataset.sortKey === k) {
      host.dataset.sortAsc = host.dataset.sortAsc === "1" ? "0" : "1";
    } else {
      host.dataset.sortKey = k;
      // Numeric/conf cols default to descending (largest first); strings ascending.
      const col = ATTR_VERDICT_COLS.find(c => c.key === k);
      host.dataset.sortAsc = (col && col.type === "str") ? "1" : "0";
    }
    _renderAttributionVerdict(hostId, ctx);
  }));
  const toggleEl = document.getElementById(showAllId);
  if (toggleEl) {
    toggleEl.addEventListener("change", () => {
      host.dataset.showAll = toggleEl.checked ? "1" : "0";
      _renderAttributionVerdict(hostId, ctx);
    });
  }
  // Open drawer on the top row by default
  if (rows[0]) _renderAttributionDrawer("attr-drawer", ctx, rows[0].cell_type);
}

function _renderAttributionDrawer(hostId, ctx, cellType) {
  const host = document.getElementById(hostId);
  if (!host) return;
  const gene = ctx.gene || "";
  host.innerHTML =
    `<div class="attr-drawer-header"><strong>${_escapeHtml(cellType)}</strong>` +
    ` &middot; <span class="muted">${_escapeHtml(gene)} / ${_escapeHtml(ctx.contrast)}</span>` +
    ` &middot; ${_allenABALink(gene)}</div>` +
    `<div class="attr-drawer-grid">` +
      `<section class="attr-section"><h5>WMB expression across 34 WMB classes <span class="muted">(wmb_kinase_expression.csv)</span></h5>` +
        `<p class="muted attr-caption">Seurat-style dot plot for ${_escapeHtml(gene)} in the Allen Whole Mouse Brain reference. Color = mean log2 expression, dot size = fraction of cells expressing. Target cell type is outlined.</p>` +
        `<div id="attr-wmb-dotplot"></div></section>` +
      `<section class="attr-section"><h5>SEA-AD supertype log2 fold change <span class="muted">(sea_ad_supertype_lfc.csv)</span></h5>` +
        `<p class="muted attr-caption">Per-supertype LFC for ${_escapeHtml(gene)} in human AD donors, grouped by subclass. Stratum (early / late / full CPS) follows the contrast pathway. Subclass median is used in the verdict table.</p>` +
        `<div id="attr-seaad-heatmap"></div></section>` +
      `<section class="attr-section"><h5>Song within-cohort OLS <span class="muted">(song_concordance.csv)</span></h5>` +
        `<p class="muted attr-caption">Factorial OLS coefficient on the per-animal pseudobulk for this cell type and pathway. Pathway is derived from the contrast prefix (App / Tau / ApTt).</p>` +
        `<div id="attr-song-table"></div></section>` +
    `</div>` +
    `<section class="attr-section attr-section-wide"><h5>Per-cell substrate-site OLS <span class="muted">(deconvolution/per_animal/site_level_ols.parquet)</span></h5>` +
      `<p class="muted attr-caption">Per-(site, contrast, cell type) β / SE / p from the CTM-native pseudo-deconvolution OLS, restricted to ${_escapeHtml(ctx.name || "")}'s substrate set in ${_escapeHtml(cellType)}. Shows what is driving the Decomp NES in the row above. Bulk β is the same site's stoichiometry β before share-reweighting; |Δβ| measures how much the per-cell estimate diverges from bulk.</p>` +
      `<div id="attr-decomp-ols-table" class="audit-scroll"></div></section>`;
  _renderWMBDotPlot("attr-wmb-dotplot", ctx, cellType);
  _renderSEAADHeatmap("attr-seaad-heatmap", ctx, cellType);
  _renderSongOLSPanel("attr-song-table", ctx, cellType);
  _renderDecompOlsTable("attr-decomp-ols-table", ctx, cellType);
}

function _renderDecompOlsTable(hostId, ctx, cellType) {
  const host = document.getElementById(hostId);
  if (!host) return;
  const cId = CONTRASTS.indexOf(ctx.contrast);
  if (ctx.kinase_id == null || cId < 0) {
    host.innerHTML = `<div class="muted">No contrast resolved.</div>`;
    return;
  }
  if (!SliceCache || typeof SliceCache.loadDecompOls !== "function") {
    host.innerHTML = `<div class="muted">Decomp OLS shards unavailable in this build.</div>`;
    return;
  }
  host.innerHTML = `<div class="muted">Loading per-cell OLS shard…</div>`;
  const reqGene = ctx.gene;
  const reqContrast = ctx.contrast;
  const reqCell = cellType;
  SliceCache.loadDecompOls(ctx.kinase_id).then(rows => {
    // Bail if the user moved on while we were fetching.
    if (ctx.gene !== reqGene || ctx.contrast !== reqContrast) return;
    const stillThis = document.getElementById(hostId);
    if (!stillThis || stillThis !== host) return;
    if (!Array.isArray(rows) || rows.length === 0) {
      host.innerHTML = `<div class="muted">No per-cell OLS shard for this kinase.</div>`;
      return;
    }
    const sub = rows.filter(r => Number(r.contrast_id) === cId
                              && String(r.wmb_class) === String(reqCell));
    if (!sub.length) {
      host.innerHTML = `<div class="muted">No substrate sites for ${_escapeHtml(reqCell)} in ${_escapeHtml(reqContrast)}.</div>`;
      return;
    }
    const lfcCol = "stoich_lfc_" + reqContrast;
    const pCol = "stoich_pval_" + reqContrast;
    const bulkBySite = new Map();
    for (const r of (ctx.olsRows || [])) {
      bulkBySite.set(String(r.site_id), {bulk_lfc: r[lfcCol], bulk_pval: r[pCol]});
    }
    sub.sort((a, b) => (Number(b.lfc) || 0) - (Number(a.lfc) || 0));
    const num = (v, d=3) => (v == null || !isFinite(v)) ? "—" : Number(v).toFixed(d);
    const rowsHtml = sub.map(r => {
      const sid = String(r.site_id);
      const bulk = bulkBySite.get(sid) || {};
      const blfc = bulk.bulk_lfc != null && isFinite(bulk.bulk_lfc) ? Number(bulk.bulk_lfc) : null;
      const dlfc = (blfc != null && isFinite(r.lfc)) ? Math.abs(Number(r.lfc) - blfc) : null;
      const pcSig = isFinite(r.pval) && Number(r.pval) < 0.05;
      const bulkSig = bulk.bulk_pval != null && isFinite(bulk.bulk_pval) && Number(bulk.bulk_pval) < 0.05;
      return `<tr>` +
        `<td>${_escapeHtml(r.gene_symbol || "")}</td>` +
        `<td class="attr-num">${_escapeHtml(sid)}</td>` +
        `<td class="motif-mono">${_escapeHtml(r.motif || "")}</td>` +
        `<td>${_escapeHtml(r.track || "")}</td>` +
        `<td class="attr-num"${pcSig ? ' style="font-weight:600"' : ''}>${num(r.lfc, 3)}</td>` +
        `<td class="attr-num">${num(r.se, 3)}</td>` +
        `<td class="attr-num"${pcSig ? ' style="font-weight:600"' : ''}>${num(r.pval, 3)}</td>` +
        `<td class="attr-num"${bulkSig ? ' style="font-weight:600"' : ''}>${num(blfc, 3)}</td>` +
        `<td class="attr-num">${num(dlfc, 3)}</td>` +
      `</tr>`;
    }).join("");
    host.innerHTML =
      `<div class="muted" style="font-size:11px;margin-bottom:4px;">${sub.length} substrate sites · sorted by per-cell β (largest first)</div>` +
      `<table class="attr-decomp-ols-table"><thead><tr>` +
        `<th>Gene</th><th>Site</th><th>Motif</th><th>Track</th>` +
        `<th title="Per-cell β: substrate-site stoichiometry coefficient from the per-(group, wmb_class) OLS, on the deconvoluted phospho signal. Bold when per-cell p < 0.05.">Per-cell β</th>` +
        `<th>SE</th>` +
        `<th title="Per-cell p-value (uncorrected). Bold at p < 0.05.">p</th>` +
        `<th title="Bulk β: same site's stoichiometry β from the bulk MEA pipeline before share-reweighting. Bold when bulk p < 0.05.">Bulk β</th>` +
        `<th title="|per-cell β − bulk β|. Large values mean the cell-type estimate diverges materially from the bulk estimate at this site.">|Δβ|</th>` +
      `</tr></thead><tbody>${rowsHtml}</tbody></table>`;
  }).catch(err => {
    console.error("decomp OLS shard fetch failed", err);
    host.innerHTML = `<div class="muted">Failed to load per-cell OLS shard: ${_escapeHtml(String(err && err.message || err))}</div>`;
  });
}

function _renderWMBDotPlot(hostId, ctx, targetCellType) {
  const host = document.getElementById(hostId);
  if (!host) return;
  const rows = (ctx.wmbRows || []).slice();
  if (rows.length === 0) {
    host.innerHTML = `<div class="muted">No WMB rows for ${_escapeHtml(ctx.gene || '')}.</div>`;
    return;
  }
  rows.sort((a, b) => (Number(b.mean_log2_expression) || 0) - (Number(a.mean_log2_expression) || 0));
  const maxExpr = Math.max(...rows.map(r => Number(r.mean_log2_expression) || 0), 1);
  const W = 720, H = 18 * rows.length + 60, padL = 160, padT = 30, padR = 40;
  const innerW = W - padL - padR;
  const x0 = padL, x1 = padL + innerW;
  const colorAt = (v) => {
    const t = Math.max(0, Math.min(1, v / maxExpr));
    // grey → deep blue ramp
    const r = Math.round(240 - 180 * t), g = Math.round(240 - 130 * t), b = Math.round(240 - 50 * t);
    return `rgb(${r},${g},${b})`;
  };
  const sizeAt = (frac) => {
    const f = Math.max(0, Math.min(1, Number(frac) || 0));
    return 2 + 9 * Math.sqrt(f);
  };
  const tickValues = [0, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0].filter(v => v <= maxExpr * 1.05);
  const xScale = (v) => x0 + (Math.max(0, Math.min(maxExpr, v)) / maxExpr) * innerW;
  const ticks = tickValues.map(v => `<line x1="${xScale(v)}" x2="${xScale(v)}" y1="${padT - 4}" y2="${padT}" stroke="#9ca3af" stroke-width="1"/>` +
    `<text x="${xScale(v)}" y="${padT - 8}" font-size="10" text-anchor="middle" fill="#6b7280">${v}</text>`).join("");
  const dots = rows.map((r, i) => {
    const expr = Number(r.mean_log2_expression) || 0;
    const frac = Number(r.fraction_cells_expressing) || 0;
    const cx = xScale(expr);
    const cy = padT + 18 * i + 9;
    const isTarget = r.cell_type === targetCellType;
    const stroke = isTarget ? "#111827" : "#cbd5e0";
    const strokeW = isTarget ? 2 : 0.8;
    const labelClass = isTarget ? "attr-dot-label attr-dot-label-target" : "attr-dot-label";
    const title = `${r.cell_type}: log2 expr = ${expr.toFixed(2)}, fraction = ${frac.toFixed(2)}, specificity = ${(Number(r.specificity_score) || 0).toFixed(3)}`;
    return `<g><title>${_escapeHtml(title)}</title>` +
      `<text x="${x0 - 8}" y="${cy + 3.5}" text-anchor="end" font-size="11" class="${labelClass}">${_escapeHtml(r.cell_type)}</text>` +
      `<line x1="${x0}" x2="${x1}" y1="${cy}" y2="${cy}" stroke="#e5e7eb" stroke-dasharray="2,2"/>` +
      `<circle cx="${cx}" cy="${cy}" r="${sizeAt(frac).toFixed(1)}" fill="${colorAt(expr)}" stroke="${stroke}" stroke-width="${strokeW}"/>` +
      `</g>`;
  }).join("");
  const legend = `<g transform="translate(${padL}, ${H - 22})">` +
    `<text x="0" y="0" font-size="10" fill="#6b7280">Color: log2 expression (0 → ${maxExpr.toFixed(1)})  ·  Size: fraction of cells expressing (0 → 1)</text>` +
    `</g>`;
  host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" preserveAspectRatio="xMidYMid meet" class="attr-svg">` +
    `<line x1="${x0}" x2="${x1}" y1="${padT}" y2="${padT}" stroke="#9ca3af" stroke-width="1"/>` +
    ticks + dots + legend +
    `</svg>`;
}

function _renderSEAADHeatmap(hostId, ctx, targetCellType) {
  const host = document.getElementById(hostId);
  if (!host) return;
  const stratumByPathway = {App: "early", Tau: "late", ApTt: "full"};
  const pathway = _attrPathwayFromContrast(ctx.contrast);
  const stratum = stratumByPathway[pathway] || "full";
  const rows = (ctx.seaSuperRows || []).filter(r => r.stratum === stratum);
  if (rows.length === 0) {
    host.innerHTML = `<div class="muted">No SEA-AD supertype rows for ${_escapeHtml(ctx.gene || '')} (stratum: ${_escapeHtml(stratum)}).</div>`;
    return;
  }
  // Group by subclass
  const bySubclass = new Map();
  for (const r of rows) {
    const sc = r.subclass || "(unknown)";
    if (!bySubclass.has(sc)) bySubclass.set(sc, []);
    bySubclass.get(sc).push(r);
  }
  const subclasses = Array.from(bySubclass.keys()).sort((a, b) => {
    if (a === targetCellType) return -1;
    if (b === targetCellType) return 1;
    return a.localeCompare(b);
  });
  const allLfcs = rows.map(r => Number(r.supertype_lfc) || 0);
  const maxAbs = Math.max(...allLfcs.map(Math.abs), 0.5);
  const cellW = 22, cellH = 16, padL = 170;
  let maxCols = 0;
  for (const arr of bySubclass.values()) maxCols = Math.max(maxCols, arr.length);
  const W = padL + cellW * maxCols + 30;
  const H = subclasses.length * cellH + 50;
  const lfcColor = (v) => {
    const m = Math.min(Math.abs(v) / maxAbs, 1);
    const alpha = 0.15 + 0.75 * m;
    if (v > 0) return `rgba(197, 48, 48, ${alpha.toFixed(3)})`;
    if (v < 0) return `rgba(43, 108, 176, ${alpha.toFixed(3)})`;
    return "#f3f4f6";
  };
  const cells = subclasses.map((sc, i) => {
    const arr = bySubclass.get(sc).slice().sort((a, b) => (Number(b.supertype_lfc) || 0) - (Number(a.supertype_lfc) || 0));
    const isTarget = sc === targetCellType;
    const labelClass = isTarget ? "attr-hm-label-target" : "";
    const median = arr.map(r => Number(r.supertype_lfc) || 0).sort((a, b) => a - b)[Math.floor(arr.length / 2)] || 0;
    const cellsRow = arr.map((r, j) => {
      const v = Number(r.supertype_lfc) || 0;
      const x = padL + j * cellW;
      const y = i * cellH + 30;
      return `<g><title>${_escapeHtml(r.supertype)}: LFC = ${v.toFixed(3)}</title>` +
        `<rect x="${x}" y="${y}" width="${cellW - 1}" height="${cellH - 1}" fill="${lfcColor(v)}" stroke="#fff"/>` +
        `</g>`;
    }).join("");
    const median_str = `median ${median.toFixed(2)} (n=${arr.length})`;
    return `<g><text x="${padL - 8}" y="${i * cellH + 30 + 11}" text-anchor="end" font-size="11" class="${labelClass}">${_escapeHtml(sc)}</text>` +
      cellsRow +
      `<text x="${padL + maxCols * cellW + 6}" y="${i * cellH + 30 + 11}" font-size="10" fill="#6b7280">${median_str}</text></g>`;
  }).join("");
  const legend = `<g transform="translate(${padL}, ${H - 14})"><text x="0" y="0" font-size="10" fill="#6b7280">stratum: ${_escapeHtml(stratum)} CPS · color: red = up in AD, blue = down · one square per supertype, grouped by subclass</text></g>`;
  host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="100%" preserveAspectRatio="xMidYMid meet" class="attr-svg">` +
    cells + legend + `</svg>`;
}

function _renderSongOLSPanel(hostId, ctx, targetCellType) {
  const host = document.getElementById(hostId);
  if (!host) return;
  // Schema migrated from 3-pathway to 9-contrast. Fall back to legacy pathway
  // key if the contrast column isn't present on the loaded rows.
  const _useContrast = (ctx.songCdRows || []).some(r => r.contrast != null);
  const targetKey = _useContrast ? ctx.contrast : _attrPathwayFromContrast(ctx.contrast);
  const keyCol = _useContrast ? "contrast" : "pathway";
  const rows = (ctx.songCdRows || []).filter(r => r.cell_type === targetCellType);
  if (rows.length === 0) {
    host.innerHTML = `<div class="muted">No Song concordance rows for ${_escapeHtml(ctx.gene || '')} × ${_escapeHtml(targetCellType)}.</div>`;
    return;
  }
  const num = (v, d=3) => (v == null || !isFinite(Number(v))) ? "—" : Number(v).toFixed(d);
  const sciNum = (v) => (v == null || !isFinite(Number(v))) ? "—" : Number(v).toExponential(2);
  const tbody = rows.map(r => {
    const isTarget = r[keyCol] === targetKey;
    return `<tr${isTarget ? ' class="attr-song-selected"' : ''}>` +
      `<td>${_escapeHtml(r[keyCol])}${isTarget ? ' <span class="attr-badge attr-badge-info">selected</span>' : ''}</td>` +
      `<td class="attr-num" style="background:${_attrLfcColor(Number(r.song_lfc))}">${num(r.song_lfc, 3)}</td>` +
      `<td class="attr-num">${num(r.song_se, 3)}</td>` +
      `<td class="attr-num">${sciNum(r.song_pval)}</td>` +
      `<td class="attr-num">${num(r.song_fdr, 3)}</td>` +
      `<td class="attr-num">${num(r.n_animals, 0)}</td>` +
      `</tr>`;
  }).join("");
  const headerLabel = _useContrast ? "Contrast" : "Pathway";
  const lfcTitle = _useContrast
    ? "Factorial OLS coefficient at this contrast (10-param design with timepoint interactions). Pseudobulk log2(CPM+1), males only."
    : "Factorial OLS coefficient: App = β_App; Tau = β_Tau; ApTt = β_App + β_Tau + β_Int. Pseudobulk log2(CPM+1), males only, pooled across timepoints.";
  const pvalTitle = _useContrast
    ? "Two-sided p-value for the OLS contrast t-statistic with df_resid = n_animals − 10."
    : "Two-sided p-value for the OLS coefficient with df_resid = n_animals − 4.";
  const fdrTitle = `Benjamini–Hochberg FDR computed within (cell type, ${_useContrast ? "contrast" : "pathway"}).`;
  host.innerHTML =
    `<table class="attr-song-table">` +
      `<thead><tr>` +
        `<th>${headerLabel}</th>` +
        `<th title="${lfcTitle}">β (log2 LFC)</th>` +
        `<th title="Standard error of β.">SE</th>` +
        `<th title="${pvalTitle}">p-value</th>` +
        `<th title="${fdrTitle}">FDR</th>` +
        `<th title="Animals contributing to the OLS fit for this cell type.">n animals</th>` +
      `</tr></thead><tbody>${tbody}</tbody>` +
    `</table>`;
}

function _setAuditSelectors(ctx) {
  const siteSelect = document.getElementById("audit-site-select");
  if (siteSelect) {
    const current = siteSelect.value;
    const siteRows = ctx.siteRows || [];
    siteSelect.innerHTML = siteRows.slice(0, 300).map(r =>
      `<option value="${_escapeHtml(r.site_id)}">${_escapeHtml(r.site_id)} · ${_escapeHtml(r.gene_symbol || "")}</option>`
    ).join("");
    if (current && siteRows.some(r => String(r.site_id) === current)) siteSelect.value = current;
    else if (siteRows.some(r => String(r.site_id) === "2488")) siteSelect.value = "2488";
    else if (siteRows[0]) siteSelect.value = siteRows[0].site_id;
    siteSelect.onchange = () => renderActiveKinaseAuditTab(Store.state.selection.kinase);
  }
  const sampleSelect = document.getElementById("audit-sample-select");
  if (sampleSelect) {
    const current = sampleSelect.value || "plex2_130c_sn_mean";
    const cols = Object.keys((ctx.rawMatrix || [])[0] || {}).filter(c => c.endsWith("_sn_mean"));
    sampleSelect.innerHTML = cols.map(c => `<option value="${_escapeHtml(c)}">${_escapeHtml(c)}</option>`).join("");
    sampleSelect.value = cols.includes(current) ? current : (cols.includes("plex2_130c_sn_mean") ? "plex2_130c_sn_mean" : cols[0] || "");
    sampleSelect.onchange = () => renderActiveKinaseAuditTab(Store.state.selection.kinase);
  }
}

async function _loadKinaseAuditContext(kinase_id, seq) {
  _ensureKinaseIndexes();
  const K = PAYLOAD.kinases;
  const ki = _kinaseIdxById.get(kinase_id);
  if (ki == null) return null;
  const name = K.name[ki];
  const contrast = _selectedAuditContrast(K, ki);
  // Resolve track-suffixed audit keys: ST kinases load the unsuffixed files
  // (raw_phospho_normalized, mea_stoichiometry, ...), pY kinases load the
  // _pY siblings produced by kinase_attribution._track_output.
  const residueType = (K.residue_type && K.residue_type[ki]) || "ST";
  const tk = (base) => residueType === "Y" ? base + "_pY" : base;

  const [stoichRows, rawRows, olsRows, rawMatrix, stoichMatrix, uaRows,
         normRows, sampleRows, winsorRows, globalRows, subsRows,
         wmbAllRows, songCdAllRows, seaSuperAllRows] = await Promise.all([
    AuditDataStore.load(tk("mea_stoichiometry")),
    AuditDataStore.load(tk("mea_raw_phospho")).catch(() => []),
    AuditDataStore.load(tk("site_level_ols")),
    AuditDataStore.load(tk("raw_phospho_normalized")),
    AuditDataStore.load(tk("stoichiometry_matrix")),
    AuditDataStore.load("unified_attribution_full").catch(() => AuditDataStore.load("unified_attribution")),
    AuditDataStore.load("normalization_summary"),
    AuditDataStore.load("sample_mapping"),
    AuditDataStore.load(tk("winsorized_sites")),
    AuditDataStore.load(tk("mea_global_shift")),
    AuditDataStore.load(tk("mea_substrate_sets")).catch(() => []),
    AuditDataStore.load("wmb_kinase_expression").catch(() => []),
    AuditDataStore.load("song_concordance").catch(() => []),
    AuditDataStore.load("sea_ad_supertype_lfc").catch(() => []),
  ]);
  if (seq !== _kinaseAuditSeq || Store.state.selection.kinase !== kinase_id) return null;

  const meaStoich = stoichRows.filter(r => r.kinase === name);
  const meaRaw = rawRows.filter(r => r.kinase === name || r.kinase === K.gene_symbol[ki]);
  const leadRow = meaStoich.find(r => r.contrast === contrast) || meaStoich[0] || {};
  const motifs = new Set(String(leadRow["Leading substrates"] || "")
    .split(";").map(_normMotif).filter(Boolean));
  const motifBySite = new Map();
  for (const r of rawMatrix) if (motifs.has(_normMotif(r.motif))) motifBySite.set(String(r.site_id), r.motif);
  for (const r of stoichMatrix) if (motifs.has(_normMotif(r.motif))) motifBySite.set(String(r.site_id), r.motif);
  const siteIds = new Set(motifBySite.keys());
  const siteRows = olsRows.filter(r => siteIds.has(String(r.site_id))).map(r => ({
    ...r,
    motif: motifBySite.get(String(r.site_id)) || r.motif || "",
  }));
  const attrRows = uaRows.filter(r => r.kinase === name || r.gene_symbol === K.gene_symbol[ki]);
  const geneUpper = String(K.gene_symbol[ki] || "").toUpperCase();
  const wmbRows = (wmbAllRows || []).filter(r =>
    String(r.gene_symbol || "").toUpperCase() === geneUpper);
  const songCdRows = (songCdAllRows || []).filter(r =>
    String(r.gene_symbol || "").toUpperCase() === geneUpper);
  const seaSuperRows = (seaSuperAllRows || []).filter(r =>
    String(r.gene_symbol || "").toUpperCase() === geneUpper);

  // Substrate-set sites for this kinase + contrast (kinase library's substrate
  // gene set restricted to this contrast's prerank universe). This is what GSEA
  // walks for this kinase, and is upstream of the MEA leading-edge result.
  const substrateMotifs = new Set();
  for (const r of (subsRows || [])) {
    if (r.kinase === name && r.contrast === contrast) substrateMotifs.add(_normMotif(r.motif));
  }
  const substrateMotifBySite = new Map();
  for (const r of stoichMatrix) {
    const m = _normMotif(r.motif);
    if (m && substrateMotifs.has(m)) substrateMotifBySite.set(String(r.site_id), r.motif);
  }
  const substrateSiteIds = new Set(substrateMotifBySite.keys());
  const substrateSiteRows = olsRows.filter(r => substrateSiteIds.has(String(r.site_id))).map(r => ({
    ...r,
    motif: substrateMotifBySite.get(String(r.site_id)) || r.motif || "",
  }));

  return {
    kinase_id, ki, name, gene:K.gene_symbol[ki], contrast,
    residueType,
    meaStoich, meaRaw, leadRow, siteIds, siteRows, attrRows, olsRows,
    wmbRows, songCdRows, seaSuperRows,
    substrateMotifs, substrateSiteIds, substrateSiteRows, subsRows,
    rawMatrix, stoichMatrix, normRows, sampleRows, winsorRows, globalRows,
  };
}

function renderNumberTrace(rawMatrix, stoichMatrix, normRows, sampleRows, hostId) {
  const host = document.getElementById(hostId || "audit-number-trace");
  if (!host) return;
  const sid = _selectedAuditSite();
  const sample = _selectedAuditSample();
  const raw = rawMatrix.find(r => String(r.site_id) === String(sid)) || {};
  const sto = stoichMatrix.find(r => String(r.site_id) === String(sid)) || {};
  const sm = sampleRows.find(r => r.column_name === sample) || {};
  const norm = Object.fromEntries((normRows || []).map(r => [r.key, r.value]));
  const rows = [
    {step:"Selected site", source:"site_id", value:sid},
    {step:"Sample column", source:"sample_mapping.csv", value:sample},
    {step:"Animal / condition", source:"sample_mapping.csv", value:[sm.animal_id, sm.genotype, sm.timepoint].filter(Boolean).join(" / ")},
    {step:"Normalized phospho", source:"raw_phospho_normalized.csv", value:raw[sample]},
    {step:"Stoichiometry", source:"stoichiometry_matrix.csv", value:sto[sample]},
    {step:"Matched protein", source:"stoichiometry_matrix.csv", value:sto.matched_protein},
    {step:"IRS method", source:"normalization_summary.json", value:norm.normalization_method || ""},
    {step:"Raw workbook reference", source:"pipeline provenance", value:"Referenced by generated CSV lineage; raw workbooks are not embedded in v1."},
  ];
  _renderAuditTable(hostId || "audit-number-trace", "number_trace", rows, ["step","source","value"], "stoichiometry_matrix");
}

function renderSourceCatalog(listId, detailId) {
  const list = document.getElementById(listId || "audit-source-list");
  const detail = document.getElementById(detailId || "audit-source-detail");
  if (!list || !detail) return;
  const tables = _auditManifest();
  const keys = Object.keys(tables);
  if (!keys.includes(_sourceCatalogKey)) _sourceCatalogKey = keys[0];
  list.innerHTML = keys.map(k => {
    const t = tables[k];
    return `<button class="${k === _sourceCatalogKey ? "active" : ""}" data-key="${_escapeHtml(k)}">` +
      `<strong>${_escapeHtml(t.label || k)}</strong><br><span class="muted">${(t.row_count || 0).toLocaleString()} rows · ${(t.column_count || 0).toLocaleString()} cols</span></button>`;
  }).join("");
  list.querySelectorAll("button").forEach(btn => btn.addEventListener("click", () => {
    _sourceCatalogKey = btn.dataset.key;
    renderSourceCatalog(listId, detailId);
  }));
  const t = tables[_sourceCatalogKey] || {};
  detail.innerHTML = `<dl class="audit-kv">` +
    `<dt>Raw path</dt><dd>${_escapeHtml(t.source_path || "")}</dd>` +
    `<dt>Viewer path</dt><dd>${_escapeHtml(t.relative_path || "")}</dd>` +
    `<dt>Rows / columns</dt><dd>${(t.row_count || 0).toLocaleString()} / ${(t.column_count || 0).toLocaleString()}</dd>` +
    `<dt>Searchable columns</dt><dd>${_escapeHtml((t.columns || []).map(c => c.raw).join(", "))}</dd>` +
    `</dl><div id="audit-source-preview"></div>`;
  _renderAuditTable("audit-source-preview", _sourceCatalogKey, t.preview || [],
    (t.columns || []).slice(0, 12).map(c => c.raw), _sourceCatalogKey);
}

async function renderActiveKinaseAuditTab(kinase_id) {
  const body = document.getElementById("kinase-audit-body");
  if (!body || kinase_id == null) return;
  const tab = _activeKinaseAuditTab();
  document.querySelectorAll(".kinase-audit-tabs button").forEach(btn =>
    btn.classList.toggle("active", btn.dataset.auditTab === tab));
  const seq = _kinaseAuditSeq;
  body.innerHTML = '<div class="muted">Loading audit data...</div>';
  try {
    const ctx = await _loadKinaseAuditContext(kinase_id, seq);
    if (!ctx || seq !== _kinaseAuditSeq) return;
    _setAuditSelectors(ctx);
    const sample = _selectedAuditSample();
    const siteCols = _siteOlsColumns(ctx.contrast);
    const existingSiteCols = _existingCols(ctx.siteRows, siteCols);
    const wantedMea = ["kinase", "contrast", "ES", "NES", "p-value", "FDR", "Subs fraction", "Leading substrates"];

    if (tab === "measurement-trace") {
      body.innerHTML = `<p class="kinase-stage-note">Raw-to-stoichiometry receipt for the selected kinase and contrast's leading-substrate sites. The Sample control selects one animal/channel column; each row shows raw PTM, raw parent protein, IRS-normalized values, log2 transforms, and the stoichiometry subtraction used downstream.</p><div id="audit-measurement-trace"></div>`;
      const traceRows = await MeasurementTraceStore.load(sample, ctx.residueType);
      if (seq !== _kinaseAuditSeq) return;
      const rows = _substrateSiteRows(traceRows, ctx.siteIds, 500);
      _renderAuditTable("audit-measurement-trace", "measurement_trace", rows,
        ["site_id","gene_symbol","motif","protein_gene","matched_protein","raw_phospho","raw_protein","irs_phospho","irs_protein","log2_irs_phospho","log2_irs_protein","stoichiometry"],
        false);
    } else if (tab === "site-stats") {
      body.innerHTML = `<p class="kinase-stage-note">OLS contrast details for the selected kinase's leading-substrate phosphosites. Each row is one phosphosite, not one sample. The selected contrast controls which stoichiometry and raw-phospho effect columns are shown; n_obs_stoich is the total count of usable stoichiometry sample columns available for that site.</p><div id="audit-site-stats"></div>`;
      _renderAuditTable("audit-site-stats", "site_level_ols", ctx.siteRows, existingSiteCols, "site_level_ols");
    } else if (tab === "mea-input") {
      const prep = _buildPreparedMeaInput(ctx);
      const shiftRow = (ctx.globalRows || []).find(r => r.contrast === ctx.contrast) || null;
      const shiftVal = shiftRow ? Number(shiftRow.median_shift) : null;
      const winsorAll = _winsorContrastSummary(ctx.winsorRows, ctx.contrast);
      const totalSites = prep.prerank ? prep.prerank.total : null;
      const rankNote = AuditDataStore.fileMode
        ? `<span class="muted"> &middot; rank_in_contrast unavailable under file:// (serve over HTTP for the full prerank).</span>`
        : (prep.prerank ? ` &middot; ranked across ${totalSites.toLocaleString()} sites` : `<span class="muted"> &middot; rank could not be computed (missing shift or winsor bounds)</span>`);
      const formula = `<code>OLS &beta;<sub>stoich</sub> &minus; median shift &rarr; centered &rarr; winsorize [lo, hi] &rarr; clipped &rarr; GSEA prerank</code>`;
      const winsorClippedInSet = prep.rows.filter(r => r.was_winsorized === "yes").length;
      const winsorHeadline = (prep.bounds
          ? `bounds [${prep.bounds[0].toFixed(3)}, ${prep.bounds[1].toFixed(3)}] &middot; ${winsorAll.n.toLocaleString()} sites clipped across the contrast &middot; ${winsorClippedInSet} of this kinase's substrate sites clipped`
          : `<span class="muted">No winsorization receipts for this contrast.</span>`);
      const subsCount = (ctx.substrateMotifs && ctx.substrateMotifs.size) || 0;
      const leSubset = prep.rows.filter(r => r.in_leading_edge === "yes").length;
      const fallbackNote = (prep.sourceMode === "leading_edge_fallback")
        ? `<div class="muted" style="margin-top:.4em">mea_substrate_sets.csv unavailable; falling back to the leading-edge subset. Run <code>pixi run enrich</code> to materialize substrate-set receipts.</div>`
        : "";
      body.innerHTML =
        `<section class="audit-panel"><h4>Step 1 &middot; Global shift <span class="muted">(mea_global_shift.csv)</span></h4>` +
        `<p class="kinase-stage-note">Median stoichiometry LFC across the contrast's ranked sites${shiftVal != null ? `: <strong>${shiftVal.toFixed(4)}</strong>` : ""}. Subtracted from every ranked site before GSEA so the prerank is centered at zero. Contrast-level, not kinase-specific.</p>` +
        `<p class="kinase-stage-note muted"><strong>Why center?</strong> GSEA scores how a kinase's substrate set is concentrated at the top vs. bottom of the ranked list. If the entire contrast has a global up- or down-shift (e.g. a small bulk-level imbalance in normalization), that shift moves <em>every</em> kinase's ranks in one direction and inflates one tail's NES regardless of biology. Subtracting the contrast-level median forces equal numbers of positive and negative ranks so a significant NES reflects substrate-set concentration relative to background, not the global drift.</p>` +
        `<div id="audit-mea-shift"></div></section>` +
        `<section class="audit-panel"><h4>Step 2 &middot; Winsorization <span class="muted">(winsorized_sites.csv)</span></h4>` +
        `<p class="kinase-stage-note">Centered LFCs clipped to the 1st/99th percentile so individual sites cannot dominate the prerank. ${winsorHeadline}</p></section>` +
        `<section class="audit-panel audit-wide"><h4>Step 3 &middot; Prepared MEA input for this kinase <span class="muted">(mea_substrate_sets.csv &times; site_level_ols)</span></h4>` +
        `<p class="kinase-stage-note">One row per site whose motif is in this kinase's substrate set per the kinase library at threshold KL_THRESH. <strong>${subsCount.toLocaleString()}</strong> motifs &rarr; <strong>${prep.rows.length.toLocaleString()}</strong> sites in this contrast's prerank universe; ${leSubset} flagged as leading edge in the MEA result. Each row walks from the OLS &beta; through the median-shift correction and the winsor clip into the prerank position GSEA used to score this kinase. Sort by Rank to walk the ranked list as GSEA did.${fallbackNote}</p>` +
        `<div id="audit-mea-prepared"></div></section>`;
      _renderAuditTable("audit-mea-shift", "mea_global_shift", shiftRow ? [shiftRow] : [],
        ["contrast","median_shift","mean_before","pct_pos_before","pct_pos_after"], "mea_global_shift");
      _renderAuditTable("audit-mea-prepared", "mea_input_derived", prep.rows,
        ["rank_in_contrast","site_id","gene_symbol","motif","n_obs_stoich","raw_lfc","centered_lfc","clipped_lfc","was_winsorized","in_leading_edge"],
        false);
    } else if (tab === "mea-score") {
      const leadRow = ctx.leadRow || {};
      const rawRow = (ctx.meaRaw || []).find(r => r.contrast === ctx.contrast) || {};
      const compactMea = ["kinase","contrast","ES","NES","p-value","FDR","Subs fraction"];
      const fileNote = AuditDataStore.fileMode
        ? `<div class="muted" style="margin-top:.4em">Running enrichment requires the full prerank — serve over HTTP to render it.</div>`
        : "";
      body.innerHTML = `<p class="kinase-stage-note">The score for ${_escapeHtml(ctx.name)} on ${_escapeHtml(ctx.contrast)}: how the kinase's substrate set (Step 3) concentrates in the contrast prerank. Stoichiometry track is the primary signal; raw phospho is shown alongside for cross-track sanity.${fileNote}</p>` +
        `<section class="audit-panel"><h4>Score for ${_escapeHtml(ctx.contrast)}</h4>` +
        `<div id="audit-mea-scorecard"></div></section>` +
        `<section class="audit-panel"><h4>Running enrichment for ${_escapeHtml(ctx.contrast)}</h4>` +
        `<p class="kinase-stage-note">GSEA walk recomputed at view time from the cached prerank. The curve steps up at every substrate hit (weighted by |clipped LFC|) and down at every miss. Peak ES and the leading-edge prefix are marked. Tie-breaking among ~2% of sites with duplicated clipped values may differ from gseapy's internal order.</p>` +
        `<div id="audit-mea-running" style="height:300px"></div></section>` +
        `<section class="audit-panel"><h4>NES across all contrasts</h4>` +
        `<p class="kinase-stage-note">Stoichiometry NES bars: full-saturation when FDR &lt; threshold, faded when not significant. The selected contrast is outlined in black. Raw phospho NES shown as paired open diamonds. Click a bar to switch the selected contrast.</p>` +
        `<div id="audit-mea-trajectory" style="height:220px"></div></section>` +
        `<section class="audit-panel"><h4>Stoichiometry vs raw phospho for ${_escapeHtml(ctx.contrast)} <span class="muted">(mea_stoichiometry.csv vs mea_raw_phospho.csv)</span></h4>` +
        `<p class="kinase-stage-note">Per-metric comparison of the same kinase &times; contrast scored against two preprocessing tracks. Stoichiometry is primary; raw phospho is the sensitivity check. Δ = stoichiometry − raw. Sign-flipping or significance divergence flags abundance-driven vs activity-driven signals.</p>` +
        `<div id="audit-mea-comparison"></div></section>` +
        `<section class="audit-panel audit-wide"><h4>Per-cell-type decomposition for ${_escapeHtml(ctx.contrast)} <span class="muted">(kinase_enrichment_wmb.csv)</span></h4>` +
        `<p class="kinase-stage-note">Pseudo-deconvoluted MEA NES per WMB class for this kinase &times; contrast, sorted by NES. Bars are filled when FDR &lt; threshold, faded otherwise. The vertical line marks the bulk NES from the live pipeline (solid black = bulk significant, dashed gray = ns). Comparing the spread of class bars to the bulk line shows whether the bulk signal localizes to a class, is averaged across many, or is masked by canceling classes.</p>` +
        `<div id="audit-mea-decomp"></div></section>`;
      _renderMeaScorecard("audit-mea-scorecard", leadRow, rawRow, ctx);
      _renderRunningEnrichmentPlot("audit-mea-running", ctx);
      _renderMeaTrajectory("audit-mea-trajectory", kinase_id, ctx);
      try { _renderDecompPanel("audit-mea-decomp", kinase_id, ctx, leadRow); }
      catch (decompErr) {
        console.error("decomp panel failed", decompErr);
        const dh = document.getElementById("audit-mea-decomp");
        if (dh) dh.innerHTML = `<div class="muted">Decomposition panel failed: ${_escapeHtml(String(decompErr && decompErr.message || decompErr))}</div>`;
      }
      const cmpRows = _buildMeaComparisonRows(leadRow, rawRow);
      const diag = _diagnoseRawAbsence(ctx, rawRow);
      const diagBanner = diag
        ? `<div class="kinase-stage-note muted" style="margin-bottom:.6em">⚠ ${diag.note}</div>`
        : "";
      const cmpHost = document.getElementById("audit-mea-comparison");
      if (cmpHost) {
        cmpHost.innerHTML = diagBanner + `<div id="audit-mea-cmp-table"></div>`;
        _renderAuditTable("audit-mea-cmp-table", "mea_track_comparison", cmpRows,
          ["metric","stoich","raw","delta"], false);
      }
    } else if (tab === "attribution") {
      body.innerHTML =
        `<section class="audit-panel audit-wide"><h4>Verdict across cell types <span class="muted">for ${_escapeHtml(ctx.name)} / ${_escapeHtml(ctx.contrast)}</span></h4>` +
        `<div id="attr-verdict"></div></section>` +
        `<section class="audit-panel audit-wide"><h4>Evidence drawer</h4>` +
        `<div id="attr-drawer"></div></section>` +
        `<section class="audit-panel"><h4>Raw attribution rows <span class="muted">(unified_attribution.csv)</span></h4>` +
        `<div id="audit-attribution"></div></section>`;
      _renderAttributionVerdict("attr-verdict", ctx);
      _renderAuditTable("audit-attribution", "unified_attribution", ctx.attrRows,
        ["kinase","gene_symbol","contrast","cell_type","combined_confidence","wmb_specificity","wmb_mean_log2_expression","wmb_fraction_cells_expressing","sea_ad_lfc","song_lfc","combined_score","evidence_basis"],
        "unified_attribution");
    }
  } catch (e) {
    if (seq !== _kinaseAuditSeq) return;
    console.error("audit tab failed", e);
    const msg = e && (e.message || e.toString && e.toString()) || String(e);
    body.innerHTML = `<div class="muted">Audit table load failed: ${_escapeHtml(msg)}</div>`;
  }
}

