
// attribution_view.js — shared accordion verdict-table engine for all cohorts.
// Exposes AttributionView.render(hostId, ctx, manifest) where manifest is a
// per-cohort declaration (columns, sections, getRows, dedupCmp, defaultSort,
// rowVisible, bulkAnchor, superGroups).
//
// Leaf cell renderers defined here are global so cohort-specific files
// (kinase_fivexfad.js) can reference them until those cohorts are refactored
// in later sprints.

// ---- Leaf cell renderers (global scope — referenced by kinase_fivexfad.js) --

function _detGateCell(detected, frac, where) {
  const ctxNoun = where || "this cell type";
  // null detection = no call exists for this kinase here (e.g. a reference's
  // probe panel does not cover it) — distinct from a measured "not detected".
  if (detected == null) {
    return `<span class="muted" title="No detection call for ${ctxNoun} (kinase outside this panel).">n/a</span>`;
  }
  const detBool = detected === true || detected === "True" || detected === "true";
  const fracPct = (frac != null && isFinite(frac)) ? (Number(frac) * 100).toFixed(0) + "%" : "";
  if (!detBool) {
    return `<span class="song-det-no" title="Not detected in ${ctxNoun} (${fracPct ? fracPct + ' of cells' : 'frac n/a'} < 10% threshold)">✗ ${fracPct}</span>`;
  }
  return `<span class="song-det-yes" title="Detected in ${ctxNoun} (${fracPct ? fracPct + ' of cells' : ''}; ≥10% threshold passed)">✓ ${fracPct}</span>`;
}

function _concTierCell(tier) {
  const tierN = Number(tier) || 0;
  if (tierN <= 0) return `<span class="muted" title="At or below the even share across all cell types (no concentration tier)">—</span>`;
  const cls = tierN >= 10 ? "vhi" : tierN >= 5 ? "hi" : tierN >= 2 ? "mid" : "lo";
  return `<span class="badge ${cls}" title="Concentration ≥${tierN}× the even 1/N share of total expression across all cell types">≥${tierN}×</span>`;
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

// Middle header row: spans the leaf columns that share a `sub` label under one
// labeled parent; consecutive columns without `sub` collapse to a blank spacer.
function _attrSubGroupRow(cols) {
  let html = `<tr class="attr-verdict-subgroup">`;
  let i = 0;
  while (i < cols.length) {
    if (!cols[i].sub) {
      let span = 0;
      while (i < cols.length && !cols[i].sub) { span++; i++; }
      html += `<th class="attr-subgroup-spacer" colspan="${span}"></th>`;
    } else {
      const sub = cols[i].sub;
      const label = cols[i].subLabel || sub;
      const title = cols[i].subTitle ? ` title="${_escapeHtml(cols[i].subTitle)}"` : "";
      let span = 0;
      while (i < cols.length && cols[i].sub === sub) { span++; i++; }
      html += `<th class="attr-subgroup-th" colspan="${span}"${title}>${label}</th>`;
    }
  }
  return html + `</tr>`;
}

// Confidence pill cell: shown only on the kinase's dominant cell type; a muted
// dot on all other rows so the pill is not read as a per-cell-type claim.
function _attrVerdictConfCell(r) {
  const isHome = r.specificity_celltype && r.cell_type === r.specificity_celltype;
  if (!isHome) {
    return `<span class="muted" title="Confidence is a kinase-level property (cell-type exclusivity), shown on this kinase's dominant cell type${r.specificity_celltype ? ': ' + _escapeHtml(r.specificity_celltype) : ''}.">·</span>`;
  }
  const dir = (r.direction_tier && r.direction_tier !== "none")
    ? " · disease-direction concordance: " + String(r.direction_tier).replace("_", " ") : "";
  const tip = _escapeHtml((r.confidence_basis || "none") + dir);
  return `<span class="${_attrConfidenceClass(r.confidence_tier)}" title="${tip}">${_escapeHtml((r.confidence_tier || "").replace("_", " "))}</span>`;
}

// Sort comparator for a single column.
function _attrVerdictCmp(a, b, key, type, asc) {
  const dir = asc ? -1 : 1;
  if (type === "num") {
    const va = (a[key] == null || !isFinite(a[key])) ? null : Number(a[key]);
    const vb = (b[key] == null || !isFinite(b[key])) ? null : Number(b[key]);
    return numCmp(va, vb, dir);
  } else if (type === "conf") {
    const va = _CONF_RANK[a[key]] ?? -1;
    const vb = _CONF_RANK[b[key]] ?? -1;
    return numCmp(va, vb, dir);
  } else {
    const va = (a[key] || "").toString();
    const vb = (b[key] || "").toString();
    return asc ? va.localeCompare(vb) : vb.localeCompare(va);
  }
}

// ---- Shared section renderers ------------------------------------------------

// eff band labels for the specificity-verdict section.
function _specEffBand(eff) {
  if (eff == null || !isFinite(eff)) return {txt:"n/a", badge:""};
  if (eff <= 1.5) return {txt:"≈ one cell type (≤ 1.5)", badge:"vhi"};
  if (eff <= 3.0) return {txt:"a few cell types (≤ 3)", badge:"mid"};
  return {txt:"broadly expressed (> 3)", badge:"lo"};
}

// Module-scope formatters used by _renderSpecificityVerdictShell and adapters.
const _specPct = (v) => (v == null || !isFinite(v)) ? "—" : (Number(v) * 100).toFixed(0) + "%";
const _specF2  = (v) => (v == null || !isFinite(v)) ? "—" : Number(v).toFixed(2);

// §0 cohort-agnostic shell — owns the <section> wrapper, the attr-spec-grid
// two-block layout, eff-band display (via _specEffBand), confidence pill (via
// _attrConfidenceClass), and the shared pct/f2 formatters. Does NOT reference
// any cohort-specific field directly.
//
// adapter shape:
//   { conf, basis, eff,
//     thisCellTitle,    // string: cell type name for the block header
//     thisCellDlHtml,   // pre-rendered <dt>/<dd> pairs HTML (inside a <dl>)
//     thisCellExtraHtml,// optional: extra HTML after the <dl> (e.g. collapse list)
//     kinaseTitleHtml,  // string: block header (e.g. "This kinase overall")
//     kinaseDlHtml,     // pre-rendered <dt>/<dd> pairs HTML (inside a <dl>)
//     reconcileSentence,// pre-built reconcile sentence (cohort supplies noun)
//   }
function _renderSpecificityVerdictShell(adapter) {
  const conf = adapter.conf || "none";
  const eff  = (adapter.eff != null && isFinite(adapter.eff)) ? Number(adapter.eff) : null;
  const band = _specEffBand(eff);

  const thisCell =
    `<div class="attr-spec-block"><div class="attr-spec-h">${_escapeHtml(adapter.thisCellHeaderLabel || "This cell type")} — ${_escapeHtml(adapter.thisCellTitle || "")}</div>` +
    `<dl class="attr-spec-dl">${adapter.thisCellDlHtml || ""}</dl>` +
    (adapter.thisCellExtraHtml || "") +
    `</div>`;

  const kinaseBlock =
    `<div class="attr-spec-block"><div class="attr-spec-h">${adapter.kinaseTitleHtml || "This kinase overall"}</div>` +
    `<dl class="attr-spec-dl">${adapter.kinaseDlHtml || ""}</dl>` +
    `<p class="attr-spec-reconcile">${adapter.reconcileSentence || ""}</p></div>`;

  return `<section class="attr-section attr-section-wide attr-spec-section">` +
    `<h5>§0 · Specificity verdict <span class="muted">— why the confidence pill reads "${_escapeHtml(conf.replace('_', ' '))}"</span></h5>` +
    `<div class="attr-spec-grid">${thisCell}${kinaseBlock}</div></section>`;
}

// §0 — Song specificity verdict. Builds a Song adapter and calls the shell.
function _renderSpecificityVerdict(row, ctx, kstats) {
  const conf = row.confidence_tier || "none";
  const eff = (row.song_unit_effective_n != null && isFinite(row.song_unit_effective_n)) ? Number(row.song_unit_effective_n) : null;
  const band = _specEffBand(eff);
  const effNative = (row.song_effective_n != null && isFinite(row.song_effective_n)) ? Number(row.song_effective_n) : null;
  const detected = row.song_detected === true;
  const totalEven = kstats.nClustersTotal > 0 ? 1 / kstats.nClustersTotal : null;
  const shareTotal = (row.song_concentration_of_total != null && isFinite(row.song_concentration_of_total))
    ? Number(row.song_concentration_of_total) : null;
  const tier = Number(row.song_concentration_tier) || 0;
  const tierTip = totalEven != null
    ? `≥${tier||1}× the even ${_specPct(totalEven)} share of total expression (1/${kstats.nClustersTotal} cell types)`
    : "";
  const tierBadge = tier > 0
    ? `<span class="badge ${tier>=10?'vhi':tier>=5?'hi':tier>=2?'mid':'lo'}" title="${_escapeHtml(tierTip)}">≥${tier}×</span>`
    : `<span class="muted" title="${_escapeHtml(tierTip)}">— (at or below the all-cell-types even share)</span>`;

  let thisCellExtraHtml = "";
  if (row.specificity_collapsed) {
    const SU = (PAYLOAD.specificity_units && PAYLOAD.specificity_units.units) || {};
    const kids = (SU[row.specificity_unit] && SU[row.specificity_unit].children) || [];
    const items = kids.map(cl => {
      const cr = kstats.byCell.get(cl);
      const det = cr && cr.song_detected === true;
      const fr = cr ? _specPct(cr.song_fraction_cells_expressing) : "—";
      return `<li>${det ? "✓" : "✗"} ${_escapeHtml(cl)} <span class="muted">${fr} cells</span></li>`;
    }).join("");
    thisCellExtraHtml = `<div class="attr-spec-children"><div class="muted">This unit collapses ${kids.length} over-split Song clusters:</div><ul class="attr-spec-kidlist">${items}</ul></div>`;
  }

  const thisCellDlHtml =
    `<dt>Cells expressing</dt><dd>${_specPct(row.song_fraction_cells_expressing)}${detected ? "" : ' <span class="muted">(below 10%; specificity denominator still uses all clusters)</span>'}</dd>` +
    `<dt>Share of total expr</dt><dd>${_specPct(shareTotal)}` +
      (totalEven != null ? ` <span class="muted">(even share = ${_specPct(totalEven)} across all ${kstats.nClustersTotal} cell types)</span>` : "") + `</dd>` +
    `<dt>Concentration</dt><dd>${tierBadge}</dd>` +
    `<dt>Specificity unit</dt><dd>${_escapeHtml(row.specificity_unit_label || "—")}` +
      (row.specificity_collapsed ? ` <span class="muted">(collapsed)</span>` : "") + `</dd>`;

  const reconcileSentence =
    `<strong>eff = ${_specF2(eff)}</strong> — the effective number of specificity units ${_escapeHtml(ctx.gene || "")} concentrates in ` +
    `(1 / Σ unit share², over all specificity units). ${band.txt} → confidence ` +
    `<strong>${_escapeHtml(conf.replace("_", " "))}</strong>` +
    (eff != null && eff > 3 && tier >= 5
      ? `, even though this cell type alone holds ≥${tier}× the even share of total expression.`
      : ".");

  const kinaseDlHtml =
    `<dt>Detected in</dt><dd>${kstats.nDetected} / ${kstats.nClustersTotal} cell types <span class="muted">(${kstats.nUnitsDetected} / ${kstats.nUnitsTotal} specificity units; detection is shown separately from specificity denominator)</span></dd>` +
    `<dt>Effective # units (eff)</dt><dd><span class="badge ${band.badge}">${_specF2(eff)}</span> <span class="muted">${band.txt}</span></dd>` +
    `<dt>Subtype spread</dt><dd>${_specF2(effNative)} <span class="muted">effective # transcriptomic clusters over all clusters (sub-types within a cell type — not the tier input)</span></dd>` +
    `<dt>Confidence</dt><dd><span class="${_attrConfidenceClass(conf)}" title="${_escapeHtml(row.confidence_basis || '')}">${_escapeHtml(conf.replace('_', ' '))}</span></dd>`;

  return _renderSpecificityVerdictShell({
    conf,
    eff,
    thisCellTitle: row.cell_type,
    thisCellDlHtml,
    thisCellExtraHtml,
    kinaseTitleHtml: "This kinase overall",
    kinaseDlHtml,
    reconcileSentence,
  });
}

// WMB Seurat-style dot plot for the clicked cell type.
function _renderWMBDotPlot(hostId, ctx, targetCellType) {
  const host = document.getElementById(hostId);
  if (!host) return;
  const rows = (ctx.wmbRows || []).slice();
  if (rows.length === 0) {
    host.innerHTML = `<div class="muted">No WMB rows for ${_escapeHtml(ctx.gene || '')}.</div>`;
    return;
  }
  rows.sort((a, b) => (Number(b.mean_log2_expression) || 0) - (Number(a.mean_log2_expression) || 0));
  const _c2w = (PAYLOAD.specificity_units && PAYLOAD.specificity_units.cluster_to_wmb_class) || {};
  const targetWmbClass = _c2w[targetCellType] || targetCellType;
  const maxExpr = Math.max(...rows.map(r => Number(r.mean_log2_expression) || 0), 1);
  const W = 720, H = 18 * rows.length + 60, padL = 160, padT = 30, padR = 40;
  const innerW = W - padL - padR;
  const x0 = padL, x1 = padL + innerW;
  const colorAt = (v) => {
    const t = Math.max(0, Math.min(1, v / maxExpr));
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
    const isTarget = r.cell_type === targetWmbClass;
    const stroke = isTarget ? "#111827" : "#cbd5e0";
    const strokeW = isTarget ? 2 : 0.8;
    const labelClass = isTarget ? "attr-dot-label attr-dot-label-target" : "attr-dot-label";
    const tier = Number(r.concentration_tier) || 0;
    const concStr = tier > 0 ? `≥${tier}× even share of total expr` : "≤ even share of total expr";
    const title = `${r.cell_type}: log2 expr = ${expr.toFixed(2)}, fraction = ${frac.toFixed(2)}, ${concStr}`;
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

// SEA-AD per-supertype LFC heatmap for the clicked cell type.
function _renderSEAADHeatmap(hostId, ctx, targetCellType) {
  const host = document.getElementById(hostId);
  if (!host) return;
  const stratumByPathway = {App: "early", Tau: "late", ApTt: "full"};
  const pathway = String(ctx.contrast || "").split("_")[0] || "";
  const stratum = stratumByPathway[pathway] || "full";
  const rows = (ctx.seaSuperRows || []).filter(r => r.stratum === stratum);
  if (rows.length === 0) {
    host.innerHTML = `<div class="muted">No SEA-AD supertype rows for ${_escapeHtml(ctx.gene || '')} (stratum: ${_escapeHtml(stratum)}).</div>`;
    return;
  }
  const bySubclass = new Map();
  for (const r of rows) {
    const sc = r.subclass || "(unknown)";
    if (!bySubclass.has(sc)) bySubclass.set(sc, []);
    bySubclass.get(sc).push(r);
  }
  const _c2s = (PAYLOAD.specificity_units && PAYLOAD.specificity_units.cluster_to_seaad_subclass) || {};
  const targetSubclasses = new Set(_c2s[targetCellType] || []);
  const subclasses = Array.from(bySubclass.keys()).sort((a, b) => {
    const at = targetSubclasses.has(a), bt = targetSubclasses.has(b);
    if (at && !bt) return -1;
    if (bt && !at) return 1;
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
    const isTarget = targetSubclasses.has(sc);
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

// Within-cohort OLS panel (Song concordance shard). Renamed from
// _renderSongOLSPanel for cohort-neutral naming; the Song manifest passes
// `SliceCache.loadSongConcordance` as the loader.
function _renderWithinCohortOLSPanel(hostId, ctx, targetCellType, loader) {
  const host = document.getElementById(hostId);
  if (!host) return;
  if (!loader || typeof loader !== "function") {
    host.innerHTML = `<div class="muted">Within-cohort OLS shards unavailable in this build.</div>`;
    return;
  }
  const gene = ctx.gene || "";
  const reqGene = gene;
  const reqContrast = ctx.contrast;
  const reqCell = targetCellType;
  host.innerHTML = `<div class="muted">Loading Song shard…</div>`;
  loader(gene).then(allRows => {
    if (ctx.gene !== reqGene || ctx.contrast !== reqContrast) return;
    const stillThis = document.getElementById(hostId);
    if (!stillThis || stillThis !== host) return;
    if (!Array.isArray(allRows) || allRows.length === 0) {
      host.innerHTML = `<div class="muted">No Song LFC shard for ${_escapeHtml(gene)}.</div>`;
      return;
    }
    const rows = allRows.filter(r => r.cell_type === reqCell);
    if (rows.length === 0) {
      host.innerHTML = `<div class="muted">No Song LFC rows for ${_escapeHtml(gene)} × ${_escapeHtml(reqCell)}.</div>`;
      return;
    }
    const num = (v, d=3) => (v == null || !isFinite(Number(v))) ? "—" : Number(v).toFixed(d);
    const sciNum = (v) => (v == null || !isFinite(Number(v))) ? "—" : Number(v).toExponential(2);
    const tbody = rows.map(r => {
      const isTarget = r.contrast === reqContrast;
      return `<tr${isTarget ? ' class="attr-song-selected"' : ''}>` +
        `<td>${_escapeHtml(r.contrast)}${isTarget ? ' <span class="attr-badge attr-badge-info">selected</span>' : ''}</td>` +
        `<td class="attr-num" style="background:${_attrLfcColor(Number(r.song_lfc))}">${num(r.song_lfc, 3)}</td>` +
        `<td class="attr-num">${num(r.song_se, 3)}</td>` +
        `<td class="attr-num">${sciNum(r.song_pval)}</td>` +
        `<td class="attr-num">${num(r.song_fdr, 3)}</td>` +
        `<td class="attr-num">${num(r.n_animals, 0)}</td>` +
        `</tr>`;
    }).join("");
    const lfcTitle = "Factorial OLS coefficient at this contrast (10-param design with timepoint interactions). Pseudobulk log2(CPM+1), males only.";
    const pvalTitle = "Two-sided p-value for the OLS contrast t-statistic with df_resid = n_animals − 10.";
    const fdrTitle = "Benjamini–Hochberg FDR computed within (cell type, contrast).";
    host.innerHTML =
      `<table class="attr-song-table">` +
        `<thead><tr>` +
          `<th>Contrast</th>` +
          `<th title="${lfcTitle}">β (log2 LFC)</th>` +
          `<th title="Standard error of β.">SE</th>` +
          `<th title="${pvalTitle}">p-value</th>` +
          `<th title="${fdrTitle}">FDR</th>` +
          `<th title="Animals contributing to the OLS fit for this cell type.">n animals</th>` +
        `</tr></thead><tbody>${tbody}</tbody>` +
      `</table>`;
  }).catch(err => {
    console.error("Song LFC shard fetch failed", err);
    host.innerHTML = `<div class="muted">Failed to load Song shard: ${_escapeHtml(String(err && err.message || err))}</div>`;
  });
}

// Per-cell-type decomposition OLS substrate-site table.
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
    if (ctx.gene !== reqGene || ctx.contrast !== reqContrast) return;
    const stillThis = document.getElementById(hostId);
    if (!stillThis || stillThis !== host) return;
    if (!Array.isArray(rows) || rows.length === 0) {
      host.innerHTML = `<div class="muted">No per-cell OLS shard for this kinase.</div>`;
      return;
    }
    const sub = rows.filter(r => Number(r.contrast_id) === cId
                              && String(r.cell_type) === String(reqCell));
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
        `<th title="Per-cell β: substrate-site stoichiometry coefficient from the per-(group, cell_type) OLS, on the deconvoluted phospho signal. Bold when per-cell p < 0.05.">Per-cell β</th>` +
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

// Song-cluster → WMB-class → SEA-AD-subclass crosswalk line.
function _refCrosswalkLine(cellType) {
  const SU = PAYLOAD.specificity_units || {};
  const wmb = (SU.cluster_to_wmb_class || {})[cellType] || "—";
  const sea = (SU.cluster_to_seaad_subclass || {})[cellType];
  const seaTxt = (sea && sea.length) ? sea.join(", ") : "— (not in SEA-AD MTG taxonomy)";
  return `<div class="attr-crosswalk" title="Vocabulary crosswalk: the WMB class and SEA-AD subclass(es) this Song cluster maps to. The reference panels below outline these rows.">` +
    `<span class="attr-xw-tag">Reference mapping</span> ` +
    `<span class="attr-xw-vocab">Song</span> ${_escapeHtml(cellType)} ` +
    `<span class="attr-xw-arrow">→</span> <span class="attr-xw-vocab">WMB</span> ${_escapeHtml(wmb)} ` +
    `<span class="attr-xw-arrow">→</span> <span class="attr-xw-vocab">SEA-AD</span> ${_escapeHtml(seaTxt)}</div>`;
}

// Human-location corroboration line (§1 inset, Song cohort).
function _humanLocationLine(row) {
  const v = (row.human_location_score != null && isFinite(row.human_location_score)) ? Number(row.human_location_score) : null;
  if (v == null) return `<p class="muted attr-caption">Human location score: n/a for this cell type.</p>`;
  const f2 = (x) => (x != null && isFinite(x)) ? Number(x).toFixed(2) : "—";
  const strong = v >= 1.0;
  return `<p class="attr-caption">Human location score: <strong>${v.toFixed(2)}</strong> log2 over brain mean` +
    (strong
      ? ` <span class="attr-badge attr-badge-info">strong ≥ 1.0 — corroborates</span>`
      : ` <span class="muted">(below the 1.0 corroboration threshold)</span>`) +
    ` <span class="muted">(SEA-AD ${f2(row.seaad_location_score)} · HBCA ${f2(row.hbca_location_score)}, max taken)</span></p>`;
}

// Allen Brain Atlas external links.
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

// ---- Shared detail renderer (called per expanded row) -----------------------

// Renders the inline accordion detail body. Each section in manifest.sections
// is called with (sectionHostId, ctx, row); the section renderer manages its
// own async and inner DOM updates.
function _renderAttributionDetailShared(hostId, ctx, row, kstats, manifest) {
  const host = document.getElementById(hostId);
  if (!host) return;
  const cellType = row && row.cell_type || "";
  const gene = ctx.gene || "";

  // Build section markup stubs (empty divs with stable ids).
  const sectionStubs = (manifest.sections || []).map((sec, si) => {
    const secHostId = `${hostId}-sec${si}`;
    return {sec, secHostId};
  });

  // Crosswalk (Song-specific; rendered if manifest says so).
  const crosswalkHtml = manifest.renderCrosswalk
    ? _refCrosswalkLine(cellType)
    : "";

  // Build the section layout HTML. The manifest may supply a renderDetailLayout
  // function to control how sections are arranged (e.g. to wrap some in a grid
  // div). If not supplied, sections are rendered flat in order.
  let sectionLayoutHtml;
  if (manifest.renderDetailLayout) {
    sectionLayoutHtml = manifest.renderDetailLayout(sectionStubs, ctx, row, kstats);
  } else {
    sectionLayoutHtml = sectionStubs.map(({sec, secHostId}) => {
      const titleHtml = sec.title || "";
      const captionHtml = sec.caption || "";
      const wide = sec.wide ? " attr-section-wide" : "";
      return `<section class="attr-section${wide}">` +
        (titleHtml ? `<h5>${titleHtml}</h5>` : "") +
        (captionHtml ? `<p class="muted attr-caption">${captionHtml}</p>` : "") +
        `<div id="${secHostId}"></div></section>`;
    }).join("");
  }

  host.innerHTML =
    `<div class="attr-drawer-header"><strong>${_escapeHtml(cellType)}</strong>` +
    ` &middot; <span class="muted">${_escapeHtml(gene)} / ${_escapeHtml(ctx.contrast)}</span>` +
    ` &middot; ${_allenABALink(gene)}</div>` +
    crosswalkHtml +
    sectionLayoutHtml;

  // Fire each section's render function (regardless of layout variant).
  for (const {sec, secHostId} of sectionStubs) {
    sec.render(secHostId, ctx, row, kstats);
  }
}

// ---- Main engine ------------------------------------------------------------

const AttributionView = (() => {
  function render(hostId, ctx, manifest) {
    const host = document.getElementById(hostId);
    if (!host) return;

    const allRows = manifest.getRows(ctx);
    if (allRows.length === 0) {
      host.innerHTML = `<div class="muted">No attribution rows in ${_escapeHtml(ctx.contrast || "")}.</div>`;
      return;
    }

    // Defensive dedup: keep canonical row per (contrast_id, cell_type).
    const deduped = new Map();
    for (const r of allRows) {
      const k = manifest.dedupKey(r);
      const prev = deduped.get(k);
      if (!prev || manifest.dedupCmp(r, prev) < 0) deduped.set(k, r);
    }
    const rows = Array.from(deduped.values());

    // Let the manifest attach any derived columns (e.g. decomp NES, bulk_match).
    if (manifest.attachDerived) manifest.attachDerived(rows, ctx);

    // Sort.
    const cols = manifest.columns;
    const visibleCols = cols.filter(c => !c.hidden);
    const defaultSortKey = manifest.defaultSort.key;
    const defaultSortAsc = !!manifest.defaultSort.asc;
    const sortKey = host.dataset.sortKey || defaultSortKey;
    const sortAsc = host.dataset.sortAsc != null ? host.dataset.sortAsc === "1" : defaultSortAsc;
    const sortCol = cols.find(c => c.key === sortKey) || cols.find(c => c.key === defaultSortKey) || cols[cols.length - 1];
    rows.sort((a, b) => _attrVerdictCmp(a, b, sortCol.key, sortCol.type, sortAsc));
    // Apply tiebreak only when sorting by the default key (e.g. confidence_tier).
    if (manifest.sortTiebreak && sortCol.key === defaultSortKey) {
      rows.sort((a, b) => {
        const primary = _attrVerdictCmp(a, b, sortCol.key, sortCol.type, sortAsc);
        if (primary !== 0) return primary;
        return manifest.sortTiebreak(a, b);
      });
    }

    // Row visibility.
    const showAllId = `${hostId}-show-all`;
    const showAll = host.dataset.showAll === "1";
    const visibleRows = (manifest.rowVisible && !showAll)
      ? rows.filter(manifest.rowVisible)
      : rows;
    const hiddenCount = rows.length - visibleRows.length;

    // Render each verdict row + its paired hidden detail row.
    // Column c.render(r) must return the complete <td>...</td> string.
    // Columns without c.render get a default attr-num cell.
    const _numFmt = (v, d=3) => (v == null || !isFinite(v)) ? "" : Number(v).toFixed(d);
    const tbody = visibleRows.map((r, i) => {
      const cells = visibleCols.map(c => {
        if (c.render) return c.render(r, _numFmt, ctx);
        return `<td class="attr-num">${_escapeHtml(String(r[c.key] == null ? "" : r[c.key]))}</td>`;
      }).join("");
      const verdictTr = `<tr data-row-idx="${i}" data-cell-type="${_escapeHtml(r.cell_type)}" aria-expanded="false" class="attr-verdict-row">` +
        cells +
        `</tr>`;
      const detailTr = `<tr class="attr-detail-row" data-detail-idx="${i}" hidden>` +
        `<td colspan="${visibleCols.length}"><div class="attr-detail-host" id="${hostId}-detail-${i}"></div></td></tr>`;
      return verdictTr + detailTr;
    }).join("");

    // Column headers.
    const headCells = visibleCols.map(c => {
      const arrow = (c.key === sortCol.key) ? (sortAsc ? " ▲" : " ▼") : "";
      const title = c.title ? ` title="${_escapeHtml(c.title)}"` : "";
      return `<th class="attr-verdict-th" data-sort-key="${c.key}"${title}>${c.label}${arrow}</th>`;
    }).join("");

    // Super-group header row.
    const superHead = manifest.superGroupRow
      ? manifest.superGroupRow(visibleCols)
      : "";

    // Bulk anchor block above the table.
    const bulkData = manifest.bulkAnchor ? manifest.bulkAnchor(ctx) : null;
    const bulkAnchorHtml = bulkData ? _renderBulkAnchor(bulkData, ctx) : "";

    // Exclusivity summary line (Song cohort provides this via manifest).
    const exclHtml = manifest.exclSummary ? manifest.exclSummary(rows, ctx) : "";

    // Show-all toggle.
    let toggleHtml = "";
    if (manifest.showAllLabel && hiddenCount > 0) {
      toggleHtml = `<div class="attr-verdict-toggle"><label><input type="checkbox" id="${showAllId}"${showAll ? " checked" : ""}> ${manifest.showAllLabel} <span class="muted">(${hiddenCount} hidden)</span></label></div>`;
    } else if (manifest.showAllLabel && showAll && rows.length > 0) {
      toggleHtml = `<div class="attr-verdict-toggle"><label><input type="checkbox" id="${showAllId}" checked> Showing all cell types</label></div>`;
    }

    // Explainer collapsible (manifest supplies raw HTML or null).
    const explainerHtml = manifest.explainerHtml || "";

    host.innerHTML =
      bulkAnchorHtml +
      exclHtml +
      `<table class="attr-verdict-table">` +
        `<thead>${superHead}${_attrSubGroupRow(visibleCols)}<tr>${headCells}</tr></thead><tbody>${tbody}</tbody>` +
      `</table>` +
      toggleHtml +
      explainerHtml;

    // Compute kstats for the detail renderer (kinase-level detection summary).
    const kstats = manifest.buildKstats ? manifest.buildKstats(rows, ctx) : null;

    // Single-expand accordion.
    const _attrCollapseAll = () => {
      host.querySelectorAll("tr.attr-verdict-row").forEach(r => {
        r.classList.remove("attr-verdict-selected");
        r.setAttribute("aria-expanded", "false");
        const ch = r.querySelector(".attr-chevron"); if (ch) ch.textContent = "▸";
      });
      host.querySelectorAll("tr.attr-detail-row").forEach(dr => {
        dr.hidden = true;
        const h = dr.querySelector(".attr-detail-host"); if (h) h.innerHTML = "";
      });
    };
    const _attrExpand = (idx) => {
      _attrCollapseAll();
      const vr = host.querySelector(`tr.attr-verdict-row[data-row-idx="${idx}"]`);
      const dr = host.querySelector(`tr.attr-detail-row[data-detail-idx="${idx}"]`);
      if (!vr || !dr) return;
      vr.classList.add("attr-verdict-selected");
      vr.setAttribute("aria-expanded", "true");
      const ch = vr.querySelector(".attr-chevron"); if (ch) ch.textContent = "▾";
      dr.hidden = false;
      _renderAttributionDetailShared(`${hostId}-detail-${idx}`, ctx, visibleRows[idx], kstats, manifest);
    };

    host.querySelectorAll("tr.attr-verdict-row").forEach(tr => tr.addEventListener("click", () => {
      const idx = Number(tr.dataset.rowIdx);
      if (tr.getAttribute("aria-expanded") === "true") { _attrCollapseAll(); return; }
      _attrExpand(idx);
    }));
    host.querySelectorAll("th.attr-verdict-th").forEach(th => th.addEventListener("click", () => {
      const k = th.dataset.sortKey;
      if (host.dataset.sortKey === k) {
        host.dataset.sortAsc = host.dataset.sortAsc === "1" ? "0" : "1";
      } else {
        host.dataset.sortKey = k;
        const col = cols.find(c => c.key === k);
        host.dataset.sortAsc = (col && col.type === "str") ? "1" : "0";
      }
      render(hostId, ctx, manifest);
    }));
    const toggleEl = document.getElementById(showAllId);
    if (toggleEl) {
      toggleEl.addEventListener("change", () => {
        host.dataset.showAll = toggleEl.checked ? "1" : "0";
        render(hostId, ctx, manifest);
      });
    }

    // Auto-open the top visible row.
    if (visibleRows[0]) _attrExpand(0);
  }

  return {render};
})();

// Bulk anchor block renderer (shared, called from the engine).
function _renderBulkAnchor(data, ctx) {
  const num = (v, d=3) => (v == null || !isFinite(v)) ? "—" : Number(v).toFixed(d);
  const bulkNes = data.nes;
  const bulkFdr = data.fdr;
  const bulkSig = bulkFdr != null && isFinite(bulkFdr) && bulkFdr < 0.25;
  const bulkDir = (bulkNes != null && isFinite(bulkNes))
    ? (bulkNes > 0 ? `<span class="attr-bulk-up">↑ NES = +${num(bulkNes, 2)}</span>`
                   : `<span class="attr-bulk-down">↓ NES = ${num(bulkNes, 2)}</span>`)
    : `<span class="attr-bulk-ns">NES n/a</span>`;
  const bulkFdrTxt = (bulkFdr != null && isFinite(bulkFdr))
    ? `FDR = ${num(bulkFdr, 3)}${bulkSig ? "" : " (n.s.)"}` : "FDR n/a";
  const signNote = data.signNote || "sign of the bulk NES is the reference direction every column below is checked against. <strong>Positive NES = kinase more active in disease; negative = more active in WT.</strong>";
  return `<div class="attr-bulk-anchor">Bulk MEA anchor for ${_escapeHtml(data.contrast || ctx.contrast || "")}: ` +
    `<span class="attr-bulk-pill" title="Sign convention: + NES = kinase substrates concentrated among sites with higher stoichiometry (log2 phospho − log2 protein) in disease vs WT.">${bulkDir} · ${bulkFdrTxt}</span> ` +
    `<span class="muted">— ${signNote}</span></div>`;
}
