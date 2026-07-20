
// attribution_manifest_song.js — Song cohort attribution manifest.
// Consumed by AttributionView.render("attr-verdict", ctx, SONG_MANIFEST).
// Declares Song's attribution spine, super-groups, data hooks, and section list.

const SONG_MANIFEST = (() => {

  // ---- Column definitions --------------------------------------------------
  // These mirror the former ATTR_VERDICT_COLS array verbatim. The engine
  // dispatches rendering per column via render(r, numFmt).

  const _num = (v, d=3) => (v == null || !isFinite(v)) ? "" : Number(v).toFixed(d);

  const columns = [
    {
      key: "cell_type", label: "Cell type", type: "str", group: "id", title: "",
      render(r, _numFmt, ctx) {
        const binFlag = r.wmb_binary_expressed === true || String(r.wmb_binary_expressed).toLowerCase() === "true";
        const expBadge = binFlag
          ? ""
          : `<span class="attr-badge attr-badge-warn" title="Mean log2 expression < 1 OR fewer than 10% of cells detect the gene in this cell type. The enrichment score may be elevated because the gene is barely expressed anywhere.">low expr</span>`;
        const _sbk = ctx && ctx.kinase_id != null ? ((PAYLOAD.subclass_breakdown || {})[String(ctx.kinase_id)] || {}) : {};
        const _sbTip = _sbk[r.cell_type] || "";
        const _sbAttr = _sbTip ? ` title="WMB subclass breakdown: ${_escapeHtml(_sbTip)}"` : "";
        return `<td class="attr-celltype"${_sbAttr}><span class="attr-chevron" aria-hidden="true">▸</span> ${_escapeHtml(r.cell_type)}${_sbTip ? ' <span class="attr-subclass-marker" aria-hidden="true">ⓘ</span>' : ''} ${_attrVerdictConfCell(r)} ${_motifPeerCell(r)} ${expBadge}</td>`;
      },
    },
    {
      key: "confidence_tier", label: "Location confidence", type: "conf", group: "attr", hidden: true,
      title: `Kinase-level confidence: how exclusively this kinase is expressed in one cell type (curated specificity units — over-split ${COHORT_LABELS.song} clusters collapsed, distinct cell types kept separate), from within-cohort ${COHORT_LABELS.song} (primary) corroborated by the WMB atlas (at its class level) and human references. Shown on the kinase's dominant cell type only — it is a per-kinase property, not per cell type.`,
      render(r, _numFmt, _ctx) { return `<td>${_attrVerdictConfCell(r)}</td>`; },
    },
    {
      key: "song_detected", label: "Detected", type: "num", group: "attr",
      sub: "song", subLabel: COHORT_LABELS.song,
      subTitle: `Within-cohort ${COHORT_LABELS.song} snRNA. Standard detection metric for this cell type.`,
      title: `${COHORT_LABELS.song}: is the kinase expressed in ≥10% of cells in this cell type? ✓ = detected (% of cells shown), ✗ = not detected. Normalization-free presence gate. Sorts by detection.`,
      render(r, _n, _c) { return `<td class="attr-num">${_detGateCell(r.song_detected, r.song_fraction_cells_expressing, "this cell type")}</td>`; },
    },
    {
      key: "song_concentration_tier", label: "Conc", type: "num", group: "attr",
      sub: "song", subLabel: COHORT_LABELS.song,
      subTitle: `Within-cohort ${COHORT_LABELS.song} snRNA. Standard detection metric for this cell type.`,
      title: `${COHORT_LABELS.song} concentration tier: ≥2×/5×/10× the even 1/N share of total expression across all cell types (comparable across kinases). — = at or below even share. Sorts by tier.`,
      render(r, _n, _c) { return `<td class="attr-num">${_concTierCell(r.song_concentration_tier)}</td>`; },
    },
    {
      key: "wmb_detected", label: "Detected", type: "num", group: "attr",
      sub: "wmb", subLabel: "WMB",
      subTitle: "Whole-mouse-brain atlas cross-check. Standard detection metric for the matched WMB class.",
      title: "WMB cross-check: is the kinase expressed in ≥10% of WMB cells in this class? ✓ = detected (% shown), ✗ = not. Sorts by detection.",
      render(r, _n, _c) { return `<td class="attr-num">${_detGateCell(r.wmb_detected, r.wmb_fraction_cells_expressing, "this WMB class")}</td>`; },
    },
    {
      key: "wmb_concentration_tier", label: "Conc", type: "num", group: "attr",
      sub: "wmb", subLabel: "WMB",
      subTitle: "Whole-mouse-brain atlas cross-check. Standard detection metric for the matched WMB class.",
      title: "WMB concentration tier: ≥2×/5×/10× the even 1/N share of total expression across all WMB classes (comparable across kinases). — = at or below even share. Sorts by tier.",
      render(r, _n, _c) { return `<td class="attr-num">${_concTierCell(r.wmb_concentration_tier)}</td>`; },
    },
    {
      key: "sea_ad_lfc", label: "SEA-AD LFC", type: "num", group: "attr",
      title: "SEA-AD log2 fold change in human AD vs control, median across SEA-AD supertypes mapped to this subclass. Stratum (early / late / full CPS) is selected from the contrast pathway. Color: red = up in AD, blue = down.",
      render(r, _n, _c) {
        return r.sea_ad_lfc == null || !isFinite(r.sea_ad_lfc)
          ? `<td class="attr-num attr-empty">—</td>`
          : `<td class="attr-num attr-num-lfc" style="background:${_attrLfcColor(r.sea_ad_lfc)}">${_num(r.sea_ad_lfc, 3)}</td>`;
      },
    },
    {
      key: "song_lfc", label: `${COHORT_LABELS.song} LFC`, type: "num", group: "attr",
      title: `${COHORT_LABELS.song} log2 fold change from within-cohort snRNA-seq factorial OLS (β at this contrast — 10-param design, time-resolved). Color: red = up in disease genotype, blue = down.`,
      render(r, _n, _c) {
        return r.song_lfc == null || !isFinite(r.song_lfc)
          ? `<td class="attr-num attr-empty">—</td>`
          : `<td class="attr-num attr-num-lfc" style="background:${_attrLfcColor(r.song_lfc)}">${_num(r.song_lfc, 3)}</td>`;
      },
    },
    {
      key: "decomp_nes", label: "Decomp NES", type: "num", group: "decomp",
      title: `Decomposition NES from the CTM-native proportional decomposition (per-cell-type kinase MEA on bulk phospho ranking weighted by snRNA share for the kinase's substrate set). Same join key as ${COHORT_LABELS.song} LFC. Hypothesis-strength signal — see Methods.`,
      render(r, _n, _c) {
        return r.decomp_nes == null || !isFinite(r.decomp_nes)
          ? `<td class="attr-num attr-empty">—</td>`
          : `<td class="attr-num attr-num-lfc" style="background:${_attrLfcColor(r.decomp_nes)}">${_num(r.decomp_nes, 2)}</td>`;
      },
    },
    {
      key: "decomp_fdr", label: "Decomp FDR", type: "num", group: "decomp",
      title: "Decomposition MEA FDR for this (kinase, contrast, cell type) row. < 0.25 is the standard MEA gate.",
      render(r, _n, _c) {
        const sig = r.decomp_fdr != null && isFinite(r.decomp_fdr) && r.decomp_fdr < 0.25;
        return r.decomp_fdr == null || !isFinite(r.decomp_fdr)
          ? `<td class="attr-num attr-empty">—</td>`
          : `<td class="attr-num"${sig ? ' style="font-weight:600"' : ''}>${_num(r.decomp_fdr, 3)}</td>`;
      },
    },
    {
      key: "bulk_match", label: "vs Bulk", type: "num", group: "decomp",
      title: "Sign agreement between Decomp NES and the bulk MEA NES for this kinase × contrast. Bold ✓/✗ when Decomp FDR < 0.25; muted when not. Hover any cell for the underlying values.",
      render(r, _numFmt, _ctx) {
        // _ctx is not passed here; bulkNes/bulkFdr are pre-attached by attachDerived.
        if (r.bulk_match == null) return `<td class="attr-num attr-empty">—</td>`;
        const agree = r.bulk_match > 0;
        const sig = Math.abs(r.bulk_match) === 2;
        const glyph = agree ? "✓" : "✗";
        const color = agree ? "#15803d" : "#b91c1c";
        const style = sig
          ? `color:${color};font-weight:700`
          : `color:#94a3b8;font-weight:500`;
        const tip = _escapeHtml(r._bulkMatchTip || "");
        return `<td class="attr-num" style="${style};text-align:center" title="${tip}">${glyph}</td>`;
      },
    },
  ];

  // ---- Super-group header --------------------------------------------------

  function superGroupRow(cols) {
    const _grpCounts = cols.reduce((acc, c) => { acc[c.group] = (acc[c.group]||0)+1; return acc; }, {});
    return `<tr class="attr-verdict-supergroup">` +
      `<th class="attr-supergroup-spacer" colspan="${_grpCounts.id || 0}"></th>` +
      `<th class="attr-supergroup-attr" colspan="${_grpCounts.attr || 0}" title="Cell-type attribution evidence. Each component is compared against the bulk MEA direction at this contrast.">Attribution (vs bulk direction)</th>` +
      `<th class="attr-supergroup-decomp" colspan="${_grpCounts.decomp || 0}" title="Per-cell-type pseudo-deconvolution MEA. A second look at the bulk phospho ranking re-projected by snRNA share.">Decomposition cross-check</th>` +
    `</tr>`;
  }

  // ---- Data hooks ----------------------------------------------------------

  function getRows(ctx) {
    const verdictFilter = {
      disease:   (ctx.contrast || "").split("_")[0] || "",
      timepoint: ((ctx.contrast || "").match(/_(\d+mo)$/) || ["",""])[1] || "",
      celltype: "", confidence: "",
    };
    return ctx.kinase_id != null
      ? getScopedAttribution(ctx.kinase_id, verdictFilter)
      : [];
  }

  function dedupKey(r) { return `${r.contrast_id}|${r.cell_type}`; }

  function dedupCmp(a, b) { return _cmpCanonicalAttribution(a, b); }

  // Attach decomp NES/FDR and bulk_match to each row in-place.
  function attachDerived(rows, ctx) {
    const _K = ViewerPayload.kinases();
    const _bulkNes = (_K && _K["NES_" + ctx.contrast]) ? _K["NES_" + ctx.contrast][ctx.kinase_id] : null;
    const _bulkFdr = (_K && _K["FDR_" + ctx.contrast]) ? _K["FDR_" + ctx.contrast][ctx.kinase_id] : null;
    const numFmt = (v, d=3) => (v == null || !isFinite(v)) ? "—" : Number(v).toFixed(d);
    for (const r of rows) {
      const dk = `${ctx.kinase_id}|${r.contrast_id}|${r.cell_type}`;
      const d = _decompByKey ? _decompByKey.get(dk) : null;
      r.decomp_nes = d ? d.nes : null;
      r.decomp_fdr = d ? d.fdr : null;
      if (r.decomp_nes == null || !isFinite(r.decomp_nes) || r.decomp_nes === 0
          || _bulkNes == null || !isFinite(_bulkNes) || _bulkNes === 0) {
        r.bulk_match = null;
        r._bulkMatchTip = "";
      } else {
        const agree = (r.decomp_nes > 0) === (_bulkNes > 0);
        const sig = r.decomp_fdr != null && isFinite(r.decomp_fdr) && r.decomp_fdr < 0.25;
        r.bulk_match = agree ? (sig ? 2 : 1) : (sig ? -2 : -1);
        r._bulkMatchTip =
          `Bulk NES = ${numFmt(_bulkNes, 2)}` +
          (_bulkFdr != null && isFinite(_bulkFdr) ? ` (FDR ${numFmt(_bulkFdr, 3)})` : "") +
          ` · Decomp NES = ${numFmt(r.decomp_nes, 2)}` +
          (r.decomp_fdr != null && isFinite(r.decomp_fdr) ? ` (FDR ${numFmt(r.decomp_fdr, 3)})` : "") +
          (sig ? "" : " · Decomp not significant (FDR ≥ 0.25)");
      }
    }
    // Stash bulk values for bulkAnchor.
    attachDerived._lastBulkNes = _bulkNes;
    attachDerived._lastBulkFdr = _bulkFdr;
  }

  const defaultSort = {key: "confidence_tier", asc: false};

  function sortTiebreak(a, b) {
    return (_songSpecificityRank(b) - _songSpecificityRank(a)) ||
           ((b.decomp_agrees_bulk ? 1 : 0) - (a.decomp_agrees_bulk ? 1 : 0));
  }

  // Show all cell types. Detection is an evidence column, not a row gate.
  const rowVisible = null;

  function bulkAnchor(ctx) {
    const _K = ViewerPayload.kinases();
    const nes = (_K && _K["NES_" + ctx.contrast]) ? _K["NES_" + ctx.contrast][ctx.kinase_id] : null;
    const fdr = (_K && _K["FDR_" + ctx.contrast]) ? _K["FDR_" + ctx.contrast][ctx.kinase_id] : null;
    return {contrast: ctx.contrast, nes, fdr};
  }

  // Exclusivity summary line above the table.
  function exclSummary(rows, _ctx) {
    const _kt = rows[0] || {};
    const _exclUnit = _kt.specificity_unit_label || _kt.specificity_celltype || "";
    let _collapseNote = "";
    if (_kt.specificity_collapsed) {
      const SU = (PAYLOAD.specificity_units && PAYLOAD.specificity_units.units) || {};
      const kids = (SU[_kt.specificity_unit] && SU[_kt.specificity_unit].children) || [];
      _collapseNote = ` <span class="muted">(collapsed from ${kids.length} ${COHORT_LABELS.song} clusters — `
        + `top: ${_escapeHtml(_kt.specificity_celltype || "")}; see rows below)</span>`;
    }
    return `<div class="attr-excl-summary">Cell-type exclusivity (kinase-level): ` +
      `<span class="${_attrConfidenceClass(_kt.confidence_tier || 'none')}" title="${_escapeHtml(_kt.confidence_basis || '')}">${_escapeHtml((_kt.confidence_tier || 'none').replace('_', ' '))}</span>` +
      (_exclUnit ? ` · <strong>${_escapeHtml(_exclUnit)}</strong>` : "") +
      _collapseNote +
      `</div>`;
  }

  // kstats for the detail renderer (per-kinase detection summary).
  function buildKstats(rows, _ctx) {
    const _SU = (PAYLOAD.specificity_units && PAYLOAD.specificity_units.units) || {};
    const _C2U = (PAYLOAD.specificity_units && PAYLOAD.specificity_units.cluster_to_unit) || {};
    const _detRows = rows.filter(r => r.song_detected === true);
    return {
      nDetected: _detRows.length,
      nUnitsDetected: new Set(_detRows.map(r => _C2U[r.cell_type]).filter(Boolean)).size,
      nClustersTotal: rows.length,
      nUnitsTotal: Object.keys(_SU).length,
      byCell: new Map(rows.map(r => [r.cell_type, r])),
    };
  }

  // No show-all toggle: all Levy-t5 clusters are visible by default.
  const showAllLabel = "";

  // Crosswalk: Song cohort uses the reference-mapping line.
  const renderCrosswalk = true;

  // Explainer collapsible.
  const explainerHtml = `<details class="attr-explainer"><summary>How to read attribution confidence</summary>` +
    `<div class="attr-explainer-body">` +
    `<p>Confidence is a <strong>kinase-level</strong> label: how <em>exclusively</em> the kinase is expressed in a single cell type, scored over <strong>curated specificity units</strong>. ${COHORT_LABELS.song} over-splits some cell types into many clusters (e.g. excitatory neurons → 6 pyramidal subtypes); those are <strong>collapsed</strong> into one unit (shown as an expandable parent) so a pan-class kinase is not penalized. Cell types that are genuinely distinct (e.g. endothelial vs. pericyte) stay separate. It prioritizes the <strong>within-cohort ${COHORT_LABELS.song} snRNA</strong> and treats the reference atlases as corroboration:</p>` +
    `<table class="attr-explainer-table" style="margin-bottom:8px;">` +
      `<thead><tr><th>Source</th><th>Role</th></tr></thead><tbody>` +
      `<tr><td><strong>${COHORT_LABELS.song}</strong></td><td><strong>Primary.</strong> Within-cohort snRNA from the same animals. Sets the tier from how concentrated the kinase's expression is in one unit (effective number of cell types = 1 / Σ&nbsp;share² over the units).</td></tr>` +
      `<tr><td><strong>WMB</strong></td><td>Corroborates: does the healthy mouse atlas place the kinase in the same cell class? Promotes the tier, never required.</td></tr>` +
      `<tr><td><strong>SEA-AD / HBCA</strong></td><td>Corroborates: does the human brain reference (when its location signal is strong) point to the same class? Promotes the tier, never vetoes.</td></tr>` +
      `</tbody></table>` +
    `<ul>` +
      `<li><strong><span class="attr-conf attr-conf-very-high">very high</span></strong> — ${COHORT_LABELS.song} localizes the kinase to essentially one cell type (≲1.5 effective units) <em>and</em> a reference (WMB or human) agrees on that cell class.</li>` +
      `<li><strong><span class="badge hi">high</span></strong> — ${COHORT_LABELS.song} concentrates the kinase in one cell type and a reference agrees; or ${COHORT_LABELS.song} places it very tightly (≲1.5) on its own, not yet corroborated.</li>` +
      `<li><strong><span class="badge mid">moderate</span></strong> — ${COHORT_LABELS.song} concentrates the kinase in one cell type, but no reference corroborates that cell class.</li>` +
      `<li><strong><span class="badge lo">low</span></strong> — expressed broadly across cell types (not cell-type-specific).</li>` +
      `<li><strong>none</strong> — no measurable within-cohort ${COHORT_LABELS.song} expression distribution.</li>` +
    `</ul>` +
    `<p class="muted">Confidence is per kinase, so in the table above the pill is shown on the kinase's <em>dominant</em> cell type. A collapsed unit lists the ${COHORT_LABELS.song} clusters it covers (its child rows appear below). The prior disease-direction concordance is preserved in each pill's tooltip.</p>` +
    `</div></details>`;

  // ---- Detail sections in Song's exact order --------------------------------
  // §crosswalk handled by renderCrosswalk=true above.
  // §0 specificity verdict → §1 WMB dot plot → §2 SEA-AD heatmap + Song OLS → §3 decomp OLS.

  const sections = [
    {
      id: "specificity",
      title: null,  // rendered by _renderSpecificityVerdict internally
      caption: null,
      wide: true,
      render(secHostId, ctx, row, kstats) {
        const host = document.getElementById(secHostId);
        if (!host) return;
        host.innerHTML = _renderSpecificityVerdict(row, ctx, kstats);
      },
    },
    {
      id: "wmb",
      title: `§1 · Expression — WMB reference <span class="muted">(wmb_kinase_expression.csv)</span>`,
      caption: null,  // caption rendered inline below
      wide: false,
      render(secHostId, ctx, row, _kstats) {
        const host = document.getElementById(secHostId);
        if (!host) return;
        const gene = ctx.gene || "";
        const cellType = row.cell_type || "";
        host.innerHTML =
          `<p class="muted attr-caption">Seurat-style dot plot for ${_escapeHtml(gene)} across Allen Whole Mouse Brain classes. Color = mean log2 expression, dot size = fraction of cells expressing. The clicked cell type's WMB class is outlined.</p>` +
          `<div id="${secHostId}-plot"></div>` +
          _humanLocationLine(row);
        _renderWMBDotPlot(`${secHostId}-plot`, ctx, cellType);
      },
    },
    {
      id: "seaad",
      title: `§2 · Disease direction — SEA-AD <span class="muted">(sea_ad_supertype_lfc.csv)</span>`,
      caption: null,  // caption rendered inline below
      wide: false,
      render(secHostId, ctx, row, _kstats) {
        const host = document.getElementById(secHostId);
        if (!host) return;
        const gene = ctx.gene || "";
        const cellType = row.cell_type || "";
        host.innerHTML =
          `<p class="muted attr-caption">Per-supertype human AD-vs-control LFC for ${_escapeHtml(gene)}, grouped by subclass; stratum (early / late / full CPS) follows the contrast. The subclass matching this cell type is outlined. Direction-tier evidence — does its activity move with disease — not specificity.</p>` +
          `<div id="${secHostId}-hm"></div>`;
        _renderSEAADHeatmap(`${secHostId}-hm`, ctx, cellType);
      },
    },
    {
      id: "song-ols",
      title: `§2 · Disease direction — ${COHORT_LABELS.song} OLS <span class="muted">(song_concordance)</span>`,
      caption: null,
      wide: false,
      render(secHostId, ctx, row, _kstats) {
        const host = document.getElementById(secHostId);
        if (!host) return;
        const gene = ctx.gene || "";
        const cellType = row.cell_type || "";
        host.innerHTML =
          `<p class="muted attr-caption">Within-cohort factorial OLS coefficient (β) on the per-animal pseudobulk for this cell type, across contrasts. The selected contrast is highlighted.</p>` +
          `<div id="${secHostId}-tbl"></div>`;
        const loader = SliceCache && typeof SliceCache.loadSongConcordance === "function"
          ? (g) => SliceCache.loadSongConcordance(g) : null;
        _renderWithinCohortOLSPanel(`${secHostId}-tbl`, ctx, cellType, loader);
      },
    },
    {
      id: "decomp-ols",
      title: `§3 · Mechanism — per-cell substrate-site OLS <span class="muted">(site_level_ols.parquet)</span>`,
      caption: null,
      wide: true,
      render(secHostId, ctx, row, _kstats) {
        const host = document.getElementById(secHostId);
        if (!host) return;
        const cellType = row.cell_type || "";
        const kinaseName = ctx.name || "";
        host.innerHTML =
          `<p class="muted attr-caption">Per-(site, contrast, cell type) β / SE / p from the Levy-t5 pseudo-decomposition OLS, restricted to ${_escapeHtml(kinaseName)}'s substrate set in ${_escapeHtml(cellType)}. Shows what drives the Decomp NES. Bulk β is the same site's stoichiometry β before share-reweighting; |Δβ| measures how much the per-cell estimate diverges from bulk.</p>` +
          `<div id="${secHostId}-tbl" class="audit-scroll"></div>`;
        _renderDecompOlsTable(`${secHostId}-tbl`, ctx, cellType);
      },
    },
  ];

  return {
    // Song detail layout: reproduces the original attr-drawer-grid structure.
    // Sections are laid out in the exact order and nesting of the old
    // _renderAttributionDetail: §0 specificity (wide, no outer wrapper),
    // then a grid div wrapping §1 WMB + §2 SEA-AD + §2 Song OLS,
    // then §3 decomp outside the grid (wide).
    // sectionStubs = [{sec, secHostId}] in sections array order.
    renderDetailLayout(sectionStubs, _ctx, _row, _kstats) {
      // Map by section id for lookup.
      const byId = {};
      for (const {sec, secHostId} of sectionStubs) byId[sec.id] = secHostId;

      const specHostId = byId["specificity"];
      const wmbHostId = byId["wmb"];
      const seaadHostId = byId["seaad"];
      const songOlsHostId = byId["song-ols"];
      const decompHostId = byId["decomp-ols"];

      return (
        // §0 specificity — renders its own <section> from _renderSpecificityVerdict
        `<div id="${specHostId}"></div>` +
        // §1/§2 grid
        `<div class="attr-drawer-grid">` +
          `<section class="attr-section"><h5>${sections.find(s => s.id === "wmb").title}</h5>` +
            `<div id="${wmbHostId}"></div></section>` +
          `<section class="attr-section"><h5>${sections.find(s => s.id === "seaad").title}</h5>` +
            `<div id="${seaadHostId}"></div></section>` +
          `<section class="attr-section"><h5>${sections.find(s => s.id === "song-ols").title}</h5>` +
            `<div id="${songOlsHostId}"></div></section>` +
        `</div>` +
        // §3 decomp — wide, outside grid
        `<section class="attr-section attr-section-wide"><h5>${sections.find(s => s.id === "decomp-ols").title}</h5>` +
          `<div id="${decompHostId}" class="audit-scroll"></div></section>`
      );
    },

    columns,
    superGroupRow,
    getRows,
    dedupKey,
    dedupCmp,
    attachDerived,
    defaultSort,
    sortTiebreak,
    rowVisible,
    bulkAnchor,
    exclSummary,
    buildKstats,
    showAllLabel,
    renderCrosswalk,
    explainerHtml,
    sections,
  };
})();
