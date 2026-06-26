
// attribution_manifest_fivexfad.js — 5xFAD cohort attribution manifest.
// Consumed by AttributionView.render("f5-audit-body", ctx, FIVEXFAD_MANIFEST).
// ctx = { group, age, gene } where group is the 5xFAD kinase group object.
//
// Age/tissue scoping: expressed here in getRows() — no forked render code.
// Decomp-only fallback: when no native 5xFAD snRNA attribution rows exist for
// the kinase, cell-type decomp MEA rows are promoted as skeleton attribution
// rows (confidence_tier = "none", _decomp_only = true).

const FIVEXFAD_MANIFEST = (() => {

  // ---- Helpers ---------------------------------------------------------------

  const _f5num = (v) => {
    if (v == null) return null;
    if (typeof v === "string" && v.trim() === "") return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  };

  const _num = (v, d=3) => (v == null || !isFinite(v)) ? "" : Number(v).toFixed(d);

  // ---- Column definitions ---------------------------------------------------
  // Attribution spine: cell_type carries the location-confidence pill;
  // confidence_tier remains an internal sort key. Visible evidence columns:
  // fivexfad_detected, fivexfad_concentration_tier, fivexfad_lfc,
  // wmb_detected, wmb_concentration_tier, sea_ad_lfc,
  // decomp_nes, decomp_fdr, decomp_agrees_bulk.

  const columns = [
    {
      key: "cell_type", label: "Cell type", type: "str", group: "id",
      title: `${COHORT_LABELS.fivexfad} new_clusters cell type on the shared 46-cluster mouse spine.`,
      render(r, _numFmt, _ctx) {
        const conf = r.confidence_tier || "none";
        const tip = _escapeHtml(r.confidence_basis || conf);
        const isHome = r.fivexfad_top_celltype && r.cell_type === r.fivexfad_top_celltype;
        const chip = isHome
          ? `<span class="${_attrConfidenceClass(conf)}" title="${tip}">${_escapeHtml(conf.replace("_", " "))}</span>`
          : `<span class="muted" title="Location confidence is a kinase/tissue-level property, shown on this kinase's dominant ${COHORT_LABELS.fivexfad} cell type${r.fivexfad_top_celltype ? ': ' + _escapeHtml(r.fivexfad_top_celltype) : ''}.">·</span>`;
        return `<td class="attr-celltype"><span class="attr-chevron" aria-hidden="true">▸</span> ${_escapeHtml(r.cell_type || "")} ${chip}</td>`;
      },
    },
    {
      key: "confidence_tier", label: "Location confidence", type: "conf", group: "attr", hidden: true,
      title: `${COHORT_LABELS.fivexfad} snRNA cell-type exclusivity confidence tier. Hover the chip for the evidence basis. Shown for all rows (${COHORT_LABELS.fivexfad} has no collapsed-unit hierarchy).`,
      render(r, _numFmt, _ctx) {
        const conf = r.confidence_tier || "none";
        const tip = _escapeHtml(r.confidence_basis || conf);
        return `<td><span class="${_attrConfidenceClass(conf)}" title="${tip}">${_escapeHtml(conf.replace("_", " "))}</span></td>`;
      },
    },
    {
      key: "fivexfad_detected", label: "Detected", type: "num", group: "attr",
      sub: "f5", subLabel: COHORT_LABELS.fivexfad,
      subTitle: `Within-cohort ${COHORT_LABELS.fivexfad} snRNA (new_clusters). Standard detection metric for this cell type.`,
      title: `${COHORT_LABELS.fivexfad} snRNA: is the kinase transcript detected in ≥10% of cells in this cell type? ✓ = detected (% of cells shown), ✗ = not detected.`,
      render(r, _n, _c) {
        return `<td class="attr-num">${_detGateCell(r.fivexfad_detected, r.fivexfad_fraction_cells_expressing, `this ${COHORT_LABELS.fivexfad} cell type`)}</td>`;
      },
    },
    {
      key: "fivexfad_concentration_tier", label: "Conc", type: "num", group: "attr",
      sub: "f5", subLabel: COHORT_LABELS.fivexfad,
      subTitle: `Within-cohort ${COHORT_LABELS.fivexfad} snRNA (new_clusters). Standard detection metric for this cell type.`,
      title: `${COHORT_LABELS.fivexfad} concentration tier: ≥2×/5×/10× the even 1/N share of total expression across all cell types. — = at or below even share.`,
      render(r, _n, _c) { return `<td class="attr-num">${_concTierCell(r.fivexfad_concentration_tier)}</td>`; },
    },
    {
      key: "fivexfad_lfc", label: "snRNA LFC", type: "num", group: "attr",
      title: `${COHORT_LABELS.fivexfad} snRNA TG-vs-WT log-expression difference for the selected tissue and age. Color: red = up in TG, blue = down. Info-only — not the confidence gate.`,
      render(r, _n, _c) {
        const v = _f5num(r.fivexfad_lfc);
        return v == null
          ? `<td class="attr-num attr-empty">—</td>`
          : `<td class="attr-num attr-num-lfc" style="background:${_attrLfcColor(v)}">${_num(v, 3)}</td>`;
      },
    },
    {
      key: "wmb_detected", label: "Detected", type: "num", group: "attr",
      sub: "wmb", subLabel: "WMB",
      subTitle: "Whole-mouse-brain atlas cross-check. Standard detection metric for the matched WMB class.",
      title: "WMB cross-check: is the kinase detected in ≥10% of WMB cells in this class? ✓ = detected (% shown), ✗ = not.",
      render(r, _n, _c) {
        return `<td class="attr-num">${_detGateCell(r.wmb_detected, r.wmb_fraction_cells_expressing, "this WMB class")}</td>`;
      },
    },
    {
      key: "wmb_concentration_tier", label: "Conc", type: "num", group: "attr",
      sub: "wmb", subLabel: "WMB",
      subTitle: "Whole-mouse-brain atlas cross-check. Standard detection metric for the matched WMB class.",
      title: "WMB concentration tier: ≥2×/5×/10× the even 1/N share of total expression across all WMB classes. — = at or below even share.",
      render(r, _n, _c) { return `<td class="attr-num">${_concTierCell(r.wmb_concentration_tier)}</td>`; },
    },
    {
      key: "sea_ad_lfc", label: "SEA-AD LFC", type: "num", group: "attr",
      title: "Human SEA-AD AD-vs-control log2 fold change mapped to this cell type where available. Color: red = up in AD, blue = down.",
      render(r, _n, _c) {
        const v = _f5num(r.sea_ad_lfc);
        return v == null
          ? `<td class="attr-num attr-empty">—</td>`
          : `<td class="attr-num attr-num-lfc" style="background:${_attrLfcColor(v)}">${_num(v, 3)}</td>`;
      },
    },
    {
      key: "decomp_nes", label: "Decomp NES", type: "num", group: "decomp",
      title: `${COHORT_LABELS.fivexfad} per-cell-type decomposition MEA NES for this kinase, tissue, age, and cell type.`,
      render(r, _n, _c) {
        const v = _f5num(r.decomp_nes);
        return v == null
          ? `<td class="attr-num attr-empty">—</td>`
          : `<td class="attr-num attr-num-lfc" style="background:${_attrLfcColor(v)}">${_num(v, 2)}</td>`;
      },
    },
    {
      key: "decomp_fdr", label: "Decomp FDR", type: "num", group: "decomp",
      title: `${COHORT_LABELS.fivexfad} per-cell-type decomposition MEA FDR for this kinase, tissue, age, and cell type. < 0.25 is the standard MEA gate.`,
      render(r, _n, _c) {
        const v = _f5num(r.decomp_fdr);
        const sig = v != null && v < 0.25;
        return v == null
          ? `<td class="attr-num attr-empty">—</td>`
          : `<td class="attr-num"${sig ? ' style="font-weight:600"' : ""}>${_num(v, 3)}</td>`;
      },
    },
    {
      key: "decomp_agrees_bulk", label: "Bulk match", type: "str", group: "decomp",
      title: `Whether significant per-cell decomposition MEA has the same NES sign as the bulk ${COHORT_LABELS.fivexfad} anchor. Bold = significant decomp FDR < 0.25.`,
      render(r, _n, _c) {
        if (r.decomp_agrees_bulk === "yes") {
          return `<td><span class="badge hi" title="Decomp FDR passes the gate and NES sign matches bulk.">match</span></td>`;
        }
        if (r.decomp_agrees_bulk === "opposes") {
          return `<td><span class="badge lo" title="Decomp FDR passes the gate but NES sign opposes bulk.">oppose</span></td>`;
        }
        return `<td class="attr-empty">—</td>`;
      },
    },
  ];

  // ---- Super-group header ---------------------------------------------------

  function superGroupRow(cols) {
    const _grpCounts = cols.reduce((acc, c) => { acc[c.group] = (acc[c.group]||0)+1; return acc; }, {});
    return `<tr class="attr-verdict-supergroup">` +
      `<th class="attr-supergroup-spacer" colspan="${_grpCounts.id || 0}"></th>` +
      `<th class="attr-supergroup-attr" colspan="${_grpCounts.attr || 0}" title="Cell-type attribution evidence from ${COHORT_LABELS.fivexfad} snRNA, WMB, and SEA-AD reference layers.">Attribution (vs bulk direction)</th>` +
      `<th class="attr-supergroup-decomp" colspan="${_grpCounts.decomp || 0}" title="Per-cell-type ${COHORT_LABELS.fivexfad} decomposition MEA using matched snRNA new_clusters weights.">Decomposition cross-check</th>` +
    `</tr>`;
  }

  // ---- Data hooks -----------------------------------------------------------

  // ctx = { group, age, gene }
  // Age/tissue scoping is done here, not in render code.
  // Decomp-only fallback: when no native 5xFAD snRNA attribution rows exist,
  // promote cell-type decomp MEA rows as skeleton rows.
  function getRows(ctx) {
    const group = ctx && ctx.group;
    if (!group) return [];
    const age = ctx.age;

    // Native 5xFAD snRNA attribution rows, scoped to this tissue and age.
    const attrRows = _f5AttrRowsForGroup(group);

    // Decomp rows for this group (cell-type MEA, already loaded).
    const decompRows = _f5CelltypeMeaRowsForGroup(group);

    // Rows to iterate: native if available, else decomp-only fallbacks.
    const sourceRows = attrRows.length
      ? attrRows
      : decompRows.map(d => ({
          cell_type: d.cell_type,
          confidence_tier: "none",
          confidence_basis: `No native ${COHORT_LABELS.fivexfad} snRNA attribution row available for this kinase; showing decomposition MEA only.`,
          tissue: group.tissue,
          age_months: age,
          kinase: group.kinase,
          gene_symbol: group.gene_symbol || group.kinase,
          _decomp_only: true,
        }));

    // Join with decomp MEA data for each row.
    const meaRow = _f5SelectedRow(group);
    const bulkNes = _f5num(meaRow && meaRow.NES);
    const fdrGate = Store.state.filters.fdr;

    return sourceRows.map(r => {
      const d = _f5CelltypeMeaFor(group, r.cell_type);
      const dNes = _f5num(d && d.NES);
      const dFdr = _f5num(d && d.FDR);
      const sig = dFdr != null && dFdr < fdrGate;
      const agrees = sig && dNes != null && bulkNes != null && dNes !== 0 && bulkNes !== 0
        ? ((dNes > 0) === (bulkNes > 0))
        : false;
      return Object.assign({}, r, {
        decomp_nes: dNes,
        decomp_fdr: dFdr,
        decomp_agrees_bulk: agrees ? "yes" : (sig ? "opposes" : ""),
      });
    });
  }

  function dedupKey(r) {
    // Rows are already scoped to one tissue+age by getRows; dedup on cell_type.
    return String(r.cell_type || "");
  }

  function dedupCmp(a, b) {
    // Prefer higher confidence tier; then higher 5xFAD concentration.
    const cr = (_CONF_RANK[b.confidence_tier] || 0) - (_CONF_RANK[a.confidence_tier] || 0);
    if (cr) return cr;
    return (_f5num(b.fivexfad_concentration) || -1) - (_f5num(a.fivexfad_concentration) || -1);
  }

  const defaultSort = {key: "confidence_tier", asc: false};

  function sortTiebreak(a, b) {
    // Tiebreak under confidence_tier: higher 5xFAD concentration, then WMB tier.
    return ((_f5num(b.fivexfad_concentration) || -Infinity) - (_f5num(a.fivexfad_concentration) || -Infinity)) ||
           ((_f5num(b.wmb_concentration_tier) || -1) - (_f5num(a.wmb_concentration_tier) || -1));
  }

  // Hide low/none confidence rows unless "Show all" is toggled.
  function rowVisible(r) {
    return ["very_high", "high", "moderate"].includes(r.confidence_tier || "");
  }

  function bulkAnchor(ctx) {
    const group = ctx && ctx.group;
    const age = ctx && ctx.age;
    const row = group && age != null ? (group.rows.get(age) || null) : null;
    const nes = _f5num(row && row.NES);
    const fdr = _f5num(row && row.FDR);
    return {
      contrast: `TG_vs_WT_${age}mo`,
      nes,
      fdr,
      signNote: `sign of the bulk NES is the reference direction for decomposition agreement. <strong>Positive NES = kinase more active in ${COHORT_LABELS.fivexfad} TG; negative = more active in WT.</strong>`,
    };
  }

  const showAllLabel = "Show all cell types";

  const renderCrosswalk = false;

  const explainerHtml = `<details class="attr-explainer"><summary>How to read attribution confidence</summary>` +
    `<div class="attr-explainer-body">` +
    `<p>Confidence is a <strong>per-row</strong> label: how exclusively the kinase is expressed in this cell type (effective number of cell types computed from ${COHORT_LABELS.fivexfad} <code>new_clusters</code> within the selected tissue), corroborated by WMB and SEA-AD references.</p>` +
    `<table class="attr-explainer-table" style="margin-bottom:8px;">` +
      `<thead><tr><th>Source</th><th>Role</th></tr></thead><tbody>` +
      `<tr><td><strong>${COHORT_LABELS.fivexfad} snRNA</strong></td><td><strong>Primary.</strong> Tissue-specific matched ${COHORT_LABELS.fivexfad} location evidence using <code>new_clusters</code> on the shared 46-label mouse spine.</td></tr>` +
      `<tr><td><strong>WMB</strong></td><td>Corroborates: does the healthy mouse atlas place the kinase in the same cell class?</td></tr>` +
      `<tr><td><strong>SEA-AD</strong></td><td>Human Alzheimer's disease direction where a mapped cell-type effect is available. Info-only — not the confidence gate.</td></tr>` +
      `<tr><td><strong>Decomp NES / FDR</strong></td><td>Per-cell-type kinase MEA cross-check after projecting raw phosphosite signal with matched snRNA weights.</td></tr>` +
      `</tbody></table>` +
    `<ul>` +
      `<li><strong><span class="attr-conf attr-conf-very-high">very high</span></strong> — kinase localizes to essentially one cell type (≲1.5 effective <code>new_clusters</code>) and WMB/SEA-AD agrees.</li>` +
      `<li><strong><span class="badge hi">high</span></strong> — concentrated in one cell type with reference corroboration; or very tight on its own (≲1.5), not yet corroborated.</li>` +
      `<li><strong><span class="badge mid">moderate</span></strong> — concentrated in one cell type (eff ≤ 3), but not corroborated by a reference.</li>` +
      `<li><strong><span class="badge lo">low</span></strong> — broadly expressed (eff > 3).</li>` +
      `<li><strong>none</strong> — no measurable ${COHORT_LABELS.fivexfad} snRNA expression distribution for this tissue and age.</li>` +
    `</ul>` +
    `<p class="muted">Disease direction (snRNA LFC) is an info-only axis — it is never the confidence gate. A significant bulk MEA does not gate the pill; those gate values appear as info fields on each row.</p>` +
    `</div></details>`;

  // ---- §0 local specificity verdict (5xFAD-specific fields) -----------------
  // The shared _renderSpecificityVerdict reads Song-specific fields
  // (song_unit_effective_n, song_detected, specificity_unit_label, etc.).
  // 5xFAD uses fivexfad_effective_n, fivexfad_detected, fivexfad_concentration_tier.

  function _f5RenderSpecificityVerdict(secHostId, ctx, row) {
    const host = document.getElementById(secHostId);
    if (!host) return;

    const conf = row.confidence_tier || "none";
    const eff = _f5num(row.fivexfad_unit_effective_n);   // pill driver (curated units, Song convention)
    const rawEff = _f5num(row.fivexfad_effective_n);      // native new_clusters breadth (subtype spread, NOT the tier input)
    const detected = row.fivexfad_detected === true || row.fivexfad_detected === "true" || row.fivexfad_detected === "True";
    const frac = _f5num(row.fivexfad_fraction_cells_expressing);
    const tier = Number(row.fivexfad_concentration_tier) || 0;
    const concOfTotal = _f5num(row.fivexfad_concentration_of_total);
    const wmbDetected = row.wmb_detected === true || row.wmb_detected === "true" || row.wmb_detected === "True";
    const wmbFrac = _f5num(row.wmb_fraction_cells_expressing);
    const wmbTier = Number(row.wmb_concentration_tier) || 0;
    const seaLfc = _f5num(row.sea_ad_lfc);
    const snrnaSamples = `${_specF2(row.n_snrna_samples_wt)} WT / ${_specF2(row.n_snrna_samples_tg)} TG`;
    const snrnaCells = `${_specF2(row.n_cells_wt)} WT / ${_specF2(row.n_cells_tg)} TG`;
    const f5Lfc = _f5num(row.fivexfad_lfc);
    const band = _specEffBand(eff);

    const tierBadge = tier > 0
      ? `<span class="badge ${tier>=10?'vhi':tier>=5?'hi':tier>=2?'mid':'lo'}">≥${tier}×</span>`
      : `<span class="muted">— (at or below the all-cell-types even share)</span>`;

    const thisCellDlHtml =
      `<dt>${COHORT_LABELS.fivexfad} cells expressing</dt><dd>${_specPct(frac)}${detected ? "" : ' <span class="muted">(below 10%; specificity denominator still uses all units)</span>'}</dd>` +
      `<dt>Share of total expr</dt><dd>${_specPct(concOfTotal)}</dd>` +
      `<dt>Concentration</dt><dd>${tierBadge}</dd>` +
      `<dt>snRNA samples</dt><dd>${_escapeHtml(snrnaSamples)}</dd>` +
      `<dt>snRNA cells</dt><dd>${_escapeHtml(snrnaCells)}</dd>` +
      `<dt>snRNA LFC (TG vs WT)</dt><dd>${f5Lfc == null ? '<span class="muted">—</span>' : `<span class="attr-num-lfc" style="background:${_attrLfcColor(f5Lfc)}">${f5Lfc.toFixed(3)}</span>`} <span class="muted">(info-only)</span></dd>` +
      `<dt>WMB detected</dt><dd>${_detGateCell(wmbDetected, wmbFrac, "this WMB class")}</dd>` +
      `<dt>WMB concentration</dt><dd>${_concTierCell(wmbTier)}</dd>` +
      `<dt>SEA-AD LFC (AD vs control)</dt><dd>${seaLfc == null ? '<span class="muted">—</span>' : `<span class="attr-num-lfc" style="background:${_attrLfcColor(seaLfc)}">${seaLfc.toFixed(3)}</span>`}</dd>` +
      `<dt>Cluster source</dt><dd>${_escapeHtml(row.cluster_source || "new_clusters")}</dd>`;

    const reconcileSentence =
      `<strong>eff = ${_specF2(eff)}</strong> — effective number of curated <strong>specificity units</strong> this kinase concentrates in ` +
      `(1 / Σ unit share² over all selected-tissue units; over-split <code>new_clusters</code> collapsed to units). ${band.txt} → confidence ` +
      `<strong>${_escapeHtml(conf.replace("_", " "))}</strong>.`;

    const kinaseDlHtml =
      `<dt>Effective # units (eff)</dt><dd>${eff != null ? `<span class="badge ${band.badge}">${_specF2(eff)}</span>` : "—"} <span class="muted">${band.txt}</span></dd>` +
      `<dt>Subtype spread</dt><dd>${rawEff != null ? _specF2(rawEff) : "—"} <span class="muted">effective # native new_clusters over all clusters (not the tier input)</span></dd>` +
      `<dt>Confidence</dt><dd><span class="${_attrConfidenceClass(conf)}" title="${_escapeHtml(row.confidence_basis || '')}">${_escapeHtml(conf.replace('_', ' '))}</span></dd>` +
      `<dt>Top cell type</dt><dd>${_escapeHtml(row.fivexfad_top_celltype || "—")}</dd>`;

    host.innerHTML = _renderSpecificityVerdictShell({
      conf,
      eff,
      thisCellTitle: row.cell_type || "",
      thisCellDlHtml,
      kinaseTitleHtml: `This kinase (${_escapeHtml(ctx.gene || "")})`,
      kinaseDlHtml,
      reconcileSentence,
    });
  }

  // ---- §4 per-cell substrate-site OLS (5xFAD-local, uses celltype_ols_shards) --
  // Genuinely distinct from the shared _renderDecompOlsTable, which uses
  // SliceCache.loadDecompOls (Song decomp OLS with contrast_id indexing).
  // 5xFAD's per-cell OLS shard uses _f5LoadCelltypeOls, filtered by
  // tissue/track/contrast/cell_type — different schema and loader.

  function _f5RenderCelltypeOlsSection(secHostId, ctx, row) {
    const host = document.getElementById(secHostId);
    if (!host) return;
    const group = ctx && ctx.group;
    if (!group) {
      host.innerHTML = `<div class="muted">No group context for per-cell OLS.</div>`;
      return;
    }
    const age = ctx.age;
    const contrast = `TG_vs_WT_${age}mo`;
    const cellType = row && row.cell_type || "";
    host.innerHTML = `<div class="muted">Loading per-cell substrate-site OLS shard...</div>`;
    _f5LoadCelltypeOls(group).then(payload => {
      if (!host || !_f5StillSelected(group)) return;
      const rows = ((payload && payload.rows) || []).filter(r =>
        r.tissue === group.tissue
        && r.track === group.track
        && r.contrast === contrast
        && r.cell_type === cellType
      );
      if (!rows.length) {
        host.innerHTML = `<div class="muted">No per-cell substrate-site OLS rows for ${_escapeHtml(cellType)} in ${_escapeHtml(contrast)}.</div>`;
        return;
      }
      rows.sort((a, b) => Math.abs(_f5num(b.lfc) || 0) - Math.abs(_f5num(a.lfc) || 0));
      host.innerHTML = _f5SmallTable(rows.slice(0, 200), [
        {key: "site_id", label: "Site", fmt: _f5SiteCell, html: true},
        {key: "gene_symbol", label: "Gene"},
        {key: "motif", label: "Motif"},
        {key: "lfc", label: "Per-cell beta", fmt: _f5FmtShort},
        {key: "se", label: "SE", fmt: _f5FmtShort},
        {key: "pval", label: "p", fmt: _f5FmtShort},
        {key: "fdr", label: "FDR", fmt: _f5FmtShort},
        {key: "n_wt", label: "WT"},
        {key: "n_tg", label: "TG"},
      ]);
    });
  }

  // ---- Detail sections in accordion order ----------------------------------
  // §0 specificity verdict (5xFAD-local: reads fivexfad_* fields)
  // §1 WMB dot plot (shared _renderWMBDotPlot)
  // §2 SEA-AD heatmap (shared _renderSEAADHeatmap)
  // §3 per-cell substrate-site OLS (5xFAD-local: reads celltype_ols_shards)

  const sections = [
    {
      id: "specificity",
      title: null,
      caption: null,
      wide: true,
      render(secHostId, ctx, row, _kstats) {
        _f5RenderSpecificityVerdict(secHostId, ctx, row);
      },
    },
    {
      id: "wmb",
      title: `§1 · Expression — WMB reference <span class="muted">(wmb_kinase_expression.csv)</span>`,
      caption: null,
      wide: false,
      render(secHostId, ctx, row, _kstats) {
        const host = document.getElementById(secHostId);
        if (!host) return;
        const gene = ctx.gene || "";
        const cellType = row.cell_type || "";
        host.innerHTML =
          `<p class="muted attr-caption">Seurat-style dot plot for ${_escapeHtml(gene)} across Allen Whole Mouse Brain classes. Color = mean log2 expression, dot size = fraction of cells expressing. The clicked cell type's WMB class is outlined.</p>` +
          `<div id="${secHostId}-plot"></div>`;
        _renderWMBDotPlot(`${secHostId}-plot`, ctx, cellType);
      },
    },
    {
      id: "seaad",
      title: `§2 · Disease direction — SEA-AD <span class="muted">(sea_ad_supertype_lfc.csv)</span>`,
      caption: null,
      wide: false,
      render(secHostId, ctx, row, _kstats) {
        const host = document.getElementById(secHostId);
        if (!host) return;
        const gene = ctx.gene || "";
        const cellType = row.cell_type || "";
        host.innerHTML =
          `<p class="muted attr-caption">Per-supertype human AD-vs-control LFC for ${_escapeHtml(gene)}, grouped by subclass. The subclass matching this cell type is outlined. Direction-tier evidence — does its activity move with disease — not specificity.</p>` +
          `<div id="${secHostId}-hm"></div>`;
        _renderSEAADHeatmap(`${secHostId}-hm`, ctx, cellType);
      },
    },
    {
      id: "celltype-ols",
      title: `§3 · Mechanism — per-cell substrate-site OLS <span class="muted">(${COHORT_LABELS.fivexfad} celltype_ols_shards)</span>`,
      caption: null,
      wide: true,
      render(secHostId, ctx, row, _kstats) {
        const host = document.getElementById(secHostId);
        if (!host) return;
        const cellType = row.cell_type || "";
        const gene = ctx.gene || "";
        host.innerHTML =
          `<p class="muted attr-caption">Substrate-site TG-vs-WT effects for ${_escapeHtml(cellType)}, restricted to ${_escapeHtml(gene)}'s kinase-library substrate set in the selected tissue, track, and age.</p>` +
          `<div id="${secHostId}-tbl" class="audit-scroll"></div>`;
        _f5RenderCelltypeOlsSection(`${secHostId}-tbl`, ctx, row);
      },
    },
  ];

  // ---- Detail layout --------------------------------------------------------
  // §0 specificity verdict (wide, no outer wrapper — renders its own <section>).
  // §1 WMB + §2 SEA-AD in a grid.
  // §3 per-cell OLS — wide, outside the grid.

  function renderDetailLayout(sectionStubs, _ctx, _row, _kstats) {
    const byId = {};
    for (const {sec, secHostId} of sectionStubs) byId[sec.id] = secHostId;

    const specHostId = byId["specificity"];
    const wmbHostId = byId["wmb"];
    const seaadHostId = byId["seaad"];
    const olsHostId = byId["celltype-ols"];

    return (
      `<div id="${specHostId}"></div>` +
      `<div class="attr-drawer-grid">` +
        `<section class="attr-section"><h5>${sections.find(s => s.id === "wmb").title}</h5>` +
          `<div id="${wmbHostId}"></div></section>` +
        `<section class="attr-section"><h5>${sections.find(s => s.id === "seaad").title}</h5>` +
          `<div id="${seaadHostId}"></div></section>` +
      `</div>` +
      `<section class="attr-section attr-section-wide"><h5>${sections.find(s => s.id === "celltype-ols").title}</h5>` +
        `<div id="${olsHostId}" class="audit-scroll"></div></section>`
    );
  }

  return {
    columns,
    superGroupRow,
    getRows,
    dedupKey,
    dedupCmp,
    defaultSort,
    sortTiebreak,
    rowVisible,
    bulkAnchor,
    showAllLabel,
    renderCrosswalk,
    explainerHtml,
    sections,
    renderDetailLayout,
  };
})();
