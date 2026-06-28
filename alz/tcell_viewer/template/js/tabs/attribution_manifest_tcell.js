
// attribution_manifest_tcell.js — T-cell cohort attribution manifest.
// Consumed by AttributionView.render("attr-verdict", ctx, TCELL_MANIFEST).
//
// Two orthogonal within-cohort axes (neither is "specificity" — that word is
// reserved for the NSCLC reference):
//   cell-TYPE confidence (CD8/CD4/Treg) drives the confidence pill (kinase-level)
//   state enrichment (tcell_state_enrichment) is the activation-continuum axis.
//
// Column spine:
//   cell_type (a STATE) with cell-type-confidence pill · tcell_detected (Detected)
//   · tcell_state_enrichment (Enrichment, gated on detection) · tcell_lfc
//   (LFC — info) · tcell_concordance (vs Bulk — info) · tcell_consistency
//   (Timecourse — info) · nsclc_detected (NSCLC Detected)
//
// Sections (accordion, in order):
//   §0 Confidence verdict (cell-type axis sets the pill; state axis is info;
//      NSCLC corroborator)
//   §1 Within-cohort transcript trace (local)
//   §2 NSCLC cell-type strip (local)
//   WMB dot plot, SEA-AD heatmap, decomp OLS — dropped (not a mouse-brain cohort;
//   no decomp layer): per plan §"Section drops".

const TCELL_MANIFEST = (() => {

  const _num = (v, d=3) => (v == null || !isFinite(v)) ? "" : Number(v).toFixed(d);

  // State-enrichment cell: this state's transcript as a fold over the kinase's
  // MEAN expression across all adequately-sampled ProjecTILs states (gene-agnostic
  // baseline), so a kinase concentrated in one state scores a high fold and it is
  // not saturated by the transcriptional homogeneity of ProjecTILs states (where a
  // dominance-of-total concentration tier cannot localize). Reuses the shared badge
  // palette (vhi/hi/mid/lo). Local to T-cell (no other cohort has this axis).
  function _tcellEnrichCell(fold) {
    const v = Number(fold);
    if (fold == null || !isFinite(v)) {
      return `<span class="muted" title="No finite state-enrichment value">—</span>`;
    }
    const lbl = v.toFixed(1) + "×";
    const cls = v >= 3 ? "hi" : v >= 2 ? "mid" : v >= 1.5 ? "lo" : "";
    if (!cls) {
      return `<span class="muted" title="Within ~1.5× of the kinase's baseline (mean) T-cell state — broadly expressed, not state-enriched">${lbl}</span>`;
    }
    return `<span class="badge ${cls}" title="${lbl} the kinase's baseline (mean-state) expression — enriched in this state relative to a typical T-cell state">${lbl}</span>`;
  }

  // ---- Column definitions ---------------------------------------------------

  const columns = [
    {
      key: "cell_type", label: "Cell state", type: "str", group: "id", title: "",
      render(r, _numFmt, _ctx) {
        return `<td class="attr-celltype"><span class="attr-chevron" aria-hidden="true">▸</span> ${_escapeHtml(r.cell_type)} ${_attrVerdictConfCell(r)}</td>`;
      },
    },
    {
      key: "confidence_tier", label: "Cell-type confidence", type: "conf", group: "id", hidden: true,
      title: "Kinase-level cell-type confidence: how concentrated this kinase is in one T-cell TYPE (CD8 / CD4 / Treg), from the donor's own scRNA, corroborated by the independent NSCLC reference detecting the kinase in its T_NK cell type. tier ≥2 (≥2× the even 1/3 share) → moderate/high; broad → low. Shown once, on the kinase's headline state row — it is a per-kinase property, not per state.",
      render(r, _numFmt, _ctx) { return `<td>${_attrVerdictConfCell(r)}</td>`; },
    },
    {
      key: "tcell_detected", label: "Detected", type: "num", group: "attr",
      sub: "within", subLabel: "Within-cohort attribution (vs bulk direction)",
      subTitle: "Within-cohort standard detection metric, pooled across all scRNA days. Computed in alz/cross_reference/tcell_within_cohort.py.",
      title: "Is the kinase's transcript detected in this state? ✓/✗ = fraction of cells expressing ≥ 10% (normalization-free presence), with the % shown. Sorts by detection.",
      render(r, _numFmt, _ctx) {
        return `<td class="attr-num">${_detGateCell(r.tcell_detected, r.tcell_fraction_expressing, "this T-cell state")}</td>`;
      },
    },
    {
      key: "tcell_state_enrichment", label: "Enrichment", type: "num", group: "attr",
      sub: "within", subLabel: "Within-cohort attribution (vs bulk direction)",
      subTitle: "Within-cohort detection + state enrichment, pooled across all scRNA days. Computed in alz/cross_reference/tcell_within_cohort.py.",
      title: "State enrichment — this state's transcript level as a fold over the kinase's BASELINE (mean expression across all adequately-sampled states, ≥50 cells). A kinase concentrated in one state scores a high fold; a broadly-expressed kinase sits near 1×. An undetected state shows no badge (—). ≥3× strong / ≥2× moderate / ≥1.5× mild; <1.5× = broadly expressed (not state-enriched). Sorts by fold.",
      render(r, _numFmt, _ctx) {
        const det = r.tcell_detected === true || r.tcell_detected === "True" || r.tcell_detected === "true";
        return `<td class="attr-num">${det ? _tcellEnrichCell(r.tcell_state_enrichment)
          : '<span class="muted" title="Not detected in this state (fraction < 10%) — no state enrichment">—</span>'}</td>`;
      },
    },
    {
      key: "tcell_lfc", label: "LFC", type: "num", group: "attr",
      sub: "within", subLabel: "Within-cohort attribution (vs bulk direction)",
      subTitle: "Within-cohort standard detection metric, pooled across all scRNA days. Computed in alz/cross_reference/tcell_within_cohort.py.",
      title: "Pseudobulk transcript log2 fold change vs the d2 baseline at this contrast day (mean per-cell log-expression difference). Color: red = up, blue = down. No p-value — a single donor has no biological replicates (see Methods). INFO ONLY: OR≈1 between transcript direction and bulk NES.",
      render(r, _numFmt, _ctx) {
        return r.tcell_lfc == null || !isFinite(r.tcell_lfc)
          ? `<td class="attr-num attr-empty" title="Transcript absent at this day or the d2 baseline — no fold-change defined. Expected where the kinase is not expressed in this state.">—</td>`
          : `<td class="attr-num attr-num-lfc" style="background:${_attrLfcColor(r.tcell_lfc)}">${_num(r.tcell_lfc, 3)}</td>`;
      },
    },
    {
      key: "_decomp_state_nes", label: "Decomp NES", type: "num", group: "attr",
      sub: "within", subLabel: "Within-cohort attribution (vs bulk direction)",
      subTitle: "Per-state deconvoluted kinase NES from the raw (non-stoichiometry-corrected) projected-state MEA (mea_projected_state_raw.csv). Available after the state_mea gate run.",
      title: "Deconvoluted per-state NES: the kinase MEA NES for this T-cell state from the raw projected-state ssGSEA decomposition. The raw (not stoichiometry-corrected) track is used because stoich correction cancels the per-state decomposition share, collapsing NES to the bulk value. Empty until the state_mea gate run completes and produces mea_projected_state_raw.csv.",
      render(r, _numFmt, ctx) {
        // Read from the decomposition_index via _decompByKey, keyed by
        // (kinase_id | contrast_id | cell_type). kinase_id is the per-donor
        // integer index carried on ctx (== attribution_index.kinase_id); cell_type
        // is this row's ProjecTILs state.
        if (typeof _decompByKey === "undefined" || !_decompByKey) {
          return `<td class="attr-num attr-empty" title="No projected-state MEA data (gate dependency).">—</td>`;
        }
        if (typeof _ensureKinaseIndexes === "function") _ensureKinaseIndexes();
        const kinase_id = ctx && ctx.kinase_id;
        const CONTRASTS_ARR = (typeof CONTRASTS !== "undefined") ? CONTRASTS : [];
        const cid = CONTRASTS_ARR.indexOf(ctx && ctx.contrast);
        const state = r.cell_type || r.state || "";
        if (kinase_id == null || cid < 0 || !state) {
          return `<td class="attr-num attr-empty" title="No decomp data for this kinase/contrast/state.">—</td>`;
        }
        const key = `${kinase_id}|${cid}|${state}`;
        const entry = _decompByKey.get(key);
        if (!entry || entry.nes == null || !isFinite(entry.nes)) {
          return `<td class="attr-num attr-empty" title="No decomp NES for ${state} (state MEA pending).">—</td>`;
        }
        const fdrStr = (entry.fdr != null && isFinite(entry.fdr))
          ? ` FDR ${entry.fdr.toExponential(1)}` : "";
        return `<td class="attr-num attr-num-lfc" style="background:${_attrLfcColor(entry.nes)}"
          title="Decomp NES ${entry.nes.toFixed(2)}${fdrStr}">${_num(entry.nes, 2)}</td>`;
      },
    },
    {
      key: "tcell_consistency", label: "Timecourse", type: "num", group: "attr",
      sub: "within", subLabel: "Within-cohort attribution (vs bulk direction)",
      subTitle: "Within-cohort standard detection metric, pooled across all scRNA days. Computed in alz/cross_reference/tcell_within_cohort.py.",
      title: "Count of contrast days {d13, d17, d20} where the transcript moves concordantly with the bulk NES. Credibility comes from timecourse consistency, not a per-day p-value.",
      render(r, _numFmt, _ctx) {
        const consist = r.tcell_consistency || 0;
        return `<td class="attr-num" title="${consist} of 3 contrast days (d13/d17/d20) move concordantly with bulk.">${consist}<span class="muted" style="font-size:10px;"> / 3</span></td>`;
      },
    },
    {
      key: "nsclc_detected", label: "Detected", type: "num", group: "ext",
      sub: "nsclc", subLabel: "NSCLC reference (detection)",
      subTitle: "Independent human reference: 10x 897k-cell NSCLC, detection at the matched T-cell state.",
      title: "Independent reference (10x 897k-cell NSCLC): is the kinase actually present in THIS exact T-cell state? Fraction of reference cells expressing it (✓ = ≥10% of cells express it — the single cross-cohort detection floor, identical to the within-cohort scRNA). n/a = the kinase is outside the NSCLC probe panel. Sorts by detection.",
      render(r, _numFmt, _ctx) { return `<td class="attr-num">${_detGateCell(r.nsclc_detected, r.nsclc_frac, "this T-cell state (NSCLC reference)")}</td>`; },
    },
  ];

  // ---- Super-group header row -----------------------------------------------

  function superGroupRow(cols) {
    const _grpCounts = cols.reduce((acc, c) => { acc[c.group] = (acc[c.group]||0)+1; return acc; }, {});
    return `<tr class="attr-verdict-supergroup">` +
      `<th class="attr-supergroup-spacer" colspan="${(_grpCounts.id || 0)}"></th>` +
      `<th class="attr-supergroup-attr" colspan="${(_grpCounts.attr || 0)}" title="Within-cohort cell-state attribution: transcript enrichment across this cohort's own ProjecTILs T-cell states + concordance vs the bulk kinase-MEA direction at this contrast.">Within-cohort attribution (vs bulk direction)</th>` +
      `<th class="attr-supergroup-ext" colspan="${(_grpCounts.ext || 0)}" title="Independent human reference: 10x 897k-cell NSCLC (ProjecTILs/scGate T-states + marker-labeled non-T cell types). Detection at the matched T-state — is the kinase actually present where the within-cohort attribution localized it?">NSCLC reference (detection)</th>` +
    `</tr>`;
  }

  // ---- Data hooks -----------------------------------------------------------

  function getRows(ctx) {
    return ctx.kinase_id != null
      ? getScopedAttribution(ctx.kinase_id, { day: ctx.contrast || "", celltype: "" })
      : [];
  }

  function dedupKey(r) { return `${r.contrast_id}|${r.cell_type}`; }

  const _enr = r => (r.tcell_state_enrichment != null && isFinite(r.tcell_state_enrichment))
    ? r.tcell_state_enrichment : 0;

  // Keep the highest-confidence row per (contrast, cell_type); tie-break by enrichment.
  function dedupCmp(a, b) {
    const ra = _CONF_RANK[a.confidence_tier] ?? -1;
    const rb = _CONF_RANK[b.confidence_tier] ?? -1;
    if (ra !== rb) return ra > rb ? -1 : 1;
    return _enr(b) - _enr(a);
  }

  const defaultSort = {key: "confidence_tier", asc: false};

  function sortTiebreak(a, b) {
    return _enr(b) - _enr(a);
  }

  // Always show all states — no filter (de-gate directive: all 14 ProjecTILs
  // states listed for the human to read; sort reorders, nothing drops).
  const rowVisible = null;

  function bulkAnchor(ctx) {
    const _K = ViewerPayload.kinases();
    const nes = (_K && _K["NES_" + ctx.contrast]) ? _K["NES_" + ctx.contrast][ctx.kinase_id] : null;
    const fdr = (_K && _K["FDR_" + ctx.contrast]) ? _K["FDR_" + ctx.contrast][ctx.kinase_id] : null;
    return {
      contrast: ctx.contrast,
      nes,
      fdr,
      signNote: "sign of the bulk NES is the reference direction the concordance column is checked against. <strong>Positive NES = kinase more active at this day than at d2.</strong>",
    };
  }

  // No exclusivity summary line (T-cell uses confidence_basis in the location-confidence pill
  // tooltip; the §0 confidence verdict section provides the full breakdown).
  // No Show-all toggle (rowVisible = null, every row is always visible).
  // No reference crosswalk (no WMB/SEA-AD vocabulary in T-cell).

  const explainerHtml =
    `<details class="attr-explainer"><summary>How to read within-cohort T-cell attribution</summary>` +
    `<div class="attr-explainer-body">` +
    `<p>The <strong>location-confidence</strong> pill is the unified kinase-level exclusivity confidence (how exclusively the kinase localizes to one ProjecTILs state), corroborated by the independent NSCLC reference. It is a per-kinase property — the pill is shown on the kinase's top state only; all other rows show a muted dot. Click any row to expand the confidence detail.</p>` +
    `<p>The <strong>within-cohort</strong> axes (Detected / Enrichment / LFC / vs Bulk / Timecourse) localize the <strong>bulk</strong> kinase-activity signal to a T-cell ProjecTILs state using this cohort's own paired scRNA. The <strong>NSCLC detection</strong> column is an <strong>independent human reference</strong> (10x 897k-cell) corroborating whether the kinase is actually present in that exact state.</p>` +
    `<table class="attr-explainer-table" style="margin-bottom:8px;">` +
      `<thead><tr><th>Axis</th><th>What it tells you</th></tr></thead><tbody>` +
      `<tr><td><strong>Location confidence</strong></td><td>Kinase-level exclusivity (unified): effective number of states (inverse-Simpson breadth over all states) + NSCLC corroboration. Shown on the top state; muted dot elsewhere.</td></tr>` +
      `<tr><td><strong>Detected</strong></td><td>Within-cohort detection in this cohort's own scRNA, pooled across all days: fraction ≥ 10% (normalization-free presence).</td></tr>` +
      `<tr><td><strong>Enrichment</strong></td><td>State enrichment — this state's transcript as a fold over the kinase's <em>baseline</em>: its mean expression across all adequately-sampled ProjecTILs states. The T-cell enrichment metric asks "how many fold above this kinase's typical T-cell state?" so a kinase concentrated in one state scores high — it is not flattened by the transcriptional homogeneity of ProjecTILs states (where a dominance-of-total concentration tier saturates and cannot localize). ≥3× strong / ≥2× moderate / ≥1.5× mild; &lt;1.5× = broadly expressed.</td></tr>` +
      `<tr><td><strong>LFC / vs Bulk</strong></td><td>Does the transcript move the same direction as the bulk kinase activity at this day? <strong>Info only</strong> — concordance between kinase activity and its own mRNA is at chance (OR≈1, verified 2026-06-03; same in the published mouse Song method). Never used to filter.</td></tr>` +
      `<tr><td><strong>Timecourse</strong></td><td>How many of the three contrast days (d13/d17/d20) agree in direction with bulk. Credibility from timecourse consistency, not per-day significance.</td></tr>` +
      `<tr><td><strong>NSCLC Detected</strong></td><td>Independent 10x 897k-cell NSCLC reference: fraction of cells in this exact T-state that express the kinase (✓ = ≥10% of cells — the single cross-cohort detection floor). An independent corroborator of the within-cohort attribution. n/a = outside the NSCLC probe panel.</td></tr>` +
      `</tbody></table>` +
    `<p><strong>No p-value / FDR.</strong> Donor1 is a single donor with one scRNA library per day — no biological replicates, so a per-(state, day) significance test would be pseudoreplication. Direction + magnitude + timecourse consistency + NSCLC corroboration are reported instead.</p>` +
    `</div></details>`;

  // ---- §0 Confidence verdict — T-cell local renderer ----------------------
  // The shared _renderSpecificityVerdict reads Song-specific fields
  // (song_unit_effective_n etc.) and is not usable for T-cell. This T-cell
  // local renderer covers the same logical content: detection, state enrichment,
  // eff breadth, and the NSCLC corroborator. It is referenced from this manifest
  // only — no shared code is forked.

  function _renderTcellConfidenceVerdict(secHostId, ctx, row) {
    const host = document.getElementById(secHostId);
    if (!host) return;
    const conf = row.confidence_tier || "none";
    const basis = row.confidence_basis || "";
    // tcell_top_celltype is the cell TYPE (CD8 / CD4 / Treg) — the axis the
    // confidence pill is computed on. tcell_effective_n is the per-STATE
    // cell-state spread (info only; it does NOT set the confidence pill).
    const ctType = row.tcell_top_celltype || "";
    const eff = (row.tcell_effective_n != null && isFinite(row.tcell_effective_n))
      ? Number(row.tcell_effective_n) : null;
    const band = _specEffBand(eff);
    const detected = row.tcell_detected === true || row.tcell_detected === "True" || row.tcell_detected === "true";

    // Kinase-level peak STATE enrichment (activation-continuum axis): the largest
    // per-state fold-over-baseline across the kinase's ELIGIBLE states (ineligible
    // states are NaN upstream, so they never win).
    let peakEnr = null, peakEnrState = "";
    if (ctx.kinase_id != null) {
      for (const rr of getScopedAttribution(ctx.kinase_id, { day: "", celltype: "" })) {
        const e = rr.tcell_state_enrichment;
        if (e != null && isFinite(e) && (peakEnr == null || e > peakEnr)) {
          peakEnr = e; peakEnrState = rr.cell_type;
        }
      }
    }

    // NSCLC corroborator from the row (per-STATE NSCLC detection at the
    // crosswalked ProjecTILs state). Separate from the cell-type-axis T_NK
    // corroboration that sets the pill.
    const nsclcDet = row.nsclc_detected;
    const nsclcFrac = row.nsclc_frac;
    let corrLine;
    if (nsclcDet == null) {
      corrLine = `<span class="muted">Outside the NSCLC probe panel — reference cannot corroborate or refute.</span>`;
    } else if (nsclcDet === true || nsclcDet === "True" || nsclcDet === "true") {
      corrLine = `<span class="attr-badge attr-badge-info">✓ corroborated</span> NSCLC detects this kinase in the matched ${_escapeHtml(row.cell_type)} state (${_specPct(nsclcFrac)} of cells — expressed)`;
    } else {
      corrLine = `<span class="muted">✗ not detected</span> — NSCLC does NOT detect this kinase in the matched ${_escapeHtml(row.cell_type)} state (${_specPct(nsclcFrac)} of cells — not expressed).`;
    }

    const thisCellDlHtml =
      `<dt>Cells expressing</dt><dd>${_specPct(row.tcell_fraction_expressing)}${detected ? "" : ' <span class="muted">(below 10%; not eligible for state enrichment)</span>'}</dd>` +
      `<dt>State enrichment</dt><dd>${detected ? _tcellEnrichCell(row.tcell_state_enrichment) : '<span class="muted" title="Not detected in this state — no state enrichment">—</span>'} <span class="muted">vs the kinase's baseline (mean) state</span></dd>` +
      `<dt>NSCLC corroborator</dt><dd>${corrLine}</dd>`;

    const reconcileSentence =
      `The confidence pill reflects <strong>cell-type confidence</strong> (CD8 / CD4 / Treg), ` +
      `not state breadth: <strong>${_escapeHtml(conf.replace("_", " "))}</strong> — ${_escapeHtml(basis)}.` +
      (peakEnr != null && peakEnr >= 1.5
        ? ` Separately, on the activation-state axis the transcript peaks ${peakEnr.toFixed(1)}× in ${_escapeHtml(peakEnrState)} over its typical state.`
        : ` On the activation-state axis it is not state-enriched (no detected state ≥1.5× its median).`);

    const kinaseDlHtml =
      `<dt>Cell type</dt><dd>${_escapeHtml(ctType) || "<span class='muted'>—</span>"} <span class="muted">— the type (CD8 / CD4 / Treg) it concentrates in; sets the confidence pill</span></dd>` +
      `<dt>Peak state enrichment</dt><dd>${peakEnr == null
        ? "<span class='muted'>— (no detected, ≥50-cell state)</span>"
        : `${_tcellEnrichCell(peakEnr)} <span class="muted">in ${_escapeHtml(peakEnrState)} — activation-state axis (info, does not set the pill)</span>`}</dd>` +
      `<dt>Cell-state spread</dt><dd><span class="badge ${band.badge}">${_specF2(eff)}</span> <span class="muted">effective # ProjecTILs states (info; not the confidence axis)</span></dd>` +
      `<dt>Cell-type confidence</dt><dd><span class="${_attrConfidenceClass(conf)}" title="${_escapeHtml(basis)}">${_escapeHtml(conf.replace("_", " "))}</span></dd>` +
      `<dt>Basis</dt><dd class="muted" style="font-size:11px;">${_escapeHtml(basis)}</dd>`;

    host.innerHTML = _renderSpecificityVerdictShell({
      conf,
      eff,
      thisCellHeaderLabel: "This cell state",
      thisCellTitle: row.cell_type,
      thisCellDlHtml,
      kinaseTitleHtml: "This kinase overall",
      kinaseDlHtml,
      reconcileSentence,
    });
  }

  // ---- §1 Within-cohort transcript trace — T-cell local -------------------
  // Per-state timecourse: all contrast days for this cell type.
  // No p-value — single donor, no biological replicates.

  function _renderTcellTranscriptTrace(secHostId, ctx, row) {
    const host = document.getElementById(secHostId);
    if (!host) return;
    const cellType = row.cell_type || "";
    const cellRows = ctx.kinase_id != null
      ? getScopedAttribution(ctx.kinase_id, { day: "", celltype: cellType })
      : [];
    const num = (v, d=3) => (v == null || !isFinite(v)) ? "" : Number(v).toFixed(d);
    let traceBody;
    if (cellRows.length === 0) {
      traceBody = `<div class="muted">No within-cohort rows for ${_escapeHtml(cellType)}.</div>`;
    } else {
      const detected = cellRows[0].tcell_detected;
      const frac = cellRows[0].tcell_fraction_expressing;
      const enr = cellRows[0].tcell_state_enrichment;
      const byDay = cellRows.slice().sort((a, b) =>
        String(CONTRASTS[a.contrast_id]).localeCompare(String(CONTRASTS[b.contrast_id])));
      const trRows = byDay.map(r => {
        const day = CONTRASTS[r.contrast_id] || "";
        const lfc = r.tcell_lfc;
        const lfcCell = lfc == null || !isFinite(lfc)
          ? `<td class="attr-num attr-empty" title="Transcript absent at this day or the d2 baseline — no fold-change defined. Expected for a state where the kinase is not expressed.">—</td>`
          : `<td class="attr-num attr-num-lfc" style="background:${_attrLfcColor(lfc)}">${num(lfc, 3)}</td>`;
        const concCell = `<td class="attr-num attr-num-lfc" style="background:${_attrLfcColor(r.tcell_concordance)}">${num(r.tcell_concordance, 3)}</td>`;
        return `<tr><td>${_escapeHtml(day)}</td><td class="attr-num">${num(r.nes, 2)}</td>${lfcCell}${concCell}</tr>`;
      }).join("");
      const detHtml = _detGateCell(detected, frac, "this T-cell state");
      const enrHtml = _tcellEnrichCell(enr);
      traceBody =
        `<p class="muted attr-caption">Within-cohort: ${detHtml} ${enrHtml} ` +
        `(detection shown separately; state enrichment = fold over the kinase's baseline mean state). ` +
        `Each row is one contrast day: bulk NES, the pseudobulk transcript LFC, and their sign concordance. No p-value (single-donor timecourse).</p>` +
        `<table class="attr-verdict-table"><thead><tr>` +
          `<th>Contrast</th><th>Bulk NES</th><th>Transcript LFC</th><th>Concordance</th>` +
        `</tr></thead><tbody>${trRows}</tbody></table>`;
    }
    host.innerHTML = traceBody;
  }

  // ---- Sections (accordion detail) -----------------------------------------
  // §0 confidence verdict (T-cell-local renderer, NSCLC corroborator)
  // §1 within-cohort transcript trace (T-cell-local)
  // §2 NSCLC cell-type strip (references the local _renderNSCLCCellTypeStrip
  //    defined in kinase_audit.js — T-cell-specific, not shared)
  // WMB dot plot, SEA-AD heatmap, decomp OLS: dropped (not applicable).

  const sections = [
    {
      id: "confidence",
      title: null,
      caption: null,
      wide: true,
      render(secHostId, ctx, row, _kstats) {
        _renderTcellConfidenceVerdict(secHostId, ctx, row);
      },
    },
    {
      id: "transcript-trace",
      title: `§1 · Within-cohort transcript trace <span class="muted">(unified_attribution_tcells.csv)</span>`,
      caption: null,
      wide: true,
      render(secHostId, ctx, row, _kstats) {
        const host = document.getElementById(secHostId);
        if (!host) return;
        _renderTcellTranscriptTrace(secHostId, ctx, row);
      },
    },
    {
      id: "nsclc-strip",
      title: `§2 · NSCLC reference detection by cell type <span class="muted">(nsclc_kinase_expression.csv)</span>`,
      caption: null,
      wide: true,
      render(secHostId, ctx, _row, _kstats) {
        // _renderNSCLCCellTypeStrip is defined in kinase_audit.js (T-cell-specific,
        // not shared). It reads ctx.nsclcRows which is loaded by _loadKinaseAuditContext.
        if (typeof _renderNSCLCCellTypeStrip === "function") {
          _renderNSCLCCellTypeStrip(secHostId, ctx);
        } else {
          const host = document.getElementById(secHostId);
          if (host) host.innerHTML = `<div class="muted">NSCLC cell-type strip unavailable.</div>`;
        }
      },
    },
  ];

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
    explainerHtml,
    sections,
    // renderCrosswalk omitted — T-cell has no WMB/SEA-AD crosswalk.
    // exclSummary omitted — pill tooltip carries the basis; §0 has the full breakdown.
    // buildKstats omitted — T-cell confidence verdict is T-cell-local.
    // attachDerived omitted — no decomp NES/bulk_match to derive.
    // showAllLabel omitted — rowVisible is null, so every row is shown and no toggle needed.
  };
})();
