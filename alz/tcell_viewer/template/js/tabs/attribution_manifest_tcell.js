
// attribution_manifest_tcell.js — T-cell cohort attribution manifest.
// Consumed by AttributionView.render("attr-verdict", ctx, TCELL_MANIFEST).
//
// Two orthogonal within-cohort axes (neither is "specificity" — that word is
// reserved for the NSCLC reference):
//   lineage confidence (CD8/CD4) drives the confidence pill (kinase-level)
//   state enrichment (tcell_state_enrichment) is the activation-continuum axis.
//
// Column spine:
//   cell_type (a STATE) with cell-type-confidence pill · tcell_detected (Detected)
//   · tcell_state_enrichment (Enrichment, gated on detection) · tcell_lfc
//   (LFC — info) · tcell_concordance (vs Bulk — info) · tcell_consistency
//   (Timecourse — info). Raw state cell counts accompany the detection call.
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
  // MEAN expression across all observed evidence-backed states (gene-agnostic
  // baseline), so a kinase concentrated in one state scores a high fold. Reuses the shared badge
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
      title: "Kinase-level lineage confidence: how concentrated this kinase is in CD8 versus CD4 states from the donor's own scRNA. The independent NSCLC reference contributes only a coarse T/NK presence check; it is not mapped to these evidence-backed states. Shown once on the kinase's headline state row.",
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
      title: "State enrichment — this state's transcript level as a fold over the kinase's baseline mean across all observed evidence-backed states. Detection and the raw pooled state cell count are shown separately; no minimum-cell gate is applied.",
      render(r, _numFmt, _ctx) {
        const det = r.tcell_detected === true || r.tcell_detected === "True" || r.tcell_detected === "true";
        return `<td class="attr-num">${det ? _tcellEnrichCell(r.tcell_state_enrichment)
          : '<span class="muted" title="Not detected in this state (fraction < 10%) — no state enrichment">—</span>'}</td>`;
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
  ];

  // ---- Super-group header row -----------------------------------------------

  function superGroupRow(cols) {
    const _grpCounts = cols.reduce((acc, c) => { acc[c.group] = (acc[c.group]||0)+1; return acc; }, {});
    return `<tr class="attr-verdict-supergroup">` +
      `<th class="attr-supergroup-spacer" colspan="${(_grpCounts.id || 0)}"></th>` +
      `<th class="attr-supergroup-attr" colspan="${(_grpCounts.attr || 0)}" title="Within-cohort cell-state attribution across this cohort's evidence-backed per-cell states.">Within-cohort attribution (vs bulk direction)</th>` +
    `</tr>`;
  }

  // ---- Data hooks -----------------------------------------------------------

  function getRows(ctx) {
    // Day-invariant: fetch all contrasts. The verdict table renders one row per
    // cell_type; per-day quantities (LFC, Decomp NES, bulk NES) live in each
    // row's expand detail as heat-strips over the 5-day MEA axis. This lets the
    // table render for the scRNA-less days (d15/d19) too — localization is
    // pooled across all scRNA days and does not depend on the day selector.
    return ctx.kinase_id != null
      ? getScopedAttribution(ctx.kinase_id, { day: "", celltype: "" })
      : [];
  }

  // Dedup on cell_type only: the surviving row's day-invariant columns
  // (tcell_detected, tcell_fraction_expressing, tcell_state_enrichment,
  // confidence_tier) are byte-identical across contrasts, so which contrast
  // wins the dedupCmp is lossless for those columns.
  function dedupKey(r) { return `${r.cell_type}`; }

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

  // Always show every evidence-backed state; sorting never drops rows.
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
    `<p>The <strong>location-confidence</strong> pill is a kinase-level CD8-versus-CD4 lineage call from this cohort's own scRNA. It is shown on the kinase's top state only; all other rows show a muted dot.</p>` +
    `<p>The <strong>within-cohort</strong> axes localize the bulk kinase-activity signal to the evidence-backed per-cell states. The independent NSCLC reference remains available as a separate coarse cell-type panel, but is not crosswalked to these states.</p>` +
    `<table class="attr-explainer-table" style="margin-bottom:8px;">` +
      `<thead><tr><th>Axis</th><th>What it tells you</th></tr></thead><tbody>` +
      `<tr><td><strong>Location confidence</strong></td><td>Kinase-level CD8/CD4 lineage concentration relative to the even 1/2 share, with independent coarse NSCLC T/NK detection as corroboration. Shown on the top state; muted dot elsewhere.</td></tr>` +
      `<tr><td><strong>Detected</strong></td><td>Within-cohort detection in this cohort's own scRNA, pooled across all days: fraction ≥ 10% (normalization-free presence).</td></tr>` +
      `<tr><td><strong>Enrichment</strong></td><td>This state's transcript as a fold over the kinase's mean expression across all observed evidence-backed states. Detection and pooled state cell count are shown as raw evidence; no minimum-cell gate is applied.</td></tr>` +
      `<tr><td><strong>LFC / vs Bulk</strong></td><td>Does the transcript move the same direction as the bulk kinase activity at this day? <strong>Info only</strong> — concordance between kinase activity and its own mRNA is at chance (OR≈1, verified 2026-06-03; same in the published mouse Song method). Never used to filter.</td></tr>` +
      `<tr><td><strong>Timecourse</strong></td><td>How many of the three contrast days (d13/d17/d20) agree in direction with bulk. Credibility from timecourse consistency, not per-day significance.</td></tr>` +
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
    // tcell_top_celltype is the lineage (CD8 / CD4) — the axis the
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

    const thisCellDlHtml =
      `<dt>Cells expressing</dt><dd>${_specPct(row.tcell_fraction_expressing)}${detected ? "" : ' <span class="muted">(below 10%; not eligible for state enrichment)</span>'}</dd>` +
      `<dt>State cells</dt><dd>${row.tcell_state_n_cells == null ? "—" : Number(row.tcell_state_n_cells).toLocaleString()} <span class="muted">cells pooled across observed days</span></dd>` +
      `<dt>State enrichment</dt><dd>${detected ? _tcellEnrichCell(row.tcell_state_enrichment) : '<span class="muted" title="Not detected in this state — no state enrichment">—</span>'} <span class="muted">vs the kinase's baseline (mean) state</span></dd>`;

    const reconcileSentence =
      `The confidence pill reflects <strong>lineage confidence</strong> (CD8 / CD4), ` +
      `not state breadth: <strong>${_escapeHtml(conf.replace("_", " "))}</strong> — ${_escapeHtml(basis)}.` +
      (peakEnr != null && peakEnr >= 1.5
        ? ` Separately, on the activation-state axis the transcript peaks ${peakEnr.toFixed(1)}× in ${_escapeHtml(peakEnrState)} over its typical state.`
        : ` On the activation-state axis it is not state-enriched (no detected state ≥1.5× its mean-state baseline).`);

    const kinaseDlHtml =
      `<dt>Lineage</dt><dd>${_escapeHtml(ctType) || "<span class='muted'>—</span>"} <span class="muted">— CD8 or CD4 concentration; sets the confidence pill</span></dd>` +
      `<dt>Peak state enrichment</dt><dd>${peakEnr == null
        ? "<span class='muted'>— (no detected state)</span>"
        : `${_tcellEnrichCell(peakEnr)} <span class="muted">in ${_escapeHtml(peakEnrState)} — activation-state axis (info, does not set the pill)</span>`}</dd>` +
      `<dt>Cell-state spread</dt><dd><span class="badge ${band.badge}">${_specF2(eff)}</span> <span class="muted">effective # evidence-backed states (info; not the confidence axis)</span></dd>` +
      `<dt>Lineage confidence</dt><dd><span class="${_attrConfidenceClass(conf)}" title="${_escapeHtml(basis)}">${_escapeHtml(conf.replace("_", " "))}</span></dd>` +
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

  // Heat-strip renderer over the full CONTRASTS axis. `valuesByCid` is a
  // dense array of length CONTRASTS.length whose entries are the numeric
  // value for that contrast index, or `null` for a "no scRNA library" gap.
  // Diverging red (≥0) / blue (<0) with per-strip saturation normalized to
  // `maxAbs`. The cell whose contrast == `selCid` (a CONTRASTS index) gets a
  // `.npc.sel` outline so the global contrast selector keeps a tie-in
  // without gating the strip. Reuses the existing .tcell-nes-profile CSS
  // grammar (kinase_explorer.js:_renderNesProfile — but no fdr/sig here,
  // since attribution has no per-day p-values).
  function _renderAttrHeatStrip(valuesByCid, opts) {
    const {maxAbs = 0, selCid = -1, gapTitle = "", tipLabel = "value", tipFmt = (v) => v.toFixed(2)} = opts || {};
    const cells = [];
    const colLabels = [];
    for (let ci = 0; ci < CONTRASTS.length; ci++) {
      const c = CONTRASTS[ci];
      const v = valuesByCid[ci];
      const isGap = v == null;
      const isSel = ci === selCid;
      const dayLabel = String(c).replace(/_d2$/, "");
      let bg = "#fff", cls = "npc", tip;
      if (isGap) {
        cls += " gap";
        tip = gapTitle
          ? `${c}: ${gapTitle}`
          : `${c}: no scRNA library at ${dayLabel} — ${tipLabel} undefined`;
      } else if (isFinite(v)) {
        if (maxAbs > 0) {
          const a = Math.min(1, Math.abs(v) / maxAbs);
          const rgb = v >= 0 ? [197,48,48] : [43,108,176];
          bg = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${(0.15 + 0.85 * a).toFixed(3)})`;
        }
        tip = `${c}: ${tipLabel} ${tipFmt(v)}`;
      } else {
        cls += " gap";
        tip = `${c}: ${tipLabel} n/a`;
      }
      if (isSel) cls += " sel";
      cells.push(`<div class="${cls}" style="background:${bg};" title="${_escapeHtml(tip)}"></div>`);
      colLabels.push(`<span title="${_escapeHtml(c)}">${_escapeHtml(dayLabel)}</span>`);
    }
    const countStyle = `--nes-profile-count:${CONTRASTS.length};`;
    return `<div class="nes-profile-wrap tcell-nes-profile attr-heat-strip" style="${countStyle}">` +
      `<div class="nes-profile-col-labels">${colLabels.join("")}</div>` +
      `<div class="nes-profile-cell">${cells.join("")}</div>` +
      `</div>`;
  }

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

      // ---- Heat-strips over the full MEA axis ---------------------------
      // The bulk NES / bulk FDR live on PAYLOAD.kinases keyed by contrast; we
      // pull one value per CONTRASTS entry so the strip is populated at all 5
      // MEA days (there is no scRNA gap for bulk activity).
      const _K = ViewerPayload.kinases();
      const kid = ctx.kinase_id;
      const bulkByCid = new Array(CONTRASTS.length).fill(null);
      const bulkFdrByCid = new Array(CONTRASTS.length).fill(null);
      if (kid != null && _K) {
        for (let ci = 0; ci < CONTRASTS.length; ci++) {
          const c = CONTRASTS[ci];
          const arrN = _K["NES_" + c];
          const arrF = _K["FDR_" + c];
          if (arrN && arrN[kid] != null && isFinite(arrN[kid])) bulkByCid[ci] = arrN[kid];
          if (arrF && arrF[kid] != null && isFinite(arrF[kid])) bulkFdrByCid[ci] = arrF[kid];
        }
      }

      // LFC (per-state, per-day) — only defined on scRNA days. Days without
      // an attribution row (d15/d19) stay null → rendered as gap cells.
      const lfcByCid = new Array(CONTRASTS.length).fill(null);
      for (const r of cellRows) {
        if (r.contrast_id != null && r.tcell_lfc != null && isFinite(r.tcell_lfc)) {
          lfcByCid[r.contrast_id] = r.tcell_lfc;
        }
      }

      // Decomp NES (per-state, per-day) from the decomposition_index. Same
      // gapping rule: scRNA-less days → null → gap.
      const decByCid = new Array(CONTRASTS.length).fill(null);
      const decFdrByCid = new Array(CONTRASTS.length).fill(null);
      if (typeof _ensureKinaseIndexes === "function") _ensureKinaseIndexes();
      if (typeof _decompByKey !== "undefined" && _decompByKey && kid != null) {
        for (let ci = 0; ci < CONTRASTS.length; ci++) {
          const entry = _decompByKey.get(`${kid}|${ci}|${cellType}`);
          if (entry && entry.nes != null && isFinite(entry.nes)) {
            decByCid[ci] = entry.nes;
            if (entry.fdr != null && isFinite(entry.fdr)) decFdrByCid[ci] = entry.fdr;
          }
        }
      }

      // maxAbs is normalized PER KINASE (max |v| across that kinase's
      // states×days for LFC/Decomp; across the 5 days for bulk NES) so a
      // flat state reads flat and cross-state magnitude is comparable.
      const bulkMax = Math.max(0, ...bulkByCid.filter(v => v != null && isFinite(v)).map(Math.abs));
      let lfcMax = 0, decMax = 0;
      if (kid != null) {
        for (const r of getScopedAttribution(kid, { day: "", celltype: "" })) {
          if (r.tcell_lfc != null && isFinite(r.tcell_lfc)) lfcMax = Math.max(lfcMax, Math.abs(r.tcell_lfc));
        }
        if (typeof _decompByKey !== "undefined" && _decompByKey) {
          for (const [k, v] of _decompByKey.entries()) {
            if (!k.startsWith(`${kid}|`)) continue;
            if (v && v.nes != null && isFinite(v.nes)) decMax = Math.max(decMax, Math.abs(v.nes));
          }
        }
      }

      const selCid = (typeof CONTRASTS !== "undefined" && ctx && ctx.contrast)
        ? CONTRASTS.indexOf(ctx.contrast) : -1;

      const bulkStrip = _renderAttrHeatStrip(bulkByCid, {
        maxAbs: bulkMax, selCid,
        tipLabel: "bulk NES",
        tipFmt: (v) => v.toFixed(2),
        gapTitle: "no bulk NES for this contrast",
      });
      const lfcStrip = _renderAttrHeatStrip(lfcByCid, {
        maxAbs: lfcMax, selCid,
        tipLabel: "transcript log2FC",
        tipFmt: (v) => v.toFixed(3),
        gapTitle: "no scRNA library at this day — transcript LFC undefined",
      });
      const decStrip = _renderAttrHeatStrip(decByCid, {
        maxAbs: decMax, selCid,
        tipLabel: "decomp NES",
        tipFmt: (v) => v.toFixed(2),
        gapTitle: "no scRNA library at this day — decomp NES undefined",
      });

      const stripsHtml =
        `<div class="attr-heat-strip-stack">` +
          `<div class="attr-heat-strip-row"><span class="attr-heat-strip-label" title="Kinase-level bulk MEA activity (per-kinase, per-day) — the anchor the transcript is asked to concord with. Populated at all 5 MEA days.">Bulk NES</span>${bulkStrip}</div>` +
          `<div class="attr-heat-strip-row"><span class="attr-heat-strip-label" title="Pseudobulk transcript log2 fold change for ${_escapeHtml(cellType)} vs the d2 baseline. Only defined on scRNA days (d13/d17/d20); d15/d19 are ✗ gap cells.">LFC</span>${lfcStrip}</div>` +
          `<div class="attr-heat-strip-row"><span class="attr-heat-strip-label" title="Per-state deconvoluted kinase NES for ${_escapeHtml(cellType)} from the raw projected-state MEA. Only defined on scRNA days.">Decomp NES</span>${decStrip}</div>` +
        `</div>`;

      // ---- Exact-value numeric table (kept below the strips) ------------
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
        `(detection and enrichment are pooled across all scRNA days; both are day-invariant). ` +
        `Below: bulk NES / transcript LFC / decomp NES over the full 5-day MEA axis. ` +
        `d15/d19 have no scRNA library — LFC/decomp render as gap cells; bulk NES is populated at all 5 days. ` +
        `The currently-selected contrast is outlined.</p>` +
        stripsHtml +
        `<p class="muted attr-caption" style="margin-top:.8em;">Exact per-day values (scRNA days only):</p>` +
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
