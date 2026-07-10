# Re-run Matt's T-cell analysis after cell-cycle regression

## Status

Implemented on 2026-07-10. The previous cycle-regression report changed Matt's
analytical narrative and has been replaced by the faithful rerun. Do not relabel
downstream inputs, rerun Incytr, or update the viewer until this report is reviewed.

## Purpose

The deliverable is Matt's analysis with one preprocessing change:

```text
original per-cell expression
        ↓
regress S.Score and G2M.Score donor-wide
        ↓
run Matt's original marker-panel, clustering, relabel, and conclusion workflow
```

This is not an audit of Matt's report. It must not introduce a new evidence framework,
new label vocabulary, new biological questions, or a historical-versus-corrected
comparison narrative.

## Sources to preserve

1. `tcell_state_labeling_evidence_matt.html` — historical narrative and method
   baseline.
2. `tcell_label_notes.txt` — questions to answer within Matt's existing structure.
3. `tcell_matt_report_expected.json` — machine-readable inventory of Matt's marker
   panels, displayed tables, state counts, and AUROCs.

The dated HTML snapshot remains provenance. The live report is not treated as
byte-immutable; the reproducible object is Matt's method.

## Invariants

The rerun must preserve:

- Matt's section sequence and section titles;
- the ProjecTILs `functional.cluster` labels as the labels being evaluated;
- Matt's versioned marker panels;
- per-cell panel score as the mean of across-cell z-scored marker expression;
- Mann–Whitney AUROC with half credit for ties;
- Matt's implemented type-panel background for exact method continuity;
- same-lineage sibling backgrounds for functional-state panels;
- Seurat clusters as independent supporting evidence;
- Matt's progression from lineage, to supported/unsupported substates, to a collapsed
  biological label set.

The report may change a conclusion only when the cycle-regressed rerun changes the
underlying evidence.

## The single analytical change

For every donor and marker gene, fit across all projected cells:

```text
log-normalized expression = intercept + beta_S × S.Score + beta_G2M × G2M.Score + residual
```

Use the residual in Matt's marker and marker-panel calculations. Preserve the raw
expression separately. Do not use binary `Phase`, a positivity cutoff, or cell
exclusion.

For cluster support, run Matt's Seurat workflow after `ScaleData(...,
vars.to.regress = c("S.Score", "G2M.Score"))`, then use the resulting clusters,
markers, and UMAP in the same supporting role as the original report. Cell-cycle genes
may still be expressed, but they must not create a `proliferating` identity or be used
to reject an exhaustion label in this artificially stimulated experiment.

## Report contract

The revised report must retain these top-level sections exactly:

1. What this document shows
2. Gene glossary
3. Step 1 — CD8 vs CD4 identity is certain
4. Step 2 — Cytotoxic / exhaustion sub-states hold up
5. Step 3 — CD4-helper sub-states (Th17, Tfh) are not supported
6. Step 4 — TPEX folds into TEX; EOMES / EM / Treg remain weak
7. Step 5 — Proliferation is a second axis, and some clusters are not T cells
8. Step 6 — What the data supports

Within that structure:

- present cycle-regressed values as the analysis result, not as a sensitivity layer;
- retain Matt's table shapes and explanatory style where possible;
- use the cycle-regressed Seurat clusters and UMAP for cluster support;
- explain in Step 5 that induced cell cycle was regressed and is not a biological
  labeling axis;
- remove `CD4 proliferating` from the final biological vocabulary;
- keep the original conclusions where adjusted evidence remains materially the same;
- do not foreground discrepancies in the historical HTML.

## Reproducibility checks

- Matt's unadjusted method reproduces all recorded AUROCs and target-cell counts.
- Barcode joins are one-to-one and preserve all projected cells.
- Residual marker values are orthogonal to S and G2/M scores.
- Changing binary `Phase` cannot change adjusted results.
- Donors remain separate.
- The rendered report's top-level section sequence matches Matt's report.
- The dedicated runner renders the report from declared outputs without invoking
  state-label, Incytr, index, or viewer producers.

## Review gate

The report is the only deliverable in this phase. Downstream label mapping,
deconvolution, Incytr, report-index, and viewer changes require separate approval.
