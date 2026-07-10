# Reproduce Matt's T-cell report and remove induced cell-cycle signal

## Status

Implemented on 2026-07-10; the revised evidence report is awaiting scientific review.
Do not relabel cells, rerun Incytr, update the viewer, or change the downstream state
spine until that review is complete.

## Authoritative documents

1. `outputs/reports/tcell_labeling/tcell_state_labeling_evidence_matt.html` — the
   original report and narrative/method baseline. A dated snapshot preserves the
   historical version; the live report is not treated as byte-immutable.
2. `tcell_label_notes.txt` — the scientific review questions the revision must answer.
3. This plan — the implementation contract for a reproducible revision.

The divergent winning-marker-panel report is archived under
`outputs/reports/tcell_labeling/archive/divergent_marker_classifier_2026-07-10/`
and is out of scope.

## Decision

Reproduce Matt's original method before modifying it:

- retain the original per-cell ProjecTILs `functional.cluster` labels as the labels
  being evaluated;
- calculate per-cell marker and marker-panel AUROC with Matt's original marker sets;
- use opposite-lineage cells for CD4/CD8 type questions and same-lineage sibling
  states for functional-substate questions;
- retain Seurat clustering as independent supporting evidence, not as a mechanism
  that forces every cell in a cluster into one state;
- do not introduce a winning-marker-panel classifier or a new state vocabulary.

Then add one controlled extension: remove the experimentally induced cell-cycle
component from marker expression and rerun the *same* AUROC questions. Cell cycle is
a nuisance signal in this experiment; it must not be used to accept, reject, merge,
or rename a T-cell state.

## What “cell-cycle regression removal” means

The marker extractor already produces a compact cells × marker-genes matrix from the
Seurat log-normalized RNA `data` layer. Join it by barcode to the existing Seurat
`S.Score` and `G2M.Score` values. For each donor and marker gene, fit the donor-wide
ordinary least-squares model:

```text
log-normalized expression = intercept + beta_S × S.Score + beta_G2M × G2M.Score + residual
```

Use the residual as the cycle-regressed marker value. Fit across the donor, not within
a ProjecTILs state, so the target label cannot influence regression. Do not use the
binary `Phase` call, a positivity cutoff, or a cell-exclusion rule. Adding a constant
back to residuals is unnecessary because AUROC is rank-based and panel genes are
z-scored before averaging.

The compact regression is preferred over reading Seurat `scale.data`: it reproduces
the intended nuisance removal without materializing a multi-gigabyte dense matrix.
Regression coefficients are implementation diagnostics, not biological scores and
must not appear as viewer labels or exported analysis scores.

## Work plan

### 1. Freeze and inventory the original

- Record the historical report SHA-256 as provenance rather than a runtime gate.
- Keep the dated archive snapshot in
  `outputs/reports/tcell_labeling/archive/matt_original_snapshot_2026-07-10/`.
- Extract the report's displayed marker panels, target states, comparison backgrounds,
  AUROC values, tables, and conclusions into a machine-readable reproduction fixture.
- Record the exact input paths and row counts used by the original analysis.

### 2. Restore the unadjusted Matt pipeline

- Restore a single marker-set source matching the report.
- Restore the marker extractor for log-normalized RNA, joined strictly by barcode.
- Restore per-marker AUROC and panel-score AUROC:
  - per-cell panel score = mean of marker expression z-scored across cells;
  - type panels = target lineage versus opposite lineage;
  - state panels = target state versus same-lineage sibling states;
  - AUROC = Mann–Whitney U divided by `n_target × n_comparison`, with half credit
    for ties.
- Emit unadjusted results into a new `reproduced_unadjusted/` directory; never
  overwrite the historical files used by the original report.
- Compare reproduced values with the values extracted from the original HTML.
  Any mismatch must be explained before cycle regression is added.

### 3. Add cycle-regressed marker evidence

- Implement the donor-wise gene-level regression defined above as a pure, tested
  transformation of the compact marker matrix.
- Preserve raw log-normalized marker values alongside residualized values with explicit
  units/names; never replace raw evidence silently.
- Recalculate the same per-marker and per-panel AUROC tables with identical labels and
  backgrounds.
- Emit results into `cycle_regressed/` with unambiguous `unadjusted_auroc`,
  `cycle_regressed_auroc`, and `auroc_difference` columns.
- Do not create a thresholded “cycle-corrected confidence” score.

### 4. Address `tcell_label_notes.txt` without changing the backbone

- **TIM-3/TOX in CD8 exhausted:** show unadjusted and cycle-regressed per-gene AUROC,
  detection fraction, and raw expression distribution. State whether their absence is
  unchanged after nuisance removal.
- **Cytotoxic versus exhausted:** keep them as separate biological questions. Do not
  use the former `CD8 cytotoxic/exhausted` combined label in the revised conclusion.
  Quantify overlap of the two continuous panels without turning overlap into a new type.
- **Cycling and exhaustion:** remove “cycling therefore not exhausted” reasoning.
  Report only whether marker separation remains after cycle regression.
- **CD4 resting/naive terminology:** do not retain this non-standard final label without
  marker support. Report the evaluated reference state and its naive/memory evidence.
- **Negative markers:** after exact reproduction, add a clearly labeled sensitivity
  analysis using predefined loss-of-memory markers for exhaustion. Keep original
  unsigned AUROC as the primary reproduction result.
- **CD8 separation:** show the complete adjusted per-marker and panel AUROC rather than
  inventing a stronger categorical call.
- **TPEX versus TEX:** directly compare TPEX with TEX on TCF7, LEF1, SELL, CCR7, and IL7R
  before deciding whether the states should merge.
- **Cluster forcing:** keep all primary tests per cell. Use Seurat or cycle-regressed
  clusters only as an independent visualization of heterogeneity.

### 5. Build a separate revised report

- Reconstruct Matt's report source as
  `tcell_state_labeling_evidence_cycle_regressed.qmd` and render
  `tcell_state_labeling_evidence_cycle_regressed.html`.
- Preserve Matt's section order, gene glossary, tables, and explanatory style wherever
  the underlying result is unchanged.
- Clearly distinguish three evidence layers:
  1. original unadjusted Matt result;
  2. cycle-regressed result using the same question;
  3. explicitly labeled sensitivity analyses requested in the notes.
- Replace the original proliferation section with a nuisance-removal methods/QC section.
- Build the revision as a separate report so the historical method and the new
  extension remain easy to compare.

### 6. Review gate before downstream work

Present the revised report and an explicit proposed label mapping for scientific review.
Only after approval may another task re-key deconvolution, Incytr, the report index, or
the viewer. The current plan does not authorize those changes.

## Tests and reproducibility checks

- Barcode joins are one-to-one and preserve all original projected cells.
- A fixed Mann–Whitney fixture verifies AUROC and tie handling.
- Original unadjusted report values reproduce from scripts and declared inputs.
- Regressed marker residuals are numerically orthogonal to `S.Score` and `G2M.Score`
  when the design matrix has full rank.
- Shuffling or changing the binary `Phase` column cannot change adjusted results.
- AUROC target/background counts are emitted with every result.
- Donors are processed and reported separately.
- The dated original snapshot remains recoverable, while analytical reproduction is
  verified from the method fixture and declared inputs.
- The revised QMD renders from a clean process using only declared outputs.

## Acceptance criteria

- Matt's original method is reproducible, and the dated historical report remains
  recoverable from its snapshot.
- The unadjusted method is fully reproducible from versioned scripts.
- Cycle regression is a single isolated preprocessing step; all downstream AUROC logic
  is shared with the unadjusted analysis.
- No state is rejected because it cycles, and no state is assigned from cell cycle.
- Every question in `tcell_label_notes.txt` is answered with per-cell evidence.
- No downstream state spine changes before explicit report approval.

## Explicitly out of scope

- A winning-marker-panel classifier.
- New definitive labels or score cutoffs.
- Incytr/deconvolution reruns.
- Viewer/index updates.
- Functional or terminal-exhaustion claims unsupported by the assay.
