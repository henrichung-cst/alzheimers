# T-cell labeling report workspace

The active scientific review is intentionally limited to three documents:

1. [Original Matt report](tcell_state_labeling_evidence_matt.html) — historical
   evidence, narrative, and method baseline; the dated archive preserves its original
   version without treating the live report as byte-immutable.
2. [`tcell_label_notes.txt`](../../../tcell_label_notes.txt) — reviewer questions.
3. [Revision plan](../../../docs/plans/tcell-matt-report-cycle-regression-revision.md)
   — reproduce Matt's method, then remove induced cell-cycle signal before rerunning
   the same per-cell AUROC questions.

The original report has a checksum-verified backup under
`archive/matt_original_snapshot_2026-07-10/`. Superseded exploratory report work is
under `archive/divergent_marker_classifier_2026-07-10/` and is not authoritative.

The current deliverable is
[`tcell_state_labeling_evidence_cycle_regressed.html`](tcell_state_labeling_evidence_cycle_regressed.html):
Matt's analysis rerun after donor-wise S/G2M regression. It preserves Matt's section
order, marker questions, cluster/UMAP support, and conclusion flow; it is not an audit
or replacement labeling framework. Rebuild it with `pixi run tcells-matt-cycle-report`.

No downstream label, deconvolution, Incytr, index, or viewer change should proceed
until the revised report is reviewed.
