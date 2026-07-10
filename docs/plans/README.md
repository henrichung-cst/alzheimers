# docs/plans — index

This folder holds **plans**: forward-looking specs for work that is open or in flight.
Implemented specs live in [`../foundation/`](../foundation/); completed-program records and
shipped plans live under [`../../archive/archived_plans/`](../../archive/archived_plans/).

Status legend: **active** = in-flight / approved-pending · **deferred** = parked.

---

## In flight

| Doc | Status | What |
|---|---|---|
| [`kinase_sidechain_incytr_graph/`](kinase_sidechain_incytr_graph/) | active | one-spine + kinase-sidechain cytoscape tab — `/orchestrate`-decomposed subplan set (`_index.md` + 01–04); all four parts unbuilt |
| [`ad_geneuse_unpin_from_sce4.md`](ad_geneuse_unpin_from_sce4.md) | open decision | un-pin AD Incytr `gene.use` from sce4's frozen node sets to derived DEG∪prG — untouched, decision pending |
| [`tcell-proteome-transcriptome-correlation.md`](tcell-proteome-transcriptome-correlation.md) | active | genome-wide donor2/day2 bulk-protein vs pseudobulk-transcript Spearman — standalone CSV, unbuilt |
| [`tcell-matt-report-restoration.md`](tcell-matt-report-restoration.md) | active WIP | Matt cluster-relabel labeling (Seurat clusters → small biological set) as canonical; ProjecTILs permanently retired. User-owned; gates the t-cell Incytr re-run |
| [`deployment/todo9_viewer_aws_deployment.md`](deployment/todo9_viewer_aws_deployment.md) | partial | viewer AWS deployment scaling — Option A (`deploy_viewer.sh` s3 sync) shipped; B/C/D open |

## Backlog

Candidate work, one file per item. Delete a file when it ships — git holds the history.

| Doc | Theme | What |
|---|---|---|
| [`tcell-incytr-trend-timepoint-coverage.md`](tcell-incytr-trend-timepoint-coverage.md) | T-cell | verify Incytr trends cover all 3 timepoints |
| [`tcell-data-structure-verification.md`](tcell-data-structure-verification.md) | T-cell | confirm donor1/donor2 data shape matches the pipeline's assumptions |
| [`incytrdb-provenance-audit.md`](incytrdb-provenance-audit.md) | Incytr | IncytrDB source, freshness, correct mouse/human version per dataset |
| [`incytr-sankey-diagram.md`](incytr-sankey-diagram.md) | Incytr | add sankey view + collapse excitatory clusters (chord already ships) |
| [`disease-direction-site-level-early-change.md`](disease-direction-site-level-early-change.md) | Cross-cohort | Disease Direction Stage 4 — site-level early-change phosphosites |
| [`cross-species-specificity-guard.md`](cross-species-specificity-guard.md) | Cross-cohort | review constraint: no single-mouse-celltype target that is broad in human |
| [`kinase-regulation-network.md`](kinase-regulation-network.md) | Kinase hierarchy | PhosphoSite-based regulation network + observed disease overlay (precedes family discrimination) |
| [`kinase-family-discrimination.md`](kinase-family-discrimination.md) | Kinase hierarchy | separate same-family kinases by cell-type specificity (needs the network first) |
| [`methods-workflow-diagrams.md`](methods-workflow-diagrams.md) | Docs | workflow flowcharts for AD + T-cell pipelines |
| [`positive-controls-list.md`](positive-controls-list.md) | Docs | curated per-cohort positive-control artifact |
| [`tmt-paper-imac-replication.md`](tmt-paper-imac-replication.md) | External | fetch TMT-paper IMAC data, compare kinase enrichment (exploratory) |
| [`never-read-sidecar-cleanup.md`](never-read-sidecar-cleanup.md) | Refactor carryover | stop emitting `_raw`/`_all`/`_per_cluster` sidecars |
| [`cohort-namespace-frozen-layer-moves.md`](cohort-namespace-frozen-layer-moves.md) | Refactor carryover | migrate held-back frozen Incytr/decomposition paths into `alz/cohorts/*` |

Current-state snapshot cross-referencing every plan against commits: [`plans_state_audit_2026-07-10.md`](plans_state_audit_2026-07-10.md). Sequencing, cross-cutting threads, and blocking adjudications: [`implementation_sequencing.md`](implementation_sequencing.md). The T-cell labeling direction is resolved — canonical = Matt cluster-relabel, ProjecTILs permanently retired.

---

## Where the rest went

- **Orchestration program (Themes A–H)** — all four waves merged and tagged 2026-06-25. The
  program index (`meta_plan.md` / `_contracts.md` / `p4_dag.md`) and the per-theme build record
  are archived under
  [`../../archive/archived_plans/orchestration/`](../../archive/archived_plans/orchestration/).
  Its live contracts were promoted: cohort-rename (C2) → `foundation/cohort_contract.md §2.5`,
  kinase-trend vocabulary → `foundation/kinase_explorer_attribution.md`, backbone-grain spec →
  `foundation/backbone_incytr_track.md`.
- **`standard_attribution_metric.md`** — the implemented cross-cohort metric spec (cited by path
  in 5 source files); moved to [`../foundation/standard_attribution_metric.md`](../foundation/standard_attribution_metric.md).
- **Completed standalone plans** — moved to
  [`../../archive/archived_plans/standalone_done/`](../../archive/archived_plans/standalone_done/).
