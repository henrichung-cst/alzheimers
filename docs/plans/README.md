# docs/plans — index

This folder holds **plans**: forward-looking specs for work that is open or in flight.
Implemented specs live in [`../foundation/`](../foundation/); completed-program records and
shipped plans live under [`../../archive/archived_plans/`](../../archive/archived_plans/).

Status legend: **active** = in-flight / approved-pending · **deferred** = parked.

---

## In flight

| Doc | Status | What |
|---|---|---|
| [`ad_geneuse_unpin_from_sce4.md`](ad_geneuse_unpin_from_sce4.md) | active | un-pin AD Incytr `gene.use` from sce4's frozen node sets to derived DEG∪prG — decided (un-pin); needs AD re-run + viewer rebuild |
| [`tcell-matt-report-restoration.md`](tcell-matt-report-restoration.md) | active | per-cell marker assignment is the labeling standard (by-cluster rejected as day-confounded; ProjecTILs corroboration-only). Labels verified current 2026-07-16 — no longer gates the t-cell Incytr re-run |
| [`deployment/todo9_viewer_aws_deployment.md`](deployment/todo9_viewer_aws_deployment.md) | partial | viewer AWS deployment scaling — Option A (`deploy_viewer.sh` s3 sync) shipped; B/C/D open |

## Backlog

Candidate work, one file per item. Delete a file when it ships — git holds the history.

| Doc | Theme | What |
|---|---|---|
| [`tcell_apriori_expectations.md`](tcell_apriori_expectations.md) | T-cell | blind literature-grounded exhaustion prediction set → `docs/reference/tcell_apriori_expectations.md` (unbuilt) |
| [`incytr-sankey-diagram.md`](incytr-sankey-diagram.md) | Incytr | add sankey view + collapse excitatory clusters (chord already ships) |
| [`disease-direction-site-level-early-change.md`](disease-direction-site-level-early-change.md) | Cross-cohort | Disease Direction Stage 4 — site-level early-change phosphosites |
| [`kinase-regulation-network.md`](kinase-regulation-network.md) | Kinase hierarchy | PhosphoSite-based regulation network + observed disease overlay (precedes family discrimination) |
| [`kinase-family-discrimination.md`](kinase-family-discrimination.md) | Kinase hierarchy | separate same-family kinases by cell-type specificity (needs the network first) |
| [`methods-workflow-diagrams.md`](methods-workflow-diagrams.md) | Docs | workflow flowcharts for AD + T-cell pipelines |
| [`tmt-paper-imac-replication.md`](tmt-paper-imac-replication.md) | External | fetch TMT-paper IMAC data, compare kinase enrichment (exploratory) |
| [`cohort-namespace-frozen-layer-moves.md`](cohort-namespace-frozen-layer-moves.md) | Refactor carryover | migrate held-back frozen Incytr/decomposition paths into `alz/cohorts/*` |

Sequencing, cross-cutting threads, and blocking adjudications: [`implementation_sequencing.md`](implementation_sequencing.md). The T-cell labeling direction is resolved — canonical = per-cell marker assignment, ProjecTILs corroboration-only.

---

## Where the rest went

- **Orchestration program (Themes A–H)** — all four waves merged and tagged 2026-06-25. The
  program index (`meta_plan.md` / `_contracts.md` / `p4_dag.md`) and the per-theme build record
  are archived under
  [`../../archive/archived_plans/orchestration/`](../../archive/archived_plans/orchestration/).
  Its live contracts were promoted: cohort-rename (C2) → `foundation/cohort_contract.md §2.5`,
  kinase-trend vocabulary → `foundation/kinase_explorer_attribution.md`, backbone-grain spec →
  `foundation/backbone_incytr_track.md`.
- **Kinase sidechain Incytr graph** — shipped 2026-07-17 (subplans 01–07). Architecture promoted to
  [`../foundation/kinase_sidechain_incytr_graph.md`](../foundation/kinase_sidechain_incytr_graph.md);
  the subplan set is archived under
  [`../../archive/archived_plans/kinase_sidechain_incytr_graph/`](../../archive/archived_plans/kinase_sidechain_incytr_graph/).
- **`standard_attribution_metric.md`** — the implemented cross-cohort metric spec (cited by path
  in 5 source files); moved to [`../foundation/standard_attribution_metric.md`](../foundation/standard_attribution_metric.md).
- **Completed standalone plans** — moved to
  [`../../archive/archived_plans/standalone_done/`](../../archive/archived_plans/standalone_done/).
