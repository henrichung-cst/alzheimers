# docs/plans — index

This folder holds **plans**: forward-looking specs for work that is open or in flight.
Implemented specs live in [`../foundation/`](../foundation/); completed-program records and
shipped plans live under [`../../archive/archived_plans/`](../../archive/archived_plans/).

Status legend: **active** = in-flight / approved-pending · **deferred** = parked.

---

## Plans

| Doc | Status | What |
|---|---|---|
| [`incytr_rerun_ksg_ptm_backbone_2026-06-29.md`](incytr_rerun_ksg_ptm_backbone_2026-06-29.md) | in progress | three-cohort Incytr re-run (KsG + Ack/KGG PTM + backbone) on branch `feat/incytr-backbone-refactor` |
| [`kinase_sidechain_incytr_graph/`](kinase_sidechain_incytr_graph/) | active | one-spine + kinase-sidechain cytoscape tab — `/orchestrate`-decomposed subplan set (`_index.md` + 01–04); backend edge-model + viewer tab unbuilt |
| [`ad_geneuse_unpin_from_sce4.md`](ad_geneuse_unpin_from_sce4.md) | active | open decision — un-pin AD Incytr `gene.use` from sce4's frozen node sets to derived DEG∪prG |
| [`deployment/todo9_viewer_aws_deployment.md`](deployment/todo9_viewer_aws_deployment.md) | active | viewer AWS deployment scaling plan (options A/B) |
| [`TODO.md`](TODO.md) | living | theme backlog (A–H) — the master list of candidate work |

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
