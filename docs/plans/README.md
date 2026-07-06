# docs/plans — index

Planning docs live here. Two bodies of work:

1. **Orchestration program** — the gated 4-wave parallel-implementation experiment (Themes A–H).
   **All four waves executed and tagged** (`orchestration-w1…w4-2026-06-25`); these docs are kept
   **live for review** as the build record. Top-level docs at this root; per-theme plans under
   `theme_<x>/`.
2. **Standalone topic plans** — viewer / analysis work that predates or runs parallel to the
   orchestration, grouped by subject under `attribution/`, `tcell/`, `fivexfad/`, `deployment/`.

Completed / superseded **standalone** plans have been moved out to
[`archive/archived_plans/standalone_done/`](../../archive/archived_plans/standalone_done/) — they
are no longer indexed here. The wave program stays in place regardless of completion.

Status legend: **active** = in-flight / approved-pending · **done (live)** = executed, kept here as
the build record · **deferred** = parked.

---

## Active work

The only in-flight effort is the **Theme-B Incytr regeneration** (KsG + PTM + backbone), plus the
standalone topic plans below.

| Doc | What |
|---|---|
| [`incytr_rerun_ksg_ptm_backbone_2026-06-29.md`](incytr_rerun_ksg_ptm_backbone_2026-06-29.md) | three-cohort Incytr re-run orchestration (overnight, operator-gated) |
| [`theme_b/backbone_incytr_track.md`](theme_b/backbone_incytr_track.md) | **authoritative** backbone-grain spec — read before touching backbone/Incytr-viewer code |
| [`theme_b/ksg_kinase_imputed_nodes.md`](theme_b/ksg_kinase_imputed_nodes.md) | KsG admission layer design |

### `attribution/`
| Doc | Status |
|---|---|
| [`standard_attribution_metric.md`](attribution/standard_attribution_metric.md) | active — the one cross-cohort metric definition (rollout gated); cited by several source files |

### `tcell/` — T-cell viewer & NSCLC detection
| Doc | Status |
|---|---|
| [`tcell_viewer_vocab_schema_2026-06-26.md`](tcell/tcell_viewer_vocab_schema_2026-06-26.md) | active — fixed vocabulary + detection schema consolidation |
| [`tcell_viewer_quality_fixes_2026-06-26.md`](tcell/tcell_viewer_quality_fixes_2026-06-26.md) | active — W1-gate quality fixes |
| [`tcell_celltype_and_state_specificity.md`](tcell/tcell_celltype_and_state_specificity.md) | active — two-column cell-type vs state metric |
| [`tcell_viewer_coherence_fixes.md`](tcell/tcell_viewer_coherence_fixes.md) | active — coherence-fix audit |
| [`nsclc_within_cohort_detection_comparison.md`](tcell/nsclc_within_cohort_detection_comparison.md) | active — NSCLC-reference vs within-cohort per-state detection |

### `fivexfad/` — 5xFAD (MouseC2) viewer
| Doc | Status |
|---|---|
| [`fivexfad_evidence_panel.md`](fivexfad/fivexfad_evidence_panel.md) | active — Evidence panel wiring (Song mirror) |

### `deployment/`
| Doc | Status |
|---|---|
| [`todo9_viewer_aws_deployment.md`](deployment/todo9_viewer_aws_deployment.md) | active — viewer AWS deployment scaling plan (options A/B) |

---

## Orchestration program (completed — kept live for review)

All four waves merged and tagged on 2026-06-25. These remain here as the design + build record.

| Doc | What |
|---|---|
| [`meta_plan.md`](meta_plan.md) | The 4-wave gated orchestration plan; phase/status tracker. **Start here.** |
| [`_contracts.md`](_contracts.md) | Cross-theme contracts (C2, C1, B3, B5, F1, F2). B3/B5 remain the convention the active Incytr work cites. |
| [`p4_dag.md`](p4_dag.md) | Dependency DAG + git/worktree topology; wave assignment. |
| [`TODO.md`](TODO.md) | Living theme backlog (A–H); open items remain (B1, B2, C4, E1/E2, G1, H1). |

| Theme dir | Subject | Key docs |
|---|---|---|
| [`theme_a/`](theme_a/) | T-cell cohort (theme A) | `plan.md`, `audit.md` |
| [`theme_b/`](theme_b/) | Incytr backbone grain + kinase→pathway — **active** (see above) | `backbone_incytr_track.md` (authoritative), `b4_plan.md`, `ksg_kinase_imputed_nodes.md`, `b2_plan.md` (deferred), `incytr_viewer_schema.md` (historical baseline) |
| [`theme_c/`](theme_c/) | Cohort naming · genotype split · disease-direction · cross-species | `c1`/`c2`/`c3`/`c5` plans + audits, `kinase_trend_refactor.md` |
| [`theme_d/`](theme_d/) | Substrate comparator engine | `d1_plan.md` |
| [`theme_f/`](theme_f/) | Signed-sort + CSV-export sweep | `f1_plan.md`, `f2_plan.md`, `f_audit.md` |
| [`theme_g/`](theme_g/) | Positive controls (check-controls skill) | `g2_plan.md` |
