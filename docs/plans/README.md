# docs/plans — index

Planning docs live here. Two bodies of work:

1. **Orchestration program** — the gated 4-wave parallel-implementation experiment (Themes A–H).
   Top-level docs at this root; per-theme plans under `theme_<x>/`.
2. **Standalone topic plans** — viewer / analysis work that predates or runs parallel to the
   orchestration, grouped by subject under `attribution/`, `tcell/`, `fivexfad/`, `deployment/`.

Status legend: **active** = being worked / approved-pending · **done** = implemented, kept as the
build record · **deferred** = parked · **superseded** = a newer doc in the same dir replaces it.

---

## Orchestration program (root + `theme_<x>/`)

| Doc | What |
|---|---|
| [`meta_plan.md`](meta_plan.md) | The 4-wave gated orchestration plan; phase/status tracker. **Start here.** |
| [`_contracts.md`](_contracts.md) | Cross-theme contracts every theme cites (C2, C1, B3, B5, F1, F2). |
| [`p4_dag.md`](p4_dag.md) | Dependency DAG + git/worktree topology; wave assignment. |

| Theme dir | Subject | Key docs |
|---|---|---|
| [`theme_a/`](theme_a/) | T-cell cohort (orchestration theme A) | `plan.md`, `audit.md` |
| [`theme_b/`](theme_b/) | Incytr backbone grain + kinase→pathway | **[`backbone_incytr_track.md`](theme_b/backbone_incytr_track.md) (authoritative)**, `b4_plan.md`, `b2_plan.md` (deferred), `incytr_viewer_schema.md` (historical baseline) |
| [`theme_c/`](theme_c/) | Cohort naming · genotype split · disease-direction · cross-species | `c1`/`c2`/`c3`/`c5` plans + audits |
| [`theme_d/`](theme_d/) | Substrate comparator engine | `d1_plan.md` |
| [`theme_f/`](theme_f/) | Signed-sort + CSV-export sweep | `f1_plan.md`, `f2_plan.md`, `f_audit.md` |
| [`theme_g/`](theme_g/) | Positive controls (check-controls skill) | `g2_plan.md` |

---

## Standalone topic plans

### `attribution/` — cell-type attribution & specificity
| Doc | Status |
|---|---|
| [`standard_attribution_metric.md`](attribution/standard_attribution_metric.md) | active — the one cross-cohort metric definition (confirmed 2026-06-20, rollout gated); cited by several source files |
| [`attribution_specificity_audit.md`](attribution/attribution_specificity_audit.md) | done — the audit `standard_attribution_metric` supersedes |
| [`attribution_view_consolidation.md`](attribution/attribution_view_consolidation.md) | done — Song renderer consolidation (S0–S4) |
| [`attribution_drawer_redesign.md`](attribution/attribution_drawer_redesign.md) | done — implementation history; live SSOT is `docs/foundation/kinase_explorer_attribution.md` |
| [`cross_reference_exclusivity_regrouping.md`](attribution/cross_reference_exclusivity_regrouping.md) | done — confidence-pill recalculation |

### `tcell/` — T-cell viewer & NSCLC detection
| Doc | Status |
|---|---|
| [`tcell_viewer_vocab_schema_2026-06-26.md`](tcell/tcell_viewer_vocab_schema_2026-06-26.md) | active — fixed vocabulary + detection schema consolidation |
| [`tcell_viewer_quality_fixes_2026-06-26.md`](tcell/tcell_viewer_quality_fixes_2026-06-26.md) | active — W1-gate quality fixes |
| [`tcell_celltype_and_state_specificity.md`](tcell/tcell_celltype_and_state_specificity.md) | active — two-column cell-type vs state metric |
| [`tcell_cell_state_vs_cell_type.md`](tcell/tcell_cell_state_vs_cell_type.md) | superseded by `tcell_celltype_and_state_specificity.md` |
| [`tcell_viewer_coherence_fixes.md`](tcell/tcell_viewer_coherence_fixes.md) | active — coherence-fix audit |
| [`nsclc_within_cohort_detection_comparison.md`](tcell/nsclc_within_cohort_detection_comparison.md) | active — NSCLC-reference vs within-cohort per-state detection |

### `fivexfad/` — 5xFAD (MouseC2) viewer
| Doc | Status |
|---|---|
| [`fivexfad_detection_metric_migration.md`](fivexfad/fivexfad_detection_metric_migration.md) | done 2026-06-22 — standard-detection-metric migration (Phase 3) |
| [`fivexfad_evidence_panel.md`](fivexfad/fivexfad_evidence_panel.md) | active — Evidence panel wiring (Song mirror) |

### `deployment/`
| Doc | Status |
|---|---|
| [`todo9_viewer_aws_deployment.md`](deployment/todo9_viewer_aws_deployment.md) | active — viewer AWS deployment scaling plan (options A/B) |
