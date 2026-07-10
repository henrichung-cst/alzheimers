# docs/plans — state audit (2026-07-10)

Branch `feat/incytr-backbone-refactor`. Each plan cross-referenced against commits + on-disk files. Verdicts: DONE / PARTIAL / NOT-STARTED / SUPERSEDED / OPEN-DECISION.

## Bottom line

- **README index is the most stale artifact.** 7 `tcell-*.md` plans + `foundation_docs_audit_2026-07-08.md` + the `tcell-state-relabeling-and-correlation/` subdir are all absent from the index table, despite being the newest work on the branch.
- **4 T-cell plans are DONE and belong in the archive**, not `docs/plans/`.
- **Live tension in the T-cell cluster:** committed HEAD ships the per-cell (`..._percell`) labeling that plans #3/#8 delivered; the uncommitted working tree re-points everything to a plan-less `..._markertypes` method; and plan #1 (Matt cycle-regression) proposes unwinding the per-cell relabel entirely. No committed plan documents the markertypes direction. **This is a decision to make, not a doc fix.**
- **Two committed plans need a status refresh** (`incytr_rerun_...` internal status table is stale; deployment Option A has shipped).

---

## Per-plan verdicts

### T-cell cluster

| Plan | Verdict | Evidence |
|---|---|---|
| `tcell-relabel-incytr-rerun.md` | **DONE** | 31e61ca, f50e14d, a7d53d2. `tcell_state_labels.py`, `cells/*_state_labels.csv`, `state_audit.json`, re-keyed R extractors all present. → **archive** |
| `tcell-report-folder-consolidation.md` | **DONE** | `outputs/reports/` collapsed to `tcell_labeling/` + `tcell_viewer/`; retired folders under `archive/tcell_report_consolidation_20260709/`. → **archive** |
| `tcell-viewer-percell-state-update.md` | **DONE** (committed) | f12336c. `paths.py` default `..._percell`, low-signal keys stripped, roster validated against audit. WIP re-points to `..._markertypes` (see tension). → **archive** |
| `tcell-topgene-state-validation.md` | **DONE → superseded** | Outputs exist but archived; approach retired by the consolidation plan. → **archive/delete** |
| `tcell-percell-axis-labeling.md` | **SUPERSEDED** | Continuous-axis prototype (`..._axes/` tree) lost to #3's categorical labels; not adopted. → **archive/delete** |
| `tcell-matt-report-cycle-regression-revision.md` | **NOT-STARTED** | Self-labeled "Planning only"; only §1 freeze snapshots exist. **Proposes reverting the per-cell direction #3/#8 shipped.** → keep, needs decision |
| `tcell-state-relabeling-and-correlation.md` + `/02_...md` | **NOT-STARTED** | No `tcell_proteome_transcriptome_correlation.py`, no output CSV. Two files = one deliverable (parent + detailed twin). → keep, consolidate to one |

**Undocumented WIP:** working tree has a 225-line rewrite of `tcell_state_labels.py`, deleted evidence `.qmd`, and viewer/runner re-pointed to `outputs/reports/incytr_pair_mode_tcells_markertypes/`. No plan covers this.

### Incytr / backbone / deployment

| Plan | Verdict | Evidence |
|---|---|---|
| `incytr_rerun_ksg_ptm_backbone_2026-06-29.md` | **DONE (compute+wiring); doc STALE** | KsG (build_ksg_attribution.py), backbone all cohorts/grains on disk, payload ported to fivexfad.py + tcell viewer (ee66ba4, 790b203). Doc's status table + "Phase 2 DEFERRED" are wrong. PTM design changed: `derive_phospho_from_ptm.py` deleted (416fb2c); 5xFAD reads `ACK_FILE`/`KGG_FILE` deconvoluted directly — no `wide_ptm/`. → **refresh status then archive** |
| `deployment/todo9_viewer_aws_deployment.md` | **PARTIAL** | Option A shipped: `alz/runners/supporting/deploy_viewer.sh` (3-pass `s3 sync` + cache tiers), pixi `deploy-*-viewer`. Option B (shard manifest) NOT-STARTED; C/D not started. → keep, mark A done |
| `ad_geneuse_unpin_from_sce4.md` | **OPEN-DECISION** | Untouched: `SCE4_GENEUSE_DIR` export, `extract_sce4_geneuse.R`, `use_frozen_geneuse` branch all live. → keep |
| `kinase_sidechain_incytr_graph/` (01–04) | **NOT-STARTED (all 4)** | No `kinase_kinase_edges.py`; bridge `--cohort` still `song/fivexfad/all`; no sidechain/interactome refs in viewers; no `incytr_sidechains.js`. Note: plan's `alz/viewer_shared/` path is wrong (actual: `alz/viewer/template/js/`). → keep, fix path assumption |

### Meta docs

| Doc | Verdict | Action |
|---|---|---|
| `foundation_docs_audit_2026-07-08.md` | **DONE** — all C1 + D1–D23 applied (544dfde/907f8e9) | archive |
| `README.md` (index) | **STALE** — see below | rewrite index |
| `TODO.md` | living, many items now DONE-unmarked | see below |

---

## TODO.md theme status (commit-verified)

- **DONE (unmarked):** A1, A2, A5, B3, B4, B5, C1, C2, C3, C5, D1, F1, F2
- **PARTIAL:** B2 (chord done, no sankey/EN-collapse), G1 (terminology rewrite done, no diagrams), G2 (`check-controls` skill done, no curated list), C3 (Stage 4 site-level only queued), I2 (bulk migrated, frozen-layer paths held back)
- **NOT-STARTED:** A3, A4, B1, C4, E1, E2, H1, I1

---

## Recommended updates (awaiting go)

1. **Rewrite `README.md` index** — add the missing tcell cluster + foundation audit; correct statuses (incytr_rerun → done-on-branch, deployment → partial/A-shipped).
2. **Archive DONE plans** to `archive/archived_plans/` : `tcell-relabel-incytr-rerun`, `tcell-report-folder-consolidation`, `tcell-viewer-percell-state-update`, `tcell-topgene-state-validation`, `foundation_docs_audit_2026-07-08`, and `incytr_rerun_ksg_ptm_backbone` (after status refresh).
3. **Delete superseded:** `tcell-percell-axis-labeling.md` (git holds it).
4. **Consolidate** the two `tcell-state-relabeling-and-correlation` files into one open plan.
5. **Mark TODO.md** DONE items with `[DONE]` / move to its archive section; leave PARTIAL/NOT-STARTED.
6. **Decide the T-cell labeling direction** (percell vs markertypes vs Matt cycle-regression) — a real fork, not a doc edit. The three plans (#1, committed #3/#8, undocumented WIP) are mutually exclusive.
