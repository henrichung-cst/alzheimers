# Sce4 fix propagation to canonical pipeline (2026-05-23)

## Context

The six-fix chain that reproduces sce4's `Top300_table` (599/600 Ndnf×Ndnf, 600/600 Microglia→Cholinergic, max `|Δ sclog2FC|` = 0 on R/E/T) was developed in `bench/regen/`. Most fixes are already in the canonical driver (`alz/incytr_pair/incytr_commandline.R`), but:

1. The canonical pipeline has never been **end-to-end verified** against sce4.
2. Downstream integration code (`alz/integration/`) hardcodes Cal_scFC's old epsilon (`1e-5`) — fix #6 broke that assumption.
3. Comments in `emit_expr_bygroup.R` and `build_transcript_trace.py` still reference `mean_method="mean"` (pre-fix #3).
4. The bench drivers (`run_pair_posstrict.R`, `run_single_pair*.R`) are duplicates of production logic with the same six fixes applied. Per anti-shim, the duplicates should be deleted or formally repurposed as the regression harness.

Goal: ensure that `bash alz/runners/main/run_pair_mode_pipeline.sh` on the Song dataset (or any new dataset following the README workflow) produces sce4-equivalent output, with the parity locked into a CI-style check.

## Audit summary — fix presence in canonical code

| # | Fix | Canonical location | Status |
|---|---|---|---|
| 1 | DG no-cap | `alz/incytr_pair/build_input_gene_list.R` | ✅ already no cap (uses `avg_log2FC > 1.5 & p_val < 1e-4 ∪ HEG`) |
| 2 | `pmax(pr_*, 1)` floor | `alz/incytr_pair/incytr_commandline.R:261-267` | ✅ in |
| 3 | `Expr_bygroup(mean_method = NULL)` (trimean) | `incytr_commandline.R:382` | ✅ in |
| 4 | `Cal_SigProb(correction = 0.01)` | `incytr_commandline.R:389` | ✅ in |
| 5 | `cutoff_SigProb = 0, cutoff_PDS = 0` | `incytr_commandline.R:213-214,388,414` | ✅ in |
| 6 | `Cal_scFC(correction = 0.01)` | `incytr_commandline.R:435` | ✅ in (applied 2026-05-23) |

All six fixes are in the canonical driver. The remaining work is **propagation, verification, documentation, and consolidation**.

## Plan

### Phase 1 — Downstream epsilon sync (blocking the canonical run)

`alz/integration/verify_pathway_round_trip.py` recomputes `_sclog2FC` from transcript expression and checks float16-round-trip equality to the driver's output. It assumes Cal_scFC ε = `1e-5`. After fix #6 it must use ε = `0.01` or every transcript-trace verification fails.

- [ ] `verify_pathway_round_trip.py:92` — `EPSILON_SC = 1e-5` → `EPSILON_SC = 0.01`. Update the inline comment.
- [ ] `verify_pathway_round_trip.py:30` — docstring "Epsilon = 1e-5 matches Cal_scFC default" → "ε = 0.01 matches `incytr_commandline.R:435` `Cal_scFC(correction = 0.01)`".
- [ ] `build_normalized_substrate.py:470` — docstring "Transcript uses a separate epsilon = 1e-5 (Cal_scFC default)" → "ε = 0.01 (`Cal_scFC(correction = 0.01)` after sce4 fix #6)".
- [ ] `build_transcript_trace.py:209-210` and `emit_expr_bygroup.R:8` — comments referencing `Expr_bygroup(..., mean_method = "mean")` are now stale (driver uses `NULL` = trimean). Update.

**Verification step within Phase 1:** read `emit_expr_bygroup.R` end-to-end. If its emitted matrix is used downstream as a viewer input that assumes arithmetic mean, the matrix itself needs to be regenerated with trimean. If it only emits raw data the viewer doesn't depend on, just fix the comments.

**Acceptance:** `pixi run python alz/integration/verify_pathway_round_trip.py` on existing canonical Incytr outputs runs to completion with no transcript-trace failures.

### Phase 2 — Canonical end-to-end run + sce4 parity verification

The point of this plan: prove that `run_pair_mode_pipeline.sh` produces a Microglia→Cholinergic block matching sce4. Until that's verified, "the fixes are in production" is a claim about code, not behavior.

- [ ] **Force a clean rebuild** of `data/derived/incytr_inputs/input_gene_list.csv` from the Song Seurat (`build_pair_seurat.R` → `build_input_gene_list.R`). Old `input_gene_list.csv` may have been built under the top-500-cap version of fix #1. Compare row counts pre/post.
- [ ] **Run the canonical pipeline** for the `App_2mo` contrast only first (cheap subset): `bash alz/incytr_pair/run_pair_mode.sh` after manually filtering to one contrast. Confirm exit code 0, no `correction` warnings in log.
- [ ] **Extract Microglia → Cholinergic.Neurons block** from the canonical parquet output (`outputs/reports/incytr_pair_mode/wide/ma_2mo_AppP_ma_2mo_WTyp_incytr_output.parquet`).
- [ ] **Compare to `bench/sce4_DEG_PRG_Top300_table_10302025.csv`** using the same path-key join logic as `bench/regen/check_scfc_fix.R`. Acceptance bar: 600/600 paths recovered after SigProb filter, max `|Δ sclog2FC|` = 0.0000 on R/E/T positions. App-as-Ligand residual logged separately.
- [ ] **Run the full 9 contrasts** if the App_2mo check passes.

**Acceptance:** a new artifact `outputs/reports/incytr_pair_mode/sce4_parity_report.csv` (or equivalent) shows per-pair recall and per-position `max |Δ|` for at least the two cross-validated pairs.

### Phase 3 — Lock the regression with a pixi task

Per project memory `feedback_no_intentional_wrong_behavior` and the project's "every divergence is a defect" framing, a future change to any of the six fix sites must trigger a visible failure — not silently regress to the 12/600 baseline.

- [ ] Move `bench/regen/check_scfc_fix.R` to `alz/incytr_pair/verify_sce4_parity.R`, generalized to take a `--contrast`, `--sender`, `--receiver` and read the canonical parquet output.
- [ ] Add a pixi task `verify-incytr-sce4` that runs the parity check against the canonical output for the two known-good pairs (Microglia → Cholinergic, Ndnf × Ndnf).
- [ ] Wire it into `run_pair_mode_pipeline.sh` as a final step (similar pattern to `verify_decomposition.py --all` in the smoke runner).
- [ ] On failure, the task exits non-zero with a per-position max-|Δ| table and the path-key recall count.

**Acceptance:** `pixi run verify-incytr-sce4` exits 0 against the current canonical output, non-zero if any fix is reverted.

### Phase 4 — CLAUDE.md + pipeline-contract documentation

The six fixes are project-specific correctness invariants. They belong in CLAUDE.md alongside the existing pair-mode invariants ("pair pvalue untrustworthy", "Cal_pairwise_grid is the entry point").

- [ ] Add a "Pair-mode Incytr — sce4 parity constants" section to `CLAUDE.md` listing the six call-site overrides with file:line refs to `incytr_commandline.R` and `build_input_gene_list.R`. One line per fix. Reference this plan document.
- [ ] Update `docs/foundation/live_pipeline_contract.md` Phase 1b/1c entries to call out the six overrides explicitly (currently the contract describes the run order but not the parameter overrides).
- [ ] Add a brief note to `alz/incytr_pair/README.md` (if present) or `incytr_commandline.R`'s top-of-file docstring summarizing the six-fix chain in one paragraph with a pointer to `bench/bench.md`.

### Phase 5 — Bench consolidation (anti-shim)

The bench drivers (`run_pair_posstrict.R`, `run_single_pair.R`, `run_single_pair_sce4dg.R`, `run_single_pair_posstrict.R`, `run_pair_mode_pipeline.sh`) re-implement the production driver with the same six fixes. Per anti-shim, parallel-mode code paths shouldn't coexist after the pivot lands.

Options, in preferred order:

1. **Delete the duplicates.** Replace bench drivers with thin wrappers that invoke the production driver with parameterized sender/receiver. The bench harness becomes purely a verification harness (Phase 3 artifact), not a parallel pipeline.
2. **Keep bench drivers explicitly as an investigation snapshot.** Add a top-of-file note: "frozen at fix-chain-v6; do not edit. The production driver at `alz/incytr_pair/incytr_commandline.R` is authoritative." Useful only if we expect to do another sce4-parity investigation.

The choice depends on whether more sce4-parity investigation is anticipated. Default: Option 1 (delete), revisit if a new parity question arises.

- [ ] Pick option 1 or 2.
- [ ] Execute: either delete duplicate drivers and migrate the smoke-probe utilities (`pick_probe_genes.R`, `check_scfc_fix.R`, etc.) under `alz/incytr_pair/`, or freeze with a top-of-file frozen note.
- [ ] Update `bench/bench.md` to point to canonical production paths instead of bench drivers.

## Dependencies and ordering

Phase 1 must complete before Phase 2 (otherwise the round-trip verifier fails for unrelated reasons). Phase 3 depends on Phase 2 (the parity script needs the canonical output to compare against). Phases 4 and 5 are independent and can run in parallel with Phase 3.

## Out of scope

- The App-as-Ligand/EM residual (1 gene, both pairs). Logged in `bench/bench.md` Investigation Plan §2 — separate effort.
- The Psen1 missing path (Ndnf×Ndnf). Same — `bench/bench.md` Investigation Plan §2.
- DESeq2-from-h5ad LFC reproduction. Superseded by fix #6 — the `lfc_reproduce_from_h5ad.py` artifact will not land.
- DB version archeology (Ndnf×Ndnf's 609 extras). Independent of this plan.
- Per-contrast pvalue stability (`Investigation Plan §5`). Independent.

## Effort estimate

| phase | likely effort |
|---|---|
| 1 — epsilon sync | 30–60 min (4 file edits + one verify-script test run) |
| 2 — canonical run + parity | 1–3 h (depends on whether the input_gene_list rebuild is needed; full 9-contrast run is ~30 min on this box) |
| 3 — regression task | 1 h (script generalization + pixi wiring) |
| 4 — documentation | 30 min |
| 5 — bench consolidation | 1 h if option 1, 15 min if option 2 |

## Decision point

Before starting Phase 5, confirm whether more sce4-parity investigation is expected. If yes → option 2 (freeze bench, keep). If no → option 1 (delete duplicates, move probes into `alz/`).
