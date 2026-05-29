# Implementation plan — Lever A (kinase EI hoist) + Lever B (sparse cache build)

**Date:** 2026-05-28
**Investigation:** `optimization_levers_2026-05-28.md`
**Strict-fidelity gate:** bit-identical output, `max |Δ| = 0` across every
numeric column on the existing 2-pair fidelity benchmark.

## Goals

- Land two independent, bit-identical optimizations on the Incytr package
  source.
- Keep each lever in **one dedicated commit** so either can be reverted
  cleanly via `git revert <sha>` without touching the other.
- Gate every commit through the existing fidelity infrastructure before it
  lands.
- Update NEWS.md in the same commit as the code change so the changelog and
  the diff move together.

## Branch policy

- One feature branch per lever, off master:
  - `perf/lever-A-kinase-ei-hoist`
  - `perf/lever-B-sparse-precompute-sorted-ranks`
- Levers are independent — they can be developed in parallel, merged in
  either order. Neither depends on the other for correctness or test pass.
- Each branch must rebase cleanly on master before merge. No merge commits.
- Only the user pushes / merges / opens PRs. The assistant does not push.

## Fidelity infrastructure (already in repo, do not modify)

| asset | path | role |
|---|---|---|
| comparison script | `alzheimers/bench/fidelity/compare_pair_outputs.R` | bit-identical diff across all numeric + categorical columns of pair-mode output |
| frozen 2-pair driver | `alzheimers/bench/fidelity/run_one_pair_v2_frozen.R` | invokes `Cal_pairwise_grid` on a frozen 2-pair input, writes parquet |
| frozen parity baseline | `alzheimers/bench/fidelity/parity_frozen_out/` | reference output (current master before this work) |
| derived parity baseline | `alzheimers/bench/fidelity/parity_derived_out/` | reference output on alz-derived inputs |
| frozen validation script | `alzheimers/bench/fidelity/validate_frozen_parity.sh` | runs frozen driver + compare; exits non-zero on `max |Δ| > 0` |
| derived validation script | `alzheimers/bench/fidelity/validate_derived_parity.sh` | same on alz-derived inputs |
| testthat suite | `incytr/tests/testthat/` | 385 unit tests including `test-golden_output.R` (golden fixture regression) and `test-sce4_defaults.R` (sce4 parity defaults) |

A change passes the fidelity gate iff:
1. `pixi run test` (R package testthat, run from `incytr/`) reports
   `385 PASS, 0 FAIL`.
2. `bash bench/fidelity/validate_frozen_parity.sh` exits 0 with
   `max |Δ| = 0` on every reported column.
3. `bash bench/fidelity/validate_derived_parity.sh` exits 0 with
   `max |Δ| = 0` on every reported column.

All three are mandatory per lever before commit.

---

## LEVER A — hoist `compute_kinase_invariant_per_condition` out of the per-pair loop

### Scope

One commit on `perf/lever-A-kinase-ei-hoist`. No source files touched
besides what's required for this lever; the diff should make the lever
obvious without ambiguity.

### Files to change

| file | change | est. lines |
|---|---|---|
| `incytr/R/kinases.R` | split `prep_kinase_invariants`; extract `prep_kinase_ei_invariants` | ~80 (split) |
| `incytr/R/grid.R` | call hoisted EI computation once at `Cal_pairwise_grid` entry; pass result through `run_one` | ~15 |
| `incytr/NAMESPACE` | unchanged — new helper stays internal | 0 |
| `incytr/man/*.Rd` | regenerated via `pixi run document` | mechanical |
| `incytr/NEWS.md` | one bullet documenting the lever | ~10 |

### Implementation steps

1. **Audit pass (read-only).** Re-read `kinases.R:66-296` end-to-end with
   the explicit goal of confirming `compute_kinase_invariant_per_condition`
   has zero pair-dependency on its arguments. Verify that the only data
   the function uses from `object` is `@data`, `@idents`, `cells_cond` (per
   condition), and the kinase gene set derived from `kldata` ∩ pathway
   genes. Document any cross-cutting reference found that contradicts the
   pair-invariance claim. If found, stop and revise the plan.

2. **Design the split.** New signature:
   ```r
   # New helper, runs once per contrast.
   prep_kinase_ei_invariants(object, kinase_gene_union,
                             mean_method, fold_threshold)
     -> list(ei_by_cond = list(<cond1> = <ei_df>, <cond2> = <ei_df>))
   ```
   The `kinase_gene_union` argument is the union of kinase genes across
   the full pair grid (computed once at `Cal_pairwise_grid` entry from
   `union(@pathways$Ligand, $Receptor, $EM, $Target) ∩ kldata$gene`).
   `prep_kinase_invariants` keeps its current signature but accepts an
   optional `ei_by_cond` argument; when supplied, Part 3
   (`compute_kinase_invariant_per_condition` calls) is skipped and the
   passed-in dict is used. When NULL, current behavior is preserved
   exactly — this preserves the single-pair entry points used by tests.

3. **Wire through `Cal_pairwise_grid`.** At `grid.R` ~line 200 (before the
   per-pair loop), compute `kinase_gene_union` from the full grid's
   pathway set, build `ei_by_cond` once via `prep_kinase_ei_invariants`,
   pass it to each `run_one` invocation. `run_one` forwards it to
   `prep_kinase_invariants`.

4. **Fidelity check #1 — testthat.** From `incytr/`:
   `pixi run test`. Required: `385 PASS, 0 FAIL`. If any test fails, fix
   before proceeding. The `test-prep_kinase_invariants.R` test (if it
   exists) and `test-Cal_pairwise_grid.R` are the most likely to surface
   plumbing bugs.

5. **Fidelity check #2 — frozen parity.** From `alzheimers/`:
   `bash bench/fidelity/validate_frozen_parity.sh`. Required:
   `max |Δ| = 0` on every column. Any `Δ > 0` is a regression — stop and
   investigate before committing.

6. **Fidelity check #3 — derived parity.** From `alzheimers/`:
   `bash bench/fidelity/validate_derived_parity.sh`. Required:
   `max |Δ| = 0` on every column. Same stop rule.

7. **Regenerate Rd files.** From `incytr/`: `pixi run document`. Stage the
   resulting `man/*.Rd` changes alongside the R changes.

8. **Update NEWS.md.** Single bullet under the next-version heading
   describing the split, the fidelity gate it passed, and the mechanism
   (pair-invariant work hoisted, output bit-identical).

9. **Commit.** One commit, conventional message:
   ```
   perf(kinases): Lever A — hoist EI computation out of per-pair loop

   <body>

   Fidelity: testthat 385/385 PASS, frozen + derived parity max|Δ|=0.
   ```
   Stage only: `R/kinases.R`, `R/grid.R`, `man/*.Rd`, `NEWS.md`. Nothing
   else. If `git status` shows anything else dirty, sort it out before
   committing.

### Rollback path

`git revert <lever-A-sha>` on master restores prior behavior in one commit.
The Rd regeneration is part of the same commit, so the revert is clean.

---

## LEVER B — sparse-aware `precompute_sorted_ranks` (eliminate the 432 MB dense cast)

### Scope

One commit on `perf/lever-B-sparse-precompute-sorted-ranks`. C++ kernel
addition plus the R-side wrapper update and the `as.matrix` removal at the
single call site. No changes to `run_permutation_exceed_raw`.

### Files to change

| file | change | est. lines |
|---|---|---|
| `incytr/src/grouped_quartile.cpp` | new `precompute_sorted_ranks_sparse_raw` C++ function accepting dgCMatrix `x`/`i`/`p` SEXPs; same dense output as the existing dense-input variant | ~150 |
| `incytr/src/init.c` | register the new entry point | ~3 |
| `incytr/R/cpp11.R` | regenerated via `pixi run document` (if cpp11) or manual update if Rcpp pattern | mechanical |
| `incytr/R/grouped_quartile.R` | wrapper dispatches on `inherits(mat, "dgCMatrix")` → sparse path; relaxes the existing `stop("must be numeric matrix")` | ~15 |
| `incytr/R/analysis.R` | remove the `if (inherits(mj, "dgCMatrix")) mj <- as.matrix(mj)` cast at line 725; pass sparse through directly | -1 |
| `incytr/tests/testthat/test-grouped_quartile.R` (or new `test-precompute_sorted_ranks.R`) | add bit-identity test: dense input vs sparse input on the same matrix produces byte-identical `sorted_values` and `cell_ranks` | ~30 |
| `incytr/NEWS.md` | one bullet | ~10 |

### Implementation steps

1. **Audit pass (read-only).** Re-read
   `src/grouped_quartile.cpp:206-267` (`precompute_sorted_ranks_raw`).
   Confirm: the dense path's only use of the input matrix is per-gene
   value extraction → `std::stable_sort` → assignment back to
   `sorted_values` and `cell_ranks`. No global cross-gene structure is
   used. This is the structural premise of the sparse variant: process
   each gene independently from its sparse column slice.

2. **Design the sparse algorithm (per gene).** For gene `g`:
   - Extract `(value, cell_idx)` pairs from the dgCMatrix where row == g.
     Costs `O(nnz_g)` per gene via standard CSC traversal (one pass over
     `p`-array boundaries with `i`-array filter on `i == g`, or — better —
     pre-transpose once to dgRMatrix for `O(1)` gene-row access).
   - Sort the `nnz_g` non-zero entries by value (`std::stable_sort`).
   - Write `sorted_values[g, 0..n_cells - nnz_g - 1] = 0.0` (the zero
     block), `sorted_values[g, n_cells - nnz_g..n_cells - 1] = sorted
     non-zero values`.
   - Build `cell_ranks[g, cell_idx]`: zero-cells get ranks
     `0..n_cells - nnz_g - 1` (assigned in original-cell-order for
     `std::stable_sort` tie equivalence with the dense path);
     non-zero-cells get their post-sort positions.

3. **Critical correctness check.** The dense path uses `std::stable_sort`
   on the full per-gene column. For ties at value 0.0, stable_sort
   preserves original order. The sparse path must reproduce that exact
   tie-breaking: zero-cells receive ranks in original cell-index order.
   Validate this in step 5 below; this is the single most likely place for
   a `max|Δ| > 0` regression.

4. **Implement the C++ kernel.** Add
   `precompute_sorted_ranks_sparse_raw(SEXP x, SEXP i, SEXP p,
   int n_genes, int n_cells)` returning `list(sorted_values, cell_ranks)`
   of the same shapes and storage modes as the dense variant. Register in
   `init.c`. Avoid intermediate full-matrix allocation — the entire
   purpose is to never materialize the dense form.

5. **Add unit test.** In `test-precompute_sorted_ranks.R` (new file or
   appended): construct a randomized 200×500 dense matrix with ~90% zeros,
   convert to dgCMatrix, run both `precompute_sorted_ranks(dense_mat)` and
   `precompute_sorted_ranks(sparse_mat)`. Assert `identical()` on both
   `sorted_values` and `cell_ranks` outputs. Run on 5 different random
   seeds. The test must pass before any subsequent step.

6. **Update R wrapper.** In `R/grouped_quartile.R`, dispatch on input
   class: dgCMatrix → sparse kernel, dense matrix → existing kernel.
   Remove the `stop("must be numeric matrix")` guard. Preserve all other
   pre-checks (positive dims, etc.).

7. **Remove the cast at the call site.** In `R/analysis.R:725`, delete
   the `if (inherits(mj, "dgCMatrix")) mj <- as.matrix(mj)` line. Verify
   by re-reading the surrounding loop that no later code in the same
   function assumes `mj` is dense.

8. **Fidelity check #1 — testthat.** From `incytr/`: `pixi run test`.
   Required: `385 + new test PASS, 0 FAIL`. The new
   `test-precompute_sorted_ranks.R` is the local guard; `test-golden_output.R`
   is the end-to-end guard.

9. **Fidelity check #2 — frozen parity.** Same as Lever A step 5.
   Required `max |Δ| = 0`.

10. **Fidelity check #3 — derived parity.** Same as Lever A step 6.
    Required `max |Δ| = 0`.

11. **Memory verification (informational, not gating).** Re-run
    `bench/perf/permtest_memory_profile.R` and confirm the
    `12_precompute_sorted_ranks` tracepoint shows the same cache size
    (~170 MB) but the RSS at that tracepoint no longer reflects a 432 MB
    transient. Record the delta in the commit body.

12. **Regenerate cpp11 / Rd if needed.** `pixi run document`. Stage.

13. **Update NEWS.md.** One bullet describing the new sparse entry point,
    the removed cast, and the measured transient elimination.

14. **Commit.** One commit, conventional message:
    ```
    perf(grouped_quartile): Lever B — sparse-aware precompute_sorted_ranks

    Eliminates 432 MB dense-cast transient in Permutation_test cache build
    at alz scale. dgCMatrix input now flows through a new sparse C++ entry
    point that produces identical sorted_values + cell_ranks output.

    Fidelity: testthat 385/385 + new sparse-equivalence test PASS,
    frozen + derived parity max|Δ|=0. Measured transient delta: ...
    ```
    Stage only: `src/grouped_quartile.cpp`, `src/init.c`, `R/cpp11.R`,
    `R/grouped_quartile.R`, `R/analysis.R` (1 line), the new test file,
    `NEWS.md`, regenerated `man/*.Rd` if any. Nothing else.

### Rollback path

`git revert <lever-B-sha>` on master restores the dense cast and the
existing kernel-only behavior in one commit. The C++ entry point added by
this commit is removed by the revert; downstream callers (only the wrapper
at `R/grouped_quartile.R`) revert with it.

---

## Cross-lever stop rules

- If any of the three fidelity gates (testthat / frozen parity / derived
  parity) reports `max |Δ| > 0` on a column, **do not commit**. Diagnose
  the regression first. Most likely sources, in order:
  1. Lever B: zero-cell rank tie-breaking diverges from `std::stable_sort`.
  2. Lever A: `kinase_gene_union` excluded a kinase gene that some pair
     needed → `ei_df` lookup returns NA where current code returned a
     value.
  3. Either lever: an upstream object hash differs (column-order change)
     that propagates through.

- The fidelity infrastructure compares output schemas as well as numeric
  values. A column-order or column-name change is a regression even if all
  numbers match.

- Each lever's commit must pass all three gates in **isolation** (on its
  own branch, with master rebased in). If you combine the levers
  prematurely and one regresses, you cannot revert just one.

## Merge order

Either order works since the levers are independent. Recommended:

1. Land Lever A first. Smaller blast radius (R-only, no C++).
2. Rebase Lever B on the updated master. Re-run all three fidelity gates
   on Lever B after rebase. Land.

Alternative (parallel): hand off Lever B's C++ work in parallel with
Lever A's R work. Re-run fidelity on whichever lands second after rebase.

## Out of scope (do not touch in these commits)

- `Cal_foldchange` aFC quantile semantics (Lever F/G blocker — separate
  proposal).
- Sequential permutation (Lever E — separate proposal).
- Per-cluster omics FC cache (Lever C — separate proposal, ship later if
  desired).
- DB layer pre-filter cache (Lever D — small prize, ship later if
  desired).
- Any cleanup / refactor not directly required by the lever. Keep the
  diffs surgical so reverts stay clean.

## What "done" looks like

- master contains two commits, in order:
  - `perf(kinases): Lever A — hoist EI computation out of per-pair loop`
  - `perf(grouped_quartile): Lever B — sparse-aware precompute_sorted_ranks`
- Each commit's body lists the fidelity gates that passed.
- Either commit can be reverted with `git revert <sha>` without breaking
  the other.
- NEWS.md documents both levers.
- The previous retrospective's "memory floor is hit" claim is contradicted
  by measurable evidence: Lever A removes ~100-150 MB × 961 redundant
  allocations per contrast; Lever B removes a 432 MB transient per fork.

## Approval gate

Plan requires user approval before any source-modifying step. After
approval, the assistant proceeds lever-by-lever, running the fidelity
gates locally and stopping for user review at each commit boundary.
