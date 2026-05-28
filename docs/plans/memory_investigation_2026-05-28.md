# Pair-mode memory investigation — where the 14 GB actually lives

**Date:** 2026-05-28
**Status:** investigation only — no code changes proposed in this document.
**Context:** the 2026-05-27 audit failed to find memory wins (retrospective:
`pairmode_memory_audit_retrospective_2026-05-28.md`). User asked to look
again *with measurement* before any further pivot.

## Methodology

1. Re-read every file in the per-pair compute chain end-to-end:
   `Cal_pairwise_grid` (`incytr/R/grid.R`) →
   `fused_score_cutoff_paths` (`incytr/R/analysis.R:123–250`) →
   `Cal_scFC` / `Pathway_evaluation` (`incytr/R/analysis.R`,
   `incytr/R/evaluation.R`) →
   `prep_kinase_invariants` / `apply_kinase_to_pair` (`incytr/R/kinases.R`) →
   `Cal_PDS` (`incytr/R/evaluation.R`) →
   **`Permutation_test`** (`incytr/R/analysis.R:672–870`) — *this is the file
   I had not read carefully when writing the audit retrospective.*
2. Cross-referenced against measured RSS in the existing
   `bench/perf/post_step24_remeasure/run.log` (W=2 cell, NBOOT=100, 5-pair
   subset, full driver).
3. Sized every allocation that scales with `n_paths`, `n_genes`, or
   `n_cells` against the alz 46-cluster substrate dimensions (28,320
   genes; ~50k cells across the two conditions; ~5k genes/cluster median).

## Per-component memory accounting

From the driver log:
- pre-substrate RSS: 4,232 MB (Data.input + R session + Incytr namespace)
- after `rm(Data.input)`: 3,024 MB → **Data.input = ~1.2 GB**
- after `Expr_bygroup` substrate built: 4,070 MB → **substrate = ~1.0 GB**
- per-pair HWM during the boot loop: pair 5 (376k rows) = **5,013 MB**

Parent held state at start of pair loop: **~4 GB**. Per-worker peak during
the boot loop: ~5 GB. The ~1 GB delta between parent and worker HWM is the
chunk worth understanding — multiplied by W=3 it's where the cgroup peak
goes from 4 to ~14 GB.

### What the per-worker delta is composed of

`Permutation_test` builds these structures *inside the pair tail*, so they
live in the worker, not the parent:

| structure | file:line | dimension | size at alz scale |
|---|---|---|---|
| `permutation` matrix | `analysis.R:684` | `nC × nboot` int | `~50k × 100 × 4 B` = **20 MB** |
| `cond_mats[[j]]` (dense per condition) | `analysis.R:704–709` | `geneuse × cells_in_cond` double | `~2k × ~25k × 8 B = 400 MB × 2 conditions` = **~800 MB** |
| `cond_caches[[j]]` (sorted values + cell ranks) | `analysis.R:733` | same shape, 12 B/cell | `~2k × ~25k × 12 B × 2 conditions` = **~1.2 GB** if cache fits |
| `obs_vals`, `init_exceed`, FDR scratch | `analysis.R:801–806` | `n_pathways × n_conditions` | **~3 MB** (188k × 2 × 8 B) |
| Cal_foldchange transients (the audit target) | `analysis.R:177–186` | full 1.56M-row vectors | **~75 MB** transient |
| SigProb compute intermediates (`L*R`, hill, …) | `analysis.R:154–159` | same | **~75 MB** transient |

**The cache-vs-fallback fork is on line 718–736:**

```r
perm_cache_budget <- getOption("incytr.perm_cache_max_bytes", 500 * 1024^2)
...
if (cache_bytes > perm_cache_budget) {
  cond_caches <- vector("list", n_conditions)
  use_cached_perm <- FALSE
  break
}
```

At alz scale `cache_bytes ≈ 1.2 GB` for two conditions. The default budget
is **500 MB**, so the cache is *disabled* for heavy pairs and the loop
falls back to running grouped_quartile across `cond_mats` on every boot
iteration. In that mode `cond_mats` stays alive (~800 MB) through the
entire boot loop. (Line 745–751: when `use_cached_perm`, `cond_mats` is
nulled before the loop. When not, it's not.)

**The single largest addressable per-worker chunk is `cond_mats` and/or
`cond_caches` — both are dense `geneuse × cells_in_cond` matrices, one per
condition. Not `Cal_foldchange`, not the path enumeration, not the
deterministic scoring path. This is what the audit missed.**

Cross-check: per-worker delta (5,013 - 4,070) = 943 MB matches the
`cond_mats` size at this pair (~800 MB) plus small transients (~150 MB).

## How big is the actual lever?

Per-worker savings if you cut `cond_mats` / `cond_caches` in half:
~400–600 MB. At W=3 that's **~1.2–1.8 GB off cgroup peak (10–12%).**

If you eliminate the dense conversion entirely (keep sparse): savings
depend on the sparsity ratio. scRNA-seq is typically 5–15% nonzero;
`as.matrix(dgCMatrix)` blows it up to dense. Keeping sparse would save
roughly `(1 - density) × cond_mats_size` = `0.85 × 800 MB ≈ 680 MB` per
worker, **~2 GB cgroup at W=3 (14% of peak).**

This is a real, measurable, structural lever. Not a 1% trim.

## Why this was missed in the audit

The audit doc (`pairmode_memory_audit_2026-05-27.md` §"Where the bytes go")
focused on the *deterministic scoring path* — pathway enumeration,
SigProb, fold-change. That section traced sources 1–3 (memory growth in
`@pathways`, fork COW from `gc()`, Cal_foldchange's full-vector
allocation) and never tabulated the Permutation_test workspace.

Permutation_test consumed ~70% of pair wall time at NBOOT=100, so it was
acknowledged as the *runtime* hot spot. But the audit's memory hypothesis
was about the enumeration table size, not about the dense gene × cell
submatrices that Permutation_test materializes. Both can be true; the
audit only investigated one.

## Ranked levers (by estimated memory leverage, structural feasibility)

Each lever below is a *measurement target*, not a committed plan.
Numbers are estimates from code reading; would need empirical
confirmation before any commit.

### Tier 1: structural, no behavior change

1. **Keep `cond_mats` sparse through the cache build.**
   - Current: `analysis.R:725` does `if (inherits(mj, "dgCMatrix")) mj <- as.matrix(mj)` then calls `precompute_sorted_ranks(mj)`. The C++ kernel takes a dense matrix.
   - Lever: write a sparse-input variant of `precompute_sorted_ranks` (the per-cell rank for a column of mostly-zero values is degenerate and can be cached differently). Or: keep the kernel dense but build the cache in column chunks so peak is `chunk × cells × 12 B`, not `geneuse × cells × 12 B`.
   - Estimated saving: **~600–800 MB per worker; ~2 GB cgroup at W=3.**
   - Behavior: bit-identical (same numerical output, just smaller working set).
   - Effort: moderate. Touches `src/grouped_quartile.cpp` (already a project file, mature). Needs careful sparse-aware ranking algorithm.

2. **Raise `perm_cache_max_bytes` default OR drop the dense fallback.**
   - Current: 500 MB default forces the slower, larger-footprint non-cached path on heavy pairs.
   - The cached path is *both faster and lower-memory* (cond_mats is released after cache build, line 745–746). The 500 MB budget was meant to protect tight-RAM hosts but it actually makes things worse on heavy data: the alternative is keeping `cond_mats` alive (~800 MB) AND running the slower R loop.
   - Lever: either (a) raise default to 1.5 GB (cached path becomes the dominant footprint, but smaller than cond_mats), or (b) when the budget fails, *fail loudly* rather than silently fall back to a larger footprint.
   - Estimated saving: **~200 MB per worker; ~600 MB cgroup at W=3.**
   - Behavior: bit-identical.
   - Effort: small. One-line option default change + a comment.

3. **Quantize the substrate to `float` (single precision) in cond_mats/cond_caches only.**
   - SigProb values are in [0,1]; trimean expressions are bounded log-counts. `float` has 7 decimal digits of precision — far more than the downstream `cutoff_SigProb=0` filter cares about.
   - R doesn't have native `float` arrays, so this would mean keeping the bytes as `float` in C++ (the cpp kernel already operates on these matrices) and exposing them to R as integers or via a raw vector.
   - Estimated saving: **~400 MB per worker; ~1.2 GB cgroup at W=3** (half the current 12-B-per-cell cache).
   - Behavior: numerically very close but not bit-identical. Would fail a strict `max_abs_diff = 0` gate on whatever decimal place float-vs-double diverges (typically 8th–9th decimal). sce4 parity check uses `max(|Δ|) > 0.0001` tolerance — would still PASS.
   - Effort: large. Touches cpp kernels and the R/C++ ABI for the cache.

### Tier 2: behavior-aware

4. **Reduce geneuse for the permutation by pre-filtering low-variance pathway genes.**
   - Current: `geneuse <- unique(c(pw$Ligand, pw$Receptor, pw$EM, pw$Target))` — every gene that appears in any surviving pathway.
   - Many of those genes have near-zero expression in all condition cells. Including them in `cond_mats` is dead weight — their permuted SigProb is always ~0, so they don't move the exceed count.
   - Lever: filter geneuse to genes with non-trivial expression variance in either condition. Lossy — drops pathways whose only differentiator is a near-zero gene.
   - Estimated saving: depends on the filter threshold. A 50% reduction in geneuse → ~400 MB per worker; ~1.2 GB cgroup at W=3.
   - Behavior: not bit-identical. Need a domain decision about which gene filter is acceptable.
   - Effort: moderate.

5. **`Cal_foldchange` aFC quantile decoupling** (the audit retrospective's named follow-up).
   - Now I can size it correctly. Per-worker transient saving: ~75 MB. At W=3: ~225 MB cgroup. **~1.5% of peak.**
   - Behavior: changes aFC values for rows near the cutoff edge in defined but biologically nontrivial ways.
   - Effort: small code-wise; large validation-wise.
   - **Verdict from this investigation: not worth chasing as a memory move.** The win is too small relative to Tier 1 candidates. Pursue only if domain review independently decides post-cutoff `th` is more correct.

### Tier 3: architectural

6. **PSOCK instead of fork for the pair-worker layer.**
   - Current: `parallel::mclapply` forks the whole address space. Per-worker memory = parent_held + worker_dirty.
   - PSOCK workers start cold and you ship them only what they need.
   - Estimated effect: lower per-worker peak (no inherited substrate) but higher startup cost (cluster init + data serialization) and higher absolute memory because workers can't COW-share read-only state.
   - **Trade-off, not a win.** Worth keeping in mind as a fallback if fork pressure gets worse.

7. **DuckDB-ify the deterministic scoring path** (audit's out-of-scope item).
   - Already on the dependency list. Would eliminate `paths_dt` materialization in R entirely; SigProb compute becomes a streamed columnar query.
   - High ceiling, large effort. Outside any short-term iteration.

## Validation plan if any Tier 1 lever is pursued

1. Add `gc(reset=TRUE)` + `object.size()` instrumentation at the
   key checkpoints inside `Permutation_test` (cache build, cache active,
   boot loop entry, boot loop exit). Run W=1 on a heavy pair, confirm
   the estimated sizes above match what's actually allocated.
2. Apply the chosen lever behind an option (default off).
3. Re-run `bench/perf/exp_b_cell.sh 3 1 18G` with the option on, compare
   `memory.peak` and per-pair walls against the current Step 4 baseline.
4. Run `bench/fidelity/validate_derived_parity.sh` — sce4 parity must
   PASS at `max(|Δ|) ≤ 0.0001` on all four node columns.
5. If Tier 1 #3 (float quantization), also run `compare_pair_outputs.R`
   with a tolerance argument to characterize the magnitude of the
   double-vs-float drift on a known pair output.

## Honest framing

The audit *did* miss the largest addressable memory lever — it's in
`Permutation_test`, not the scoring path. The bug it fixed
(`Cal_PDS` row alignment) was real and valuable. But the *memory*
question was not actually investigated where the memory is.

This investigation now puts the levers on the table with concrete
estimates. Tier 1 candidates 1 and 2 together could plausibly drop
cgroup peak from ~14 GB to ~11 GB at W=3 (a real ~20% reduction,
behavior-preserving). Tier 1 #3 could push further. Tier 2 and 3 are
behavior or architecture decisions, not refactors.

**Open question for the next direction-setting:** is the goal to lower
memory enough to admit W=4 (which would unlock wall-time on
long-pair-list workloads — the audit's W=4 ceiling was the
heaviest-pair runtime AND the memory budget; the latter would lift if
Tier 1 lands), or is the current footprint acceptable and the
correctness/cleanup work has higher priority?
