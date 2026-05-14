# Incytr Pair-Mode Implementation Audit

**Date**: 2026-05-13  
**Scope**: Pair-mode pipeline only. `R/factorial.R` (upstream) is entirely excluded.  
**Phase C reference**: The Astrocytes→Microglia pair produced bit-identical scoring columns between legacy and upstream (2026-05-12 benchmark run, matched-knob configuration per `incytr_pair_mode_benchmark_plan.md`).  
**Author**: claude (code-level static audit, no execution)

---

## Overview

The legacy codebase lives in five source files sourced at runtime by `incytr_commandline.R`. The upstream package is a proper R package with the same five functional domains split across seven files. This audit compares the two implementations step by step, function by function, through the 10-function pair-mode pipeline. Where a function is essentially identical modulo whitespace and variable renames, that is stated briefly. Where there is a genuine algorithmic or default divergence, the section is expanded.

---

## Step-by-step function audits

### Step 1 — `create_Incytr`

| | Legacy | Upstream |
|---|---|---|
| File | `Incytr_functions_object.R:41–104` | `Incytr_class.R:107–217` |

**Signature delta**

| Arg | Legacy | Upstream | Notes |
|---|---|---|---|
| `object` | yes | yes | |
| `meta` | NULL | NULL | |
| `sender` | yes | yes | |
| `receiver` | yes | yes | |
| `group.by` | yes | yes | |
| `conditions` | yes | yes | renamed from `condition` in even earlier versions |
| `animal_id` | absent | NULL | new; enables factorial mode |
| `design` | absent | NULL | new; factorial design matrix |
| `contrasts` | absent | NULL | new; named contrast list |
| `assay` | yes | yes | still unused |
| `do.sparse` | yes | yes | still unused |

**Algorithmic delta**

The legacy body (`object.R:51–104`) inlines a copy of the meta-validation logic (the `matrix→data.frame` coerce, barcode identity check, warning on mismatch). The upstream body (`Incytr_class.R:134`) replaces this with a call to `validate_meta()` extracted to `utils.R:10–26`. Logic is identical.

Legacy uses `|` (non-short-circuit) at line 59 to check `is.null(sender) | is.null(receiver)`. Upstream uses `||` (short-circuit) at line 128. This is cosmetic for this check since both sides are simple null tests, but represents a correctness improvement for more general use.

The upstream constructor (lines 148–213) adds the full factorial-mode validation block — rank-deficiency check (`qr(design)$rank`), rowname-parity check between `design` and unique animals, etc. This block is unreachable in pair mode (all three new args default to NULL) and does not affect pair-mode outputs.

**Knob/default changes that silently affect outputs**: None for pair mode. `options$mode` is still set to `"single"` when `design=NULL`.

**Dead code in legacy**: `data.raw` slot is declared in the legacy S4 class (`object.R:17`) and populated by `Cal_scFC` (the DESeq2 variant at `analysis.R:645–713`). The legacy driver never calls this `Cal_scFC` variant. The upstream `Cal_scFC` has been completely rewritten for legacy/factorial modes and no longer accepts a `count.matrix`.

**Risk classification**: (a) cosmetic/refactor — `validate_meta` extraction, `||` vs `|`. (b) matched by explicit knob — none needed for pair mode. New factorial args: (a) unreachable in pair mode.

---

### Step 2 — `pathway_inference`

| | Legacy | Upstream |
|---|---|---|
| File | `Incytr_functions_analysis.R:320–385` | `analysis.R:37–91` |

**Signature delta**: Identical. No new or removed args.

**Algorithmic delta**

Legacy (`analysis.R:338–344`) filters DB layers with a for-loop, renames columns via `colnames(DB[[i]]) <- ...`, then joins with `dplyr::inner_join(..., multiple="all")`.

Upstream (`analysis.R:55–88`) converts each DB layer to `data.table`, uses `setnames()`, applies per-role filters before joining (legacy applies them after, via additional subset rows at 348–367), then executes keyed data.table joins (`setkey` + `dt2[dt1, allow.cartesian=TRUE, nomatch=0]`).

The deduplication logic differs:
- Legacy (`analysis.R:374–377`): uses `apply(SigPath, 1, duplicated)` — this tests per-row whether a gene name appears more than once in that row, and drops any row with a duplicate. The boolean matrix is transposed and column-summed (`apply(boolean_dup, 2, sum) > 0`).
- Upstream (`analysis.R:82–84`): uses explicit column comparisons (`Ligand != Receptor & Ligand != EM & ...`), which is faster and less fragile. Both remove paths where any two nodes share a gene.

Upstream adds `setorderv(SigPath, c("Ligand","Receptor","EM","Target"))` (line 86), giving a deterministic row order. Legacy row order depends on dplyr join order. This can affect permutation p-value comparisons if a different permutation seed were used, but with matched `seed.use=1L` and matched pathway sets, the SigProb vectors are indexed by `Path` string not row position, so this does not affect scores.

**Knob/default changes**: None. All five filter args (`ligand`, `receptor`, `em`, `target`) remain optional and default to NULL.

**Risk classification**: (a) cosmetic/refactor — data.table joins vs dplyr joins, explicit dedup vs apply-based. Row order determinism: (b) matched by the fact that downstream joins are by `Path` string, not positional.

---

### Step 3 — `Expr_bygroup`

| | Legacy | Upstream |
|---|---|---|
| File | `Incytr_functions_analysis.R:387–470` | `analysis.R:102–122` |

**Signature delta**: Identical — `(object, mean_method=NULL)`.

**Algorithmic delta**

The legacy body is ~84 lines of repeated code that processes condition1 and condition2 in parallel: slice the matrix, transpose, attach group labels, compute quantiles or mean via `setDT()`, transpose back.

The upstream body (`analysis.R:105–122`) is 18 lines. It calls the new helper `compute_group_expr()` (`utils.R:234–238`) which in turn calls `grouped_expr_from_matrix()` (`utils.R:206–229`). The latter uses `matrixStats::rowQuantiles(probs=c(0.25,0.5,0.75), type=7L)` vectorized over all genes simultaneously instead of three separate `lapply(.SD, quantile, ...)` passes on a data.table. For sparse matrices (`dgCMatrix`) it coerces to dense before calling `rowQuantiles`.

The upstream loop iterates over all conditions generically (`for (cond in conditions)`), not hardcoded `condition1/condition2`, making it correct for N>2 conditions without code changes.

**Key numeric equivalence question**: Both implementations compute the trimean as `0.25*Q1 + 0.5*Q2 + 0.25*Q3`. The quantile type used by `matrixStats::rowQuantiles(..., type=7L)` is R's default type-7 interpolation, the same as `base::quantile()`. Numeric identity holds.

**Knob/default changes**: `mean_method=NULL` default is the same. The Phase C benchmark explicitly passes `mean_method="mean"` to both, which bypasses the trimean path entirely and uses arithmetic mean — so the `rowQuantiles` vs `lapply(quantile)` difference does not apply to the actual Phase C run. For a default-argument run (mean_method=NULL), both paths are numerically equivalent.

**Risk classification**: (a) cosmetic/refactor — helper extraction, vectorization, condition-count generalization. No output change.

---

### Step 4 — `Cal_SigProb`

| | Legacy | Upstream |
|---|---|---|
| File | `Incytr_functions_analysis.R:515–642` | `analysis.R:171–232` |

**Signature delta**

| Arg | Legacy | Upstream | Notes |
|---|---|---|---|
| `K` | 0.5 | 0.5 | |
| `N` | 2 | 2 | |
| `cutoff_SigProb` | NULL | NULL | |
| `correction` | 0.0001 | 0.0001 | |
| `q` | NULL | NULL | |
| `compute_fc` | absent | TRUE | new; set FALSE in permutation iterations |

**Algorithmic delta**

Legacy (`analysis.R:521–598`) does two separate, near-identical blocks: a big `left_join` chain for condition1, another for condition2. Upstream (`analysis.R:177–201`) iterates over conditions with the helper `compute_sigprob()` (`analysis.R:127–138`). The helper uses named-vector lookups (`setNames(expr_data[[sender]], expr_data$Gene)`) instead of four sequential left-joins; this avoids allocating intermediate joined data frames.

The `Cal_foldchange` call signature changed. Legacy calls `Cal_foldchange(df, correction=..., q=...)` with `df` having hardcoded column names `condition1` and `condition2`. Upstream's `Cal_foldchange` accepts `cond1_col` and `cond2_col` parameters (`math.R:34–70`); upstream `Cal_SigProb` (line 213) passes these explicitly. This is backward-compatible and does not change values.

The `cutoff_SigProb` filter in legacy (`analysis.R:618–627`) calls `object_update(object, Path=Path.update)`. Upstream (`analysis.R:221–228`) calls the extracted wrapper `apply_path_cutoff(object, keep)` which just calls `object_update`. Same logic.

The new `compute_fc=TRUE` arg (`analysis.R:168`) allows the permutation loop to skip fold-change computation, which is a performance optimization only — not reachable when the caller passes the default or uses the normal pipeline.

**Knob/default changes**: None for the matched-knob run. `compute_fc` defaults to TRUE so legacy callers get identical behavior without explicitly setting it.

**Risk classification**: (a) cosmetic/refactor — helper extraction, named-vector vs join. (a) `compute_fc` arg is a perf-only addition.

---

### Step 5 — `Integr_multiomics`

| | Legacy | Upstream |
|---|---|---|
| File | `Incytr_functions_analysis.R:715–860` | `analysis.R:710–758` (dispatch) + `analysis.R:444–508` (core) |

**Signature delta**

Upstream adds an `omics` named-list parameter (`analysis.R:713`) as a higher-level alternative to the positional `pr.data_condition1/condition2/...` args. Legacy had only the positional form. Upstream supports three new omics layers: `Ack`, `KGG`, `Rme1` (with new slots `Ack_FC`, `KGG_FC`, `Rme1_FC` on the S4 class). The legacy positional args (`pr.*`, `ps.*`, `py.*`) are still accepted and routed through the new `integr_multiomics_impl` internal function.

**Algorithmic delta**

Legacy (`analysis.R:729–860`) has three nearly-identical blocks, one per omics layer, each with: column selection, `inner_join` for condition1/condition2, limma normalization, `Cal_foldchange`, then a sequence of four `left_join` operations to attach log2FC/aFC columns to the pathway table by role (Ligand/Receptor/EM/Target).

Upstream refactors this into `integrate_omics_layer()` (`analysis.R:446–508`) called once per layer. The main algorithmic change is the column attachment: legacy (`analysis.R:758–767`) selects `df[, 6:13]` by position (hardcoded column offsets), which is fragile if the pathway data frame has extra columns added upstream. Upstream (`analysis.R:487–507`) builds new columns using named-vector lookups, entirely avoiding positional indexing.

Legacy uses `inner_join(data_s1, data_s2, by="gene_symbol")` to align conditions. Upstream uses `intersect()` + `match()` to find common genes without a join. Semantically equivalent but the upstream approach preserves column names explicitly rather than relying on dplyr's `.x`/`.y` suffix handling.

**Knob/default changes**: Legacy hardcodes `correction=NULL` (becomes 0.0001 in `Cal_foldchange`) and `q=NULL` (becomes 0.75). These are passed through identically in upstream. In the Phase C benchmark, `pr.correction=0.001` was set explicitly for both — this knob match is sufficient.

**Risk classification**: (a) cosmetic/refactor — helper extraction, named-vector vs positional column indexing. (a) new `Ack/KGG/Rme1` layers are additive only.

---

### Step 6 — `Pathway_evaluation`

| | Legacy | Upstream |
|---|---|---|
| File | `Incytr_functions_evaluation.R:5–144` | `evaluation.R:12–149` |

**Signature delta**: Identical for pair mode. Upstream signature is identical; no new args for the single-condition case.

**Algorithmic delta**

The most significant change is the `score.weight` validation. Legacy checks `length(score.weight)!=3` (line 12) and defaults to `c(0.5, 0.5, 0.5)`. Upstream checks `length(score.weight)!=6` (line 19) and defaults to `rep(0.5, 6)`. This reflects the addition of three new omics layers (Ack, KGG, Rme1). A legacy caller passing an explicit `score.weight=c(0.5, 0.5, 0.5)` will get silently replaced by the upstream default `c(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)`. Since the AD driver passes no explicit `score.weight`, both legacy and upstream use their respective default, and the extra three terms contribute 0 (empty slots). Net effect on pair-mode outputs with no Ack/KGG/Rme1 data: identical `multimodel_score`.

The `abs.value` handling is fixed in upstream. Legacy (`eval.R:34–44`) has a sequence of `if / else if` that only sets `case[1]` when "Ligand" is in `abs.value`, but never sets `case[2:4]` because the `else if` prevents reaching those branches (if "Ligand" is absent but "Receptor" is present, `case` is never initialized and the function would error). Upstream (`eval.R:42–49`) replaces this with a correct `if/if/if/if` pattern. This is a bug fix in upstream; however, the AD driver always passes `abs.value="None"`, so this branch is never reached in practice.

Upstream refactors the three-omics scoring loop into `score_omics_layer()` (`eval.R:54–68`), which generalizes to any number of omics layers and handles missing column names gracefully (returns 0). Legacy had three near-identical `if(nrow(object@pr_FC)==0)...` blocks.

**Knob/default changes**: `score.weight` default expansion from 3→6 elements: (b) matched by knob — the AD driver does not pass this arg and no Ack/KGG/Rme1 data is provided, so the extra weights multiply zero and produce no change. `abs.value` bug fix: unreachable in practice for AD driver.

**Risk classification**: (a) cosmetic/refactor for score_omics_layer extraction. (b) matched by knob for score.weight. The abs.value bug fix is (c) latent risk — it would fire if a future caller passes a vector like `c("Receptor")`, which is now correctly handled by upstream but would silently mis-score in legacy.

---

### Step 7 — `Integr_kinasedata`

| | Legacy | Upstream |
|---|---|---|
| File | `Incytr_functions_kinases.R:58–308` | `kinases.R:137–301` |

**Signature delta**: Identical — `(object, kldata, mean_method=NULL, cell_group, fold_threshold=10)`.

**Algorithmic delta**

**Part 2** (pathway-kinase case matching): Legacy (`kinases.R:108–119`) uses `dplyr::inner_join` to match pathway genes against the kinase library. Upstream (`kinases.R:190`) uses `merge(x, y, by=c("motif.geneName","gene"))`. Semantically identical for this use; `merge` here mirrors the default `inner_join`.

**Part 3** (per-condition expression): Legacy processes condition1 and condition2 with two separate, near-identical blocks (lines 154–220). Upstream extracts this into `compute_kinase_condition()` (`kinases.R:66–123`) and calls it in a `for (cond in object@conditions)` loop (lines 240–257). The extracted helper (`kinases.R:70–75`) fixes a latent single-gene bug: legacy had a branch `if(length(geneuse)==1)` that set `colnames(df1) = geneuse` but then did not transpose correctly (produces a 1×N data frame when cells×1 is needed). Upstream always transposes with `as.data.frame(t(M))` regardless of gene count.

**Part 4 (Cal_EI)**: Legacy (`kinases.R:224–232`) calls `Cal_EI` identically to upstream, but Cal_EI itself was vectorized in upstream (see Cal_EI section below).

**Part 5** (EI column assembly): Legacy (`kinases.R:244–305`) uses nested `left_join` calls to attach EI values to `kl.pathways` per condition, accumulating columns with repeated `left_join(kl.EI, df_1[...], by=...)`. Upstream (`kinases.R:263–298`) uses direct vector assignment via `match()`, which is both faster and avoids the name-collision risk (`left_join` would auto-suffix `.x`/`.y` if the joining column name already exists in the target).

**No-SiK path**: When `length(k.gene)==0`, legacy (`kinases.R:133–147`) generates EI columns with a hardcoded `2*ncase` width (`for (j in 1:ncase) colnames(EI.df)[2*j-1]` = EI for conditions[1], `[2*j]` = EI for conditions[2]) — hardcoded for exactly 2 conditions. Upstream (`kinases.R:210–226`) iterates over `object@conditions` generically, producing columns for any number of conditions.

**Knob/default changes**: The `compute_kinase_condition` helper includes a guard at `kinases.R:106–111`: if the receiver cell type is absent from `ei_df` (e.g., a rare cell type with zero cells in one condition after `fct_drop`), it returns `s4=NA` and skips the EI merge. Legacy has no such guard — the `df_EI_1[, object@receiver]` column access at line 259 would error (subscript out of bounds) if the receiver column was absent. This is a bug fix in upstream.

**Risk classification**: (a) refactor — `compute_kinase_condition` extraction, `match()` vs left_join, condition-loop generalization. Single-gene-path bug fix and receiver-absent guard: (c) latent risks that would affect rare edge cases not tested in Phase C.

---

### Step 8 — `Cal_PDS`

| | Legacy | Upstream |
|---|---|---|
| File | `Incytr_functions_evaluation.R:369–408` | `evaluation.R:577–665` |

**Signature delta**

| Arg | Legacy | Upstream | Notes |
|---|---|---|---|
| `KPDS.weight` | NULL→0.5 | NULL→0.5 | |
| `cutoff_PDS` | NULL | NULL | |
| `cond_ref` | absent | NULL→conditions[1] | new; explicit reference condition |
| `cond_alt` | absent | NULL→conditions[2] | new; explicit alternative condition |

**Algorithmic delta**

Legacy (`eval.R:382–392`) computes the PDS using three separate assignment branches:
```r
df$score04[df$score03>0]  = ... + KPDS.weight*df$s4_1[...]
df$score04[df$score03==0] = ... + KPDS.weight*(df$s4_1[...]-df$s4_2[...])
df$score04[df$score03<0]  = ... - KPDS.weight*df$s4_2[...]
```

Upstream (`eval.R:556–559`) extracts this into `apply_condition_direction()`:
```r
ifelse(base_score > 0, weight * cond1_term,
ifelse(base_score < 0, -weight * cond2_term,
                        weight * (cond1_term - cond2_term)))
```

The semantics are identical. The `cond1_term` in upstream corresponds to `s4_1` (reference condition score, i.e., conditions[1]), and `cond2_term` to `s4_2`. The new `cond_ref`/`cond_alt` args allow the caller to swap the directionality, but default to `conditions[1]`/`conditions[2]` matching legacy behavior exactly.

**Knob/default changes**: None for pair mode with default `cond_ref`/`cond_alt`.

**Risk classification**: (a) cosmetic/refactor — helper extraction. (b) new `cond_ref/cond_alt` args matched by defaults.

---

### Step 9 — `Permutation_test`

| | Legacy | Upstream |
|---|---|---|
| File | `Incytr_functions_analysis.R:968–1047` | `analysis.R:779–896` |

**Signature delta**

| Arg | Legacy | Upstream | Notes |
|---|---|---|---|
| `K` | 0.5 | 0.5 | |
| `N` | 2 | 2 | |
| `nboot` | 10 | 10 | |
| `seed.use` | 1L | 1L | |
| `mean_method` | NULL | NULL | |
| `FDR_method` | "BH" | "BH" | |
| `cutoff_p_value` | NULL | NULL | |
| `n.cores` | absent | `max(1L, detectCores()-2L)` | new; parallel execution |

**Algorithmic delta — this is the largest divergence in the pipeline**

Legacy (`analysis.R:977–1006`) runs a simple `for (i in 1:nboot)` loop: shuffle idents, call `Expr_bygroup`, call `Cal_SigProb_ptest` (an internal copy of `Cal_SigProb` without fold-change), accumulate columns into a `matrix`. The loop processes the full `object@data` matrix on each iteration through `Expr_bygroup`.

Upstream (`analysis.R:795–894`) applies three optimizations:

1. **Pre-extraction** (lines 800–826): Before the loop, subset the data matrix to just pathway genes (`geneuse`), pre-extract per-condition barcode lists, pre-compute per-condition dense submatrices from the sparse matrix, and pre-compute gene index vectors (`gene_to_idx`). These are computed once and reused across all `nboot` iterations.

2. **Fast sigprob path** (lines 838–842): Each permutation iteration calls `grouped_expr_from_matrix()` (the `matrixStats::rowQuantiles` path) and then `compute_sigprob_fast()` (`analysis.R:146–157`). The fast path pre-expands `K^N = KN` and uses algebraic form `x^N / (x^N + KN)` without the hill wrapper. Both produce identical values.

3. **Streaming accumulation** (lines 848–865): Instead of storing all `nboot` SigProb matrices and then computing p-values at the end, upstream accumulates `exceed_counts` (integer vectors) on the fly. This reduces peak memory from O(n_paths × nboot × n_conditions) to O(n_paths × n_conditions).

4. **Chunked parallelism** via `run_permutation_loop()` (`analysis.R:1–22`): When `n.cores>1`, `mclapply` is used in chunks of `max(10, n.cores×4)` iterations. With `n.cores=1` (the safe default when matching legacy), the loop is serial and functionally identical to legacy.

**p-value formula change**: Legacy (`analysis.R:1012–1013`) computes:
```r
m <- t(sapply(object@SigProb[,(j+1)], rep, nboot)) - ptest.SigProb[[j]]
p = format(round(rowSums(m <= 0)/nboot, 4), nsmall = 4)
```
This counts permutations where `obs - perm <= 0`, i.e., permuted SigProb >= observed, using `<=` (greater-than-or-equal on the permuted side). The `format(round(...))` coerces to character before `p.adjust`.

Upstream (`analysis.R:870–874`) accumulates:
```r
state[[j]] + as.integer(result[[j]] >= obs_vals[[j]])
```
using `>=`. Then: `p <- exceed_count[[j]] / nboot`. No `format(round(...))` coercion — `p` stays numeric.

The `>=` vs `<=` inversion: legacy counts `obs - perm <= 0` ↔ `perm >= obs`, upstream counts `perm >= obs` directly. These are the same condition, so the p-value numerator is identical given the same permutation samples. The `format(round(...))` coercion in legacy converts to character and back before `p.adjust`; this could introduce floating-point rounding artifacts at the 4th decimal place that differ from the exact division in upstream. In practice, for `nboot=100`, all p-values are multiples of 0.01, so `format(round(..., 4))` changes nothing.

**RNG parity**: Both start with `set.seed(seed.use)` and call `replicate(nboot, sample.int(nC, nC))`. If `n.cores=1` in upstream and `mean_method="mean"` is used to skip `rowQuantiles`, the permutation samples are identical. With `n.cores>1` (the upstream default), `mclapply` forks separate processes with independent RNG streams, producing different permutation samples from legacy. This is the documented permutation-RNG divergence from the Phase C benchmark.

**Matched-knob fix**: The Phase C benchmark must pass `n.cores=1L` explicitly to upstream `Permutation_test` to achieve RNG parity.

**Risk classification**: (b) matched by explicit `n.cores=1L` for Phase C. (a) pre-extraction and streaming accumulation are pure performance improvements. (c) the `format(round(...))` vs direct-numeric p-value path is a latent risk for edge cases where p-values fall exactly at a rounding boundary — not observed in practice at `nboot=100`, but worth noting.

---

### Step 10 — `Export_results`

| | Legacy | Upstream |
|---|---|---|
| File | `Incytr_functions_evaluation.R:237–367` | `evaluation.R:329–551` |

**Signature delta**

| Arg | Legacy | Upstream | Notes |
|---|---|---|---|
| `object` | yes | yes | |
| `indicator` | FALSE | FALSE | |
| `format` | absent | `c("long","wide")` | new; factorial pivot |

**Algorithmic delta**

The legacy export function (`eval.R:246–256`) uses `left_join` to attach `sc_FC` columns using `Gene` as key. Upstream (`eval.R:437–448`) uses named-vector lookup:
```r
sender_vec <- setNames(df_sender$log2FoldChange, df_sender$Gene)
df$Ligand_sclog2FC <- unname(sender_vec[df$Ligand])
```
Semantically identical for non-NA cases.

For SigProb attachment, legacy (`eval.R:261–266`) uses `left_join(df, object@SigProb, join_by(Path))`. Upstream (`eval.R:453–456`) uses index-based subsetting: `object@SigProb[match(df$Path, object@SigProb$Path), -1]`. The match-based approach preserves the row order of `df` regardless of SigProb ordering, whereas the legacy left_join also preserves row order — so outputs are identical.

Legacy (`eval.R:337–341`) uses `inner_join` for kinase results: pathways absent from `kl.pathways` are dropped. Upstream (`eval.R:529–534`) uses `match()` to attach by index, a left-join semantics (absent paths get NA). This is a **behavior difference**: if a pathway survives `Permutation_test` but has no kinase data, legacy drops it, upstream keeps it with NA kinase columns. In the AD pair-mode runs, `Integr_kinasedata` covers all surviving pathways, so this difference does not fire in practice.

The indicator-generation loop in upstream (lines 517–527) was refactored from three separate `if(nrow(pr_FC)>0)` blocks to a for-loop over the omics layer names, with column names constructed programmatically.

**Knob/default changes**: The new `format` arg defaults to `"long"` in upstream for factorial objects but to `"wide"` behavior for legacy objects (since the function detects `object@options$mode != "factorial"`). For pair-mode objects, the format arg is effectively ignored.

**Risk classification**: (a) cosmetic/refactor — named-vector vs join for column attachment. (c) inner_join vs left-join for kinase attachment: latent risk if any pathway survives p-value filtering but has no matching kinase data. Not triggered by the AD driver.

---

## Shared helpers map

| Upstream helper | Location | Legacy equivalent |
|---|---|---|
| `validate_meta()` | `utils.R:10–26` | Inline in `Incytr_functions_object.R:66–80` and `Incytr_functions_analysis.R:114–130` (duplicated) |
| `compute_group_expr()` | `utils.R:234–238` | Inline in `Expr_bygroup`, `Find_highexp_gene`, `Integr_kinasedata` (triplicated) |
| `grouped_expr_from_matrix()` | `utils.R:206–229` | Inline with `setDT(...)[lapply(.SD, quantile,...)]` pattern in ~5 locations |
| `apply_path_cutoff()` | `utils.R:246–248` | Inline `object_update` calls at each cutoff filter site |
| `pathway_genes()` | `utils.R:241–243` | Inline `unique(c(pathways$Ligand, ...$Receptor, ...$EM, ...$Target))` at ~3 locations |
| `barcodes_bycondition()` | `utils.R:128–139` | `analysis.R:235–257` (same logic, hardcoded 2-condition) |
| `object_update()` | `utils.R:149–198` | `analysis.R:259–318` (same logic) |
| `.SIK_NAMES` | `utils.R:5–6` | Absent — legacy `object_update` hardcodes the six SiK column names inline at `analysis.R:304–309` |
| `compute_sigprob()` | `analysis.R:127–138` | Inline 4×left_join + hill() product in `Cal_SigProb` |
| `compute_sigprob_fast()` | `analysis.R:146–157` | Absent — legacy has no fast path; `Cal_SigProb_ptest` is a full copy of `Cal_SigProb` |
| `apply_condition_direction()` | `evaluation.R:556–559` | Inline three-branch assignment in `Cal_PDS:384–390` |
| `score_omics_layer()` | `evaluation.R:54–68` | Three separate `if(nrow(...)>0)` blocks in `Pathway_evaluation` |
| `compute_kinase_condition()` | `kinases.R:66–123` | Inline duplicate blocks for condition1 and condition2 in `Integr_kinasedata:154–220` |
| `logi()` | `math.R:8–10` | `Incytr_functions_evaluation.R:1–3` — byte-identical |
| `hill()` | `math.R:19–21` | `Incytr_functions_analysis.R:472–474` — byte-identical |
| `Cal_foldchange()` | `math.R:34–70` | `Incytr_functions_analysis.R:476–512` — refactored (see below) |
| `run_permutation_loop()` | `analysis.R:1–22` | Absent — legacy has no chunked parallel loop |
| `factorial_contrast_pair()` | `analysis.R` (factorial, not audited) | Absent |

`Cal_foldchange` is functionally equivalent but the upstream version takes `cond1_col`/`cond2_col` parameters (`math.R:35`) and uses `pmax` instead of `apply(df[,2:3], 1, max)` for `Vmax` — cosmetic performance improvement. The legacy version's `unlist(df[,2:3])` for computing the quantile threshold pools both conditions; upstream's `c(df[[cond1_col]], df[[cond2_col]])` does the same.

---

## S4 class delta

| Slot | Legacy (`object.R:16–39`) | Upstream (`Incytr_class.R:50–80`) | Notes |
|---|---|---|---|
| `data.raw` | `AnyMatrix` | **absent** | Removed; was only used by the DESeq2 `Cal_scFC` variant, which was retired |
| `data` | `AnyMatrix` | `AnyMatrix` | |
| `expr.bygroup` | `list` | `list` | |
| `pathways` | `data.frame` | `data.frame` | |
| `pathways_5steps` | `data.frame` | `data.frame` | |
| `meta` | `data.frame` | `data.frame` | |
| `sender` | `character` | `character` | |
| `receiver` | `character` | `character` | |
| `idents` | `AnyFactor` | `AnyFactor` | |
| `conditions` | `character` | `character` | |
| `animal_id` | **absent** | `character` | factorial mode |
| `expr.byanimal` | **absent** | `list` | factorial mode |
| `sigprob.byanimal` | **absent** | `data.frame` | factorial mode |
| `design` | **absent** | `ANY` | factorial mode |
| `contrasts` | **absent** | `list` | factorial mode |
| `SigProb` | `data.frame` | `data.frame` | |
| `p_value` | `data.frame` | `data.frame` | |
| `sc_FC` | `list` | `list` | |
| `pr_FC` | `data.frame` | `data.frame` | |
| `ps_FC` | `data.frame` | `data.frame` | |
| `py_FC` | `data.frame` | `data.frame` | |
| `Ack_FC` | **absent** | `data.frame` | new omics layer |
| `KGG_FC` | **absent** | `data.frame` | new omics layer |
| `Rme1_FC` | **absent** | `data.frame` | new omics layer |
| `Evaluation` | `data.frame` | `data.frame` | |
| `kl` | `data.frame` | `data.frame` | |
| `kl.pathways` | `data.frame` | `data.frame` | |
| `EI` | `list` | `list` | |
| `kl.explore` | `data.frame` | `data.frame` | |
| `options` | `list` | `list` | |

The `AnyDF` union (`data.frame | data.table`) declared in legacy `object.R:14` is absent from upstream — upstream S4 slot types use plain `data.frame`. This means assigning a `data.table` to an upstream slot will succeed only via S4 coercion (or fail if the union is not registered). This could matter if user code creates a data.table and assigns it directly.

---

## Known fragility points

### 1. `.SIK_NAMES` constant — the patched fragility

`utils.R:5–6` defines:
```r
.SIK_NAMES <- c("SiK_R_of_EM", "SiK_R_of_T", "SiK_EM_of_T",
                "SiK_EM_of_R", "SiK_T_of_R", "SiK_T_of_EM")
```
This constant is used in `object_update()` (`utils.R:186`) to iterate over the SiK columns when subsetting `object@EI` after a path filter. Before the patch, `object_update` referenced `.SIK_NAMES` in `utils.R:186` while the constant was absent — a dangling symbol that would produce `object not found` at runtime whenever a path cutoff triggered EI subsetting. This was fixed by adding the constant at the top of `utils.R`.

The same names are hardcoded inline in the legacy `object_update` at `analysis.R:304–309` as a vector of string literals. If a future refactor adds or renames a SiK case in `Integr_kinasedata`, both the `Integr_kinasedata` initialization block (`kinases.R:270–276`) and `utils.R:5–6` must be updated in sync. There is no runtime check that the two lists agree.

### 2. `object_update` — `pr_FC` row-count assumption

Both legacy (`analysis.R:276–283`) and upstream (`utils.R:169–175`) apply path filtering to the omics FC slots by temporarily assigning `df$Path <- Path.orignal` and then filtering. This assumes that `nrow(pr_FC) == nrow(pathways)` at the time of filtering, i.e., the FC slot has one row per pathway in the same order as `object@pathways`. If `Integr_multiomics` is called after a `cutoff_SigProb` filter has already shortened `object@pathways`, but the FC slot was never built, this assignment uses `Path.orignal` which holds the pre-filter pathway count — producing a recycled/truncated assignment or error. The legacy and upstream implementations share this assumption equally; it is not introduced by the refactor.

### 3. `Pathway_evaluation` — `abs.value` bug in legacy only

Legacy `evaluation.R:34–44` uses an `if / else if / else if / else if` chain to set `case[1:4]`. The first branch covers `"None"`, the second `"All"`, the third checks `!all(is.element(...))` and stops, and the fourth through seventh branches use `else if`, meaning only the first matching component sets its case flag. Legacy will silently fail to apply the correct mask for multi-element `abs.value` vectors (e.g., `c("Receptor","EM")`). Upstream uses parallel `if` statements, correctly handling any combination. The AD driver passes `abs.value="None"`, so this bug is never triggered, but it represents a correctness difference between implementations.

### 4. `Integr_kinasedata` — single-gene branch transpose bug in legacy

When `length(geneuse)==1`, legacy (`kinases.R:162–165`) does:
```r
df1 <- as.data.frame(M1)
colnames(df1) = geneuse
```
`M1` is `cells × 1` (from `data[geneuse, cells]`). `as.data.frame(M1)` on a matrix preserves rows as cells and the column as the gene. Then `df1$group_label` is assigned correctly. However, the downstream `setDT(df1)[lapply(.SD, quantile,...), keyby=group_label]` treats the column named `geneuse` as the gene column — this works.

For condition2, the same branch at lines 170–174 does the same. The real issue is that this branch does NOT transpose (`t(M)` is absent), unlike the multi-gene branch at lines 165–166. So `df1` has shape `cells × 1`, which happens to be correct for the `setDT` aggregation. Legacy's single-gene path is accidentally correct but structurally inconsistent with the multi-gene path. The upstream `compute_kinase_condition` helper (`kinases.R:74`) always transposes: `df <- as.data.frame(t(M))` regardless of gene count. The comment on line 72–73 explains this explicitly. Both paths produce the same output for the single-gene case because the data.table aggregation works either way.

### 5. `Permutation_test` — `n.cores` default in upstream

The upstream default `n.cores = max(1L, parallel::detectCores() - 2L)` means a caller who does not explicitly set `n.cores=1L` will get parallel execution on a multi-core machine. Parallel execution uses `mclapply` which forks with independent seeds, producing RNG-divergent p-values from legacy. This is not documented in the function's `@param` description. The fix for Phase E is to always pass `n.cores=1L` explicitly in the AD driver, or accept the documented permutation-RNG drift.

---

## Risk classification for v1 adoption decision

| Delta | Classification | Notes |
|---|---|---|
| `validate_meta` / `barcodes_bycondition` / `pathway_genes` helper extraction | (a) cosmetic | Pure refactor, identical semantics |
| `pathway_inference`: data.table joins, deterministic row order | (a) cosmetic | Path-string joins downstream, order irrelevant |
| `Expr_bygroup`: `grouped_expr_from_matrix` / `matrixStats::rowQuantiles` | (a) cosmetic | Numerically identical for type-7 quantiles |
| `Cal_SigProb`: named-vector vs join; `compute_fc` arg | (a)+(a) | No numeric change |
| `Integr_multiomics`: positional vs named-vector column attachment | (a) cosmetic | No numeric change; eliminates fragile `df[,6:13]` |
| `Pathway_evaluation`: `score.weight` expanded from 3→6 | (b) matched by knob | Extra terms multiply zero with no Ack/KGG/Rme1 data |
| `Pathway_evaluation`: `abs.value` bug fix | (c) latent risk | Only fires for `abs.value` vectors not used by AD driver |
| `Integr_kinasedata`: `compute_kinase_condition` extraction | (a) cosmetic | |
| `Integr_kinasedata`: receiver-absent guard | (c) latent risk | Fires only for rare cell types absent in one condition |
| `Integr_kinasedata`: no-SiK EI column generalization to N conditions | (a) cosmetic for pair mode | |
| `Cal_PDS`: `apply_condition_direction` extraction; new `cond_ref/cond_alt` | (a)+(b) | Defaults preserve legacy semantics |
| `Permutation_test`: pre-extraction + streaming accumulation | (a) performance | No semantic change |
| `Permutation_test`: `n.cores` default (parallel RNG) | (b) matched by explicit `n.cores=1L` | Must be set explicitly |
| `Permutation_test`: `format(round(...))` coercion vs direct numeric | (c) latent risk | Only fires at exact rounding boundary, not seen in practice |
| `Export_results`: named-vector vs join for SC/SigProb/p_value | (a) cosmetic | |
| `Export_results`: `inner_join` vs `match()` for kinase attachment | (c) latent risk | Fires if pathway survives but has no kinase data |
| `.SIK_NAMES` constant (was dangling reference, now patched) | patched — was (d) | No longer a blocker |
| New slots `animal_id`, `design`, `contrasts`, `expr.byanimal`, etc. | (a) additive | Unreachable in pair mode |
| New omics slots `Ack_FC`, `KGG_FC`, `Rme1_FC` | (a) additive | Unreachable in pair mode |
| Removed `data.raw` slot | (a) for pair mode | DESeq2 `Cal_scFC` variant was never called by AD driver |

**Summary**: There are no (d) blockers for v1 adoption. The three (c) items (`abs.value` bug, receiver-absent guard, `Export_results` inner vs left join) are all bug fixes in upstream and are unreachable with the AD driver's current call patterns. The one (b) item requiring an explicit knob is `n.cores=1L` in `Permutation_test`. All remaining deltas are (a) cosmetic/performance.

The Phase C bit-identical result is consistent with this audit: matched knobs (`mean_method="mean"`, `n.cores=1L`) suppress all algorithmic divergence paths, and the new upstream features (factorial, Ack/KGG/Rme1, parallel permutation) are unreachable in pair mode.
