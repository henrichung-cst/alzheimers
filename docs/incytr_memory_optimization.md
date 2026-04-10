# Incytr Memory Optimization Investigation

**PR**: henrichung-cst/alzheimers#1  
**Branch**: `subclass-attribution`  
**Incytr source**: `/home/hchung/Projects/work/incytr/`

## Problem

The Phase 1 Incytr integration pipeline OOMs on a 30GB RAM machine. The pipeline successfully completes through PDS computation (expression scoring, kinase structural + activity integration) but crashes during `Permutation_test()`. Earlier runs with larger pathway counts also OOMed during `Integr_kinase_enrichment()` and `Export_results()`.

## Environment

- 30GB RAM, 8GB swap
- R via `micromamba run -n incytr`
- Incytr installed from `/home/hchung/Projects/work/incytr/` via `R CMD INSTALL`
- Dataset: 3,161 cells × 30,567 genes (Song snRNA-seq, 4mo males, WT vs App)
- 22 cell types, sender=Microglia-PVM (185 cells), receiver=L5 IT (37 cells)

## Memory profile at crash point

| Stage | Peak RSS | Status |
|-------|----------|--------|
| Load expression matrix (30K × 3K sparse) | ~1 GB | OK |
| Load IncytrDB (raw: 7.7M edges) | ~4 GB | OK |
| Filter DB to expressed genes | ~4 GB | OK |
| `pathway_inference` (3,544 pathways at 50% threshold) | ~6 GB | OK |
| `Expr_bygroup` + `Cal_SigProb` + `Cal_scFC` | ~8 GB | OK |
| `Export_results` (expression-only baseline) | ~10 GB | OK |
| `Integr_kinasedata` (61K filtered kldata rows) | ~12 GB | OK |
| `Integr_kinase_enrichment` (18K filtered kl_output rows) | ~16 GB | OK |
| `Cal_PDS` | ~18 GB | OK |
| `Permutation_test` (nboot=50, n.cores=1) | >30 GB | **OOM** |

## Bottleneck 1: `Permutation_test` (critical)

**File**: `incytr/R/analysis.R:467-570`

**What it does**: For each of `nboot` permutations, shuffles cell identity labels and recomputes `Expr_bygroup` + `Cal_SigProb` for all pathways. Collects results in a `nrow(SigProb) × nboot` matrix per condition.

**Why it OOMs**: The function pre-densifies the expression matrix at line 491:

```r
dense_cond1 <- as.matrix(object@data[geneuse, celluse$condition1])
```

With `geneuse` = ~2,300 unique pathway genes and `celluse$condition1` = ~1,600 cells, `dense_cond1` is a 2,300 × 1,600 dense matrix (~30MB). Two conditions = ~60MB. This is fine.

The real cost is inside `run_one_permutation()` (line 496): each call to `grouped_expr_from_matrix` + `compute_sigprob` operates on 3,544 pathways. With `nboot=50` and `lapply` (single-core), R accumulates all 50 results before garbage collecting. The `results` list (line 522-523) holds 50 × 2 vectors of length 3,544 = 354K doubles × 8 bytes ≈ only 2.7MB. So the results list itself isn't the problem.

**Likely cause**: The per-permutation computation creates large intermediate data.tables inside `compute_sigprob` and `grouped_expr_from_matrix` that aren't freed between iterations because R's garbage collector doesn't run between `lapply` iterations when memory pressure is gradual. By permutation ~30-40, accumulated unfree'd temporaries push past 30GB.

**Suggested fixes** (in order of implementation difficulty):

1. **Force GC between permutations** (minimal change):
   ```r
   results <- vector("list", nboot)
   for (i in seq_len(nboot)) {
     results[[i]] <- run_one_permutation(i)
     if (i %% 10 == 0) gc(verbose = FALSE)
   }
   ```

2. **Streaming p-value computation** (avoids storing all permutation results):
   ```r
   # Instead of collecting nboot vectors then computing p-values,
   # accumulate counts incrementally
   exceed_count <- matrix(0L, nrow(object@SigProb), n_conditions)
   for (i in seq_len(nboot)) {
     perm_result <- run_one_permutation(i)
     for (j in seq_len(n_conditions)) {
       obs <- object@SigProb[, j + 1]
       exceed_count[, j] <- exceed_count[, j] + (perm_result[[j]] >= obs)
     }
     if (i %% 10 == 0) gc(verbose = FALSE)
   }
   p_values <- exceed_count / nboot
   ```
   This eliminates the `ptest.SigProb` matrices (lines 525-533) and the `results` list entirely.

3. **Pathway batching** (for very large pathway counts):
   Split pathways into batches of N (e.g., 1,000), run permutation per batch, merge p-values. This bounds memory regardless of pathway count.

## Bottleneck 2: `Integr_kinase_enrichment` scaling (secondary)

**File**: `incytr/R/kinases.R:768-827`

With `kldata = NULL` (Scenario C), `Integr_kinase_enrichment` calls `filter_kl_evidence_to_pathways()` which cross-references kl_evidence against all pathway genes. With our pre-filtering (18K kl_output rows × 3,544 pathways), this completed. But at the 20% expression threshold (788K pathways × 25K kl_output), it OOMed.

When both `kldata` and `kl_output` are provided (Scenario D, line 782), it falls through to `Integr_kinasedata()` which does a more complex cross-join. Our wrapper pre-filters kldata to pathway genes (101K → 61K rows) which helps.

**Suggested fix**: Inside `Integr_kinasedata`, filter `kldata` to pathway genes early (before any cross-join). This is currently done by the caller but could be built into the function:
```r
# At top of Integr_kinasedata, after validation:
pathway_genes <- unique(c(pathdf$Receptor, pathdf$EM, pathdf$Target))
kldata <- kldata[kldata$gene %in% pathway_genes | 
                 kldata[["motif.geneName"]] %in% pathway_genes, ]
```

## Bottleneck 3: IncytrDB size (background)

**File**: `incytr/data/*.rda`

| Layer | Edges | Approx memory |
|-------|-------|---------------|
| Layer 1 (L→R) | 6,707 | ~1 MB |
| Layer 2 (R→EM) | 3,511,888 | ~1.5 GB |
| Layer 3 (EM→T) | 4,232,358 | ~1.8 GB |

Loading all three layers consumes ~3.3GB before any analysis begins. The wrapper pre-filters to expressed genes (reducing to 2.8M + 3.7M edges), but the full tables must be loaded first.

**Suggested fix** (longer-term): Store DB layers as on-disk indexed files (e.g., `fst` or `arrow` format) with `from`/`to` indices, loading only edges matching the gene set. This would cut initial memory from ~3.3GB to <100MB for typical analyses.

## Reproduction

```bash
cd /home/hchung/Projects/work/alzheimers

# Python adapters (all verified, produce intermediates/)
micromamba run -n alzheimers python3 code/integration/adapters/export_expression.py
micromamba run -n alzheimers python3 code/integration/adapters/export_kldata.py
micromamba run -n alzheimers python3 code/integration/adapters/export_kl_output.py
micromamba run -n alzheimers python3 code/integration/adapters/export_phospho.py

# R pipeline (OOMs at Permutation_test on 30GB)
micromamba run -n incytr Rscript code/integration/wrappers/run_incytr.R
```

The R wrapper's last successful output before crash:
```
=== Integrating kinase evidence ===
  kldata (filtered to pathway genes): 61281 rows, 369 kinases
  Structural kinase integration complete.
  kl_output (filtered): 18849 rows, 114 kinases
  Activity kinase integration complete.
Computing PDS...
=== Permutation testing (3 seeds x 50 permutations) ===
  Seed 1...Interrupted system call ; error code 4
```

## Key files

| File | Purpose |
|------|---------|
| `incytr/R/analysis.R:467-570` | `Permutation_test` — primary OOM source |
| `incytr/R/analysis.R:229-290` | `grouped_expr_from_matrix`, `compute_sigprob` — called per permutation |
| `incytr/R/kinases.R:768-827` | `Integr_kinase_enrichment` — secondary scaling issue |
| `incytr/R/kinases.R:399-600` | `Integr_kinasedata` — kldata cross-join |
| `alzheimers/code/integration/wrappers/run_incytr.R` | Wrapper that triggers the OOM |
| `alzheimers/code/integration/intermediates/` | Pre-computed adapter outputs (expression, kldata, kl_output, phospho) |

## Acceptance criteria

1. `Permutation_test(nboot=50, n.cores=1)` completes on 30GB RAM with 3,544 pathways and 3,161 cells
2. No regression in permutation p-value accuracy (same results as current implementation on small datasets)
3. `Integr_kinasedata` internally filters kldata to pathway genes (defensive, regardless of caller pre-filtering)
