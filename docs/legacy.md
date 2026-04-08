<legacy_method> # Legacy Deconvolution Method: Proportional Decomposition
## Overview
Before A_obs + DESP was adopted, the pipeline used a **proportional decomposition** method that distributed bulk phosphoproteomics signal across cell types using per-gene scRNA-seq expression shares. This document describes what that method did, how it differs from the current approach, and what properties it preserved that the replacement lost.
---
## The Legacy Formula
For each gene `g`, sample `s`, and cell-type cluster `c`:
```
deconvoluted(g, s, c) = size_factor(s, c) × bulk_MS(g, s) × scRNA(g, s, c) / total_scRNA(g, s)
```
Where:
- **`bulk_MS(g, s)`** — raw bulk mass spec intensity for gene `g` in sample `s`
- **`scRNA(g, s, c)`** — aggregated scRNA-seq expression of gene `g` in cluster `c` for sample `s` (from `aggexp.csv`)
- **`total_scRNA(g, s)`** — sum of scRNA expression of gene `g` across ALL clusters in sample `s`: `Σ_c scRNA(g, s, c)`
- **`size_factor(s, c)`** — cluster size correction: `total_cells(s) / cells_in_cluster(s, c)`
### In Plain Terms
The bulk MS signal for each gene is apportioned to each cell-type cluster **proportional to that cluster's share of the total scRNA expression for that specific gene in that specific sample**. The size factor then corrects for cluster size (smaller clusters get scaled up, larger clusters get scaled down).
### What Makes It Condition-Specific
The critical term is **`scRNA(g, s, c) / total_scRNA(g, s)`** — the fraction of gene `g`'s scRNA signal that comes from cluster `c` in sample `s`. Because `aggexp.csv` contains per-sample aggregated expression:
- Sample `ma_4mo_WTyp` has its own scRNA expression profile per cluster
- Sample `ma_4mo_AppP` has a **different** scRNA expression profile per cluster
- Therefore, the same gene's signal gets distributed differently across cell types depending on the sample/condition
**This is why the legacy method does NOT have the rank-1 problem.** Each sample's decomposition uses sample-specific scRNA weights, so the resulting phosphosite profiles genuinely differ between conditions.
---
## Input Data
### aggexp.csv (Aggregated scRNA Expression)
- **Shape:** ~1,078 rows × ~30,568 columns
- **Rows:** One per (cluster, sample) combination — e.g., `Astrocytes` in `ma_2mo_WTyp`
- **Columns:** ~30,567 gene symbols + 1 `Sample` column
- **Values:** Aggregated (summed or averaged) scRNA-seq expression per gene per cluster per sample
- **Size:** 215 MB
- **Key property:** Expression values vary by sample, so the same gene in the same cluster has different values across conditions
### yuyu_clustersize.csv (Cluster Sizes)
- **Shape:** 46 clusters × 24 samples
- **Values:** Number of cells in each cluster for each sample
- **Ontology:** Fine-grained 46-class labels (e.g., `Excitatory-Rorb`, `Foxp2-Excitatory-Neurons-layers-6-and-2-3`, `Erbb4-VIP-inhibitory-neurons`)
### Bulk Mass Spec
- Standard bulk phosphoproteomics CSV (16K phosphosites × 24 samples)
---
## Key Properties
### Condition-Specific Variation: PRESERVED
Cross-condition LFC correlations for the legacy output average **r ≈ 0.4–0.6**, compared to r ≈ 1.0 for the current A_obs + DESP output:
| Cell Type | Timepoint | Legacy avg r | Current avg r |
|-----------|-----------|:------------:|:-------------:|
| Glut (Excitatory) | 2mo | 0.538 | 1.000 |
| Glut (Excitatory) | 4mo | 0.393 | -0.333* |
| Microglia | 2mo | 0.500 | 1.000 |
| Microglia | 4mo | 0.412 | -0.333* |
| Astrocytes | 2mo | 0.438 | 1.000 |
| OPCs | 4mo | 0.420 | -0.333* |
| Oligodendrocytes | 6mo | 0.596 | 1.000 |
*(-0.333 values in the current method are artifacts of near-zero LFC variance interacting with the pseudocount)*
**Interpretation:** In the legacy output, knowing that a phosphosite is upregulated under Ttau tells you relatively little about whether it's also upregulated under AppP (r ≈ 0.4). In the current output, it tells you everything (r ≈ 1.0) — because the profiles are identical.
### Cell-Type Resolution
- **Legacy:** 46 fine-grained clusters (e.g., `Excitatory-Rorb`, `Excitatory-Pyramidal`, `Foxp2-Excitatory-Neurons-layers-6-and-2-3`)
- **Current:** 8 coarse clusters after label mapping (Glut, Gaba, Astrocytes, Microglia, Oligodendrocytes, OPCs, Endothelial cells, Medium spiny neurons)
The legacy method operated at higher cell-type resolution. The 46→8 collapse was a separate decision in the transition, not inherent to A_obs + DESP.
---
## Known Problems with the Legacy Method
The legacy method was replaced due to several implementation issues identified in the audit (`deconv/docs/desp_vs_legacy_audit.md`):
### 1. Label Mapping Failure
The `remove_number()` function attempted to strip trailing digits from cluster names to match them across samples, but it was fragile:
- For 23 of 24 samples, most or all labels failed to map correctly
- Labels containing meaningful digits were corrupted: `Foxp2-Excitatory-Neurons-layers-6-and-2-3` → `Foxp2-Excitatory-Neurons-layers-`
- 1,032 of 1,078 label instances were unmapped
### 2. Fabricated Signal for Missing Combinations
Missing (sample, cluster) combinations were filled with rows of pseudocount values (`min_positive / 10000 ≈ 4.6e-9`). This injected false positive signal where none existed. Out of 1,078 total rows, 1,058 were fabricated.
### 3. Pseudocount Poisoning
All zero values in `aggexp.csv` were replaced with the pseudocount before decomposition:
```python
new = list(map(lambda x: str(min/10000) if x == "0" else x, i))
```
This transforms true absence of expression into positive signal, creating systematic noise.
### 4. Zero-Count Instability
When a cluster had zero cells in a sample, the size factor `total_cells / 0` produced infinity. No guard was in place.
### 5. Gene Filtering
Only genes present in both bulk MS and `aggexp.csv` were retained (6,687 of 6,746 proteins). 59 proteins were dropped.
---
## Why It Was Replaced
The audit concluded that the implementation failures (label corruption, fabricated signal, infinite scale factors) made the legacy output unreliable. A_obs + DESP was adopted as a more principled approach:
- **A_obs** uses validated cell-count fractions from matched scRNA-seq
- **DESP** solves a regularized optimization problem rather than applying a heuristic formula
- No pseudocount imputation or label-mapping heuristics
- Full feature coverage (all bulk proteins retained)
---
## What the Replacement Lost
The transition to A_obs + DESP fixed the implementation bugs but **introduced the rank-1 structural limitation**:
| Property | Legacy (Proportional) | Current (A_obs + DESP) |
|----------|:---------------------:|:----------------------:|
| Condition-specific LFC profiles | Yes (r ≈ 0.4–0.6) | No (r ≈ 1.0) |
| Per-gene, per-sample decomposition weights | Yes (from aggexp.csv) | No (single X matrix) |
| Cell-type resolution | 46 clusters | 8 clusters |
| Mathematically principled | No (heuristic formula) | Yes (regularized regression) |
| Label handling | Broken (1032/1078 unmapped) | Correct |
| Zero handling | Pseudocount imputation | True zeros preserved |
| Fabricated signal | Yes (1058 fabricated rows) | No |
| Feature coverage | 6,687 genes | All 6,746 genes |
**The core trade-off:** The legacy method used richer information (per-gene, per-sample, per-cluster scRNA expression from `aggexp.csv`) but applied it through a buggy heuristic. The replacement uses a sound mathematical framework but discards all per-sample variation by design.
---
## Why the Legacy Method Preserved Condition-Specific Signal
The key difference is **what information feeds into the decomposition weights**:
### Legacy: Per-Gene, Per-Sample scRNA Shares
```
weight(g, s, c) = scRNA(g, s, c) / Σ_c scRNA(g, s, c)
```
This weight varies across:
- **Genes** — different genes have different expression patterns across cell types
- **Samples** — the same gene in the same cell type has different expression across conditions/timepoints because `aggexp.csv` contains per-sample aggregated expression
- **Cell types** — by definition
Because the weight varies by sample, two conditions (e.g., WTyp and AppP) distribute the same gene's bulk signal differently across cell types → the resulting profiles differ by condition.
### Current: Global Cell-Type Profiles × Scalar Fractions
```
deconv(g, s, c) = A_obs(s, c) × X_DESP(g, c)
```
- `X_DESP(g, c)` is the same for all samples — one value per (gene, cell type)
- `A_obs(s, c)` varies by sample but is the same for all genes — one scalar per (sample, cell type)
Because `X_DESP` doesn't vary by sample, all conditions get identical gene-level profiles per cell type, scaled by a single composition scalar.
### The Information Gap
The legacy method consumed **~30,567 genes × 46 clusters × 24 samples = ~33.7 million** scRNA expression values to determine decomposition weights.
The current method consumes **24 samples × 10 cell types = 240** composition fractions. All gene-level variation in decomposition weights comes from DESP solving a single regression across all samples.
---
## Implications for a New Approach
Any replacement method should aim to:
1. **Preserve condition-specific variation** (like legacy) — the decomposition weights should differ between conditions for the same gene and cell type
2. **Use mathematically principled inference** (like DESP) — not heuristic formulas with pseudocount imputation
3. **Handle missing data and edge cases correctly** (like DESP) — no fabricated signal, no infinite scale factors
4. **Leverage the rich per-sample scRNA data** (like legacy) — the `aggexp.csv` matrix contains genuine per-sample, per-cluster, per-gene expression that is currently unused by DESP
The ideal approach would combine DESP's mathematical rigor with the legacy method's use of per-sample scRNA expression as decomposition weights. The `aggexp.csv` data (or an equivalent per-sample, per-cluster expression matrix) is the key input that enables condition-specific decomposition — it should not be reduced to a scalar composition matrix before deconvolution.
---
## File Locations
```
Legacy script (git):          git show e148ba4:deconv/code/protein-ms-by-cell-type.py
Legacy outputs:               data/incytr_collections/song/proteomics/legacy/ps_yuyu_deconvoluted.csv
Legacy aggexp input:          data/incytr_collections/song/method_records/legacy_deconvolution_20250721/inputs/aggexp.csv
Legacy cluster sizes:         data/incytr_collections/song/method_records/legacy_deconvolution_20250721/inputs/yuyu_clustersize.csv
Audit document:               deconv/docs/desp_vs_legacy_audit.md
Transition document:          data/incytr_collections/song/method_records/aobs_desp_standardized/docs/deconvolution-transition-aobs-desp.md
```
 </legacy_method>
