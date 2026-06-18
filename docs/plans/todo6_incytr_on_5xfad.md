# Plan: Run Pair-Mode Incytr on the 5xFAD Cohort

Date: 2026-06-18

---

## 1. Context and Template

The Song AD pair-mode Incytr pipeline runs `Incytr::Cal_pairwise_grid` over the
31-cluster levy_t5 spine for each of 9 contrasts (3 genotypes × 3 timepoints).
The 5xFAD extension is a direct analogue: same 31-cluster spine, same driver
(`alz/incytr_pair/incytr_commandline.R`), two tissues (cortex, hippocampus) × 4
timepoints = 8 contrasts. Everything — runner, builder scripts, pixi tasks, gene-use
derivation path — is already written. This plan documents the current state, the
one missing input, and two data-shape questions that must be answered before
declaring the run clean.

---

## 2. Song Pipeline Inputs — Complete Map

Every input `incytr_commandline.R` needs for the Song AD run:

| Input | Source | Song path |
|---|---|---|
| Seurat object | `build_pair_seurat.R` | `data/derived/incytr_inputs/incytr_obj.rds` |
| pr deconvoluted | `export_decomposition_for_pair.py` | `data/derived/incytr_inputs/pr_yuyu_deconvoluted.csv` |
| ps deconvoluted | same | `data/derived/incytr_inputs/ps_yuyu_deconvoluted.csv` |
| py deconvoluted | same | `data/derived/incytr_inputs/py_yuyu_deconvoluted.csv` |
| allmarkers | `build_input_gene_list.R` | `data/derived/incytr_inputs/allmarkers.csv` |
| kldata | Song kinase library | `data/derived/incytr_inputs/kldata.csv` |
| sce4 frozen gene.use | `extract_sce4_geneuse.R` | `data/incytr_frozen/sce4_geneuse/<c1>_<c2>.csv` |
| Species | env `SPECIES=mouse` | — |
| Gene key col | env `PR_GENE_COL=Gene Symbol` | — |

Song's deconvolution uses the **provenance** path:
`P_c = (N_total/N_c) × bulk × (specific_c / Σ_46 specific)` where `bulk` =
frozen per-group medians of raw Spectronaut quantities (`pr_median.csv`, ~40–55 MS
intensity units for typical proteins); `specific_c` = frozen `aggexp.csv`
(SCT/model-normalized sums, unrecoverable from the h5ad).

---

## 3. 5xFAD Data Inventory — What Exists

### 3a. Primary proteomics (Spectronaut TSVs)

All files on disk; the `.sne` blocker noted in prior memory is resolved. All four
contrast-relevant assays are parseable:

| Tissue | Assay | Path |
|---|---|---|
| cortex | total | `primary/cortex/proteomics/total/26Mar2026_MaleCtx5xFAD_Total_ProteinSiteReport.tsv` |
| cortex | IMAC/ST | `primary/cortex/proteomics/imac/20260527_104420_102325_LD_5xfad_IMAC_cortex_PTMSiteReport.tsv` |
| cortex | pY | `primary/cortex/proteomics/py/26Mar2026_MaleCtx5xFAD_pY_PTMSiteReport.tsv` |
| hippocampus | total | `primary/hippocampus/proteomics/total/20260501_094814_5xFAD_hippocampus_3-6-9-12mo_totalproteome_Report.tsv` |
| hippocampus | IMAC/ST | `primary/hippocampus/proteomics/imac/23Mar2026_MaleHippo5XFAD_IMAC_EnsemblDB_PTMSiteReport.tsv` |
| hippocampus | pY | `primary/ensembl_corrections/hippocampus/py/23Mar2026_MaleHippo5XFAD_pY_CorrectEnsembleDB__PTMSiteReport.tsv` |

### 3b. scRNA

`data/datasets/5xFAD/primary/scrna/reclustering/fivex_renamed_from_merged.RDS` (791 MB,
present). Condition vocabulary: `<geno>_<age>` (e.g. `TG_3mo`, `WT_9mo`). 31 named
clusters after dropping unnamed `cluster-N` entries. Both tissues extracted from
the single RDS.

### 3c. Derived Incytr inputs — current status

All upstream-derived files exist:

| File | Tissue | Status |
|---|---|---|
| `incytr_obj.rds` | cortex | OK (351 MB) |
| `incytr_obj.rds` | hippocampus | OK (398 MB) |
| `pr_deconvoluted.csv` | cortex | OK (36 MB; 8272 genes × 241 value cols) |
| `ps_deconvoluted.csv` | cortex | OK (107 MB; 39666 sites × 241 value cols) |
| `py_deconvoluted.csv` | cortex | OK (9.9 MB; 2734 sites × 241 value cols) |
| `pr_deconvoluted.csv` | hippocampus | OK (27 MB; 6852 genes × 219 value cols) |
| `ps_deconvoluted.csv` | hippocampus | OK (71 MB; 26961 sites × 219 value cols) |
| `py_deconvoluted.csv` | hippocampus | OK (6.7 MB; 1765 sites × 219 value cols) |
| `allmarkers.csv` | cortex | **MISSING** |
| `allmarkers.csv` | hippocampus | **MISSING** |
| `kldata.csv` (symlink) | cortex | **MISSING** (created by runner's `ensure_kldata()`) |
| `kldata.csv` (symlink) | hippocampus | **MISSING** (created by runner's `ensure_kldata()`) |

Column naming in deconvoluted files: `<condition>_<cell_type>` (e.g.
`TG_3mo_Microglia`). The driver's `slice_omics()` strips `^<condition>_` anchored prefix — this
matches the 5xFAD condition vocabulary exactly.

Mass-identity verified in `decompose_manifest.json`: max abs relative error
< 4×10⁻¹⁵ for all (tissue, channel, condition) combinations.

Kinase MEA has already been run for both tissues, both tracks (st/py), all 4 ages
(see `outputs/reports/kinase_attribution_5xfad/`). Contrast QC shows all 8 contrasts
are `primary` status with n_wt = 2–5, n_tg = 3–5 per age.

---

## 4. Per-Input Shape and Normalization Comparison: Song vs 5xFAD

### 4a. Deconvoluted protein values — magnitude difference

| Cohort | Gene example | Typical value in deconvoluted pr |
|---|---|---|
| Song | Gnai3 (ma_2mo_AppP_Astrocytes) | ~125 (arbitrary MS intensity unit from limma/median pipeline) |
| 5xFAD | Gnai3 (TG_3mo_Astrocytes) | ~46,000–70,000 (2^global-median-anchored log2 Spectronaut quantity) |

The ~400–1000x difference is because:

- **Song** `pr_median.csv` was built by Yuyu's lab in their normalized MS units
  (post-limma, median-grouped). These are raw DIA intensities normalized by a
  sample-level process that compresses the dynamic range.
- **5xFAD** `pr_bulk_linear.csv` was built by `_linear_group_bulk()` in
  `alz/cohorts/fivexfad/ingest.py`: `2^(global-median-centered log2 quantity)`,
  averaged per group. The exponentiation produces values in the 10³–10⁵ range
  for abundant proteins.

**This is NOT a RAISE item.** The Incytr driver's `proteomics_gene()` call pipes
`pr_1` and `pr_2` (already sliced to per-cluster columns) through
`limma::normalizeBetweenArrays()` (quantile normalization) before computing
log2FC and aFC. QN re-maps values by within-column rank, so the absolute scale
cancels and fold changes are scale-invariant. After the `pmax(pr, 1)` floor
(which does nothing for either cohort — Song values 40–130 and 5xFAD values
40k–130k are both already well above 1), both cohorts are processed identically
by the engine. The T-cell cohort also uses the same `2^log2_quantity` linear path
with values in the ~1k–20k range, and it runs without issue.

The within-cohort scale consistency is what matters: for each 5xFAD contrast,
`TG_Nmo` and `WT_Nmo` columns come from the same normalization chain, so their
ratio is correct.

### 4b. Transcript share (aggexp) — assay differences

| Cohort | Assay | Layer | Aggregation |
|---|---|---|---|
| Song | SCT (model-normalized) | `data` slot | `AggregateExpression(slot="data")` — one unrecoverable step |
| 5xFAD | `originalexp` (log-normalized RNA) | `data` layer | sparse matrix multiplication (`sub_dat %*% ind`) |

The aggexp is used only as a transcript share: `specific_c / Σ_cell_types specific`.
The share is a ratio that cancels absolute normalization differences. The key requirement
(met by both) is that the `data` layer contains log-normalized counts so that
aggregation sums are proportional to expression. The `originalexp` assay in the 5xFAD
Seurat is standard Seurat log-normalization (verified: the extract script checks that
the `data` layer is non-empty and the extraction succeeded).

**Not a RAISE item.**

### 4c. Cluster spine

5xFAD drops unnamed `cluster-N` cells (cohort-intrinsic filter) and resolves to the
same 31 levy_t5 clusters as Song. Confirmed by `decompose_manifest.json`:
`"cell_types": [31 entries]`. Name cross-check against
`data/incytr_frozen/v2_46clusters/spines/levy_t5/cluster_spine.csv` runs in
`build_5xfad_seurat.R` (stops on unknown names). Value columns in the deconvoluted
files show 31 cell types × 8 conditions = 248 possible; actual column counts
(241 cortex, 219 hippocampus) are lower because some cell types have 0 cells in
some conditions — expected and handled by `slice_omics()`.

**Not a RAISE item.**

### 4d. Condition metadata path in the driver

The Song Seurat carries `Group` = `ma_<age>_<geno>`; the driver uses `Group` when
present (line 134). The 5xFAD Seurat from `build_5xfad_seurat.R` sets `obj$condition`
(not `obj$Group`), so the driver falls through to the `condition` branch (line 136).
The T-cell cohort uses the same branch. The driver comment at line 131 explicitly names
this: "T-cell Seurat sets `condition` directly ... Use Group when present, otherwise
trust the builder's `condition`."

**Not a RAISE item.**

### 4e. Gene-key column

Song uses `PR_GENE_COL="Gene Symbol"` (two words, space). 5xFAD uses
`PR_GENE_COL="gene_symbol"` (underscore). Both `build_5xfad_seurat.R` and
`fivexfad_decompose.py` emit `gene_symbol`. The runner sets
`PR_GENE_COL="gene_symbol"` explicitly. The T-cell runner uses the same setting.

**Not a RAISE item.**

### 4f. SCE4_GENEUSE_DIR

Song sets `SCE4_GENEUSE_DIR` to use sce4's frozen per-pair node sets. 5xFAD has no
sce4 reference. `run_pair_mode_5xfad.sh` leaves `SCE4_GENEUSE_DIR` unset, so the
driver derives `DEG ∪ prG` per cluster per contrast — the same path as T-cells.

**Not a RAISE item.**

### 4g. Transgene force-include

The driver runs: `TRANSGENES <- intersect(c("App","Psen1","Mapt"), rownames(Data.input))`.
5xFAD is a hAPP/hPSEN1 model (Thy1 promoter knock-in); the mouse reference genome
captures both the transgenic and endogenous copies under `App` and `Psen1` symbols.
The force-include adds them to every cluster's `prg_by_cluster`. For 5xFAD, `App` and
`Psen1` should have genuine high expression in TG animals (unlike the Song AD model
where transgene expression in deconvoluted Microglia was flat). The force-include is
additive-idempotent: if `App` already passes `|aFC| > 1` it's already in `prG`; the
add changes nothing. If it doesn't (because it's excluded by the abundance adjustment
at low-copy clusters), the force-include adds it.

**Not a RAISE item** — the behavior is the same as Song and T-cells; the biology of
5xFAD's App/Psen1 expression is irrelevant to the pipeline's correctness.

---

## 5. RAISE Items

### RAISE-1: Bulk normalization units are not directly comparable between cohorts

The 5xFAD `pr_bulk_linear` values are ~400–1000× larger than Song's `pr_median`
values for the same proteins. This is a normalization-pipeline difference, not an
error: each cohort's deconvolution and Incytr run is self-contained, so within-cohort
fold changes are correct. **Cross-cohort comparison of absolute deconvoluted protein
levels is not meaningful** without renormalization. If any downstream analysis (viewer,
integration report) plans to compare 5xFAD deconvoluted values numerically to Song's
deconvoluted values, this must be addressed first.

This plan does not address cross-cohort comparison. If that use case arises, the fix
is to re-derive 5xFAD `pr_bulk_linear` from a normalized scale that matches Song's
(e.g. median-centering to the same reference set) or to use only fold-change metrics
(PDS, sclog2FC) which are scale-invariant. Raise to user before implementing any
cross-cohort value comparison.

### RAISE-2: Small sample counts at some contrasts

Cortex 6mo: n_wt=2, n_tg=4. Cortex 9mo: n_wt=2, n_tg=3. Hippocampus 9mo: n_wt=2,
n_tg=3. For Incytr pair-mode, the key statistic is `SigProb` (fraction of bootstrapped
resamples where the path's direction is consistent). With n=2 per group, the cell-count
support for the per-(cluster, condition) expression is lower. `SigProb` will be
pessimistic for cell types with few cells in the sparse groups.

This is not a pipeline error — it mirrors the Song cohort's sparse coverage for some
genotype × timepoint cells (e.g. Cholinergic-Neurons at 2mo in Song had 1 cell/condition).
The driver runs all pairs regardless of cell-count sparsity. The significance filter
(`SigProb > 0.1`) will naturally attenuate low-support paths. No pipeline change needed.

**Raise to user only if they expect uniform coverage across all 8 contrasts.** The
n=2 WT groups at 6mo and 9mo are a proteomics sample-count limitation, not a pipeline
gap.

---

## 6. What Is Missing Before Running

Two items block the `preflight()` check in `run_pair_mode_5xfad.sh`:

### 6a. allmarkers.csv (both tissues) — must build

```
pixi run 5xfad-build-incytr-gene-list
```

This runs `build_5xfad_input_gene_list.R cortex` then
`build_5xfad_input_gene_list.R hippocampus`. Each reads the per-tissue
`incytr_obj.rds` and calls `FindAllMarkers(logfc.threshold=0.1, only.pos=TRUE)` with
the `presto` backend (sequential plan — the comment in the script explains why
`multisession` deadlocks on 241 idents). Output: `allmarkers.csv` under each tissue
input dir.

Expected runtime: substantial (241 Type_condition idents × 40k+ cortex cells). Plan
to run this as a background shell job, not interactively.

### 6b. kldata.csv symlink — auto-created by runner

`run_pair_mode_5xfad.sh`'s `ensure_kldata()` creates the symlink at run time. No
explicit step needed.

---

## 7. Execution Plan

All steps in order; no step should be skipped.

### Step 1: Build allmarkers (both tissues)

```bash
pixi run 5xfad-build-incytr-gene-list
# produces:
#   data/derived/5xfad_incytr_inputs/cortex/allmarkers.csv
#   data/derived/5xfad_incytr_inputs/hippocampus/allmarkers.csv
```

Verify after:

```bash
python3 -c "
import pandas as pd
for t in ['cortex', 'hippocampus']:
    df = pd.read_csv(f'data/derived/5xfad_incytr_inputs/{t}/allmarkers.csv')
    print(t, df.shape, df.columns.tolist()[:5])
    assert {'gene','cluster','avg_log2FC','p_val'}.issubset(df.columns), 'missing required cols'
    print('  cluster sample:', df.cluster.value_counts().head(3).to_dict())
"
```

The `cluster` column must have entries matching `<cell_type>_<condition>` format
(e.g. `Microglia_TG_3mo`) to satisfy the driver's `paste0(cl, "_", c(condition1, condition2))`
lookup.

### Step 2: Smoke run (cortex, TG_3mo vs WT_3mo, nboot=2)

```bash
bash alz/incytr_pair/run_pair_mode_5xfad.sh --smoke cortex
# output: outputs/reports/incytr_pair_mode_5xfad/cortex/wide_smoke/
```

The smoke run confirms: driver loads both Seurat and deconvoluted CSVs, `slice_omics`
finds columns, `proteomics_gene` returns a non-empty prG set, `Cal_pairwise_grid`
completes at least one pair. Check the log for:
- `[pair-driver] per-cluster gene.use sizes:` — should show median > 0
- `[pair-driver] precompute-trimean gene union:` — should be several thousand genes
- No `Error` or `0 paths enumerated for all pairs` messages

If the smoke succeeds (parquet exists and is non-zero), proceed.

### Step 3: Full run

```bash
NPAIR_WORKERS=3 N_CHUNK_MULT=8 bash alz/incytr_pair/run_pair_mode_5xfad.sh
# outputs:
#   outputs/reports/incytr_pair_mode_5xfad/cortex/wide/    (4 parquets)
#   outputs/reports/incytr_pair_mode_5xfad/hippocampus/wide/  (4 parquets)
# significance filter applied per tissue after all contrasts complete
```

Or via pixi:

```bash
pixi run 5xfad-incytr
```

The runner is resumable: existing parquets are skipped. The significance filter runs
automatically per tissue after all 4 contrasts complete (`SigProb > 0.1 AND |PDS| >= 0.2`,
uncapped, no FDR arm). No sce4 parity gate applies (5xFAD has no reference).

Memory note: The 5xFAD Seurat objects are larger than Song's (351–398 MB on disk vs
Song's ~27k cells). The RSS budget comments in the Song runner (N_CHUNK_MULT=48 for
heavy contrasts) may apply here. Start with the default `N_CHUNK_MULT=8` from the runner
and monitor RSS; raise if needed.

---

## 8. Expected Output Shape

| Tissue | Contrasts | Pairs per contrast | Parquets |
|---|---|---|---|
| cortex | 4 (TG_{3,6,9,12}mo vs WT_Nmo) | 31² = 961 | 4 |
| hippocampus | 4 | 31² = 961 | 4 |

Each parquet: rows = pathways (Sender → Ligand → Receptor → EM → Target chains),
columns include `Sender`, `Receiver`, `Ligand`, `Receptor`, `EM`, `Target`,
`SigProb_<c1>`, `SigProb_<c2>`, `PDS`, `TPDS`, `PPDS`, `PhPDS`, `Ligand_pr_log2FC`,
`Ligand_sclog2FC`, etc.

After filtering: `SigProb > 0.1 (either condition) AND |PDS| >= 0.2` applied
in-place to `wide/`. The filtered files replace the unfiltered ones in `wide/`.

---

## 9. Open Questions / Raise to User

1. **Cross-cohort comparison**: Is the plan to compare 5xFAD Incytr outputs
   numerically against Song outputs (e.g. PDS magnitude comparison)? PDS and
   sclog2FC are fold-change-based and scale-invariant, so cross-cohort comparison
   of those metrics is valid. Raw deconvoluted protein values should not be compared
   directly. Raise if the integration/viewer layer needs to display them side by side.

2. **Small-n contrasts (n_wt=2)**: Cortex 6mo and 9mo, hippocampus 9mo have only 2
   WT animals. Incytr does not gate on sample count; `SigProb` will be conservative
   for these contrasts. Accept this or flag these as lower-confidence contrasts in
   downstream reporting?

3. **No sce4-style reference for 5xFAD**: The 5xFAD run uses derived `DEG ∪ prG`
   (the T-cell path), not a frozen pre-computed gene.use set. There is no external
   published reference to validate path-set fidelity against. The run will produce
   output, but there is no parity gate analogous to `verify-incytr-sce4`. Accept this
   or plan a synthetic benchmark pair?

4. **Viewer integration**: The 5xFAD Incytr outputs are not yet wired into the unified
   viewer. That would require adding 5xFAD contrasts to the viewer's payload builder
   (analogous to how T-cell outputs are surfaced). Not planned in this step; raise if
   needed.

5. **allmarkers runtime**: `FindAllMarkers` on 241 Type_condition idents (31 clusters
   × 8 conditions) may take 30–60+ min even with `presto`. The cortex object has 40k
   cells; hippocampus is similar. Run as a background job and check for completion
   before proceeding to the smoke run.

---

## 10. Files Touched in This Plan

**No production code changes.** All scripts are already written. The plan only builds
missing inputs and runs existing tasks.

Build inputs:
- `alz/incytr_pair/build_5xfad_input_gene_list.R` (existing, run via pixi task)

Run:
- `alz/incytr_pair/run_pair_mode_5xfad.sh` (existing)
- driver: `alz/incytr_pair/incytr_commandline.R` (existing, no changes)

Output root: `outputs/reports/incytr_pair_mode_5xfad/`

Relevant SSOT references:
- `alz/incytr_pair/README.md` — method/application boundary
- `alz/cohorts/fivexfad/ingest.py` — `run_export_bulk()` provenance
- `alz/ingest/fivexfad_decompose.py` — deconvolution provenance
- `alz/ingest/fivexfad_scrna_extract.R` — aggexp extraction
- `data/derived/5xfad_incytr_inputs/{cortex,hippocampus}/decompose_manifest.json` — verified mass-identity
