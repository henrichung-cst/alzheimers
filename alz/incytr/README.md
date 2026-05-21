# alz/incytr/

Pair-mode Incytr production code for the AD phosphoproteomics project.
All math, scoring, and pathway construction lives in the upstream `incytr` R
package (`~/Projects/work/incytr/`); this directory holds the AD-side scripts
that build inputs, run the pair-mode grid, emit the transcript substrate, and
reshape outputs.

## Role of this module vs `alz/integration/`

| Module | Responsibility |
|---|---|
| `alz/incytr/` | Run-time drivers: build Seurat inputs, run `Incytr::Cal_pairwise_grid`, emit transcript substrate, orchestrate the per-contrast loop |
| `alz/integration/` | Consume outputs: reshape wide parquets into `receiver_cache/` for the viewer; build transcript-trace shards; manage config and cluster-spine loading |

## Entry point

```bash
bash alz/incytr/run_pair_mode.sh
```

Runs the 9 contrast pair-mode grid (3 diseases × 3 timepoints) and writes
wide parquets to `outputs/reports/incytr_pair_mode/wide/`.

Full end-to-end (inputs → R driver → viewer reshape):
```bash
bash alz/runners/main/run_pair_mode_pipeline.sh
```

## File inventory

Files are relocated here from `bench/` as Phase 2 of the Merged Evidence
Panel epic (`docs/plans/merged_evidence_panel.md`) completes. Placeholders
below are filled in by the corresponding phase items.

| File | Role | Phase item |
|---|---|---|
| `run_pair_mode.sh` | Per-contrast loop driver; calls `incytr_commandline.R` for each of 9 contrasts | 2.4 |
| `incytr_commandline.R` | R driver: calls `Incytr::Cal_pairwise_grid`; writes one wide parquet per contrast | 2.2 |
| `reconstruct_labels.R` | Post-processing helper: re-attaches cluster labels to driver output | 2.2 |
| `reconstruct_node_fc.R` | Post-processing helper: adds node LFC columns to driver output | 2.2 |
| `emit_expr_bygroup.R` | Transcript-substrate emitter: per-(cluster, Group) mean of `originalexp@data`; writes `outputs/reports/decomposition/levy_t5/transcript_per_cluster.parquet` | 2.2 |
| `build_pair_inputs.sh` | Input-prep orchestrator: calls `build_pair_seurat.R` and `build_input_gene_list.R`; writes to `data/derived/incytr_inputs/` | 2.3 |
| `build_pair_seurat.R` | Builds `incytr_obj.rds` from the snRNA-seq data | 2.3 |
| `build_input_gene_list.R` | Builds `input_gene_list.csv` for the driver | 2.3 |
| `export_decomposition_for_pair.py` | Reshapes per_cluster parquets into yuyu CSV format for the R driver; writes to `data/derived/incytr_inputs/` | 2.4 |

## Data layout

```
data/derived/incytr_inputs/          ← R driver inputs
   incytr_obj.rds
   pr_yuyu_deconvoluted.csv
   ps_yuyu_deconvoluted.csv
   py_yuyu_deconvoluted.csv
   kldata.csv                        (symlink → data/datasets/song/kinase/kldata_pspy.csv)
   allmarkers.csv
   HEG_df.csv
   input_gene_list.csv

outputs/reports/incytr_pair_mode/wide/   ← R driver outputs (9 wide parquets)
   ma_2mo_AppP_ma_2mo_WTyp_incytr_output.parquet
   ma_2mo_Ttau_ma_2mo_WTyp_incytr_output.parquet
   ma_2mo_ApTt_ma_2mo_WTyp_incytr_output.parquet
   ma_4mo_AppP_ma_4mo_WTyp_incytr_output.parquet
   ma_4mo_Ttau_ma_4mo_WTyp_incytr_output.parquet
   ma_4mo_ApTt_ma_4mo_WTyp_incytr_output.parquet
   ma_6mo_AppP_ma_6mo_WTyp_incytr_output.parquet
   ma_6mo_Ttau_ma_6mo_WTyp_incytr_output.parquet
   ma_6mo_ApTt_ma_6mo_WTyp_incytr_output.parquet

outputs/reports/decomposition/levy_t5/transcript_per_cluster.parquet
   ← transcript substrate (produced by emit_expr_bygroup.R)
   schema: (cluster, group, gene, value)
   31 clusters × 12 groups × ~30k genes
```

## Invariants

- **31² = 961 sender × receiver pairs per contrast.** Rank-deficient clusters emit NaN.
- **Active spine: levy_t5** (31 clusters, `min_cells = 5`, no rank gate). No other spine is reachable from code.
- **Pair pvalue is untrustworthy** — filter/rank on `|PDS|`, not pvalue.
- **Transcript substrate is contrast-invariant.** `emit_expr_bygroup.R` runs once per spine build, not per contrast.
- **Paths are resolved from repo root** via `git rev-parse --show-toplevel`; scripts can be invoked from any cwd.

## Status

As of 2026-05-20 (Merged Evidence Panel epic, Phase 2 in progress):
- `emit_expr_bygroup.R` output relocated to canonical substrate path (Item 1.1 done).
- Remaining files pending relocation from `bench/` (Items 2.2–2.4 pending).
