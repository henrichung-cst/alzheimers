# alz/incytr_pair/

Pair-mode Incytr production code for the AD phosphoproteomics project.
All math, scoring, and pathway construction lives in the upstream `incytr` R
package (`~/Projects/work/incytr/`); this directory holds the AD-side scripts
that build inputs, run the pair-mode grid, emit the transcript substrate, and
reshape outputs.

## Role of this module vs `alz/integration/`

| Module | Responsibility |
|---|---|
| `alz/incytr_pair/` | Run-time drivers: build Seurat inputs, run `Incytr::Cal_pairwise_grid`, emit transcript substrate, orchestrate the per-contrast loop |
| `alz/integration/` | Consume outputs: reshape wide parquets into `receiver_cache/` for the viewer; build transcript-trace shards; manage config and cluster-spine loading |

## Entry point

```bash
bash alz/incytr_pair/run_pair_mode.sh
```

Runs the 9 contrast pair-mode grid (3 diseases × 3 timepoints) and writes
wide parquets to `outputs/reports/incytr_pair_mode/wide/`.

Full end-to-end (inputs → R driver → viewer reshape):
```bash
bash alz/runners/main/run_pair_mode_pipeline.sh
```

Build driver inputs only:
```bash
bash alz/incytr_pair/build_pair_inputs.sh
```

## File inventory

| File | Role |
|---|---|
| `run_pair_mode.sh` | Per-contrast loop driver; calls `incytr_commandline.R` for each of 9 contrasts. Reads from `data/derived/incytr_inputs/`; writes to `outputs/reports/incytr_pair_mode/wide/`. |
| `incytr_commandline.R` | R driver: calls `Incytr::Cal_pairwise_grid`; writes one wide parquet per contrast to `outputs/reports/incytr_pair_mode/wide/`. |
| `reconstruct_labels.R` | Post-processing helper: re-attaches cluster labels to driver output. |
| `reconstruct_node_fc.R` | Post-processing helper: adds node LFC columns to driver output. |
| `emit_expr_bygroup.R` | Transcript-substrate emitter: per-(cluster, Group) mean of `originalexp@data`; writes `outputs/reports/decomposition/levy_t5/transcript_per_cluster.parquet`. Run once per spine build, not per contrast. |
| `build_pair_inputs.sh` | Input-prep orchestrator: calls `build_pair_seurat.R`, `export_decomposition_for_pair.py`, and `build_input_gene_list.R`; writes to `data/derived/incytr_inputs/`. |
| `build_pair_seurat.R` | Builds `incytr_obj.rds` from the snRNA-seq data. Writes to `data/derived/incytr_inputs/incytr_obj.rds`. |
| `build_input_gene_list.R` | Builds `input_gene_list.csv` for the driver. Writes to `data/derived/incytr_inputs/`. |
| `export_decomposition_for_pair.py` | Reshapes `{protein,phospho,phospho_pY}_per_cluster.parquet` into yuyu CSV format for the R driver. Writes `{pr,ps,py}_yuyu_deconvoluted.csv` to `data/derived/incytr_inputs/`. |
| `pair_to_receiver_cache.py` | Reshapes the 9 wide driver outputs from `outputs/reports/incytr_pair_mode/wide/` into the long-form `receiver_cache/` layout the unified viewer consumes. Invoked by `alz/runners/main/run_pair_mode_viewer_build.sh`. |

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
