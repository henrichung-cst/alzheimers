# Incytr Integration

A thin AD-specific wrapper around the upstream `incytr` R package
(`~/Projects/work/incytr/`). All math, scoring, and pathway construction
lives in the package; this directory holds the AD-side helpers that consume
pair-mode outputs and manage the cluster spine and config.

## Status (2026-05-20)

The **factorial** Incytr path was archived 2026-05-18. Pair-mode is the active
path, driven by `alz/runners/main/run_pair_mode_pipeline.sh`. The archived
factorial plumbing lives at `archive/incytr_factorial_2026-05-18/` — recover
from there if needed.

The active cluster spine is **Levy-t5** (31 clusters, `min_cells = 5`, no
rank gate). Earlier spines (WMB-34, Levy-19) are superseded and no longer
reachable from code.

## Role of this module vs `alz/incytr/`

| Module | Responsibility |
|---|---|
| `alz/incytr/` | Run-time drivers: build Seurat inputs, run `Incytr::Cal_pairwise_grid`, emit transcript substrate, orchestrate the per-contrast loop |
| `alz/integration/` | Consume outputs: reshape wide parquets into `receiver_cache/` for the viewer; build transcript-trace shards; manage config and cluster-spine loading |

## Files

| File | Role |
|---|---|
| `config_integration.py` | Filter values, design columns, contrast vectors, paths. `load_cluster_spine()` is consumed by `alz/snrna_proportions.py`, `alz/decomposition/verify_decomposition.py`, and the viewer. |
| `build_cluster_spine.py` | Run-once generator: builds the Levy-t5 31-cluster spine CSV from the Levy lab cluster key + barcode-to-cluster table. Outputs to `data/incytr_frozen/v2_46clusters/spines/<name>/cluster_spine.csv`. |
| `extract_cluster_assignments.R` | Run-once generator: emits `barcode_to_cluster.csv` and `cell_metadata.csv` from the legacy `incytr_obj.rds`. |
| `plot_cluster_spine.py` | Diagnostic plots over `cluster_spine.csv`. |
| `build_seaad_bridge.py` | One-shot: generates `cluster_to_seaad_supertype.csv` (hand-curated cluster → SEA-AD supertype map). |
| `build_yuyu_kldata.py` | Builds the `kldata_pspy.csv` symlinked into `data/derived/incytr_inputs/kldata.csv` for the R driver. |
| `pair_to_receiver_cache.py` | Reshapes pair-mode output from `outputs/reports/incytr_pair_mode/wide/` (9 wide parquets) into the long-form `receiver_cache/` layout the unified viewer consumes. Invoked by `alz/runners/main/run_pair_mode_viewer_build.sh`. |
| `build_transcript_trace.py` | Builds per-cluster transcript-trace shards under `audit_sources/transcript_trace/` from the canonical substrate at `outputs/reports/decomposition/levy_t5/transcript_per_cluster.parquet`. |
