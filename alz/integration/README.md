# Incytr Integration

The **consumer** side of the pair-mode Incytr pipeline. All scoring math,
fold-change, and pathway construction live in the `Incytr` R package
(`~/Projects/work/incytr/`); the run-time drivers live in `alz/incytr_pair/`.
This directory holds the AD-side helpers that consume pair-mode outputs, build
the substrates the unified viewer's Evidence tab reads, and own the cluster-spine
config.

## Status (2026-05-29)

The active cluster spine is **Levy-t5** (31 clusters, `min_cells = 5`, no rank
gate). The factorial Incytr path was archived 2026-05-18
(`archive/incytr_factorial_2026-05-18/`); earlier spines (WMB-34, Levy-19) are
superseded and no longer reachable from code. Pair-mode is driven by
`alz/runners/main/run_pair_mode_pipeline.sh`.

## Role of this module vs `alz/incytr_pair/`

| Module | Responsibility |
|---|---|
| `alz/incytr_pair/` | Run-time drivers: build Seurat inputs, run `Incytr::Cal_pairwise_grid`, emit transcript substrate, orchestrate the per-contrast loop |
| `alz/integration/` | Consume outputs: reshape wide parquets into viewer substrates, manage config and cluster-spine loading, verify the round-trip |

## Separation of concerns

These are Python files; they cannot `library(Incytr)`. The boundary is: Incytr
scoring math is **produced** by the R driver, and the only place this directory
re-derives any of it is to **verify** the driver's stored output — never as a
second source of truth.

- **One verification pass.** `verify_pathway_round_trip.py` independently
  recomputes each node's `*_log2FC` from the substrates and asserts agreement
  with the stored value (`ROUNDTRIP_TOL = 1e-4`). It is a sampled spot-check by
  default (`SAMPLE_ROWS_PER_CONTRAST = 100`, reproducible per `--seed`; strict
  mode does the full grid). It emits **no** data. The build step
  (`build_normalized_substrate.py`) only *produces* the substrate — it does not
  carry its own copy of the recompute.
- **Parity constants are shared.** The `Cal_foldchange` / `Cal_scFC` corrections
  (`EPSILON_OMICS = 1e-3`, `EPSILON_SC = 0.01`) live in
  `alz/shared/incytr_constants.py`, mirroring the R driver — not redefined per
  file.
- **`normalize_quantiles`** (`build_normalized_substrate.py`) is a numpy port of
  `limma::normalizeQuantiles`, which Incytr applies via the CRAN `limma` package
  inside `integrate_omics_layer`. The R driver does not emit per-condition
  normalized means, so the viewer needs this substrate to reconstruct LFC
  client-side. It is the one legitimate parallel implementation here; its
  correctness is the point of the round-trip verification above.

## Files

| File | Role |
|---|---|
| `config_integration.py` | Filter values, design columns, contrast vectors, paths. `load_cluster_spine()` is consumed by `alz/reference/snrna_proportions.py`, `alz/decomposition_mea/verify_decomposition.py`, and the viewer. |
| `build_cluster_spine.py` | Run-once generator: builds the Levy-t5 31-cluster spine CSV from the Levy lab cluster key + barcode-to-cluster table. Step A of `run_pair_mode_pipeline.sh`. |
| `build_normalized_substrate.py` | Produces per-(cluster, contrast, layer) limma-normalized condition means under `audit_sources/omics_trace_normalized/`. Consumed by the viewer + `verify_pathway_round_trip.py`. |
| `build_omics_trace.py` | Per-cluster raw-value omics-trace shards (protein + phospho pS/pT + pY) under `audit_sources/omics_trace/`. |
| `build_transcript_trace.py` | Per-cluster transcript pseudobulk shards under `audit_sources/transcript_trace/` from `outputs/reports/decomposition/levy_t5/transcript_per_cluster.parquet`. |
| `verify_pathway_round_trip.py` | Round-trip LFC verification harness (sampled default, `--strict` full grid, `--seed` to rotate the sample). Run inside the viewer build. |
| `build_yuyu_kldata.py` | One-time generator: builds `kldata_pspy.csv` (symlinked into `data/derived/incytr_inputs/kldata.csv`) via the `kinase_library` package. |
| `build_seaad_bridge.py` | One-time generator: hand-curated `cluster_to_seaad_supertype.csv` (Levy-t5 cluster → SEA-AD supertype). Direct crosswalks only. |
| `extract_cluster_assignments.R` | One-time generator: emits `barcode_to_cluster.csv` and `cell_metadata.csv` from the legacy `incytr_obj.rds`. |

`build_normalized_substrate.py`, `build_omics_trace.py`,
`build_transcript_trace.py`, and `verify_pathway_round_trip.py` are invoked
lazily from `alz/build_unified_viewer.py` (pixi `viewer`); the three generators
above are run by hand on data refresh and their outputs are frozen prerequisites.
