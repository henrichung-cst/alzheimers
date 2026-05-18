# Incytr Integration

A thin AD-specific wrapper around the upstream `incytr` R package
(`~/Projects/work/incytr/`). All math, scoring, and pathway construction
lives in the package; this directory holds the AD-side helpers that build
inputs and consume outputs.

## Status (2026-05-18)

The **factorial** Incytr path was archived. Pair-mode is the active path,
driven by `alz/runners/main/run_pair_mode_pipeline.sh` (the runner formerly
known as `run_change_requests.sh`) and the bench scripts under
`bench/incytr_pair_19/` / `bench/run_pair_mode_19.sh`. The archived factorial
plumbing lives at `archive/incytr_factorial_2026-05-18/` — recover from
there if needed.

## Files

| File | Role |
|---|---|
| `config_integration.py` | Filter values, design columns, contrast vectors, paths. `load_cluster_spine()` is consumed by `alz/snrna_proportions.py`, `alz/decomposition/verify_decomposition.py`, and the viewer. |
| `build_cluster_spine.py` | Run-once generator: builds the 19-cluster spine CSV from the legacy Levy lab cluster key + barcode-to-cluster table. Outputs to `data/incytr_frozen/v2_46clusters/spines/<name>/cluster_spine.csv`. |
| `extract_cluster_assignments.R` | Run-once generator: emits `barcode_to_cluster.csv` and `cell_metadata.csv` from the legacy `incytr_obj.rds`. |
| `plot_cluster_spine.py` | Diagnostic plots over `cluster_spine.csv`. |
| `build_seaad_bridge.py` | One-shot: generates `cluster_to_seaad_supertype.csv` (hand-curated 19-cluster → SEA-AD supertype map). |
| `build_yuyu_kldata.py` | Builds the `kldata_pspy.csv` consumed by pair-mode (`bench/incytr_pair_*/incytr input/kldata.csv`). |
| `pair_to_receiver_cache.py` | Reshapes pair-mode output (9 wide parquets) into the long-form `receiver_cache/` layout the unified viewer consumes. Invoked by `alz/runners/main/run_pair_mode_viewer_build.sh`. |
