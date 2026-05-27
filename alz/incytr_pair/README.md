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
| `export_decomposition_for_pair.py` | Runs the **provenance deconvolution** `P_c = (N_total/N_c)×bulk×(specific_c/Σ_46 specific)` (min/10000 imputation): transcript share from frozen `aggexp.csv`, size factors from the Song h5ad, bulk from frozen `pr/imac/py_median.csv`. Emits 31-spine × 12 male-group `{pr,ps,py}_yuyu_deconvoluted.csv`. Replaces the prior levy_t5 parquet-reshape (see `docs/plans/sce4_decomposition_reconciliation_2026-05-24.md`). |
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

## Running on this box (memory-bounded, cgroup-wrapped)

The host has **30 GiB total RAM**. A pair-mode contrast's peak working set
(main R + Permutation_test allocations) can reach ~24 GB on heavy contrasts.
A bare `bash alz/runners/main/run_pair_mode_pipeline.sh` will OOM-kill Claude
or the desktop session if it runs away. Always launch inside a systemd user
scope so the cgroup contains a runaway to itself.

### Canonical invocation

```bash
rm -f outputs/reports/change_requests/.state/E2.done   # if re-running E2
systemd-run --user --scope --slice=alz-incytr.slice \
  -p MemoryMax=24G -p MemorySwapMax=0 \
  --unit=alz-incytr-$(date +%s) \
  --description="Pair-mode regen" \
  bash -c 'NPERM_WORKERS=1 NPAIR_WORKERS=1 CHUNK_PARALLEL=1 N_CHUNK_MULT=48 \
           bash alz/runners/main/run_pair_mode_pipeline.sh --rerun E2 --workers 1' \
  > /tmp/pair_mode_capped.log 2>&1 &
```

### Why these flags (do not change without a reason and a test)

- `MemoryMax=24G` — hard wall. Above this the cgroup OOM-kills *inside the
  scope only*; the host (and Claude) stay alive.
- **No `MemoryHigh`.** A soft throttle of 20G caused 30 % of wall time to be
  spent in kernel reclaim — pairs ran at 38 min/pair vs the 1–2 min/pair
  bench precedent. The kernel hit `memory.high` 3.4 M times in 2.5 h.
- `MemorySwapMax=0` — shared box, no swap thrash.
- `NPERM_WORKERS=1` — single-core permutation. Each perm fork inherits the
  main R's ~12 GB via COW and writes private pages during scoring; 4 forks
  push the cgroup past 24 G on heavy pairs. Going to 2 *might* work; do not
  go to 4.
- `N_CHUNK_MULT=48` (~20 pairs/chunk) — smaller than the upstream default
  `N_CHUNK_MULT=8` (~121 pairs/chunk). The default was sized for a 30+ GB
  host without a cgroup cap; under 24 G it OOMs around pair 14 of 20 because
  `chunk_fn` writes one sub-parquet per pair (rather than accumulating the
  full chunk in memory), but R's own heap-fragmentation residue still grows.
- `CHUNK_PARALLEL=1`, `NPAIR_WORKERS=1` — one chunk subprocess at a time, one
  pair at a time inside the chunk. Two simultaneous chunks at ~13–15 GB each
  blow the cap.

### Monitoring

- Per-pair progress: `tail -f /tmp/pair_mode_capped.log | grep pair-driver`
- Live cgroup state:
  ```bash
  scope=$(systemctl --user list-units --type=scope --no-legend --state=active \
          'alz-incytr-*.scope' | awk '{print $1}' | head -1)
  cgdir=/sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/alz.slice/alz-incytr.slice/$scope
  cat $cgdir/memory.events    # high/oom counters; should all stay 0
  cat $cgdir/memory.current   # current RSS in bytes
  ```
- Sub-shards landing on disk:
  `find outputs/reports/incytr_pair_mode/wide/.shards -type f | wc -l`

### Killing a runaway

```bash
scope=$(systemctl --user list-units --type=scope --no-legend --state=active \
        'alz-incytr-*.scope' | awk '{print $1}' | head -1)
systemctl --user stop "$scope"
```

This terminates *only* the cgroup. Claude and any other user processes are
unaffected.

### Wall-time expectation

Bench-precedent per-pair time on this box (NPERM_WORKERS=1): ~60–90 s for
typical pairs, up to ~4–5 min on the densest sender×receiver combinations
(large excitatory clusters with ~1.3 M output rows). Full 9-contrast run is
8 649 pairs × ~75 s ≈ **~180 h wall (~7.5 days)**, plus chunk-startup
overhead. Plan accordingly.

### After the run

```bash
pixi run verify-incytr-sce4   # regression gate: confirms sce4 parity on two known pairs
bash alz/runners/main/run_pair_mode_pipeline.sh   # picks up sentinels; runs E3/E4/I/V
```
