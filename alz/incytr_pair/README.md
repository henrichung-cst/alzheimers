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

## Separation of concerns: method (`../incytr`) vs application (this dir)

**`Incytr` is the method (a public package we release); `alz` is one application
of it.** Anything intrinsic to the method — scoring, fold-change, gene-set
selection, expression aggregation — must be **called** from the `Incytr` package,
never copied or reimplemented here. This directory holds only AD-specific glue:
loading AD inputs, file paths, the contrast loop, and call-site parameter overrides
for sce4 parity. Do not re-inline method logic to avoid a package change — that is
the exact "secret duplication" this boundary exists to prevent.

Method functions this app calls (all from `Incytr::`):

| What | Package function |
|---|---|
| Pairwise grid scoring (SigProb, PDS, scFC, kinase) | `Cal_pairwise_grid` |
| High-expression genes (HEG), global cutoff scope | `Find_highexp_gene_batch` |
| Per-cluster proteomically-regulated genes (prG) | `proteomics_gene(strict=TRUE)` |
| Pair-invariant trimean substrate (`expr_bygroup`) | `precompute_expr_bygroup` |
| Trimean kernel for the transcript substrate | `Incytr:::grouped_weighted_quartile` (in `emit_expr_bygroup.R`) |
| Core-budget assertion, results export | `assert_core_budget`, `Export_results` |

Legitimately app-side (stays here): `slice_omics`, the `pmax(pr,1)` floor
(`floor_pr`), `dg_by_cluster = DEG ∪ prG` assembly, `label_node`, the DuckDB shard
concat, per-pair scheduling, RSS monitors, and the sce4 call-site overrides
(`mean_method=NULL`, `correction=0.01`, `cutoff_*=0`, `pr.correction=0.001`,
`fold_threshold=10`, strict `>1` prG cutoff). Three method computations were
reimplemented inline and moved to the package on 2026-05-29 (byte-identical swap;
audit: `docs/plans/incytr_pair_reorg_2026-05-29.md`).

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

Mouse AD cohort (9 contrasts) and T-cell cohort (per-donor) share `incytr_commandline.R`
and `filter_significant_paths.py`; the cohorts differ only in their input builders.

| File | Role |
|---|---|
| `run_pair_mode.sh` | Per-contrast loop driver (mouse AD); calls `incytr_commandline.R` then `filter_significant_paths.py` for each of 9 contrasts. Reads `data/derived/incytr_inputs/`; writes `outputs/reports/incytr_pair_mode/wide/`. |
| `run_pair_mode_tcells.sh` | Per-contrast loop driver (T-cell cohort); reuses the same `incytr_commandline.R`. pixi task `tcells-incytr`. |
| `incytr_commandline.R` | R driver: calls `Incytr::Cal_pairwise_grid`; writes one wide parquet per contrast. Node labels (DEG/prG) and per-node FC are written inline. Env-parameterized for mouse vs human (`SPECIES`, `CHANNELS`, file/gene-col vars). |
| `filter_significant_paths.py` | Row-subset significance filter applied after each contrast: `(SigProb_<disease> > 0.1 OR SigProb_<WTyp> > 0.1) AND abs(PDS) >= 0.2`. Parity-preserving. |
| `emit_expr_bygroup.R` | Transcript-substrate emitter: per-(cluster, Group) trimean of `originalexp@data` via `Incytr:::grouped_weighted_quartile`; writes `outputs/reports/decomposition/levy_t5/transcript_per_cluster.parquet`. Run once per spine build, not per contrast. |
| `build_pair_inputs.sh` | Input-prep orchestrator (mouse): calls `build_pair_seurat.R`, `export_decomposition_for_pair.py`, `build_input_gene_list.R`; writes `data/derived/incytr_inputs/`. |
| `build_pair_seurat.R` | Builds `incytr_obj.rds` from the snRNA-seq data. |
| `build_input_gene_list.R` | Builds `input_gene_list.csv` (DEG ∪ HEG). HEG via `Incytr::Find_highexp_gene_batch(cutoff_scope="global")`. |
| `build_tcells_seurat.R` | T-cell-cohort Seurat builder (per donor). pixi `tcells-build-incytr-seurat`. |
| `build_tcells_input_gene_list.R` | T-cell-cohort gene-list builder (per donor). pixi `tcells-build-input-gene-list`. |
| `export_decomposition_for_pair.py` | Runs the **provenance deconvolution** `P_c = (N_total/N_c)×bulk×(specific_c/Σ_46 specific)` (min/10000 imputation): transcript share from frozen `aggexp.csv`, size factors from the Song h5ad, bulk from frozen `pr/imac/py_median.csv`. Emits 31-spine × 12 male-group `{pr,ps,py}_yuyu_deconvoluted.csv`. See `docs/plans/sce4_decomposition_reconciliation_2026-05-24.md`. |
| `pair_to_receiver_cache.py` | Reshapes the 9 wide driver outputs into the long-form `receiver_cache/` layout the unified viewer consumes. Also exports `_sanitize_celltype`, imported by 4 `alz/integration/` modules. |
| `verify_sce4_parity.py` | Regression gate (`pixi run verify-incytr-sce4`): confirms the sce4-parity overrides still reproduce 573/600 + max \|Δ sclog2FC\|=0 on the two benchmark pairs by reading the existing wide parquets. |
| `__init__.py` | Package marker (enables `from incytr_pair.* import` in `alz/integration/`). |

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
