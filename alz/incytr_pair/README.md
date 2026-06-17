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
| Per-cluster proteomically-regulated genes (prG) | `proteomics_gene(strict=TRUE)` |
| Pair-invariant trimean substrate (`expr_bygroup`) | `precompute_expr_bygroup` |
| Trimean kernel for the transcript substrate | `Incytr:::grouped_weighted_quartile` (in `emit_expr_bygroup.R`) |
| Core-budget assertion, results export | `assert_core_budget`, `Export_results` |

Legitimately app-side (stays here): `slice_omics`, the `pmax(pr,1)` floor
(`floor_pr`), the per-contrast DEG arm (this contrast's two conditions' markers
from `allmarkers.csv`) and `dg_by_cluster = DEG ∪ prG` assembly, `label_node`, the DuckDB shard
concat, per-pair scheduling, RSS monitors, and the sce4 call-site overrides
(`mean_method=NULL`, `correction=0.01`, `cutoff_*=0`, `pr.correction=0.001`,
`fold_threshold=10`, strict `>1` prG cutoff). Three method computations were
reimplemented inline and moved to the package on 2026-05-29 (byte-identical swap;
audit: `docs/integrations/kinase_incytr_integration.md`).

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

## Production Defaults and Input Provenance

The canonical AD Incytr input bundle is:

```text
data/derived/incytr_inputs/
```

Production runs must take transcriptomics, markers, protein, phospho, and kinase inputs from that
single matched bundle. Do not mix files from `data/incytr_frozen/v2_46clusters/incytr input/`,
`data/incytr_frozen/sce4_source/deconvolution_with_new_clusters_20250721/`,
`data/derived/incytr_inputs_source_ps_diag/`, or `data/derived/_sce4_input_scratch/` into a
production run. Those roots are diagnostic/provenance material only unless the output directory and
documentation explicitly mark the run as forensic.

Default AD and T-cell output gating is floor-gated and uncapped:

```text
(SigProb_condition1 > 0.1 OR SigProb_condition2 > 0.1) AND abs(PDS) >= 0.2
```

Do not add a p-value/FDR/q-value gate to canonical deliverables. Do not apply the per
sender/receiver Top300 up/down cap to canonical Song or T-cell viewer outputs. Top300 is available
only as an explicit sce4 table-compatibility diagnostic (`filter_significant_paths.py --top300`)
because it is rank-sensitive to the PDS drift documented in the sce4 reproduction audit.

`filter_significant_paths.py --exclude-transgenes` is available only for the explicit AD/sce4
transgene-excluded sensitivity analysis. It removes paths touching `App`, `Psen1`, or `Mapt` in
Ligand/Receptor/EM/Target before any optional cap. It is not a hidden default and should be called
out in downstream results.

Before relying on a run or config, check path provenance:

```bash
pixi run python alz/incytr_pair/audit_incytr_input_provenance.py
```

## File inventory

Mouse AD cohort (9 contrasts) and T-cell cohort (per-donor) share `incytr_commandline.R`
and `filter_significant_paths.py`; the cohorts differ only in their input builders.

| File | Role |
|---|---|
| `run_pair_mode.sh` | Per-contrast loop driver (mouse AD); calls `incytr_commandline.R` then `filter_significant_paths.py` for each of 9 contrasts. Reads `data/derived/incytr_inputs/`; writes `outputs/reports/incytr_pair_mode/wide/`. |
| `run_pair_mode_tcells.sh` | Per-contrast loop driver (T-cell cohort); reuses the same `incytr_commandline.R`. pixi task `tcells-incytr`. |
| `incytr_commandline.R` | R driver: calls `Incytr::Cal_pairwise_grid`; writes one wide parquet per contrast. Node labels (DEG/prG) and per-node FC are written inline. Env-parameterized for mouse vs human (`SPECIES`, `CHANNELS`, file/gene-col vars). |
| `filter_significant_paths.py` | Row-subset significance filter applied after each contrast: `(SigProb_<disease> > 0.1 OR SigProb_<WTyp> > 0.1) AND abs(PDS) >= 0.2`, uncapped by default. Optional `--top300` applies the sce4 per-pair top-300 up/down cap for compatibility diagnostics only. Optional `--exclude-transgenes` removes `App`/`Psen1`/`Mapt` paths for AD sensitivity analyses only. |
| `emit_expr_bygroup.R` | Transcript-substrate emitter: per-(cluster, Group) trimean of `originalexp@data` via `Incytr:::grouped_weighted_quartile`; writes `outputs/reports/decomposition/levy_t5/transcript_per_cluster.parquet`. Run once per spine build, not per contrast. |
| `build_pair_inputs.sh` | Input-prep orchestrator (mouse): calls `build_pair_seurat.R`, `export_decomposition_for_pair.py`, `build_input_gene_list.R`; writes `data/derived/incytr_inputs/`. |
| `build_pair_seurat.R` | Builds `incytr_obj.rds` from the snRNA-seq data. |
| `build_input_gene_list.R` | Builds `allmarkers.csv` (one-vs-rest `FindAllMarkers`, `Type_condition` idents, run broad at `logfc.threshold=0.1` to match sce4's frozen table). The driver assembles the per-contrast gene.use (this contrast's two conditions' DEG ∪ prG) from this — no HEG (dropped 2026-05-31), no pre-collapsed `input_gene_list.csv`. |
| `build_tcells_seurat.R` | T-cell-cohort Seurat builder (per donor). pixi `tcells-build-incytr-seurat`. |
| `build_tcells_input_gene_list.R` | T-cell-cohort `allmarkers.csv` builder (per donor); same per-contrast-DEG contract as the mouse path. pixi `tcells-build-input-gene-list`. |
| `export_decomposition_for_pair.py` | Runs the **provenance deconvolution** `P_c = (N_total/N_c)×bulk×(specific_c/Σ_46 specific)` (min/10000 imputation): transcript share from frozen `aggexp.csv`, size factors from the Song h5ad, bulk from frozen `pr/imac/py_median.csv`. Emits 31-spine × 12 male-group `{pr,ps,py}_yuyu_deconvoluted.csv`. See `archive/sce4_reproduction_2026-06-08/README.md`. |
| `pair_to_receiver_cache.py` | Reshapes the 9 wide driver outputs into the long-form `receiver_cache/` layout the unified viewer consumes. Also exports `_sanitize_celltype`, imported by 4 `alz/integration/` modules. |
| `verify_sce4_parity.py` | Regression gate (`pixi run verify-incytr-sce4`): regenerates the two benchmark pairs unfiltered (nboot=0) and confirms the sce4-parity overrides still reproduce 599/600 + max \|Δ sclog2FC\|=0 on R/EM/T (App-transgene residual exempt). |
| `verify_sce4_full.R` | Full sce4 reproduction gate (`pixi run verify-incytr-sce4-full`): compares every gated path tuple in the 9 unfiltered wide parquets against sce4's pre-cap pairwise RDS files. Fails loudly until all 9 RDS files are present. |
| `audit_incytr_input_provenance.py` | Lightweight scanner that reports canonical, diagnostic, and suspicious Incytr input-root references in scripts/docs/configs. Use before treating a run as production. |
| `__init__.py` | Package marker (enables `from incytr_pair.* import` in `alz/integration/`). |

The sce4-reproduction forensic probes (`audit_*`, `forensic_sce4_afc.R`,
`run_sce4_full_unfiltered.sh`) and the redundant `launch_pair_mode.sh` launcher
were archived to `archive/sce4_reproduction_2026-06-08/` on 2026-06-08 — the
reproduction is solved and they are referenced only by the investigation log
`archive/sce4_reproduction_2026-06-08/README.md`. The active regression gates
(`verify_incytr_sce4.sh`, `verify_sce4_parity.py`, `verify_sce4_full.R`) and the
production gene.use source (`extract_sce4_geneuse.R`) stay here.

## Data layout

```
data/derived/incytr_inputs/          ← R driver inputs
   incytr_obj.rds
   pr_yuyu_deconvoluted.csv
   ps_yuyu_deconvoluted.csv
   py_yuyu_deconvoluted.csv
   kldata.csv                        (symlink → data/datasets/song/kinase/kldata_pspy.csv)
   allmarkers.csv                    (driver derives per-contrast DEG from this)

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
pixi run verify-incytr-sce4-full   # full gate: requires all 9 pre-cap sce4 RDS files
bash alz/runners/main/run_pair_mode_pipeline.sh   # picks up sentinels; runs E3/E4/I/V
```

For sce4 reproduction runs where permutation p-values are not needed, set
`FULL_NBOOT=0` when calling `run_pair_mode.sh`; the path-set gates use SigProb
and PDS, not permutation p-values. Set `FORCE_RERUN=1` if `wide/` already
contains stale parquets from a non-frozen-geneuse run.
