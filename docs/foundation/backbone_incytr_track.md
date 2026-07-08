# Backbone / Incytr-viewer track — authoritative

Single source of truth for the Incytr backbone grain. Read this before touching any
backbone/Incytr-viewer code.

## Core principle — a backbone is a pathway with fewer nodes

The grain selector only changes *which of the 4 nodes (L, R, EM, T) define the entity*:

| grain | nodes | = |
|---|---|---|
| Full | L-R-EM-T | the pathway itself (default) |
| R-EM-T | R, EM, T | drop Ligand |
| L-R-EM | L, R, EM | drop Target |
| R-EM | R, EM | drop Ligand + Target |

Everything else is the same machinery: per-(entity × contrast) rows (up to 9 contrasts = 3 diseases
× 3 timepoints), same columns, same DEG/prG badges, same Top/CellType toggle, same per-score gates,
same Evidence + Scores drawer (scoped to surviving nodes), same heatmap (counts entities). Sliding
the grain coarser never changes what a row *means* — only how many nodes define it. There is **no
separate recurrence mode, no representative/argmax PDS, no aggregation over timepoints, ever.**

## Backbone scoring — engine re-score on the sub-chain

A backbone gets a real PDS computed on its own nodes by the **same scorer** (`score_spine`), not
derived from the full paths sharing its spine. Every PDS term restricts cleanly because each
factorizes over nodes/edges (`evaluation.R`, `analysis.R`, `kinases.R`):

| term | full path | backbone rule |
|---|---|---|
| SigProb → TPDS | `hill(L·R)·hill(R·EM)·hill(EM·TG)` | product of **surviving contiguous edges** (R-EM-T → `hill(R·EM)·hill(EM·TG)`; L-R-EM → `hill(L·R)·hill(R·EM)`; R-EM → `hill(R·EM)`); `TPDS=logi(SigProb)` |
| PPDS / PhPDS_ps / PhPDS_py / Ack / KGG / Rme1 | `rowMeans(logi[L,R,EM,T]_layer)` | `rowMeans` over **surviving node FC columns** |
| SiK (KPDS) | 6 kinase→substrate cases among R/EM/T | keep cases whose **both** positions survive (Ligand never participates) |
| PDS | `Cal_PDS(multimodel + KPDS·SiK)` | same assembly on the restricted terms |

The grains are exactly the contiguous sub-chains, so SigProb is a clean edge product — no
variable-length-pathway refactor. Scoring uses the production `style="aFC"` fold-changes held in
engine memory, which is why emission is engine-integrated (the `wide/` parquet strips `_aFC`
columns; re-scoring from `wide/` drifts PDS past the `|PDS|≥0.2` floor). ε=0.01 and the canonical
floor (`SigProb>0.1 either AND |PDS|≥0.2`) are preserved.

## Wiring

- **Scorer** — `score_spine` / `score_spine_from_expr` in `~/Projects/work/incytr/R/evaluation.R`
  (NAMESPACE export + `tests/testthat/test-score_spine.R`). Full-grain re-score is byte-identical to
  `Cal_PDS`.
- **Emission** — engine-integrated in `Cal_pairwise_grid` (`incytr/R/grid.R`). Each pair-mode run
  writes `<BACKBONE_OUT_DIR>/<grain>/<contrast>_backbone_output.parquet` (+ `.shards/`), parallel to
  the pathway `wide/`. Grains: R-EM, L-R-EM, R-EM-T. Path-scoring stays byte-identical → sce4 parity
  preserved (see `docs/reference/incytr_sce4_reproduction.md`).
- **Per-cohort output dir** — `BACKBONE_OUT_DIR` is set per cohort by the runners
  (`alz/incytr_pair/run_pair_mode_5xfad.sh`, `run_pair_mode_tcells.sh`) and read in
  `incytr_commandline.R`, so each cohort's backbone lands alongside its own `wide/` rather than the
  shared Song path.
- **Viewer payload** — `write_incytr_backbone_grains` (`alz/viewer/shared/incytr_index.py`) emits
  per-grain heatmap count tensors and shards for **all three cohorts**: Song (`cohorts/song.py`),
  5xFAD (`cohorts/fivexfad.py`), and t-cell (`alz/tcell_viewer/slices_incytr.py`). R-EM/L-R-EM are
  inlined; R-EM-T is served via a global index + `SliceCache` shards. Ack/KGG/Rme1 score columns are
  gated — surfaced only where the cohort emits them (no empty columns).
- **Merged viewer screen** — one Incytr tab with a persistent left filter panel and a Table/Heatmap
  switch in the main pane (`alz/viewer_shared/template/js/tabs/incytr_{pathways,heatmap}.js`,
  `incytr_global_index.js`, `incytr_state.js`). Unified `IncytrFilter` namespace, unified PDS-sign
  vocab (both/up/down). Heatmap-cell click switches the pane in place.
- **Grain selector** (Full / L-R-EM / R-EM-T / R-EM; hidden when a cohort has no `backbone_grains`)
  **+ within-disease timepoint-combination filter** (multiselect timepoints + all/any). The timepoint
  predicate is evaluated **within a single disease** — a backbone at 2mo/4mo in one disease and only
  6mo in another does NOT qualify for "all 3". This filter *is* the recurrence predicate.
- **Grain-drill navigation** — `_ipNavigateDrill(targetGrain, searchText)` switches the grain and
  injects space-separated gene tokens into the visible `ip-search` box (criteria stay visible and
  user-editable — no hidden mode flip). A single `↩ Back to <grain>` toolbar button snapshots the
  full filter state for one-level restore. The per-grain `backbone_spine_index` (spine-key → present
  (sender,receiver) pairs) is built via streamed DuckDB `SELECT DISTINCT` and loaded on-demand by
  `loadBackboneSpineIndex(grain)` (`04_slice_cache.js`, LRU + `DecompressionStream`) only on widen.

## Full data

Full runs come from the memory-capped overnight runners under `alz/incytr_pair/regeneration/`
(`run_backbone_overnight_5xfad.sh`, `run_backbone_overnight_tcells.sh`, `run_backbone_overnight_all.sh`):
pair-mode grid → viewer build → kinase-incytr bridge → viewer build, capped via `systemd-run`,
logged, resumable. Per the multi-hour-job rule these are run by the operator in `tmux`, one cohort at
a time, not launched in-session.

## Superseded — do not reintroduce

- **Standalone recurrence reduction** (`backbone_reduction.py`, `backbone_rem_t.parquet`,
  `backbone_rank` / `n_timepoints_present` / `n_conditions_present` / `is_cholinergic_target`):
  recurrence is a live within-disease timepoint filter, not a precomputed ranked table.
- **Recurrence "mode" as a separate viewer surface**: it is a grain + a filter, not its own screen.
- There is no representative/argmax PDS — every backbone carries a real per-contrast PDS from
  `score_spine`.
