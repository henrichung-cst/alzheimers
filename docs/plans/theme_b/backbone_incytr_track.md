# Backbone / Incytr-viewer track — authoritative

Single source of truth for the Incytr backbone work in Wave 1 / Theme B. Consolidates what was
spread across `b5_plan`, `b5_audit`, `b2_plan`, `backbone_fold_into_build_*`, and the prior
`incytr_viewer_refactor` draft. Read this before touching any backbone/Incytr-viewer code.

Companion docs that stay live:
- [`incytr_viewer_schema.md`](incytr_viewer_schema.md) — baseline audit of the two Incytr tabs (reference).
- [`b4_plan.md`](b4_plan.md) — kinase→pathway bridge (`#Backbones`/`#Paths`), implemented.
- [`b2_plan.md`](b2_plan.md) — backbone sankey, **deferred** (see Open threads).

## Position in the 4-wave plan

The top-level program is the parallel-orchestration experiment in [`../meta_plan.md`](../meta_plan.md):
Themes A–H delivered as gated Waves 1–4. This track is the Theme-B backbone slice of Wave 1.

The original Wave-1 design (B5) was a **standalone recurrence reduction** (`backbone_reduction.py`)
that pre-ranked R-EM-T spines into `backbone_rem_t.parquet` for a B2 sankey. **That approach is
superseded** (see Superseded). The backbone is now a first-class *grain* of the pathway itself,
emitted by the scoring engine and navigated live in the merged viewer.

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

## Implemented & verified (workflow run `wf_82154a8b-932`, 2026-06-29)

All settled fact; the sce4 parity gate (`pixi run verify-incytr-sce4`) passed throughout.

- **`score_spine` / `score_spine_from_expr`** in `~/Projects/work/incytr/R/evaluation.R` (+ NAMESPACE
  export, `tests/testthat/test-score_spine.R`, 28 tests). Full-grain re-score is **byte-identical**
  to `Cal_PDS` (`max_abs_diff = 0.0` on the 1,283-path benchmark pair).
- **Engine-integrated emission** in `Cal_pairwise_grid` (`incytr/R/grid.R`): each pair-mode run
  writes `outputs/reports/incytr_pair_mode/backbone/<grain>/<contrast>_backbone_output.parquet`
  (+ `.shards/`), parallel to the pathway `wide/`. Path-scoring path stays byte-identical →
  **sce4 parity preserved** (gated path-set == sce4 Allpathway, transgene-exempt; R/T sclog2FC
  max|Δ|=0). Grains: R-EM, L-R-EM, R-EM-T.
- **Viewer payload** (`alz/viewer/shared/incytr_index.py`, `alz/viewer/cohorts/song.py`): per-grain
  heatmap count tensors, payload sizing (R-EM/L-R-EM inline; R-EM-T via global index + `SliceCache`
  shards), and Ack/KGG/Rme1 score-column **gating** (surfaced only where the cohort emits them —
  absent for this AD cohort and t-cells, no empty columns).
- **Merged viewer screen** (`alz/viewer_shared/template/js/tabs/incytr_{pathways,heatmap}.js` +
  `incytr_global_index.js`, `incytr_state.js`, both `body.html`, `02_ui_chrome.js`): one Incytr tab,
  persistent left filter panel, Table/Heatmap switch in the main pane, unified `IncytrFilter`
  namespace, unified PDS-sign vocab (both/up/down). Heatmap-cell click switches the pane in place.
- **Grain selector** (Full / L-R-EM / R-EM-T / R-EM; hidden when no `backbone_grains`) + **within-
  disease timepoint-combination filter** (multiselect {2mo,4mo,6mo} + all/any). The timepoint
  predicate is evaluated **within a single disease** — a backbone at 2mo/4mo in AppP and only 6mo
  in Ttau does NOT qualify for "all 3". This filter *is* the recurrence predicate that B5's
  `backbone_rank` used to precompute.
- **Tissue gating fix** — 5xFAD-only (`ctxIsFivexfad` guard), no longer leaks under Song.
- **Cleanup** — the superseded `backbone_reduction.py` and its `reduce()` call in the bridge were
  removed; `#Backbones`/`#Paths` (from `kinase_participation.csv`) are unaffected.

### B-6 + J-5 grain-drill / expansion navigation (2026-06-29)

Lets the user move between grains anchored on a row, with a transparent filter mechanism and a
one-level undo.

- **B-6 `backbone_spine_index`** — per-grain on-demand artifact (spine-key → present
  (sender,receiver) pairs), the sibling of `gene_node_index_shard`. Built in `song.py` via streamed
  DuckDB `SELECT DISTINCT` (never a whole-frame read); R-EM 346 / L-R-EM 399 / R-EM-T 97,012 spines.
  Loaded by `loadBackboneSpineIndex(grain)` (`04_slice_cache.js`, LRU + `DecompressionStream`) only
  on widen, self-gated per grain.
- **Drill = transparent search-bar filter.** `_ipNavigateDrill(targetGrain, searchText)` switches
  the grain and injects **space-separated** gene tokens into the visible `ip-search` box (expand-to-
  Full appends sender+receiver so the within-pair scope is visible, not a hidden mode flip; collapse
  injects the coarser grain's spine genes). The criteria are visible and user-editable — no
  `|`-joined single-literal token, no silent `ipMode:"pair"` switch.
- **Single Back button** (`↩ Back to <grain>`, in the toolbar next to the grain selector). Snapshots
  the full filter state + page into `_ipRuntime.drillReturn` before each drill; one-level restore
  (re-drilling replaces the snapshot); cleared on Reset. Mirrored into both `viewer/` and
  `tcell_viewer/` `body.html` (the grain selector lives in both). This replaced the in-drawer
  breadcrumb so the toolbar button is the sole navigation affordance.
- **"Expands to" drawer sub-tab** (alongside Evidence/Scores) carries the expand / collapse / widen
  initiators; widen groups present pairs by receiver cell type (count-first).
- **Tests** — `alz/viewer/verify_backbone_spine_index.py`, **35 passing**: key/grain-partial-order
  logic, per-grain payload presence, and a 7-case synthetic multi-pair / 3-receiver widen-grouping
  test (the cross-cell-type path the 1-pair fixture can't exercise). sce4 parity preserved
  (`verify-incytr-sce4` PASS); both viewers build clean.
- **Residual** — the **visual** widen gate awaits the overnight full load (1-pair fixture is
  degenerate for cross-cell-type grouping). The Back button + transparent drill *are* exercisable on
  the fixture now (part of the P4 click-through below).

## Current runtime state

The viewer is built on a **1-pair fixture** (Microglia → Cholinergic-Neurons, 9 contrasts) so the
full 961-pair run drops in with zero code change (nothing hardcodes pair name/count/size/fixture
path — adversarially audited). Built artifacts: `outputs/reports/unified_viewer/index.html`,
`.../tcell_viewer/index.html`.

The full data comes from the overnight runner **`alz/incytr_pair/run_backbone_overnight.sh`**
(`tmux new -s incytr` → `bash …`): pair-mode grid → viewer build → kinase-incytr bridge → viewer
build, memory-capped via `systemd-run`, logged, resumable.

**Human gates outstanding:** (P4) click-through of the merged screen + grain/timepoint/tissue +
the grain-drill Back button and transparent search-bar drill on the fixture now; (P5) re-confirm
after the overnight full load, unified + t-cell, including the widen → all-cell-types grouping.

## Open threads / next

### B2 sankey — deferred

[`b2_plan.md`](b2_plan.md) was designed on the deleted `backbone_rem_t.parquet` + `backbone_rank`
schema, so its current spec is stale. The merged table+heatmap + grain selector + within-disease
timepoint filter + the B-6/J-5 drill already deliver backbone navigation and recurrence, so a
separate recurrence-ranked sankey may be redundant. **Decision parked** (drop vs. rebuild on the
engine grains) until the drill ships and the full data loads.

### `_contracts.md §B5`

Rewritten to current truth (engine grain + live recurrence filter). No agent should consume
`backbone_rem_t.parquet` or `backbone_rank` — both are gone.

## Superseded — do not reintroduce

- **Standalone recurrence reduction** (`backbone_reduction.py`, `reduce()` in the bridge,
  `backbone_rem_t.parquet`): removed. Recurrence is a live within-disease timepoint filter, not a
  precomputed ranked table.
- **`backbone_rank` / `n_timepoints_present` / `n_conditions_present` / `is_cholinergic_target`
  schema**: removed. There is no representative/argmax PDS — every backbone carries a real per-
  contrast PDS from `score_spine`.
- **Recurrence "mode" as a separate viewer surface**: removed — it is a grain + a filter.
