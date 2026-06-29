# Incytr viewer refactor — unified filter panel + backbone grain

Refactors the two shared Incytr tabs (`alz/viewer_shared/template/js/tabs/incytr_{pathways,heatmap}.js`)
into **one merged Incytr screen**, and adds a **node-grain selector** that treats a backbone as a
pathway with a shorter node spine — scored by the engine on its own nodes, identically to a full path.
Propagates to the unified *and* t-cell viewers.

Supersedes the prior draft's "backbone recurrence mode": there is **no separate mode, no representative
PDS, no `n_timepoints_present`/`backbone_rank` schema, no DuckDB reduction.** A backbone is a pathway.

## Core principle (settled by grilling 2026-06-29)

**A backbone is a pathway with fewer nodes.** The grain selector only changes *which of the 4 nodes
(L, R, EM, T) define the entity*:

| grain | nodes | = |
|---|---|---|
| Full | L-R-EM-T | the pathway itself (today's behavior, default) |
| R-EM-T | R, EM, T | drop Ligand |
| L-R-EM | L, R, EM | drop Target |
| R-EM | R, EM | drop Ligand + Target |

Everything else is the **same machinery**: per-(entity × contrast) rows (up to 9 contrasts =
3 diseases × 3 timepoints), same columns, same DEG/prG badges, same Top/CellType toggle, same per-score
gates, same Evidence + Scores drawer (scoped to the surviving nodes), same heatmap (counts entities).
Sliding the grain coarser never changes what a row *means* — only how many nodes define it.

## Backbone scoring — engine re-score on the sub-chain (settled)

A backbone gets a real PDS computed on its own nodes by the **same scorer**, not derived/aggregated from
the full paths sharing its spine. Every PDS term restricts cleanly because each factorizes over
nodes/edges (`evaluation.R:51-95`, `analysis.R:296`, `kinases.R:71`):

| term | full path | backbone rule |
|---|---|---|
| SigProb → TPDS | `hill(L·R)·hill(R·EM)·hill(EM·TG)` | product of **surviving contiguous edges** (R-EM-T → `hill(R·EM)·hill(EM·TG)`; L-R-EM → `hill(L·R)·hill(R·EM)`; R-EM → `hill(R·EM)`); `TPDS=logi(SigProb)` |
| PPDS / PhPDS_ps / PhPDS_py | `rowMeans(logi[L,R,EM,T]_layer)` | `rowMeans` over **surviving node FC columns** |
| SiK (KPDS) | 6 kinase→substrate cases among R/EM/T | keep cases whose **both** positions survive (Ligand never participates → R-EM-T's SiK = full path's; L-R-EM/R-EM keep only the two R↔EM cases) |
| PDS | `Cal_PDS(multimodel + KPDS·SiK)` | same assembly on the restricted terms |

The grains are exactly the **contiguous** sub-chains, so SigProb is a clean edge product — no
variable-length-pathway refactor, no hardcoded-4-node blocker. All inputs are already in scope: the
pair-invariant trimean substrate (`precompute_expr_bygroup`), the kinase library, and the per-node FC
columns in `wide/`.

## Build-side work

### B-1. Incytr package: score an arbitrary contiguous sub-spine
Add a sub-spine scoring path in `~/Projects/work/incytr` that, given a node subset ∈
{R-EM, L-R-EM, R-EM-T}, computes SigProb (surviving edge product), the omics node-means, and SiK
(surviving-position cases), then `Cal_PDS`. Reuses `hill`, `logi`, `score.weight`, `KPDS.weight`,
the ε=0.01 parity constants, and the canonical floor. Faithful restriction of the existing scorer —
not a new formula. Verify on Full = byte-identical to today's `PDS`.

### B-2. App-side driver: emit backbone `wide/` outputs per grain
A driver (parallel to the existing pair-mode flow) that, per (sender, receiver, contrast, grain), groups
the enumerated paths onto the grain's distinct spines and calls B-1, writing backbone `wide/` parquet
parallel to the existing pathway `wide/`. Same schema (per-node FC for surviving nodes, `SigProb_<cond>`,
`SiK_score_<cond>`, scores), same floor (`SigProb>0.1 either AND |PDS|≥0.2`). Distinct-spine counts
(gated): R-EM ~19.7k, L-R-EM ~27.9k, R-EM-T ~2.78M. **Memory-safe**: scoring is vectorized arithmetic
over grouped spines (no enumeration explosion); run DuckDB-streamed under a `systemd-run` memory cap;
never `pd.read_parquet` the R-EM-T output whole.

### B-3. Heatmap count tensors per grain
The heatmap counts entities for the active contrast. Compute the existing dense count tensor
(senders × receivers × contrasts × pvalThr × |PDS|Thr, + signed) **once per grain** (4 grains incl. Full)
over that grain's entity set. Bounded, inline-able.

### B-4. Payload + sizing
R-EM / L-R-EM ship whole (~28k rows, low-MB). **R-EM-T (2.78M) loads like Full**: Top mode reads a
global ranked index (cap stated in-UI); Cell Type mode shards by sender→receiver via `SliceCache`.
No cap on the small grains.

### B-5. Ack/KGG/Rme1 — 5xFAD only, do not surface empty
These channels are **not produced** in the Song/this-run `wide/` output (verified: no `_Ack_/_KGG_/_Rme1_`
columns). They self-gate via `block.score_columns`; surfacing them elsewhere would ship empty columns
(honesty rule). Extend `_INCYTR_SCORE_COLS` to carry `Ack_score`/`KGG_score`/`Rme1_score` **only when the
producing cohort emits them** — already the per-block mechanism; no empty columns added to Song/t-cell.

## JS work

### J-1. Merge into one Incytr screen
Collapse the two `TAB_MANIFEST` entries into **one**: a persistent **left filter side panel** +
a main pane with a **table / heatmap switch**. The heatmap-cell click stops being a tab jump — it
switches the main pane to the table with the panel's filters already reflecting the click. t-cell
inherits the collapse.

### J-2. Left filter panel — unified surface, one `IncytrFilter`
Sectioned vertical panel; **controls that don't apply to the active view are hidden** (not greyed —
vertical layout absorbs show/hide without reflow). One namespace, unified vocab (PDS sign =
`both/up/down` everywhere). Controls: grain · timepoint-combination · Top/CellType · Disease · Sender ·
Receiver · pvalue< · |PDS|≥ · PDS-sign · per-score gates · gene search · sparse-cell · Tissue (5xFAD
only) · trend. View-specific (hidden off-view): table row-cap/pagination; heatmap color-scale/axis-limit/
timeline.

### J-3. Grain selector + timepoint-combination filter
- **Grain**: `Full / L-R-EM / R-EM-T / R-EM`, default Full. Switches the entity source (B-2/B-4); the
  dropped node's column shows "—". No other rendering change.
- **Timepoint combination**: multiselect over {2mo, 4mo, 6mo} + all/any toggle (reuses the "Recur in"
  idiom). An **entity-level recurrence predicate** evaluated **within a single disease**, combined with
  the Disease filter: keep entities for which *some one disease's* surviving contrasts cover the chosen
  timepoints (all = AND, any = OR). The timepoint count is taken per-disease-row, never over the
  row-union across diseases — a backbone that appears at 2mo/4mo in AppP and only 6mo in Ttau does **not**
  qualify for "all 3 timepoints," because that is two separate disease signals, not one trajectory.
  Selecting all three + all = "present at 2mo and 4mo and 6mo of the same genotype." Qualifying entities
  render their per-contrast rows normally; the trajectory is read in the Scores drawer.

### J-4. Tissue-gating fix
Reproduce "Tissue visible under Song" against the live build first (visual authoritative), then fix
`_syncFivexfadTissueToggle` so the control is truly 5xFAD-only in the unified panel.

## Phasing (gated)
- **P0** — reproduce tissue bug; confirm grain factorization on Full (re-score == today's PDS, byte-exact).
- **P1** — B-1 package sub-spine scorer + verify Full parity.
- **P2** — B-2 driver → backbone `wide/`; size + floor checks, memory-capped.
- **P3** — B-3 tensors + B-4 payload/sizing + B-5 score-col gating.
- **P4** — J-1 merged screen + J-2 unified panel (no new filters yet). **Gate: browser, both cohorts.**
- **P5** — J-3 grain + timepoint filter; J-4 tissue fix. **Gate: browser, unified + t-cell.**
