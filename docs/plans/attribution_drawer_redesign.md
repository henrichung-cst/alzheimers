# Attribution drawer — refactor & redesign

> Current architecture source of truth:
> [`docs/foundation/kinase_explorer_attribution.md`](../foundation/kinase_explorer_attribution.md).
> This file is implementation history for the Song AD attribution redesign; use
> the foundation doc for live cross-cohort Attribution tab behavior.

**Status:** IMPLEMENTED (all phases 0–6 landed; viewer rebuilt, 20 attribution
tests green).
**Scope:** the per-cell-type evidence drawer on the Audit tab — now
`_renderAttributionDetail` + `_renderSpecificityVerdict` and the sub-renderers in
`alz/viewer/template/js/tabs/kinase_audit.js`.

**Phase 0 deviations (producer additions Phase 0 proved necessary):**
1. **The eff the pill uses was never exported.** `song_effective_n` is the
   *native* 31-cluster eff; the confidence tier is set by the *unit-level* eff
   (`1/Σ unit_share²`) computed in `specificity_class.py` and discarded. They
   diverge for collapsed kinases (CDK9: native 5.64 vs unit 2.55 → `high`).
   Showing the native eff in §0 would have re-created the confusion it set out to
   fix. Added `song_unit_effective_n` to the `assign_specificity_class` output →
   `unified_attribution.csv` → `attribution_index`; §0 uses it. Required a
   `pixi run attribute` regen, not just a viewer rebuild.
2. **The WMB-class crosswalk was not in the payload.** `cluster_to_unit` is not a
   cluster→WMB-class map (only equals it for collapsed units). Added
   `cluster_to_wmb_class` and `cluster_to_seaad_subclass` to the
   `specificity_units` payload (`_build_specificity_units`).

---

## 1. What the drawer is today

The Audit tab shows a per-(kinase × contrast × cell-type) **verdict table**
(`_renderAttributionVerdict`, line 665). Clicking a cell-type row opens the
**evidence drawer** (`_renderAttributionDrawer`, line 895) in the side panel
`#attr-drawer`. The drawer has four sub-renderers:

| Section | Renderer | Shows | Source |
|---|---|---|---|
| WMB expression dot plot | `_renderWMBDotPlot` (997) | mean log2 expr × fraction cells, across WMB classes | `ctx.wmbRows` (`wmb_kinase_expression.csv`) |
| SEA-AD supertype LFC heatmap | `_renderSEAADHeatmap` (1049) | per-supertype disease LFC, grouped by subclass | `ctx.seaSuperRows` (`sea_ad_supertype_lfc.csv`) |
| Song within-cohort OLS | `_renderSongOLSPanel` (1109) | β / SE / p / FDR of the disease contrast | `SliceCache.loadSongConcordance` shard |
| Per-cell substrate-site OLS | `_renderDecompOlsTable` (923) | per-site β driving the Decomp NES | `SliceCache.loadDecompOls` shard |

**The through-line:** every section is about **disease direction** (LFC, OLS)
or **mechanism** (decomp site OLS). None of it is about *expression
specificity*.

---

## 2. Why it is out of date (findings, verified)

1. **Conceptual mismatch — the headline problem.** The drawer is the
   deep-dive for the **confidence pill**, but that pill was redesigned this
   cycle into a **cell-type-exclusivity** verdict (curated specificity units,
   `eff = 1/Σ share²`, detection-gated — see
   [`docs/foundation/specificity_confidence.md`](../foundation/specificity_confidence.md)).
   The drawer shows **zero specificity evidence**: no `eff`, no detected-set
   share, no count of cell types detected, no unit membership, no collapse
   reveal. So clicking a row to "understand the verdict" shows direction and
   mechanism — a different axis than the pill. This is exactly the TGFBR1
   confusion: the Microglia row reads 91% of cells / ≥10× concentration, yet
   the pill is `low` (eff ≈ 4.1, detected in 21/31 cell types), and the drawer
   offers nothing to reconcile it.

2. **Dead field `specificity_score`.** `_renderWMBDotPlot` line 1033 builds its
   tooltip from `r.specificity_score`. That column was retired; the live
   `wmb_kinase_expression.csv` header is
   `…detected,concentration,concentration_tier` — no `specificity_score`. The
   tooltip therefore always prints `specificity = 0.000`. The live signal is
   `concentration` / `concentration_tier`.

3. **Cross-vocabulary highlight silently fails.** Both reference panels promise
   "target cell type is outlined" by string-equality:
   - WMB dot plot (1029): `r.cell_type === targetCellType` — `r.cell_type` is a
     **WMB class**; `targetCellType` is a **Song Levy cluster**.
   - SEA-AD heatmap (1068, 1088): `sc === targetCellType` — `sc` is a **SEA-AD
     subclass**; again a Song cluster.
   These vocabularies differ (e.g. Song `Vascular-Leptomeningeal-Cells` vs WMB
   `33 Vascular`), so the outline matches only on lucky name collisions and is
   absent the rest of the time. The crosswalk to fix it
   (`specificity_units.cluster_to_unit`: cluster → WMB class) is already in the
   payload.

4. **Legacy `pathway` shim.** `_renderSongOLSPanel` (1130–1134) branches on
   `_useContrast` and falls back to a legacy `pathway` key; the producer mirrors
   this (`evidence.py:289`, `contrast` else `pathway`). The schema migrated to
   the 9-contrast key. Per the anti-shim rule, the legacy branch should be
   removed once Phase 0 confirms the live shard is contrast-keyed.

5. **Stale captions.** The SEA-AD caption (908) says "Subclass median is used in
   the verdict table." The verdict no longer uses subclass-median LFC; human
   references corroborate via `human_location_score`. Captions and the
   explainer legend describe the pre-redesign model.

---

## 3. Redesign principles

- **The drawer must explain the verdict it is attached to.** Specificity is the
  pill's axis, so specificity leads. Direction and mechanism stay, but as
  clearly-labelled secondary evidence tied to `direction_tier`, not as the
  whole story.
- **No grouping shown without its parts** (the standing viewer rule): a
  collapsed unit's child Song clusters are revealed in-drawer.
- **Real vocabulary only** (`eff`, "specificity unit", "concentration",
  `confidence_tier` / `direction_tier`) — no reintroduced `specificity_score`,
  no invented terms.
- **Vocab-correct the cross-reference highlights** via the existing
  cluster → WMB-class crosswalk; never string-match across vocabularies.
- **Kill dead code and shims in the same pass** (anti-shim).
- **Reuse, don't greenfield** — keep the SVG dot plot / heatmap / shard-loading
  scaffolding; change what they read and how they're framed.

---

## 4. Container: inline expandable rows (replaces the bottom drawer)

The deep-dive moves **into the verdict table**. Clicking a Cell-type cell
toggles an inline **detail row** (`<tr class="attr-detail-row"><td colspan=N>…`)
directly beneath it, holding the §0–§3 content below. The separate "Evidence
drawer" `<section>` + `#attr-drawer` host (kinase_audit.js:1453–1454) is
**removed** — a pivot replaces, it does not coexist (no bottom panel kept
alongside the inline expander).

Interaction decisions:
- **Single-expand (accordion) — decided.** Opening a row collapses any other
  open one. This preserves today's single-selection behavior, keeps it obvious
  where you are, and bounds the async/SVG cost to one detail at a time.
- **Lazy render.** The detail content renders only when a row is expanded
  (exactly as the drawer renders on-demand today), into the detail row's
  `<td>` instead of `#attr-drawer`. The four sub-renderers already take a
  `hostId`, so they are reused verbatim against the inline container's id.
- **Expand affordance.** The Cell-type cell gets a chevron (▸/▾) and
  `aria-expanded`; the whole row toggle replaces the current
  select-row-highlights behavior.
- **Default + re-sort.** On (re)render the top row auto-expands (mirrors the
  current "open the top row" default); sorting re-renders and re-opens the top
  row.
- **Width.** The detail `<td>` content (720 px dot plot, heatmap) lives in its
  own `overflow-x:auto` container so the table body never scrolls horizontally.

The §0–§3 content below is unchanged by this; only its container moves.

## 4b. Detail content (the §0–§3 sections)

Reorder and add one headline section; keep the existing three as secondary.

**§0 — Specificity verdict (NEW, top).** The reconciliation the drawer is
missing. Two blocks:
- *This cell type* (clicked row): % of cells expressing
  (`song_fraction_cells_expressing`), share of detected-set expression
  (`song_concentration`, shown as a raw % — surfaced nowhere today),
  concentration tier (`song_concentration_tier`) with its baseline spelled out
  (even share = 1/n_detected), and the specificity unit it belongs to
  (`specificity_unit_label`). If the unit is **collapsed**, list its sibling
  Song clusters with per-cluster detection from
  `PAYLOAD.specificity_units.units[unit].children`.
- *This kinase overall*: detected in **N / 31 cell types** (N / 17 units),
  **eff** (`song_effective_n`) with the band `≤1.5 one · ≤3 a few · >3 broad`,
  and a **full reconciliation with the eff math shown (decided — Option B):** a
  2–3 line explanation, always visible (not behind a toggle), e.g. *"Confidence
  scores exclusivity as eff = 1 / Σ(unit share²). Microglia holds 46% of
  detected-set expression (0.46² = 0.21 of Σ = 0.24); the remaining ~20 detected
  cell types add the rest → eff ≈ 4.1. Since 4.1 > 3, the kinase is broadly
  expressed → `low`, even though Microglia alone is ≥10× its even share."* The
  numbers are filled per kinase from `song_concentration` (folded onto units)
  and `song_effective_n`.

**§1 — Expression across references.** The WMB dot plot, fixed: tooltip reads
`concentration` / `concentration_tier` (drop `specificity_score`); the target
outline maps the clicked Song cluster → its WMB class via `cluster_to_unit`
before matching. **Human location score lives here (decided — Option A):** show
`human_location_score` (= max of `seaad_location_score` / `hbca_location_score`,
log2 over brain mean) for the clicked cell type's WMB class, so "where is it
expressed" spans mouse (WMB) + human in one section, beside the specificity
evidence it corroborates. This is the human signal the pill actually uses; it is
distinct from the SEA-AD disease LFC, which stays in §2. Flag when the score is
**strong** (≥ 1.0 log2, `HUMAN_STRONG_LOG2_SPECIFICITY`) — the threshold at
which human corroborates — to back the §0 corroboration line.

**§2 — Disease direction.** The SEA-AD heatmap + Song OLS, reframed as the
**`direction_tier`** evidence (this answers "does its activity move with
disease", the pill's *prior* meaning, now `direction_tier`). Vocab-correct the
SEA-AD target highlight the same way; rewrite the stale "subclass median" caption.

**§3 — Mechanism.** The per-cell substrate-site OLS, unchanged in intent (what
drives the Decomp NES); caption refreshed.

---

## 5. Data availability

All §0 inputs already ship — **no Stage-3 rerun**:
- `song_concentration`, `song_fraction_cells_expressing`,
  `song_concentration_tier`, `song_effective_n`, `specificity_unit`,
  `specificity_unit_label`, `specificity_collapsed` — all on `attribution_index`.
- `n_detected` — count of `song_detected` rows already in the table.
- collapse children + cluster → WMB-class — `PAYLOAD.specificity_units`.

Phase 0 verifies: the live `song_concordance` shard key (`contrast` vs
`pathway`); that `cluster_to_unit` covers every cluster the dot plot shows; and
that the drawer receives the clicked **row object** (not just the cell-type
string) so §0 can read its per-row fields without re-deriving.

---

## 6. Phases

- **Phase 0 — contract check.** Confirm shard key (`contrast` vs `pathway`),
  crosswalk coverage, the verdict table's column count (for the detail row's
  `colspan`), and that the clicked **row object** is reachable so the detail can
  read its per-row fields without re-deriving.
- **Phase 1 — inline-expand container (relocate only, no content change).**
  Convert the "Evidence drawer" panel into an expandable detail row in the
  verdict table; remove the `#attr-drawer` section; point the four existing
  sub-renderers at the inline host. Single-expand accordion, lazy render,
  chevron affordance, top-row default. Verify parity with the old drawer before
  changing any content.
- **Phase 2 — cleanup.** Remove the `specificity_score` tooltip ref →
  `concentration`/`concentration_tier`; remove the `pathway` fallback if Phase 0
  confirms it dead.
- **Phase 3 — vocab-correct highlights.** Map Song cluster → WMB class (and the
  SEA-AD subclass equivalent) for the "target outlined" logic in both panels.
- **Phase 4 — add §0 Specificity verdict** (the headline), incl. collapse-child
  reveal and the reconciliation line.
- **Phase 5 — reframe & captions.** Reorder into §0–§3, rewrite captions and the
  Audit explainer legend to the current model (define `eff`; `confidence_tier`
  = where, `direction_tier` = disease direction).
- **Phase 6 — rebuild & verify.** `node --check`; rebuild under the 8 G cap;
  spot-check TGFBR1 (§0 reconciles ≥10×↔`low`), a `very_high` (BTK), and a
  collapsed-unit kinase (CDK9). Hard-refresh note.

## 7. Out of scope
Producer/payload schema (unless Phase 0 finds a gap); the verdict table columns
(the redesign deliberately moves depth into the drawer, not new columns); other
tabs (Explorer/Crosstable were already fixed; 5xFAD/Human/tcell drawers are
separate).
