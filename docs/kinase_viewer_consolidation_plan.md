# Kinase Viewer Data-Fidelity & Filter Consolidation Plan

Status: DRAFT — awaiting approval before implementation.

## Goal

Eliminate data discordance in the Kinase tab by collapsing to **one source of truth per quantity** and **one filter bar that every component respects**.

## Problems being solved (from audit)

1. Pill HIGH-criterion ≠ Verdict HIGH rows (different upstream tables)
2. NES Profile color scale is global, not filter-scoped
3. `_sigCount` sort key ignores filters; row-inclusion respects them — sort contradicts visible set
4. `attribution_index` is pre-filtered to high+moderate; verdict reads full table — counts don't match
5. Three same-row visualizations operate on three different contrast scopes with no labeling
6. Filter state scattered across `window._keFilters`, `Store.state.filters.contrast`, `keSortCol`, `keNesLensIdx`, `keSearch`, `host.dataset.*`, `Store.state.view.kinaseAuditTab`, local `ctx.contrast`

## Architecture

### Single source of truth: `PAYLOAD.attribution_index`

Build one columnar index containing **every** `(kinase, contrast, cell_type)` row from `unified_attribution_full.csv`, including `low` and `none` tiers. Columns:

```
kinase_id, contrast_id, cell_type,
combined_confidence, combined_score,
wmb_specificity, sea_ad_lfc, song_lfc, concordance_source,
nes, fdr  -- denormalized from kinase_activity_matrix for fast filter ops
```

All five derived quantities — pill, cell-types count, verdict rows, sort keys, NES Profile — read from this index applying the unified filter at render time. Drop:
- `has_high_conf_attribution` baked-in flag (derive live)
- Separate `unified_attribution_full` audit CSV fetch (already in payload)
- `_attrList`, `_attrCellSet`, `_highCtxSet` precomputed maps (compute under filter)

### Single filter bar (top of Kinase tab)

One sticky bar replaces the current scattered controls:

```
[Search ____________] [Disease ▾] [Timepoint ▾] [Cell type ▾] [Confidence ▾] [FDR ≤ 0.25 ◯─] [Trajectory ▾] [Reset]
```

Backed by one state object:
```js
window.KinaseFilter = {
  search: "",
  disease: "any",      // any | App | Tau | ApTt
  timepoint: "any",    // any | 2mo | 4mo | 6mo
  celltype: "any",     // any | <34 WMB classes>
  confidence: "any",   // any | high | moderate | low
  fdr: 0.25,
  trajectory: "any"
}
```

Removed UI elements:
- Per-tab contrast picker (derived from disease+timepoint; if both ≠ any, exact contrast is determined; if either is `any`, components show "scoped to N contrasts" label)
- Sample picker on detail panel (move to detail-only sub-control if needed)
- Verdict table's local `showAll` toggle (replaced by Confidence filter)
- NES Profile lens cycle (`keNesLensIdx`) — folded into Disease filter
- Audit-subtab contrast dropdown
- The four-dropdown ke-toolbar-multi (replaced by the unified bar)

Persisted to localStorage as `kinaseFilter.v2`.

### Component-by-component contract

| Component | Source | Filter scope | Label |
|---|---|---|---|
| Ranked table row inclusion | `attribution_index` | search + trajectory + (disease+tp scope) + celltype + confidence | n/a |
| Confidence pill | `attribution_index` filtered to current scope; HIGH = ∃ row with `combined_confidence == "high"` | full active filter | "HIGH (in App+4mo)" or "HIGH" if no scope |
| NES Profile glyph | `kinase_activity_matrix` | always renders 9 cells; **color scale = max\|NES\| over visible kinases × scoped contrasts**; non-scoped contrast cells dimmed 40% | tooltip names scope |
| Cell-types cell | `attribution_index` filtered to scope | full active filter | "3 cell types in App+4mo" |
| MEA bar chart | `kinase_activity_matrix` | always 9 bars; bars outside scope dimmed; selected contrast bordered | scope label above |
| Attribution Verdict table | `attribution_index` | full active filter; if disease+tp narrow to single contrast, that contrast; else aggregates across scoped contrasts with `contrast` column shown | scope label above table |
| Sort: NES, n_sig, n_attributed | derived under filter scope | filter scope | header tooltip |

### Smart default

When user clicks a kinase row, if no contrast is uniquely determined by the filter, set disease+tp to the contrast containing the kinase's highest-confidence row (high>moderate>low). This replaces both `peak_NES` fallback and the contrast picker.

## Implementation phases

**Phase 1 — Payload (Python)**
- `_build_attribution_index` in `build_unified_viewer.py`: rebuild from `unified_attribution_full.csv`, no tier filter, denormalize NES/FDR
- Drop `has_high_conf_attribution` from `_build_kinases_slice` (or keep but mark deprecated; viewer ignores it)
- Drop separate audit-CSV fetch for `unified_attribution_full` (now in payload)

**Phase 2 — Filter state + UI (JS)**
- New `KinaseFilter` state object with subscribe pattern
- New unified filter bar HTML/CSS replacing `ke-toolbar-multi`
- localStorage v2 schema; one-time migration ignores v1
- Remove `keSortCol` global → fold into `KinaseFilter.sort`; remove `keNesLensIdx`, `keSearch`
- Remove contrast picker, sample picker, verdict showAll toggle, audit subtab contrast dropdown

**Phase 3 — Render path**
- `getScopedContrastIds(filter)` helper used by every component
- `getScopedAttribution(kinase_id, filter)` helper returning filtered rows
- Rewrite: `_buildKinaseRowModel` (drop precomputed maps), `_renderNesProfile` (filter-scoped maxAbs), `_renderCellTypesCell` (filter-scoped count), `renderKinaseExplorer` pill (live from index), `_renderAttributionVerdict` (no local state, reads filter), `_renderMeaTrajectory` (dim non-scoped bars), `_keCompare` (sort keys derived under scope)
- Add scope-label component shown above each visualization

**Phase 4 — Smart default + cleanup**
- Click handler on row sets filter to peak-confidence contrast if filter is `any/any`
- Remove dead code: `SCORE_*` remnants, `_attrList`/`_attrCellSet`/`_highCtxSet` if unused, audit-context `ctx.contrast` derivation

**Phase 5 — Verify**
- Audit kinases AURA, AURB, plus 3 random others against raw `unified_attribution_full.csv` to confirm pill ↔ verdict ↔ cell-types count agree under every filter state
- Run `pixi run live` end-to-end is NOT required (viewer-only change); rebuild viewer with `python code/build_unified_viewer.py`

## Out of scope

- Backbone tab, Pathway tab, Cross-entity tab — unchanged
- Attribution-recovery Python pipeline — unchanged (we read its outputs)
- Any retroactive provenance fixes to existing CSVs

## Risks

- Payload size: full `unified_attribution_full` is ~9 contrasts × 34 WMB classes × ~400 kinases = ~122k rows. Columnar JSON ≈ 6-8 MB uncompressed; gzip ~1.5 MB. Acceptable.
- localStorage v1 → v2: users with bookmarked filter state will lose it. Acceptable, document in changelog.

## Approval

Reply "go" to proceed with Phase 1, or specify changes.
