# Decouple the T-cell attribution table from the contrast filter

## Context

The attribution subtab (kinase audit → attribution) renders a **Verdict across cell
types** table: one row per ProjecTILs T-cell state for the selected kinase. Today
the table is filtered by the globally-selected contrast day (`ctx.contrast`) via
`getScopedAttribution(kinase_id, {day: ctx.contrast})`. Two problems result:

1. **The table empties for scRNA-less days.** The viewer's contrast axis is the 5
   MEA/proteomics days (d13, d15, d17, d19, d20), but attribution rows only exist
   for the 3 days that also have a scRNA library (d13, d17, d20). Selecting **d15
   or d19** yields zero rows → "no attribution rows", which falsely reads as "this
   kinase has no cell-state localization at d15." The localization is in fact fully
   known and day-invariant.

2. **The day filter carries no localization information.** Verified on ACVR2A:
   `tcell_detected`, `tcell_fraction_expressing`, and `tcell_state_enrichment` are
   byte-identical across d13/d17/d20 (they are pooled across all scRNA days). Only
   `tcell_lfc`, `Decomp NES`, and the bulk `NES` vary by day. The contrast selection
   only picks *which day's LFC/NES to overlay* — a per-day detail, not a scope for
   the whole table.

**Intended outcome:** the attribution table renders for **any** selected contrast
(the localization is day-invariant), and the day-varying quantities (transcript
LFC, per-state Decomp NES, bulk NES) are shown as **heat-strips over the full 5-day
MEA axis inside each row's expand detail** — with the scRNA-less days (d15/d19)
drawn as explicit "no scRNA" gap cells. This turns the old empty-table bug into an
informative time-course view and makes the pooled-vs-per-day distinction legible.

Design decisions (locked with the user): **heat-strips** (reuse the existing NES-
profile cell grammar, not literal bars); strips are **expand-only** (rows already
expand); the table itself becomes day-invariant; the header drops the `/ {day}`.

## Files to modify

All changes are in the T-cell viewer template + its Python builder is untouched
(the data already supports this). **No edit to the shared engine**
`alz/viewer_shared/template/js/tabs/attribution_view.js`.

### 1. `alz/tcell_viewer/template/js/tabs/attribution_manifest_tcell.js`

- **`getRows(ctx)` (≈line 156):** stop scoping to the day — fetch all contrasts:
  `getScopedAttribution(ctx.kinase_id, {day: "", celltype: ""})`. This makes the
  table independent of `ctx.contrast`, so it renders for d15/d19 too.
- **`dedupKey(r)` (≈line 162):** key on `cell_type` only (drop `contrast_id`), so
  the engine's dedup Map collapses to one row per state. `dedupCmp` (highest
  confidence, tiebreak enrichment) is unchanged and lossless here — the surviving
  row's day-invariant columns are identical across contrasts.
- **Drop the two day-varying column definitions from `columns[]`:** `tcell_lfc`
  (≈81–91) and `_decomp_state_nes` (≈92–123). The remaining columns are all
  day-invariant: Cell state (+ confidence pill), Detected, Enrichment, Timecourse,
  NSCLC Detected.
- **`_renderTcellTranscriptTrace` (§1 renderer, ≈304–341):** this is where the
  heat-strips go. It already re-fetches the state's rows across all days
  (`getScopedAttribution(ctx.kinase_id, {day:"", celltype: cellType})`). Add, above
  the existing numeric per-day table:
  - a **bulk NES** strip (per-kinase, 5-day axis) — the activity anchor;
  - a **transcript LFC** strip (per-state, 5-day axis; d15/d19 = gap cells);
  - a **Decomp NES** strip (per-state, 5-day axis; from `_decompByKey`, key
    `` `${kinase_id}|${cid}|${state}` ``; d15/d19 = gap cells).
  Keep the existing numeric table below the strips as the exact-value readout.
- **Add a local heat-strip helper** `_renderAttrHeatStrip(valuesByCid, {maxAbs,
  selCid, gapTitle})` — generalizes the color/saturation logic of `_renderNesProfile`
  (`kinase_explorer.js:417`): iterate `CONTRASTS`, diverging red `[197,48,48]` for
  ≥0 / blue `[43,108,176]` for <0, saturation `0.15 + 0.85·min(1,|v|/maxAbs)`. A
  `null` value renders a muted `.npc.gap` cell titled "no scRNA library at {day} —
  transcript/activity undefined". The cell whose contrast == `ctx.contrast` gets a
  `.npc.sel` outline so the global selector keeps a subtle tie-in without gating.
  `maxAbs` is normalized **per kinase** (max |value| across that kinase's
  states×days for LFC/Decomp; across the 5 days for bulk NES) so a flat state reads
  flat and cross-state magnitude is comparable.

### 2. `alz/tcell_viewer/template/js/tabs/kinase_audit.js`

- **Header (≈line 945):** replace `Verdict across cell types … for {name} / {contrast}`
  with a day-invariant title, e.g. `Cell-state localization for {name}` plus a muted
  sub-line "pooled across all scRNA days · per-day transcript & activity in each
  row's detail". Drop the `ctx.contrast` reference.
- **`exportVerdictCsv` (≈966–981):** drop the per-day fields `tcell_lfc` /
  `tcell_concordance` from the exported keys (they would now be the arbitrary
  dedup-winner's day). Export localization only: `cell_type, confidence_tier,
  tcell_detected, tcell_fraction_expressing, tcell_state_enrichment,
  tcell_consistency`. Exact per-day values remain available via the existing **Raw
  attribution rows** table (`audit-attribution`, all contrasts).

### 3. `alz/tcell_viewer/template/styles.css`

- Reuse the existing `.tcell-nes-profile` / `.nes-profile-cell` / `.npc` grammar
  (≈429–449) — it already width-adapts to N contrasts via `--nes-profile-count`.
- Add `.npc.gap` (muted / diagonal-hatch fill, distinct from a zero-value cell) and
  `.npc.sel` (outline for the selected contrast). Add a small wrapper class for the
  §1 strip stack + day-label row if the existing `.nes-profile-col-labels` needs a
  variant.

## First implementation step

Copy this plan to `docs/plans/tcell/tcell_attribution_decouple_contrast.md` (project
rule: durable plans live in the repo, not the ephemeral staging path).

## Verification

1. `pixi run tcell-viewer` — rebuild; confirm exit 0 and it reports the donor1
   kinase slice + `decomposition_index (state MEA)` rows (Decomp strips need it).
2. `grep -c "npc gap\|_renderAttrHeatStrip" outputs/reports/tcell_viewer/index.html`
   — confirm the new render path is inlined into the built HTML.
3. Serve the viewer over HTTP (lazy parquet sidecars): open donor1 → kinase audit →
   attribution subtab.
   - Select **d15** (or d19): the table **still renders all states** (no "no
     attribution rows"). This is the core fix.
   - Expand a state with a detected transcript (e.g. a control kinase): the §1
     detail shows three strips on the 5-day axis — bulk NES filled at all 5 days;
     LFC and Decomp NES filled at d13/d17/d20 with **gap cells at d15/d19**; the
     currently-selected contrast cell is outlined.
   - Switch the contrast d13→d20: the **table does not change**; only the outlined
     cell in the strips moves. Confirms decoupling.
   - ACVR2A regression: its Enrichment/Detected are identical across any contrast
     selection (matches the pooled data verified during planning).
4. Export CSV from the verdict table → confirm no `tcell_lfc`/`tcell_concordance`
   columns; per-day values still exportable from the Raw attribution rows table.
