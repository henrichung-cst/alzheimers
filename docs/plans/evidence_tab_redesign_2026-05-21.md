# Plan — Evidence Tab Redesign (Information-Dense Matrix)

**Date:** 2026-05-21
**Scope:** `alz/viewer/template/js/widgets/evidence_row.js` + Evidence-related CSS in `alz/viewer/template/styles.css`. No changes to substrate builds, payload shape, or upstream Incytr.

## Problem

The current Evidence panel (Phase 3 of the merged-evidence epic) is functionally correct but visually noisy. The layout wastes vertical space, surfaces inconsequential numeric noise as eye-catching failures, and uses unclear language for the three distinct "missing data" states. Specific pain points called out:

1. **Vertical stacking.** Each of the 4 node columns stacks 4 layer blocks vertically; phospho layers further fan out into per-site sub-rows. A pathway with 6 phospho sites across pS+pY renders 30+ stacked SVGs per row — far below the screen's information capacity.
2. **FAIL threshold is too tight for the display precision.** A Δ of `1.18e-4` between stored and recomputed LFC is well within numerical round-trip tolerance, but at 2-decimal display it shows as `0.702 vs 0.702` — visually identical — yet the UI screams `FAIL stored=0.702 recomputed=0.702 Δ=1.18e-4` in bold red. The build-time assertion (`verify_pathway_round_trip.py`, 1e-4 abs) is correctly load-bearing; the **UI** threshold should be display-aware.
3. **Dots cover the mean-value text.** `_renderDotBarSvg` paints the mean number at `y = barY - 2` directly above the bar; deterministic dot jitter lands several dots over the same text.
4. **Three "missing" states blur together.** `no data`, `n/a`, and `LFC 0.000` currently look similar at a glance but mean different things:
   - **no gene on node** (e.g. Receptor empty in this pathway row) — the slot is structurally absent
   - **gene measured but absent from this layer** (e.g. Apoe has no pY sites)
   - **gene measured, value is genuinely zero** (true negative)
5. **Phospho numbers' meaning is unclear.** A reader sees a bar labelled `0.42` under "Phospho (pS/pT)" with no indication whether that's raw IRS-normalized intensity, log2, per-site, per-gene aggregated, or something else. Incytr aggregates phospho sites → gene by `mean` upstream, but the dot-bar shows per-site rows for transparency — the two coexist without a label distinguishing them.

## Goal

A glanceable 4×4 matrix that lets a reader vertically trace pathway evidence from raw measurements → group means → derived LFC → Incytr's stored value, with explicit units, unambiguous missing-data states, and only flagging mismatches that matter at 2-decimal display precision.

## Design

### 1. Replace vertical stacks with a true matrix

Lay out as a CSS-grid table:

```
              | Transcript | Protein | Phospho pS | Phospho pY |
--------------|------------|---------|------------|------------|
  Ligand   X  |   ▮▯  +0.31|  ▯▮ -0.12|    n/a    |    n/a    |
  Receptor Y  |   ▯▮ -0.08 |  ▮▯ +0.45| ▯▮ -0.21▾ |    n/a    |
  EM       Z  |   ▮▯ +1.10 |  ▮▯ +0.88|    n/a    | ▯▮ -0.04  |
  Target   W  |   ▮▯ +0.22 |   n/a   | ▯▮ -0.55▾ |    n/a    |
```

- 4 rows × 4 columns. Each cell ≈ 110px × 44px. Whole grid fits in ~600×220 px (vs. current ~1100×900).
- One cell = one micro dot-bar plot + an LFC chip on the right edge. No per-site fan-out at the top level (see §4).
- Row labels (`Ligand`, `Receptor`, etc.) carry the gene symbol and the cluster name as `node · gene · cluster` in muted small-caps; tooltip shows full cluster.
- Column headers carry the layer name + unit string and a `?` tooltip with the exact transformation: e.g. `Protein · log₂(IRS-norm + ε)`.

### 2. Micro dot-bar plot (~110×44 px)

- Two adjacent vertical bars per cell: WT (left, grey) and Disease (right, accent). Bar height = mean. Bar width = `n` (number of animals, capped — usually 3, so visually constant).
- Per-animal dots overlaid as a small horizontal strip **below** the bars, in a dedicated 8px band, not on top. Dot radius 2px, opacity 0.6.
- Mean label moves to the right-edge LFC chip area — never inside the SVG.
- Y-axis hidden; bars share a common per-cell scale (max of |WT mean|, |Disease mean|). Add a thin zero baseline only when both means are positive (rare for transcript/protein), else implicit.
- Tooltip on hover: `WT n=3 mean=0.74 [0.62, 0.78, 0.82]` / `ApTt n=3 mean=0.91 [0.85, 0.92, 0.96]`.

### 3. LFC chip — replace FAIL banner

Right-edge chip per cell, two-row mini:

```
+0.31      ← stored value, mono font, sign-coloured
✓ Δ<.005   ← only shown when |Δ| visible at 2-dec; otherwise blank
```

Rules:

| State | Display |
|---|---|
| Stored & recomputed agree at 3 dec (|Δ| < 0.005) | Stored value only, no decoration. (Build-time 1e-4 assertion is the source of truth — UI doesn't need to advertise sub-display agreement.) |
| Stored & recomputed differ at 2-dec (|Δ| ≥ 0.005 AND |Δ|/max(|stored|,0.1) ≥ 0.01) | Show stored in normal weight, recomputed below in italic, small `⚠` icon with tooltip `recomputed=X stored=Y Δ=Z` |
| No stored value (gene not in Incytr output for this node) | `LFC —` muted, no chip |
| Gene present in Incytr but no substrate row (can't recompute) | Show stored only, small `i` icon with tooltip "no substrate to verify" |

The build-time `verify_pathway_round_trip.py` keeps the strict 1e-4 absolute threshold. The UI threshold is purely a "would a reader see a discrepancy at 2-decimal display" filter.

### 4. Phospho — gene-level by default, sites on disclosure

Top-level phospho cell shows the **gene-aggregated** value (mean across sites within each animal × group, matching Incytr's upstream `summarise_all(mean)` in `bench/incytr_pair_levy_t5/incytr_commandline.R` — confirmed in Item 3.1). This is what Incytr's stored `*_ps_log2FC` / `*_py_log2FC` actually consumes.

A small `▾` caret in the cell footer reveals a popover (not inline) with the per-site dot-bar strips, one row per `site_id`. Closes on outside click. This keeps the per-site detail one click away without bloating the default view, and explicitly labels it as "Site-level detail (informational; Incytr aggregates to gene-mean before LFC)".

### 5. Three "missing" states get distinct treatment

| Cause | Visual | Tooltip |
|---|---|---|
| Node has no gene (e.g. `Receptor` blank in row) | empty cell, light grey hatched background | "No gene on this node for this pathway" |
| Gene not measured in this layer (no shard rows for this gene × layer) | `n/a` in muted italic, no SVG | "Gene `X` has no sites in `phospho_py` for cluster `Y`" |
| Measured, value zero (rows exist, all values=0) | dot-bar with bars at zero, dots at zero line | "Measured; both arms have zero intensity" |

The `LFC —` text is reserved exclusively for the LFC chip when there's no stored value. The cell body never says `LFC —`.

### 6. Sub-cell affordances retained from current build

- Arm colour: WT=`#777`, Disease=`#a3203c` (matches the rest of the viewer).
- Hover tooltip lists the underlying animal IDs + per-animal values (current build already has the data, just needs to render).
- Click-to-copy on the LFC chip copies `recomputed=X stored=Y Δ=Z` for QA workflows.

### 7. Header strip (above grid)

Single-line meta: `App_2mo · sender=Astrocytes · receiver=Basal-Ganglia · males-only · n=3 vs n=3`. Removes the current `dots=animals · bars=mean` legend (legend moves into a `?` tooltip on the grid).

## Files to modify

| File | Change |
|---|---|
| `alz/viewer/template/js/widgets/evidence_row.js` | Rewrite `EvidencePanel.render` to build a single `<table>` (or CSS-grid 5-col layout) instead of 4 stacked `.ev-col` blocks. Replace `_renderDotBarSvg` with `_renderMicroDotBar` (smaller SVG, dots-below-bars, no in-SVG label). Replace `_renderLfcSlot` with `_renderLfcChip` using the 2-state UI threshold above. Add phospho gene-aggregation pre-pass (mean across `site_id` per animal × group), with site-level rows reachable via popover. Add three explicit missing-state branches. Move legend to tooltip. |
| `alz/viewer/template/styles.css` (lines 568–595) | Replace `.ev-grid`, `.ev-col`, `.ev-layer-block`, `.ev-site-block` with `.ev-matrix` (5-col CSS-grid: label + 4 layers), `.ev-row`, `.ev-cell` (110×44). Drop `.ev-lfc-fail` red banner — replace with `.ev-lfc-warn` (small icon + tooltip). Add `.ev-cell-empty-hatch` for "no gene on node" state. Update `@media` breakpoint to drop the matrix to 3-col (transcript + protein + collapsed phospho) at <900px. |
| `docs/plans/merged_evidence_panel.md` | Append a closing note pointing to this plan as the UI iteration follow-up; do not reopen the epic. |

## Non-goals

- No changes to the omics_trace / omics_trace_normalized parquet shards, to `build_omics_trace.py`, or to `build_normalized_substrate.py`. The 1e-3 / 1e-5 epsilons, the limma-normalized substrate, and the gene-level aggregation in the driver all stay as-is.
- No change to `verify_pathway_round_trip.py`. Its 1e-4 absolute threshold remains the build-time gate — only the **UI** threshold for surfacing mismatches changes.
- No change to the underlying Incytr pair-mode driver or `Cal_pairwise_grid` semantics.
- No new payload meta keys unless strictly required. (Plan currently requires none — all behaviour drives off existing `omics_trace`, `omics_trace_normalized`, `transcript_trace` meta.)
- No removal of phospho site-level data — it stays one click away in the popover.

## Verification

1. **Visual smoke.** Open the user's reference pathway (`Apoe|App|Stk11|Cttnbp2`, Astrocytes → Basal-Ganglia-GABAergic-Neurons, `ApTt_2mo`). The full 4×4 matrix must be visible without scrolling on a 1440-wide window. All four nodes show transcript + protein cells; pS/pY cells either show data or one of the three distinct missing-state visuals.
2. **No false-positive FAIL.** The historically-flagged `Δ=1.18e-4` cell must render as plain stored value with no decoration. Hover tooltip must still show recomputed value for QA.
3. **Mismatch still surfaced when it matters.** Hand-edit a shard row to perturb a recomputed value by 0.05 from stored; reload; the cell must show the `⚠` icon with the tooltip listing both numbers.
4. **Phospho aggregation correct.** Pick a gene with ≥2 phospho sites for the receiver cluster. The top-level cell's mean must equal `mean(site_means)` per animal × group. Open the popover; per-site bars must sum-match the aggregated value to ≤1e-6.
5. **Round-trip still passes at build time.** `pixi run viewer` (which now calls `verify.verify(strict=False)` per Item 3.5) must still report `0 failures` across 961 shards. No build-time threshold changed.
6. **Three missing states distinguishable.** Find one pathway exhibiting each of the three missing-data causes; screenshot the cells; confirm visually distinct.

## Out-of-scope follow-ups (not included)

- Adding a "compare two pathways" side-by-side mode.
- Persisting popover-expanded site state across pathway navigation.
- Per-cell sparkline of the time course (2mo/4mo/6mo) — would require a separate substrate join.
