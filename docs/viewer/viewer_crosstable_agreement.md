# Unified Viewer — Crosstable Agreement View

## Goal

Reframe the unified viewer's **Crosstable** tab from a flat side-by-side mouse|human join into a view that **highlights agreement/intersection** in kinase activity between **song (mouse)** and **mukesh (human)**, matching the visual language of the per-dataset Kinase tabs. The tab's point is intersection/agreement — whether the two datasets concur — not a side-by-side column placement.

## Behavior

Master/detail layout, like the Kinase tabs:

- **Master** ~9 columns: identity + M-glyph + Agree + H-glyph + M/H specificity tier. No per-sample columns, Columns panel, or legend.
- **`#kx-detail` panel** (right of a draggable splitter) = cross-dataset comparison with a verdict header + 2 sub-tabs (NES Activity, Cell-type Specificity), each Mouse|Human, **reusing the kinase-tab renderers verbatim**: `_renderKinaseNesPlot` / `_renderKinaseCelltypeEvidence` (mouse), `_khRenderNESAcrossDonors` / `_khRenderAttribution` (human).
- Selection is **local** (no `SET_SELECTION`).
- **Full union** of both datasets' kinases; join key = kinase abbreviation + residue (species-neutral, **no ortholog map**).
- **Direction + co-significance** classification recomputed **live** against the FDR slider. Categories: concordant-up, concordant-down, discordant, mouse-only, human-only, neither.
- Compute is client-side JS only (`js/tabs/kinase_crosstable.js`) — no `build_payload` / Python change.

## Conventions when extending this tab

- Reuse existing cell renderers/glyphs — **do not greenfield** (see `feedback_no_reimplementing_shared_viewer` → CLAUDE.md "Viewer ports = lift, not rewrite").
- Sign convention: **+ = up in disease** in both datasets (see `reference/incytr_sce4_reproduction.md` / lfc sign convention).
- Append new requirements to the backlog before implementing.
