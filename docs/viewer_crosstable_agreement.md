# Unified Viewer — Crosstable Agreement View

Source: `project_crosstable_agreement_view.md` (multi-session work started 2026-06-04).

> The memory referenced an SSOT plan at `docs/plans/crosstable_agreement_view_2026-06-04.md`, which is **not present on disk**. This doc is the surviving record; recreate the plan file if you resume and want a backlog.

## Goal

Reframe the unified viewer's **Crosstable** tab from a flat side-by-side mouse|human join into a view that **highlights agreement/intersection** in kinase activity between **song (mouse)** and **mukesh (human)**, matching the visual language of the per-dataset Kinase tabs. The original tab computed nothing about whether the two datasets concur — it just placed columns side by side; the point of the tab is intersection/agreement.

## Current state (Iteration 2, shipped 2026-06-04, was uncommitted at write time)

Master/detail layout, like the Kinase tabs:

- **Master** slimmed to ~9 columns: identity + M-glyph + Agree + H-glyph + M/H specificity tier. All per-sample columns, the Columns panel, and the legend were removed.
- **`#kx-detail` panel** (right of a draggable splitter) = cross-dataset comparison with a verdict header + 2 sub-tabs (NES Activity, Cell-type Specificity), each Mouse|Human, **reusing the kinase-tab renderers verbatim**: `_renderKinaseNesPlot` / `_renderKinaseCelltypeEvidence` (mouse), `_khRenderNESAcrossDonors` / `_khRenderAttribution` (human).
- Selection is **local** (no `SET_SELECTION`).
- Shared-code fix: `_khRenderAttribution` donor-select → `body.querySelector`. Splitter centralized as `_wireSplitter` in `02_ui_chrome.js` (ka/kh/kx).
- **Known v1 limitation:** human NES-bar click re-targets `#kh-detail`.

## Iteration 1 (earlier, also uncommitted at write time) — all in `alz/viewer/template/`

`js/tabs/kinase_crosstable.js` (+ `body.html`, `styles.css`). Locked decisions:

- Grouped collapsible table (**no scatter**).
- **Full union** of both datasets' kinases (was mouse-keyed, which hid human-only kinases).
- **Direction + co-significance** classification recomputed **live** against the FDR slider.
- Join key = kinase abbreviation + residue (species-neutral, **no ortholog map**).
- Compute is client-side JS only — no `build_payload` / Python change.
- Categories: concordant-up, concordant-down, discordant, mouse-only, human-only, neither.

## Conventions when extending this tab

- Reuse existing cell renderers/glyphs — **do not greenfield** (see `feedback_no_reimplementing_shared_viewer` → CLAUDE.md "Viewer ports = lift, not rewrite").
- Sign convention: **+ = up in disease** in both datasets (see `incytr_sce4_reproduction.md` / lfc sign convention).
- Append new requirements to the backlog before implementing.
