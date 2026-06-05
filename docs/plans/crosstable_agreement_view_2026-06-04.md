# Crosstable → cross-dataset agreement view

**Status:** iteration 1 shipped (uncommitted as of 2026-06-04). Multi-session work in progress.
**SSOT for this work.** Supersedes the staging plan `~/.claude/plans/we-want-to-make-curried-moon.md`.

## Goal

Reframe the unified-viewer **Crosstable** tab from a flat side-by-side mouse|human join
into a view that **highlights intersections / agreements** in kinase activity between
**song (mouse)** and **mukesh (human)**, in the same visual language as the per-dataset
Kinase tabs (NES rank, brain-tissue specificity, detail).

## Locked design decisions (user-confirmed 2026-06-04)

- **Layout:** table grouped by agreement category, collapsible group headers with counts.
  **No scatter plot** (considered, rejected).
- **Row universe:** **full union** — every kinase in *either* dataset (mouse-only,
  human-only, shared). The tab was mouse-keyed before and hid human-only kinases.
- **Agreement rule:** **direction + co-significance**, significance recomputed **live**
  against the FDR slider (NOT the build-time `n_sig_*` columns).
- Cross-dataset join key = kinase abbreviation + residue (`name|residue`). Abbreviations
  are species-neutral; **no ortholog mapping** is applied or needed at the MEA level.
- Compute is **client-side JS only** — all inputs already in PAYLOAD, significance must
  track the live FDR slider. No Python / `build_payload` change.

## Agreement model (implemented)

Per row, stamped live by `_kxComputeAgreement(rows, fdrGate)`:
- `mouseSig` = any mouse contrast `_fdr[c] < fdrGate`. Mouse summary NES = `peak_NES`.
- `humanSig` = ≥1 AD donor `FDR < fdrGate`; human summary NES = **median of AD-donor NES
  among donors significant at the gate** (significance + sign share one live threshold).

| Category          | Condition                                       |
|-------------------|-------------------------------------------------|
| `concordant-up`   | both sig & both summary NES > 0                 |
| `concordant-down` | both sig & both summary NES < 0                 |
| `discordant`      | both sig & signs differ                         |
| `mouse-only`      | mouseSig & !humanSig                            |
| `human-only`      | humanSig & !mouseSig                            |
| `neither`         | !mouseSig & !humanSig                           |

`_agreeScore` (within-group rank): concordant `+|mNes|·|hNes|`; discordant `−|mNes|·|hNes|`;
mouse/human-only `|nes|·0.1`; neither `0`. Group order is fixed (concordant first).

## Current state — file/function inventory

All in `alz/viewer/template/`:

- **`js/tabs/kinase_crosstable.js`**
  - `_kxState` default `sortKey:"agree_score"`; new `collapsed:Set` (per-category fold).
  - `_kxDefaultCols` + `_kxRenderColsPanel`: `show_agree / show_m_profile / show_h_profile`.
  - `_kxBuildIndexes`: union loop appends human-only rows (`_humanOnly:true`, `peak_NES:null`).
  - `_KX_AGREE_ORDER`, `_KX_AGREE_META` (labels/badge classes/tooltips).
  - `_kxMedian`, `_kxComputeAgreement`, `_kxAgreeCategoryCell`.
  - Glyphs: `_kxMakeNesProfileRow` + `_kxMouseGlyphCell` (reuse `_renderNesProfile` from
    kinase_explorer.js via adapter); `_kxHumanGlyphCell` (local strip — same `.npc` markup
    + color as `_khRenderProfile`, kept local because `_KH`/`_KHState` are null until the
    human tab renders).
  - `_kxBuildHeader` / `_kxBuildRow`: `M-Profile | Agree | H-Profile` inserted after Family;
    rows tagged `data-humanonly` → click selects `kinaseHuman` vs `kinase`.
  - `_kxSortRows`: signed `agree_score` branch. `_kxRenderTable`: bucket → per-group sort →
    collapsible group-header rows; count line shows concordant ↑/↓ + discordant tallies.
- **`body.html`** (crosstable panel ~L373): `#kx-agree-legend` badge strip above table.
- **`styles.css`**: `#kx-table tr.kx-group` header rows (NOT sticky — `<thead>` is already
  sticky at top:0); `.kx-mglyph/.kx-hglyph/.kx-agree-col` column styling.

## Verification (iteration 1)

`pixi run viewer` → fresh `index.html` (payload `generated_at` 2026-06-04T18:59Z); JS
`node --check` clean; all new symbols inlined; logic harness confirmed all six categories,
live-FDR median (excludes sub-gate donors), human-only-no-mouse-data, and FDR-tightening
flips. **Not yet browser-eyeballed** — open the HTML, hard-refresh, confirm a populated
Human-only group appears.

## Iteration 2 — master/detail with a cross-dataset detail panel (SHIPPED 2026-06-04, uncommitted)

**Result:** master is now slim + grouped; detail panel reuses the kinase-tab renderers.
User tweaks folded in: **legend dropped** (verdict header carries it); **selection is LOCAL**
(no `SET_SELECTION` dispatch — `selectedKey` in crosstable state). Cols disclosure panel
removed entirely (`_kxRenderColsPanel`/`_kxDefaultCols` deleted) — the slim master is fixed.
Verified: `node --check` clean on all 3 touched JS; all cross-file renderers resolve to exactly
one def in the bundle; a Node-VM DOM smoke (synthetic shared / mouse-only / human-only kinases)
passed every path — idle prompt, both sub-tabs dispatch to the right renderers, donor primed to
first finite-NES AD donor, mouse-only→human placeholder, human-only→mouse placeholder, verdict
header renders. Browser eyeball still pending (Plotly layout not exercised in the VM).

**Known v1 limitation:** `_khRenderNESAcrossDonors`' donor-bar click re-targets the human tab's
`#kh-detail` and mutates `_KHState.auditDonor` (shared global) — clicking a bar in the crosstable
re-renders the hidden human tab rather than the crosstable's own chart. Harmless but not ideal;
fix later by parameterizing the click callback. Logged in "Later iterations".

### Plan as built

**Problem:** the wide per-sample grid (M-bulk × 9 contrasts, per-donor × ~10 donors, raw+tier
specificity) is invaluable but unreadable. **Goal:** bring the **detail-panel design** from the
per-dataset Kinase tabs into the crosstable — NOT by stuffing the wide rows into a panel, but by
making the crosstable **master/detail** where the detail is a *cross-dataset comparison*. The
per-sample NES relocates from the grid into the detail's NES bar charts (every contrast/donor =
a bar), exactly the kinase-tab pattern.

**Decisions (user-confirmed 2026-06-04):**
- Layout: **master left / detail right + draggable splitter** (reuse `explorer-layout
  kinase-audit-layout` + `.ka-splitter` + `.detail-card`, as `tab-kinase` body.html L22–103).
- Master: **fully slim** — Kinase · Gene · Res · Family · M-glyph · **Agree** · H-glyph ·
  M-spec tier · H-spec tier (~9 narrow cols). Drop all per-contrast / per-donor / raw-spec cols.
- Detail v1: **two sub-tabs**, each a two-column **Mouse | Human** comparison under a verdict header.

**Reuse (verbatim — no greenfield):** every target renderer takes a host-id/body arg and
self-inits its indexes; `_KH`/`_KHState` are populated at boot so human charts work even if that
tab was never opened.
- Activity — Mouse `_renderKinaseNesPlot(hostId, kid)` (kinase_audit.js:479, NES across 9
  contrasts, self-contained, no ctx/click wiring) | Human `_khRenderNESAcrossDonors(hostId, row,
  donor)` (kinase_human.js:786, NES across donors, host-id arg).
- Specificity — Mouse `_renderKinaseCelltypeEvidence(hostId, kid)` (kinase_audit.js:502,
  cell-type evidence AuditTable, self-contained) | Human `_khRenderAttribution(body, row)`
  (kinase_human.js:1274, Levy-T5 cell-type attribution table).

**Single shared-code touch:** `_khRenderAttribution` line 1435 — change
`document.getElementById("kh-attr-donor-select")` → `body.querySelector("#kh-attr-donor-select")`
so the donor-select wiring scopes to its own panel (today it grabs the first match in document
order = the human tab's element). The onChange already re-renders in place into `body`, so this
is the only change; it also hardens the human tab. No other renderer needs edits.

### Changes by file

- **`body.html`** (tab-crosstable, ~L373): wrap the master `.card` + a new
  `<div class="ka-splitter" id="kx-splitter">` + `<section class="detail-card" id="kx-detail">`
  in `<div class="explorer-layout kinase-audit-layout">`. Toolbar + Columns disclosure stay in
  the master card. Drop the now-redundant agreement legend OR move it into the master card header.
- **`js/tabs/kinase_crosstable.js`:**
  - Slim `_kxBuildHeader`/`_kxBuildRow`: remove m_bulk / m_decomp / h_ad / h_ctrl / spec_raw /
    spec_tier column loops; add one **M-spec** tier (`_kxWmbTierBadge` at selected cluster) and
    one **H-spec** tier (`_kxLog2TierBadge`, SEA-AD MTG at selected cluster).
  - Prune `_kxDefaultCols` / `_kxRenderColsPanel` to the agreement-view + spec toggles (the
    per-sample groups are gone — this is the pivot, not a hidden mode).
  - Selection: store `Store.state.view.crosstable.selectedKey` (= `name|residue`, stable across
    both datasets) on row click; render the detail from it. Resolve mouse row from `_KX_ROWS`,
    human row from `_khAllRows().find(id===human.kid)`. (Keep the existing `SET_SELECTION`
    dispatch for cross-tab linkage.)
  - New `_kxRenderDetail()`: idle placeholder when no selection; else verdict header
    (category badge + `Mouse peak {peak_NES} ({nSig}/9 sig) · Human median {hNes}
    ({donorsSig}/{tested} sig)`), sub-tab nav (`detailTab` state, default "activity"), and a
    `.kx-detail-grid` two-column body. Activity → the two NES plots; Specificity → the two
    cell-type tables. Missing side (mouse-only / human-only) → "Not measured in {species}"
    placeholder. Set `_KHState.auditDonor` to the human row's first finite-NES AD donor before
    calling the human renderers.
  - Call `_kxRenderDetail()` from `_kxRenderTable` (after table paint) and on
    selectedKey / detailTab change.
- **`js/02_ui_chrome.js`:** extract the `ka-splitter` drag block (L309–345) into
  `_wireSplitter(splitterId, storageKey, minW, maxW)`; call it for `ka-splitter`, `kh-splitter`,
  and the new `kx-splitter` (consolidation, removes the duplicate in kinase_human.js:1614).
- **`styles.css`:** reuse `explorer-layout`/`kinase-audit-layout`/`detail-card`/`ka-splitter`.
  Add `.kx-detail-grid` (2-col grid) + `.kx-detail-col` header + `.kx-detail-placeholder`.

### Verification (iteration 2)

`pixi run viewer`; open HTML, hard-refresh, Crosstable tab. Check: master is slim + grouped;
selecting a concordant kinase shows the verdict header + both NES plots side by side; mouse-only
→ human side placeholder, human-only → mouse side placeholder; Specificity sub-tab shows mouse
cell-type evidence + human attribution tables; splitter drags + persists; FDR slider still
regroups the master and refreshes the detail. Confirm the human attribution donor-select inside
the crosstable changes only the crosstable panel (the L1435 scoping fix).

## Later iterations (backlog)

- [ ] Detail sub-tab 3: **Substrate/Score** (running-enrichment + scorecard per side) — needs the
      async audit-context loader (`_loadKinaseAuditContext`) for `ctx.meaRaw` etc.
- [ ] Fix the human NES-bar click in the crosstable detail (currently re-targets `#kh-detail` +
      mutates `_KHState.auditDonor`) — parameterize `_khRenderNESAcrossDonors`' click callback so
      it stays within the crosstable panel.
- [ ] (pending user spec)

## How to resume cold

1. Read this doc + the memory `project_crosstable_agreement_view`.
2. `git status` — iteration 1 may be committed by then; if dirty, the 3 files above are it.
3. Rebuild + eyeball: `pixi run viewer`, open `outputs/reports/unified_viewer/index.html`,
   hard-refresh, Crosstable tab (visible in both Mouse and Human modes).
