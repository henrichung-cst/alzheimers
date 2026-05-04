# Unified Viewer — Refined-Minimal Review Pass

Status: proposed.
Scope: edits confined to `code/build_unified_viewer.py` (embedded JS / HTML / CSS). No payload schema changes, no Python data-loading changes, no new tabs.

The prior IA refactor (tab grouping, adaptive filter bar, prerequisite cards, breadcrumbs, URL hash state) is **already implemented** — `TAB_MANIFEST`, `FILTER_REASONS`, `syncFilterBarToTab`, `syncBreadcrumb`, `renderUnmetPrerequisite`, and the breadcrumb nav are all in place. This plan only addresses what remains.

## Goal

Resolve the central pain point: **interpretation guidance is scattered across five surfaces** — native `title=` tooltips, `metric-help` ⓘ icons, inline `.callout` blocks, muted helper paragraphs, and empty-state copy. They drift in tone and force the reader to scan multiple zones to understand a single number. Unify them into:

1. A canonical metric glossary (data layer).
2. A collapsible per-tab "How to read" drawer (long-form home).
3. Tooltips become short labels only, all sourced from the glossary.

## Non-goals

- No table or chart redesign — they work.
- No color or palette changes.
- No new tabs / removed tabs / renamed tabs.
- No framework migration — stays a single self-contained HTML.
- No revisiting the IA refactor that already shipped.

## Approach

### 1. Canonical glossary (`METRIC_DEFS`)

One JS object placed alongside `TAB_MANIFEST` (~line 1704), keyed by metric ID:

```js
const METRIC_DEFS = {
  support:   { label: "Support",
               short: "How much signal a kinase pushes into this pathway. Bigger = stronger driver.",
               long:  "Sum of |NES| × IDF × pair-attribution across edges.",
               howToRead: "Use this to rank top driver candidates." },
  direction: { label: "Direction",
               short: "Signed Support: + = more active in disease, − = less, ~0 = mixed.",
               long:  "Σ(concordance × |support|) across edges.",
               howToRead: "High Support + strong sign = clean driver. Near-zero = weaker candidate." },
  tpds:      { label: "TPDS", short: "Pathway directional score per contrast.", … },
  // … contrast, direction-filter, fdr, score, support-filter, receiver,
  //   confidence-tier, concordance, backbone, trend, etc.
};
```

Consumed by:
- `metric-help` ⓘ tooltips (`title` → `short`).
- Column headers (`<th title>` → `short`).
- The drawer (renders `long` + `howToRead`).
- Optionally a Glossary section in the Methods tab — same data, free win.

This kills tooltip drift: edits land in one object.

### 2. Per-tab "How to read" drawer

Right-aligned collapsible panel. Toggle button in the top-right of the content area; state persisted per tab in `localStorage`. Default collapsed. Width when open: ~320px, pushes content (no overlay).

Content is a small registry, not free-form HTML:

```js
const TAB_GUIDE = {
  pathway: {
    purpose: "Find pathways that pass both significance nulls and inspect their driving kinases.",
    primary: "Driving Kinases table — Support ranks; Direction signs the rank.",
    cues: [
      { metric: "support",   when: "Top of the list" },
      { metric: "direction", when: "Sign disagrees with bulk genotype direction" },
    ],
    pitfalls: [ "Trend column counts evidence, not magnitude — use Direction for magnitude." ],
  },
  kinase: { … },
  temporal: { … },
  graph: { … },
  // …
};
```

Drawer template: **What this tab answers** → **How to read the primary view** → metric cues (each pulls `howToRead` from `METRIC_DEFS`) → **Common pitfalls**. One template, populated from data.

Replaces every existing inline `.callout "How to read this"` block.

### 3. Tooltip pass

Every existing `title=` on a header, ⓘ icon, or filter label is rewritten to consume `METRIC_DEFS[key].short`. Inline-written long sentences are removed from the markup. Tooltips become labels, not paragraphs.

### 4. Typography pass (light)

- One display face + one body face (selected during implementation, with system fallbacks). `<link>` with `font-display: swap`.
- Tighten the type scale to a 1.2 modular scale; reduce the count of distinct sizes.
- Harmonize spacing tokens (`--space-1..6`) — current CSS has ad-hoc `margin: 12px 8px` patterns.

No color, no shadows, no new components.

### 5. Cleanup pass

- Remove inline `.callout "Reading a row"` blocks superseded by the drawer.
- Remove ad-hoc muted helper paragraphs that duplicate drawer content.
- Audit `metric-help` icons: keep on column headers and section h4s; remove decoratively duplicated ones.

## Critical files

`code/build_unified_viewer.py` — only file edited. Sections touched (line numbers approximate):

- **CSS** (~880–940): drawer styles, typography variables, spacing tokens.
- **HTML template** (~1003–1100): drawer toggle button + drawer container.
- **JS data layer** (~1704–1790): `METRIC_DEFS`, `TAB_GUIDE` added next to existing `TAB_MANIFEST` / `FILTER_REASONS`.
- **JS routing** (~4370–4440): `syncDrawer(activeTab)` added to the existing Store subscriber alongside `syncFilterBarToTab` / `syncBreadcrumb`.
- **Render functions** (~2937, 3340, 3790, 3867, …): rewrite tooltips to read from `METRIC_DEFS`; remove inline callouts now covered by the drawer.

No edits to lines 1–1000 (data loading / payload build).

## Verification

1. **Build:** `pixi run viewer`. Confirm `METRIC_DEFS`, `TAB_GUIDE` present in the embedded `<script>`; no JS console errors.
2. **Glossary parity:** every column header `title=` and every ⓘ tooltip resolves to a `METRIC_DEFS[key].short` — no orphan strings. Grep emitted JS for any remaining hard-coded long-form tooltip text.
3. **Drawer:** open on each tab → content matches `TAB_GUIDE[tab]`. Toggle persists across reloads. Default collapsed.
4. **Regression diff:** Pathway / Graph / Temporal Backbone / Additivity Backbone tables and plots byte-identical (no logic changes to those render fns). Spot-check 2–3 contrasts.
5. **Tone audit:** read each tab's drawer cold — does someone unfamiliar with the pipeline get the gist? If a sentence reaches for jargon, it stays in `long`, not `howToRead`.

## Sequencing

1. Land `METRIC_DEFS` + rewrite all existing tooltips to consume it. Cosmetic-only, low risk.
2. Land drawer + `TAB_GUIDE`; remove the now-redundant inline callouts.
3. Typography + spacing-token pass.
4. Cleanup pass: remove decorative duplicate ⓘ icons, dead helper paragraphs.

Each step is independently shippable; we can stop after any of them if it turns out to be enough.

## Out of scope

- Anything in the data-loading / payload build.
- New / removed / renamed tabs.
- Chart or table redesign.
- Color, palette, shadow, ornament changes.
- Migration off self-contained HTML.
- The shipped IA refactor (`TAB_MANIFEST`, filter-bar adaptivity, prerequisites, breadcrumbs, URL state).
