# Unified Viewer — Filter & Tab Redesign Plan

Status: design draft awaiting approval. Implementation has not started.
Scope: `code/build_unified_viewer.py` filter bar, `TAB_MANIFEST`, the
Sender × Receiver tab, the Graph tab, and removal of the Evidence Audit tab.

## Problem

The current viewer carries a six-control global filter bar (`contrast`,
`direction`, `receiver`, `support`, `fdr`, `score`) regardless of which
tab is active. Most tabs consume only a slice of that bar, so the rest
is rendered dimmed with a tooltip. On Overview, Results, Methods, and
Audit the entire bar is dimmed. On Signal Map, Sender × Receiver, and
Kinase, four of six controls are dimmed. The bar's purpose — "set scope
once, see it everywhere" — is mostly illusory.

Two further failure modes:

1. **Word collisions.** Global `direction` (drop up or down rows) and
   the local Sender × Receiver `Mode` toggle (Count vs Direction split
   of the same data) argue over the same word with different semantics.
   Global `contrast` is a single-value picker; Pathway's local `pe-cset`
   chip set is a richer "any/all/exact match" picker that does the same
   conceptual job differently.
2. **Single-contrast bias.** Two tabs require `contrast ≠ ALL` to
   render. Single-contrast snapshots are rarely the most informative
   biological read; trajectories across the 3×3 (3 genotypes × 3
   timepoints) are. The header filter pushes users toward the less
   informative view.

## Principles

The redesign rests on three rules:

- **A filter is global only if ≥3 tabs consume it with the same
  meaning.** Anything below that bar moves to the tab that owns it.
- **View-mode toggles never live in the global bar.** A control that
  changes how a figure is drawn (Count vs Direction, Kinase vs
  Backbone, Concentric vs Force) belongs next to the figure.
- **Slice the 3×3, do not summarize it.** Trajectory-aware tabs show
  three contrasts of real data side-by-side along one axis, never an
  average across an axis.

## Global filter bar — three controls

After the redesign the always-visible bar contains:

| Filter | Kind | Consumed by |
|---|---|---|
| `receiver` | scope | Temporal, Additivity, Pathway, Graph |
| `support` (pathway evidence tier) | scope | Temporal, Additivity, Pathway, Graph |
| `fdr` | threshold | Kinase, Temporal, Additivity, Pathway |

All three are consumed by ≥3 tabs with consistent meaning. Static and
trajectory tabs (Overview, Results, Methods, Signal Map, Sender ×
Receiver) will not dim them — they simply do not apply, and the bar
hides controls that no current tab consumes rather than greying them.

Removed from the global bar:

- `contrast` — eliminated. Two tabs that previously required a single
  contrast (Sender × Receiver, Graph) are redesigned to slice the 3×3
  trajectory-first; the rest either ignored contrast already or carry
  their own richer picker (Pathway's `pe-cset`).
- `direction` — eliminated. As scope it was used only on Pathway and
  Graph; both can express signed reads through their local controls or
  through edge styling. As a "split-by-sign" view-mode it lives on
  Sender × Receiver as a local toggle.
- `score` — eliminated as global. Different tabs use different score
  scales (TPDS, observed score, NES); a single numeric box meant
  different things on each tab. Each tab gets a local threshold in its
  own units.

## Hard data-load rule — passing both nulls only

The viewer payload only carries backbones × contrasts that passed both
permutation tests (kinase enrichment + receiver-specific structure).
Failed-null rows are filtered upstream during payload construction.

Consequences:

- No "passing both nulls" toggle anywhere in the UI.
- The Evidence Audit tab is removed (its purpose was to expose pass /
  fail status across contrasts; with failed contrasts absent from the
  payload it has nothing to audit).
- The `requires` gate on Audit and the breadcrumb chip for backbone
  selection driven by Audit go away.

## Per-tab control inventory after the redesign

| Tab | Global filters consumed | Local controls |
|---|---|---|
| Overview | — | — |
| Results | — | — |
| Signal Map | — | — |
| Sender × Receiver | — | axis toggle, anchor picker, mode (count/direction split) |
| Temporal | fdr, support, receiver | level, metric, tissue, local \|TPDS\| threshold |
| Additivity | fdr, support, receiver | level, timepoint scope, local score threshold |
| Kinase | fdr | search, composite-score preset + sliders |
| Pathway | fdr, support, receiver | search, `pe-cset` chips + match mode, local \|TPDS\| threshold |
| Graph | support, receiver | genotype picker, timepoint slider, min-degree, local \|TPDS\| threshold, top-N cap |
| Methods | — | — |

## Sender × Receiver — collapse-axis design

Default view: three 22×22 matrices side-by-side, each a real contrast,
no averaging. Color scale pinned across all 9 contrasts (already
implemented this session).

Local controls:

- **Axis toggle**: "Compare genotypes" | "Compare timepoints"
- **Anchor picker**: when axis = compare-genotypes, anchor is a
  timepoint (2/4/6mo) and the three matrices are App, Tau, ApTt at
  that timepoint. When axis = compare-timepoints, anchor is a genotype
  (App / Tau / ApTt) and the three matrices are 2mo, 4mo, 6mo for that
  genotype.
- **Mode**: Count (log10 1+n) | Direction (n_up − n_down). Existing
  `sm-mode` toggle, retained.

Keyboard:

- `←` / `→` — step the anchor
- `↑` / `↓` — flip the axis (and reset anchor sensibly)

Default: axis = compare-timepoints, anchor = ApTt. The double-genotype
trajectory across time is the most biologically loaded read.

Subtitle reads e.g. "App, Tau, ApTt at 2mo" or "App at 2mo, 4mo, 6mo".
Keyboard hint chip stays visible.

## Graph — timepoint-slider design

Default view: single graph showing one (genotype, timepoint) snapshot.
Pure snapshot rendering — no persistence overlay, no compositing — for
performance.

Local controls:

- **Genotype picker**: App | Tau | ApTt (required).
- **Timepoint slider**: 2mo / 4mo / 6mo, with optional play button to
  auto-advance.
- **Min-degree**: existing 1/2/5/10/20/50 selector, retained.
- **\|TPDS\| ≥ X threshold**: local slider. Default at the 50th–75th
  percentile of \|TPDS\| within the current snapshot — most edges
  hidden by default, drag down to reveal weak edges.
- **Top-N edges cap**: optional safety net (e.g. "show top 200 by
  \|TPDS\|"). Off by default; flip on if a snapshot stalls.

Default: App, 2mo. Earliest disease, simplest first comparison.

### Performance plan

- Precompute Cytoscape elements for the three timepoints of the active
  genotype on genotype change. Slider moves are visibility flips, not
  rebuilds.
- Edge fade transitions on visibility change are cheap; rebuild costs
  are not. Avoid the rebuild path on slider movement.
- The hard-rule (passing both nulls only) prunes the universe of edges
  upstream — typical snapshots should land in the low hundreds, not
  thousands.

## Local thresholds replacing the global `score`

Each tab that previously consumed global `score` gets a tab-local
threshold in its native units:

- Temporal: \|TPDS\| ≥ X (when metric is mean \|TPDS\|) or mean score ≥ X
- Additivity: observed score ≥ X
- Pathway: \|TPDS\| ≥ X
- Graph: \|TPDS\| ≥ X (covered above)

Defaults: zero (show everything within the passing-both-nulls universe).
Each tab's local threshold persists across tab switches via the existing
`view` state slice.

## State / URL shape changes

The `Store.state.filters` slice loses `contrast`, `direction`, `score`,
`graphNodeIds`. It keeps `receiver`, `support`, `fdr`, `sender`.

The `Store.state.view` slice gains:

- `senderMatrixAxis` ("genotype" | "timepoint")
- `senderMatrixAnchor` (one of the 3 anchors for the active axis)
- `graphGenotype` ("App" | "Tau" | "ApTt")
- `graphTimepoint` ("2mo" | "4mo" | "6mo")
- per-tab score thresholds: `temporalScoreMin`, `additivityScoreMin`,
  `pathwayScoreMin`, `graphScoreMin`

The URL serializer / deserializer is updated to match. Old URLs with
`?c=App_2mo&d=up` are not back-compat — the filters they reference no
longer exist. Acceptable: the viewer is internal.

`TAB_MANIFEST` is updated: every entry's `filters` array references
only `receiver`, `support`, or `fdr`. The `requires` gates on
Sender × Receiver and Graph are removed (both tabs render at default
without a single-contrast pick). Audit is removed entirely.

## Migration order

The renderer can support both filter shapes during migration via a
feature flag, but the data substrate change (passing-both-nulls only)
and the Audit tab removal are coupled and should ship together.

Proposed order:

1. **Payload** — add the passing-both-nulls upstream filter to
   `build_unified_viewer.py` payload assembly. Verify with grep on the
   emitted JSON that no failed-null rows remain.
2. **Audit tab removal** — delete `tab-audit` markup, `TAB_MANIFEST.audit`,
   `renderAudit`, the breadcrumb path, and the Pathway → Audit nav link.
3. **Global bar shrink** — remove `contrast`, `direction`, `score`
   markup, state slices, URL keys, and `FILTER_REASONS` rows. Adjust
   `syncFilterBarToTab` to hide rather than dim non-applicable filters.
4. **Sender × Receiver redesign** — rewrite the tab around axis +
   anchor controls. Update `TAB_GUIDE.senders` drawer copy to match.
   Remove the `requires` gate.
5. **Graph redesign** — add genotype picker + timepoint slider, swap
   precompute strategy, add the local \|TPDS\| and top-N controls.
   Update `TAB_GUIDE.graph`. Remove the `requires` gate.
6. **Per-tab local thresholds** — add the tab-local score / \|TPDS\|
   inputs to Temporal, Additivity, Pathway. Update their drawers.
7. **Drawer rewrites** — every drawer that referenced the removed
   global filters needs a copy pass. Continue the six-section,
   mechanism-grounded rewrites started this session, tab by tab with
   review before edit.

## Rollback cost per step

- Payload + Audit removal: cheap to revert via git.
- Global bar shrink: medium — touches every tab's render path that
  reads from `Store.state.filters`. Reverting requires restoring state
  shape and every consumer.
- Sender × Receiver / Graph redesigns: contained to those tab render
  functions and their toolbars. Drawer copy is data, easy to revert.
- Local thresholds: contained per tab.

## Resolved design decisions

These were open questions on the first draft; resolved by user review.

**Pathway tab — trajectory-shorthand picker.** Replace the 9-chip
`pe-cset` with named trajectory buttons. Six buttons, one click each:

- *App trajectory* — backbones passing in any of App_2mo / App_4mo / App_6mo
- *Tau trajectory* — backbones passing in any of Tau_2mo / Tau_4mo / Tau_6mo
- *ApTt trajectory* — backbones passing in any of ApTt_2mo / ApTt_4mo / ApTt_6mo
- *2mo cross-section* — backbones passing in any of App_2mo / Tau_2mo / ApTt_2mo
- *4mo cross-section* — backbones passing in any of App_4mo / Tau_4mo / ApTt_4mo
- *6mo cross-section* — backbones passing in any of App_6mo / Tau_6mo / ApTt_6mo

Plus an "All contrasts" default. Each button is a single click; the
match mode is implicitly *any*. Loses the "exactly these and no others"
power-user mode — acceptable, that mode was niche and the trajectory
framing is consistent with the rest of the redesign.

**Sender × Receiver axis flip — last-used-on-axis.** Track each axis's
last anchor independently in `view` state:
`senderMatrixLastAnchorByAxis = { genotype: "2mo", timepoint: "ApTt" }`.
On `↑` / `↓` axis flip, set the anchor to the last value the user used
on the destination axis (or its default if the user has not visited
that axis yet: 2mo for compare-genotypes, ApTt for compare-timepoints).
Costs four lines of state; matches navigation expectation.

**Graph top-N cap — separate control, default off.** The Graph toolbar
has both a `|TPDS| ≥ X` slider and a `max edges` numeric input. The cap
defaults to blank (no cap); users who hit a slow snapshot flip it on.
Keeping them separate avoids conflating "what's the threshold" with
"what's the rendering budget."
