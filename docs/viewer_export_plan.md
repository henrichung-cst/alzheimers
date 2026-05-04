# Unified Viewer — Per-Tab Export Plan

Lets the collaborator copy the current view into an AI chatbot for follow-up
analysis. Scope: the four interactive analysis tabs (`temporal`, `additivity`,
`kinase`, `pathway`) plus `senders` and `graph`. Excluded: `overview`,
`results`, `signal`, `methods`.

## Contract

"Export = exactly the view on screen." The exported rows are the rows the
active filter predicate admits, in the current sort order, honoring any
built-in display cap the tab already enforces (e.g., kinase tab's top-200).
No second cap layer. No raw-data fallback in the bundle — long analyses keep
using the underlying CSVs in `outputs/reports/`.

## Bundle structure (same shape for every tab)

Output is a single Markdown document, copy-to-clipboard primary, download as
`.md` secondary. Three sections in fixed order:

1. **Context header** — tab name, every active filter, every active
   selection chip, and denominators that anchor the visible slice.
   Format: a labeled key/value list, then a one-line denominator sentence.

   ```
   Tab: Pathway
   Receiver: Astro_NT
   Support: pathway-confirmed
   FDR: < 0.25
   |TPDS|: ≥ 0.30
   Trajectory: ApTt-peaking
   Selection: backbone BB_4129
   ───
   Showing 142 of 25,839 ApTt-only chains (out of 55,859 total tested);
   142 pass |TPDS| ≥ 0.30 at FDR < 0.25 with the chosen receiver.
   ```

2. **Methods preamble** — pulled verbatim from the tab's `TAB_GUIDE`
   `method` paragraphs plus `shows.lead`. Already mechanism-grounded
   from the Step 7 drawer rewrite, which is exactly the framing an LLM
   needs to interpret the table without re-deriving the assay.

3. **Visible table** — Markdown fenced table of the rows on screen, in
   sort order, with the columns the tab actually shows. Numeric columns
   keep the same precision as the rendered cells.

## Architecture

**Single-source predicate.** Each tab already has a function that filters
the payload into its render slice. The export adapter must call that same
function, not re-implement filtering. This is the only invariant — drift
between screen and export is the failure mode we are designing against.

**Per-tab adapter** — small object with three fields:

```js
{
  collect: () => ({ filters, denominators, rows }),  // uses the same predicate as render
  columns: [ { key, label, format } ],
  methodsKey: "pathway",                              // index into TAB_GUIDE
}
```

The shared `exportTab(tab)` helper runs `collect`, formats the bundle,
and hands it to the clipboard / download UI. Tab-specific knowledge stays
in the adapter; bundle assembly stays shared.

**UI placement.** A small "Export view" button in each tab's local
toolbar, next to the tab's existing controls (not in the global filter
bar). Tooltip: "Copy this view as Markdown for an AI chatbot."

## Per-tab specifics

### temporal
- **Filters captured**: mode (kinase|backbone), |TPDS| ≥, FDR, receiver.
- **Denominators**: passing rows / total rows tested in the active mode,
  per genotype × timepoint cell shown on screen.
- **Columns**: entity (kinase or backbone id), genotype, timepoint, NES
  or TPDS, FDR, passing-flag.

### additivity
- **Filters captured**: mode, score min, FDR, receiver.
- **Denominators**: chains plotted / total chains at this timepoint;
  summary of the additivity ratio bin (sub / additive / super) counts.
- **Columns**: chain id, App score, Tau score, ApTt score, predicted
  (App+Tau), ratio, timepoint.

### kinase
- **Filters captured**: FDR; selection (receiver, support if set).
- **Denominators**: rows shown / 240 kinases, plus how many pass FDR
  alone and how many additionally pass any selection chip.
- **Columns**: kinase, gene symbol, per-contrast NES, per-contrast FDR,
  trajectory label, top supporting cell types, backbone count.

### pathway
- **Filters captured**: trajectory button, |TPDS| ≥, receiver, support, FDR.
- **Denominators**: rows shown / chains in the trajectory bucket / total
  chains tested (55,859); contrasts the selection passes in.
- **Columns**: chain id, sender, receiver, ligand→…→target nodes, TPDS
  per genotype × timepoint, passing-contrast count, top driving kinases.

### senders
- **Filters captured**: receiver, support.
- **Denominators**: sender rows shown / 22 senders; chain counts per
  sender for the active receiver.
- **Columns**: sender, n chains, n significant chains, top receivers,
  top driving kinases.

### graph
- **Filters captured**: receiver, support, current node selection set.
- **Denominators**: nodes drawn / total backbones for this receiver;
  edges drawn / total edges in slice.
- **Columns**: node id (backbone), degree, n contrasts passing, top
  associated kinases. (Edges export as a second small table: source,
  target, weight.)

## Open questions

- **Ordering of duplicate filters** — temporal and pathway both capture
  FDR + |TPDS|; do we want a canonical key order in the header so a
  diffed export is stable? Default: yes, alphabetical by filter key.
- **JSON variant** — not in scope unless the collaborator's chatbot
  workflow demands it. Markdown first.

## Implementation order

1. Shared `exportTab(tab)` helper + clipboard/download UI primitive.
2. `kinase` adapter (cleanest: single table, simple filters).
3. `temporal` adapter.
4. `pathway` adapter (largest table, validates the cap-by-render-state rule).
5. `additivity`, `senders`, `graph` adapters.
6. Visual QA in browser — confirm header denominators match what's drawn.
