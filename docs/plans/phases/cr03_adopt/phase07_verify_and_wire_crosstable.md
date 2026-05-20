# Phase 7 — Verify + wire Crosstable tab into TAB_MANIFEST

**Status:** todo
**Depends on:** Phase 6
**Reversible:** the wiring change is a small fixup commit; verification is read-only.

## Goal

End-to-end smoke of the reconciled viewer. Wire the Crosstable tab into the runtime manifest — the CR-03 branch added `kinase_crosstable.js` and registered it in `body.html` / `index.html.j2`, but did not add the runtime `TAB_MANIFEST` entry, so the tab is currently invisible.

## Preflight

```bash
git rev-parse HEAD                          # = Phase 6 HEAD
test -f alz/viewer/template/js/tabs/kinase_crosstable.js \
  && echo "crosstable JS present" \
  || echo "MISSING — branch state is unexpected"
```

Locate the manifest:

```bash
grep -rn 'TAB_MANIFEST' alz/viewer/template/
# Expect a definition in one of the boot/state/wiring JS files.
```

Inspect the existing manifest entries to learn the schema (label, id, module reference, ordering, optional gating predicate).

Inspect `kinase_crosstable.js` for the PAYLOAD fields it accesses:

```bash
grep -nE '\bPAYLOAD\.[A-Za-z_.]+' alz/viewer/template/js/tabs/kinase_crosstable.js | \
  sort -u
```

Cross-reference each field against the current PAYLOAD (open the rebuilt index.html in browser DevTools and check `PAYLOAD` keys). If any field is missing post-reconcile, document and either extend the builder or scope the Crosstable to populated columns. Do not register a tab that will throw on load.

## Steps

### 7.1 — Wire Crosstable into TAB_MANIFEST

Add the manifest entry matching the schema you observed in preflight. Place it ordering-wise near the other Kinase tabs (Explorer, Human, Audit).

```bash
git add <manifest file>
git commit -m "feat(viewer): wire Crosstable tab into TAB_MANIFEST

Branch a7d104d..175c85b added kinase_crosstable.js and the body.html slot
but never registered the tab in the runtime manifest, so it was invisible.
Registers it alongside the other Kinase tabs."
```

### 7.2 — Full build

```bash
pixi run python alz/build_unified_viewer.py 2>&1 | tee /tmp/cr03-final-build.log | tail -40
```

Watch RSS in a sidecar terminal. Expect peak under 4 GB given the env-controlled DuckDB budget. If it spikes higher, abort and capture the trace point — the Phase 0 pagination commit was supposed to remove the major memory hazards.

### 7.3 — Browser smoke

Hard-refresh (Ctrl+Shift+R). For each tab, confirm it loads without console errors and renders meaningful content:

- [ ] Kinase Explorer — Family column visible
- [ ] Kinase Human — Family column visible; Cell-type specificity sub-tab populated from `celltype_specificity.csv`
- [ ] **Crosstable** (new) — table loads, sort works, column-visibility groups toggle
- [ ] Kinase Audit
- [ ] Incytr Pathways — per-shard pagination active (pager controls visible), trajectory chips render, Recur-in filter works, temporal-detail bar chart renders, Measurement Trace panel opens on row expansion

Spot-check in DevTools console:

```js
PAYLOAD.meta.generated_at        // matches the build timestamp from /tmp/cr03-final-build.log
PAYLOAD.human.celltype_specificity  // non-null object
PAYLOAD.kinase[0].family            // present
```

### 7.4 — Build artifacts on disk

```bash
ls -lh outputs/reports/unified_viewer/index.html
ls outputs/reports/unified_viewer/edge_slices/
ls outputs/reports/unified_viewer/audit_sources/transcript_trace/ | head
```

All three should exist. `index.html` should be roughly the size of the previous build (70–80 MB).

## Failure handling

If any tab throws on load:
1. Capture the console error.
2. If it's a missing PAYLOAD field that you missed in preflight, decide between extending the builder (small) or scoping the tab to omit the missing data. Document either way.
3. If it's a JS syntax error from a bad merge, identify the file:line, fix, commit, rebuild.

If RSS spikes during build, **do not** apply a row cap. Diagnose the actual hot path (`tracemalloc` snapshot, or grep for the obvious culprits: `fetchdf` over large tables, `iterrows`, full-file `read_csv`). The viewer audit lists the known hazards.

## What the next phase needs from you

In your Implementation Log entry:
- The crosstable-wiring commit SHA.
- The full-build peak RSS.
- A tick-list of which tabs passed smoke and any open issues filed for follow-up.
- An explicit go/no-go for Phase 8 push.
