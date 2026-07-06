# "Related pathways" drawer — simplify to a read-only data panel

Status: implemented
Scope: `alz/viewer_shared/template/js/tabs/incytr_pathways.js`,
`alz/viewer_shared/template/js/tabs/incytr_global_index.js`,
`alz/viewer_shared/template/js/04_slice_cache.js`,
the two per-cohort `body.html` + `styles.css` (Song + T-cell), and the
backbone-spine-index build in `alz/viewer/cohorts/song.py` / `fivexfad.py` + its verifier.
Build: `pixi run python alz/build_unified_viewer.py --html` + hard refresh.

Supersedes `expands_to_drawer_redesign.md` (that doc proposed *relabelling* the navigation
buttons; this removes them).

## Goal

The drawer tab (id `expands-to`, label "Related pathways") becomes **read-only**. It shows two
facts about the selected row and performs **no navigation** — no grain re-scope, no search-box
seeding, no Back affordance:

1. **# full pathways** collapsed into this backbone (backbone grains only) — the row's `n_paths`.
2. **Related cell-type pairs** — every `(sender, receiver)` pair carrying this row's path/spine,
   grouped by receiver. Shown at **every grain**, Full included.

## Data sources (both already shipped — no new build artifact)

- `n_paths` is already a materialized column on every backbone-grain row
  (`song.py:874`, index manifest `:1042`). Read `r.n_paths` directly. (Full rows have no
  `n_paths` column — the count is meaningful only where rows aggregate multiple Targets, so
  omit it at Full grain.)
- "Pairs carrying this path" comes from the **global index**, which is built per grain — Full and
  every backbone grain (`song.py:1951` / `:1205`) — and already covers *every* pathway across all
  pairs (`incytr_global_index.js:4`). `pathRows(ident)` (`:474`) already scans it; the only reason
  it returns one pair is the `sid/rid` equality at `:506`.

## Changes

### 1. New `IncytrGlobalIndex.pairsForPath(ident)` — `incytr_global_index.js`

Add a sibling to `pathRows`: identical node resolution (present nodes matched, `"—"` sentinels for
nodes absent at this grain — `:498-499`), but **drop the sender/receiver filter** and, instead of
materializing rows, collect the distinct `(sender, receiver)` set:

```js
function pairsForPath(ident) {
  // resolve recWant/emWant/ligWant/tgtWant exactly as pathRows
  const seen = new Set(); const out = [];
  for (let i = 0; i < d.nrows; i++) {
    if (rec[i] !== recWant || em[i] !== emWant) continue;
    if (ligWant >= 0 && lig && lig[i] !== ligWant) continue;
    if (tgtWant >= 0 && tgt && tgt[i] !== tgtWant) continue;
    const k = sid[i] * gi.receiver_vocab.length + rid[i];
    if (seen.has(k)) continue;
    seen.add(k);
    out.push([gi.sender_vocab[sid[i]], gi.receiver_vocab[rid[i]]]);
  }
  return out;   // cache like pathRows
}
```

Export it (`:515`). Because absent nodes match-any, a backbone row matches by its surviving nodes
(spine semantics) and a Full row matches by the exact 4-tuple — one function covers both grains.

### 2. Rewrite `_ipApplyExpandsToWidenPanel` → source from the global index — `incytr_pathways.js:2084`

Replace the `SliceCache.loadBackboneSpineIndex(grain)` body with:
`await IncytrGlobalIndex.ensureLoaded(); const pairs = IncytrGlobalIndex.pairsForPath(ident);`
then the existing group-by-receiver render verbatim. `ensureLoaded()` fetches the per-grain binary
regardless of mode (the manifest is present in pair mode too — the "top mode only" contract governs
the *table* render path, not whether the binary can be loaded). **Verify this in pair mode** (see
Verification). Rename the function to `_ipRenderRelatedPairs`.

### 3. Rewrite `_ipRenderExpandsToPanel` — `incytr_pathways.js:2007`

Delete **Section A entirely** (`:2018-2054`: `canExpand`/`canCollapse`/`canBack`, every
`data-ip-drill` button). The panel becomes:

- Backbone grain → one line `<N> full pathways collapse into this backbone` from `r.n_paths`.
- All grains → "Related cell-type pairs" rendered **directly** (no checkbox/toggle): a host div the
  async `_ipRenderRelatedPairs` fills. Drop the `widen` gating and the `hasSpineIndex` branch.

### 4. Delete the drill machinery (now unreachable) — `incytr_pathways.js`

- `_ipNavigateDrill` (`:1969`), `_ipDrillBack` (`:1984`), `_IP_GRAIN_COARSER` (`:1951`),
  `_ipCoarserGrains` (`:1957`).
- `_ipRuntime.drillReturn` (`:145`) and `_ipRuntime.widenKeys` (`:144`) + every read/write
  (`:481-485`, `:1052`, `:1074-1077`, `:1113-1140`, `:1762`, `:1786-1789`, `:1819-1853`, `:2638`).
- The `data-ip-drill` button click handlers in **both** table click blocks (`:1082-1110` and
  `:1794-1822`).
- The grain-selector Back button: `#ip-drill-back` markup (`body.html:316` Song, `:169` T-cell),
  its listener (`:2648-2651`), and its show/hide block (`:478-485`).

### 5. Retire the backbone-spine-index sidecar (superseded by the global-index scan)

- `04_slice_cache.js`: delete `loadBackboneSpineIndex` (`:229`) and its export (`:260`).
- `song.py` / `fivexfad.py`: delete the `backbone_spine_index` build block (`song.py:1123-1172`),
  the `backbone_spine_index` key on the grain payload (`:1239`), `_BACKBONE_SPINE_INDEX_FILENAME`
  (`incytr_index.py:48`), and `BACKBONE_GRAIN_SPINE_EXPR` if no other reader remains.
- Delete `alz/viewer/verify_backbone_spine_index.py` and any `pixi` task referencing it.
- `_ipSpineKey` (`:1933`): keep only if still used after the rewrite; otherwise delete.

### 6. Stylesheets

Remove `.ip-drill-btn`, `.ip-drill-btn--primary`, `.ip-drill-btn code` from
`alz/viewer/template/styles.css` (`:726-732`) and `alz/tcell_viewer/template/styles.css`
(`:667-673`). Keep `.ip-related*` classes (still used by the read-only panel).

## Out of scope

- No change to grain mechanics or the grain selector itself — switching detail level stays in the
  selector dropdown, which never used the drill path.
- No new payload column — `n_paths` and the global index already exist.

## Verification

- `node --check` on the two edited JS files.
- `pixi run python alz/build_unified_viewer.py --html`, hard refresh, walk:
  1. Backbone row (e.g. L-R-EM) → panel shows `n_paths` count + the related-pairs table; no buttons.
  2. Full row → panel shows related-pairs only (no count line); confirm pairs enumerate.
  3. **Pair mode**: open the drawer on a row and confirm related-pairs still populate (global index
     fetched on demand) — this is the one behavior the old sidecar provided that must not regress.
  4. Grep the built `index.html` for `data-ip-drill` / `ip-drill-back` → zero hits.
- Confirm Song and T-cell viewers render identically (shared JS; per-cohort CSS only).
