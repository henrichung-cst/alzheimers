# kinase_sidechain_incytr_graph — subplan 04: cytoscape sidechain panel

**Goal:** Render **one pathway spine at a time** (L→R→EM→T, always drawn and central) decorated with
its kinase sidechains and extended kinase→kinase chains, as a new **detail sub-tab** in the Incytr
pathways table — driven by the row the user already expanded. Wired for song / 5xFAD / t-cells.

**Scope**
- In: A `Sidechains` sub-tab in the pathways row detail panel, beside `Evidence` / `Scores` /
  `Related pathways`. On open it reads subplan 03's `incytr_sidechains` slice, pulls the row's 4 node
  genes' terminal kinases for the row's contrast, walks the interactome to draw sidechains + extended
  chains, scales edge thickness with the evidence continuum, and distinguishes `both`-provenance edges.
  Degrades gracefully when the slice is absent.
- Out: No backend or payload changes (01/02/03). No new top-level tab and **no `TAB_MANIFEST` edits**
  (see "Navigation" below). No new picker/menu — navigation is the existing row expander. No
  all-pathways view, seed, or top-N. **No progressive click-to-expand** (deferred). No new CDN
  dependency.

**Files touched:** `alz/viewer_shared/template/js/tabs/incytr_sidechains.js` (new),
`alz/viewer_shared/template/js/tabs/incytr_pathways.js` (wiring, 4 sites below),
`alz/viewer_shared/template/index.html.j2` (one include line).

**Risk tier:** **low** — the most code, but it fails *visibly*: the acceptance check renders the panel
and looks at it. A wrong spine, missing sidechains, uniform edge thickness, or an undistinguished
`both` edge is visible on inspection, not silent. Reuses the shared detail-panel machinery +
cytoscape verbatim. Low.

## Navigation — read this before designing anything

**There is no path-selection event, and no cross-tab path-selection channel.** Do not look for one; do
not build one. Verified against the live code: no `CustomEvent`/`dispatchEvent` exists anywhere in the
viewer JS. What exists is `_ipRuntime.openKeys` (`incytr_pathways.js:140`) — a module-local `Set` of
expanded row keys backing an **inline accordion detail panel** inside the pathways table, whose sub-tab
bar is built in `_ipRenderDetailPanel` (`:1788`). `IncytrFilter` (`incytr_state.js`) is the only
cross-tab channel and carries **pair** granularity (sender/receiver/disease), not a path row.

The sidechain graph is therefore a **4th sub-tab in that detail panel**, not a top-level tab: it
inherits the row (`r`) and its key (`rk`) directly from the panel it renders in, so navigation costs
zero new state. `r` already carries everything needed — `r.Ligand`, `r.Receptor`, `r.EM`, `r.Target`
(the 4 spine genes) and `r.contrast`.

**The 4 wiring sites in `incytr_pathways.js`** (both tables have an independent copy of the switcher —
patch both or the panel silently 404s on one table):
1. `_ipRenderDetailPanel` (`:1788`) — add `btn("sidechains", "Sidechains")` to `tabBar` and an
   `activeTab === "sidechains"` branch returning `tabBar` + a host div (mirror the `expands-to` branch).
2. Top-table click handler (`:1073-1088`) — add `if (tab === "sidechains") _isRenderSidechains(rk, r);`
3. Second-table click handler (`:1745-1760`) — the same line.
4. Both `openKeys` re-render loops (`:1090-1096` and the `isOpen` branch at `:1658-1663`) — route
   `sidechains` to the renderer the way `trajectory` / `expands-to` are routed.

Gate the button on slice availability (the way `hasTrajIdx` gates `Scores` and `showExpandsTo` gates
`Related pathways`): no slice for the active context → **no button**, not a button onto an empty panel.

**Registering the new file:** JS includes are an explicit list, not a glob — add
`{{ raw('js/tabs/incytr_sidechains.js') -}}` to `alz/viewer_shared/template/index.html.j2` (with the
other `incytr_*` includes). Both viewers render this same shared template and resolve `js/tabs/*.js`
local-dir-first then shared, so **one line covers both viewers**; `alz/viewer/verify_template.py`
checks every include resolves. Everything concatenates into a single `<script>` block, so top-level
`function` declarations are in one scope — `_ipRuntime` and the `_ip*` helpers are reachable from the
new file regardless of include order.

**Reuse (do not rebuild):**
- `window.cytoscape` 3.30.4 — **already CDN-loaded in `head.html` (verified) and currently unused.**
  Use it; add NO new dependency.
- `_ipRenderRelatedPairs` (`incytr_pathways.js:1889`) — **the template for this panel.** Async,
  guards on availability, degrades to a `muted` message, and re-checks
  `document.getElementById(hostId)` after every `await` (the panel may have closed mid-load). Copy that
  shape. Do **not** copy `incytr_chord.js`'s "unavailable" branch (`:80`) — it fires when **d3 is
  missing**, a different failure than an absent slice.
- `SliceCache` (`04_slice_cache.js`) / the `geneIndex` lazy sidecar (`incytr_pathways.js:155`,
  `:639-776`) — the on-demand fetch + `{url, data, error, promise}` cache pattern, including the
  `file://` fetch-block message. Fetch the shard **once per context**, not per row.
- `ViewerPayload.activeContext()` (`00_payload_adapter.js:30`) — returns the context id that keys the
  index directly. Do not re-derive a cohort/tissue mapping.

## Operating rules (non-negotiable — this repo's standing constraints)

- ANTI-SHIM: a pivot REPLACES the old approach; it does not coexist with it. No feature flag
  defaulting to the old mode, no `if name == "old"` branch, no legacy wrapper, no fallback to
  superseded logic, no env-var escape hatch, no config key toggled `enabled: false`. Update
  docstrings, comments, README, and runner scripts in the SAME pass. NO TOMBSTONES: when you remove
  something, delete it outright — do not leave a pointer recording that it once existed ("formerly X",
  "removed — see Y", a struck line, a status annotation on an emptied entry). Git holds the history.

- MEMORY SAFETY (shared box, OOM crashes other users' jobs): assume every dataset is too big for RAM
  until proven otherwise. Never call `pd.read_parquet` / `read_parquet` / `read.csv` / `fread` /
  `arrow::read_*` on a multi-GB file — not to "have a look", not inside a script you author (the
  validator/derive step you write is what crashes the box). The memory bill is row-count × col-count
  decompressed, not file size. Stream instead: DuckDB (`COPY … TO`, SQL aggregation/join),
  `pyarrow.parquet.ParquetFile(...)` row-group iteration, `arrow::open_dataset()` with explicit column
  selection — return only scalars or a bounded result. Check size first (`ls -lh` / `du -h` / parquet
  metadata). Run any step that could touch a large artifact under a cap:
  `systemd-run --user --scope -p MemoryMax=<N>G -p MemorySwapMax=0 …`. If you must load something big,
  say so first and wait for confirmation.

- VERIFICATION USES `pixi run`: bare `python` is the system interpreter and lacks project deps
  (pyarrow/duckdb/…). Always `pixi run python …` or the relevant `pixi run <task>`. Run
  `pixi task list` before assuming a task name.

- GIT: do not push, open PRs, force-push, `git reset --hard`, or any destructive git op without
  explicit instruction. Commit messages carry NO LLM attribution / Co-Authored-By trailer. NEVER
  commit data files (`.rds`, `.csv`, `.parquet`, `.mtx`, `.xlsx`, `.h5ad`, or other data artifacts).

- HONESTY OVER POLISH: no `TODO` / "coming soon" / "to be implemented" / placeholder in user-facing
  surfaces — omit an incomplete feature entirely. Do not ship known-incorrect output (nulls, stale
  joins, wrong vocabularies) to keep a diff small; fix the upstream cause. A bad benchmark result is a
  finding to report, not a metric to relax. **Do not render a visual that misleads about data quality**
  — see the `celltype_match` prohibition below, which is a live instance of this rule, not a hypothetical.

- NO COLLABORATOR CONTACT — EVER: do not propose emailing/asking/requesting anything from external
  collaborators. All work proceeds from artifacts on disk + what is derivable from them. Missing
  intermediates (`*.rds`, derived CSVs, cached matrices) are derivable — reconstructing them from raw
  inputs + method IS the task, not a blocker.

- TOOL DISCIPLINE: standard commands are shadowed by modern replacements (`grep`→`rg`, `cat`→`bat`,
  `find` is a wrapper). GNU/POSIX flags break in plain shell calls — use `command grep` / `command find`
  / the absolute binary when you need real coreutils behavior. In fish scripts, use builtins
  (`string match -r`, `string replace`, `count`), and never use `_` as a loop variable (read-only).

## The payload — exact shape (verified on disk, do not re-derive)

`PAYLOAD.edge_slice_ref.incytr_sidechains_index` → `sidechains_index.json`:

```
{ schema_version: 1, slice_type: "incytr_kinase_sidechains",
  by_context: { <context_id>: { url, interactome_edge_count,
                                interactome_node_count, terminal_edge_count } } }
```

Context ids are exactly `ViewerPayload.activeContext()` values — `song_ad`, `fivexfad_cortex`,
`fivexfad_hippocampus` (unified viewer) and `donor1` (t-cell viewer). Each shard is gzipped columnar
JSON (0.1–1.2 MB), parallel arrays, **not** row objects:

- `interactome` — `source_gene`, `target_gene`, `provenance`, `weight`, `weight_lit`, `weight_motif`,
  `in_vivo_refs`, `in_vitro_refs`, `n_motif_contrasts`, `motif_contrasts`. Kinase→kinase,
  **contrast-agnostic** (weights are max-|PDS| across contrasts). This is what the chain walk traverses.
- `terminal_edges` — `source_gene`, `target_gene`, `role`, `contrast`, `celltype_match`, `provenance`,
  `weight`, `weight_lit`, `weight_motif`, `best_abs_pds`. Kinase→pathway-node, **per-contrast**.

**The join:** `terminal_edges` where `target_gene` ∈ {`r.Ligand`, `r.Receptor`, `r.EM`, `r.Target`}
AND `contrast === r.contrast`. `role` ∈ {`Ligand`, `Receptor`, `EM`, `Target`} names the spine position
the kinase attaches to — **all four positions carry sidechains** (song: Target 97625, EM 8592,
Receptor 4239, Ligand 2981). Then walk `interactome` upstream from each terminal kinase for the
extended chains. `weight` is the evidence continuum — `norm(log1p(in_vivo_refs)) + norm(|PDS|)`, where
the `both` corroboration bonus IS the sum.

**Two traps in `weight`, both measured:**
- **`weight` can be exactly 0, and a 0-thickness edge is an invisible edge.** 275 of song's 3701
  interactome edges (265 of donor1's 6676) are zero-weight — psp-only edges with no in-vivo refs and no
  motif support. They are real edges. **Floor the rendered thickness at a visible minimum**; do not map
  weight linearly onto thickness from zero, or ~7% of the chain silently disappears. This is the one
  silent failure mode in an otherwise loud subplan. (Terminal edges have no zero-weight rows.)
- **`[0,2]` is the theoretical range; the observed max is ~1.64** (song 1.639, donor1 1.629) because
  the two normalized terms rarely peak together. Scale thickness against the **max of the loaded
  shard**, not a hardcoded 2.0, or the top ~18% of the range is never used.

**Curated specifics (invariants for this subplan's files)**

- **`celltype_match` MUST NOT drive any visual — no badge, no color, no filter, no count.** It is
  present in the shard but **not renderable**: `false` conflates "tested, negative" with "never tested",
  and the shard carries no marker distinguishing them. For t-cells, `D1_d15_vs_d2` and `D1_d19_vs_d2`
  have **exactly 0** `celltype_match=true` across 138,745 edges — those days fall outside
  `tcell_within_cohort.CONTRAST_DAYS`, so they were never tested. Rendering that reads as biology ("no
  attributable kinases at d15") when it is a coverage boundary. Ignore the column.
- **The provenance legend belongs on the interactome, not the terminal edges.** In `terminal_edges`,
  `psp` is **0 by design** (the terminal map is motif-anchored — settled, do not "fix") and `both` is
  ~0.06% (song: motif 113365 / both 72). Only the kinase→kinase `interactome` carries a real 3-way
  split (song: 2901 motif / 778 psp / 22 both). Distinguish `both` on the chain edges; do not build a
  3-class legend for terminal edges where one class can never appear.
- **Viewer ports = lift, not rewrite.** Reuse `viewer_shared/template/js/tabs/*.js` /
  `js/widgets/*.js` / `js/01_state.js` verbatim; reshape the payload to `PAYLOAD.*`. Greenfielding a
  new app.js ships a permanently inferior surface.
- **Frontend contract:** `docs/foundation/viewer_frontend_contract.md`.
- **The spine is never hidden** (source plan) — emphasizing the L→R→EM→T pathway is the whole point;
  weight-thinning applies to sidechains, never the spine. A spine node with zero terminal kinases still
  renders.
- After `pixi run viewer`, **hard-refresh (Ctrl+Shift+R / Cmd+Shift+R)** — the built HTML inlines
  PAYLOAD, so a normal refresh serves stale data. Check `PAYLOAD.meta.generated_at` in DevTools.

**Acceptance / verify:**
- Expanding a pathways row and opening `Sidechains` renders that row's spine + sidechains, in **both
  viewers** (unified: song + 5xFAD; t-cell: donor1). The spine is central and always drawn.
- Edge thickness scales with `weight`; **the panel re-reads the row's `contrast`** — two rows differing
  only in contrast show different terminal edges. A `both`-provenance chain edge is visually distinct
  from `motif`/`psp`. **Every edge in the drawn subgraph is visible**, including zero-weight ones.
- The `Sidechains` button is absent (not empty) when the active context has no slice; with the slice
  absent entirely the pathways table still renders and the other sub-tabs still work.
- `pixi run python alz/viewer/verify_template.py` passes (the new include resolves).
- No `celltype_match` reference in the shipped JS: `command grep -n celltype_match
  alz/viewer_shared/template/js/tabs/incytr_sidechains.js` returns nothing.

**Declared dependencies:** **03** (data) — renders the `incytr_sidechains` slice 03 emits; verified on
disk for all four contexts. Not file-disjoint from the pathways tab any more: this subplan edits
`incytr_pathways.js` (4 wiring sites) rather than registering a top-level tab, so it must not run
concurrently with other work in that file.
