# Pathways detail tab — subplan 02: Frontend Pathways tab (JS)

**Goal:** Add a new "Pathways" tab to the kinase detail pane that renders the
inlined participation summary always, lazy-loads the per-kinase edge table over
HTTP on expand, and cross-links each edge into the Incytr Pathways tab.

**Scope**
- In (all within `alz/tcell_viewer/template/js/tabs/kinase_audit.js`):
  - Conditionally append `{id:"pathways", label:"Pathways"}` to
    `KINASE_AUDIT_TABS` (`:28`) **only when** the payload block
    `payload.kinase_incytr_participation` is present (donor1). Absent → no tab.
  - Add an `else if (tab === "pathways")` branch in `renderActiveKinaseAuditTab`
    (`:824`; branches read `ctx` from `_loadKinaseAuditContext`, `ctx.name` is
    the kinase). The `attribution` branch (`:937`) is the current last — append
    after it.
  - **Summary (always, works under `file://`):** headline `#pathways` (N and
    N/total %) and `#backbones` (M) from `payload.kinase_incytr_participation[ctx.name].counts`,
    plus three breakdown lists (`by_role`, `by_contrast`, `by_receiver`).
  - **Edge table (summary-first, lazy, HTTP-only):** collapsed behind an
    expander. On expand over HTTP, `AuditDataStore.load("kinase_incytr_edges")`,
    slice rows to `ctx.name`, render with `AuditTable` (sortable; default sort
    `pathways` desc, tiebreak `|signed_nes|`). Under `file://`
    (`AuditDataStore.fileMode`) the expander shows a "serve over HTTP for the
    edge table" note; the summary still renders.
  - **Cross-nav per edge row** (filter handoff, reuse the exact heatmap field
    set — see Curated specifics).
  - **Empty state:** kinase with 0 participating edges → "no observed pathway
    edges for this kinase" (its headline counts are 0).
- Out:
  - **No Python edits.** The payload block, columnar arrays, and edge sidecar
    are produced by subplan 01. Build against a fixture matching the pinned
    contract in `_index.md` until 01's output is available.
  - **Do NOT edit `kinase_explorer.js`** — the Explorer columns are unchanged
    (revision #1); this subplan does not touch them.
  - **Do NOT edit `01_state.js`** — `AuditDataStore` (`:400`) and `AuditTable`
    (`:457`) are consumed verbatim, not modified.
  - No Cytoscape/graph, no per-site table, no full motif-peer roster (badge shows
    detected/informative counts only). No donor2 rendering of this tab.

**Files touched:** `alz/tcell_viewer/template/js/tabs/kinase_audit.js`

**Risk tier:** medium — most of it fails loudly (`node --check` catches syntax; a
broken render is visible; the tab-inclusion gate throws or is empty). The one
silent-ish path is the cross-nav filter handoff: a wrong `receiverIn` / gene /
contrast could narrow the Incytr table to the wrong shortlist without erroring.
The acceptance check spot-verifies one cross-link lands a pathway with that gene,
role, and receiver.

**Reuse (do not rebuild — viewer ports are lift, not rewrite):**
- `AuditDataStore.load` (`01_state.js:400`) + `AuditTable` (`01_state.js:457`) —
  lazy fetch/cache and sortable table, verbatim. `AuditDataStore.fileMode` is the
  `file://` switch; it returns `block.preview` (empty for this sidecar) when in
  file mode.
- The existing `else if (tab === ...)` branches in this file as the structural
  template for the new branch.
- Cross-nav precedent: the `IncytrFilter.set(...)` + `SET_VIEW activeTab:"incytr"`
  + `_setIncytrPane("table")` sequence used by `temporal_v2.js:526` and the
  heatmap click handler.

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
  finding to report, not a metric to relax.

- NO COLLABORATOR CONTACT — EVER: do not propose emailing/asking/requesting anything from external
  collaborators. All work proceeds from artifacts on disk + what is derivable from them. Missing
  intermediates (`*.rds`, derived CSVs, cached matrices) are derivable — reconstructing them from raw
  inputs + method IS the task, not a blocker.

- TOOL DISCIPLINE: standard commands are shadowed by modern replacements (`grep`→`rg`, `cat`→`bat`,
  `find` is a wrapper). GNU/POSIX flags break in plain shell calls — use `command grep` / `command find`
  / the absolute binary when you need real coreutils behavior. In fish scripts, use builtins
  (`string match -r`, `string replace`, `count`), and never use `_` as a loop variable (read-only).

**Curated specifics (invariants for this subplan's files)**
- **Consume the pinned contract in `_index.md` exactly.** Read the summary from
  `payload.kinase_incytr_participation[ctx.name]` (`counts` / `by_role` /
  `by_contrast` / `by_receiver`); load the edge table with
  `AuditDataStore.load("kinase_incytr_edges")` and slice by `ctx.name`. Edge CSV
  columns and order are fixed there.
- **Breakdown semantics for labeling:** `by_role` is role-membership, NOT a
  partition (a pathway reached at >1 role is counted under each) — label it as
  such so the sum-of-roles > `#pathways` is not read as a bug. `by_contrast` /
  `by_receiver` DO partition `#pathways` (each sums to it). `#backbones` is the
  distinct Receptor∪EM union, not a sum.
- **Cross-nav field set (reuse verbatim, do not invent levers):**
  ```
  IncytrFilter.set({ pair:null, ipMode:"top", senderIn:[],
                     receiverIn:[receiver], searchText:target_gene,
                     /* disease, timepoint — only if the edge contrast maps */ });
  Store.dispatch({type:"SET_VIEW", key:"activeTab", value:"incytr"});
  _setIncytrPane("table");
  ```
  `receiverIn` + `searchText`(gene) always apply. `searchText` is an **any-role
  exact** gene match (`incytr_pathways.js` `_ipRowHasExactGeneSearchValue`), so it
  does not pin the edge's role — acceptable; role is shown in this tab. Add
  `disease`+`timepoint` from the edge contrast (`D1_d13_vs_d2`, row form) ONLY if
  it maps cleanly onto the t-cell Incytr `(disease, timepoint)` axis
  (`_ipAxisParts` / `ViewerPayload.contrastAxis`) — verify this mapping at
  implementation; if it does not map, omit contrast (receiver+gene still lands
  the right shortlist). Do NOT add a per-role search lever.
- **`file://` degradation is a real requirement.** Under `AuditDataStore.fileMode`
  the sidecar `preview` is empty by design (01 registers it empty) — the summary
  must render from the inlined block and the expander must show the note, never a
  broken/empty table.
- **Tab-inclusion gate:** the tab exists only when `payload.kinase_incytr_participation`
  is present; donor2 (no kinase MEA) has no block and must never show this tab.
- **Column-tooltip trap (this repo):** T-cell viewer column header tooltips render
  from `METRIC_DEFS` in `01_state.js` via `applyMetricTooltips()`, which
  OVERWRITES static `<th title>` text at boot. This tab is a detail-pane tab, not
  an Explorer column, so it likely needs no METRIC_DEFS change — but if you add any
  tooltip that should match an Explorer metric, edit the `METRIC_DEFS` `short:`
  field, not raw HTML (a raw `<th title>` edit is inert). Do not edit `01_state.js`
  for anything else.

**Acceptance / verify:**
- `node --check alz/tcell_viewer/template/js/tabs/kinase_audit.js` passes.
- Against a fixture (or 01's built payload): a donor1 kinase shows the Pathways
  tab; the summary headline equals the Explorer `#pathways`/`#backbones` for that
  kinase; the three breakdowns render; `by_contrast`/`by_receiver` visibly sum to
  `#pathways`.
- Served over HTTP: expanding the edge table fetches, slices to the kinase, and
  renders sorted by `pathways` desc; every visible row has `pathways ≥ 1`.
- Under `file://`: summary renders, expander shows the note (no broken table).
- A kinase with 0 participating edges shows the empty state.
- Spot-check one edge's cross-link: clicking it switches to the Incytr Pathways
  tab (table pane) and the narrowed table contains a pathway with that gene at
  that role and receiver (and contrast if the mapping was wired).
- The other four detail tabs still render unaffected; hard-refresh after any
  build (viewer inlines PAYLOAD into `index.html`).

**Declared dependencies:** none — parallel-safe (file-disjoint from subplan 01).
Consumes the pinned contract in `_index.md`; build against a fixture matching it.
Final integration render depends on 01's built payload being present.
