# kinase_sidechain_incytr_graph — subplan 04: cytoscape sidechain tab

**Goal:** A viewer tab that renders **one pathway spine at a time** (L→R→EM→T, always drawn and
central) decorated with its kinase sidechains and extended kinase→kinase chains, driven from the
existing pathways-table row selection, wired for song / 5xFAD / t-cells.

**Scope**
- In: A new cytoscape tab reading subplan 03's `edge_slice_ref` payload; on a path selection (from
  `incytr_pathways.js`) it pulls the 4 node genes' terminal kinases and walks the interactome to draw
  the sidechains + extended chains, edge thickness scaling with the evidence continuum and re-weighting
  with the selected pathway's contrast; `both`-provenance edges visually distinct; offline degrades
  gracefully. Register the tab in `TAB_MANIFEST` for both viewers.
- Out: No backend or payload changes (01/02/03). No new picker/menu — navigation is the existing table
  row selection. No all-pathways view, seed, or top-N. **No progressive click-to-expand** (deferred).
  No new CDN dependency.

**Files touched:** `alz/viewer_shared/template/js/tabs/incytr_sidechains.js` (new),
`alz/viewer/template/js/02_ui_chrome.js` (TAB_MANIFEST), `alz/tcell_viewer/template/js/02_ui_chrome.js`
(TAB_MANIFEST).

**Risk tier:** **low** — the most code, but it fails *loudly*: the acceptance check renders the tab and
looks at it. A wrong spine, missing sidechains, uniform edge thickness, or an undistinguished `both`
edge is visible on inspection, not silent. Reuses the shared tab machinery + cytoscape verbatim. Low.

**Reuse (do not rebuild):**
- `window.cytoscape` 3.30.4 — **already CDN-loaded in `head.html` and currently unused.** Use it; add
  NO new dependency.
- `incytr_chord.js` — its offline-degrade pattern (graceful when the slice is absent). Copy it.
- `incytr_pathways.js` — the selection event this tab subscribes to. Do not add a duplicate picker.
- The shared tab machinery in `viewer_shared/template/js/` (`01_state.js`, widgets) — reuse verbatim,
  reshape the payload to the `PAYLOAD.*` contract; do not greenfield an app.js.

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
- **Viewer ports = lift, not rewrite.** Reuse `viewer_shared/template/js/tabs/*.js` / `js/widgets/*.js`
  / `js/01_state.js` verbatim; the new tab reshapes the payload to `PAYLOAD.*`. Greenfielding a new
  app.js ships a permanently inferior surface. Genuinely-N/A features drop at `TAB_MANIFEST` level, not
  silently reimplemented.
- **Register in `TAB_MANIFEST` in BOTH viewers** — `alz/viewer/template/js/02_ui_chrome.js` (song +
  5xFAD) and `alz/tcell_viewer/template/js/02_ui_chrome.js` (t-cell). A tab absent from a cohort's
  manifest simply doesn't render for that cohort.
- **Frontend contract:** `docs/foundation/viewer_frontend_contract.md`.
- **The spine is never hidden** (source plan) — emphasizing the L→R→EM→T pathway is the whole point;
  weight-thinning applies to sidechains, never the spine.
- After `pixi run viewer`, **hard-refresh (Ctrl+Shift+R / Cmd+Shift+R)** — the built HTML inlines
  PAYLOAD, so a normal refresh serves stale data. Check `PAYLOAD.meta.generated_at` in DevTools.

**Acceptance / verify:**
- Selecting a path row in `incytr_pathways.js` renders that spine + sidechains in the new tab for
  **all three cohorts**; the spine is central and always drawn.
- Edge thickness scales with the evidence continuum and **re-weights when the selected pathway's
  contrast changes**; a `both`-provenance edge is visually distinct from `motif`/`psp`.
- With the slice absent (offline), the tab degrades gracefully (matches `incytr_chord.js`).

**Declared dependencies:** **03** (data) — renders the `edge_slice_ref` payload keys 03 emits.
File-disjoint from 03 (new JS + manifest edits vs. Python slice writers); the dependency is the payload
contract, so this tab can be authored in parallel against that contract and verified once 03 lands.
