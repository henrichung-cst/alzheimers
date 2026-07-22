# Pathways detail tab — subplan 01: Backend compute & delivery (Python)

**Goal:** Extend the existing terminal-edge participation scan to also emit the
per-kinase breakdowns and per-edge pathway counts, deliver them as one inlined
payload block plus one lazy edge sidecar — computed in a single scan, with the
existing Explorer columns left reading their current arrays.

**Scope**
- In:
  - Extend `_incytr_pathway_participation` (`alz/tcell_viewer/slices_kinase.py:609`)
    to return, per kinase, alongside the existing `incytr_pathway_count` /
    `incytr_backbone_count`: `by_role` / `by_contrast` / `by_receiver`
    distinct-row buckets, and a per-edge pathway-count lookup.
  - In `alz/tcell_viewer/slices_incytr.py`, from `_read_terminal_edges(donor)`
    (`:866`) build the compact participating-edge rows for the sidecar (drop the
    heavy `sites` / `motif_peer_roster` JSON; drop count-0 edges), reusing
    `_terminal_contrast_to_row` (`:854`) for the `contrast` column.
  - In `alz/build_tcell_viewer.py` (~`:245-250`): assemble the inlined
    `payload["kinase_incytr_participation"]` block (donor1-gated on
    `mea_kinase_donor`); write the edge sidecar CSV; register it in the audit
    manifest. **Keep** setting `kslice["incytr_pathway_count"]` /
    `["incytr_backbone_count"]` from the same scan result (revision #1).
  - Register the sidecar via `_audit_csv_meta` (`alz/tcell_viewer/slices_audit.py:143`),
    inside `build_tcell_audit_manifest` (`:326`) or the build assembly — match an
    existing entry's shape exactly, empty `preview`.
- Out:
  - **No JS edits.** Do not touch `kinase_audit.js` or `kinase_explorer.js`
    (subplan 02 owns the tab; the Explorer columns stay as-is per revision #1).
  - No new scan pass — extend the one composite-key scan already in
    `_incytr_pathway_participation`; richer return, not a parallel computation.
  - No `sites` / `motif_peer_roster` in the sidecar (counts only).
  - No count-0 edges in the sidecar.
  - No change to the terminal-edge rule itself (`kinase_kinase_edges.py`).

**Files touched:**
`alz/tcell_viewer/slices_kinase.py`,
`alz/tcell_viewer/slices_incytr.py`,
`alz/tcell_viewer/slices_audit.py`,
`alz/build_tcell_viewer.py`

**Risk tier:** high — the composite-key masks, the per-edge count map, and the
count-0 drop produce numeric counts a wrong join/mask would corrupt *plausibly*
(a count of 40 vs 42 looks fine). The strongest guard is that the acceptance
check ties the breakdown sums back to the headline AND the headline back to the
independently-derived Explorer column values — reproduce those exactly.

**Reuse (do not rebuild):**
- `_incytr_pathway_participation` (`slices_kinase.py:609`) — the existing
  composite-key `isin` scan over the ~1.87M-row global index. Extend its return;
  it already computes `#pathways`/`#backbones`.
- `_read_terminal_edges` / `_terminal_contrast_to_row` (`slices_incytr.py:866`/`:854`).
- `_audit_csv_meta` (`slices_audit.py:143`) and the empty-preview shim pattern
  (`slices_audit.py:235-244`) — reuse verbatim for the sidecar manifest entry.

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
- **Revision #1 (load-bearing):** compute the counts once in the extended scan
  and populate BOTH the existing `incytr_pathway_count` / `incytr_backbone_count`
  columnar arrays (Explorer JS reads these unchanged at `kinase_explorer.js:205-208`)
  AND the new `kinase_incytr_participation` block. Do NOT re-source the Explorer
  columns from the new block; do NOT edit any JS. One computation, no rewire, no
  two-representations-in-flight.
- **Distinct-row union semantics** (must match the headline exactly):
  `pathways` = |union of the three role masks|; `backbones` = |Receptor∪EM| (≤ pathways);
  `by_role` is role-membership (a pathway row reached at >1 role counts under
  EACH role — NOT a partition); `by_contrast` and `by_receiver` partition
  `pathways` (each row has exactly one contrast / one receiver → their values
  each sum to `pathways`). Compute breakdowns via `np.bincount` of the union
  mask's rows' `contrastId`/`receiverId`, map ids→names via the manifest vocabs.
- **Contract shape is pinned** — see `_index.md` "Pinned interface contract".
  The block keys and the sidecar CSV column set/order there are what subplan 02
  consumes; do not deviate. Sidecar manifest key = `kinase_incytr_edges`,
  `preview` empty.
- **Schema type-matching:** the sidecar `contrast` column must be row form
  (`_terminal_contrast_to_row`) to match the CONTRASTS vocabulary used elsewhere
  in the viewer. Match existing column dtypes when adding provenance columns.
- **Roles are Target/EM/Receptor only** (no Ligand); `owning_cluster == receiver`.
  Terminal-edge tuple `(kinase, target_gene, role, contrast, owning_cluster)` is
  already unique in `terminal_edges.csv` (no aggregation needed).
- **Data source sizes:** `terminal_edges.csv` is 78 MB / 53,280 rows (bounded —
  the existing `_read_terminal_edges` reads it fine); the global index is
  ~1.87M rows and is what the composite-key scan already streams. Do not add a
  second full-index materialization. Run the build memory-capped.
- The sidecar CSV is **build output** (written under the viewer output dir) —
  gitignored, never committed.

**Acceptance / verify:**
- Run the T-cell viewer build memory-capped:
  `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 pixi run <tcell-viewer-build-task>`
  (confirm the task name with `pixi task list`).
- Load the built `payload["kinase_incytr_participation"]` and assert, for a
  spot-check kinase: `sum(by_contrast.values()) == counts.pathways`;
  `sum(by_receiver.values()) == counts.pathways`; `|Receptor∪EM| == counts.backbones`;
  `counts.backbones <= counts.pathways`.
- Assert `counts.pathways` / `counts.backbones` for several kinases equal the
  values the Explorer columns show (i.e. equal the `incytr_pathway_count` /
  `incytr_backbone_count` arrays on the kinases slice) — the arrays and the block
  must agree because they come from one scan.
- Read the sidecar back (streamed / `nrows`, do not slurp if large): every row
  has `pathways >= 1`; column set/order matches the pinned contract; the
  largest-participation kinase (~1554 edges) is present with that many rows.
- Confirm the manifest entry for `kinase_incytr_edges` has a non-empty
  `relative_path`, empty `preview`, and `columns` matching the CSV.
- Report the inlined `index.html` size delta and the sidecar file size.

**Declared dependencies:** none — parallel-safe (file-disjoint from subplan 02).
Produces the pinned contract in `_index.md` that 02 consumes.
