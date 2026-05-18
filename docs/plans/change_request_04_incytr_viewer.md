# Change request 02 — Incytr pathways viewer (multi-contrast, temporal detail, trajectory)

## Goal

Make the Incytr Pathways tab answer three new questions:

1. *Which pathways recur across multiple disease contrasts (App, Tau, ApTt)?*
2. *For a chosen pathway, how does its PDS evolve across 2 / 4 / 6 mo?*
3. *Which pathways follow a stereotyped trajectory (always-up, going-up, etc.)?*

The data backbone is pair-mode Incytr (9 separate runs over disease × timepoint),
not factorial Incytr. Trajectory and multi-contrast filters are computed
client-side over the 9 pair runs already loaded for the selected
(sender, receiver) scope.

## Locked decisions

- Trajectory schema: coarse label + sign-vector tooltip.
  Categories: `always-up`, `always-down`, `monotonic-up`, `monotonic-down`,
  `early-only`, `late-onset`, `mixed`, `flat`. Sign vector across
  (2,4,6) per contrast shown on hover.
- "Flat" threshold for sign-vector classification: `|PDS| < 0.01`
  AND `pvalue ≥ 0.05` (both must hold to be "flat"). This mirrors the
  existing `tv2` defaults in `temporal_v2.js`.
- Multi-contrast filter: AND across selected contrasts (a pathway
  passes only if it has a non-flat row in *every* selected contrast at
  the active pvalue/|PDS| gates).

## Surface map

- `alz/build_unified_viewer.py` — builds the payload's
  `incytr_pathways` block and per-(sender, receiver) shards in
  `edge_slices/`. Today this is fed by the factorial cache.
- `alz/viewer/template/js/tabs/incytr_pathways.js` — table tab.
- `alz/viewer/template/js/tabs/incytr_heatmap.js` — entry point;
  clicks set `IncytrFilter` then route here.
- `alz/viewer/template/js/tabs/incytr_state.js` — shared
  `IncytrFilter` store; persisted under `incytrFilter.v1`.
- `alz/viewer/template/body.html:199` — `#tab-incytrpathways` panel
  with `ip-ms-*` multiselects, `ip-slider-p`, `ip-slider-pds`,
  `ip-reset`, `ip-count`, `ip-table-wrap`.

## Data contract (pair-mode source)

Each pair-mode run emits per-`(path, contrast)` rows where
`contrast = "<disease>_<timepoint>"`. The shard already loaded by the
table (`r.contrast`, `r.PDS`, `r.pvalue`, etc.) is exactly this shape —
nothing to change in shard format. What changes:

- `incytr_pathways` payload block gains a `trajectory_index`:
  `{ path_id → { contrast: "App|Tau|ApTt", traj_label: <coarse>, sign_vec: "u/f/d" × 3 } }`.
  Computed in the Python build at viewer-build time, not in the
  browser (cheaper than re-deriving per scope change).
- A `path_id` column is added to every shard row so the trajectory
  index joins cleanly. Use the existing `Path` string if unique, else
  hash of `(sender, receiver, Path)`.

## Implementation steps

### A. Python (`alz/build_unified_viewer.py`)

1. After loading the pair-mode receiver cache, group rows by `(sender,
   receiver, Path)` and split into 3 disease sub-frames. For each:
   - Read PDS at 2, 4, 6 mo (NaN if missing).
   - Apply the flat rule (`|PDS| < 0.01 AND pvalue ≥ 0.05` → 'f',
     else sign of PDS → 'u'/'d').
   - Map sign vector → coarse label:
     - `uuu` → always-up
     - `ddd` → always-down
     - `fuu` / `fud` / `uuu(monotonic increasing magnitude)` → monotonic-up etc.
     - `uff` / `ffu` → early-only / late-onset
     - `fff` → flat
     - anything else → mixed
   - Write `(path_id, contrast, traj_label, sign_vec)` to
     `trajectory_index`.
2. Compute a `contrast_recur` summary per path:
   `{ path_id → set_of_contrasts_with_any_significant_timepoint }` for
   the multi-contrast filter shortcut. Inline this as
   `incytr_pathways.recur_index`.
3. Bump the payload block's `version` field; older JS will fall back to
   the existing filter set (no trajectory chips, no temporal detail).

### B. JS state (`incytr_state.js`)

- Add to `IncytrFilter` defaults:
  `trajLabels: []`, `recurContrasts: []` (e.g., `["App","Tau"]` means
  the pathway must show up in both), `detailRowKey: null`.
- Bump persistence key to `incytrFilter.v2` to avoid stale chips.

### C. JS table tab (`incytr_pathways.js`)

1. **Trajectory chips** above the table:
   - Render one chip per coarse label, toggleable. Selecting any chip
     filters rows whose `trajectory_index[path_id][contrast].traj_label`
     matches.
   - Hover on the chip → tooltip of the sign vector "u/f/d at 2/4/6 mo
     for this contrast".
2. **Multi-contrast filter**:
   - Add a `Recur in` multiselect with options `App / Tau / ApTt`.
     If user picks `App` and `Tau`, AND the rows.
   - Use `recur_index` for the fast-path filter, then fall back to the
     loaded shards when both are selected (because the table already
     has all 9 contrasts for the loaded scope).
3. **Temporal detail row**:
   - Reuse the existing per-row expander (today shows the 4×7 fold-
     change matrix). Add a sibling tab "Trajectory" inside the
     expanded panel.
   - Render a small bar chart: x = 2/4/6 mo, grouped by App / Tau /
     ApTt, y = PDS. Use the same micro-bar pattern as
     `kinase_audit.js`'s `audit-mea-trajectory` (search
     `_renderMeaTrajectory`). Sign by color (red up, blue down) and
     greyed when below the flat threshold.
4. Trajectory tag is also surfaced in a new table column
   `trajectory`, sortable and filtered by the chip set.

### D. Heatmap (`incytr_heatmap.js`)

- No data change; add a faint trajectory glyph to each (sender,
  receiver) cell showing the dominant pathway trajectory in that cell
  (only when one label covers ≥ 50% of pathways). Low priority — gate
  behind a flag and ship after the table changes land.

## Open questions for you

- Trajectory monotonicity test: should I require strict monotonicity
  in |PDS| magnitude or just monotonicity in the sign sequence?
  Default I'd use is sign-monotonic (e.g., `fud` → monotonic-down).
- For the "Recur in" filter — do you want the gate to require the
  pathway to be significant in *all* selected contrasts (current
  pvalue/|PDS| sliders applied) or just *present* (a row exists)?
  Default: significant.

## Risks

- Pair-mode Incytr produces 9 separate cache directories; the viewer
  build needs to fan-in across them. Plumbing change in
  `build_unified_viewer.py`'s shard loop.
- `path_id` collisions across (sender, receiver) pairs would break the
  trajectory join. Hash key includes both.
- The current shard size (per (sender, receiver)) is already O(few MB).
  Adding `trajectory` and `sign_vec` per row is negligible.

## Sequencing note

The viewer JS work (B, C) can be developed against the existing
`incytr_factorial` cache by faking a trajectory_index from the cached
contrasts — fine for UI iteration. The Python build (A) must wait on
pair-mode Incytr from change request 02 (spine re-threshold).
