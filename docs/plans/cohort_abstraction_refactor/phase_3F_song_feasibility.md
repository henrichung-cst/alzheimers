# Phase 3F — Song Feasibility Report

Date: 2026-06-17
Status: feasibility (read-only; no code change)
Grounded against: callers read in full — `alz/bulk_mea/enrich.py` (lines 422–480),
`alz/bulk_mea/mechanism.py` (full), `alz/decomposition_mea/enrich_celltype.py` (full),
`alz/ctrl_outlier_audit/concordance_overlap_AD_excl_01_03.py` (full);
reference read: `alz/core/mea_runner.py`, `alz/core/fivexfad_celltype_mea_adapter.py`,
`alz/ingest/fivexfad_celltype_mea.py`, `docs/plans/cohort_abstraction_refactor/phase_3_design.md`.

---

## Caller 1 — Song bulk MEA (`alz/bulk_mea/enrich.py:455`, function `main()`)

### Call shape

| Dimension | Value |
|---|---|
| `_run_mea` calls per track | **1** (stoich only) |
| Loop dimension | `config.PHOSPHO_TRACKS` — ST and pY tracks |
| `lfc_key` | `"stoich_lfc"` |
| `track` | `track_name` (the track dict key, e.g. `"st"` or `"py"`) |

**Raw-LFC note (design doc §Divergences):** `raw_lfc` is computed and stored in
`results_by_contrast` by `_fit_and_contrast`, but `main()` passes only `"stoich_lfc"` to
`_run_mea`. Raw-LFC is NOT run through a second `_run_mea` call here — it is used only
in `mechanism.py` (caller 2). This design note is already in the phase_3_design.md and
must stay as-is.

### Output shape

Writes **5 files per track** (not 4):

1. `site_level_ols{suffix}.csv` — site-level OLS output (NOT a `_run_mea` output)
2. `mea_stoichiometry{suffix}.csv` — the long MEA table (runner's `mea_*` output)
3. `mea_global_shift{suffix}.csv`
4. `winsorized_sites{suffix}.csv`
5. `mea_substrate_sets{suffix}.csv`

The runner's 4-table-write shell covers files 2–5. `site_level_ols.csv` is an OLS output
written before `_run_mea` is called — it is not produced by `_run_mea` and the runner has
no mechanism to write it. This is the first cohort-specific divergence.

The MEA long-table filename is `mea_stoichiometry{suffix}.csv`, not
`mea_perdonor{suffix}.csv` or `mea_timecourse{suffix}.csv`. `long_table_stem` would need
to be `"mea_stoichiometry"` (no infix; suffix = track's `output_suffix`, e.g. `""` for ST
and `"_pY"` for pY).

Output path is produced via `config.track_output(filename, track_cfg)` which prepends
`KINASE_ATTRIBUTION_OUTPUT_DIR` and inserts the track suffix before the extension. The
runner currently passes `out_dir` + filename constants — it does not call `track_output`.
An adapter would need to replicate this path-composition or accept the right `out_dir`
per-unit.

### Cohort-specific vs runner-invariant

**Cohort-specific:**
- Input loading (`pd.read_csv(stoich_path)`, `pd.read_csv(raw_path)`)
- Sample mapping + filtering (`_filter_samples`, `analysis_mode`)
- Contrast construction (`_fit_and_contrast` — OLS + contrast computation)
- Output path composition (`config.track_output`)
- Writing `site_level_ols.csv` (before `_run_mea`)

**Runner-invariant (shared):**
- `_run_mea` call
- Writing the 4 MEA tables

### Pattern fit

This maps cleanly to the **table-writing shell** pattern: one `_run_mea` call per unit
(per track), four standard table writes, no accumulation. The adapter would implement
`iter_units` (once per track), `load_inputs`, `build_contrasts`, and set
`long_table_stem="mea_stoichiometry"` with the track suffix as `unit.suffix`.

**Escape hatches needed:**
1. The fifth output (`site_level_ols.csv`) is written BEFORE `_run_mea` in `main()`.
   An adapter's `write_aggregates` hook (called after the 4-table write) could handle
   this, but only if `site_ols` is stashed somewhere the adapter can recover it after
   `_run_mea` returns. The runner currently passes only `RunResult`
   (mea_df/shift_df/wins_df/substrate_df) to `write_aggregates` — `site_ols` is not in
   the result and not a `_run_mea` output. The adapter would need to retain `site_ols` as
   instance state during `build_contrasts` and flush it in `write_aggregates`. This is
   awkward but does not require the runner to change — it is adapter-side stashing.
2. The conditional write `if mea_df is not None and len(mea_df) > 0` for
   `mea_stoichiometry.csv` differs from the runner's unconditional `mea_df.to_csv`.
   The runner already records an empty-mea SkipRecord but still calls `write_aggregates`
   gated on `not mea_df.empty` (line 418), so an empty result skips `write_aggregates`
   while the 3 other tables still get written. The Song caller writes `mea_stoichiometry`
   only if non-empty but always writes the other 3. The runner's behavior matches this for
   the other 3 tables but would write an empty `mea_stoichiometry.csv` (zero rows) rather
   than omitting it. This is a minor layout divergence.

**Net complexity assessment:** Both escape hatches are adapter-side. The runner code
itself does not need modification. The `site_ols` stash is slightly awkward but is not
a new seam in the shared code.

### Recommendation: **FOLD VIA TABLE-WRITING SHELL (low priority)**

The pattern fits. The `site_ols` stash is the only structural quirk; it stays in the
adapter and does not infect the runner. Anti-over-abstraction check passes — the runner
removes the 4-table write duplication without adding a new escape hatch to shared code.

**Uncertainty:** The conditional `mea_stoichiometry.csv` write (omit if empty vs write
zero-row CSV) is a minor behavior difference. A future implementer should match the
canonical behavior exactly — omit if empty — either by an `if not mea_df.empty` guard in
`write_aggregates` or by using `_call_mea_unit` directly and writing manually. If this
corner case is judged load-bearing, note it in the parity checklist.

---

## Caller 2 — Song mechanism MEA (`alz/bulk_mea/mechanism.py:67`, function `_run_track_raw_mea`)

### Call shape

| Dimension | Value |
|---|---|
| `_run_mea` calls per track | **1** |
| Loop dimension | tracks (`st`, `py`) |
| `lfc_key` | `"raw_lfc"` |
| `track` | `track_cfg["name"]` |

### Output shape

Writes **1 file per track** (`mea_raw_phospho{suffix}.csv`), then concatenates across
tracks and produces two cross-track files:
- `mechanism_annotation.csv` — the per-(kinase, contrast) classification table
- `unified_attribution.csv` — the canonical production output, mutated in-place by merging
  `mechanism_annotation` in

The runner's 4-table-write shell writes 4 files per unit. For this caller:
- The caller writes ONLY `mea_raw` (one of the 4). It explicitly discards `shift_df`,
  `wins_df`, `substrate_df` (the return value is `mea_raw, _, _, _`).
- After collecting per-track `mea_raw` results it runs `_classify_mechanisms` and
  `_merge_mechanism_into_unified` — logic entirely outside `_run_mea`.

### Cohort-specific vs runner-invariant

**Cohort-specific:**
- The entire mechanism classification and unified-attribution merge
- Deliberately discards 3 of the 4 `_run_mea` outputs
- Per-track OLS for raw phospho via `_prepare_raw_ols` (different from Song bulk's
  `_fit_and_contrast` which also fits stoich)
- Cross-track concatenation and final output writes

**Runner-invariant:**
- The `_run_mea` call itself

### Pattern fit

The 4-table-write shell does NOT fit: the caller explicitly drops 3 of 4 outputs and
does not write standard tables. Forcing this into `run_unit` would require the runner to
suppress 3 of its 4 writes, which inverts the design premise and adds a new "partial-write"
mode to shared code.

The `mea_caller` injection pattern also does not cleanly fit: `_run_track_raw_mea`
returns the `mea_raw` DataFrame to `step_mechanism_annotation`, which then handles
cross-track logic. The injection point is a single `_run_mea` call with no accumulation
loop around it — there is no enclosing `run()` function that takes a `mea_caller`
parameter. Wiring the runner here would require restructuring `mechanism.py`'s call graph
to add an injectable `mea_caller` parameter to `_run_track_raw_mea` and then wrapping it
— adding abstraction for one call site with a non-standard output contract.

**Escape hatch assessment:** Fitting this caller requires EITHER:
(a) A new "discard 3 of 4 outputs" mode in the runner (anti-shim: more complex than the
    duplication removed), OR
(b) An adapter that calls `_call_mea_unit`, ignores 3 of 4 outputs, and is wired in
    via `mea_caller` injection into `_run_track_raw_mea` — doable but yields no
    measurable benefit: the skip/empty guard is the only shared behavior, and the current
    inline call already handles the empty check (`if mea_raw is None`).

### Recommendation: **LEAVE CUSTOM**

The output contract is non-standard (1 of 4 outputs used), the caller is a supplementary
diagnostic stage (not the primary production MEA), and fitting it adds net complexity to
either shared code or the adapter. The anti-over-abstraction guard fires. Leave
`mechanism.py` calling `_run_mea` directly.

---

## Caller 3 — Song decomposition per-cluster MEA (`alz/decomposition_mea/enrich_celltype.py:359`)

### Call shape

| Dimension | Value |
|---|---|
| `_run_mea` calls per unit | **1** |
| Loop dimension | `clusters` (31 spine clusters) |
| `lfc_key` | `"lfc"` |
| `track` | `track_cfg` (the resolved track dict) |

Note: `track` is passed as a dict (`track_cfg`), not a string. **RESOLVED (verified
2026-06-17):** `_run_mea` normalizes its `track` arg internally via
`config.resolve_track(track)` (`alz/bulk_mea/enrich.py:218`), which is idempotent —
it accepts either a string key (`"st"`/`"py"`, what the 5xFAD celltype caller passes)
**or** a pre-resolved `track_cfg` dict (what this caller passes). So a fold needs NO
reconciliation: `MeaUnit.track` is opaque and forwarded verbatim to `_run_mea`; storing
the dict there is fine even though the field is typed `str`. (Optionally a fold could
pass `track_cfg["name"]` for type cleanliness, but it is not required for correctness.)

### Output shape

Accumulate-and-write-at-end pattern — structurally identical to `fivexfad_celltype_mea`:

- Appends `(mea_df, shift_df, wins_df, subs_df)` to per-list accumulators
- Also accumulates `site_ols` (a per-cluster OLS output, not a `_run_mea` output)
- Inserts a `"cluster"` column into each DataFrame before accumulating
- After all clusters: concatenates and writes:
  - `mea_per_cluster{suffix}.parquet` — long MEA (parquet, not CSV)
  - `site_level_ols_per_cluster{suffix}.parquet` — OLS (parquet, not CSV)
  - `mea_global_shift_per_cluster{suffix}.csv`
  - `winsorized_sites_per_cluster{suffix}.csv`
  - `mea_substrate_sets_per_cluster{suffix}.csv`
  - `enrich_audit{suffix}.json`

Key structural differences vs the runner's 4-table-write shell:
1. **Parquet outputs** for the two main tables (not CSV). The runner always writes CSV.
2. **`cluster` column insertion** into each DataFrame before accumulation — this happens
   BETWEEN `_run_mea` returning and the output being written.
3. **`site_ols` accumulation** alongside the `_run_mea` outputs — `site_ols` is not a
   `_run_mea` output.
4. **Single concatenated write at end** — not per-unit writes.
5. **Per-cluster audit JSON** — not the runner's SkipRecord manifest.

### Structural twin comparison with `fivexfad_celltype_mea.py`

The 3D adapter (`fivexfad_celltype_mea_adapter.py`) demonstrated that the
accumulate-and-write-at-end pattern folds via **`mea_caller` injection**:

| Attribute | `fivexfad_celltype_mea.py` (3D) | `enrich_celltype.py` (Song decom) |
|---|---|---|
| Loop dimension | `(tissue, track, cell_type)` | `cluster` |
| `lfc_key` | `"lfc"` | `"lfc"` |
| Per-unit column insert | `cell_type`, `track`, `tissue` | `cluster` |
| Extra per-unit output | none (site_ols returned separately) | `site_ols` accumulated too |
| Final write format | parquet + CSV mix | parquet + CSV mix |
| Has a `run(mea_caller=None)` parameter | YES (designed for injection) | NO |
| Audit output | `fivexfad_celltype_mea_audit.json` | `enrich_audit.json` |

The 3D pattern works because `fivexfad_celltype_mea.run()` already accepts a `mea_caller`
parameter. `enrich_celltype.py`'s `main()` does NOT have a `mea_caller` parameter — the
injection point does not exist yet.

**Fitting via `mea_caller` injection would require:**
1. Refactoring `enrich_celltype.py`'s `main()` to accept a `mea_caller` parameter and
   thread it into the cluster loop (analogous to how `fivexfad_celltype_mea.run()` threads
   it into `_fit_one_celltype`).
2. An adapter that builds a `MeaUnit` per cluster, calls `runner._call_mea_unit`, and
   inserts the `cluster` column into the returned DataFrames.
3. The adapter also needs to handle `site_ols` accumulation — which is NOT passed through
   the `_run_mea` / `mea_caller` interface at all. `site_ols` is produced by
   `_ols_for_cluster` BEFORE `_run_mea` is called, meaning the adapter would need to
   capture it from outside the `mea_caller` closure.

**Track dict vs string — RESOLVED (verified 2026-06-17):** The call passes `track=track_cfg`
(a dict). `_run_mea` calls `config.resolve_track(track)` internally (`enrich.py:218`), which
is idempotent and accepts a string key OR a pre-resolved dict. `MeaUnit.track` is opaque and
forwarded verbatim, so this works with NO reconciliation. Not a prerequisite.

### Recommendation: **FOLD VIA `mea_caller` INJECTION (medium priority, after 3E)**

The structural twin relationship with 3D is genuine — `lfc_key="lfc"`, accumulate-and-
write-at-end, parquet/CSV mix. The 3D adapter is the right template. The refactor of
`enrich_celltype.py` to add a `mea_caller` parameter is the main prerequisite.

**Net complexity assessment:** The adapter does NOT inflate the runner. The required
change to `enrich_celltype.py` (add `mea_caller=None` parameter + thread to the
`_run_mea` call) is the same transformation already applied to `fivexfad_celltype_mea.py`
for 3D. The `site_ols` accumulation remains in `enrich_celltype.py`'s own loop — it is
never touched by the runner.

**Parity strategy:** Same as 3D — bounded per-cluster spot-check: for one
representative cluster, compare the per-cluster slice of the final concatenated parquets
against canonical. Because `_run_mea` is frozen and the only runner contribution is the
`_call_mea_unit` skip guard, input-equivalence implies output-equivalence for any cluster
that was not skipped in canonical.

**Flag:** None outstanding. (The `track`-dict question is resolved above —
`_run_mea`→`config.resolve_track` accepts the dict verbatim.)

---

## Caller 4 — Ctrl-outlier audit (`alz/ctrl_outlier_audit/concordance_overlap_AD_excl_01_03.py:124`)

### Call shape

| Dimension | Value |
|---|---|
| `_run_mea` calls per invocation | **2** (one per custom group contrast: AD-vs-clean, suspect-vs-clean) |
| Loop dimension | tracks (`st`, `py`) × 2 contrasts — 4 calls total |
| `lfc_key` | `"stoich_lfc"` |
| `track` | track name string (`"st"` or `"py"`) |

### Output shape

The script's purpose is boolean overlap analysis, NOT writing the standard MEA tables.
Outputs:
- `overlap_AD8_sus_clean.csv` — kinase overlap set with per-group agreement counts
- `substrates_leading_edge.csv` — leading-edge substrate sites shared across contrasts
- `MANIFEST.md` — provenance + result summary

The `_run_mea` outputs (particularly `mea_df` and `substrate_df`) are used INLINE as
intermediate data: `mea_df` drives `compute_overlap` (NES/FDR intersection logic) and
`substrate_df` drives `compute_substrates` (leading-edge site extraction). The 3 other
outputs (`shift_df`, `_outlier_df`) are explicitly discarded at the call site.

### Is this a production producer worth folding?

**No.** This is a one-off audit script with the following characteristics:
- Writes bespoke outputs (overlap CSV, substrate CSV, MANIFEST) — not MEA tables
- Uses `_run_mea` as a means to an end (get NES/FDR + leading-edge substrates), not to
  produce the standard 4-table set
- Constructs custom group contrasts (AD-vs-clean, suspect-vs-clean) from raw matrix
  columns — NOT the production contrasts defined in `CONTRAST_COEFS`
- Has a single fixed entry point (`main()`) with no `mea_caller` injection point
- Is self-contained and not called from any pipeline node

Folding it would add zero benefit: the runner's skip guard and 4-table write are
irrelevant to this script's purpose (it skips those tables entirely). Injecting the
runner would require restructuring the `_group_contrast` function to accept a `mea_caller`
parameter, and then building an adapter that passes only `mea_df` and `substrate_df` back
to the caller — a net increase in complexity for zero gain.

### Recommendation: **OUT OF SCOPE (audit script, not a production producer)**

Leave unchanged. The anti-over-abstraction guard fires immediately: the runner exists to
share the 4-table write + skip loop across production MEA producers, not to route every
`_run_mea` call in the codebase through a common framework.

---

## Overall Verdict

| Caller | Recommendation | Priority | Pattern |
|---|---|---|---|
| Song bulk (`enrich.py:455`) | Fold | Low | Table-writing shell |
| Song mechanism (`mechanism.py:67`) | Leave custom | — | N/A |
| Song decomposition (`enrich_celltype.py:359`) | Fold | Medium (after 3E) | `mea_caller` injection |
| Ctrl-outlier audit (`concordance_overlap_AD_excl_01_03.py:124`) | Out of scope | — | N/A |

### Recommended fold order

1. **Song decomposition** first, if folded at all — it is the structural twin of 3D
   and the `mea_caller` refactor to `enrich_celltype.py` is a known-small edit. Parity
   strategy mirrors 3D exactly: bounded per-cluster spot-check.
2. **Song bulk** second — the table-writing shell fit is clean, and the only adapter
   complexity is stashing `site_ols` in instance state and flushing it in
   `write_aggregates`. The conditional-write corner case on `mea_stoichiometry.csv` (omit
   if empty) must be tested explicitly in the parity check.

### What is NOT folded and why

- **Mechanism** — non-standard output contract (1 of 4 used), supplementary diagnostic
  stage, no adapter benefit without adding a partial-write mode to shared code.
- **Ctrl-outlier audit** — one-off audit script with bespoke outputs; folding it adds
  complexity and removes no meaningful duplication.

### Parity note

Song is the reference cohort. Both fold candidates (bulk + decomposition) must use
**end-to-end re-run parity** (not fingerprint-only) per the phase_3_design.md §Parity
strategy, with the independent re-run on a track/cluster the verifier picks, not the
implementer. For Song bulk, the ST track is the minimum (cheaper, no `_pY` suffix); for
Song decomposition, one representative cluster suffices.

### Flagged uncertainty

1. **`track` dict vs string in `_run_mea`**: `enrich_celltype.py` passes `track_cfg`
   (dict) to `_run_mea`; all other callers pass a string. Whether `_run_mea` internally
   calls `config.resolve_track` (which accepts both) must be confirmed before implementing
   the decomposition adapter. If it requires a string, the adapter extracts
   `track_cfg["name"]`.
2. **Conditional `mea_stoichiometry.csv` write**: the Song bulk caller omits the long
   table file when `mea_df` is empty/None; the runner writes a zero-row CSV. This is a
   layout difference, not just content parity. The parity check must assert file existence
   only when the canonical does, not merely content when located.
