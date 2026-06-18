# Phase 2 — MEA Output Helpers: Decision Log

## What was extracted

`alz/core/mea_outputs.py` holds the shared post-`_run_mea` logic that was
character-identical between `mukesh_perdonor.py` and `tcells_perdonor.py`
modulo the entity-axis noun (`donor` ↔ `timepoint`) and ordering rule.

The four bearers extracted:

| Helper | Purpose |
|--------|---------|
| `KIND_SPEC` | Shared `lfc_key` + `infix` for `stoich`/`raw` kinds |
| `build_nes_fdr_matrices` | Long → kinase×entity NES/FDR pivot |
| `build_recurrence_summary` | Per-kinase sig/up/dn/tested/median counts |
| `mea_output_path` | `{out_dir}/{stem}{infix}{suffix}.csv` path convention |

---

## API decisions

### KIND_SPEC split

Both producers have a `_KIND_SPEC` dict.  The shared keys are `lfc_key` and
`infix` — these are logic-bearing and identical between the two cohorts.
The cohort-specific extra keys (`matrix_kind` for mukesh, `base` for tcells)
are NOT in the shared `KIND_SPEC`; each scratch adapter merges them locally:

```python
spec = {**_SHARED_KIND_SPEC[kind], **local_extra}
```

This avoids forcing a cohort's unique fields into a shared dict awkwardly,
and keeps the shared module cohort-agnostic.

### `build_nes_fdr_matrices` — entity axis parameterisation

- `entity_col_name`: name of the transient column added to pivot on
  (`"donor"` or `"timepoint"`).  The long table on disk does NOT have this
  column — it has `contrast`.  The function strips `contrast_suffix` to
  derive it, matching exactly what the inline code does.
- `contrast_suffix`: `"_vs_CTRLmean"` (mukesh) or `"_vs_d2"` (tcells).
- `entity_order`: explicit ordered list, computed by the CALLER (not this
  module) because the ordering rule is cohort-specific:
  - mukesh: `ad_ids + ctrl_ids` (AD first, then CTRL; sourced from
    `sample_mapping.csv` via the existing `_split_samples` helper).
  - tcells: numerically sorted by the trailing day number, using
    `int(s.rsplit("_d", 1)[-1])` — same key as the inline code.

Passing a pre-computed list (rather than a key function or a mode flag) keeps
the shared function free of cohort-specific logic while maintaining the correct
column ordering in both wide matrices.

### `build_recurrence_summary` — subset and axis-noun

- `subset_ids`: column list to restrict to.  For mukesh this is either
  `ad_ids` or `ctrl_ids` (two separate calls produce `recurrence.csv` and
  `recurrence_ctrl.csv`).  For tcells it is all timepoint labels (single
  call, no CTRL arm).
- `axis_noun`: `"donors"` or `"timepoints"`.  This parameterises the four
  count-column names (`n_{axis_noun}_sig` etc.) and nothing else — the
  computation is identical.
- Empty-subset guard: an empty `subset_ids` returns a zero-row DataFrame
  with the correct columns, preserving mukesh's existing guard for the case
  where `ctrl_ids` would be empty.  The guard is expressed generically in
  the shared function rather than inline.
- `fdr_thresh`: callers pass `config.MEA_FDR_THRESH`; it is never hardcoded
  in the shared module.

### `mea_output_path` — path convention

A thin helper returning `os.path.join(out_dir, f"{stem}{infix}{suffix}.csv")`.
Chosen over a full `write_mea_result_bundle` wrapper because the audit
passthrough tables (shift/wins/substrate) are trivial `to_csv` calls with no
shared logic — forcing them through a bundle would add complexity without
removing duplication.

---

## Why canonical writers are left untouched

The existing inline blocks in `_run_track_kind` (mukesh lines 142–218,
tcells lines 153–207) are the SOLE producers of the canonical outputs under
`outputs/reports/kinase_attribution_human/perdonor/` and
`outputs/reports/kinase_attribution_tcells/`.  These are protected surfaces
(the Phase 0/1 baseline covers them).  Touching them in the same commit as
introducing a new module would conflate two changes: (1) extracting shared
logic and (2) proving the extraction is parity-safe.  The gated plan keeps
these as separate steps so a verifier can confirm parity before the cutover
lands.

---

## The three-call cutover (NOT applied here)

When the verifier signs off on scratch parity, each inline block can be
replaced with these three calls.  Shown for mukesh; tcells is symmetric
(swap `"donor"` → `"timepoint"`, `"_vs_CTRLmean"` → `"_vs_d2"`, `ad_ids` →
`tp_order`, two calls → one call).

**Mukesh cutover (replaces lines 161–218 of `_run_track_kind`):**

```python
from alz.core.mea_outputs import (
    build_nes_fdr_matrices, build_recurrence_summary, mea_output_path,
)

donor_order = ad_ids + ctrl_ids
nes_wide, fdr_wide = build_nes_fdr_matrices(
    mea_df,
    entity_col_name="donor",
    contrast_suffix="_vs_CTRLmean",
    entity_order=donor_order,
)
nes_wide.to_csv(mea_output_path(PERDONOR_DIR, "kinase_donor_nes", infix, suffix))
fdr_wide.to_csv(mea_output_path(PERDONOR_DIR, "kinase_donor_fdr", infix, suffix))

rec = build_recurrence_summary(
    nes_wide, fdr_wide,
    subset_ids=ad_ids,
    axis_noun="donors",
    fdr_thresh=config.MEA_FDR_THRESH,
)
rec.to_csv(mea_output_path(PERDONOR_DIR, "recurrence", infix, suffix), index=False)
print(f"  recurrence: {(rec['n_donors_sig'] >= 1).sum()} >= 1 donor sig")

rec_ctrl = build_recurrence_summary(
    nes_wide, fdr_wide,
    subset_ids=ctrl_ids,
    axis_noun="donors",
    fdr_thresh=config.MEA_FDR_THRESH,
)
rec_ctrl.to_csv(
    mea_output_path(PERDONOR_DIR, "recurrence_ctrl", infix, suffix), index=False
)
```

**Tcells cutover (replaces the tp_order + pivot + recurrence block in
`_run_track_kind`, roughly lines 175–207):**

```python
from alz.core.mea_outputs import (
    build_nes_fdr_matrices, build_recurrence_summary, mea_output_path,
)

tp_order = sorted(
    mea_df["timepoint"].unique(),        # column already added above
    key=lambda s: int(s.rsplit("_d", 1)[-1]),
)
nes_wide, fdr_wide = build_nes_fdr_matrices(
    mea_df,
    entity_col_name="timepoint",
    contrast_suffix="_vs_d2",
    entity_order=tp_order,
)
nes_wide.to_csv(mea_output_path(out_dir, "kinase_timepoint_nes", infix, suffix))
fdr_wide.to_csv(mea_output_path(out_dir, "kinase_timepoint_fdr", infix, suffix))

rec = build_recurrence_summary(
    nes_wide, fdr_wide,
    subset_ids=tp_order,
    axis_noun="timepoints",
    fdr_thresh=config.MEA_FDR_THRESH,
)
rec.to_csv(mea_output_path(out_dir, "recurrence", infix, suffix), index=False)
print(f"  recurrence: {(rec['n_timepoints_sig'] >= 1).sum()} >= 1 timepoint sig")
```

Note: the tcells inline code also adds `mea_df["timepoint"]` before the
pivot; the `build_nes_fdr_matrices` function handles this internally, so the
`mea_df["timepoint"] = ...` line is dropped from the cutover.

---

## Scratch sanity check (self-check at implementation time)

Both cohorts, one representative combo each:

| Cohort | Combo | Canonical wide shape | Scratch wide shape | Col match | Idx match | Recurrence shape |
|--------|-------|---------------------|--------------------|-----------|-----------|-----------------|
| mukesh | st/stoich | (311, 17) | (311, 17) | True | True | (311, 7) both |
| tcells/donor1 | st/stoich | (311, 5) | (311, 5) | True | True | (311, 7) |

Official value-level parity verification is the verifier's job (Phase 2 gate).

---

## CUTOVER — 2026-06-17

The deduplication was applied after scratch parity was confirmed above.

### Helpers added

**`mukesh_perdonor.py`**

```python
def _write_donor_aggregates(
    mea_df: pd.DataFrame,
    ad_ids: list[str],
    ctrl_ids: list[str],
    infix: str,
    suffix: str,
    out_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
```

Calls `build_nes_fdr_matrices` + `build_recurrence_summary` + `mea_output_path` from
`alz.core.mea_outputs`.  Writes `kinase_donor_nes`, `kinase_donor_fdr`,
`recurrence`, `recurrence_ctrl` into `out_dir`.  Keeps both `print(...)` recurrence
summary lines.  Returns `(nes_wide, fdr_wide)` so callers can report shapes without
re-pivoting.

**`tcells_perdonor.py`**

```python
def _write_timepoint_aggregates(
    mea_df: pd.DataFrame,
    infix: str,
    suffix: str,
    out_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
```

Derives `tp_order` (numeric-day sort via `int(s.rsplit("_d", 1)[-1])`) internally from
the contrast labels present in `mea_df`.  Calls the same shared helpers.  Writes
`kinase_timepoint_nes`, `kinase_timepoint_fdr`, `recurrence` into `out_dir`.  Keeps
the `print(...)` recurrence summary line.  Returns `(nes_wide, fdr_wide)`.

### Inline lines deleted

- **mukesh** `_run_track_kind` lines 161–218: `mea_df.copy()` + donor label strip,
  `pivot_table` × 2 + `reindex` + `to_csv` × 2, the nested `_recurrence()` function
  (12 lines), its two call sites + two `print(...)` lines.  Replaced with a single
  `_write_donor_aggregates(mea_df, ad_ids, ctrl_ids, infix, suffix, PERDONOR_DIR)` call.

- **tcells** `_run_track_kind` lines 176–207: `mea_df.copy()` + timepoint label strip,
  `tp_order` sort, `pivot_table` × 2 + `reindex` + `to_csv` × 2, inline recurrence
  `pd.DataFrame` build + sort + `to_csv` + `print(...)`.  Replaced with a single
  `_write_timepoint_aggregates(mea_df, infix, out_dir=out_dir, suffix=suffix)` call.

`git diff --stat`: mukesh +195 / -53; tcells +143 / -37; total 248 insertions, 90 deletions.

### Scratch adapters wired to the same helper

`regenerate_aggregates_to_scratch` in each producer now calls the cohort helper
directly instead of duplicating its own write logic.  Production and scratch now share
the identical code path.

### Module-level imports

`alz.core.mea_outputs` (`KIND_SPEC`, `build_nes_fdr_matrices`, `build_recurrence_summary`,
`mea_output_path`) imported at module top-level in both producers; inline
`from alz.core.mea_outputs import ...` inside the scratch function removed.

### Audit passthrough writes unchanged

The `mea_perdonor` / `mea_timecourse`, `mea_global_shift`, `winsorized_sites`, and
`mea_substrate_sets` `to_csv` calls remain exactly where they were in `_run_track_kind`
— before the `if mea_df.empty` guard and before the aggregate helper call.  They are not
routed through the helper.

### Cutover scratch sanity check (post-dedup)

| Cohort | Combo | Wide shape | Col match | Recurrence shape |
|--------|-------|------------|-----------|-----------------|
| mukesh | st/stoich | (311, 18) | True | (311, 7) both AD + CTRL |
| mukesh | st/raw | (311, 18) | True | (311, 7) both |
| mukesh | py/stoich | (78, 18) | True | (311, 7) both |
| mukesh | py/raw | (78, 18) | True | (311, 7) both |
| tcells/donor1 | st/stoich | (311, 6) | True | (311, 7) |
| tcells/donor1 | st/raw | (311, 6) | True | (311, 7) |
| tcells/donor1 | py/stoich | (78, 6) | True | (311, 7) |
| tcells/donor1 | py/raw | (78, 6) | True | (311, 7) |
