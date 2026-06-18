# Phase 3 — Shared MEA Runner: Decision Log (Wave 3A)

Date: 2026-06-17  
Wave: 3A — runner skeleton + Mukesh adapter (scratch only)  
Status: implementation CLOSED; parity pending external verifier

---

## Runner API (`alz/core/mea_runner.py`)

### `MeaUnit` (dataclass)

```python
@dataclass
class MeaUnit:
    track: str                                     # "st" | "py"
    kind: str                                      # "stoich" | "raw"
    lfc_key: str                                   # "stoich_lfc" | "raw_lfc"
    out_dir: str | Path                            # scratch output directory
    infix: str                                     # "" | "_raw"
    suffix: str                                    # "" | "_pY"
    motif_series: pd.Series | None                 # populated by load_inputs
    site_ids: np.ndarray | None                    # populated by load_inputs
    gene_symbols: np.ndarray | None                # populated by load_inputs
    results_by_contrast: dict[str, dict[str, np.ndarray]] | None  # build_contrasts
    meta: dict                                     # cohort-specific passthrough
```

`meta` carries `ad_ids`, `ctrl_ids`, and the loaded `matrix` DataFrame.  The
runner does not interpret `meta`; `write_aggregates` extracts what it needs.

### `MeaAdapter` (Protocol)

```python
class MeaAdapter(Protocol):
    def iter_units(self) -> Iterable[MeaUnit]: ...
    def load_inputs(self, unit: MeaUnit) -> MeaUnit: ...
    def build_contrasts(self, unit: MeaUnit) -> MeaUnit: ...
    def skip_check(self, unit: MeaUnit) -> tuple[bool, str | None]: ...
    def write_aggregates(self, result: RunResult) -> None: ...
    def write_provenance(self, skips: list[SkipRecord]) -> None: ...
```

`write_aggregates` and `write_provenance` are no-op by default — cohorts that
wrote nothing before keep writing nothing.

### `MeaRunner`

```python
class MeaRunner:
    def __init__(self, kinase_enrich_module: Any) -> None: ...
    def capture_fingerprint(self, unit: MeaUnit) -> dict: ...           # input-equiv mode
    def run_unit(self, unit, adapter, *, skip_check_fn=None) -> RunResult | None: ...
    def run_all(self, adapter: MeaAdapter) -> list[RunResult]: ...
    @property
    def skips(self) -> list[SkipRecord]: ...
```

`run_unit` owns steps 2–5 per `phase_3_design.md`: skip check → `_run_mea` →
4-table write → `write_aggregates` hook.  Returns `None` on skip.

`capture_fingerprint` is the input-equivalence primitive: emits shape/dtype/NaN
count/sha256 fingerprints of all `_run_mea` kwargs without calling it.

---

## Adapter shape (`alz/core/mukesh_mea_adapter.py`)

`MukeshMeaAdapter(scratch_dir, mapping, tracks)` is a thin connector:

| Method | Delegates to |
|--------|-------------|
| `iter_units` | yields `MeaUnit` for each `(track, kind)` combo |
| `load_inputs` | `_load_track_matrix` (from `mukesh_perdonor.py`) |
| `build_contrasts` | `_split_samples` + `_build_donor_deltas` |
| `skip_check` | returns `(True, reason)` if `motif_series is None` (raw track absent) |
| `write_aggregates` | `_write_donor_aggregates` (from `mukesh_perdonor.py`) |
| `write_provenance` | JSON skip log to `scratch_dir/skip_log.json` |

No new logic.  The adapter is a wiring harness, not a reimplementation.

---

## Input-equivalence design

`capture_fingerprint` compares:
- scalar fields: `track`, `lfc_key`, `infix`, `suffix`, `n_contrasts`
- `contrast_keys`: sorted list of contrast names
- `motif_series`: length, null count, first/last 3, sha256 over string join
- `site_ids`, `gene_symbols`: shape, dtype, NaN count, first/last 3, sha256
  (numeric arrays: tobytes after float64/int64 cast; string arrays: newline-joined)
- `contrasts[name]`: per-contrast shape/dtype/NaN/sha256 of the lfc array

Identical fingerprints + frozen deterministic `_run_mea` (fixed seed +
permutations) ⇒ identical outputs, by construction.  No permutation re-run
required.

---

## What stayed cohort-specific

- **AD/CTRL split and donor ordering** — computed inside the adapter from
  `sample_mapping.csv` via `_split_samples`.  Runner does not know about donor
  groups.
- **Contrast construction** — `_build_donor_deltas` is Mukesh-specific;
  T-cell will have its own delta builder.
- **`write_aggregates`** — the wide NES/FDR pivot + AD/CTRL recurrence is
  cohort-specific in the sense that it requires knowing `ad_ids`/`ctrl_ids`
  from `meta`.  The underlying helper (`_write_donor_aggregates`) is Phase-2
  shared, but the runner only calls it via the adapter hook.
- **`write_provenance`** — Mukesh historically wrote nothing here; the adapter
  writes a minimal JSON skip log.  The runner does not mandate a format.
- **Output directory layout** — canonical uses `PERDONOR_DIR`; runner uses
  `scratch_dir/{track}/{kind}/`.  Both paths coexist during the migration window.

---

## Opt-in runner entry in `mukesh_perdonor.py`

Added `_run_via_runner(scratch_dir, tracks)` and `--runner-scratch-dir DIR`
CLI flag.  The inline `_run_track_kind` canonical block is untouched.  The
two orchestration paths coexist during the Phase-3 migration window.

```
pixi run python alz/ingest/mukesh_perdonor.py --runner-scratch-dir <DIR> [--track py|st|both]
```

---

## Parity results (Wave 3A, 2026-06-17)

### (A) Input-equivalence

| Unit | Status | Compared |
|------|--------|---------|
| st/stoich | PASS | all fields: track, lfc_key, infix, suffix, n_contrasts, contrast_keys, motif_series, site_ids, gene_symbols, all 17 contrast arrays |
| st/raw | PASS | same |
| py/stoich | PASS | same |
| py/raw | PASS | same |

Command:
```
pixi run python alz/core/phase3_parity_harness.py \
  --scratch-dir outputs/reports/refactor_audit/phase_3/mukesh_new
```

### (B) End-to-end spot-check (py/stoich)

Runner output: `outputs/reports/refactor_audit/phase_3/mukesh_new/py/stoich/`

| File | Rows (runner) | Rows (canon) | Cols match | Worst \|Δ\| |
|------|--------------|--------------|-----------|-------------|
| mea_perdonor_pY.csv | 1326 | 1326 | YES | 0.00e+00 |
| kinase_donor_nes_pY.csv | 78 | 78 | YES | 0.00e+00 |
| recurrence_pY.csv | 78 | 78 | YES | 0.00e+00 |

Exact numerical identity (worst |Δ| = 0).

Command:
```
pixi run python alz/core/phase3_parity_harness.py \
  --scratch-dir outputs/reports/refactor_audit/phase_3/mukesh_new \
  --run-e2e
```

---

## Future cutover (as text, not applied)

When the verifier clears Wave 3A parity:

1. In `mukesh_perdonor.py::_run_track`, replace the two `_run_track_kind(...)` 
   calls with a `MeaRunner.run_all(adapter)` call.
2. Delete `_run_track_kind` and `_run_track`.
3. The `--runner-scratch-dir` flag becomes the only entry point; rename it
   `--out-dir` and default it to `PERDONOR_DIR` for the production run.
4. `_write_donor_aggregates`, `regenerate_aggregates_to_scratch`, and
   `--scratch-dir` (Phase-2 proof harness) are superseded; remove them.

This is gated: the cutover is a separate approved step, not part of Wave 3A.

---

## Abstraction friction / anti-over-abstraction notes

The Mukesh fit was natural.  Three observations:

1. **`meta` dict on `MeaUnit`** is a mild smell — it passes `ad_ids`/`ctrl_ids`
   and the loaded matrix from `load_inputs` to `write_aggregates` via the unit
   object.  A cleaner alternative would be for `write_aggregates(result, loaded_inputs)`
   to receive a second argument, but that changes the Protocol signature.  The
   `meta` approach is used by the canonical code implicitly (same variables in
   scope), so the adapter mirrors reality rather than cleaning it up prematurely.

2. **`motif_series.reset_index(drop=True)`** was needed in the adapter
   (`load_inputs`) but NOT in the canonical `_run_track_kind` (which uses
   `matrix["motif"]` directly without reset).  Added in the adapter to avoid
   index-alignment issues when fingerprinting.  This difference is benign (same
   values, same sha256 after both are reset) — but it surfaced that the canonical
   path relies on a positional-alignment assumption that `_run_mea` satisfies
   internally.  Input-equivalence confirmed both paths produce the same sha256.

3. **5xFAD** is expected to be harder (double `_run_mea` + concat per unit for
   bulk; pseudobulk join + reweighted signal for celltype).  The runner's
   `run_unit` primitive works per `_run_mea` call; the adapter's
   `write_standard_tables` override is the natural extension point.  No change
   to the runner is anticipated.

---

# Phase 3 — Shared MEA Runner: Decision Log (Wave 3B)

Date: 2026-06-17  
Wave: 3B — T-cell adapter (scratch only)  
Status: implementation CLOSED; parity pending external verifier

---

## Adapter shape (`alz/core/tcells_mea_adapter.py`)

`TcellsMeaAdapter(scratch_dir, donors, tracks)` is a thin connector:

| Method | Delegates to |
|--------|-------------|
| `iter_units` | yields `MeaUnit` for each `(donor, track, kind)` combo |
| `load_inputs` | reads CSV via `_matrix_path`; records `matrix_absent` rich skip dict |
| `build_contrasts` | `_baseline_and_days` + `_build_timepoint_deltas`; records `no_motif` rich skip dict |
| `skip_check` | returns `(True, reason)` if matrix absent or n_motif == 0 |
| `write_aggregates` | `_write_timepoint_aggregates` (Phase-2 helper in tcells_perdonor.py); records `mea_ran` + `mea_empty` |
| `write_provenance` | per-donor `mea_manifest.json` in T-cell's exact canonical schema |

Units enumerate as `donor × track × kind` (e.g. `donor1/st/stoich`, ...,
`donor2/py/raw`).  Output dir per unit is `<scratch>/<donor>/mea/`.

No new logic.  The adapter is a wiring harness.

---

## Runner — no changes

`mea_runner.py` was NOT modified.  T-cell's richer skip handling is managed
entirely within the adapter's `load_inputs` / `build_contrasts` /
`write_aggregates` hooks using the existing `skip_check` + runner skip-guard
path.  No runner extension was needed.

---

## Manifest schema faithfulness

T-cell's canonical `mea_manifest.json` has a richer skip schema than Mukesh's
skip log:
- `matrix_absent`: `{donor, track, kind, reason, path}`
- `no_motif`: `{donor, track, kind, reason, n_sites}`
- `mea_empty`: `{donor, track, kind, reason, n_sites}`

The adapter records these as full rich dicts in `_rich_skips[donor]` during
`load_inputs` / `build_contrasts` / `write_aggregates` respectively, then emits
them verbatim in `write_provenance`.  The runner's `SkipRecord` (which holds a
simpler `unit_label`/`reason`/`detail` tuple) is used only for the runner's
internal skip list; the manifest uses the adapter-owned rich dicts.

This is an intentional design choice: the runner's `SkipRecord` is the minimal
shared primitive; adapters that historically had a richer skip schema (T-cell)
maintain that schema independently in `write_provenance`.

---

## Opt-in runner entry in `tcells_perdonor.py`

Added `_run_via_runner(scratch_dir, donors, tracks)` and `--runner-scratch-dir
DIR` / `--track` CLI flags.  Inline `_run_track_kind` and `_run_donor` are
untouched.  Two coexisting paths during the Phase-3 migration window.

```
pixi run python alz/ingest/tcells_perdonor.py --runner-scratch-dir <DIR> [--donor both|donor1|donor2] [--track both|st|py]
```

**Important:** run `--donor both --track both` (defaults) in a single
invocation to get the correct cumulative manifest.  Running tracks separately
creates a fresh adapter per invocation, so `mea_ran` in the manifest only
reflects that invocation's track subset.  This matches the canonical
`_run_donor` behaviour where all combos accumulate in one call.

---

## Honest diff characterisation (`tcells_perdonor.py`)

The git diff against HEAD shows 157 insertions / 32 deletions.  The 32
deletions are Phase 2's extraction of the inline aggregate block from
`_run_track_kind` into `_write_timepoint_aggregates` (Phase 2 work committed in
working tree but not yet in HEAD at the time of this wave).  My Phase 3
additions are purely additive:
- New import of `_SHARED_KIND_SPEC`, `build_nes_fdr_matrices`, etc. (already
  used by `_write_timepoint_aggregates` via Phase 2)
- `_run_via_runner` function (new)
- `--runner-scratch-dir` / `--track` CLI flags in `main` (new)

`_run_track_kind` and `_run_donor` bodies are byte-identical to the Phase 2
state.  `enrich.py` diff is empty.

---

## Parity results (Wave 3B, 2026-06-17)

### End-to-end self-check (donor1 py/raw — authoritative)

Runner output: `outputs/reports/refactor_audit/phase_3/tcells_new/donor1/mea/`
Canonical: `outputs/reports/kinase_attribution_tcells/donor1/mea/`

| File | Rows (runner / canon) | Cols match | Worst \|Δ\| | Int exact |
|------|-----------------------|------------|-------------|-----------|
| mea_perdonor_raw_pY.csv | 390 / 390 | YES | 0.00e+00 | YES |
| kinase_timepoint_nes_raw_pY.csv | 78 / 78 | YES | 0.00e+00 | YES |
| kinase_timepoint_fdr_raw_pY.csv | 78 / 78 | YES | 0.00e+00 | YES |
| recurrence_raw_pY.csv | 78 / 78 | YES | 0.00e+00 | YES |

Exact numerical identity (worst |Δ| = 0 across all numeric columns).

Command:
```
pixi run python alz/ingest/tcells_perdonor.py \
  --runner-scratch-dir outputs/reports/refactor_audit/phase_3/tcells_new
```
(runs both donors, both tracks in one invocation)

### Manifest parity

| Donor | Status | Notes |
|-------|--------|-------|
| donor1 | PASS | mea_ran=['st/stoich','st/raw','py/stoich','py/raw'] matches canonical |
| donor2 | PASS | mea_ran=[] mea_skipped=4 — all fields match including path/n_sites |

### Donor2 skip-by-design

donor2/mea contains only `mea_manifest.json` — no spurious CSVs.
All 4 combos skipped: st/{stoich,raw} → matrix_absent; py/{stoich,raw} → no_motif.

---

## Abstraction friction / anti-over-abstraction notes

1. **Rich skip dicts vs `SkipRecord`**: T-cell's canonical manifest embeds
   per-skip extra fields (`path` for matrix_absent, `n_sites` for no_motif /
   mea_empty) that the runner's `SkipRecord` doesn't carry.  Rather than
   extending `SkipRecord` (breaking 3A/Mukesh) or silently dropping these
   fields, the adapter maintains a parallel `_rich_skips` dict.  Mild smell, but
   faithful to the design principle: cohorts that had richer provenance keep it.

2. **Cumulative manifest requires single invocation**: the adapter accumulates
   `_ran` and `_rich_skips` in memory across the `run_all` loop.  CLI users who
   run tracks separately get per-track manifests, not the full cumulative one.
   This is documented; the verifier should run `--donor both` (default) without
   `--track` to get the correct manifest.  A future cutover can default to a
   single entry point (like canonical `_run_donor`) and this isn't an issue.

3. **No runner changes required**: T-cell's per-donor unit dimension (vs
   Mukesh's single-donor) fit naturally into `iter_units` yielding one unit per
   `(donor, track, kind)` — the runner loop over `iter_units` handles this with
   zero modification.  Friction was lower than anticipated.

---

# Phase 3 — Correctness Fix: T-cell long-table stem

Date: 2026-06-17  
Wave: 3B post-fix — filename drift defect  
Status: CLOSED (verified)

## Defect

`MeaRunner.run_unit` (line 364) hardcoded the MEA long-table filename stem as
`"mea_perdonor"`.  This is correct for Mukesh but wrong for T-cell, whose
canonical stem is `"mea_timecourse"`.  The runner-driven T-cell path was writing
`mea_perdonor_pY.csv` / `mea_perdonor_raw_pY.csv` / etc. instead of the
canonical `mea_timecourse_*` names — output drift with identical content.

## Fix

Added `long_table_stem: str = "mea_perdonor"` field to `MeaUnit` (default
preserves Mukesh).  `TcellsMeaAdapter.iter_units()` sets
`long_table_stem="mea_timecourse"` in each yielded unit.  The runner replaces
the literal with `f"{unit.long_table_stem}{infix}{suffix}.csv"`.

Files changed:
- `alz/core/mea_runner.py` — new field on `MeaUnit`; literal replaced in `run_unit`
- `alz/core/tcells_mea_adapter.py` — `long_table_stem="mea_timecourse"` in `iter_units`
- `alz/core/mukesh_mea_adapter.py` — no change (default `"mea_perdonor"` is correct)

## Other stems confirmed identical across cohorts

`mea_global_shift`, `winsorized_sites`, `mea_substrate_sets` — both cohort
canonicals use these exact names; left as shared constants in the runner.

The aggregate stems (`kinase_donor_nes*`, `kinase_timepoint_nes*`, `recurrence*`)
are written by adapter `write_aggregates` hooks, not the runner — already
cohort-specific by design; no runner change needed for 3C/3D.

## Parity results (2026-06-17)

Mukesh regression (py/raw unit, comparing `kinase_attribution_human/perdonor/` canonical):

| File | Rows | Cols | worst\|Δ\| |
|------|------|------|------------|
| mea_perdonor_raw_pY.csv | 1326 | 10 | 0.00e+00 |
| kinase_donor_nes_raw_pY.csv | 78 | 18 | 0.00e+00 |
| recurrence_raw_pY.csv | 78 | 7 | 0.00e+00 |

T-cell fix (donor1 py/raw unit, comparing `kinase_attribution_tcells/donor1/mea/` canonical):

| File | Rows | Cols | worst\|Δ\| |
|------|------|------|------------|
| mea_timecourse_raw_pY.csv | 390 | 10 | 0.00e+00 |
| kinase_timepoint_nes_raw_pY.csv | 78 | 6 | 0.00e+00 |
| recurrence_raw_pY.csv | 78 | 7 | 0.00e+00 |

No `mea_perdonor*` files in `tcells_new/`. donor2 contains only `mea_manifest.json`.
`enrich.py` diff empty.

---

# Phase 3 — Correctness Fix: Mukesh out_dir flattened

Date: 2026-06-17
Wave: 3B post-fix — Mukesh nested-output defect
Status: CLOSED (verified)

## Defect

`MukeshMeaAdapter.iter_units()` constructed `out_dir = self.scratch_dir / track / kind`,
writing outputs into `<scratch>/st/stoich/`, `<scratch>/st/raw/`, `<scratch>/py/stoich/`,
`<scratch>/py/raw/`.  Canonical Mukesh writes ALL four combos flat into one directory
(`kinase_attribution_human/perdonor/`), disambiguated by filename infix/suffix
(`mea_perdonor_raw_pY.csv` etc.).  The nesting was unnecessary (all 4 combos produce
distinct filenames — zero collisions) and constituted deliverable drift.

## Fix

One-line change in `alz/core/mukesh_mea_adapter.py` line ~80:

```python
# Before:
out_dir = self.scratch_dir / track / kind
# After:
out_dir = self.scratch_dir
```

## T-cell adapter finding

`TcellsMeaAdapter.iter_units()` already writes flat per donor:
`out_dir = self.scratch_dir / donor / "mea"` → matches canonical
`kinase_attribution_tcells/donor{N}/mea/`.  No fix needed.

## Layout verification (re-run after fix, stale scratch deleted first)

Mukesh scratch `mukesh_new/` — relative path set vs canonical `perdonor/` (excluding
`donor_groups.json` which the runner does not own):

```
scratch == canonical  (32 files, zero diff)
```

File set (flat, no subdirs):
```
kinase_donor_fdr.csv            kinase_donor_fdr_pY.csv
kinase_donor_fdr_raw.csv        kinase_donor_fdr_raw_pY.csv
kinase_donor_nes.csv            kinase_donor_nes_pY.csv
kinase_donor_nes_raw.csv        kinase_donor_nes_raw_pY.csv
mea_global_shift.csv            mea_global_shift_pY.csv
mea_global_shift_raw.csv        mea_global_shift_raw_pY.csv
mea_perdonor.csv                mea_perdonor_pY.csv
mea_perdonor_raw.csv            mea_perdonor_raw_pY.csv
mea_substrate_sets.csv          mea_substrate_sets_pY.csv
mea_substrate_sets_raw.csv      mea_substrate_sets_raw_pY.csv
recurrence.csv                  recurrence_ctrl.csv
recurrence_ctrl_pY.csv          recurrence_ctrl_raw.csv
recurrence_ctrl_raw_pY.csv      recurrence_pY.csv
recurrence_raw.csv              recurrence_raw_pY.csv
winsorized_sites.csv            winsorized_sites_pY.csv
winsorized_sites_raw.csv        winsorized_sites_raw_pY.csv
```

`find mukesh_new/ -type d` → only `mukesh_new/` itself (no track/kind subdirs).

T-cell scratch `tcells_new/` — relative path set vs canonical:
```
donor1/mea/  →  29 files (matches donor1/mea/ exactly)
donor2/mea/  →  mea_manifest.json only (matches donor2/mea/ exactly)
```

## Content regression

| File | Shape | worst\|Δ\| |
|------|-------|------------|
| Mukesh mea_perdonor_raw_pY.csv | (1326, 10) | 0.00e+00 |
| Mukesh kinase_donor_nes_raw_pY.csv | (78, 18) | 0.00e+00 |
| Mukesh recurrence_raw_pY.csv | (78, 7) | 0.00e+00 |
| T-cell donor1 mea_timecourse_raw_pY.csv | (390, 10) | 0.00e+00 |
| T-cell donor1 kinase_timepoint_nes_raw_pY.csv | (78, 6) | 0.00e+00 |
| T-cell donor1 recurrence_raw_pY.csv | (78, 7) | 0.00e+00 |

All 6 checks: exact numerical identity (max |Δ| = 0).
`enrich.py` diff empty.

---

# Phase 3 — Wave 3C REWORK: 5xFAD bulk MEA adapter (genuine reuse)

Date: 2026-06-17
Wave: 3C rework — replace duplicated adapter with injectable mea_caller seam
Status: CLOSED (verified)

## Problem with first 3C attempt

The first 3C attempt (flagged by the verifier's parity_report_3C.md) produced
correct output but unacceptable code:

- `FivexFadBulkMeaAdapter` class (~220 LOC) was never called by `run_via_runner`.
- `fit_track` was imported into the adapter but never called.
- `run_via_runner` reimplemented ~90 lines of `fit_track`'s OLS + contrast +
  concat logic (a third copy also existed in the dead `build_contrasts` method).
- The `run_via_runner` docstring falsely claimed it called `fit_track`.
- The stoich-matrix skip guard was duplicated (once inline, once inside
  `_call_mea_unit`).

## Design: injectable mea_caller seam

The correct design keeps all OLS/contrast/concat logic in ONE place (`fit_track`
in `fivexfad.py`) and makes the MEA invocation injectable.

### `fit_track` signature change (`alz/ingest/fivexfad.py`)

```python
def fit_track(
    tissue: str,
    track: str,
    manifest: pd.DataFrame,
    mea_caller: Callable[..., tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]] | None = None,
) -> dict[str, pd.DataFrame]:
```

The `mea_caller` parameter is optional (default `None`). Inside `fit_track`:

```python
_call_mea = mea_caller if mea_caller is not None else kinase_enrich._run_mea
```

Both `_run_mea` call sites inside `fit_track` use `_call_mea(...)` with the
same positional + keyword arguments as before. The default path (`mea_caller=None`)
is byte-identical to the pre-rework canonical behavior — `kinase_enrich._run_mea`
is called directly, unchanged.

The `Callable` import was added (`from typing import Callable, Iterable`).
No other changes to `fivexfad.py`'s logic or structure.

### Thin adapter (`alz/core/fivexfad_bulk_mea_adapter.py`)

The adapter is now ~145 LOC total (down from 471 LOC). What it does:

1. Builds a `MeaRunner` instance and a `_PassAdapter` stub.
2. For each (tissue, track), checks for the stoichiometry matrix. On miss:
   appends a `SkipRecord` to `runner._skips` and `continue`.
3. Defines `_make_mea_caller(tissue, kl_track)` — a factory that returns a
   closure with the exact `kinase_enrich._run_mea` signature. Inside the closure,
   a `MeaUnit` is constructed and `runner._call_mea_unit(unit, _adapter)` is
   called. The `RunResult` is unpacked to `(mea_df, shift_df, wins_df,
   substrate_df)`. If the runner returns `None` (skip), four empty DataFrames are
   returned so `fit_track`'s concat/annotate logic stays intact.
4. Calls `fit_track(tissue, track, manifest, mea_caller=mea_caller)` — the
   returned 7-DataFrame dict is written flat to `scratch_dir`.

### What was deleted from the adapter

| Deleted | Why |
|---------|-----|
| `FivexFadBulkMeaAdapter` class (220 LOC) | Never called; dead code |
| `build_contrasts` method (~60 LOC) | Third copy of OLS logic; dead |
| Inline OLS block in `run_via_runner` (~50 LOC) | Duplicate of `fit_track` |
| Inline annotation/concat block (~40 LOC) | Duplicate of `fit_track` |
| Inline skip guard (stoich matrix) | Kept once; removed inline duplicate |
| `import fit_track` (dead import) | Now actually called |
| False docstring claiming `fit_track` was called | Replaced with truthful docstring |
| `_KIND_SPEC` dict | Only needed by the dead class/adapter pattern |
| `from alz.ingest.fivexfad import ... _build_design_matrix, _contrast_coefs, _contrast_qc, _contrast_group_counts, _age_from_contrast` | All used only in the dead/duplicate blocks; removed |
| `import numpy as np` | No longer needed in the adapter |

## Runner involvement (honest assessment)

`MeaRunner._call_mea_unit` is called once per `_run_mea` invocation (twice per
(tissue, track): once for `stoich_lfc`, once for `raw_lfc`). It contributes:
- `adapter.skip_check` call (via `_PassAdapter` stub — always passes after the
  stoich-matrix guard upstream)
- `motif_series is None` guard
- `0 non-null motifs` guard
- `results_by_contrast is empty` guard
- Empty-mea warning/SkipRecord

The OLS, contrast construction, concat assembly, and 7-file output naming are
all inside `fit_track` — the single authoritative copy.

## `mea_runner.py` changes

None. The runner was not modified. Mukesh and T-cell behavior is unaffected.

## Parity results (Wave 3C rework, 2026-06-17)

### Runner path (via `--runner-scratch-dir`)

Scratch: `outputs/reports/refactor_audit/phase_3/fivexfad_bulk_new/`
Canonical: `outputs/reports/kinase_attribution_5xfad/`

| Prefix | site_level_ols | mea_stoich | mea_raw | mea_global_shift | wins_sites | contrast_qc | substrate sha256 |
|--------|---------------|------------|---------|-----------------|------------|-------------|-----------------|
| cortex_py | Δ=0 | Δ=0 | Δ=0 | Δ=0 | Δ=0 | Δ=0 | MATCH (d6290bbfcde96919) |
| cortex_st | Δ=0 | Δ=0 | Δ=0 | Δ=0 | Δ=0 | Δ=0 | MATCH (30c4e163114ae165) |
| hippocampus_py | Δ=0 | Δ=0 | Δ=0 | Δ=0 | Δ=0 | Δ=0 | MATCH (5c8f28da53fe8e6f) |
| hippocampus_st | Δ=0 | Δ=0 | Δ=0 | Δ=0 | Δ=0 | Δ=0 | MATCH (f1bf83c1a6f3ecdd) |

Layout: 28 files in scratch == 28 canonical MEA output files (set diff empty).

### Default path (mea_caller=None — proves the seam is behavior-preserving)

Scratch: `outputs/reports/refactor_audit/phase_3/fivexfad_default_path/`
Verified prefixes: cortex_py, cortex_st

| Prefix | site_level_ols | mea_stoich | mea_raw | mea_global_shift | wins_sites | contrast_qc | substrate sha256 |
|--------|---------------|------------|---------|-----------------|------------|-------------|-----------------|
| cortex_py | Δ=0 | Δ=0 | Δ=0 | Δ=0 | Δ=0 | Δ=0 | MATCH (d6290bbfcde96919) |
| cortex_st | Δ=0 | Δ=0 | Δ=0 | Δ=0 | Δ=0 | Δ=0 | MATCH (30c4e163114ae165) |

### Canonical output guard

All 4 canonical `*_mea_stoichiometry.csv` mtimes remain 2026-06-15. No canonical
file was touched.

### enrich.py

`git diff alz/bulk_mea/enrich.py` → empty (frozen).

### Dead code

`grep` for `FivexFadBulkMeaAdapter`, `build_contrasts`, `_run_ols_all_sites`,
`_contrast_stats`, `_safe_concat`, `_annotate` in the adapter → 0 matches.

---

# Wave 3D — 5xFAD celltype adapter

Date: 2026-06-17
Status: CLOSED (PASS). Verifier `audit-pipeline` ≠ implementer `general-purpose`.

## Diagnostic resolved before implementing
User flagged "5xFAD celltype balloons 21× vs Song." That compared celltype vs Song
**bulk** — wrong baseline. Like-for-like vs Song's own per-cell-type MEA
(`alz/decomposition_mea/enrich_celltype.py` → `outputs/reports/decomposition/levy_t5/
mea_substrate_sets_per_cluster*.csv`): Song celltype 23.9M substrate rows / ~1.3 GB,
5xFAD celltype 24.9M / 1.8 GB — same scale. No ballooning; intrinsic cell-type fan-out.

## Rigor decision (human)
**Bounded per-slice.** Prove the seam on one (tissue,track,cell_type) slice run both
ways; frozen+deterministic `_run_mea` covers the rest. No full 1.8 GB re-emission.

## Seam (mirrors 3C)
`_fit_one_celltype(..., mea_caller=None)`; `_call_mea = mea_caller or kinase_enrich._run_mea`.
`run(..., out_dir=OUT_DIR, mea_caller=None)` — only the 7 output writes redirect; inputs
unchanged. Adapter `fivexfad_celltype_mea_adapter.py` routes the single call through
`runner._call_mea_unit`. Default path byte-identical.

## Parity (verifier, slice hippocampus/py/Excitatory-Pyramidal)
Runner (b) == default seam-off (a): all 7 files layout-equal, 6 data files bit-identical
(worst |Δ| 0.000e+00), ints exact, determinism scratchA==scratchA2. Frozen files empty diff.

## Open finding — stale canonical (NOT a 3D defect)
Default path ≠ on-disk canonical for `lfc`/NES (4,735/6,980 OLS rows; max |Δ| lfc 2.9e15
on pinv-unstable sites). Mechanism proven: canonical written 2026-06-15 18:08; pseudobulk
input regenerated 2026-06-16 08:59 (~15h later); zero code change since `d756bc7`. The
canonical is stale vs its current input; default code reproduces correctly from the current
pseudobulk. Structural fields + substrate row counts match (membership input-invariant).
Decision deferred: regenerate vs leave (5xFAD on hold). A re-run regenerates identically via
inline OR runner path.

## Verdict
PASS — runner faithfully reproduces the inline default celltype path (3D's scope). Stale
canonical logged separately.
