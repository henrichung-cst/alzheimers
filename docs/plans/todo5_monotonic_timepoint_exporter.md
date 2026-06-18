# Plan: T-cell monotonic timepoint exporter

**Status:** UNIMPLEMENTED — plan only.

---

## 1. Background and data structures

### Timepoint axis

Two donors; timepoints are calendar days of a T-cell exhaustion protocol, measured
from Day 2 (the baseline).

| Donor | Baseline | Later timepoints (from MEA NES wide) |
|-------|----------|--------------------------------------|
| donor1 | D1_d2 | D1_d13, D1_d15, D1_d17, D1_d19, D1_d20 (5 points) |
| donor2 | D2_d2 | D2_d5, D2_d7, D2_d9, D2_d11 (4 points) |

Monotonicity is defined over the **absolute abundance** columns (including
baseline), not over deltas. The ordered sequence is `[d2, d13, d15, d17, d19,
d20]` for donor1 and `[d2, d5, d7, d9, d11]` for donor2. Day numbers are parsed
from column names to determine order.

**Important note:** D1_d11 exists only in the total-proteome file; it is absent
from the phospho/stoichiometry tracks. The script must derive column order from
the column names present in each file, not from the sample_mapping (which lists
all channels including total-proteome-only ones).

### Kinase side

Source: **kinase_timepoint_nes** wide matrices. These are kinase × timepoint
matrices of NES values, where each column represents one post-baseline timepoint.
The baseline Day-2 NES is implicitly 0 (it is the reference), so the ordered
sequence for monotonicity is `[0, NES_d13, NES_d15, NES_d17, NES_d19, NES_d20]`
for donor1.

**Donor2 has no IMAC → no ST-track MEA at all. Its pY MEA was also skipped
(no flanking motifs in the donor2 pY ForPerseus export).** Confirmed by
`donor2/mea/` containing only `mea_manifest.json` and no NES CSV files. The
kinase exporter has exactly one NES source: donor1, two tracks (ST and pY).

Available files:
```
outputs/reports/kinase_attribution_tcells/donor1/mea/kinase_timepoint_nes.csv     # ST, 311 kinases × 5 timepoints
outputs/reports/kinase_attribution_tcells/donor1/mea/kinase_timepoint_nes_pY.csv  # pY, 78 kinases × 5 timepoints
```

Column schema: `kinase | D1_d13 | D1_d15 | D1_d17 | D1_d19 | D1_d20`

### Substrate/phospho side

Source: **raw_phospho_normalized** and/or **stoichiometry_matrix** CSVs.
These are site × timepoint matrices of median-centered log2 values, including
the baseline Day-2 column. The ordered sequence for monotonicity reads directly
from the columns (baseline included).

Both tracks, both donors:
```
outputs/reports/kinase_attribution_tcells/donor1/raw_phospho_normalized.csv       # ST, 62 807 sites × 6 cols (d2+5)
outputs/reports/kinase_attribution_tcells/donor1/raw_phospho_normalized_pY.csv    # pY
outputs/reports/kinase_attribution_tcells/donor1/stoichiometry_matrix.csv         # ST stoich
outputs/reports/kinase_attribution_tcells/donor1/stoichiometry_matrix_pY.csv      # pY stoich
outputs/reports/kinase_attribution_tcells/donor2/raw_phospho_normalized_pY.csv    # D2 pY (no ST)
outputs/reports/kinase_attribution_tcells/donor2/stoichiometry_matrix_pY.csv      # D2 pY stoich
```

Column schema: `site_id | protein_id | gene_symbol | site_position | motif | D{n}_d2 | D{n}_d{T1} | ...`

File sizes are all under 10 MB — standard pandas read is fine; DuckDB is not needed.

**Which substrate input to use:** Use `raw_phospho_normalized` (the direct
median-centered log2 phospho intensity) as the primary substrate source. The
stoichiometry matrix is the abundance-corrected signal used for MEA; it is
appropriate too and the script should accept a `--track-kind` flag
(`raw` / `stoich`, default `raw`) to allow either. Do not load both unless the
user requests it.

---

## 2. Monotonicity definition

### Predicate

For a vector `v = [v1, v2, ..., vN]` (values in chronological order,
NaN-excluded), define:

- **Monotonically increasing (weak):** `all(v[i+1] >= v[i])` for every adjacent
  pair in the NaN-filtered sequence.
- **Monotonically decreasing (weak):** `all(v[i+1] <= v[i])` for every adjacent
  pair in the NaN-filtered sequence.

Use **weak** monotonicity (≥ / ≤) as the default because real MS data has
plateau regions and this is a screen, not a strict ordering test. Expose a
`--strict` flag to tighten to strict (> / <) if desired.

**NaN handling:**
- For kinases: NES wide matrices are dense (no NaNs expected; MEA always
  produces a score for every kinase in the library).
- For substrates: NaNs occur (undetected in a run). A site with fewer than
  `--min-obs` (default 3, out of the total N timepoints including baseline) non-NaN
  values is excluded entirely. Monotonicity is evaluated over the NaN-filtered
  subsequence in chronological order.

**Constant sequences:** A flat series (all values equal) satisfies both weak
predicates. By default, exclude constants (add a `--include-constant` flag to
keep them). Implementation: after passing the weak check, additionally require
`max(v) - min(v) > 0`.

### Kinase-specific: prepend implicit baseline 0

The NES wide table has no Day-2 column because NES(baseline vs baseline) = 0
by definition. Prepend 0 as the first element before testing monotonicity so
the full time course [0, d13, d15, d17, d19, d20] is evaluated.

### Donor handling

- **Kinases:** only donor1 produces NES results. No cross-donor aggregation is
  possible or needed. Tag all kinase output rows with `donor=donor1`.
- **Substrates:** run each (donor, track) independently. A site is reported per
  (donor, track, kind) tuple where it passes. The same `site_id` can appear in
  both donors (with independent monotonicity assessments).

Do not aggregate across donors; per-donor monotonicity is the right unit for
this single-donor time-course design.

---

## 3. Script specification

### Location

`alz/cohorts/tcells/monotonic_export.py`

Rationale: cohort-scoped, alongside `ingest.py` and `mea.py`. Not wired into
`pixi.toml` since it's rarely used — run directly.

### Inputs (command-line)

```
--track-kind {raw,stoich}   substrate preprocessing to use (default: raw)
--strict                    use strict inequalities (default: weak)
--min-obs INT               minimum non-NaN timepoints per substrate site (default: 3)
--include-constant          include flat (zero-range) series that trivially pass
--out-dir PATH              output directory (default: outputs/reports/kinase_attribution_tcells/monotonic/)
```

No positional args; the script knows where the canonical files live from
`KINASE_DIR` (imported from `alz.cohorts.tcells.ingest`).

### Output files

Two CSVs written to `--out-dir`:

#### `monotonic_kinases.csv`

One row per (donor, track, kinase, direction) tuple that passes the predicate.

| Column | Description |
|--------|-------------|
| `donor` | `donor1` |
| `track` | `st` or `py` |
| `kinase` | kinase name |
| `direction` | `increasing` or `decreasing` |
| `n_timepoints` | number of timepoints (including implicit d2=0 prepend) |
| `value_d2` | implicit baseline NES = 0.0 |
| `value_d{T1}` ... `value_d{TN}` | NES at each later timepoint (wide, one col per day) |
| `range` | max(v) − min(v) |
| `slope_ols` | OLS slope of NES vs day number (float) |

The day columns are named `value_d{day}` (e.g. `value_d2`, `value_d13`, ...)
to avoid column-name collisions when stacking donors.

#### `monotonic_substrates.csv`

One row per (donor, track, kind, site, direction) tuple that passes.

| Column | Description |
|--------|-------------|
| `donor` | `donor1` or `donor2` |
| `track` | `st` or `py` |
| `kind` | `raw` or `stoich` (matches `--track-kind`) |
| `site_id` | e.g. `ZAP70_Y315` |
| `gene_symbol` | gene name |
| `site_position` | e.g. `Y315` |
| `motif` | flanking window (may be empty for donor2 pY) |
| `direction` | `increasing` or `decreasing` |
| `n_obs` | number of non-NaN timepoints used |
| `value_d2` ... `value_d{TN}` | per-timepoint log2 value (NaN where missing), one col per day |
| `range` | max − min over non-NaN values |
| `slope_ols` | OLS slope of value vs day number over non-NaN pairs |

### Algorithm skeleton

```python
def _day_order(cols: list[str]) -> list[tuple[int, str]]:
    """Return [(day_int, col_name), ...] sorted ascending."""
    ...

def _is_monotone(v: np.ndarray, strict: bool) -> str | None:
    """Return 'increasing', 'decreasing', or None. Excludes constants unless --include-constant."""
    ...

def export_kinases(out_dir, strict, include_constant):
    nes_files = {
        ("donor1", "st"): KINASE_DIR + "/donor1/mea/kinase_timepoint_nes.csv",
        ("donor1", "py"): KINASE_DIR + "/donor1/mea/kinase_timepoint_nes_pY.csv",
    }
    rows = []
    for (donor, track), path in nes_files.items():
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        day_cols = _day_order([c for c in df.columns if c != "kinase"])
        days = [d for d, _ in day_cols]
        cols = [c for _, c in day_cols]
        for _, row in df.iterrows():
            v = np.array([0.0] + [row[c] for c in cols])  # prepend implicit d2=0
            d = np.array([2] + days)
            direction = _is_monotone(v, strict)
            if direction:
                ...  # build output row
    pd.DataFrame(rows).to_csv(out_dir / "monotonic_kinases.csv", index=False)

def export_substrates(out_dir, track_kind, strict, min_obs, include_constant):
    substrate_files = {
        ("donor1", "st"): KINASE_DIR + f"/donor1/{prefix}.csv",
        ("donor1", "py"): KINASE_DIR + f"/donor1/{prefix}_pY.csv",
        ("donor2", "py"): KINASE_DIR + f"/donor2/{prefix}_pY.csv",
    }
    # where prefix = "raw_phospho_normalized" if kind=="raw" else "stoichiometry_matrix"
    ...
```

### Running the script

```bash
# default (raw phospho, weak monotonicity)
pixi run python -m alz.cohorts.tcells.monotonic_export

# stoichiometry track, strict
pixi run python -m alz.cohorts.tcells.monotonic_export --track-kind stoich --strict

# custom output directory
pixi run python -m alz.cohorts.tcells.monotonic_export --out-dir /tmp/mono_test/
```

---

## 4. Memory safety

All six substrate files are under 10 MB (checked 2026-06-18). Standard
`pd.read_csv` is fine. No DuckDB streaming needed. Load one file at a time and
discard before loading the next — do not accumulate all in memory simultaneously
(the combined phospho is ~32 MB, still fine, but keeping the pattern is good
practice).

---

## 5. Verification / sanity check

After writing both CSVs, print a summary to stdout:

```
[kinases] donor1/st: 311 kinases → N_inc increasing, N_dec decreasing (N_both: kinases monotone on BOTH tracks st+py)
[kinases] donor1/py:  78 kinases → N_inc increasing, N_dec decreasing
[substrates] donor1/st raw: 62807 sites → N_inc increasing, N_dec decreasing (NaN-excluded sites: K)
[substrates] donor1/py raw: ... sites → ...
[substrates] donor2/py raw: ... sites → ...
wrote monotonic_kinases.csv (N rows) → outputs/reports/kinase_attribution_tcells/monotonic/
wrote monotonic_substrates.csv (N rows)
```

**Spot-check:** print the top-3 increasing and top-3 decreasing kinases (by
`slope_ols`) to stdout as a sanity check. Expected: a strictly increasing NES
kinase should have `value_d2=0, value_d13 < value_d15 < ...` (weak).

**Sanity assertions in code:**
1. Every row in `monotonic_kinases.csv` must have `direction` in `{increasing, decreasing}`.
2. For increasing rows: `np.all(np.diff(v_nona) >= 0)` (or `> 0` if `--strict`).
3. For substrate rows: `n_obs >= --min-obs`.
4. `range > 0` for all rows when `--include-constant` is not set.

These can be a small `--verify` flag that re-reads the output and checks the
above (useful as a smoke test after the first run).

---

## 6. What is NOT in scope

- Not wired into `pixi.toml` (rarely used).
- No integration with the viewer or unified attribution.
- No cross-donor aggregation (intersection of monotone kinases across donor1
  tracks is left as a manual downstream filter on the CSV).
- No FDR filtering on kinases — the caller can filter the output CSV on any
  threshold.
- The stoich track for donor2 is absent (no IMAC); the script skips it silently
  with a log line, consistent with the existing `mea.py` skip contract.
