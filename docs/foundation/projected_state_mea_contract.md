# Projected State MEA Contract (Packet 0B)

## Scope

This contract defines MEA for the T-cell cohort on **state-projected substrates**.

Projected state MEA is:

- kinase enrichment (`mea`) and related inference artifacts run on
  T-cell state-projected substrate matrices (not on raw bulk matrices),
- indexed by `state` (ProjecTILs `functional.cluster`),
- with contrasts as day-to-day changes versus a donor-specific baseline day.

This is **not** direct cell-state phosphoproteomics. The model is inference on
projections derived from deconvolved matrices, using the same interpretation
convention currently used for projected cell-type MEA in Song and 5xFAD:
the kinase signal is read as enrichment over state-projected substrate activity,
not as direct phosphoproteomic observation in a purified cell state.

## Interpretation Convention

- `state` is a biological label from ProjecTILs, not a deconvolved physical
  fraction.
- `NES` direction follows baseline-versus-contrast convention used by existing
  projected cell-type MEA: positive/negative NES carries the same interpretation
  language as standard state-level MEA in the unified systems.
- If a pair has no paired raw/stoich projected evidence, it is a `not_evaluable`
  manifest outcome (do not synthesize a score or call).
- No additional mechanism score is exported. Use categorical outcomes from
  downstream mechanism attribution logic only.

## First implementation target

Packet 0B implementation target is:

- `donor = donor1`
- `track = st`
- `kind = projected_state`
- states = ProjecTILs `functional.cluster` labels

No other donor/track must be made runnable before donor1 ST is contractually
defined and documented.

## Eligibility Inputs

Projected state MEA rows are eligible only when **all** of the following are
present for the donor/track pair:

- deconvolution artifacts under
  `data/derived/tcells_incytr_inputs/{donor}/`:
  - `ps_deconvoluted.csv` for `st` (donor1 expected; donor2 likely absent),
  - `py_deconvoluted.csv` for `py`.
- cell-state metadata:
  - `scrna/cell_counts.csv`,
  - `scrna/aggexp_data.csv`,
  - optional `scrna/projectils_embeddings.csv` (visualization only; not required
    for MEA eligibility),
  - optional `scrna/state_audit.json` (for traceability only).
- deconvolution traceability:
  - `decompose_manifest.json` (if present).
- a motif-bearing projected substrate stream for the requested track:
  `site_id + gene_symbol + motif` columns for `st`/`py` inputs.

## Skip Reasons

A row is `skip_reason = null` when inputs and required QC conditions pass.

Expected `skip_reason` values for Packet 0B are:

- `missing_projection_inputs` — required deconvolution inputs or `decompose_manifest.json` absent.
- `missing_state_metadata` — required `scrna/cell_counts.csv` or `scrna/aggexp_data.csv`
  absent.
- `missing_baseline_day` — donor baseline day unavailable for projected inputs.
- `no_post_baseline_days` — no eligible contrast day after baseline.
- `state_has_no_cells` — baseline or target day has zero cells for state.
- `no_motif_sites` — track has no motif-bearing projected sites after filtering.
- `not_evaluable` — malformed row joins or non-evaluable pairing at MEA level.

Special rule:

- `donor2` + `track = py` with no usable motif support is explicitly recorded as
  `not_evaluable` (not as a fatal missing-file error), using
  `skip_reason = no_motif_sites`.

## Deconvolution Prerequisite Command

If any T-cell deconvolution input is missing, Packet 0B must not run
regeneration. Record this as a prerequisite and stop before generation:

```bash
pixi run tcells-decompose
```

## Required Manifest Fields

Each `projected_state_mea_manifest.json` row must include at minimum:

| Field | Type | Units / value shape | Description |
|---|---|---|---|
| `donor` | `str` | `donor1` / `donor2` | Cohort donor id |
| `state` | `str` | ProjecTILs label | `functional.cluster` label |
| `track` | `str` | `st` / `py` | Kinase input track |
| `kind` | `str` | `projected_state` | Must be constant for this contract |
| `baseline_day` | `str` | day key (e.g. `d2`) | Baseline used for contrasts |
| `days_available` | `list[str]` | day keys | Days present in eligible projected inputs |
| `days_run` | `list[str]` | day keys | Days actually run for MEA |
| `n_cells_by_day` | `dict[str, int]` | counts by day | Cells with valid state assignment |
| `n_sites` | `int` | site count | Total projected substrate rows used before motif filter |
| `n_motif_sites` | `int` | site count | Motif-bearing sites entering motif scoring |
| `input_files` | `list[str]` | relative paths | Files required for that manifest row |
| `skip_reason` | `str | null` | reason key / null | Non-empty only when row is not evaluable |

No additional internal gate/quality constants should be exported in this
manifest; downstream decisions should use categorical calls and these raw
counts/evidence columns.
