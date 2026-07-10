# T-cell viewer update for evidence-backed per-cell states

## Goal

Rebuild the T-cell viewer against the completed per-cell-state Incytr run while
keeping the state vocabulary consistent across pathways, kinase attribution,
projected-state MEA, traces, audit tables, and UI copy.

The viewer must not combine the current per-cell pathway outputs with historical
ProjecTILs state artifacts. It must expose raw evidence with clear units and
must not export internal gate values as analysis scores or viewer labels.

## Current mismatch

The generated payload at
`outputs/reports/tcell_viewer/tcell_viewer.payload.json.gz` still reports:

- `outputs/reports/incytr_pair_mode_tcells` as its pathway source;
- the historical 14-state donor1 ProjecTILs vocabulary in `celltypes`;
- historical ProjecTILs states in donor1 within-cohort attribution and
  projected-state MEA artifacts.

The completed production inputs instead have six donor1 states and seven donor2
states, and the filtered pathway parquets live under
`outputs/reports/incytr_pair_mode_tcells_percell/`.

These donor-scoped T-cell grids are a separate cohort contract, not a change to
the primary AD pair-mode invariant of 31 states and nine contrasts. The T-cell
runner continues to use `Incytr::Cal_pairwise_grid`, but its output shape is
derived independently from each donor's observed evidence-backed states and
later-day-versus-day-2 contrasts.

The viewer also publishes a `median_n <= 3` low-signal gate, derived exclusion
counts, and a `low_signal_endpoint` flag. That cutoff is an implementation
choice rather than a biological measurement and should not be exposed as an
analysis score. The underlying median, mean, minimum, total, and number of
observed timepoints are the interpretable evidence.

## Required changes

### 1. Pin the current pathway source

- Change `alz/tcell_viewer/paths.py` so the canonical pair-mode root is
  `outputs/reports/incytr_pair_mode_tcells_percell`.
- Allow an explicit environment override for diagnostics, while keeping the
  per-cell tree as the default.
- Update builder docstrings, provenance text, and validation reports to name the
  selected source path.
- Never fall back silently to the historical ProjecTILs tree.

### 2. Derive the viewer state roster from current evidence

- Replace `_load_donor_clusters()` in `alz/tcell_viewer/slices_kinase.py` with a
  state loader backed by `scrna/state_audit.json` or `scrna/cell_counts.csv`.
- Put state-roster loading and pathway-vocabulary validation in one shared
  T-cell viewer helper used by slices, audit provenance, and validation. No
  consumer should carry its own state-name table.
- Remove `PROJECTILS_LABEL_MAP` from the current celltype-slice path.
- Require the pathway sender/receiver vocabulary to be a subset of the donor's
  canonical state roster. Day-specific absent states are valid and must remain
  visible through the raw count evidence.
- Expected donor rosters:
  - donor1: `CD4Activated`, `CD4Naive`, `CD4Proliferating`, `CD4Resting`,
    `CD8Cytotoxic`, `CD8Exhausted`;
  - donor2: `CD4Activated`, `CD4ActivatedStress`, `CD4Proliferating`,
    `CD4Resting`, `CD8Cytotoxic`, `CD8Tex`, `CD8Tpex`.

### 3. Regenerate state-dependent kinase artifacts

- Update stale ProjecTILs wording and assumptions in
  `alz/cross_reference/tcell_within_cohort.py` before running it against the
  rebuilt `aggexp_data.csv`, `pct_expressing.csv`, and `cell_counts.csv`.
- Remove the obsolete minimum-state-cell gate from the new donor1 state axis;
  all six evidence-backed donor1 states have direct raw cell counts. Carry the
  raw count evidence rather than an internal eligibility score.
- Do not manufacture a one-to-one NSCLC/ProjecTILs crosswalk for the new state
  vocabulary. The new `CD4Activated`, `CD4Proliferating`, `CD4Resting`,
  `CD8Cytotoxic`, and donor1 `CD8Exhausted` calls do not have exact independent
  reference counterparts. Disable state-matched NSCLC corroboration unless an
  explicit biologically justified mapping is added later.
- Run `pixi run tcell-within-cohort` and confirm its shipped cell types are the
  six donor1 per-cell states.
- Run `pixi run tcell-state-mea` and confirm every current projected-state MEA
  file uses only the six donor1 states. Ensure historical state files cannot be
  discovered by the viewer after regeneration.

### 4. Remove the internal low-signal gate from the payload

- In `alz/tcell_viewer/slices_incytr.py`, retain raw `median_n`, `mean_n`,
  `min_n`, `total_n`, and `n_timepoints` fields with cell-count units.
- Remove `low_signal_median_n_threshold`, `low_signal_celltypes`,
  `low_signal_median_le_3`, `low_signal_endpoint`, and
  `pathway_counts_low_signal_excluded` from newly generated T-cell payloads.
- Update the shared T-cell rendering path so plots use the raw count fields and
  do not synthesize a default cutoff when those legacy keys are absent.
- Keep pathway counts based on the canonical production-filtered parquets; do
  not add a second hidden endpoint filter.

### 5. Update audit and UI provenance

- Replace ProjecTILs wording in `alz/build_tcell_viewer.py`, the T-cell template,
  and T-cell-specific JavaScript with “evidence-backed per-cell state” wording.
- Fix the T-cell audit manifest's deconvolution path: the manifest is at
  `<donor>/decompose_manifest.json`, not `<donor>/scrna/decompose_manifest.json`.
- Include the per-cell state-label and state-audit artifacts in the audit drawer.
- Historical `projectils_predictions.csv` and projection coordinates may be
  shown only when clearly labeled as comparison/audit artifacts; they must not
  populate current state controls.

## Verification

1. Add focused hard checks to the existing viewer validation entrypoints for the
   state-roster loader, pair-mode source override, absent day-state handling,
   and removal of legacy low-signal payload keys. Do not introduce a separate
   unit-test suite.
2. Run `pixi run tcell-within-cohort` and verify donor1 attribution contains
   exactly the six current states.
3. Run `pixi run tcell-state-mea` and verify every projected-state artifact is
   current-state keyed.
4. Run `pixi run tcell-viewer` and the viewer payload/template validators.
5. Inspect the generated payload and require:
   - pathway source is `incytr_pair_mode_tcells_percell`;
   - donor1 has six states and donor2 has seven states;
   - donor1 pathway total is 281,434 and donor2 pathway total is 1,251,773;
   - all seven contrasts are present;
   - no historical ProjecTILs-only state occurs in state controls, pathway
     endpoints, attribution rows, or projected-state MEA rows;
   - no internal low-signal threshold or derived exclusion score is exported;
   - raw cell-count evidence remains available for every state.
6. Open the built viewer and exercise donor switching, pathway heatmaps, pair
   shards, backbone grains, transcript/omics traces, kinase attribution, and the
   audit drawer.

## Implementation boundary

Do not solve this by changing only `INCYTR_PAIR_MODE_TCELLS_DIR`. The viewer is
ready to update only when every state-dependent producer and consumer above is
migrated together; otherwise the payload would silently mix incompatible state
ontologies.
