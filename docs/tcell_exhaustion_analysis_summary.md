# T-cell Exhaustion Analysis Summary

This is the short reference to use when asked to summarize the T-cell
exhaustion analysis or its viewer. It is intentionally written as a stable
interpretive summary so future queries do not need to re-read the builder,
planning notes, and generated payload unless the user asks for current output
counts.

## Cohort

The T-cell exhaustion cohort is a two-donor human time-course dataset with
matched TMT proteomics and CITE-seq/scRNA. Donor 1 has Total proteome, pY, and
IMAC channels. Donor 2 has Total proteome and pY only, so Donor 2 supports
Incytr pathway analysis but not kinase MEA.

Time courses:

- Donor 1 scRNA days: d0, d2, d9, d13, d17, d20.
- Donor 2 scRNA days: d2, d5, d7, d9, d11.
- The analysis uses day-vs-d2 contrasts.

## Core Pipeline

The pipeline projects cells onto ProjecTILs CD4/CD8 reference atlases, extracts
state-keyed scRNA aggregate expression and cell counts, exports donor/day bulk
matrices, and builds state/day substrate estimates for Incytr.

Substrate and viewer cell states are keyed on per-cell ProjecTILs
`functional.cluster`, not Seurat clusters. This is a deliberate anti-shim
decision: collapsing ProjecTILs calls onto Seurat clusters caused large,
uninterpretable loss because the partitions did not agree. Cells without a
ProjecTILs call are dropped; otherwise state labels flow through directly as
sanitized names such as `CD8Tex`, `CD8Tpex`, `CD4Th17`, and `Treg`.

Canonical run order:

```bash
pixi run ingest-tcells-scrna
pixi run install-projectils
pixi run tcells-projectils-map
pixi run tcells-scrna-extract
pixi run tcells-export-bulk
pixi run tcells-decompose
pixi run tcells-incytr
pixi run tcell-within-cohort
pixi run tcell-viewer
```

## Kinase And Attribution Interpretation

Donor 1 is the only donor with kinase MEA because Donor 1 has IMAC. Donor 2 has
no IMAC, so the viewer should show Donor 2 kinase/attribution absence as an
expected cohort limitation, not a missing-data bug.

Within-cohort T-cell attribution localizes Donor 1 bulk kinase activity to
ProjecTILs states using this cohort's own scRNA:

- Detection evidence: whether the kinase transcript is present in each state
  (fraction of cells expressing >= 10%, normalization-free). Detection is shown
  separately and does not change the specificity denominator. State enrichment
  is computed against the kinase's MEAN expression across all adequately-sampled
  ProjecTILs states (>= 50 cells), gene-agnostic in its state set, so a kinase
  concentrated in one state scores a high fold; effective number of states uses
  the same all-state expression distribution.
- Transcript change: pseudobulk expression change versus d2.
- Concordance: the sign of bulk kinase NES times the transcript change.
- Time-course consistency: how often concordance is positive across the
  attribution days.

Important interpretation rule: detection is the informative localizer.
Concordance is shown as context only and must not be treated as a score or gate,
because kinase activity is inferred from substrate phosphorylation and can be
post-translationally decoupled from the kinase's own mRNA. The current
implementation ships the full kinase x state x attribution-day grid and labels
concordance; it does not pre-drop discordant rows.

There is no per-state significance test or FDR for the T-cell attribution. Donor
1 has one donor and one scRNA library per day, so per-cell tests would be
pseudoreplication. The viewer should expose raw evidence columns and categorical
specificity bins, not fabricate inferential scores.

## Positive-Control Kinase Lists

For T-cell receptor / cytokine positive controls, distinguish the current viewer
surface from the full donor1 MEA outputs:

- The dedicated T-cell viewer and within-cohort attribution expose donor1
  primary stoichiometry MEA for both residue tracks at once, mirroring the
  unified viewer: Ser/Thr from `kinase_timepoint_nes.csv` and tyrosine from
  `kinase_timepoint_nes_pY.csv`. Raw-phospho MEA remains an audit/sensitivity
  comparison, not a separate primary browser row.
- Donor1 pY MEA does score the expected protein tyrosine kinase controls in
  `kinase_timepoint_nes_pY.csv`: LCK, ZAP70, ITK, TEC, CSK, TXK, JAK1, JAK2,
  JAK3, and TYK2. `RLK` is a literature alias; kinase-library and the output
  table use `TXK`.
- DGK-alpha and DGK-zeta map to `DGKA` and `DGKZ`. These are lipid kinases, not
  motif-scored protein kinases in kinase-library, so their absence from kinase
  MEA outputs is expected. They are present as transcripts in the donor1 scRNA
  aggregate-expression matrix and can be evaluated as expression or pathway
  context, but not as motif-based phosphosite kinase activity in this pipeline.
- Donor2 remains unavailable for kinase MEA: no Ser/Thr IMAC, and its pY export
  has no flanking motifs for kinase-library scoring.

## T-cell Viewer

The dedicated viewer is built by:

```text
alz/build_tcell_viewer.py
```

Primary generated artifacts:

```text
outputs/reports/tcell_viewer/index.html
outputs/reports/tcell_viewer/tcell_viewer.payload.json
outputs/reports/tcell_viewer/tcell_viewer.payload.json.gz
outputs/reports/tcell_viewer/edge_slices/incytr_pathways/
```

The payload uses viewer schema v2. Donor routing is through
`meta.contexts`/`selection.context` and context-specific `*.by_context` blocks.
The current context ids are `donor1` and `donor2`; old donor-specific URL
aliases are compatibility input only.

Viewer behavior:

- Header: T-cell Pathway Viewer with Donor 1 / Donor 2 toggle.
- Donor 1: kinase explorer, temporal kinase profile, kinase audit/detail,
  within-cohort attribution, Incytr heatmap, and Incytr pathway table.
- Donor 2: Incytr heatmap and pathway table; kinase MEA and attribution are
  unavailable by design.
- Kinase explorer: day-vs-d2 NES profile, FDR-based significant-contrast count,
  specificity tier, and state/cell-type pills.
- Specificity filter: opt-in narrowing only; default is `Any`.
- Concordance: displayed in attribution detail, never used to filter or rank.
- Incytr drill-downs use lazy parquet sidecars, so serve the viewer directory
  over HTTP when those panels are needed; opening the HTML via `file://` blocks
  some browser fetches.

## Where To Check If Counts Must Be Current

Use these files only when the question needs current generated-output counts or
fresh validation:

- `outputs/reports/tcell_viewer/tcell_viewer.payload.json`
- `outputs/reports/kinase_attribution_tcells/donor1/unified_attribution_tcells.csv`
- `outputs/reports/kinase_attribution_tcells/donor1/mea/`
- `outputs/reports/incytr_pair_mode_tcells/<donor>/wide/`

Useful commands:

```bash
python alz/viewer/verify_payload_contract.py \
  outputs/reports/tcell_viewer/tcell_viewer.payload.json

pixi run tcell-within-cohort
pixi run tcell-viewer
```

## Source Files

- Cohort and run-order summary: `README.md` section "T-cell exhaustion cohort".
- Per-cell ProjecTILs aggregation decision:
  `docs/tcell_exhaustion_analysis_summary.md`.
- Within-cohort attribution implementation:
  `alz/cross_reference/tcell_within_cohort.py`.
- Concordance de-gating rationale:
  `docs/tcell_exhaustion_analysis_summary.md`.
- Viewer builder:
  `alz/build_tcell_viewer.py`.
- Viewer template:
  `alz/tcell_viewer/template/`.
- Shared viewer payload contract:
  `docs/foundation/viewer_payload_contract.md`.
