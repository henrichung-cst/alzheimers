# 5xFAD Kinase MEA and Viewer Notes

This note records the 5xFAD kinase enrichment conventions used by
`alz/ingest/fivexfad.py` and the unified viewer 5xFAD kinase tab.

## Scope

5xFAD is treated as a supporting AD cohort in the unified viewer. Cortex and
hippocampus are modeled independently, then exposed as viewer filter dimensions
inside the Mouse (5xFAD) kinase surface.

Primary kinase MEA tracks are:

- IMAC/ST
- pY

Stoichiometry is the primary analysis track. Raw phospho is retained as the
sensitivity track.

## Sample Handling

The ingest manifest records tissue, assay, raw run, age, genotype, biological
sample ID, pool status, duplicate group, analysis action, and sensitivity flag.

Primary MEA uses samples with `analysis_action == "primary"`. Explicit pool runs
and sensitivity-only runs are retained in provenance but not used in primary
TG-vs-WT contrasts. Technical duplicate runs are averaged after log2 transform so
they contribute one biological sample column.

The primary sample counts are retained in contrast QC and viewer payload
provenance. They do not gate MEA contrasts.

## Log Transform Convention

Nonpositive intensity values are set to missing before log2 transform. This
matches the Song and Mukesh code paths and avoids creating artificial extreme
negative log2 values through pseudocount substitution.

The same convention is used for total proteome, IMAC/ST, and pY inputs.

## MEA Contrast Policy

The previous 5xFAD-only `MIN_REPLICATED_GROUP_N` mask has been removed. 5xFAD
now follows the Song/Mukesh convention: if a contrast has observed data and the
shared MEA path can build a ranked site vector, MEA is run. Low sample counts
remain visible as raw `n_wt` / `n_tg` evidence, not as a viewer-facing
under-replication gate.

The shared MEA behavior remains:

- drop missing LFC rows before ranking
- drop sites without kinase-library-compatible motifs
- median-center each contrast's LFC vector
- winsorize at the 1st and 99th percentiles
- skip contrasts only when the ranked site count is below `MEA_MIN_SITES`

These operations are inherited from `alz.bulk_mea.enrich._run_mea`, which is also
used by the Song/Mukesh kinase workflows.

## Viewer Behavior

The Mouse (5xFAD) kinase tab should display the same kinase-viewer structure as
the Song/Mukesh tabs where the data support it. The viewer does not display an
`under_replicated` status or gray out 6-month cells. For the current rebuilt MEA
outputs, all 5xFAD tissue/assay/track combinations contain 3, 6, 9, and 12 month
TG-vs-WT contrasts.

Viewer-facing site labels are compact gene-site labels such as `Atp1a3_S456`
while preserving the original `site_id` in hover/title metadata and detail
sidecars.

