# Crosstable 5xFAD Crossplay Integration Plan

Date: 2026-06-17

## Goal

Integrate 5xFAD cortex and hippocampus into the unified viewer Crosstable so the
tab supports pan-AD comparison across Song, Mukesh, and 5xFAD datasets. The
change should preserve Mukesh CTRL donors as visual reference only and avoid
adding unrelated AD-vs-CTRL stability filters.

## Current Context

- The target tab is the existing `Crosstable` implemented in
  `alz/viewer/template/js/tabs/kinase_crosstable.js`.
- The tab currently compares Song mouse kinase activity with Mukesh human AD
  donor kinase activity.
- Mukesh CTRL donors are rendered as a visually separated reference strip and
  are not part of agreement filtering.
- 5xFAD data already exists in `payload.supporting_5xfad` as a supporting AD
  cohort with cortex and hippocampus bulk MEA rows, attribution summaries,
  cell-type agreement summaries, and lazy detail shards.

Relevant graphify navigation communities:

- `Viewer Kinase Crosstable`
- `Crosstable Agreement Design`
- `Viewer Specificity Evidence Refactor`
- `5xFAD Lucie Manifest`

## Dataset Status Model

Represent each disease dataset as a categorical status at the active FDR:

- `song_status`: `sig_up`, `sig_down`, `nonsig_measured`, `missing`
- `mukesh_status`: AD donors only; CTRL donors remain visual reference
- `fivexfad_cortex_status`: cortex-only 5xFAD status
- `fivexfad_hippocampus_status`: hippocampus-only 5xFAD status
- `fivexfad_status`: aggregate 5xFAD status over cortex and hippocampus

The 5xFAD aggregate status is categorical:

- cortex and hippocampus significantly agree up: `sig_up`
- cortex and hippocampus significantly agree down: `sig_down`
- one tissue is significant and the other is nonsignificant or missing: the
  significant tissue's direction
- cortex and hippocampus are significant in opposite directions: `mixed_sig`
- neither tissue is significant but at least one is measured: `nonsig_measured`
- neither tissue is measured: `missing`

Do not introduce a numeric crossplay score. Agreement should be categorical and
traceable to raw NES/FDR evidence.

## Tile Layout

Extend each Crosstable master row with 5xFAD tiles adjacent to the Mukesh data:

- Song mouse glyph
- Mukesh AD donor strip
- Mukesh CTRL donor strip, still visual reference only
- 5xFAD cortex tile row: 1x4, ordered 3/6/9/12 months
- 5xFAD hippocampus tile row: 1x4, ordered 3/6/9/12 months

The 5xFAD tissue rows should reuse the existing 5xFAD visual convention:

- red/blue encodes NES sign
- saturation encodes NES magnitude
- outline marks FDR below the active threshold
- tooltip reports tissue, age, NES, FDR, and significant/non-significant status

## Reimagined Comparison Filters

Replace the current simple direction filter with comparison-oriented controls.

### Comparison Scope

Supported scopes:

- `3-way`
- `Song vs Mukesh`
- `Song vs 5xFAD`
- `Mukesh vs 5xFAD`
- `5xFAD tissue split`

### Agreement State

Supported states:

- `Any`
- `significant same direction`
- `same direction allowing nonsignificant measured`
- `opposite significant direction`
- `source-only`
- `mixed_sig`
- `missing one side`

For `3-way`, support:

- `3-way significant agreement`: Song, Mukesh AD, and aggregate 5xFAD are all
  significant and same direction.
- `3-way directional agreement`: all measured groups point in the same direction;
  nonsignificant measured groups are allowed.
- `3-way discordance`: at least two significant groups oppose each other.
- `5xFAD mixed_sig`: cortex and hippocampus significantly disagree.

For `5xFAD tissue split`, support:

- cortex-only significant
- hippocampus-only significant
- cortex and hippocampus same significant direction
- cortex and hippocampus opposite significant direction, exposed as `mixed_sig`

Do not add an `AD-specific vs CTRL-stable` filter or any other new Mukesh CTRL
comparison feature.

## Age Handling

Do not add a normal age filter in the first implementation.

Age only applies to 5xFAD, not Song or Mukesh. A normal age dropdown would look
global while only changing one dataset's status derivation. Instead:

- always render all four 5xFAD age tiles
- derive each 5xFAD tissue status from significant age evidence at the active
  FDR
- expose age-level details through tile tooltips and the detail panel

If future users need age scoping, add it later as an explicitly labeled
`5xFAD age rule` rather than a global-looking age filter.

## Detail Panel

Keep row selection as the only interaction model unless equivalent click behavior
exists for Song and Mukesh glyphs. Do not make individual 5xFAD tissue/age tiles
clickable.

For a selected kinase, extend the detail panel with:

- Song activity evidence
- Mukesh AD activity evidence plus CTRL reference strip
- 5xFAD cortex age profile and raw age rows
- 5xFAD hippocampus age profile and raw age rows
- 5xFAD cortex/hippocampus aggregate status and `mixed_sig` explanation when
  applicable
- 5xFAD cell-type attribution and bulk-vs-decomposition agreement summaries
  where compact payload evidence is already available

## Implementation Targets

- `alz/viewer/template/js/tabs/kinase_crosstable.js`
  - add 5xFAD indexes
  - derive tissue and aggregate statuses
  - render cortex/hippocampus tile rows
  - replace agreement filtering logic
  - extend detail rendering

- `alz/viewer/template/body.html`
  - update Crosstable toolbar controls
  - update table column layout for 5xFAD tissue tile rows

- `alz/viewer/template/js/tabs/kinase_fivexfad.js`
  - reuse helper logic where practical, especially 5xFAD tile/status behavior

- `alz/viewer/template/js/01_state.js`
  - update Crosstable how-to text for the three-dataset comparison model

- `docs/foundation/viewer_payload_contract.md`
  - update only if implementation requires a new compact Crosstable-specific
    5xFAD index

## Verification

After implementation:

- run the viewer template verifier
- run the payload contract verifier if payload schema or emitted payload fields
  change
- rebuild the unified viewer
- smoke-test Crosstable in browser with:
  - 3-way significant agreement
  - pairwise Song vs Mukesh
  - pairwise Song vs 5xFAD
  - pairwise Mukesh vs 5xFAD
  - 5xFAD `mixed_sig`
  - missing-data rows
- confirm Mukesh CTRL donors remain visual reference only and do not affect
  agreement filters
