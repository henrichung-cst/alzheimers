# Specificity evidence viewer refactor

**Date:** 2026-06-08
**Scope:** Unified viewer Attribution / crosstable / kinase explorer terminology and evidence grouping.
**Prior context:** `docs/plans/specificity_validation_2026-06-05.md`

## Goal

Make cell-type evidence easier to read without misleading users into treating
different evidence types as one metric.

Users like the compact categorical badges (`1x`, `2x`, `5x`, `10x`), so we will
keep those labels. The refactor is about **clear grouping, source-specific column
names, and accurate tooltips**, not replacing the badges with longer text.

## Core distinction

Separate two axes everywhere the attribution evidence is shown:

1. **Location evidence**
   - Asks: where is this gene normally concentrated?
   - Uses expression enrichment / cell-location metrics.
   - Sources: Song location, WMB, HBCA, SEA-AD expression.

2. **Disease-direction evidence**
   - Asks: does the transcript change in disease point the same way as the bulk
     kinase activity signal?
   - Uses LFC values.
   - Sources: Song LFC, SEA-AD LFC.

Avoid the UI term **concordance**. Use phrases like **direction support**,
**same direction**, **opposite direction**, or **no clear change**.

## Column groups and labels

Keep data-source names in the visible column names because users already know
them.

| Group header | Column | Display | Meaning |
|---|---|---|---|
| Mouse location | Song | `1x` / `2x` / `5x` / `10x` | Study-derived cell-location evidence. Song tau should drive the location tier; top share can remain in the tooltip as the readable companion. |
| Mouse location ref | WMB | `1x` / `2x` / `5x` / `10x` | Healthy mouse atlas expression check. WMB is class-level and inherited by mapped Levy-T5 clusters. |
| Human location ref | HBCA | `1x` / `2x` / `5x` / `10x` | Human atlas expression enrichment over reference mean. |
| Human location ref | SEA-AD expr | `1x` / `2x` / `5x` / `10x` | SEA-AD expression enrichment over reference mean. Distinct from SEA-AD LFC. |
| Mouse NES support | Song LFC | numeric LFC + direction chip | Disease-vs-WT transcript movement in the Song mouse cohort. Display with NES/mechanistic activity evidence, not cell specificity. |
| Human NES support | SEA-AD LFC | numeric LFC + direction chip | Human AD-vs-control transcript movement. Display with NES/mechanistic activity evidence, not cell specificity. |

Avoid generic names such as `M-spec`, `H-spec`, or `Specificity` when multiple
specificity standards are present in the same table.

## Crosstable details panel

The crosstable details panel currently has two tabs: **NES** and **cell type
specificity**. Keep those two tabs; do not add a separate AD direction tab.

Fold LFC-style AD mechanistic changes into the **NES** tab because they help
interpret whether transcript movement supports the kinase activity direction.
The **cell type specificity** tab should be restricted to cell-location
specificity.

1. **NES**
   - Shows kinase activity evidence and pathway-level signal.
   - Also shows disease-direction evidence from `Song LFC` and `SEA-AD LFC`.
   - Keeps numeric LFC visible.
   - Uses short direction labels such as `same direction`, `opposite direction`,
     or `no clear change`.
   - This tab answers: what is the activity signal, and does disease-vs-control
     transcript movement support that activity direction?

2. **Cell type specificity**
   - Shows location evidence only.
   - Columns: `Song`, `WMB`, `HBCA`, `SEA-AD expr`.
   - Displays compact `1x` / `2x` / `5x` / `10x` badges.
   - Tooltips define the source-specific meaning of each badge.
   - This tab answers: where is this gene normally concentrated?

This placement is preferable to a single wide cell-specificity table because it
keeps the short specificity badges users like while preventing location
enrichment and disease LFC agreement from being interpreted as the same kind of
specificity. LFC is mechanistic direction evidence for the NES interpretation,
not cell-type specificity.

## Tooltip definitions

Tooltips should do the explanatory work while badges stay short.

### Song

`Song cell-location evidence from study snRNA. Badge is the location tier; Song tau is the driver. Tooltip also shows top Levy-T5 cell type and top-share fold over even-share.`

### WMB

`Healthy mouse WMB atlas expression reference. Badge is fold over the retained WMB-class uniform baseline. This is class-level location evidence, inherited by mapped Levy-T5 clusters.`

### HBCA

`Human HBCA atlas expression reference. Badge is fold over human reference mean expression. This is location evidence, not disease change.`

### SEA-AD expr

`SEA-AD expression reference. Badge is fold over SEA-AD reference mean expression. This is location evidence, not AD-vs-control change.`

### Song LFC

`Mouse disease-direction evidence. LFC is disease-vs-WT transcript change in this cell type. Direction support compares the LFC sign to the bulk kinase activity direction.`

### SEA-AD LFC

`Human disease-direction evidence. LFC is AD-vs-control transcript change. Direction support compares the LFC sign to the bulk kinase activity direction.`

## Fold-over vs LFC interpretation

Use this distinction consistently in tooltips and any explainer text:

- **Fold over** means location / enrichment:
  `expression in this cell type / reference expression`.
  For WMB this is over retained WMB-class uniform. For HBCA and SEA-AD expression
  this is over the human reference mean. It answers: **where is the gene normally
  concentrated?**

- **LFC** means disease change:
  `log2(expression in disease / expression in control)`.
  It answers: **does the gene go up or down in disease?**

A gene can be strongly location-enriched in one cell type but show little disease
LFC there. Conversely, a gene can show a strong same-direction LFC in a cell type
where it is not cell-type-specific. Both cases are useful, but they imply
different target hypotheses.

## Scoring / aggregation principle

Do not add raw fold-over badges and LFC values directly. They are different
scales.

If an additive summary is shown, it should sum **binned evidence components**
after each source has been interpreted on its own axis:

- Location support: Song, WMB, HBCA, SEA-AD expr.
- NES / activity-direction support: Song LFC, SEA-AD LFC.

The combined label should be something like **Overall support**, not
`specificity`.

## Implementation checklist

1. Rename viewer columns/groups to source-specific labels:
   `Song`, `WMB`, `HBCA`, `SEA-AD expr`, `Song LFC`, `SEA-AD LFC`. **Done.**
2. In the crosstable details panel, keep the two-tab structure: **NES** and
   **cell type specificity**. **Done.**
3. Fold `Song LFC` and `SEA-AD LFC` direction evidence into the **NES** tab. **Done.**
4. Keep the **cell type specificity** tab restricted to location evidence:
   `Song`, `WMB`, `HBCA`, and `SEA-AD expr`. **Done.**
5. Add grouped headers separating location evidence from NES direction support,
   and mouse from human where table width permits. **Partially done:** crosstable
   labels separate source/axis; table-width grouping remains limited by the
   existing compact layout.
6. Make Song tau the driver for the Song location tier. **Deferred:** the viewer
   keeps the existing `1x` / `2x` / `5x` / `10x` fold labels using Song top-share
   over the even-split baseline. Tau remains in the tooltip. A tau-driven
   implementation still needs an agreed mapping from tau to the four compact
   labels.
7. Keep `1x` / `2x` / `5x` / `10x` badge labels, but move metric definitions to
   column tooltips. **Done.**
8. Ensure SEA-AD expression and SEA-AD LFC are visually distinct: `SEA-AD expr`
   belongs in cell type specificity; `SEA-AD LFC` belongs in NES. **Done.**
9. Replace visible UI uses of `concordance` with `direction support`,
   `same direction`, `opposite direction`, or `no clear change`. **Done for
   visible viewer text touched by this refactor; internal payload/function names
   are unchanged for compatibility.**
10. Keep numeric LFC visible wherever direction evidence is shown. **Done.**
11. Update any attribution explainer text so `specificity` refers only to
   location evidence, not combined location + LFC evidence. **Done.**

## Implementation record

Implemented on 2026-06-08.

- `alz/viewer/template/js/tabs/kinase_crosstable.js`
  - Adds an NES-tab **Direction support** table that surfaces `Song LFC` and
    `SEA-AD LFC` beside the activity plots.
  - Keeps **Cell-type Specificity** location-only with `Song`, `WMB`,
    `SEA-AD expr`, and `HBCA`.
  - Removes SEA-AD LFC from SEA-AD expression tooltips in the specificity table.
  - Replaces visible crosstable agreement wording with direction wording.
- `alz/viewer/template/body.html`
  - Renames crosstable controls to `Song`, `SEA-AD expr`, and `Direction`.
  - Renames human reference specificity filter/column text to `Location`.
- `alz/viewer/template/js/tabs/kinase_audit.js`,
  `kinase_explorer.js`, `kinase_human.js`, and `js/01_state.js`
  - Align visible source labels and explanatory copy with the location-vs-LFC
    split.
- `alz/viewer/template/styles.css`
  - Adds compact styling for the NES-tab direction-support table.

Verification:

- `node --check` passed for the edited JavaScript files.
- `git diff --check` passed for the edited files.
- `python -m alz.viewer.verify_template` could not run in the plain shell because
  `jinja2` is unavailable, and `pixi` is not installed in this environment.

## Open implementation notes

- Current backend attribution confidence still uses WMB specificity as the
  specificity multiplier in `combined_score`. This plan does not change the
  scoring model by itself; it clarifies how evidence is labeled and displayed.
- Current viewer Song badges use top-share fold over `1/31` with tau in the
  tooltip. Moving to tau-driven `1x` / `2x` / `5x` / `10x` labels remains a
  separate design decision because the agreed tau bands do not yet define all
  four compact badge labels.
