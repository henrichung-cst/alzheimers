# Kinase Viewer Design Brief

Status: design handoff brief.
Audience: frontend/product designer reviewing the kinase tab UX.
Implementation surface: `code/build_unified_viewer.py`, generated to `outputs/reports/unified_viewer/index.html`.

## Intent

The kinase viewer is not meant to be a simple ranked list. Its purpose is to let a reader audit a kinase-level claim from the top-level interpretation down to the underlying source rows that produced it.

The core user question is:

> Why is this kinase being highlighted, and which exact rows support that claim?

The viewer should therefore support two modes at once:

1. Lightweight browsing across all kinases.
2. A traceable audit path for a selected kinase.

The current redesign keeps the ranked kinase table as the entry point, then turns the right-hand detail area into a kinase audit workbench.

## Primary User Tasks

The frontend should make these workflows feel direct:

1. Find a kinase by rank, name, family, gene, contrast activity, cell-type attribution, or backbone breadth.
2. Select a kinase and immediately understand its high-level summary: NES trajectory, significant contrasts, top cell type, backbone count, and composite score.
3. Inspect the exact MEA rows that support the kinase in stoichiometry-normalized and raw phospho analyses.
4. Drill from leading substrates into site-level OLS rows.
5. Trace one phosphosite/sample number through normalized phospho and stoichiometry matrices.
6. Inspect cell-type attribution rows and final hypothesis table rows.
7. Open the catalog of source tables, preview them, and export filtered or full source data.

## Information Architecture

The kinase tab is organized as a two-column workspace.

### Left Column: Kinase Browser

The left column remains the fast scanning surface:

- Search by kinase or gene.
- Sort by kinase name, family, gene, significant contrast count, peak NES, top cell type, attribution confidence, backbone count, or composite score.
- Click a row to pin a kinase and populate the audit workbench.

Design intent:

- Keep this column dense and efficient.
- It is a selector and ranking tool, not the full evidence explanation.
- Avoid turning it into a dashboard of every possible field.

### Right Column: Kinase Audit Workbench

The selected kinase opens a sequence of evidence tabs. The first tab should be kinase-relevant data, not setup metadata. Sample mapping stays available through Source Tables and method/context copy because it explains sample columns but does not itself support a kinase-level claim.

1. **Summary**
   - Kinase name, family, gene, trajectory label, selected audit contrast.
   - NES mini-trajectory across disease contrasts.
   - Composite score, significant contrast count, peak NES, top cell type.

2. **Measurement Trace**
   - First active walkthrough tab after selecting a kinase.
   - Shows the selected kinase/contrast leading-substrate sites for one selected sample column.
   - Places raw phospho, raw parent protein, IRS-normalized phospho, IRS-normalized parent protein, log2 transforms, and final stoichiometry side by side.
   - Makes the formula explicit: `stoichiometry = log2(IRS phospho) - log2(IRS parent protein)`.

3. **OLS Details**
   - Parses `Leading substrates` for the selected kinase and contrast.
   - Joins motifs/sites to `site_level_ols.csv`.
   - Shows one row per phosphosite substrate, not one row per sample.
   - Prioritizes the selected contrast's model-derived fields: stoichiometry LFC, stoichiometry p-value, stoichiometry FDR, raw phospho LFC, and raw phospho FDR.
   - Shows `n_obs_stoich` as the total number of biological sample columns with usable stoichiometry values for that site; this is a site-level availability count, not a contrast-specific estimate.
   - Keeps `matched_protein` visible so users can see whether stoichiometry correction was possible for the site.

4. **MEA Preparation**
   - This is the bridge between the site-level OLS table and the final kinase-level MEA score.
   - Shows how selected-contrast OLS site effects are prepared as the ranked substrate input for motif enrichment.
   - Answers which kinase substrate sites enter MEA, what each site's OLS value was, whether median-centering/global shift was applied, whether any value was winsorized, and what value/rank was passed forward when available.
   - Includes leading substrate membership plus preprocessing receipts such as `mea_global_shift.csv` and `winsorized_sites.csv`.

5. **MEA Score**
   - Side-by-side source rows from:
     - `mea_stoichiometry.csv`
     - `mea_raw_phospho.csv`
   - Shows `ES`, `NES`, `p-value`, `FDR`, `Subs fraction`, and `Leading substrates`.
   - Includes the NES mini-trajectory across disease contrasts.

6. **Attribution**
   - Filtered rows from `unified_attribution.csv`.
   - Lets the user inspect the exact cell-type attribution records.
   - Includes compact cell-type evidence and existing backbone support.

7. **Final Hypothesis**
   - Rows pulled from:
     - `kinase_activity_matrix.csv`
     - `celltype_evidence_table.csv`
     - `kinase_hypothesis_table.csv`
   - A `source` field identifies which table each row came from.

8. **Source Tables**
   - Catalog of all audit source files.
   - Shows raw path, viewer-relative path, row/column counts, searchable columns, previews, and export controls.
   - Includes `sample_mapping.csv` for users who need to decode sample/channel columns.

## Data Traceability Model

Every displayed claim should have a visible route to source rows.

The route is:

`ranked kinase row`
→ `summary metrics`
→ `leading substrates`
→ `measurement trace rows`
→ `OLS details rows`
→ `MEA preparation receipts`
→ `MEA score rows`
→ `attribution rows`
→ `final hypothesis tables`
→ `source table catalog`

## MEA Preparation Stage

The MEA Preparation tab should sit immediately after OLS Details and immediately before MEA Score.

Conceptually, this stage answers:

> How did the per-site OLS effects become the ranked substrate input used to score this kinase?

The user has just seen one row per phosphosite in OLS Details. They have not yet seen a kinase-level NES. Before showing that NES, the viewer should make the transformation into MEA input explicit.

Recommended tab order:

1. **Measurement Trace**
   - Sample-level numeric receipt.
   - Raw phosphosite/protein values, IRS-normalized values, log2 values, and stoichiometry.

2. **OLS Details**
   - Site-level contrast model.
   - One row per phosphosite with selected-contrast stoichiometry/raw phospho LFC, p-value, and FDR fields.

3. **MEA Preparation**
   - Transformation from site-level effects into MEA-ready substrate rankings.
   - Shows leading substrate membership, global-shift/median-centering receipts, winsorization receipts, and final prepared site values where available.

4. **MEA Score**
   - Kinase-level enrichment result.
   - Shows the source MEA rows, including ES, NES, p-value, FDR, substrate fraction, and leading substrates.

The MEA Preparation tab should not feel like a loose collection of source tables. It should read as a calculation bridge:

`OLS site effect`
→ `median-centered/global-shift adjusted value`
→ `winsorized/clipped value if applicable`
→ `ranked MEA input`
→ `leading substrate membership`

If the current source files do not expose every intermediate value side by side, the UI should still show the available receipts and label gaps plainly rather than implying unrecorded transformations are present.

Useful table groups for this tab:

- **Leading Substrate Membership**
  Parsed from the selected kinase/contrast `Leading substrates` field. This tells the user which sites are relevant to the kinase-level enrichment result.

- **Global Shift / Median-Centering Receipt**
  Filtered rows from `mea_global_shift.csv`, ideally by selected contrast and analysis mode. This explains whether the input distribution was centered or shifted before enrichment.

- **Winsorized Sites**
  Filtered rows from `winsorized_sites.csv`, ideally by selected contrast and selected substrate sites. This identifies site values that were clipped to reduce outlier leverage.

- **Prepared MEA Input**
  If available from source files or derivable without changing the analysis, show the final site-level value/rank that was passed to MEA. If not available, make that absence visible and keep exports/source-table links nearby.

This traceability is more important than visual novelty. A redesigned UI should not hide source rows behind vague summaries.

## Audit Table Behavior

Audit tables use a shared `AuditTable` component. A design pass should preserve these capabilities:

- Search rows.
- Sort columns.
- Pagination or virtualized rendering for large tables.
- Sticky headers.
- Numeric alignment.
- Cell-level copy/readability via full-value tooltips.
- Export current filtered rows.
- Export full source table.
- Preserve raw column names in exports by default.
- Allow clean-header export as an explicit option.

Every table header must expose:

- Display label.
- Raw column name.
- Short definition.

This is essential because the viewer is used to audit analysis outputs where raw column names matter.

## File Loading Model

The generated HTML embeds only metadata and small previews for audit tables. Full source tables are copied beside the viewer under:

`outputs/reports/unified_viewer/audit_sources/`

When opened via `file://`, browser restrictions may block full lazy loading. In that mode, the UI should:

- Keep summary browsing usable.
- Show embedded previews.
- Clearly explain that full audit table loading requires serving `outputs/reports/unified_viewer/` over HTTP.

When served over HTTP, full CSV/JSON source tables should load on demand.

## Tables in the Audit Manifest

The current audit manifest covers:

- `sample_mapping.csv`
- `normalization_summary.json`
- `raw_phospho_normalized.csv`
- `stoichiometry_matrix.csv`
- `site_level_ols.csv`
- `mea_global_shift.csv`
- `winsorized_sites.csv`
- `mea_stoichiometry.csv`
- `mea_raw_phospho.csv`
- `unified_attribution.csv`
- `kinase_activity_matrix.csv`
- `celltype_evidence_table.csv`
- `kinase_hypothesis_table.csv`

The UI should make this catalog feel intentional, not like a debug dump. The user should understand which table answers which audit question.

## Design Priorities

Prioritize:

- Dense but readable analytical layout.
- Clear hierarchy between summary, evidence, and raw sources.
- Fast switching between kinases.
- Strong source provenance.
- Tables that are usable with long scientific column names and long substrate lists.
- Stable layout that does not jump when tables load.
- Visual cues for “summary metric” versus “source row”.

Avoid:

- Marketing-style hero sections.
- Large decorative cards that reduce data density.
- Hiding source rows behind modal-only interactions.
- Renaming raw columns in a way that breaks traceability.
- Making the composite score look like a statistical confidence measure.
- Treating the audit source catalog as secondary or optional.

## Current UX Pain Points to Optimize

Areas a frontend designer should evaluate:

- The right-side audit workbench is information-rich and may need stronger visual grouping.
- MEA evidence and substrate drilldown need clear “this row came from this file” affordances.
- Long `Leading substrates` strings can dominate table cells and may need expansion/collapse behavior.
- The Number Trace panel should read like a calculation receipt, not just another table.
- Source Tables should help users choose the right table by purpose, not only by filename.
- The file-versus-HTTP loading notice should be visible but not disruptive.
- Export controls should be easy to find without making every panel feel toolbar-heavy.

## Non-Negotiable Constraints

- Do not alter analytical results.
- Do not hand-edit generated `index.html`; change the builder.
- Keep existing pathway/backbone edge slices unchanged.
- Keep summary kinase browsing usable under `file://`.
- Preserve raw column names for provenance and exports.
- Preserve all audit source table access.

## Success Criteria

A successful redesign should let a scientist do the following without leaving the kinase tab:

1. Select `AKT1`.
2. See why it ranks highly.
3. Identify which contrasts drive its NES signal.
4. Inspect stoichiometry and raw-phospho MEA rows.
5. Move from leading substrates to site-level OLS evidence.
6. Pick a site/sample and see the numeric trace.
7. Inspect attribution and final hypothesis rows.
8. Export the relevant filtered rows or full source tables.

The ideal experience feels like an audit trail with good visual hierarchy, not a collection of unrelated tables.
