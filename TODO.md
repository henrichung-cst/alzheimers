# TODO

Open items grouped by theme. Add `[DONE <date>]` prefix when complete; move to archive at bottom.

---

## A. T-Cell Analysis & Viewer

**A1. T-cell specificity / enrichment terminology and metrics**
Formalize language in the T-cell viewer: "specificity" = reference cell-type presence from NSCLC data (per-kinase: how many of the 7 NSCLC cell types express it at ≥10% detection); "enrichment" = concentration in a T-cell state via ProjecTILs. Both metrics must be surfaced and clearly labeled. The goal is to directly measure whether a kinase is assigned to a specific cell state and what substrates define that state.
- Report per-kinase: (a) number of NSCLC cell types expressing it, (b) fraction of cells per type expressing it, (c) fraction of total expression in each type.
- Report ProjecTILs state enrichment on both our scRNA and the NSCLC reference.
- Add breakdown/sort by CD4, CD8, Treg.

**A2. T-cell attribution: replace "vs bulk" with NES deconvoluted**
In tcell attribution view, the current "vs bulk" label/calculation should be replaced with the NES deconvoluted value directly.

**A3. Incytr trends over all 3 timepoints (T-cell)**
Double-check that incytr trends are computed over all 3 T-cell timepoints, not a subset.

**A4. Verify T-cell data structure matches description**
Confirm the pipeline's understanding of the T-cell data matches:
- donor1 = IMAC data; donor2 = no IMAC data
- Both have total proteome + pY data
- Skip kinase MEA for donor2
- Different timepoints (not directly comparable, same experiment parameters)
- Both have scRNA; run incytr on both (on deconvoluted); run kinase MEA on bulk

**A5. Script: monotonically increasing/decreasing kinases in T-cell data**
Write a small, rarely-used export script for the T-cell data that outputs all kinases and substrates that are monotonically increasing or decreasing across timepoints.

---

## B. Incytr Improvements

**B1. IncytrDB audit — database provenance and versioning**
Locate the IncytrDB database backing incytr pathway calculations. Determine: (a) how Changhan maintains it, (b) how up to date it is, (c) confirm we are using the correct mouse vs. human version per dataset. (Email sent to Changhan — follow up.)

**B2. Incytr pathway visualization → sankey/chord diagram**
Replace or supplement the current incytr pathway heatmap with a sankey or chord diagram that emphasizes cell-type connections over time. Collapse some excitatory neuron clusters to reduce visual complexity. (Also see B5 for backbone filtering before building this.)

**B3. Incytr with acetylation and ubiquitination**
Expand incytr to include acetylation and ubiquitination PTM data, handled identically to phosphorylation (minimal code changes — data extension, not new logic). Generate new results for song, 5xfad, and tcell **without overwriting** existing phospho-only outputs. Store in a parallel output directory (e.g., `wide_ptm/` pattern already used for 5xFAD).

**B4. Kinase → incytr pathway integration**
Insert kinase activity data into incytr pathways based on substrate activity. For the high-confidence substrates of each active kinase, identify which genes are those substrates, then link them into the pathway graph. Enrich with cell-type and timepoint information so pathway edges can be gated by expression and disease context.

**B5. Incytr pathway backbone / reduction**
Develop a pathway reduction strategy:
- Collapse paths by what is common across all timepoints first, then conditions (Cholinergic is a priority anchor).
- Rank by backbone (core L→R→EM→Target path, not just the widest enumeration).
- Add a specificity filter on pathway genes (e.g., mean specificity of member genes).
- Consider a min-cell-count weight on log fold changes.

**B6. Compare incytr on mukesh vs SEA-AD**
Run and compare incytr results on the mukesh human AD cohort and the Allen SEA-AD atlas. If the results are concordant, that would be a notable positive finding worth reporting.

---

## C. Cross-Cohort & Disease Interpretation

**C1. Song viewer: split into 3 sub-cohorts (App, Tau, ApTt)**
Song data contains 3 underlying AD genotypes. Expose them as separate sub-cohorts in the viewer where meaningful. Rule: cell-type specificity is NOT split (whole-cohort reference stays unified); NES and pathway results ARE split per genotype.

**C2. Cohort naming convention**
Standardize cohort display names away from researcher last names (Song, Mukesh) toward a biologically descriptive convention. Candidates: `MouseCohortApp / MouseCohortTau / MouseCohortApTt`, or `MouseAD1 / MouseAD2`, or similar. Decide and apply consistently across viewer labels, axis titles, and export filenames.

**C3. Disease-direction focus: up/down ranking and early markers**
Priority analytic lens for AD cohorts:
- Show kinase/substrate changes as directional (up vs. down) per genotype column (App / Tau / ApTt), not only as agreement grids.
- Ranking is critical — surface what has the biggest changes.
- Identify phosphosites that change early ("suspicious" / early-change phosphosites).
- Flag proteins that can be secreted into CSF or blood as potential early diagnostic markers.
- For a given gene list: show which are also present in AD data (late-stage), with LFC column.

**C4. Cross-species specificity guard**
Operational constraint to enforce in review: never finalize a target where a kinase is assigned to a single mouse cell type but the human reference shows it expressed across multiple cell types. Check human cell-type specificity (SEA-AD or NSCLC) for every candidate before reporting.

**C5. Compare mukesh 50-kinase overlap with 5xFAD (substrate direction)**
Take the 50 kinases with agreement between mukesh human AD, human preclinical, and human controls. In the 5xFAD mouse dataset, check: (a) shared direction (up/down), (b) substrate composition overlap (Venn), (c) substrate-direction correlation. Flag kinases where substrate agreement breaks down between mouse and human — disagreements are biologically interesting.

---

## D. Substrate Analysis

**D1. Cross-cohort substrate phosphosite comparator**
Build a substrate comparator tool: select a pool of kinases from one cohort (e.g., most active in mukesh human AD), select another pool from a second cohort, and compare underlying substrate composition (overlap, direction). Initial use case: GRF1-5 family — what separates them in terms of substrate specificity?

---

## E. Kinase Hierarchy & Family Discrimination

**E1. Kinase upstream/downstream regulation network**
Using the PhosphoSite Kinase Library (check if access is already available in-repo; fetch if not), build a kinase regulation network. Two layers:
1. Reference hierarchy: "if kinase A is upregulated, it should cause downregulation of downstream kinase B."
2. Observed overlay: what actually happens in the disease phenotype (both A and B upregulated? concordant? discordant?).
Viewer goal: click a kinase → show its reference regulation neighbors → overlay observed disease-direction arrows. Filter to kinases co-expressed in the same cell type.

**E2. Discriminating kinases within the same family**
Use snRNA cell-type specificity to separate kinases in the same family. Hypothesis: different members of a kinase family target different cell types — use that as a discriminator when MEA substrate overlap makes activity-level separation difficult.

---

## F. Viewer & UX

**F1. Sort all tables by signed values (no absolute value)**
Across all viewer tables (crosstable, kinase explorer, incytr pathways, and any others): sort by signed NES/PDS, not |NES|/|PDS|. Crosstable currently shows explicit sign but sorts by absolute value — fix to sort signed. End-user request; apply uniformly.

**F2. Standardize CSV export across all tables**
Audit all viewer tables and ensure export-to-CSV is available and consistent. Currently partially implemented; complete the remaining tables and standardize output format (CSV, sensible column names, no UI-only columns in the export).

---

## G. Documentation & Methods

**G1. Methods and workflow documentation sweep**
Update methods documentation for both the AD and T-cell pipelines. Add workflow diagrams (pictures / flowcharts) showing each step. This is a required deliverable for collaborators and publications — not a polish item.

**G2. Positive controls list**
Build and maintain a list of positive controls for each data cohort:
- AD cohort: APOE (well-understood AD mechanisms).
- Candidates to audit: PHKG1 (expected astrocyte-specific?), ATP9A (expected endothelial?).
- T-cell exhaustion cohort: TBD.
Use these to validate that specificity and enrichment metrics behave as expected.

---

## H. External Data & Extensions

**H1. TMT paper replication / IMAC fetch**
Identify the TMT-based paper that analyzed Song-like data with a different normalization strategy. Fetch their IMAC data and run kinase enrichment to compare results. Note whether their normalization removed heterogeneity relative to ours. This is an exploratory cancer extension.

---

## Archive — Completed

**[DONE 2026-06-19] NSCLC 10x reference for T-cell data**
10x Flex dataset (897,733 cells × 18,082 genes) annotated via ProjecTILs + marker lineages. 77,760 T cells in 14 states; non-T → 7 TME lineages. Two metrics per (kinase, cell_type): mean_log2(CPM+1) + fraction expressing. Audit: 79/339 covered kinases expressed nowhere in TME (tissue-restricted families = MEA false-positive candidates). Wired into viewer as NSCLC attribution tier. Commits: 1f2fa8e, dbae990. Plan: `docs/plans/todo2_tcell_specificity_reference.md`.

**[DONE 2026-06-19] Incytr on 5xFAD**
All 8 contrasts (cortex/hippocampus × 3/6/9/12mo) complete in `wide/` and `wide_ptm/`. One driver fix: `null_if_empty()` for optional PTM assays with no samples at a given timepoint. Viewer wired as `fivexfad_cortex` and `fivexfad_hippocampus` contexts. Plan: `docs/plans/todo6_incytr_on_5xfad.md`.

**[DONE 2026-06-19] Unified viewer scaling audit**
P1–P8 implemented: attribution-summary + celltype-MEA sidecars, payload 105→53 MB raw / 10.1→5.63 MB gzip. Crosstable lazy init, gene_node_index per-context sidecar, LRU cap on 5xFAD caches, T-cell viewer sidecar payload mode. Plan: `docs/plans/todo8_unified_viewer_scaling_audit.md`.
