# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stoichiometry-corrected kinase attribution pipeline for Alzheimer's disease phosphoproteomics. Integrates 72-animal TMT total proteome with phosphoproteomics to compute stoichiometry (log2 phospho − log2 protein), runs MEA (GSEA-based) kinase enrichment on stoichiometry β values, and attributes findings to cell types using unified evidence from SEA-AD transcriptomic concordance and WMB expression specificity. Primary analysis uses males-only (33 animals after outlier exclusion) to avoid hormonal confounds; full-cohort analysis is run as a sensitivity check.

The project pivoted from direct cell-type deconvolution (which failed validation) to this stoichiometry-corrected approach. See `docs/foundation/analysis_charter.md` for the authoritative scope definition and `docs/foundation/analysis_rationale.md` for the pivot logic.

## Environment Setup

Requires Python 3.11 (kinase-library compatibility). The active environment is `alzheimers` (managed by micromamba), which includes both `kinase-library` and `anndata`:
```bash
micromamba create -n alzheimers python=3.11 -y
micromamba activate alzheimers
mamba install kinase-library natsort pandas numpy matplotlib seaborn scipy requests anndata gseapy scikit-learn
```

## Running the Analysis

All scripts run from the repo root.

### Live Pipeline

The bundled front door runs all three stages in order:
```bash
bash code/runners/main/run_live_pipeline.sh
```

The **dual-track runner** runs males-only (primary) and full-cohort (sensitivity) analyses sequentially:
```bash
bash code/runners/main/run_dual_analysis.sh
```

Or run individual stages:
```bash
bash code/runners/main/run_data_ingest.sh          # data ingestion
bash code/runners/main/run_kinase_attribution.sh    # kinase attribution
bash code/runners/main/run_attribution_recovery.sh  # attribution recovery
```

**Data ingestion and characterization** (`data_ingest.py`):
```bash
python code/data_ingest.py --mapping       # §1: TMT channel-to-animal sample mapping (72 animals, 6 plexes)
python code/data_ingest.py --phospho-match # §2: Phosphosite-to-protein matching (91.7% match rate)
python code/data_ingest.py --markers       # §3: Cell-type marker protein assessment (WMB atlas, requires wmb_expression --proteome)
python code/data_ingest.py --quality       # §4: Data quality (PCA, batch effects, missingness)
python code/data_ingest.py --outliers      # §5: Statistical outlier detection (within-group robust z-scores)
python code/data_ingest.py --run           # All steps in order
python code/data_ingest.py --summary       # Print cached results
```

**Stoichiometry, MEA enrichment, and unified attribution** (`kinase_attribution.py`):

The `ANALYSIS_MODE` environment variable controls sample filtering (default: `males_only`):
```bash
python code/kinase_attribution.py --normalize          # Stage 1: IRS cross-plex normalization + stoichiometry (all 72 samples)
ANALYSIS_MODE=males_only python code/kinase_attribution.py --enrich   # Stage 2: OLS + MEA (males only, outliers excluded)
ANALYSIS_MODE=males_only python code/kinase_attribution.py --attribute # Stage 3: Unified cell-type attribution
python code/kinase_attribution.py --mechanism-annotation # Optional: raw phospho MEA + mechanism classification
python code/kinase_attribution.py --run                # All stages 1-3 in order
python code/kinase_attribution.py --summary            # Print cached results
```

**Final attribution-table assembly** (`attribution_recovery.py`):
```bash
python code/attribution_recovery.py --kinase-profiles   # S3: Kinase activity matrix + hypothesis table
python code/attribution_recovery.py --celltype-profiles # S4: Cell-type evidence table + kinase profiles
python code/attribution_recovery.py --hypothesis-tables # S3+S4: Run both new hypothesis table steps
python code/attribution_recovery.py --run               # All steps in order
python code/attribution_recovery.py --summary           # Print cached results
```

### Supporting Prerequisites

These must be run before the live pipeline if their outputs don't exist:

```bash
# External Atlas Data Acquisition (SEA-AD + WMB + Aging Mouse from Allen Institute)
python code/atlas_reference.py --aging     # Aging Mouse Brain Atlas
python code/atlas_reference.py --sea-ad    # SEA-AD MTG (139 supertypes, human AD snRNA-seq)
python code/atlas_reference.py --wmb       # Whole Mouse Brain characterization
python code/atlas_reference.py --mapping   # Cross-atlas taxonomy mapping
python code/atlas_reference.py --coverage  # Kinase gene coverage report
python code/atlas_reference.py --run       # All steps in priority order
python code/atlas_reference.py --summary   # Print cached results
# Runner: bash code/runners/supporting/run_atlas_reference.sh

# WMB Expression Export (required for unified attribution + marker assessment)
python code/wmb_expression.py --run       # Compute WMB per-cell-type kinase expression matrix
python code/wmb_expression.py --proteome  # Compute proteome-wide WMB expression (for --markers)
python code/wmb_expression.py --summary   # Print cached results
# Runner: bash code/runners/supporting/run_wmb_expression.sh

# Song snRNA-seq Integration (within-cohort evidence from paired animals)
python code/snrna_integration.py --pseudobulk    # S1: pseudobulk from 170 h5ad (28 animals → 22 subclasses)
python code/snrna_integration.py --specificity   # S2: within-cohort expression specificity
python code/snrna_integration.py --concordance   # S3: within-cohort transcriptomic concordance (males-only OLS)
python code/snrna_integration.py --run           # All stages in order
python code/snrna_integration.py --summary       # Print cached results
# Runner: bash code/runners/supporting/run_snrna_integration.sh
```

### Supplementary Diagnostics

Reviewer-response analyses that validate pipeline choices. Run after the main pipeline:

```bash
bash code/runners/supplementary/run_reviewer_diagnostics.sh   # All diagnostics
# Or individually:
python code/supplementary/fdr_stringent.py --run              # Q4: FDR < 0.10 comparison
python code/supplementary/threshold_sensitivity.py --run      # Q1: Confidence tier sweep
python code/supplementary/aggregation_robustness.py --run     # Q2: Aggregation method comparison
python code/supplementary/parent_protein_qc.py --run          # Q5: Activity-driven parent QC
```

### Standalone Utilities

```bash
python code/map_kinases_to_genes.py       # Kinase→gene symbol mapping
```

### Archived Code (under `archive/`)

All archived scripts have been moved to `archive/code/`. These represent closed paths that failed validation. See `docs/foundation/repo_surface_index.md` for the full classification. Key archived paths:

- **Direct deconvolution**: `sap_data.py`, `sap_model.py`, `sap_validate.py` — cell-type-specific condition effects not identifiable from 24-group design
- **Factor model**: `sap_preflight.py`, `sap_factor_model.py` — parameter reduction didn't overcome composition bottleneck
- **Two-compartment**: `sap_model_2comp.py` — synthetic validation r ≈ 0
- **Transcript-only rescue**: `sap_module1_de.py`, `sap_module2_triangulation.py` — concordance null (permutation p=0.56)
- **Pre-stoichiometry concordance**: `sap_module5b_analysis.py`, `sap_module5c_correlation.py` — mixed abundance/activity signal diluted concordance
- **Kinase enrichment discovery**: `kl_analysis_clusters.py`, `kl_analysis_bulk.py` — superseded by stoichiometry-corrected pipeline

## Architecture

### Live Pipeline

The live analysis is a stoichiometry-corrected MEA enrichment + unified attribution workflow:

1. **data_ingest.py** — Reads 72-animal Song TMT proteomics, maps TMT channels to experimental design, matches 14,772/16,114 phosphosites to parent proteins (91.7%), assesses marker proteins and kinase coverage, runs PCA quality control, detects statistical outliers (within-group robust z-scores)
2. **kinase_attribution.py** — IRS cross-plex normalization (all 72 samples), sample filtering (outlier exclusion + optional males-only), stoichiometry computation (`log2(phospho) − log2(protein)`), factorial OLS per site with disease×timepoint interactions (9 time-resolved contrasts), MEA (GSEA-based) kinase enrichment on median-centered + winsorized stoichiometry β values, unified cell-type attribution combining SEA-AD concordance and WMB expression specificity
3. **attribution_recovery.py** — Cross-contrast consistency analysis and final attribution table assembly
4. **plot_attribution_bubbles.py** — Visualization: per-tissue heatmaps, direction-over-time diverging bars, ApTt additivity scatter, winsorization diagnostic

### Stoichiometry Correction

The core enabling transformation:
```
stoichiometry = log2(phospho) − log2(parent protein abundance)
```
This removes parent-protein abundance confounding and exposes activity-regulated kinase signals.

### Sample Filtering

The pipeline supports two analysis modes controlled by `ANALYSIS_MODE` env var (default: `males_only`):

1. **Outlier exclusion** — `data_ingest.py --outliers` detects outliers using within-group robust z-scores (median + MAD, scaled by 1.4826 for normal consistency). Animals with |z| > `config.OUTLIER_ZSCORE_THRESH` (default 3.0) are flagged in `sample_exclusions.csv`. Applied to all 72 animals regardless of sex.

2. **Sex filtering** — In `males_only` mode, female animals are excluded to remove hormonal confounds on phosphoproteomics. This reduces the design from 72 to ~33 animals (3 per cell after outlier removal), drops the `female` covariate from the design matrix (N×10 instead of N×11), and yields 23 residual DOF.

IRS normalization always uses all 72 samples to maintain a stable per-plex reference. Filtering applies at OLS time (`step_enrich()`), not during normalization.

### Time-Resolved Contrasts

The OLS model uses disease×timepoint interaction terms to produce 9 time-resolved contrasts (3 diseases × 3 timepoints). The design matrix includes: const, App, Tau, Int, [female], time_4mo, time_6mo, App×time4, App×time6, Tau×time4, Tau×time6. Contrasts are defined in `CONTRAST_COEFS` (e.g., `App_4mo = β_App + β_App×time4`).

### MEA Kinase Enrichment

Replaces the earlier DiffPhos (Fisher's exact test on binarized sites) approach. Uses MEA (Motif Enrichment Analysis) via GSEA pre-ranked on stoichiometry β values. This provides continuous NES (Normalized Enrichment Score) per kinase per contrast, without requiring arbitrary binarization cutoffs. Significance: FDR < 0.25 (standard GSEA threshold).

**Preprocessing before GSEA ranking** (per contrast, in `_run_mea()`):

1. **Median-centering** — Subtracts the median stoichiometry LFC from all sites. Without this, a global shift in phosphorylation (e.g., net dephosphorylation at a timepoint) propagates into every kinase substrate set, making NES sign reflect the background shift rather than kinase-specific activity. The removed offsets are logged to `mea_global_shift.csv`.

2. **Winsorization** — Clips centered LFCs at the 1st/99th percentile (`config.MEA_WINSORIZE_PERCENTILE`) to prevent extreme outlier sites from inflating GSEA enrichment scores. Clipped sites are logged to `winsorized_sites.csv`.

### Unified Cell-Type Attribution

All MEA-significant kinases are evaluated against three evidence sources for cell-type attribution at the **subclass level** (24 SEA-AD subclasses, e.g., Pvalb, Sst, L2/3 IT, Microglia-PVM):

1. **SEA-AD concordance (pathway-matched)**: For each kinase gene, look up its differential expression in human AD (SEA-AD, 139 supertypes aggregated to 24 subclasses via `adata.var["Subclass"]`). The effect size file is **pathway-matched** to the mouse contrast: App contrasts use `effect_sizes_early.h5ad` (early/low-CPS donors, amyloid-dominant), Tau contrasts use `effect_sizes_late.h5ad` (late/high-CPS, tau-dominant), and ApTt contrasts use `effect_sizes.h5ad` (full CPS range). This avoids checking amyloid-pathway kinases against a tau-dominated late-stage signature (early/late Pearson r ≈ −0.12, ~48% sign flips). Concordance score = `sign(NES) * median(sea_ad_lfc)` — positive when kinase activity direction matches human AD transcriptomic change in that subclass. The stratum used is recorded in `sea_ad_stratum`. Mapping: `config.SEA_AD_PATHWAY_MAP`.

2. **WMB expression specificity**: How specifically each kinase gene is expressed in each of the 24 subclasses (Allen WMB 10Xv3 HPF dataset, 338 WMB subclasses mapped to 24 SEA-AD subclasses via keyword matching).

3. **Song within-cohort concordance**: Same-species, same-cohort evidence from paired snRNA-seq (28 animals). Uses Allen Cell Type Mapper annotations from `170_gex_celltypes_00.h5ad` (210 subclass labels → 22/24 SEA-AD subclasses). Pseudobulk expression is aggregated per (animal, subclass), then factorial OLS (males only, pooled across timepoints) estimates pathway-specific LFCs (App, Tau, ApTt). Song concordance = `sign(NES) * song_lfc`. Also provides within-cohort expression specificity as a complement to WMB. Song specificity correlates r ≈ 0.73 with WMB specificity (Pearson). Song evidence is additive — it can boost confidence but is never required.

**Weighted concordance model (Song 3× : SEA-AD 1×):** Song and SEA-AD are both evidence, weighted by reliability. Song (same-species, same-cohort, paired animals) receives 3× weight; SEA-AD (cross-species human proxy) receives 1×. When both are available: `effective_concordance = (3 × song_cs + 1 × sea_ad_cs) / 4`. When only one is available, it provides the full concordance signal. Neither reference has absolute veto power — Song can rescue attributions that SEA-AD blocks, and vice versa, proportional to signal strength. The `concordance_source` column records which references contributed ("both", "song", or "sea_ad"). Weights are configurable via `config.SONG_CONCORDANCE_WEIGHT` and `config.SEA_AD_CONCORDANCE_WEIGHT`.

Combined confidence (thresholds scale with number of cell types, expressed as multiples of uniform = 1/24):
- **High**: Song-contributed concordance + WMB specificity ≥ 2× uniform (~0.083) + |LFC| > 0.1
- **Moderate**: SEA-AD-only concordance (capped regardless of WMB tier), or Song-contributed + lower WMB
- **Low**: Weak evidence from all sources

Evidence basis labels: `three_way` (WMB + SEA-AD + Song), `within_cohort` (WMB + Song), `cross_species` (WMB + SEA-AD), `mouse_expression_only`, `song_only`, `human_concordance_only`, `weak`.

The 24 subclasses are defined once in `config.SEA_AD_SUBCLASSES` and re-exported by `atlas_reference.py`. Each subclass maps to a parent 5+1 class via `atlas_reference.SUBCLASS_TO_5PLUS1`.

### Mechanism Annotation (Supplementary)

Optionally (`--mechanism-annotation`), the pipeline can classify kinases as abundance-driven, activity-driven, or both by comparing raw phospho MEA vs stoichiometry MEA significance. This is a descriptive annotation, not a routing variable for attribution.

### Final Attribution Tables

attribution_recovery.py produces four hypothesis-generation tables. Static localization evidence (WMB, SEA-AD) is separated from the dynamic NES signal. WMB acts as a gate (kinase must be expressed) not a weight.

- **Table 1** `kinase_activity_matrix.csv` — wide NES/FDR across 9 contrasts + trajectory label (one row/kinase)
- **Table 2** `celltype_evidence_table.csv` — WMB-gated static localization evidence (one row/kinase×celltype)
- **Table 3** `kinase_hypothesis_table.csv` — kinase-first synthesis: activity profile + top cell-type candidates. **Primary downstream deliverable.**
- **Table 4** `celltype_kinase_profiles.csv` — cell-type-first synthesis: per-cell-type NES trajectories for all WMB-gated kinases

### Dependency Graph

```
config.py  ←  data_ingest.py                                    (total-proteome characterization)
config.py  ←  data_ingest.py  ←  kinase_attribution.py          (stoichiometry + MEA + unified attribution)
config.py  ←  kinase_attribution.py  ←  attribution_recovery.py (cross-contrast + final table)

Supporting:
config.py  ←  atlas_reference.py                                (SEA-AD + WMB + Aging Mouse acquisition)
config.py  ←  atlas_reference.py  ←  wmb_expression.py          (WMB expression export)
config.py  ←  snrna_integration.py                              (Song within-cohort evidence)
```

### Live Code

- `config.py` — Shared configuration: file paths, thresholds, enrichment method params, `SEA_AD_SUBCLASSES` list (24 subclasses, single source of truth), sample filtering params (`OUTLIER_ZSCORE_THRESH`, `ANALYSIS_MODE`). Still structurally mixed with legacy settings.
- `data_ingest.py` — TMT channel mapping, phosphosite-to-protein matching, marker assessment, PCA quality control, outlier detection. Outputs to `outputs/reports/data_ingest/`. Requires: `scikit-learn`, `matplotlib`.
- `kinase_attribution.py` — IRS normalization (all 72 samples), sample filtering (outlier exclusion + sex filter), stoichiometry computation, factorial OLS with disease×timepoint interactions (9 contrasts), MEA kinase enrichment (median-centered + winsorized), unified cell-type attribution (SEA-AD concordance + WMB expression specificity). Outputs to `outputs/reports/kinase_attribution/`. Requires: `kinase-library`, `gseapy`, `scikit-learn`, `matplotlib`, `anndata`.
- `attribution_recovery.py` — Cross-contrast consistency analysis, final unified attribution table. Outputs to `outputs/reports/attribution_recovery/`. Requires: `matplotlib`.
- `plot_attribution_bubbles.py` — Per-tissue heatmaps, direction-over-time diverging bars, ApTt additivity scatter, winsorization diagnostic. Outputs to `outputs/reports/attribution_recovery/bubble_plots/`. Requires: `matplotlib`, `scipy`.

### Supporting Code

- `atlas_reference.py` — External atlas acquisition: downloads WMB, Aging Mouse, and SEA-AD from Allen Institute. Produces structure reports, taxonomy mapping, kinase gene coverage. Exports cell-type taxonomy constants (`SUBCLASS_KEYWORDS`, `SEA_AD_SUBCLASSES`, `SUBCLASS_TO_5PLUS1`, `match_subclass`). Outputs to `outputs/reports/atlas_reference/`. Requires: `abc_atlas_access`, `anndata`, `boto3`.
- `wmb_expression.py` — WMB expression export: per-subclass kinase/phosphatase expression from Allen WMB 10Xv3 HPF (24 SEA-AD subclasses). Produces `outputs/reports/wmb_expression/wmb_kinase_expression.csv` consumed by kinase_attribution.py unified attribution. Requires: `anndata`, `abc_atlas_access`.
- `snrna_integration.py` — Song snRNA-seq integration: computes pseudobulk expression from paired 170_gex_celltypes_00.h5ad (63K nuclei, 28 animals, Allen Cell Type Mapper annotations → 22/24 SEA-AD subclasses), within-cohort expression specificity, and within-cohort transcriptomic concordance via factorial OLS (males-only, pooled across timepoints). Outputs to `outputs/reports/snrna_integration/`. Requires: `anndata`, `scipy`, `statsmodels`.
- `map_kinases_to_genes.py` — Kinase→gene symbol mapping utility.

### Runners

Operational shell wrappers under `code/runners/`:
- `main/run_live_pipeline.sh` — Bundled front door (data_ingest → kinase_attribution → attribution_recovery, gates on WMB prerequisite)
- `main/run_dual_analysis.sh` — **Dual-track runner**: males-only (primary) + full-cohort (sensitivity), archives outputs to `*_males_only/` and `*_full_cohort/` directories
- `main/run_data_ingest.sh` — Data ingestion wrapper
- `main/run_kinase_attribution.sh` — Kinase attribution wrapper
- `main/run_attribution_recovery.sh` — Attribution recovery wrapper
- `supporting/run_snrna_integration.sh` — Song snRNA-seq integration
- `supporting/run_atlas_reference.sh` — Atlas reference setup
- `supporting/run_wmb_expression.sh` — WMB expression export

**Caching:** External API calls (MyGene.info, Allen Brain Atlas) are cached inside `data/incytr_collections/song/analysis_cache/` to avoid redundant requests.

## Key Data Files

### Live Pipeline Inputs
- `data/incytr_collections/song/primary/proteomics/song2024_tmttotal_protein_quant_merged_labeled (2).xlsx` — 72-animal total proteome (6 plexes × 10 TMT channels)
- `data/incytr_collections/song/primary/proteomics/song_IMAC_sitequant_merged_labeled (2).xlsx` — Phospho sitequant (per-site intensities)
- `data/incytr_collections/song/primary/proteomics/song_IMAC_compositeSites_merged_labeled (2).xlsx` — Phospho composite sites
- `data/incytr_collections/song/primary/proteomics/Sample_list_72mice (1).xlsx` — TMT channel-to-animal sample mapping
- `data/incytr_collections/song/method_records/aobs_desp_standardized/inputs/A_obs_fractions.tsv` — Cell-type composition fractions (24 × 10 cell types)
- `data/external/allen_abc/` — Cached Allen Brain Cell Atlas data (WMB + Aging Mouse)
- `data/external/sea_ad/` — SEA-AD MTG processed data (h5ad, 139 supertypes): `effect_sizes.h5ad` (full CPS), `effect_sizes_early.h5ad` (early/low-CPS), `effect_sizes_late.h5ad` (late/high-CPS)

### Supporting/Cached Data
- `data/incytr_collections/song/analysis_cache/kinase_to_gene_mapping.csv` — Cached kinase→gene symbol mappings
- `data/incytr_collections/song/analysis_cache/allen_expression_cache.csv` — Cached Allen Brain Atlas expression results
- `outputs/reports/wmb_expression/wmb_kinase_expression.csv` — WMB expression matrix (required for unified attribution)

### Archived Inputs (used by archived code only)
- `data/incytr_collections/song/proteomics/ps_yuyu_deconvoluted.csv` — Deconvoluted ser/thr (archived pipeline)
- `data/incytr_collections/song/proteomics/source/imac_median.csv` — Bulk ser/thr phosphoproteomics (archived)

## Output

### Live Pipeline Outputs (under `outputs/reports/`)

- `outputs/reports/data_ingest/` — Data ingestion: sample mapping, phospho-protein matching, marker assessment, PCA plots
  - `sample_exclusions.csv` — Per-animal outlier metrics and exclusion flags
  - `pca_plots/outlier_diagnostic.png` — PCA + z-score strip plot for outlier detection
- `outputs/reports/kinase_attribution/` — Stoichiometry, MEA enrichment, and unified attribution:
  - `stoichiometry_matrix.csv`, `raw_phospho_normalized.csv` — Core normalized data (all 72 samples)
  - `mea_stoichiometry.csv` — MEA kinase enrichment results (NES, FDR per kinase per contrast)
  - `mea_global_shift.csv` — Median offsets removed per contrast before GSEA (transparency log)
  - `winsorized_sites.csv` — Sites clipped during winsorization (with original/clipped LFCs)
  - `site_level_ols.csv` — Per-site OLS results (stoichiometry + raw phospho LFC/FDR for 9 contrasts)
  - `unified_attribution.csv` — Unified cell-type attribution (SEA-AD + WMB combined)
  - `attribution_summary.json` — Summary counts by confidence, cell type, contrast
  - `mechanism_annotation.csv` — Optional: abundance/activity/both classification
- `outputs/reports/attribution_recovery/` — Hypothesis-generation tables:
  - `kinase_activity_matrix.csv` — wide NES/FDR + trajectory label (1 row/kinase)
  - `celltype_evidence_table.csv` — WMB-gated static evidence (1 row/kinase×celltype)
  - `kinase_hypothesis_table.csv` — kinase-first synthesis, **primary downstream deliverable**
  - `celltype_kinase_profiles.csv` — cell-type-first NES profiles (1 row/celltype×kinase)
  - `bubble_plots/` — Heatmaps, direction-over-time, additivity scatter, winsorization diagnostic
- `outputs/reports/atlas_reference/` — Atlas structure reports, taxonomy mapping, coverage
- `outputs/reports/wmb_expression/` — WMB expression export (supporting)

## Foundation Documentation

The `docs/foundation/` directory contains authoritative design documents:
- `analysis_charter.md` — Single front door defining the live scope, closed paths, and rules for new work
- `live_pipeline_contract.md` — Stage-by-stage runtime spec (inputs, outputs, failure modes) for the live pipeline
- `repo_surface_index.md` — Exhaustive main/supporting/archived classification of every file
- `analysis_rationale.md` — Why the project pivoted from deconvolution to stoichiometry
- `statistical_constraints.md` — Hard design limits (identifiability, DOF)
- `repo_retention_policy.md` — Active-vs-archived boundaries

## Gotchas

- **ANALYSIS_MODE controls sample filtering** — defaults to `males_only`. Set `ANALYSIS_MODE=full_cohort` for sensitivity analysis with both sexes. The mode affects `--enrich`, `--attribute`, and `--mechanism-annotation` but NOT `--normalize` (which always uses all 72 samples)
- **Outlier detection requires stoichiometry** — `data_ingest.py --outliers` reads `stoichiometry_matrix.csv`, so `kinase_attribution.py --normalize` must be run first. Falls back to total proteome if unavailable
- **Limited automated tests** — live pipeline has no unit tests; verify with `--summary` flags on each script
- **Song proteomics files must be mounted** — data_ingest.py reads Excel workbooks from `data/incytr_collections/song/primary/proteomics/`
- **WMB prerequisite** — `run_live_pipeline.sh` gates on `wmb_kinase_expression.csv` and `wmb_proteome_expression.csv`; run `run_wmb_expression.sh` first
- **Atlas cache compressed** — raw h5ad files under `data/external/allen_abc/` are zstd-compressed to save space (~115 GB → ~26 GB). Decompress with `bash code/runners/supporting/decompress_atlas_cache.sh` before re-running `wmb_expression.py`. See `data/external/allen_abc/MANIFEST.json` for provenance
- **SEA-AD data required** — Unified attribution needs SEA-AD effect sizes under `config.SEA_AD_DIR`
- **API caching** — delete files under `data/incytr_collections/song/analysis_cache/` to force re-fetch
- **WMB expression memory** — `wmb_expression.py --proteome` processes 6,308 genes across 13 regions; use `skip_regional=True` and `chunk_size=2000` to avoid OOM (~30GB RAM available)
- **A_obs is group-level** — `A_obs_fractions.tsv` has 24 rows (one per factorial group), not 72 (per-animal). Marker assessment correlates per-animal protein intensity with per-group composition, so most cell types yield q≈1.0 — this is expected
- **Do not reopen closed paths** — direct deconvolution, factor model, two-compartment, and transcript-only rescue are all closed (see charter)

### Code Intelligence

Prefer [LSP](/search_kw/2519ab10c2ea14e4599a7c2565ea0ac0) over [Grep](/search_kw/6ccc30f8b0bcd9ee2f2c9f62486726d2)/Read for code navigation — it's faster, precise, and avoids reading entire files:
- `workspaceSymbol` to find where something is defined
- `findReferences` to see all usages across the codebase
- `goToDefinition` / `goToImplementation` to jump to source
- `hover` for type info without reading the file

Use Grep only when LSP isn't available or for text/pattern searches (comments, strings, config).

After writing or editing code, check LSP diagnostics and fix errors before proceeding.

## Tool Preferences
- Use `bat` for file previewing when showing output to the user
- Use `eza -lah --git` for directory listings
