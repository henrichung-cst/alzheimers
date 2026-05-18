# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Stoichiometry-corrected kinase attribution pipeline for Alzheimer's disease phosphoproteomics. Integrates 72-animal TMT total proteome with phosphoproteomics to compute stoichiometry (log2 phospho − log2 protein), runs MEA (GSEA-based) kinase enrichment on stoichiometry β values, and attributes findings to cell types using unified evidence from SEA-AD transcriptomic concordance and WMB expression specificity. Primary analysis uses males-only (33 animals after outlier exclusion) to avoid hormonal confounds; full-cohort analysis is run as a sensitivity check.

The project pivoted from direct cell-type deconvolution (which failed validation) to this stoichiometry-corrected approach. See `docs/foundation/analysis_charter.md` for the authoritative scope definition and `docs/foundation/analysis_rationale.md` for the pivot logic.

## Environment Setup

Env managed by **pixi**; activated automatically via direnv on `cd`. Python 3.11 + kinase-library 1.7.0 + R. Package versions are pinned to match kinase-library's strict `~=` requirements (scipy 1.14, scikit-learn 1.6, pandas 2.2, seaborn 0.13, etc).

```bash
pixi install   # first-time setup
```

## Running the Analysis

All scripts run from the repo root.

### Live Pipeline

The bundled live task runs ingest → normalize → enrich → attribute → recover:
```bash
pixi run live
```

The **dual-track runner** runs males-only (primary) and full-cohort (sensitivity) analyses sequentially:
```bash
pixi run dual
```

Or run individual stages:
```bash
pixi run ingest     # data ingestion
pixi run normalize  # IRS + stoichiometry
pixi run enrich     # MEA kinase enrichment (males-only)
pixi run attribute  # unified cell-type attribution (males-only)
pixi run recover    # cross-contrast + final tables
```

**Data ingestion and characterization** (`data_ingest.py`):
```bash
python alz/data_ingest.py --mapping       # §1: TMT channel-to-animal sample mapping (72 animals, 6 plexes)
python alz/data_ingest.py --phospho-match # §2: Phosphosite-to-protein matching (91.7% match rate)
python alz/data_ingest.py --quality       # §3: Data quality (PCA, batch effects, missingness)
python alz/data_ingest.py --outliers      # §4: Statistical outlier detection (within-group robust z-scores)
python alz/data_ingest.py --run           # All steps in order
python alz/data_ingest.py --summary       # Print cached results
```

**Stoichiometry, MEA enrichment, and unified attribution** — split into four stage modules plus a summary helper:

Sample filtering (`analysis_mode`) lives in `conf/base/parameters.yml`. Default is `males_only`; `KEDRO_ENV=full_cohort` overlays `conf/full_cohort/parameters.yml` for the sensitivity analysis.
```bash
python alz/kinase_normalize.py                            # Stage 1: IRS cross-plex normalization + stoichiometry (all 72 samples)
python alz/kinase_enrich.py                               # Stage 2: OLS + MEA (males-only by default; KEDRO_ENV=full_cohort to switch)
python alz/kinase_attribute.py                            # Stage 3: Unified cell-type attribution
python alz/kinase_mechanism.py                            # Optional: raw phospho MEA + mechanism classification
python alz/kinase_summary.py                              # Print cached results
```

**Final attribution-table assembly** (`attribution_recovery.py`):
```bash
python alz/attribution_recovery.py --kinase-profiles   # S3: Kinase activity matrix + hypothesis table
python alz/attribution_recovery.py --celltype-profiles # S4: Cell-type evidence table + kinase profiles
python alz/attribution_recovery.py --hypothesis-tables # S3+S4: Run both new hypothesis table steps
python alz/attribution_recovery.py --run               # All steps in order
python alz/attribution_recovery.py --summary           # Print cached results
```

### Supporting Prerequisites

These must be run before the live pipeline if their outputs don't exist:

```bash
# External Atlas Data Acquisition (SEA-AD + WMB from Allen Institute)
python alz/atlas_reference.py --sea-ad        # SEA-AD MTG Nebula effect-size h5ads
python alz/atlas_reference.py --wmb-download  # All 13 WMB-10Xv3 log2 expression matrices (~95 GB)
python alz/atlas_reference.py --run           # SEA-AD + WMB download
# Runner: bash alz/runners/supporting/run_atlas_reference.sh

# WMB Expression Export (required for unified attribution + marker assessment)
python alz/wmb_expression.py --run       # Compute WMB per-cell-type kinase expression matrix
python alz/wmb_expression.py --proteome  # Compute proteome-wide WMB expression (for --markers)
python alz/wmb_expression.py --summary   # Print cached results
# Runner: bash alz/runners/supporting/run_wmb_expression.sh

# Song snRNA-seq Integration (within-cohort evidence from paired animals)
python alz/snrna_integration.py --pseudobulk    # S1: pseudobulk from 170 h5ad (28 animals → ~21 of 34 WMB classes)
python alz/snrna_integration.py --specificity   # S2: within-cohort expression specificity
python alz/snrna_integration.py --concordance   # S3: within-cohort transcriptomic concordance (males-only OLS)
python alz/snrna_integration.py --run           # All stages in order
python alz/snrna_integration.py --summary       # Print cached results
# Runner: bash alz/runners/supporting/run_snrna_integration.sh
```

### Supplementary Diagnostics

Reviewer-response analyses that validate pipeline choices. Run after the main pipeline:

```bash
bash alz/runners/supplementary/run_reviewer_diagnostics.sh   # All diagnostics
# Or individually:
python alz/supplementary/fdr_stringent.py --run              # Q4: FDR < 0.10 comparison
python alz/supplementary/threshold_sensitivity.py --run      # Q1: Confidence tier sweep
python alz/supplementary/aggregation_robustness.py --run     # Q2: Aggregation method comparison
python alz/supplementary/parent_protein_qc.py --run          # Q5: Activity-driven parent QC
python alz/supplementary/deconvolution_feasibility.py --run  # Q6: Marker→composition concordance via factorial OLS (closes deconvolution-feasibility, requires wmb_expression --proteome)
```

### Standalone Utilities

```bash
python alz/map_kinases_to_genes.py       # Kinase→gene symbol mapping
python alz/build_unified_viewer.py       # Interactive HTML viewer (kinase + pathway + cross-entity)
python alz/lucie_5xfad_manifest.py       # Lucie 5xFAD proteomics manifest builder
```

## Architecture

The live pipeline is a 3-stage stoichiometry-corrected MEA enrichment + unified attribution workflow. For full specifications, see the foundation docs:

- **Scope and workflow**: [`docs/foundation/analysis_charter.md`](docs/foundation/analysis_charter.md)
- **Stage-by-stage contract** (inputs, outputs, failure modes): [`docs/foundation/live_pipeline_contract.md`](docs/foundation/live_pipeline_contract.md)
- **Concordance model** (evidence sources, weights, confidence tiers): [`docs/foundation/concordance.md`](docs/foundation/concordance.md)
- **Statistical constraints** (identifiability, DOF): [`docs/foundation/statistical_constraints.md`](docs/foundation/statistical_constraints.md)
- **Pivot rationale**: [`docs/foundation/analysis_rationale.md`](docs/foundation/analysis_rationale.md)

### Pipeline Summary

1. **data_ingest.py** — TMT channel mapping, phosphosite-to-protein matching (91.7%), marker assessment, PCA QC, outlier detection
2. **kinase_normalize.py / kinase_enrich.py / kinase_attribute.py / kinase_mechanism.py** — Modular kinase pipeline: IRS normalization (all 72 samples) → stoichiometry (`log2(phospho) − log2(protein)`) → factorial OLS (9 time-resolved contrasts) + MEA kinase enrichment (median-centered + winsorized) → unified cell-type attribution (SEA-AD + WMB + Song concordance) → optional mechanism classification. `kinase_summary.py` prints cached results.
3. **attribution_recovery.py** — Cross-contrast consistency and final hypothesis tables (primary deliverable: `kinase_hypothesis_table.csv`)
4. **plot_attribution_bubbles.py** — Visualization: heatmaps, direction-over-time bars, additivity scatter

### Key Design Points

- **Stoichiometry**: `log2(phospho) − log2(protein)` removes parent-protein abundance confounding
- **Sample filtering**: `analysis_mode` Kedro parameter in `conf/base/parameters.yml` (default `males_only`); `KEDRO_ENV=full_cohort` swaps to `conf/full_cohort/parameters.yml`. IRS normalization always uses all 72 samples; filtering applies at OLS time
- **OLS model**: disease×timepoint interactions → 9 contrasts (3 diseases × 3 timepoints). Design matrix: const, App, Tau, Int, [female], time_4mo, time_6mo, App×time4, App×time6, Tau×time4, Tau×time6
- **MEA**: GSEA pre-ranked on stoichiometry β values; median-centered then winsorized (1st/99th percentile) before ranking; FDR < 0.25
- **Attribution**: 3 evidence sources (Song within-cohort, SEA-AD concordance, WMB specificity) weighted Song 3× : SEA-AD 1×; 34 WMB classes (Allen WMB published taxonomy, no silent drops); confidence tiers (high/moderate/low)

### Dependency Graph

```
config.py  ←  data_ingest.py  ←  kinase_normalize.py  ←  kinase_enrich.py  ←  kinase_attribute.py  ←  attribution_recovery.py
                                                                          ←  kinase_mechanism.py (optional, off the live path)
                                                                          ←  kinase_summary.py (read-only)

Supporting:
config.py  ←  atlas_reference.py  ←  wmb_expression.py
config.py  ←  snrna_integration.py

Integration:
kinase_enrich.py / kinase_attribute.py + snrna_integration.py  ←  alz/integration/
```

### Live Code

- `config.py` — Shared configuration: file paths, thresholds, enrichment method params, `WMB_CLASSES` list (34 WMB classes, single source of truth for the cell-type spine — used by both the kinase pipeline and the Incytr integration), `load_song_to_wmb_class_map()` (Allen Cell Type Mapper raw `subclass_name` → WMB class, built from `wmb_subclass_to_class.csv` with numeric prefix stripped; no SEA-AD intermediate), `OUTLIER_ZSCORE_THRESH`. `ANALYSIS_MODE` is a bridge attribute that loads `analysis_mode` from Kedro parameters (`conf/base/parameters.yml`, KEDRO_ENV-switchable); Phase 4 will replace direct reads with node-injected `params:analysis_mode`. Still structurally mixed with legacy settings.
- `data_ingest.py` — TMT channel mapping, phosphosite-to-protein matching, marker assessment, PCA quality control, outlier detection. Outputs to `outputs/reports/data_ingest/`. Requires: `scikit-learn`, `matplotlib`.
- `kinase_normalize.py` — Stage 1: IRS cross-plex normalization (all 72 samples) + stoichiometry computation. Per-track (`--track st|py|both`). Outputs to `outputs/reports/kinase_attribution/`. Requires: `scikit-learn`, `matplotlib`.
- `kinase_enrich.py` — Stage 2: sample filtering (outlier exclusion + sex), factorial OLS with disease×timepoint interactions (9 contrasts), MEA kinase enrichment (median-centered + winsorized). Per-track. Requires: `kinase-library`, `gseapy`.
- `kinase_attribute.py` — Stage 3: unified cell-type attribution (SEA-AD concordance + WMB expression specificity + Song within-cohort). Currently consumes the `st` track only. Requires: `anndata`.
- `kinase_mechanism.py` — Optional supplementary stage: raw-phospho MEA + abundance/activity/both classification. Reuses Stage 2 helpers via import.
- `kinase_summary.py` — Read-only: prints a cached-results summary across all four stages.
- `attribution_recovery.py` — Cross-contrast consistency analysis, final unified attribution table. Outputs to `outputs/reports/attribution_recovery/`. Requires: `matplotlib`.
- `plot_attribution_bubbles.py` — Per-tissue heatmaps, direction-over-time diverging bars, ApTt additivity scatter, winsorization diagnostic. Outputs to `outputs/reports/attribution_recovery/bubble_plots/`. Requires: `matplotlib`, `scipy`.
- `build_unified_viewer.py` — Generates the interactive HTML viewer (kinase activity → cell-type attribution → pathway backbones → cross-entity views). Reads attribution-recovery tables, MEA stoichiometry, site-level OLS, factorial backbone recurrence + permutation pvalues, and the `kinase_backbone_edges.parquet` edge index. Emits `outputs/reports/unified_viewer/index.html` + a shared JSON payload plus sharded per-entity edge slices under `edge_slices/{kinase,backbone}/` fetched on demand via `SliceCache`. Requires: `kinase-library`, `scipy`, `pyarrow`.
- `lucie_5xfad_manifest.py` — Builds a proteomics manifest for Lucie 5xFAD data integration.

### Supporting Code

- `atlas_reference.py` — External atlas acquisition: downloads SEA-AD Nebula effect-size h5ads (`effect_sizes{,_early,_late}.h5ad`) from S3 and all 13 WMB-10Xv3 log2 expression matrices into the ABC project cache. Also exports kinase/phosphatase gene-list helpers and ABC cache utilities consumed by `wmb_expression.py`. Requires: `abc_atlas_access`, `anndata`, `boto3`.
- `wmb_expression.py` — WMB expression export: per-class kinase/phosphatase expression from Allen WMB 10Xv3 (34 WMB classes, group-by on `wmb_meta["class"]`, no silent drops). Emits `outputs/reports/wmb_expression/wmb_kinase_expression.csv` (primary, class-level) + `wmb_kinase_expression_subclass.csv` (audit sidecar at WMB subclass level). Consumed by kinase_attribute.py. Requires: `anndata`.
- `snrna_integration.py` — Song snRNA-seq integration: computes pseudobulk expression from paired 170_gex_celltypes_00.h5ad (63K nuclei, 28 animals) keyed on Allen Cell Type Mapper `class_name` (rolled up to 34 WMB classes via `wmb_class_manifest.csv`); ~21 of 34 classes pass Song's confidence + animal-count gates. Within-cohort expression specificity and transcriptomic concordance via factorial OLS (males-only, pooled across timepoints). Outputs to `outputs/reports/snrna_integration/`. Requires: `anndata`, `scipy`, `statsmodels`.
- `map_kinases_to_genes.py` — Kinase→gene symbol mapping utility.

### Integration Code (Incytr)

`alz/integration/` is a thin AD-specific wrapper around the upstream `incytr` R package (`~/Projects/work/incytr/`). All math, scoring, and pathway construction live in the package; this directory only loads AD inputs, runs the engine, persists results to parquet, and defines aggregation views. See `docs/incytr_remediation_plan.md` for the rationale and `alz/integration/README.md` for the file-by-file layout.

In-tree files:

- `export_factorial_inputs.py` — h5ad → on-disk contract (`expression_*.csv/.mtx`, `animal_metadata.csv`)
- `factorial.R` — entry point; hard-fails if `Incytr::construct_factorial_paths` / `score_factorial_paths` are absent
- `load.R`, `persist.R`, `views.sql`, `run_factorial.sh` — loader, parquet writer, DuckDB views, shell shim
- `config_integration.py` — filter values, design columns, contrast vectors, paths

Run order: `pixi run install-incytr && pixi run export-factorial-inputs && pixi run incytr-factorial`.

### Per-cluster proportional decomposition (Levy-19 spine)

Branch path active alongside the bulk live pipeline. Forward projection only — `P_c = f_c × bulk` — **not** statistical deconvolution (closed). See `docs/incytr_deconvolution_pivot.md`.

Stages:
1. `alz/snrna_proportions.py --spine levy19` — per-(animal, cluster, gene) weights `f_c = (expr_c / Σ expr) × (N_total / N_c)`
2. `alz/decomposition/build_celltype_decomposition.py --spine levy19 --track both` — projects bulk phospho (IMAC + pY) and protein onto the 19-cluster spine
3. `alz/decomposition/enrich_celltype.py --spine levy19 --track {st,py}` — per-cluster factorial OLS + MEA (9 contrasts)
4. `alz/integration/export_factorial_inputs.py` + `pixi run incytr-factorial` — per-cluster Incytr scoring (`per_cluster/{pr,ps,py}/<cluster>.parquet`; upstream `Incytr::resolve_wide` list-dispatches)

End-to-end runner: `bash alz/runners/main/run_pivot_smoke.sh [--skip-normalize] [--skip-incytr]`.

Verification harness (`alz/decomposition/verify_decomposition.py --spine levy19 --all`) writes `outputs/reports/decomposition/levy19/verification.json` and checks four contracts:
- Mass identity `Σ_c [P_c × (N_c / N_total)] ≈ bulk` (per-cell-rate, **not** `Σ_c P_c = bulk`)
- Spine coverage (all 19 clusters present, no silent drops)
- Per-cluster vs bulk MEA agreement under `f_c`-weighting (Spearman ρ ≥ 0.7 per contrast)
- Incytr produces 19² = 361 sender × receiver pairs

The legacy R wrappers / Python adapters / sidecars / orchestrator shell scripts were retired during the rewrite. They are preserved under `archive/incytr_integration/` (bulk gitignored per repo allowlist; on disk only — copy back if needed). R deps (`Incytr`, `DBI`, `duckdb`, `data.table`, `arrow`) are still required.

### Runners

Operational shell wrappers under `alz/runners/`:
- `main/run_live_pipeline.sh` — Bundled front door (data_ingest → kinase_attribution → attribution_recovery, gates on WMB prerequisite)
- `main/run_dual_analysis.sh` — **Dual-track runner**: males-only (primary) + full-cohort (sensitivity), archives outputs to `*_males_only/` and `*_full_cohort/` directories
- `main/run_data_ingest.sh` — Data ingestion wrapper
- `main/run_kinase_attribution.sh` — Kinase attribution wrapper
- `main/run_attribution_recovery.sh` — Attribution recovery wrapper
- `supporting/run_snrna_integration.sh` — Song snRNA-seq integration
- `supporting/run_atlas_reference.sh` — Atlas reference setup
- `supporting/run_wmb_expression.sh` — WMB expression export

**Caching:** External API calls (MyGene.info, Allen Brain Atlas) are cached inside `data/datasets/song/analysis_cache/` to avoid redundant requests.

## Key Data Files

### Live Pipeline Inputs
- `data/datasets/song/primary/proteomics/song2024_tmttotal_protein_quant_merged_labeled (2).xlsx` — 72-animal total proteome (6 plexes × 10 TMT channels)
- `data/datasets/song/primary/phospho/song_IMAC_{sitequant,compositeSites}_merged_labeled (2).xlsx` — Phospho IMAC (pS/pT) site-level + composite
- `data/datasets/song/primary/phospho/song_pY_{sitequant,compositeSites}_merged_labeled (2).xlsx` — Phospho pY site-level + composite (1st-class sibling to IMAC)
- `data/datasets/song/primary/metadata/Sample_list_72mice (1).xlsx` — TMT channel-to-animal sample mapping
- `data/external/allen_abc/` — Cached Allen Brain Cell Atlas data (WMB)
- `data/external/sea_ad/` — SEA-AD MTG processed data (h5ad, 139 supertypes): `effect_sizes.h5ad` (full CPS), `effect_sizes_early.h5ad` (early/low-CPS), `effect_sizes_late.h5ad` (late/high-CPS)

### Supporting/Cached Data
- `data/datasets/song/analysis_cache/kinase_to_gene_mapping.csv` — Cached kinase→gene symbol mappings
- `outputs/reports/wmb_expression/wmb_kinase_expression.csv` — WMB expression matrix (required for unified attribution)

### Decomposition (CTM-native, branch-only — not in live pipeline)
`alz/deconvolution/build_wmb_decomposition.py` consumes per-(site, group) bulk medians (`imac_median.csv`, `py_median.csv`, `pr_median.csv`) and the `yuyu_samplekey.csv` MS\_ID↔SCRNA bridge. These were deleted from `data/datasets/song/proteomics/source/` on 2026-05-07; re-pull from Google Drive via `pixi run ingest-gdrive-shared` before running the branch. Outputs: `outputs/reports/deconvolution/wmb_decomposition/{ps,py,pr}_wmb_decomposition.csv` + `wmb_class_size.csv`.

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
  - `bubble_plots/` — Heatmaps, direction-over-time, additivity scatter, winsorization diagnostic
- `outputs/reports/wmb_expression/` — WMB expression export (supporting)

## Foundation Documentation

The `docs/foundation/` directory contains authoritative design documents:
- `analysis_charter.md` — Single front door defining the live scope, closed paths, and rules for new work
- `live_pipeline_contract.md` — Stage-by-stage runtime spec (inputs, outputs, failure modes) for the live pipeline
- `concordance.md` — SEA-AD concordance analysis design and pathway matching
- `analysis_rationale.md` — Why the project pivoted from deconvolution to stoichiometry
- `statistical_constraints.md` — Hard design limits (identifiability, DOF)
- `repo_retention_policy.md` — Active-vs-archived boundaries

Other documentation:
- `docs/incytr_remediation_plan.md` — Authoritative architectural plan for the kinase ↔ Incytr integration rewrite (math moves into upstream `../incytr`; this repo retains a thin AD shell)
- `docs/integrations/kinase_incytr_integration.md` — In-tree Phase 1 stub inventory + one-way handoff diagram (thin pointer to the remediation plan)
- `docs/archive/kinase_incytr_integration_pre_remediation.md` — Legacy shadow-fork architecture (historical reference only)
- `docs/archive/legacy.md` — Legacy proportional decomposition method (historical reference)
- `archive/deconvolution/docs/deconvolution_infeasibility.md` — Archived synthetic validation proving direct deconvolution is infeasible on this dataset (closed path, frozen). Source script + figures alongside under `archive/deconvolution/`.

## Gotchas

- **`analysis_mode` controls sample filtering** — Kedro parameter in `conf/base/parameters.yml`, defaults to `males_only`. Use `KEDRO_ENV=full_cohort` to overlay `conf/full_cohort/parameters.yml` for sensitivity analysis with both sexes. Affects `kinase_enrich.py`, `kinase_attribute.py`, and `kinase_mechanism.py` but NOT `kinase_normalize.py` (which always uses all 72 samples). The legacy `ANALYSIS_MODE` env var was retired in Phase 3 — setting it is silently ignored
- **Outlier detection requires stoichiometry** — `data_ingest.py --outliers` reads `stoichiometry_matrix.csv`, so `kinase_normalize.py` must be run first. Falls back to total proteome if unavailable
- **Limited automated tests** — live pipeline has no unit tests; verify with `python alz/kinase_summary.py`
- **Song proteomics files must be mounted** — data_ingest.py reads Excel workbooks from `data/datasets/song/primary/proteomics/`
- **WMB prerequisite** — `run_live_pipeline.sh` gates on `wmb_kinase_expression.csv` and `wmb_proteome_expression.csv`; run `run_wmb_expression.sh` first
- **WMB region scope** — `WMB_REGION_SCOPE` env var (default `whole_brain`) selects the regions streamed by `wmb_expression.py`. `whole_brain` uses all 13 regions, which is correct for the specificity score (its denominator is the brain-wide reference). `cortex_hpf` (Isocortex-1/2 + HPF + CTXsp) exists as a sensitivity-check toggle — see `docs/kinase_mapping_rerun_plan.md` "cortex_hpf swap" addendum for why it is not the default. Output filename is the same across scopes; the active scope is stamped to `wmb_kinase_expression.scope.json` and a scope mismatch will force a recompute
- **Atlas cache compressed** — raw h5ad files under `data/external/allen_abc/` are zstd-compressed to save space (~115 GB → ~26 GB). Decompress with `bash alz/runners/supporting/decompress_atlas_cache.sh` before re-running `wmb_expression.py`. See `data/external/allen_abc/MANIFEST.json` for provenance
- **SEA-AD data required** — Unified attribution needs SEA-AD effect sizes under `config.SEA_AD_DIR`
- **API caching** — delete files under `data/datasets/song/analysis_cache/` to force re-fetch
- **WMB expression memory** — `wmb_expression.py --proteome` processes 6,308 genes across 13 regions; use `skip_regional=True` and `chunk_size=2000` to avoid OOM (~30GB RAM available)
- **Do not reopen closed paths** — direct (statistical) deconvolution, per-cluster stoichiometry, factor model, two-compartment, and transcript-only rescue are all closed (see charter). Proportional decomposition on Levy-19 is **active** as a forward projection only
- **Per-cluster mass identity is per-cell-rate** — verification check is `Σ_c [P_c × (N_c / N_total)] ≈ bulk`, NOT `Σ_c P_c = bulk` (the `f_c` weights are per-cell rates × N_total/N_c, so literal summation overshoots)
- **Stage 6 pY track gating** — `build_celltype_decomposition.py --track py` (or `both`) requires `raw_phospho_normalized_pY.csv` from Stage 1. Smoke runner tolerates missing pY; if you need it, re-run `pixi run normalize` first
- **Integration is gated on upstream `incytr`** — `pixi run incytr-factorial` hard-fails before loading data unless `Incytr::construct_factorial_paths` and `Incytr::score_factorial_paths` are exported by the installed package. Until those land per `docs/incytr_remediation_plan.md` Phase 1, the wrapper refuses to start. R deps (`Incytr`, `DBI`, `duckdb`, `data.table`, `arrow`) are still required. Integration config lives in `alz/integration/config_integration.py`, not `alz/config.py`
- **Unified-viewer payload is inlined into `index.html`** — `build_unified_viewer.py` ships PAYLOAD as `<script type="application/json" id="payload-data">` directly in the HTML, not as a separate fetch. After `pixi run viewer` rebuilds, reload the page with a hard refresh (Ctrl+Shift+R / Cmd+Shift+R) — a soft reload serves the cached HTML and the new data won't appear. Quick check from DevTools: `PAYLOAD.meta.generated_at` should match the latest build timestamp

## Tooling & Environment

- Prefer `Grep` (text/filename search) over LSP-based symbol search unless explicitly doing semantic code navigation; the cclsp MCP server is not always available.
- When credentials are needed (Confluence, APIs), check for hidden env files first: `.env`, `.env.confluence`, `.env.local` before asking the user.
- For Confluence page updates with diagrams, render Mermaid locally to PNG images and upload as attachments — do NOT paste raw Mermaid code blocks.
- Use `bat` for file previewing when showing output to the user.
- Use `eza -lah --git` for directory listings.

## Schema & Data Conventions

- When adding provenance/metadata columns, match the existing schema's type exactly (e.g., if single-contrast uses string format for `imputed_nodes`, factorial must too).
- Aggregation queries: always verify whether stats (std, consistency) should be computed over raw route-level rows or pre-aggregated sender-level values.

## Layer-2 drive access (2026-04-19)

Live FUSE mounts for `data/gdrive_shared/` and `data/lucie_proteomics/` were retired in favor of on-demand `rclone copy` ingest tasks (per `~/Projects/work/drive_audit.md` §Phase 3).

- `pixi run ingest-gdrive-shared` → pulls into `data/raw/external/gdrive_shared/`
- `pixi run ingest-lucie-proteomics` → pulls into `data/raw/external/lucie_proteomics/`

Docs under `docs/integrations/` and `docs/archive/` that reference paths under `data/gdrive_shared/<…>/` or `data/lucie_proteomics/<…>/` are historical. If a pipeline needs those files, run the relevant ingest task first and read from `data/raw/external/<name>/<…>`.

## Workflow Conventions

- After any implementation phase, run the full test suite (`pytest` for Python, `devtools::test()` for R) and report pass/fail counts before declaring done.
- When auditing for performance or consolidation, produce the audit document FIRST and get approval before editing files — do not dive into exploratory Read/Bash loops.
- For multi-phase work (audit → plan → implement → simplify), write the plan to a file the user can approve.
- Do not enumerate options, methods, or alternatives as filler. Only list candidates that are genuinely under consideration. If one option is obviously correct, state it directly. Padding answers with inapplicable alternatives wastes the user's time and obscures the real decision.
- **Never intentionally implement the wrong behavior because the right behavior would require deeper changes.** "Smaller diff" and "fewer files touched" are non-goals. If the correct implementation requires regenerating an upstream artifact, refactoring a schema, or changing several files, do that — do not propose a half-fix that ships known-incorrect output (e.g. nulls, stale joins, wrong vocabularies) and hide the breakage behind a follow-up note. Surface the full scope of the right change and execute it.

## Workflow rules

- Run `pixi task list` to enumerate tasks before suggesting how to execute a pipeline step — do not guess task names.
- When adding a dependency, scan `pixi.toml` and PyPI constraints for `~=` pins (kinase-library forces scipy/scikit-learn/pandas/seaborn) and match them on the conda side before `pixi install`.
- Fresh collaborator data flows through `pixi run ingest-<name>`. There are no live collaborator mounts; do not hunt for `data/gdrive_shared/` or `data/lucie_proteomics/` paths.
- DuckDB spill directory is `~/.cache/duckdb` via `.envrc` to avoid OOM on tmpfs `/tmp`. If DuckDB hits disk-full, verify `.envrc` was sourced (`echo $DUCKDB_TEMP_DIR`).
- Ground kinase/biology claims against literature via the connected MCPs (PubMed, bioRxiv, Scholar Gateway) before finalizing interpretations or reports.

## Git workflow

- Commit after each logical unit of work with a conventional commit message (`feat:`, `fix:`, `refactor:`, `docs:`).
- Use `git` (Bash) for all local operations — add, commit, branch, diff, log, status.
- Use `gh` (Bash) for all GitHub operations — PRs, issues, CI status. Do not route these through a GitHub MCP.
- Do not push or open PRs without explicit instruction.
- Do not run `git reset --hard`, force-push, or any destructive operation without confirmation.
