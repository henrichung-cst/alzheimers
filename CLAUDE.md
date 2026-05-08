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
python code/data_ingest.py --mapping       # §1: TMT channel-to-animal sample mapping (72 animals, 6 plexes)
python code/data_ingest.py --phospho-match # §2: Phosphosite-to-protein matching (91.7% match rate)
python code/data_ingest.py --markers       # §3: Cell-type marker protein assessment (WMB atlas, requires wmb_expression --proteome)
python code/data_ingest.py --quality       # §4: Data quality (PCA, batch effects, missingness)
python code/data_ingest.py --outliers      # §5: Statistical outlier detection (within-group robust z-scores)
python code/data_ingest.py --run           # All steps in order
python code/data_ingest.py --summary       # Print cached results
```

**Stoichiometry, MEA enrichment, and unified attribution** — split into four stage modules plus a summary helper:

The `ANALYSIS_MODE` environment variable controls sample filtering (default: `males_only`):
```bash
python code/kinase_normalize.py                            # Stage 1: IRS cross-plex normalization + stoichiometry (all 72 samples)
ANALYSIS_MODE=males_only python code/kinase_enrich.py      # Stage 2: OLS + MEA (males only, outliers excluded)
ANALYSIS_MODE=males_only python code/kinase_attribute.py   # Stage 3: Unified cell-type attribution
python code/kinase_mechanism.py                            # Optional: raw phospho MEA + mechanism classification
python code/kinase_summary.py                              # Print cached results
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
# External Atlas Data Acquisition (SEA-AD + WMB from Allen Institute)
python code/atlas_reference.py --sea-ad        # SEA-AD MTG Nebula effect-size h5ads
python code/atlas_reference.py --wmb-download  # All 13 WMB-10Xv3 log2 expression matrices (~95 GB)
python code/atlas_reference.py --run           # SEA-AD + WMB download
# Runner: bash code/runners/supporting/run_atlas_reference.sh

# WMB Expression Export (required for unified attribution + marker assessment)
python code/wmb_expression.py --run       # Compute WMB per-cell-type kinase expression matrix
python code/wmb_expression.py --proteome  # Compute proteome-wide WMB expression (for --markers)
python code/wmb_expression.py --summary   # Print cached results
# Runner: bash code/runners/supporting/run_wmb_expression.sh

# Song snRNA-seq Integration (within-cohort evidence from paired animals)
python code/snrna_integration.py --pseudobulk    # S1: pseudobulk from 170 h5ad (28 animals → ~21 of 34 WMB classes)
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
python code/build_unified_viewer.py       # Interactive HTML viewer (kinase + pathway + cross-entity)
python code/lucie_5xfad_manifest.py       # Lucie 5xFAD proteomics manifest builder
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
- **Sample filtering**: `ANALYSIS_MODE` env var (default `males_only`); IRS normalization always uses all 72 samples, filtering applies at OLS time
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
kinase_enrich.py / kinase_attribute.py + snrna_integration.py  ←  code/integration/
```

### Live Code

- `config.py` — Shared configuration: file paths, thresholds, enrichment method params, `WMB_CLASSES` list (34 WMB classes, single source of truth for the cell-type spine), `SEA_AD_SUBCLASSES` (transitional, used only by Incytr-integration adapters and supplementary scripts), sample filtering params (`OUTLIER_ZSCORE_THRESH`, `ANALYSIS_MODE`). Still structurally mixed with legacy settings.
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

`code/integration/` is mid-rewrite. The legacy `wrappers/`, `adapters/`, `sidecar/`, `tests/`, and orchestrator shell scripts were relocated on 2026-05-08 to `~/Projects/work/incytr_integration_archive/` (see `code/integration/MOVED.txt`). The remediation plan is at `docs/incytr_remediation_plan.md`; the new architecture replaces the shadow-fork wrapper with a thin AD-specific shell that calls the upstream `incytr` R package directly.

What remains in-tree:

- `config_integration.py` — paths, thresholds, contrast definitions (kept by the remediation plan)
- `factorial.R`, `load.R`, `persist.R`, `views.sql`, `run_factorial.sh` — Phase 1 stubs for the new architecture (incomplete; awaiting the production package API in `../incytr`)
- `intermediates/` — gitignored output dir from the legacy pipeline (orphaned)

R deps (`Incytr`, `DBI`, `duckdb`, `data.table`, `arrow`) are still required by the Phase 1 stubs.

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
`code/deconvolution/build_wmb_decomposition.py` consumes per-(site, group) bulk medians (`imac_median.csv`, `py_median.csv`, `pr_median.csv`) and the `yuyu_samplekey.csv` MS\_ID↔SCRNA bridge. These were deleted from `data/datasets/song/proteomics/source/` on 2026-05-07; re-pull from Google Drive via `pixi run ingest-gdrive-shared` before running the branch. Outputs: `outputs/reports/deconvolution/wmb_decomposition/{ps,py,pr}_wmb_decomposition.csv` + `wmb_class_size.csv`.

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
- `repo_surface_index.md` — Exhaustive main/supporting/archived classification of every file

Other documentation:
- `docs/integrations/kinase_incytr_integration.md` — Source of truth for the kinase ↔ Incytr integration (scoring model, runtime modes, configuration, outputs, limitations)
- `docs/archive/legacy.md` — Legacy proportional decomposition method (historical reference)
- `archive/deconvolution/docs/deconvolution_infeasibility.md` — Archived synthetic validation proving direct deconvolution is infeasible on this dataset (closed path, frozen). Source script + figures alongside under `archive/deconvolution/`.

## Gotchas

- **ANALYSIS_MODE controls sample filtering** — defaults to `males_only`. Set `ANALYSIS_MODE=full_cohort` for sensitivity analysis with both sexes. The mode affects `kinase_enrich.py`, `kinase_attribute.py`, and `kinase_mechanism.py` but NOT `kinase_normalize.py` (which always uses all 72 samples)
- **Outlier detection requires stoichiometry** — `data_ingest.py --outliers` reads `stoichiometry_matrix.csv`, so `kinase_normalize.py` must be run first. Falls back to total proteome if unavailable
- **Limited automated tests** — live pipeline has no unit tests; verify with `python code/kinase_summary.py`
- **Song proteomics files must be mounted** — data_ingest.py reads Excel workbooks from `data/datasets/song/primary/proteomics/`
- **WMB prerequisite** — `run_live_pipeline.sh` gates on `wmb_kinase_expression.csv` and `wmb_proteome_expression.csv`; run `run_wmb_expression.sh` first
- **Atlas cache compressed** — raw h5ad files under `data/external/allen_abc/` are zstd-compressed to save space (~115 GB → ~26 GB). Decompress with `bash code/runners/supporting/decompress_atlas_cache.sh` before re-running `wmb_expression.py`. See `data/external/allen_abc/MANIFEST.json` for provenance
- **SEA-AD data required** — Unified attribution needs SEA-AD effect sizes under `config.SEA_AD_DIR`
- **API caching** — delete files under `data/datasets/song/analysis_cache/` to force re-fetch
- **WMB expression memory** — `wmb_expression.py --proteome` processes 6,308 genes across 13 regions; use `skip_regional=True` and `chunk_size=2000` to avoid OOM (~30GB RAM available)
- **Do not reopen closed paths** — direct deconvolution, factor model, two-compartment, and transcript-only rescue are all closed (see charter)
- **Integration tree is mid-rewrite** — legacy R wrappers and Python adapters moved to `~/Projects/work/incytr_integration_archive/` on 2026-05-08; see `code/integration/MOVED.txt` and `docs/incytr_remediation_plan.md`. The Phase 1 stubs (`factorial.R`, `load.R`, `persist.R`, `views.sql`, `run_factorial.sh`) remain in-tree. R deps (`Incytr`, `DBI`, `duckdb`, `data.table`, `arrow`) are still required. Config lives in `config_integration.py`, not `config.py`

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
