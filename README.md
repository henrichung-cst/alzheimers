# Alzheimer's Kinase Analysis

Stoichiometry-corrected kinase attribution pipeline for Alzheimer's disease phosphoproteomics, with intercellular signaling integration via Incytr.

Integrates 72-animal TMT total proteome with phosphoproteomics to compute stoichiometry (log2 phospho - log2 protein), runs MEA (GSEA-based) kinase enrichment on stoichiometry beta values, and attributes findings to cell types using unified evidence from SEA-AD transcriptomic concordance, WMB expression specificity, and within-cohort snRNA-seq concordance. A second integration layer connects bulk kinase activity to cell-cell signaling pathways via Incytr's four-gene pathway model across 462 sender-receiver cell-type pairs.

Primary analysis uses males-only (33 animals after outlier exclusion) to avoid hormonal confounds; full-cohort analysis is run as a sensitivity check.

## Read This First

Use the documentation by role, not by creation date:

- [`docs/foundation/analysis_charter.md`](docs/foundation/analysis_charter.md): single source of truth for the active analysis program
- [`docs/foundation/analysis_rationale.md`](docs/foundation/analysis_rationale.md): why the project pivoted away from direct deconvolution
- [`docs/foundation/live_pipeline_contract.md`](docs/foundation/live_pipeline_contract.md): exact live stage contract, outputs, and ordered run sequence
- [`docs/foundation/statistical_constraints.md`](docs/foundation/statistical_constraints.md): identifiability limits and interpretation guardrails
- [`docs/foundation/repo_retention_policy.md`](docs/foundation/repo_retention_policy.md): explicit `main` / `supporting` / `archived` file inventory
- [`docs/integrations/kinase_incytr_integration.md`](docs/integrations/kinase_incytr_integration.md): kinase ↔ Incytr integration — Phase 1 stub inventory + handoff diagram
- [`docs/incytr_remediation_plan.md`](docs/incytr_remediation_plan.md): authoritative architectural plan for the integration rewrite
- [`docs/INDEX.md`](docs/INDEX.md): map of the live documentation set
- [`docs/report_writing_checklist.md`](docs/report_writing_checklist.md): report-writing guidance derived from reviewer feedback

Historical context lives under `archive/`.

## Current Analysis Program

The live workflow has two major components:

### 1. Stoichiometry-corrected kinase attribution (bulk pipeline)

1. **Data ingestion** (`data_ingest.py`) -- TMT channel mapping, phosphosite-to-protein matching (91.7%), marker assessment, PCA quality control, statistical outlier detection
2. **Normalize → enrich → attribute** (`kinase_normalize.py` → `kinase_enrich.py` → `kinase_attribute.py`, each a CLI shim over the corresponding Kedro pipeline under `alz/pipelines/`) -- IRS cross-plex normalization (all 72 samples), stoichiometry computation, factorial OLS per site with disease x timepoint interactions (9 time-resolved contrasts), MEA kinase enrichment on median-centered + winsorized stoichiometry beta values, unified cell-type attribution combining SEA-AD concordance and WMB expression specificity
3. **Attribution recovery** (`attribution_recovery.py`, CLI shim over the `recovery` Kedro pipeline) -- Cross-contrast consistency analysis and final attribution table assembly (kinase activity matrix, cell-type evidence table, kinase hypothesis table)

### 2. Intercellular signaling integration (Incytr pipeline)

Connects bulk kinase activity evidence to cell-cell signaling pathways. Incytr models each intercellular signaling pathway as a four-gene chain (Ligand -> Receptor -> EM -> Target) between sender and receiver cell types, scored by transcriptomic co-expression (TPDS). The integration adds kinase evidence through two complementary channels:

- **Internal channel**: kinases that pass the expression threshold become pathway nodes and are scored by Incytr's built-in kinase enrichment
- **External channel**: substrate-based reranking connects kinase activity to pathway endpoints through kinase-substrate relationships, weighted by cell-type attribution confidence and inverse-frequency (IDF) weighting to control for substrate promiscuity

The pipeline runs across 462 sender-receiver pairs (22 subclasses x 21, excluding self-pairs) for the App_4mo contrast, using DuckDB-based enumeration with in-query SigProb filtering at a 10% expression detection threshold.

Important guardrails:

- Direct cell-type deconvolution from the original 24-group design is a closed path
- Transcript-only rescue is a closed path
- Factor-model and two-compartment rescue branches are archived, not live methods
- Old deconvolution outputs should be treated as provenance only
- Cell-type attribution of kinase activity is correlational; the integration is hypothesis-generating, not mechanistic validation

## Repository Layout

```text
alzheimers/
├── archive/
│   ├── alz/                       # Archived Python/R code (SAP, enrichment, side analyses)
│   ├── deconv/                     # Archived benchmark and transition workspace
│   ├── runners/                    # Archived validation and side-workflow runners
│   └── sap_docs/                   # Archived SAP design, atlas, and transition notes
├── alz/
│   ├── runners/
│   │   ├── main/                   # Main pipeline stage runners
│   │   ├── supporting/             # Supporting setup runners
│   │   └── supplementary/          # Reviewer diagnostic runners
│   ├── integration/                # Incytr integration — thin upstream client (see docs/incytr_remediation_plan.md)
│   │   ├── config_integration.py   # Filter/design/contrast constants
│   │   ├── export_factorial_inputs.py  # h5ad → on-disk contract for load.R
│   │   ├── factorial.R, load.R, persist.R, views.sql, run_factorial.sh
│   │   │                            # Entry point + AD loader + parquet writer + DuckDB views
│   │   └── README.md               # File-by-file layout
│   ├── supplementary/              # Reviewer-response diagnostic analyses
│   ├── pipelines/                  # Kedro pipelines (live arc + ingest_mapping)
│   │   ├── ingest_mapping/         # TMT channel-to-animal mapping
│   │   ├── normalize/              # IRS normalization + stoichiometry
│   │   ├── enrich/                 # Factorial OLS + MEA kinase enrichment
│   │   ├── attribute/              # Unified cell-type attribution
│   │   ├── mechanism/              # Optional: raw-phospho MEA + classification
│   │   └── recovery/               # Hypothesis-table assembly
│   ├── pipeline_registry.py        # Kedro pipeline registry
│   ├── settings.py                 # Kedro settings
│   ├── data_ingest.py              # Main: data ingestion + characterization (CLI shim)
│   ├── kinase_normalize.py         # Main: stoichiometry (CLI shim over normalize pipeline)
│   ├── kinase_enrich.py            # Main: MEA kinase enrichment (CLI shim over enrich pipeline)
│   ├── kinase_attribute.py         # Main: unified attribution (CLI shim over attribute pipeline)
│   ├── kinase_mechanism.py         # Supplementary: mechanism classification (CLI shim)
│   ├── attribution_recovery.py     # Main: hypothesis-table assembly (CLI shim over recovery pipeline)
│   ├── plot_attribution_bubbles.py # Main: attribution visualizations
│   ├── config.py                   # Supporting: shared configuration
│   ├── atlas_reference.py          # Supporting: external-reference prep (SEA-AD, WMB, Aging Mouse)
│   ├── wmb_expression.py           # Supporting: WMB expression export
│   ├── snrna_integration.py        # Supporting: Song within-cohort snRNA-seq evidence
│   ├── map_kinases_to_genes.py     # Supporting: kinase-gene mapping utility
│   ├── build_unified_viewer.py     # Standalone: unified kinase + pathway HTML viewer
│   └── lucie_5xfad_manifest.py     # Standalone: 5xFAD integration/provenance
├── data/
│   ├── datasets/song/    # Authoritative localized Song workspace
│   ├── external/                   # Allen Brain Atlas (WMB, SEA-AD) cached data
│   ├── gdrive_shared/              # Upstream collaborator archive mounts
│   └── lucie_proteomics/           # Local 5xFAD upstream proteomics sources
├── docs/
│   ├── foundation/                 # Live analysis charter, rationale, constraints, integration methodology
│   └── integrations/               # External dataset mapping and validation notes
├── outputs/                        # Generated outputs and reports
├── scripts/setup_gdrive_mounts.sh  # Mount helper for upstream data
├── environment.yml
└── README.md
```

## External Data Compression

The Allen Brain Atlas and SEA-AD reference data under `data/external/` is zstd-compressed at rest to reduce disk usage. The live pipeline reads only small pre-computed outputs (effect-size h5ad files + CSV expression matrices), not the raw atlas h5ad files.

| Tier | Contents | Uncompressed | Compressed | Notes |
|---|---|---|---|---|
| 1 | WMB full regional h5ad (13 regions) | 89 GB | ~24 GB | Redundant when subsets exist |
| 2 | Unused SEA-AD cell-level h5ad | 23.5 GB | ~5 GB | Not referenced by any pipeline code |
| 3 | WMB gene-subset h5ad (13 regions) | 51 GB | ~14 GB | Auto-decompressed by `wmb_expression.py` |

**Kept uncompressed** (runtime dependencies): `effect_sizes.h5ad`, `effect_sizes_early.h5ad`, `effect_sizes_late.h5ad` (read by `alz/kinase_attribute.py`), and `cell_metadata_with_cluster_annotation.csv` (read by ABC cache API).

**Auto-decompress:** `wmb_expression.py --run` and `--proteome` transparently decompress tier 3 subset files before computation and recompress them afterward. A sentinel file ensures cleanup even after interruption.

**Manual control:**

```bash
bash alz/runners/supporting/compress_atlas_cache.sh [tier1|tier2|tier3|WMB|sea_ad|subset]
bash alz/runners/supporting/decompress_atlas_cache.sh [WMB|subset|sea_ad|Aging]
```

## Environment Setup

Create and activate the Python environment:

```bash
mamba env create -f environment.yml
mamba activate alzheimers
```

The `alzheimers` environment requires Python 3.11 (kinase-library compatibility) and includes `kinase-library`, `anndata`, `gseapy`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`, `requests`, `natsort`.

The Incytr integration is now a thin upstream client (see `docs/incytr_remediation_plan.md`); the legacy source is preserved under `archive/incytr_integration/`. The wrapper still requires an R environment with `Incytr`, `DBI`, `duckdb`, `data.table`, `arrow`, and is gated on the production engine landing in `../incytr` per Phase 1.

If you need the study data mounts:

```bash
bash scripts/setup_gdrive_mounts.sh
```

## Running the Analysis

All scripts run from the repo root. See [`docs/foundation/live_pipeline_contract.md`](docs/foundation/live_pipeline_contract.md) for the full stage-by-stage spec (inputs, outputs, flags, failure modes).

### Bulk Pipeline

```bash
bash alz/runners/main/run_live_pipeline.sh     # all three stages in order
bash alz/runners/main/run_dual_analysis.sh     # males-only (primary) + full-cohort (sensitivity)
```

Individual stages (`run_data_ingest.sh`, `run_kinase_attribution.sh`, `run_attribution_recovery.sh`) and module-level flags (`--run`, `--summary`, per-step flags) are documented in `CLAUDE.md`.

Sample filtering is the `analysis_mode` Kedro parameter in `conf/base/parameters.yml` (default: `males_only`). Use `KEDRO_ENV=full_cohort` to overlay `conf/full_cohort/parameters.yml` for sensitivity analysis with both sexes.

### Supporting Prerequisites

Run these before the bulk pipeline if their outputs don't exist:

```bash
bash alz/runners/supporting/run_atlas_reference.sh   # SEA-AD + WMB + Aging Mouse
bash alz/runners/supporting/run_wmb_expression.sh     # WMB expression export
bash alz/runners/supporting/run_snrna_integration.sh  # Song within-cohort snRNA-seq
```

### Incytr Integration Pipeline

Thin upstream client. The legacy 462-pair × 9-contrast runner is preserved under `archive/incytr_integration/run_factorial_all_pairs.sh`; in-tree only the new wrapper remains. See [`docs/incytr_remediation_plan.md`](docs/incytr_remediation_plan.md) for the target architecture and [`alz/integration/README.md`](alz/integration/README.md) for the file-by-file layout.

### Supplementary Diagnostics

Reviewer-response analyses that validate pipeline choices. Run after the bulk pipeline:

```bash
bash alz/runners/supplementary/run_reviewer_diagnostics.sh
```

## Key Outputs

### Bulk Pipeline (`outputs/reports/`)

| Directory | Contents |
|---|---|
| `data_ingest/` | Sample mapping, phospho-protein matching, PCA plots, `sample_exclusions.csv` |
| `kinase_attribution/` | Stoichiometry matrix, MEA enrichment (`mea_stoichiometry.csv`), unified attribution (`unified_attribution.csv`), site-level OLS, winsorization logs |
| `attribution_recovery/` | **Primary deliverables**: `kinase_hypothesis_table.csv` (kinase-first synthesis), `kinase_activity_matrix.csv`, `celltype_evidence_table.csv`, bubble plots |
| `atlas_reference/` | Atlas structure reports, taxonomy mapping, kinase gene coverage |
| `wmb_expression/` | WMB per-subclass kinase/phosphatase expression matrix |
| `snrna_integration/` | Song pseudobulk, within-cohort specificity, concordance |

### Incytr Integration

Outputs are not currently regenerable in-tree. The legacy `alz/integration/intermediates/` is gitignored and orphaned by the rewrite; the new architecture targets `outputs/reports/incytr_factorial/` (not yet wired up).

## Data Surfaces

### Song workspace

The active Song dataset lives under `data/datasets/song/`. Treat this as the authoritative local workspace. Upstream collaborator material under `data/gdrive_shared/` is archive and provenance, not a runtime dependency.

### 5xFAD integrations

- `data/lucie_proteomics/` contains upstream source files
- [`docs/integrations/5xfad-lucie-manifest.json`](docs/integrations/5xfad-lucie-manifest.json) inventories local Lucie files
- [`docs/archive/alzheimers-incytr-input-validation.md`](docs/archive/alzheimers-incytr-input-validation.md) documents input mapping (archived session audit)

## Conventions

- Use the `docs/foundation/` documents as the live analytical contract
- Use `docs/foundation/repo_retention_policy.md` when deciding whether a file is `main`, `supporting`, or `archived`
- Prefer `data/datasets/song/` over ad hoc files elsewhere in `data/`
- Treat `archive/` as provenance and history, not as the default source of live methods
- The integration pipeline is hypothesis-generating: frame results as convergent functional evidence, not mechanistic pathway validation
