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
- [`docs/foundation/repo_surface_index.md`](docs/foundation/repo_surface_index.md): explicit `main` / `supporting` / `archived` file inventory
- [`docs/foundation/incytr_integration_methodology.md`](docs/foundation/incytr_integration_methodology.md): Incytr integration design and methodology
- [`docs/sap_document_map.md`](docs/sap_document_map.md): map of the live documentation set
- [`docs/integrations/integrations-structure.md`](docs/integrations/integrations-structure.md): upstream archive layout and current operational data locations
- [`docs/integrations/alzheimers-incytr-input-validation.md`](docs/integrations/alzheimers-incytr-input-validation.md): current input mapping and validation notes for Song and 5xFAD
- [`docs/report_writing_checklist.md`](docs/report_writing_checklist.md): report-writing guidance derived from reviewer feedback

Historical context lives under `archive/`.

## Current Analysis Program

The live workflow has two major components:

### 1. Stoichiometry-corrected kinase attribution (bulk pipeline)

1. **Data ingestion** (`data_ingest.py`) -- TMT channel mapping, phosphosite-to-protein matching (91.7%), marker assessment, PCA quality control, statistical outlier detection
2. **Kinase attribution** (`kinase_attribution.py`) -- IRS cross-plex normalization (all 72 samples), stoichiometry computation, factorial OLS per site with disease x timepoint interactions (9 time-resolved contrasts), MEA kinase enrichment on median-centered + winsorized stoichiometry beta values, unified cell-type attribution combining SEA-AD concordance and WMB expression specificity
3. **Attribution recovery** (`attribution_recovery.py`) -- Cross-contrast consistency analysis and final attribution table assembly (kinase activity matrix, cell-type evidence table, kinase hypothesis table, cell-type kinase profiles)

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
│   ├── code/                       # Archived Python/R code (SAP, enrichment, side analyses)
│   ├── deconv/                     # Archived benchmark and transition workspace
│   ├── runners/                    # Archived validation and side-workflow runners
│   └── sap_docs/                   # Archived SAP design, atlas, and transition notes
├── code/
│   ├── runners/
│   │   ├── main/                   # Main pipeline stage runners
│   │   ├── supporting/             # Supporting setup runners
│   │   └── supplementary/          # Reviewer diagnostic runners
│   ├── integration/                # Incytr intercellular signaling integration
│   │   ├── adapters/               # Python: snRNA-seq export, kldata, phospho, kinase support scoring
│   │   ├── wrappers/               # R: DuckDB enumeration, Incytr orchestration, bootstrap
│   │   ├── tests/                  # Integration tests
│   │   ├── config_integration.py   # Integration-specific configuration
│   │   ├── run_phase1.sh           # Single-pair runner (reference: Microglia-PVM -> L5 IT)
│   │   └── run_all_pairs.sh        # All-pairs runner (462 pairs, systemd-run memory guard)
│   ├── supplementary/              # Reviewer-response diagnostic analyses
│   ├── data_ingest.py              # Main: data ingestion + characterization
│   ├── kinase_attribution.py       # Main: stoichiometry + MEA + unified attribution
│   ├── attribution_recovery.py     # Main: attribution recovery + final table assembly
│   ├── plot_attribution_bubbles.py # Main: attribution visualizations
│   ├── config.py                   # Supporting: shared configuration
│   ├── atlas_reference.py          # Supporting: external-reference prep (SEA-AD, WMB, Aging Mouse)
│   ├── wmb_expression.py           # Supporting: WMB expression export
│   ├── snrna_integration.py        # Supporting: Song within-cohort snRNA-seq evidence
│   ├── map_kinases_to_genes.py     # Supporting: kinase-gene mapping utility
│   ├── build_viewer.py             # Standalone: interactive HTML viewer
│   └── lucie_5xfad_manifest.py     # Standalone: 5xFAD integration/provenance
├── data/
│   ├── incytr_collections/song/    # Authoritative localized Song workspace
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

## Environment Setup

Create and activate the Python environment:

```bash
mamba env create -f environment.yml
mamba activate alzheimers
```

The `alzheimers` environment requires Python 3.11 (kinase-library compatibility) and includes `kinase-library`, `anndata`, `gseapy`, `scikit-learn`, `matplotlib`, `seaborn`, `scipy`, `requests`, `natsort`.

The Incytr integration additionally requires a separate R environment (`incytr`) with the Incytr package and DuckDB. See `code/integration/run_phase1.sh` for the dual-environment orchestration.

If you need the study data mounts:

```bash
bash scripts/setup_gdrive_mounts.sh
```

## Running the Analysis

All scripts run from the repo root.

### Bulk Pipeline

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

Or run modules directly with `--run` (all stages) or individual flags:

```bash
python code/data_ingest.py --run          # all ingestion steps
python code/kinase_attribution.py --run   # normalize + enrich + attribute
python code/attribution_recovery.py --run # hypothesis tables
```

Use `--summary` on any module to print cached results without recomputing.

The `ANALYSIS_MODE` environment variable controls sample filtering (default: `males_only`). Set `ANALYSIS_MODE=full_cohort` for sensitivity analysis with both sexes. This affects `--enrich`, `--attribute`, and `--mechanism-annotation` but NOT `--normalize` (which always uses all 72 samples).

### Supporting Prerequisites

These must be run before the bulk pipeline if their outputs don't exist:

```bash
# External Atlas Data Acquisition (SEA-AD + WMB + Aging Mouse)
bash code/runners/supporting/run_atlas_reference.sh

# WMB Expression Export (required for unified attribution + marker assessment)
bash code/runners/supporting/run_wmb_expression.sh

# Song snRNA-seq Integration (within-cohort evidence from paired animals)
bash code/runners/supporting/run_snrna_integration.sh
```

### Incytr Integration Pipeline

Connects bulk kinase activity to intercellular signaling pathways. Requires the bulk pipeline to have been run first (needs MEA results and unified attribution).

**Single-pair** (reference pair: Microglia-PVM -> L5 IT):

```bash
cd code/integration
bash run_phase1.sh
```

**All 462 sender-receiver pairs:**

```bash
cd code/integration
bash run_all_pairs.sh              # full run: Python adapters + R pipeline + kinase support
bash run_all_pairs.sh --skip-adapters  # checkpoint-resume (skip Python export)
```

Environment variables for the all-pairs runner:

| Variable | Default | Description |
|---|---|---|
| `PAIR_FILTER` | (all pairs) | Filter pairs, e.g. `"Microglia-PVM:L5 IT"` or `"*:L5 IT"` |
| `FORCE_RERUN` | `0` | Set to `1` to reprocess pairs with existing results |
| `ENABLE_KINASE_IMPUTATION` | `1` | Set to `0` to disable kinase-imputed pathway expansion |
| `RUN_PERMUTATIONS` | `0` | Set to `1` to run dual null model permutation tests |
| `RUN_BOOTSTRAP` | `0` | Set to `1` to run L5 IT bootstrap sensitivity analysis |
| `MEMORY_LIMIT_GB` | `10` | R memory guard threshold |

The all-pairs pipeline runs under `systemd-run --user --scope -p MemoryMax=12G`.

### Supplementary Diagnostics

Reviewer-response analyses that validate pipeline choices. Run after the bulk pipeline:

```bash
bash code/runners/supplementary/run_reviewer_diagnostics.sh
```

## Key Outputs

### Bulk Pipeline (`outputs/reports/`)

| Directory | Contents |
|---|---|
| `data_ingest/` | Sample mapping, phospho-protein matching, PCA plots, `sample_exclusions.csv` |
| `kinase_attribution/` | Stoichiometry matrix, MEA enrichment (`mea_stoichiometry.csv`), unified attribution (`unified_attribution.csv`), site-level OLS, winsorization logs |
| `attribution_recovery/` | **Primary deliverables**: `kinase_hypothesis_table.csv` (kinase-first synthesis), `kinase_activity_matrix.csv`, `celltype_evidence_table.csv`, `celltype_kinase_profiles.csv`, bubble plots |
| `atlas_reference/` | Atlas structure reports, taxonomy mapping, kinase gene coverage |
| `wmb_expression/` | WMB per-subclass kinase/phosphatase expression matrix |
| `snrna_integration/` | Song pseudobulk, within-cohort specificity, concordance |

### Incytr Integration (`code/integration/intermediates/`)

**Per pair** (`all_pairs/{sender}__{receiver}/`, 462 subdirectories):

| File | Description |
|---|---|
| `results_full.csv` | Full integrated scores (PDS + kinase_boost) |
| `results_expronly.csv` | Expression-only scores (TPDS baseline) |
| `kinase_support_scores.csv` | Per-pathway substrate-based kinase support scores |
| `adjusted_rankings.csv` | Lambda-sweep adjusted rankings (TPDS + lambda x kinase_support) |
| `edge_list_l{1,2,3}.csv` | Per-layer edge lists with pathway counts |
| `reranking_summary.json` | Per-pair scoring statistics |

**Cross-pair** (`all_pairs/`):

| File | Description |
|---|---|
| `pair_summary.csv` | 462-row summary: sender, receiver, n_pathways, timing, status |
| `kinase_support_summary.csv` | 462-row kinase support summary |

All results include `pathway_evidence` (expression-confirmed or kinase-imputed), `imputed_nodes`, and `kinase_boost` (PDS - TPDS) columns.

## Data Surfaces

### Song workspace

The active Song dataset lives under `data/incytr_collections/song/`. Treat this as the authoritative local workspace. Upstream collaborator material under `data/gdrive_shared/` is archive and provenance, not a runtime dependency.

### 5xFAD integrations

- `data/lucie_proteomics/` contains upstream source files
- [`docs/integrations/5xfad-lucie-manifest.json`](docs/integrations/5xfad-lucie-manifest.json) inventories local Lucie files
- [`docs/integrations/alzheimers-incytr-input-validation.md`](docs/integrations/alzheimers-incytr-input-validation.md) documents input mapping

## Conventions

- Use the `docs/foundation/` documents as the live analytical contract
- Use `docs/foundation/repo_surface_index.md` when deciding whether a file is `main`, `supporting`, or `archived`
- Prefer `data/incytr_collections/song/` over ad hoc files elsewhere in `data/`
- Treat `archive/` as provenance and history, not as the default source of live methods
- The integration pipeline is hypothesis-generating: frame results as convergent functional evidence, not mechanistic pathway validation
