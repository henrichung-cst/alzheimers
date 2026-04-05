# Alzheimer's Kinase Analysis

This repository is the study workspace for Alzheimer's multi-omic kinase and signaling analysis.

The current documentation in `docs/` defines the live program as a 72-sample, stoichiometry-enabled, mechanism-stratified attribution workflow. The localized Song workspace under `data/incytr_collections/song/` is the operational dataset surface. Upstream collaborator material remains under `data/gdrive_shared/`, and retired deconvolution and older SAP records are preserved under `archive/`.

The repo now keeps executable code under `code/` or `archive/`. The labels `main`, `supporting`, and `archived` are documented classifications, not new top-level directories.

## Read This First

Use the documentation by role, not by creation date:

- [`docs/sap_document_map.md`](docs/sap_document_map.md): map of the live documentation set
- [`docs/foundation/analysis_charter.md`](docs/foundation/analysis_charter.md): single source of truth for the active analysis program
- [`docs/foundation/analysis_rationale.md`](docs/foundation/analysis_rationale.md): why the project pivoted away from direct deconvolution
- [`docs/foundation/statistical_constraints.md`](docs/foundation/statistical_constraints.md): identifiability limits and interpretation guardrails
- [`docs/foundation/repo_surface_index.md`](docs/foundation/repo_surface_index.md): explicit `main` / `supporting` / `archived` file inventory
- [`docs/foundation/live_pipeline_contract.md`](docs/foundation/live_pipeline_contract.md): exact live stage contract, outputs, and ordered run sequence
- [`docs/integrations/integrations-structure.md`](docs/integrations/integrations-structure.md): upstream archive layout and current operational data locations
- [`docs/integrations/alzheimers-incytr-input-validation.md`](docs/integrations/alzheimers-incytr-input-validation.md): current input mapping and validation notes for Song and 5xFAD
- [`docs/integrations/5xfad-lucie-manifest.json`](docs/integrations/5xfad-lucie-manifest.json): machine-readable inventory of local 5xFAD Lucie proteomics files
- [`docs/report_writing_checklist.md`](docs/report_writing_checklist.md): report-writing guidance derived from reviewer feedback

Historical context lives under `archive/`:

- [`archive/sap_docs/transitional_notes/sap_primary_path_summary.md`](archive/sap_docs/transitional_notes/sap_primary_path_summary.md)
- [`archive/sap_docs/atlas_working_notes/sap_atlas_part4.md`](archive/sap_docs/atlas_working_notes/sap_atlas_part4.md)
- [`archive/sap_docs/legacy_design/sap_24group_identifiability_record.md`](archive/sap_docs/legacy_design/sap_24group_identifiability_record.md)
- [`archive/deconv/docs/deconvolution-transition-aobs-desp.md`](archive/deconv/docs/deconvolution-transition-aobs-desp.md)

## Current Analysis Program

The live workflow described in `docs/foundation/` is:

1. integrate the 72-animal total proteome,
2. compute phospho-to-protein stoichiometry,
3. split kinase findings into abundance-driven, both, and activity-driven classes,
4. run Track A attribution on abundance-coupled classes,
5. run Track B attribution on activity-driven classes,
6. assemble the final attribution table.

Important guardrails from the foundation docs:

- direct cell-type deconvolution from the original 24-group design is a closed path
- transcript-only rescue is a closed path
- factor-model and two-compartment rescue branches are archived, not live methods
- transcriptomic concordance should not be used on activity-driven kinase sets
- old deconvolution outputs should be treated as provenance only

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
│   │   └── supporting/             # Supporting setup and demoted mixed runners
│   ├── data_ingest.py              # Main: data ingestion + characterization
│   ├── kinase_attribution.py       # Main: stoichiometry + attribution
│   ├── attribution_recovery.py     # Main: attribution recovery + final table assembly
│   ├── config.py                   # Supporting: shared configuration
│   ├── atlas_reference.py          # Supporting: external-reference prep
│   ├── wmb_expression.py           # Supporting: WMB expression export for Track B
│   ├── map_kinases_to_genes.py     # Supporting: kinase-gene mapping utility
│   └── lucie_5xfad_manifest.py     # Supporting: 5xFAD integration/provenance
├── data/
│   ├── incytr_collections/song/    # Authoritative localized Song workspace
│   ├── gdrive_shared/              # Upstream collaborator archive mounts
│   └── lucie_proteomics/           # Local 5xFAD upstream proteomics sources
├── docs/
│   ├── foundation/                 # Live analysis charter, rationale, constraints
│   └── integrations/               # External dataset mapping and validation notes
├── outputs/                        # Generated outputs and reports
├── scripts/setup_gdrive_mounts.sh  # Mount helper for upstream data
├── environment.yml
└── README.md
```

## Environment Setup

Create and activate the mixed Python/R environment:

```bash
mamba env create -f environment.yml
mamba activate alzheimers
```

Install the local `incytr` package into that environment only if you need the archived InCytr compatibility workflows:

```bash
mamba run -n alzheimers Rscript -e 'devtools::install("/home/hchung/Projects/work/incytr", dependencies=FALSE)'
```

If you need the study data mounts:

```bash
bash scripts/setup_gdrive_mounts.sh
```

## Data Surfaces

### Song workspace

The active Song dataset lives under `data/incytr_collections/song/`.

Important references:

- [`data/incytr_collections/song/INDEX.md`](data/incytr_collections/song/INDEX.md)
- [`data/incytr_collections/song/primary/INDEX.md`](data/incytr_collections/song/primary/INDEX.md)

Operational rules:

- treat `data/incytr_collections/song/` as the authoritative local workspace
- treat `data/gdrive_shared/yuyu01/` as upstream archive and provenance, not the default runtime dependency
- treat `data/incytr_collections/song/proteomics/legacy/` as preserved collaborator outputs
- treat the regenerated `pr` / `ps` / `py` files in `data/incytr_collections/song/proteomics/` as the active Song proteomics bundle

### 5xFAD integrations

The current reference for 5xFAD input mapping is [`docs/integrations/alzheimers-incytr-input-validation.md`](docs/integrations/alzheimers-incytr-input-validation.md).

Key points:

- `data/lucie_proteomics/` contains upstream source files, not direct InCytr-ready inputs
- `docs/integrations/5xfad-lucie-manifest.json` inventories the local Lucie `.sne` files
- the packaged InCytr-ready 5xFAD inputs are treated as integration targets and comparison material, not as raw provenance

## Main Workflow

Use the bundled live runner as the main front door:

```bash
bash code/runners/main/run_live_pipeline.sh
```

Equivalent explicit ordered sequence:

```bash
bash code/runners/main/run_data_ingest.sh
bash code/runners/main/run_kinase_attribution.sh
bash code/runners/main/run_attribution_recovery.sh
```

Equivalent direct module entry points:

```bash
python code/data_ingest.py --run
python code/kinase_attribution.py --run
python code/attribution_recovery.py --run
```

## Supporting Setup

These are retained because the main pipeline consumes their outputs or context, but they are not co-equal front doors:

```bash
bash code/runners/supporting/run_atlas_reference.sh
```

The WMB expression export is a standalone supporting surface at `code/wmb_expression.py`. The live pipeline expects `outputs/reports/wmb_expression/wmb_kinase_expression.csv` to exist before kinase attribution Track B runs.

```bash
bash code/runners/supporting/run_wmb_expression.sh
```

## Demoted And Archived Workflows

Legacy compatibility builders, kinase-enrichment workflows, local InCytr examples, and archived SAP validation runners remain in the repository for provenance and reproducibility, but they are not part of the main front door. Use [`docs/foundation/repo_surface_index.md`](docs/foundation/repo_surface_index.md) for the current classification before invoking any non-Module-6 surface.

## Outputs

The primary live deliverables live under `outputs/reports/`, especially:

- `outputs/reports/data_ingest/`
- `outputs/reports/kinase_attribution/`
- `outputs/reports/attribution_recovery/`
- `outputs/reports/attribution_recovery/final_attribution_table.csv`

Supporting and archived outputs may remain elsewhere under `outputs/`, but they should not be treated as the primary current-state deliverables.

## Conventions

- use the `docs/foundation/` documents as the live analytical contract
- use `docs/foundation/repo_surface_index.md` when deciding whether a file is `main`, `supporting`, or `archived`
- use `docs/integrations/` when the question is about external bundles, input mapping, or provenance
- prefer `data/incytr_collections/song/` over ad hoc files elsewhere in `data/`
- treat `archive/` as provenance and history, not as the default source of live methods
