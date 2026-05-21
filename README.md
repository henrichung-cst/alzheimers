# Alzheimer's Kinase Analysis

Stoichiometry-corrected kinase attribution pipeline for Alzheimer's disease phosphoproteomics, with intercellular-signaling integration via Incytr (pair-mode).

Integrates 72-animal TMT total proteome with phosphoproteomics to compute stoichiometry (`log2 phospho − log2 protein`), runs MEA (GSEA-based) kinase enrichment on stoichiometry β values, and attributes findings to cell types using unified evidence from SEA-AD transcriptomic concordance, WMB expression specificity, and within-cohort Song snRNA-seq concordance. A second layer connects bulk kinase activity to cell-cell signaling pathways through Incytr's four-gene pathway model.

Primary analysis uses males-only (33 animals after outlier exclusion) to avoid hormonal confounds; full-cohort analysis is run as a sensitivity check.

## Read This First

Documentation is organized by role, not by creation date:

- [`docs/foundation/analysis_charter.md`](docs/foundation/analysis_charter.md) — single source of truth for the active analysis program
- [`docs/foundation/analysis_rationale.md`](docs/foundation/analysis_rationale.md) — why the project pivoted away from direct deconvolution
- [`docs/foundation/live_pipeline_contract.md`](docs/foundation/live_pipeline_contract.md) — stage-by-stage inputs, outputs, and run order
- [`docs/foundation/statistical_constraints.md`](docs/foundation/statistical_constraints.md) — identifiability limits and interpretation guardrails
- [`docs/foundation/repo_retention_policy.md`](docs/foundation/repo_retention_policy.md) — explicit `main` / `supporting` / `archived` inventory
- [`CLAUDE.md`](CLAUDE.md) — operator-oriented overview: tasks, file layout, gotchas

Historical context lives under `archive/`.

## Current Analysis Program

The live workflow has three major components: the mouse bulk pipeline, the Incytr pair-mode signaling layer, and a parallel human (NBB / Mukesh) cohort that provides cross-species support.

### 1. Stoichiometry-corrected kinase attribution (bulk, mouse)

1. **Data ingestion** (`alz/data_ingest.py`) — TMT channel mapping, phosphosite-to-protein matching (91.7%), PCA QC, outlier detection
2. **Normalize → enrich → attribute** (`kinase_normalize.py` → `kinase_enrich.py` → `kinase_attribute.py`) — IRS cross-plex normalization (all 72 samples), stoichiometry computation, factorial OLS per site with disease × timepoint interactions (9 contrasts), MEA kinase enrichment on median-centered + winsorized β values, unified cell-type attribution combining Song within-cohort + SEA-AD concordance + WMB specificity
3. **Attribution recovery** (`attribution_recovery.py`) — cross-contrast consistency and final hypothesis-table assembly (kinase activity matrix, cell-type evidence table, **`kinase_hypothesis_table.csv`** — primary deliverable)

### 2. Intercellular signaling integration (Incytr pair-mode)

Pair-mode is the active Incytr path. It models each intercellular pathway as a four-gene chain (Ligand → Receptor → EM → Target) between sender and receiver cell types, scored against the upstream `incytr` R package (installed via `pixi run install-incytr`). Driven on the **Levy-t5** proportional-decomposition spine (31 clusters × 31 senders/receivers per contrast).

The legacy factorial Incytr engine was archived 2026-05-18 (`archive/incytr_factorial_2026-05-18/`). It is preserved on disk but no longer wired into any pixi task or runner.

### 3. Human cross-species cohort (NBB / Mukesh)

A parallel human AD cohort mirrors the mouse stoichiometry pipeline to provide cross-species support for the primary mouse hypothesis table. It is **hypothesis-generating cross-species evidence**, not an independent primary deliverable.

1. **Ingest** (`alz/ingest_mukesh.py`) — UniProt canonical-isoform cache, diagnostic pass, and reshape of NBB donor × site tables into Song-shaped artifacts (`--reshape`).
2. **Per-donor MEA** (`alz/ingest_mukesh_perdonor.py --track both`) — for each AD donor, builds a per-site delta vs the CTRL mean on two tracks (stoichiometry primary, raw phospho as abundance-vs-activity sensitivity check). MEA → kinase × donor NES matrix + recurrence summary at FDR < `MEA_FDR_THRESH` in ≥ k donors. A CTRL leave-one-out null distribution calibrates recurrence (CR-01).
3. **Human cell-type attribution** (CR-03):
   - `alz/atlas_reference.py --sea-ad-expression` — SEA-AD MTG per-supertype expression (139 cortical supertypes)
   - `alz/atlas_reference.py --hbca-download` — Allen HBCA whole-brain class-level expression
   - `alz/human_reference_expression.py --ref both` — per-cell-type kinase specificity (`log2(mean_celltype / mean_brain)`, quantile-ranked; same formula as WMB)
   - `alz/human_celltype_attribution.py` — top-N specific cell types per kinase per reference. SEA-AD supertypes and HBCA classes are rolled up directly to the Levy-t5 spine (no chained intermediate vocabularies).

Outputs land in `outputs/reports/kinase_attribution_human/` (per-donor NES, recurrence tables, `celltype_specificity.csv`) and surface in the unified viewer alongside the mouse hypothesis table.

### Per-cluster proportional decomposition (Levy-t5 branch)

Forward projection only — `P_c = f_c × bulk` — **not** statistical deconvolution. See `docs/incytr_deconvolution_pivot.md` for the contract and `docs/plans/change_request_02_spine_rethreshold.md` for the levy19 → levy_t5 rethreshold (per-(cluster, animal) cell gate relaxed from the original `≥50` down to `≥5`, rank-gate dropped — 19 strict-rank clusters → 31 clusters covering 94.5% of nuclei).

End-to-end smoke run:

```bash
bash alz/runners/main/run_pivot_smoke.sh [--skip-normalize]
```

Verification harness (`alz/decomposition/verify_decomposition.py`) checks mass identity, spine coverage, per-cluster vs bulk MEA agreement, and pair coverage.

### Important guardrails

- Direct cell-type deconvolution, transcript-only rescue, factor-model, and two-compartment rescue are **closed paths** — see `docs/foundation/analysis_charter.md`
- Cell-type attribution of kinase activity is correlational; the integration is hypothesis-generating, not mechanistic validation

## Environment Setup

The project is managed with **pixi** (activated automatically via direnv on `cd`). Python 3.11 + kinase-library 1.7.0 + R. Package versions are pinned to kinase-library's strict `~=` requirements.

```bash
pixi install   # first-time setup
```

The pair-mode Incytr path additionally requires the local R package at `~/Projects/work/incytr/`:

```bash
pixi run install-incytr
```

## Running the Analysis

All commands run from the repo root.

### Bulk pipeline

```bash
pixi run live   # ingest → normalize → enrich → attribute → recover
pixi run dual   # males-only (primary) + full-cohort (sensitivity)
```

Individual stages: `pixi run {ingest,normalize,enrich,attribute,recover}`. Sample filtering is controlled by `analysis_mode` in `conf/base/parameters.yml` (default `males_only`); set `KEDRO_ENV=full_cohort` for the sensitivity overlay.

### Supporting prerequisites

Run these before the bulk pipeline if their outputs do not exist:

```bash
pixi run atlas        # SEA-AD + WMB atlas downloads (~95 GB for WMB)
pixi run wmb-export   # WMB per-class kinase/phosphatase expression
pixi run snrna        # Song within-cohort snRNA-seq pseudobulk + concordance
```

### Per-cluster + pair-mode Incytr

```bash
bash alz/runners/main/run_pivot_smoke.sh     # per-cluster decomposition + verification (Levy-t5 spine)
bash alz/runners/main/run_pair_mode_pipeline.sh   # full pair-mode pipeline (inputs → Incytr → viewer reshape)
bash alz/incytr/run_pair_mode.sh             # Incytr invocation only (9 contrasts)
```

### Human cross-species cohort (NBB / Mukesh)

```bash
pixi run python alz/ingest_mukesh.py --reshape                  # CR-01: NBB → Song-shaped tables
pixi run python alz/ingest_mukesh_perdonor.py --track both      # CR-01: per-donor MEA + CTRL LOO
pixi run python alz/atlas_reference.py --sea-ad-expression      # CR-03: SEA-AD MTG expression
pixi run python alz/atlas_reference.py --hbca-download          # CR-03: Allen HBCA download
pixi run python alz/human_reference_expression.py --ref both    # CR-03: SEA-AD + HBCA specificity
pixi run python alz/human_celltype_attribution.py               # CR-03: top-N cell types per kinase
```

All of the above are bundled (resumable, with sentinels) in `alz/runners/main/run_pair_mode_pipeline.sh`.

### Collaborator data ingest

External study data is pulled on demand (no live FUSE mounts):

```bash
pixi run ingest-gdrive-shared       # → data/external/gdrive_shared/
pixi run ingest-lucie-proteomics    # → data/external/lucie_proteomics/
```

### Supplementary diagnostics

Reviewer-response analyses that validate pipeline choices:

```bash
bash alz/runners/supplementary/run_reviewer_diagnostics.sh
```

## Repository Layout

```text
alzheimers/
├── alz/
│   ├── data_ingest.py, kinase_normalize.py, kinase_enrich.py,
│   │   kinase_attribute.py, kinase_mechanism.py, attribution_recovery.py,
│   │   plot_attribution_bubbles.py, build_unified_viewer.py,
│   │   atlas_reference.py, wmb_expression.py, snrna_integration.py,
│   │   snrna_proportions.py, config.py, map_kinases_to_genes.py,
│   │   ingest_mukesh.py, ingest_mukesh_perdonor.py,
│   │   human_reference_expression.py, human_celltype_attribution.py
│   ├── pipelines/                  # Kedro pipelines (live arc + ingest_mapping)
│   ├── decomposition/              # Levy-t5 per-cluster proportional decomposition
│   ├── integration/                # Pair-mode Incytr helpers + cluster spine generators
│   │                                  (factorial plumbing archived 2026-05-18)
│   ├── supplementary/              # Reviewer-response diagnostics
│   └── runners/
│       ├── main/                   # Live + branch pipeline runners
│       └── supporting/             # Atlas / WMB / snRNA prep runners
├── bench/                          # Pair-mode benchmarking (incytr_pair_19/, pair input builders)
├── data/
│   ├── datasets/song/              # Authoritative Song workspace (TMT proteome + phospho + metadata)
│   ├── external/                   # Cached Allen WMB + SEA-AD reference data (zstd-compressed at rest)
│   └── incytr_frozen/              # Frozen cluster taxonomy + ligand-receptor DB snapshots
├── docs/
│   ├── foundation/                 # Charter, rationale, contract, constraints
│   ├── plans/                      # Active change-request and pivot plans
│   ├── integrations/               # Mapping + integration notes
│   └── archive/                    # Frozen historical docs
├── archive/                        # Frozen historical code (factorial Incytr, deconvolution, SAP, etc.)
├── outputs/reports/                # All generated tables, plots, manifests
├── pixi.toml                       # Task definitions + pinned deps
└── CLAUDE.md                       # Operator-oriented overview
```

## External Data Compression

Allen Brain Atlas and SEA-AD reference data under `data/external/` is zstd-compressed at rest. The live pipeline reads only small pre-computed outputs (effect-size h5ad files + CSV expression matrices), not the raw atlas h5ad files.

`wmb_expression.py` transparently decompresses subset files before computation and recompresses them afterward (sentinel-guarded). Manual control:

```bash
bash alz/runners/supporting/compress_atlas_cache.sh   [tier1|tier2|tier3|WMB|sea_ad|subset]
bash alz/runners/supporting/decompress_atlas_cache.sh [WMB|subset|sea_ad]
```

Provenance lives in `data/external/allen_abc/MANIFEST.json`.

## Key Outputs

| Directory | Contents |
|---|---|
| `outputs/reports/data_ingest/` | Sample mapping, phospho-protein matching, PCA, `sample_exclusions.csv` |
| `outputs/reports/kinase_attribution/` | Stoichiometry matrix, `mea_stoichiometry.csv`, `unified_attribution.csv`, site-level OLS, winsorization logs |
| `outputs/reports/attribution_recovery/` | **Primary**: `kinase_hypothesis_table.csv`, `kinase_activity_matrix.csv`, `celltype_evidence_table.csv`, bubble plots |
| `outputs/reports/kinase_attribution_human/` | Human NBB cohort: per-donor NES, recurrence tables, `celltype_specificity.csv` (cross-species support) |
| `outputs/reports/wmb_expression/` | WMB per-class kinase/phosphatase expression |
| `outputs/reports/snrna_integration/` | Song pseudobulk + within-cohort specificity + concordance |
| `outputs/reports/decomposition/levy_t5/` | Per-cluster decomposition + verification |
| `outputs/reports/unified_viewer/` | Interactive HTML viewer (`build_unified_viewer.py`) |

## Conventions

- Use `docs/foundation/` as the live analytical contract
- Use `docs/foundation/repo_retention_policy.md` when deciding whether a file is `main`, `supporting`, or `archived`
- Prefer `data/datasets/song/` over ad hoc files elsewhere in `data/`
- Treat `archive/` as provenance and history, not as the default source of live methods
- The integration pipeline is hypothesis-generating: frame results as convergent functional evidence, not mechanistic pathway validation
