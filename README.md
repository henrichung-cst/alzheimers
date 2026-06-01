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
- [`CLAUDE.md`](CLAUDE.md) — agent-specific operating overrides, correctness invariants, and pipeline gotchas (not a duplicate of this README)

Historical context lives under `archive/`.

## Current Analysis Program

The live workflow has three major components: the mouse bulk pipeline, the Incytr pair-mode signaling layer, and a parallel human (NBB / Mukesh) cohort that provides cross-species support.

### 1. Stoichiometry-corrected kinase attribution (bulk, mouse)

1. **Data ingestion** (`alz/ingest/song.py`) — TMT channel mapping, phosphosite-to-protein matching (91.7%), PCA QC, outlier detection
2. **Normalize → enrich → attribute** (`alz/bulk_mea/normalize.py` → `enrich.py` → `attribute.py`) — IRS cross-plex normalization (all 72 samples), stoichiometry computation, factorial OLS per site with disease × timepoint interactions (9 contrasts), MEA kinase enrichment on median-centered + winsorized β values, unified cell-type attribution combining Song within-cohort + SEA-AD concordance + WMB specificity
3. **Attribution recovery** (`alz/bulk_mea/recover.py`) — cross-contrast consistency and final hypothesis-table assembly (kinase activity matrix, cell-type evidence table, **`kinase_hypothesis_table.csv`** — primary deliverable)

### 2. Intercellular signaling integration (Incytr pair-mode)

Pair-mode is the active Incytr path. It models each intercellular pathway as a four-gene chain (Ligand → Receptor → EM → Target) between sender and receiver cell types, scored against the upstream `incytr` R package (installed via `pixi run install-incytr`). Driven on the **Levy-t5** proportional-decomposition spine (31 clusters × 31 senders/receivers per contrast).

Canonical Incytr outputs are floor-gated and uncapped: the scorer emits all paths, then
`alz/incytr_pair/filter_significant_paths.py` keeps rows with
`(SigProb_condition1 > 0.1 OR SigProb_condition2 > 0.1) AND abs(PDS) >= 0.2`. Do not add a
p-value/FDR gate for canonical outputs. Do not apply the per sender/receiver Top300 up/down cap to
Song/AD or T-cell viewer outputs; `--top300` is reserved for explicit sce4 table-compatibility
diagnostics because the cap is rank-sensitive to unresolved `PDS` drift. AD production inputs are
the matched bundle under `data/derived/incytr_inputs/`; T-cell production inputs are the matched
per-donor bundles under `data/derived/tcells_incytr_inputs/`.

The legacy factorial Incytr engine was archived 2026-05-18 (`archive/incytr_factorial_2026-05-18/`). It is preserved on disk but no longer wired into any pixi task or runner.

### 3. Human cross-species cohort (NBB / Mukesh)

A parallel human AD cohort mirrors the mouse stoichiometry pipeline to provide cross-species support for the primary mouse hypothesis table. It is **hypothesis-generating cross-species evidence**, not an independent primary deliverable.

1. **Ingest** (`alz/ingest/mukesh.py`) — UniProt canonical-isoform cache, diagnostic pass, and reshape of NBB donor × site tables into Song-shaped artifacts (`--reshape`).
2. **Per-donor MEA** (`alz/ingest/mukesh_perdonor.py --track both`) — for each AD donor, builds a per-site delta vs the CTRL mean on two tracks (stoichiometry primary, raw phospho as abundance-vs-activity sensitivity check). MEA → kinase × donor NES matrix + recurrence summary at FDR < `MEA_FDR_THRESH` in ≥ k donors. A CTRL leave-one-out null distribution calibrates recurrence (CR-01).
3. **Human cell-type attribution** (CR-03):
   - `alz/reference/atlas.py --sea-ad-expression` — SEA-AD MTG per-supertype expression (139 cortical supertypes)
   - `alz/reference/atlas.py --hbca-download` — Allen HBCA whole-brain class-level expression
   - `alz/reference/human_expression.py --ref both` — per-cell-type kinase specificity (`log2(mean_celltype / mean_brain)`, quantile-ranked; same formula as WMB)
   - `alz/cross_reference/human_celltype_attribution.py` — top-N specific cell types per kinase per reference. SEA-AD supertypes and HBCA classes are rolled up directly to the Levy-t5 spine (no chained intermediate vocabularies).

Outputs land in `outputs/reports/kinase_attribution_human/` (per-donor NES, recurrence tables, `celltype_specificity.csv`) and surface in the unified viewer alongside the mouse hypothesis table.

### 4. T-cell exhaustion cohort (Donor 1 + Donor 2)

Net-new human T-cell exhaustion cohort (ingested 2026-05-27 from Google Drive folder `1YE_h1jIyBajtm6ArxJqevJ0rt0xLKQgX`). Two donors, each with matched TMT phosphoproteomics and CITE-seq scRNA along a time course (donor1 days 0/2/9/13/17/20, donor2 days 2/5/7/9/11). Donor1 carries Total + pY + IMAC; donor2 carries Total + pY only (no IMAC → no kinase MEA on donor2; Incytr pair-mode runs pr+py on donor2, pr+ps+py on donor1).

Pipeline (run in this order):

```bash
pixi run ingest-tcells-scrna      # download raw Seurat RDS for both donors (~10 GB)
pixi run install-projectils       # one-time: install ProjecTILs + cache CD4/CD8 atlases
pixi run tcells-projectils-map    # per-cell projection onto human CD4/CD8 reference atlases
pixi run tcells-scrna-extract     # state-keyed aggexp/counts/markers (MUST run after projectils-map)
pixi run tcells-export-bulk       # linear per-day bulk matrices (pr/py/ps)
pixi run tcells-decompose         # per-(state, day) substrate via P_s = (N_total/N_s) × bulk × share
```

**Substrate is keyed on per-cell ProjecTILs `functional.cluster`, not Seurat clusters.** Cluster-level annotation was deleted (anti-shim) after losing 44.5% of donor1 to the Seurat–ProjecTILs partition mismatch. State-keyed aggregation drops only cells with no ProjecTILs call (~13% donor1, ~7% donor2 — scGate `none`-gate + doublets). See [`docs/plans/tcells_percell_aggregation_2026-05-28.md`](docs/plans/tcells_percell_aggregation_2026-05-28.md) for the rationale.

Outputs (under `data/derived/tcells_incytr_inputs/<donor>/`):
- `scrna/aggexp_data.csv`, `cell_counts.csv`, `allmarkers.csv` — substrate keyed on sanitized ProjecTILs state (`CD8Tex`, `CD4Th17`, `Treg`, …; alphanumeric only — Incytr `<condition>_<cluster>` split constraint).
- `scrna/projectils_predictions.csv` — per-cell `lineage_gate`, `functional.cluster`, `functional.cluster.conf`.
- `scrna/state_audit.json`, `extract_manifest.json`, `decompose_manifest.json` — drop accounting and per-state/day cell counts.
- `{pr,py,ps}_deconvoluted.csv` — per-(state, day) values, columns `d{day}_{state}`. Mass identity `Σ_s P_s × N_s/N_total ≈ bulk` verified to ≤ 2e-15 per channel × day.

ProjecTILs reference atlases (carmonalab figshare doi 10.6084/m9.figshare.23608308) cache at `data/external/projectils/`. The CD4 atlas is tumor-derived — no Tcm/Th1/Tprolif vocabulary. Figshare blocks programmatic download on this network's WAF; references must be hand-downloaded the first time.

### Per-cluster proportional decomposition (Levy-t5 branch)

Forward projection only — `P_c = f_c × bulk` — **not** statistical deconvolution. See `docs/incytr_deconvolution_pivot.md` for the contract and `docs/plans/change_request_02_spine_rethreshold.md` for the levy19 → levy_t5 rethreshold (per-(cluster, animal) cell gate relaxed from the original `≥50` down to `≥5`, rank-gate dropped — 19 strict-rank clusters → 31 clusters covering 94.5% of nuclei).

End-to-end decomposition rebuild (pseudobulk → proportions → decompose → per-cluster MEA → verify):

```bash
bash alz/runners/main/rerun_decomposition_chain.sh
```

Verification is split conceptually into hard invariants and diagnostics. The hard decomposition
invariants are mass identity and spine coverage. Per-cluster-vs-bulk MEA agreement is a diagnostic
concordance check, not a mathematical reconstruction identity, because MEA/GSEA NES values are
computed after ranking, centering, winsorization, and enrichment normalization. Incytr pair coverage
must be interpreted by artifact: raw scorer outputs can be checked for scorer coverage, while
filtered viewer-ready outputs are expected to contain only active sender/receiver pairs.

`python alz/decomposition_mea/verify_decomposition.py --spine levy_t5` writes the hard-gate
`verification.json` consumed by the unified viewer. Add `--include-diagnostics` for the MEA and
Incytr artifact checks; those write a separate diagnostic report by default and do not block the
viewer unless `--strict-diagnostics` is requested.

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
pixi run live   # ingest → normalize → enrich → attribute → mechanism → recover
pixi run dual   # males-only (primary) + full-cohort (sensitivity)
pixi run all    # full end-to-end build (kinase + decomposition + Incytr + human + viewer), resumable
```

Individual stages: `pixi run {ingest,normalize,enrich,attribute,mechanism,recover}`. `mechanism` runs after `attribute` — it merges mechanism annotations into `unified_attribution.csv`. Sample filtering is controlled by `analysis_mode` in `conf/base/parameters.yml` (default `males_only`); set `KEDRO_ENV=full_cohort` for the sensitivity overlay. `pixi run all` (the resumable `run_all.sh` superset) auto-downloads the WMB/SEA-AD references if missing; pass `--skip-atlas` to assume they are on disk.

### Supporting prerequisites

Run these before the bulk pipeline if their outputs do not exist:

```bash
pixi run atlas        # SEA-AD + WMB atlas downloads (~95 GB for WMB)
pixi run wmb-export   # WMB per-class kinase/phosphatase expression
pixi run snrna        # Song within-cohort snRNA-seq pseudobulk + concordance
```

### Per-cluster + pair-mode Incytr

```bash
bash alz/runners/main/rerun_decomposition_chain.sh   # per-cluster decomposition + verification (Levy-t5 spine)
bash alz/runners/main/run_pair_mode_pipeline.sh   # full pair-mode pipeline (inputs → Incytr → viewer reshape)
bash alz/incytr_pair/run_pair_mode.sh        # Incytr invocation only (9 contrasts)
```

### Human cross-species cohort (NBB / Mukesh)

```bash
pixi run python alz/ingest/mukesh.py --reshape                          # CR-01: NBB → Song-shaped tables
pixi run python alz/ingest/mukesh_perdonor.py --track both              # CR-01: per-donor MEA + CTRL LOO
pixi run python alz/reference/atlas.py --sea-ad-expression              # CR-03: SEA-AD MTG expression
pixi run python alz/reference/atlas.py --hbca-download                  # CR-03: Allen HBCA download
pixi run python alz/reference/human_expression.py --ref both            # CR-03: SEA-AD + HBCA specificity
pixi run python alz/cross_reference/human_celltype_attribution.py       # CR-03: top-N cell types per kinase
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
│   ├── shared/                     # config.py + cross-mode utilities (map_kinases_to_genes.py)
│   ├── ingest/                     # Layer 1 — bespoke per-dataset modules (song, mukesh, lucie, tcells)
│   ├── reference/                  # Atlas downloads + cross-cohort expression (atlas, wmb_expression, human_expression, snrna_integration, snrna_proportions)
│   ├── bulk_mea/                   # Mode 1 — normalize → enrich → attribute → recover (+ mechanism, summary)
│   ├── decomposition_mea/          # Mode 2 — Levy-t5 per-cluster proportional decomposition + per-cluster MEA
│   ├── incytr_pair/                # Mode 3 — pair-mode Incytr driver + receiver-cache reshaper
│   ├── cross_reference/            # Mode 4 — SEA-AD/WMB/Song evidence loaders, human cell-type attribution, human↔SEA-AD agreement
│   ├── ctrl_outlier_audit/         # Human CTRL-07/08/10 AD-like contamination audit + clean-baseline group MEA reanalysis
│   ├── integration/                # Cross-mode glue: cluster spine, omics/transcript trace, bridge builders
│   ├── pipelines/                  # Kedro pipelines for Argo orchestration (P1 ingest live; registry in pipeline_registry.py + settings.py)
│   ├── viewer/, build_unified_viewer.py     # Mouse/human unified HTML viewer
│   ├── tcell_viewer/, build_tcell_viewer.py # T-cell cohort HTML viewer (lifted from viewer/)
│   ├── supplementary/              # Reviewer-response diagnostics
│   └── runners/
│       ├── main/                   # Live + branch pipeline runners
│       └── supporting/             # Atlas / WMB / snRNA / ProjecTILs / T-cell prep runners
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
| `outputs/reports/decomposition/levy_t5/` | Per-cluster decomposition + verification; `per_animal/site_level_ols.parquet` is the published st+py per-cell OLS table consumed by the viewer |
| `outputs/reports/unified_viewer/` | Interactive HTML viewer (`build_unified_viewer.py`) plus lazy edge slices, including `edge_slices/decomp_ols/` for per-kinase substrate-site OLS evidence |

## Conventions

- Use `docs/foundation/` as the live analytical contract
- Use `docs/foundation/repo_retention_policy.md` when deciding whether a file is `main`, `supporting`, or `archived`
- Prefer `data/datasets/song/` over ad hoc files elsewhere in `data/`
- Treat `archive/` as provenance and history, not as the default source of live methods
- The integration pipeline is hypothesis-generating: frame results as convergent functional evidence, not mechanistic pathway validation
