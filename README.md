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

1. **Ingest** (`python -m alz.cohorts.mukesh.ingest --reshape`) — UniProt canonical-isoform cache, diagnostic pass, and reshape of NBB donor x site tables into Song-shaped artifacts.
2. **Per-donor MEA** (`python -m alz.cohorts.mukesh.mea --track both`) — for each AD donor, builds a per-site delta vs the CTRL mean on two tracks (stoichiometry primary, raw phospho as abundance-vs-activity sensitivity check). MEA -> kinase x donor NES matrix + recurrence summary at FDR < `MEA_FDR_THRESH` in >= k donors. A CTRL leave-one-out null distribution calibrates recurrence (CR-01).
3. **Human cell-type attribution** (CR-03):
   - `alz/reference/atlas.py --sea-ad-expression` — SEA-AD MTG per-supertype expression (139 cortical supertypes)
   - `alz/reference/atlas.py --hbca-download` — Allen HBCA whole-brain class-level expression
   - `alz/reference/human_expression.py --ref both` — per-cell-type kinase specificity (`log2(mean_celltype / mean_brain)`, quantile-ranked; same formula as WMB)
   - `alz/cross_reference/human_celltype_attribution.py` — top-N specific cell types per kinase per reference. SEA-AD supertypes and HBCA classes are rolled up directly to the Levy-t5 spine (no chained intermediate vocabularies).

Outputs land in `outputs/reports/kinase_attribution_human/` (per-donor NES, recurrence tables, `celltype_specificity.csv`) and surface in the unified viewer alongside the mouse hypothesis table.

### 4. T-cell exhaustion cohort (Donor 1 + Donor 2)

Net-new human T-cell exhaustion cohort (ingested 2026-05-27 from Google Drive folder `1YE_h1jIyBajtm6ArxJqevJ0rt0xLKQgX`). Two donors, each with matched TMT phosphoproteomics and CITE-seq scRNA along a time course (donor1 days 0/2/9/13/17/20, donor2 days 2/5/7/9/11). Donor1 carries Total + pY + IMAC; donor2 carries Total + pY only (no IMAC → no kinase MEA on donor2; Incytr pair-mode runs pr+py on donor2, pr+ps+py on donor1).

For a stable, citation-ready summary of this analysis and the dedicated T-cell
viewer, use
[`docs/reference/tcell_exhaustion_analysis_summary.md`](docs/reference/tcell_exhaustion_analysis_summary.md).

Pipeline (run in this order):

```bash
pixi run ingest-tcells-scrna      # download raw Seurat RDS for both donors (~10 GB)
pixi run tcells-label             # CITE-seq lineage + definitive biological states
pixi run tcells-scrna-extract     # evidence-state-keyed aggexp/counts/markers
pixi run tcells-export-bulk       # linear per-day bulk matrices (pr/py/ps)
pixi run tcells-decompose         # per-(state, day) substrate via P_s = (N_total/N_s) × bulk × share
```

**Substrate is keyed on evidence-backed per-cell biological states, not ProjecTILs
states.** CD4/CD8 lineage comes from the cells' CITE-seq CD4/CD8 antibody counts
with native-cluster fallback. CD8 states are named `CD8PrecursorExhausted`,
`CD8Exhausted`, `CD8Memory`, `CD8Cytotoxic`, and `CD8Effector`. ProjecTILs state,
raw confidence, and categorical reference corroboration are retained beside the
direct call rather than overwriting it. `TerminalExhausted` is not used because
many checkpoint-positive cells remain proliferative. Only explicit
myeloid/mast/NK/gamma-delta contaminant clusters are dropped. See
[`docs/reference/tcell_exhaustion_analysis_summary.md`](docs/reference/tcell_exhaustion_analysis_summary.md)
for the rationale.

Outputs (under `data/derived/tcells_incytr_inputs/<donor>/`):
- `scrna/aggexp_data.csv`, `cell_counts.csv`, `allmarkers.csv` — substrate keyed
  on sanitized biological state (`CD8PrecursorExhausted`, `CD8Exhausted`,
  `CD4Proliferating`, …; alphanumeric only).
- `outputs/reports/tcell_labeling/cells/{donor}_state_labels.csv` — per-cell
  lineage/state calls plus raw RNA and antibody evidence.
- `scrna/projectils_predictions.csv` — independent reference projection used for
  categorical corroboration, never as an authoritative override.
- `scrna/state_audit.json`, `extract_manifest.json`, `decompose_manifest.json` — drop accounting and per-state/day cell counts.
- `{pr,py,ps}_deconvoluted.csv` — per-(state, day) values, columns `d{day}_{state}`. Mass identity `Σ_s P_s × N_s/N_total ≈ bulk` verified to ≤ 2e-15 per channel × day.

ProjecTILs reference atlases (carmonalab figshare doi 10.6084/m9.figshare.23608308) cache at `data/external/projectils/`. The CD4 atlas is tumor-derived — no Tcm/Th1/Tprolif vocabulary. Figshare blocks programmatic download on this network's WAF; references must be hand-downloaded the first time.

### Per-cluster proportional decomposition (Levy-t5 branch)

Forward projection only — `P_c = f_c × bulk` — **not** statistical deconvolution. See `docs/foundation/analysis_rationale.md` and `docs/foundation/statistical_constraints.md` for the closed-path rationale, and `alz/decomposition_mea/README.md` for the current Levy-t5 decomposition contract (per-(cluster, animal) cell gate relaxed from the original `≥50` down to `≥5`, rank-gate dropped — 19 strict-rank clusters → 31 clusters covering 94.5% of nuclei).

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

Viewer payloads use schema v2. Dataset-specific builders stay separate, but the emitted JSON routes
shared blocks through `meta.contexts` and `*.by_context`. `selection.context` / `ctx=` are the
canonical frontend routing primitives; donor-specific payload aliases such as `by_donor` are not
emitted by the current T-cell payload. Old `#d=...` T-cell links are accepted only as inbound URL
compatibility and are translated into context state.

Viewer frontend modules are shared only where behavior is identical or config-driven. The current
shared template modules live under `alz/viewer_shared/template/js/`; local viewer templates override
shared files at the same path. See `docs/foundation/viewer_frontend_contract.md` before adding or
forking viewer JavaScript.

Check generated payloads with:

```bash
python alz/viewer/verify_payload_contract.py \
  outputs/reports/unified_viewer/unified_viewer.payload.json \
  outputs/reports/tcell_viewer/tcell_viewer.payload.json
```

The current validated contexts are `song_ad` for the AD unified viewer and `donor1`/`donor2` for the
T-cell viewer.

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
pixi run python -m alz.cohorts.mukesh.ingest --reshape                  # CR-01: NBB → Song-shaped tables
pixi run python -m alz.cohorts.mukesh.mea --track both                  # CR-01: per-donor MEA + CTRL LOO
pixi run python alz/reference/atlas.py --sea-ad-expression              # CR-03: SEA-AD MTG expression
pixi run python alz/reference/atlas.py --hbca-download                  # CR-03: Allen HBCA download
pixi run python alz/reference/human_expression.py --ref both            # CR-03: SEA-AD + HBCA specificity
pixi run python alz/cross_reference/human_celltype_attribution.py       # CR-03: top-N cell types per kinase
```

All of the above are bundled (resumable, with sentinels) in `alz/runners/main/run_pair_mode_pipeline.sh`.

### Collaborator data ingest

External study data is pulled on demand (no live FUSE mounts). Every Drive
source is declared in `conf/data_sources.yaml` and pulled by the shared
[`rclone-ingest`](vendor/rclone-ingest) engine (vendored as a git submodule);
the `pixi run ingest-*` tasks are thin wrappers over it.

```bash
git submodule update --init           # first checkout only (provides vendor/rclone-ingest)
pixi run rclone-ingest check          # preflight: rclone present + remotes configured
pixi run ingest-gdrive-shared         # → data/external/gdrive_shared/
pixi run ingest-lucie-proteomics      # → data/external/lucie_proteomics/
pixi run ingest-5xfad-reports         # → data/raw/external/lucie_proteomics/reports/ (parsed PTM tables)
pixi run ingest-5xfad-raw-sne         # + ~28 GB raw .sne for the 2 unparsed cells
pixi run ingest-5xfad-snrna           # 5xFAD snRNA metadata/provenance
pixi run ingest-5xfad-snrna-rds       # + 2025 reclustering RDS objects (group: scrna)
pixi run ingest-deconvolution-bulk    # → data/datasets/song/proteomics/source/ (filtered)
pixi run ingest-tcells                # T-cell cohort proteomics
pixi run ingest-tcells-scrna          # + ~10 GB scRNA .rds (group: scrna)
pixi run ingest-sce4-canonical        # sce4 parity provenance (Top300 + limma)
pixi run ingest-sce4-source           # full 7.5 GiB sce4 working dir
```

Add or edit sources in `conf/data_sources.yaml`; `pixi run rclone-ingest list`
enumerates them. The engine is pull-only — auth lives in `rclone.conf`.

To examine a folder before declaring it (read/range-only — no mount, no bulk
download), use `ls` / `peek` / `fetch`:

```bash
pixi run rclone-ingest ls tcells donor1 -R --max-depth 2     # names, sizes, mimetypes (no transfer)
pixi run rclone-ingest peek tcells donor1/proteomics/x.csv   # first 4 KB via a ranged GET
pixi run rclone-ingest fetch tcells donor1/proteomics/x.csv --dest /tmp   # grab one file
pixi run rclone-ingest ls --folder-id <ID> --remote gdrive_shared        # ad-hoc, undeclared folder
```

### Viewer deployment (S3)

After building a viewer, sync only changed files to S3:

```bash
pixi run deploy-viewer          # unified viewer → s3://voila-buc-00-prod/pocs/incytr/unified_viewer_human/
pixi run deploy-tcell-viewer    # T-cell viewer  → s3://voila-buc-00-prod/pocs/incytr/tcell_viewer/
pixi run deploy-all-viewers     # both
```

Uses `aws s3 sync --profile bioplat` — skips files whose ETag matches S3. The uncompressed
`*.payload.json` is excluded from upload; the browser loads the `.gz` sidecar.

### 5xFAD cohort

```bash
pixi run 5xfad-ingest                  # ingest raw proteomics
pixi run 5xfad-export-bulk             # build linear bulk matrices
pixi run 5xfad-mea                     # 5xFAD kinase MEA
pixi run 5xfad-snrna-attribution       # snRNA cell-type attribution (Rscript)
pixi run 5xfad-snrna-decomp-pseudobulk # per-cluster pseudobulk for decomposition
pixi run 5xfad-celltype-mea            # per-cluster MEA
pixi run 5xfad-build-incytr-seurat     # build Seurat object for Incytr inputs
pixi run 5xfad-build-incytr-gene-list  # allmarkers for both tissues (cortex + hippocampus)
pixi run 5xfad-incytr                  # run pair-mode Incytr on 5xFAD
pixi run 5xfad-incytr-decompose        # decomposition substrate for Incytr inputs
pixi run 5xfad-viewer                  # build unified viewer including 5xFAD block
pixi run 5xfad-viewer-package          # refresh only the 5xFAD block in an existing payload
```

### Verification

```bash
pixi run verify-incytr-sce4       # regression check: sce4 parity (≥599/600 recall)
pixi run verify-incytr-sce4-full  # full sce4 verification in R
```

### Supplementary diagnostics

Reviewer-response analyses that validate pipeline choices:

```bash
bash alz/runners/supplementary/run_reviewer_diagnostics.sh
```

### All pixi tasks — quick reference

| Task | What it does |
|---|---|
| **Bulk pipeline** | |
| `ingest` | Song proteomics ingestion, channel mapping, QC |
| `normalize` | IRS normalization, stoichiometry matrix |
| `enrich` | MEA kinase enrichment on stoichiometry β values |
| `attribute` | Cell-type attribution (Song + SEA-AD + WMB) |
| `mechanism` | Merge mechanism annotations into `unified_attribution.csv` |
| `recover` | Attribution recovery, `kinase_hypothesis_table.csv` |
| `live` | `ingest → normalize → enrich → attribute → mechanism → recover` |
| `dual` | Males-only (primary) + full-cohort (sensitivity) |
| `all` | Full end-to-end build via `run_all.sh` (resumable) |
| `bubbles` | Plot attribution bubble charts |
| **Reference prerequisites** | |
| `atlas` | SEA-AD + WMB atlas downloads (~95 GB for WMB) |
| `wmb-export` | WMB per-class kinase/phosphatase expression |
| `snrna` | Song within-cohort snRNA-seq pseudobulk + concordance |
| **Human cohort (NBB / Mukesh)** | |
| `human` | `human-ingest → human-perdonor → human-seaad` |
| `human-ingest` | NBB → Song-shaped tables |
| `human-perdonor` | Per-donor MEA (ST + pY tracks) |
| `human-seaad` | SEA-AD human agreement chain |
| **T-cell cohort** | |
| `ingest-tcells` | Pull T-cell proteomics from Drive |
| `ingest-tcells-scrna` | Pull T-cell scRNA RDS (~10 GB) |
| `tcells-reshape` | Reshape raw T-cell tables |
| `tcells-projectils-map` | ProjecTILs cell-state projection |
| `tcells-scrna-extract` | State-keyed aggexp + markers (after projectils-map) |
| `tcells-export-bulk` | Per-day bulk matrices (pr/py/ps) |
| `tcells-decompose` | Per-(state, day) substrate |
| `tcells-perdonor` | Per-donor MEA |
| `tcells-build-kldata` | Build kinase-library scoring input |
| `tcells-build-incytr-seurat` | Seurat objects for both donors |
| `tcells-build-input-gene-list` | allmarkers for both donors |
| `tcells-incytr` | Run pair-mode Incytr on T-cells |
| `tcell-within-cohort` | Within-cohort concordance (donor1) |
| `tcell-viewer` | Build T-cell viewer |
| **5xFAD cohort** | |
| `5xfad-ingest` | Ingest 5xFAD proteomics |
| `5xfad-export-bulk` | Build 5xFAD bulk matrices |
| `5xfad-mea` | 5xFAD kinase MEA |
| `5xfad-snrna-attribution` | snRNA cell-type attribution |
| `5xfad-snrna-decomp-pseudobulk` | Per-cluster pseudobulk |
| `5xfad-celltype-mea` | Per-cluster MEA |
| `5xfad-build-incytr-seurat` | Seurat object for Incytr |
| `5xfad-build-incytr-gene-list` | allmarkers (cortex + hippocampus) |
| `5xfad-incytr` | Pair-mode Incytr on 5xFAD |
| `5xfad-incytr-decompose` | Decomposition substrate for Incytr |
| `5xfad-viewer` | Build unified viewer with 5xFAD |
| `5xfad-viewer-package` | Refresh only 5xFAD block in existing payload |
| **Unified viewer** | |
| `viewer` | Build unified viewer (Song + Mukesh + 5xFAD) |
| `deploy-viewer` | Sync unified viewer to S3 (incremental) |
| `deploy-tcell-viewer` | Sync T-cell viewer to S3 (incremental) |
| `deploy-all-viewers` | Sync both viewers to S3 |
| **Verification** | |
| `verify-incytr-sce4` | sce4 parity regression check (≥599/600 recall) |
| `verify-incytr-sce4-full` | Full sce4 verification in R |
| **Setup** | |
| `install-incytr` | Install local Incytr R package |
| `install-projectils` | Install ProjecTILs + cache atlases (one-time) |
| **Data ingest (Drive → local)** | |
| `ingest-gdrive-shared` | `data/external/gdrive_shared/` |
| `ingest-lucie-proteomics` | `data/external/lucie_proteomics/` |
| `ingest-5xfad-reports` | 5xFAD parsed PTM tables |
| `ingest-5xfad-raw-sne` | ~28 GB raw 5xFAD .sne files |
| `ingest-5xfad-snrna` | 5xFAD snRNA metadata |
| `ingest-5xfad-snrna-rds` | 5xFAD 2025 reclustering RDS |
| `ingest-deconvolution-bulk` | Song proteomics source tables |
| `ingest-sce4-canonical` | sce4 parity provenance |
| `ingest-sce4-source` | Full 7.5 GiB sce4 working dir |

## Repository Layout

```text
alzheimers/
├── alz/
│   ├── shared/                     # config.py + cross-mode utilities (map_kinases_to_genes.py)
│   ├── ingest/                     # Layer 1 — bespoke per-dataset modules (song, lucie); mukesh/tcells/fivexfad moved to alz/cohorts/
│   ├── cohorts/                    # Cohort namespaces: mukesh/, tcells/, fivexfad/, song/
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
