# Plan: T-cell Specificity Reference (10x NSCLC)

**Goal:** Add an external cell-type specificity reference for the T-cell cohort analogous to WMB (mouse Song) and SEA-AD/HBCA (human Mukesh). The reference is the 10x "Aggregate of 900k human NSCLC and normal-adjacent cells, multiplexed 16-probe barcodes" dataset. It answers: where is a kinase expressed across human T-cell and immune subpopulations? The motivating audit: which kinases predicted by T-cell MEA are absent from this reference (never expressed in any immune cell type)?

---

## 1. How WMB and SEA-AD/HBCA are ingested and surfaced

### WMB (mouse Song cohort)

**Source:** Allen Whole Mouse Brain 10Xv3, 13 anatomical regions (~4M cells, ~32K genes). Downloaded by `alz/reference/atlas.py --wmb-download` via `abc_atlas_access` into `data/external/allen_abc/`. Gene-subset h5ads pre-extracted to `data/external/allen_abc/subsets/` via `runners/supporting/extract_wmb_gene_subset.py`.

**Processing:** `alz/reference/wmb_expression.py --run`. Streams h5ad files by chunk (5000 cells/slice). For each gene × WMB-class:
- `mean_log2_expression` = cell-weighted mean log2 expr across all regions
- `fraction_cells_expressing` = (cells with expr > 0) / n_cells
- `binary_expressed` = `mean_log2_expression > 1` **AND** `fraction_cells_expressing > 0.10`
- `specificity_score` = share of total mean log2 expr (per gene, normalized over retained WMB classes)

**Output:** `outputs/reports/wmb_expression/wmb_kinase_expression.csv` — schema: `kinase_id, gene_symbol, cell_type, mean_log2_expression, fraction_cells_expressing, specificity_score, binary_expressed, n_cells`

**Crosswalk:** `data/derived/bridges/cluster_to_wmb_class.csv` maps 31 Levy-t5 spine clusters → WMB class (1:1, hand-curated). Loaded via `config.load_cluster_to_wmb_class_map()`.

**Viewer:** `alz/bulk_mea/attribute.py` joins WMB expression onto kinase attribution rows. The viewer payload carries `wmb_specificity` and `wmb_mean_log2_expression` columns per kinase. The `wmb_specificity_uniform()` function derives the 1/N threshold at runtime from the artifact.

### SEA-AD (human Mukesh cohort)

**Source:** SEA-AD MTG h5ad (~34 GB, 1.38M nuclei × 36K genes, 139 supertypes). Downloaded by `alz/reference/atlas.py --sea-ad-expression`. Streamed via h5py with `RLIMIT_AS` cap to avoid OOM.

**Intermediate:** `data/derived/aggregates/seaad/expression_by_supertype.csv` — genes × 139 supertypes mean log2 expression (produced once, then cached). Consumed by `alz/reference/human_expression.py --ref seaad`.

**Processing:** `human_expression.py::compute_specificity()` — log2(celltype_mean / brain_mean) specificity formula. Outputs:
- `outputs/reports/human_reference_expression/seaad_kinase_specificity.csv` — kinase_id × 139 supertypes
- `outputs/reports/human_reference_expression/seaad_kinase_expression.csv` — raw mean log2 expr

**Crosswalk:** `data/derived/bridges/cluster_to_seaad_supertype.csv` maps 31 Levy-t5 clusters → SEA-AD supertypes (many-to-one with weights). `alz/cross_reference/human_celltype_attribution.py` rolls up to Levy-t5 vocabulary.

**Viewer:** `alz/cross_reference/human_celltype_attribution.py` reads both specificity matrices, rolls them up to Levy-t5 clusters, emits `outputs/reports/kinase_attribution_human/celltype_specificity.csv`. `alz/build_unified_viewer.py` sets `capabilities.human_reference = True` when the Mukesh slice is present.

---

## 2. T-cell cohort structure

- 2 donors; donor2 has no IMAC → no Ser/Thr kinase MEA; all tracks skip by design.
- Kinase MEA lives at `outputs/reports/kinase_attribution_tcells/donor1/mea/`:
  - `mea_timecourse.csv` — 1555 rows (kinase × contrast; ST track)
  - `mea_timecourse_pY.csv` — pY track
  - `kinase_timepoint_nes{,_pY}.csv` / `kinase_timepoint_fdr{,_pY}.csv` — wide matrices
- 303 kinases are significant at FDR < 0.25 in any contrast (donor1 ST).
- Within-cohort attribution source: `alz/cross_reference/tcell_within_cohort.py` derives specificity from the cohort's own scRNA (ProjecTILs states), written to `outputs/reports/kinase_attribution_tcells/donor1/unified_attribution_tcells.csv`.
- The t-cell viewer (`alz/build_tcell_viewer.py`) hardcodes `"human_reference": False` at lines 2364 and 2419 — this is the capability flag to flip when the artifact is ready.
- The `_KINASE_AUDIT_SHIMS` at line 1685–1692 already notes `"wmb_kinase_expression"` and `"sea_ad_supertype_lfc"` as not applicable. The new NSCLC reference needs its own payload key.

---

## 3. Ingest path for the 10x NSCLC dataset

### Dataset

URL: https://www.10xgenomics.com/datasets/aggregate-of-900k-human-non-small-cell-lung-cancer-and-normal-adjacent-cells-multiplexed-samples-16-probe-barcodes-1-standard

This is a **Chromium Gene Expression Flex** dataset (probe-based, not full-transcript). Key characteristics:
- ~900K cells from NSCLC and normal-adjacent tissue, 16-probe multiplex barcodes
- Cell-type annotations: the 10x Flex output includes cell-type labels from their standard annotation pipeline — expected to include T-cell subtypes (CD4+, CD8+, Treg, NK cells), myeloid (macrophages, monocytes, DCs), B cells, endothelial, epithelial, fibroblasts, etc.
- Gene universe: Flex probe panel covers ~18K genes (the "Human Transcriptome Panel" for Flex), NOT the full 20K+. Many low-expression kinases may be off-panel entirely — this must be surfaced explicitly in the audit.
- File formats available: filtered feature-barcode matrix (MEX or HDF5), per-cell type annotations in the web summary or supplementary metadata CSV.

### What to download

The 10x Genomics dataset page provides:
1. **`sample_feature_bc_matrix.h5`** — filtered feature-barcode matrix (HDF5, sparse, ~cells × genes). This is the primary input.
2. **`clusters.csv`** or web summary clustering output — per-cell cluster/type label.
3. **`analysis/` folder** — contains `clustering/` with per-cell type assignments.

**Download strategy:** Direct HTTP fetch from 10x Genomics, not via rclone-ingest (no Drive source). Use `wget` or `curl` to the 10x public URL. Files are public (no auth). Destination: `data/external/nsclc_10x/`. Add a new entry to `conf/data_sources.yaml` comment block for documentation but use a shell script for the actual fetch since rclone-ingest is Google Drive-specific.

**New pixi task:** `nsclc-ingest = "bash alz/runners/supporting/ingest_nsclc_10x.sh"` — a wrapper script that wget/curls the HDF5 + annotation CSV.

**Size concern:** The filtered feature-barcode matrix for 900K cells × 18K genes is typically 1–3 GB in HDF5 sparse format. This is manageable without full-materialization using h5py + chunked streaming, exactly as the SEA-AD MTG pipeline does. Do NOT call `read_h5` directly on the full matrix. Inspect schema first: `python -c "import h5py; f=h5py.File('...h5','r'); print(list(f.keys()))"`.

**OOM constraint:** Cap the processing script with `RLIMIT_AS` at 16 GB (same pattern as `atlas.py::download_sea_ad_expression`). Use chunk_size=5000 cells/slice for the streaming accumulator.

### Cell-type annotation source

The 10x NSCLC Flex dataset ships per-cell cluster assignments. The expected annotation vocabulary from 10x's own reference annotation includes:
- T cells: CD4+ T cells, CD8+ T cells, Regulatory T cells (Treg), NK cells
- Myeloid: Macrophages, Monocytes, Dendritic cells
- B cells / Plasma cells
- Tumor cells (NSCLC-specific)
- Stromal: Fibroblasts, Endothelial cells
- Epithelial (normal and malignant)

**Crosswalk requirement (per memory: direct levy_t5 crosswalks):** The T-cell cohort does NOT use the Levy-t5 spine — it uses ProjecTILs states (14 states: CD8CM, CD8EM, CD8MAIT, CD8Naive, CD8TEMRA, CD8Tex, CD8Tpex, CD4CTLeomes, CD4CTLexh, CD4CTLgnly, CD4Naive, CD4Tfh, CD4Th17, Treg). The NSCLC annotation vocabulary must be crosswalked directly to these ProjecTILs states, NOT via Levy-t5 (which is a brain neuron spine and irrelevant here).

**Crosswalk file:** `data/derived/bridges/nsclc_celltype_to_projectils.csv` — hand-curated, columns: `nsclc_celltype, projectils_state, weight`. The mapping is straightforward (CD8+ T cells → CD8 states, CD4+ T cells → CD4 states, Treg → Treg). Unmapped types (tumor, stromal, etc.) contribute to a pooled "other_immune" category surfaced for completeness but not rolled up into any ProjecTILs state.

---

## 4. Output metrics — precise specification

### Script: `alz/reference/nsclc_expression.py`

Mirrors `alz/reference/wmb_expression.py` in structure. Consumes:
- `data/external/nsclc_10x/sample_feature_bc_matrix.h5` (HDF5 filtered matrix)
- `data/external/nsclc_10x/cell_type_annotations.csv` (per-cell cluster assignment)

Produces: `outputs/reports/nsclc_reference/nsclc_kinase_expression.csv`

**Schema** (mirrors WMB output):

| column | type | definition |
|--------|------|------------|
| `kinase_id` | str | HGNC uppercase symbol (e.g. `AKT1`) |
| `gene_symbol` | str | same as kinase_id (human dataset, no case conversion) |
| `cell_type` | str | NSCLC annotation label (e.g. `CD8+ T cells`) |
| `mean_log2_expression` | float | mean log2(count+1) across all cells of that type |
| `fraction_cells_expressing` | float | (cells with count > 0) / n_cells |
| `binary_expressed` | bool | `mean_log2_expression > 1` AND `fraction_cells_expressing > 0.10` |
| `specificity_score` | float | share of total mean log2 expr across all cell types (per gene) |
| `n_cells` | int | cell count for this cell type |
| `probe_covered` | bool | whether the gene is in the Flex probe panel |

**Metric (a) — cell types expressing the kinase:** `binary_expressed == True` rows per kinase, listing `cell_type` values.

**Metric (b) — % of cells expressing:** `fraction_cells_expressing` column, directly readable as "X% of CD8+ T cells express AKT1."

**Expression threshold:** adopt the WMB-identical definition — `mean_log2_expression > 1` AND `fraction_cells_expressing > 0.10` — for cross-cohort comparability.

**Probe coverage column:** Because the Flex panel does not cover all genes, `probe_covered = False` kinases are structurally absent (not biologically absent). The audit must distinguish "not in panel" from "in panel but not expressed". Add `probe_covered` boolean derived by intersecting the feature list in the HDF5 with the kinase gene universe. Kinases with `probe_covered = False` are excluded from the "absent from reference" audit finding; their absence is a panel limitation, not a biology statement.

### Processing recipe

1. Open HDF5 with h5py in read-only mode (no anndata, no full materialization).
2. Extract feature names (gene symbols) from `matrix/features/name` — these are HGNC symbols for the Flex panel.
3. Mark `probe_covered` for each kinase by intersection.
4. Load per-cell type annotations from `cell_type_annotations.csv` → `{barcode: cell_type}` dict.
5. Stream CSR matrix `matrix/data` / `matrix/indices` / `matrix/indptr` in chunks of 5000 cells. For each chunk, scatter-add into per-cell-type accumulators (same `np.add.at` pattern as `atlas.py::download_sea_ad_expression`).
6. Compute mean + fraction per (kinase, cell_type). Assign `binary_expressed` and `specificity_score`.
7. Write CSV. Write scope sidecar JSON.

---

## 5. Audit deliverable

### Script: `alz/reference/nsclc_expression.py --audit`

Reads:
- `outputs/reports/nsclc_reference/nsclc_kinase_expression.csv` (produced above)
- `outputs/reports/kinase_attribution_tcells/donor1/mea/mea_timecourse.csv` (MEA predictions)

Produces: `outputs/reports/nsclc_reference/nsclc_kinase_audit.csv`

**Schema:**

| column | definition |
|--------|------------|
| `kinase` | kinase abbreviation (e.g. `AKT1`) |
| `gene_symbol` | HGNC gene symbol |
| `probe_covered` | is gene in the Flex panel? |
| `binary_expressed_any` | True if expressed (`binary_expressed`) in any cell type |
| `expressed_cell_types` | comma-delimited list of cell types where `binary_expressed=True` |
| `max_fraction_expressing` | max `fraction_cells_expressing` across all cell types |
| `max_fraction_cell_type` | cell type achieving that max |
| `is_mea_predicted` | True if kinase appears in MEA output at FDR < 0.25 in any contrast |
| `audit_flag` | one of: `expressed`, `not_expressed_in_panel`, `not_in_probe_panel` |

**Key finding:** kinases where `is_mea_predicted=True` AND `probe_covered=True` AND `binary_expressed_any=False` — these are MEA-predicted but absent from the reference (not absent due to probe gap). This is the primary audit output.

**Printed summary** (--audit stdout):
```
NSCLC reference audit: T-cell MEA kinases
  MEA-predicted (FDR<0.25): N
  Probe panel covered: M / N
  Expressed in ≥1 cell type: K / M (panel-covered)
  NOT expressed (panel-covered): M-K  ← the finding
    [list kinase names]
```

---

## 6. Viewer integration

### Payload key

New audit table key in `_KINASE_AUDIT_SHIMS` → remove it once the artifact exists. Replace the shim with a registered audit table under key `nsclc_kinase_expression` in `_register_kinase_audit_tables()` in `alz/build_tcell_viewer.py`.

The payload adds `nsclc_kinase_expression` analogously to how `wmb_kinase_expression` is shimmed today (line 1690). Once the artifact exists:
1. Remove the shim entry for `nsclc_kinase_expression` from `_KINASE_AUDIT_SHIMS`.
2. Add to `_KINASE_AUDIT_FILES` tuple: `("nsclc_kinase_expression", "../../nsclc_reference/nsclc_kinase_expression.csv", "NSCLC 10x reference (human immune/T-cell)", [])`.
3. Flip `"human_reference": False` → `True` at lines 2364 and 2419 (per-donor context capability and global capability).

The viewer Attribution tab for the T-cell cohort will show the NSCLC specificity column alongside the within-cohort ProjecTILs specificity, giving two sources: (a) the cohort's own scRNA and (b) the independent public reference.

### Payload columns surfaced per kinase

For each kinase row in the donor1 kinase slice, add:
- `nsclc_expressed_cell_types` — list of cell types where `binary_expressed=True`
- `nsclc_max_fraction` — `max_fraction_expressing` from the audit table
- `nsclc_probe_covered` — whether the gene is panel-measurable

These are parallel to `wmb_specificity` in the mouse viewer and `seaad_location_score` in the human viewer.

---

## 7. Config additions

Add to `alz/shared/config.py` (analogous to `WMB_EXPRESSION_OUTPUT_DIR` and `HUMAN_REFERENCE_OUTPUT_DIR`):

```python
NSCLC_10X_CACHE_DIR = os.path.join(EXTERNAL_DATA_DIR, "nsclc_10x")
NSCLC_10X_H5_FILE = os.path.join(NSCLC_10X_CACHE_DIR, "sample_feature_bc_matrix.h5")
NSCLC_10X_ANNOTATIONS_FILE = os.path.join(NSCLC_10X_CACHE_DIR, "cell_type_annotations.csv")
NSCLC_REFERENCE_OUTPUT_DIR = os.path.join("outputs", "reports", "nsclc_reference")
NSCLC_KINASE_EXPRESSION_FILE = os.path.join(NSCLC_REFERENCE_OUTPUT_DIR, "nsclc_kinase_expression.csv")
NSCLC_KINASE_AUDIT_FILE = os.path.join(NSCLC_REFERENCE_OUTPUT_DIR, "nsclc_kinase_audit.csv")
```

---

## 8. New pixi tasks

```toml
nsclc-ingest = "bash alz/runners/supporting/ingest_nsclc_10x.sh"
nsclc-expression = "python alz/reference/nsclc_expression.py --run"
nsclc-audit = "python alz/reference/nsclc_expression.py --audit"
tcell-viewer = { cmd = "python alz/build_tcell_viewer.py", depends-on = ["tcell-within-cohort", "nsclc-audit"] }
```

The `ingest_nsclc_10x.sh` script downloads via `wget` or `curl` to `data/external/nsclc_10x/`. The 10x Genomics download URL pattern is `https://cf.10xgenomics.com/samples/<dataset>/sample_filtered_feature_bc_matrix.h5`. The exact URL must be confirmed from the dataset page at time of execution (the slug is not guessable from the dataset description alone).

---

## 9. Execution order

1. Confirm exact download URLs from https://www.10xgenomics.com/datasets/aggregate-of-900k-human-non-small-cell-lung-cancer-and-normal-adjacent-cells-multiplexed-samples-16-probe-barcodes-1-standard — specifically: the filtered matrix HDF5 and the per-cell type annotation file (may be in `analysis/clustering/` or as a summary CSV).
2. Write `alz/runners/supporting/ingest_nsclc_10x.sh` with the confirmed URLs.
3. `pixi run nsclc-ingest` — fetch files to `data/external/nsclc_10x/`.
4. Inspect schema: `python -c "import h5py; f=h5py.File('data/external/nsclc_10x/...h5','r'); print(list(f['matrix'].keys())); print(f['matrix/features/name'][:10])"` — confirm gene symbol format and feature count.
5. Write `alz/reference/nsclc_expression.py` following the spec above.
6. Add constants to `alz/shared/config.py`.
7. `pixi run nsclc-expression` → verify `nsclc_kinase_expression.csv` row count (expect: n_kinases × n_cell_types rows).
8. `pixi run nsclc-audit` → print summary; inspect the "not expressed (panel-covered)" list.
9. Wire into `alz/build_tcell_viewer.py`: remove shim, register audit table, flip capability flag.
10. `pixi run tcell-viewer` — verify `human_reference: true` in generated payload.

---

## 10. Known risks and mitigations

| Risk | Mitigation |
|------|-----------|
| Flex probe panel covers only ~18K genes; many kinases absent | `probe_covered` column separates panel gaps from biology; audit excludes uncovered genes from findings |
| 10x annotation vocabulary not a standard ontology; mapping to ProjecTILs states requires curation | Hand-curate `nsclc_celltype_to_projectils.csv` (small, ~15 cell types); document in the file header |
| HDF5 feature names may be Ensembl IDs, not HGNC symbols (depends on 10x reference genome used) | Inspect at schema-confirmation step (step 4); if Ensembl, add a mapping step using `data/derived/caches/kinase_to_gene_mapping.csv` to cross-reference |
| Cell count imbalance (900K cells, but T-cell fraction may be small) | `n_cells` column exposes this; `fraction_cells_expressing` is normalized within each cell type so a small T-cell count gives a valid fraction — just flag small n_cells in the audit |
| OOM on 900K cells | h5py chunked streaming + RLIMIT_AS cap at 16 GB; same approach as SEA-AD (1.38M cells handled cleanly) |
