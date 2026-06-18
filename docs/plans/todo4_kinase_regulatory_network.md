# Plan: Kinase Regulatory Network Extension

**Goal:** Build an extension to all kinase MEA analyses that models kinase-kinase regulatory relationships, visualizes the network/hierarchy, and tests whether observed disease-driven activity changes (MEA NES up/down) corroborate or contradict expected regulatory directionality from the Phosphosite Kinase Library.

---

## 1. Tool and Data Availability

### kinase-library (already in env)
`kinase-library==1.7.0` is installed in the pixi default env and imported successfully (`pixi run python -c "import kinase_library; print(kinase_library.__version__)"`). The version pin is already reconciled in `pixi.toml` (numpy~=1.26.4, pandas~=2.2.3, scipy~=1.14.1).

What the library **does** provide:
- `kinase_library.get_scored_phosphoproteome('ser_thr')` — 82,735 sites × 311 kinase scoring matrix (Ochoa phosphoproteome). Of those sites, 3,820 map to kinase-gene substrates (343 unique kinase-substrate genes), per `pp['gene'].isin(kinase_genes)`. These are the raw material for constructing kinase-kinase regulatory edges from motif-scoring evidence.
- `kinase_library.get_kinome_info()` — 400 kinases with FAMILY, TYPE (ser_thr/tyrosine), UNIPROT_ID, GENE_NAME.
- `kinase_library.get_kinase_family()` — already called by `build_unified_viewer.py` line 938.

What the library **does not** provide:
- Regulatory-consequence annotation (activating vs. inhibiting) for kinase-on-kinase phosphorylation events. The scored phosphoproteome gives "which kinase is predicted to phosphorylate which site on kinase B," but not the downstream activity effect.

### PhosphoSitePlus Regulatory Sites (external, not on-disk)
PhosphoSitePlus distributes `Regulatory_sites.gz` as a free (registration-gated) download. This file records, per phosphosite: `GENE`, `MOD_RSD`, `DOMAIN`, `ON_FUNCTION`, `ON_PROCESS`, `ON_PROT_INTERACT`, `ON_OTHER_INTERACT`, `PMIDs`, `CST_Catalog#`, `notes`. The `ON_FUNCTION` field includes entries like `"induced: kinase activity"`, `"inhibited: kinase activity"`, `"induced: protein binding"`. This is the authoritative source for regulatory sign on individual sites.

**Fetching:** Download `Regulatory_sites.gz` from https://www.phosphosite.org/staticDownloads → "Regulatory sites" → mouse+human. Registration required (free). Place at `data/external/phosphosite/Regulatory_sites.gz`. Add to `conf/data_sources.yaml` as a manual (non-rclone) external dependency with a provenance note (no automated sync, update annually).

### OmniPath / SignaLink (alternative, not installed)
`omnipath` and `pypath` are not in the pixi env. OmniPath aggregates multiple regulatory databases including NetworKIN and PhosphoSitePlus; it can be queried via its REST API (`https://omnipathdb.org/interactions`) without a local install. It is a reasonable fallback if the PSP download is delayed; however, PSP is preferred because it is the authoritative source used by the kinase library's substrate scoring. Plan for PSP primary, OmniPath REST as a fallback.

---

## 2. Kinase MEA Outputs Across All Cohorts

All cohorts share the same long-table schema: `kinase, ES, NES, p-value, FDR, Subs fraction, Leading substrates, contrast, residue_type, track`.

| Cohort | Primary MEA file(s) | Contrasts | Notes |
|---|---|---|---|
| Song (mouse-AD) | `outputs/reports/kinase_attribution/mea_stoichiometry.csv`, `mea_raw_phospho.csv`, `*_pY.csv` | App_{2,6mo}, Tau_{2,6mo}, ApTt_{2,6mo} | Bulk. `unified_attribution.csv` is the merged downstream table. Decomposition per-cluster: `outputs/reports/decomposition/levy_t5/mea_per_cluster.parquet` (53k rows, `cluster` + `kinase` + `NES` + `FDR` + `contrast`) |
| Mukesh (human-AD) | `outputs/reports/kinase_attribution_human/perdonor/mea_perdonor.csv`, `*_pY.csv` | `{AD-NN}_vs_CTRLmean` (per-donor) | Per-donor NES; aggregate in `kinase_donor_nes.csv` / `recurrence.csv` |
| T-cell | `outputs/reports/kinase_attribution_tcells/donor1/mea/mea_timecourse.csv`, `*_raw.csv`, `*_pY.csv` | `D1_{dN}_vs_d2` | Per-timepoint NES; donor2 has no IMAC → kinase MEA on bulk only, no per-donor decomp |
| 5xFAD | `outputs/reports/kinase_attribution_5xfad/{cortex,hippocampus}_{st,py}_mea_{stoichiometry,raw_phospho}.csv` | WT vs 5xFAD (separate by tissue and track) | 5xFAD proteomics on hold per memory note; 8 MEA files exist but cohort is not fully integrated into the viewer |

The extension must join on `kinase` (kinase library abbreviation, e.g. `CDK5`) in all cases. The `gene_symbol` column in the unified attribution (derived via `alz/shared/map_kinases_to_genes.py` → `data/derived/caches/kinase_to_gene_mapping.csv`) maps abbreviation → HGNC gene symbol for external database lookups.

**NES direction convention:** `NES > 0` = kinase activity up in disease (per global LFC sign rule). This applies to all MEA outputs without exception.

---

## 3. Co-Expression Evidence Per Cohort

The goal is to gate "these two kinases could plausibly regulate each other in this tissue" via co-expression in the same cell type.

### Mouse-AD (Song/5xFAD) — WMB reference
`outputs/reports/wmb_expression/wmb_kinase_expression.csv` — schema: `kinase_id, gene_symbol, cell_type, mean_log2_expression, fraction_cells_expressing, specificity_score, binary_expressed, n_cells`. Nine WMB superclasses (e.g., `01 IT-ET Glut`, `34 Immune`).

`outputs/reports/wmb_expression/wmb_kinase_expression_subclass.csv` provides finer resolution.

**Co-expression gate:** both kinases have `binary_expressed = True` in the same WMB cell_type. Soft version: both have `fraction_cells_expressing > 0.1` and `mean_log2_expression > 2`. The existing `unified_attribution.csv` already carries `wmb_binary_expressed` and `wmb_class` per kinase; the extension joins pairs on matching `wmb_class`.

### Human-AD (Mukesh) — SEA-AD reference
`outputs/reports/human_reference_expression/seaad_kinase_specificity.csv` — 309-column wide matrix: rows = kinase, columns = SEA-AD supertype clusters. Values are specificity scores (z-scores from the `alz/reference/human_expression.py` pipeline).

`outputs/reports/human_reference_expression/seaad_kinase_expression.csv` — complementary expression values.

**Co-expression gate:** both kinases have `seaad_kinase_specificity > threshold` (e.g., > 0) in the same supertype, or use the `celltype_specificity.csv` in `kinase_attribution_human/` which already ranks per-kinase top cell types.

### T-cell — within-cohort T-cell reference
`outputs/reports/kinase_attribution_tcells/donor{1,2}/tcell_specificity.csv` — schema: `gene, state, tcell_specificity, tcell_mean_log2_expression`. T-cell states are the ProjectILS/Seurat clusters (CD4CTLeomes, CD4CTLexh, etc.).

**Co-expression gate:** both kinases have `tcell_specificity > 0` in the same T-cell state. Note: T-cell cohort is a mechanistic (not disease-vs-control) cohort; regulatory corroboration here tests whether the regulatory logic holds during T-cell exhaustion dynamics.

### Per-cluster decomposition (Song only)
`mea_per_cluster.parquet` carries cluster-level NES. The `mea_substrate_sets_per_cluster.csv` (>1GB, do not load into memory) can be queried via DuckDB to extract leading-edge substrates per cluster. Co-expression at cluster level is from `outputs/reports/song_expression/` (check `pixi task list` → `snrna` for details).

---

## 4. Regulatory-Edge Data Model

### 4a. Building the edge table

**Step 1: Extract kinase-substrate pairs from the scored phosphoproteome**

```python
# pseudo-code
spp = kl.get_scored_phosphoproteome('ser_thr')   # 82k sites × 311 kinases
pp  = kl.get_phosphoproteome()                    # 88k rows: uniprot, gene, position, residue, Sequence
kinome = kl.get_kinome_info()
kinase_genes = set(kinome['GENE_NAME'].dropna().str.upper())

# Sites on kinase proteins (substrate_gene ∈ kinase_genes)
kk_sites = pp[pp['gene'].str.upper().isin(kinase_genes)].copy()  # ~3,820 sites, 343 genes

# For each site, the scoring row in spp gives the predicted upstream kinase
# Subset spp to these kinase-substrate sites
kk_scores = spp.loc[kk_sites['Sequence'].values]  # align on motif sequence (index of spp)
# Top predicted upstream kinase per site (e.g., top-1 or top-3 by percentile)
# kk_scores: sites × 311 kinases → take argmax / nlargest per row
```

This produces: `(upstream_kinase, substrate_gene, site, motif_score)` triplets. The site context (which residue on the substrate kinase) is critical for the PSP regulatory lookup.

**Step 2: Join PSP Regulatory_sites for consequence sign**

After `data/external/phosphosite/Regulatory_sites.gz` is downloaded:

```
PSP schema: GENE, PROTEIN, ACC_ID, HU_CHR_LOC, MOD_RSD, SITE_GRP_ID, 
            ORGANISM, MW_kD, DOMAIN, ON_FUNCTION, ON_PROCESS, 
            ON_PROT_INTERACT, ON_OTHER_INTERACT, PMIDs, CST_Catalog#, notes
```

Join on `(substrate_gene, MOD_RSD)` where `MOD_RSD` encodes residue+position (e.g., `S473-p`). Filter `ON_FUNCTION` for kinase-activity consequences:
- `induced: kinase activity` or `increased: kinase activity` → `regulatory_sign = +1` (phosphorylation activates)
- `inhibited: kinase activity` or `decreased: kinase activity` → `regulatory_sign = -1` (phosphorylation inhibits)
- Discard sites with ambiguous or absent `ON_FUNCTION`

**Edge schema (canonical CSV: `data/derived/kinase_regulatory_edges.csv`)**:

| Column | Type | Description |
|---|---|---|
| `upstream_kinase` | str | Kinase-library abbreviation of the upstream kinase (phosphorylator) |
| `upstream_gene` | str | HGNC gene symbol (from `kinase_to_gene_mapping.csv`) |
| `substrate_kinase` | str | Kinase-library abbreviation of the downstream kinase (phosphorylated) |
| `substrate_gene` | str | HGNC gene symbol |
| `site_id` | str | `{gene}_{residue}{position}`, e.g. `CDK5_S159` |
| `motif` | str | 15-aa motif sequence |
| `motif_score_percentile` | float | Kinase-library percentile rank of upstream_kinase scoring this site |
| `regulatory_sign` | int | +1 (activating) or -1 (inhibiting); from PSP ON_FUNCTION |
| `psp_pmids` | str | Supporting PMIDs from PSP |
| `source` | str | `psp` (primary) or `omnipath` (fallback) |
| `track` | str | `ser_thr` or `tyrosine` (residue type of the regulatory site) |

**Edge quality gates:**
- `motif_score_percentile > 75` (top quartile predicted substrate)
- `regulatory_sign` is not null (PSP annotation required)
- Both `upstream_kinase` and `substrate_kinase` must appear in `kl.get_kinase_list()` so they can be looked up in MEA outputs

**Producing this file:** Script `alz/shared/build_kinase_regulatory_edges.py`. Depends on PSP download being in place. Cache-regenerable (same pattern as `map_kinases_to_genes.py`). Add pixi task `kinase-regulatory-edges`.

### 4b. Corroboration join

For each cohort contrast, join the edge table against MEA outputs:

```
Edge table:  upstream_kinase, substrate_kinase, regulatory_sign
MEA table:   kinase, NES, FDR, contrast (one row per kinase per contrast)
```

Join twice:
- `edge.upstream_kinase` → `mea.kinase` → get `upstream_NES`, `upstream_FDR`
- `edge.substrate_kinase` → `mea.kinase` → get `substrate_NES`, `substrate_FDR`

**Corroboration call schema:**

| Column | Type | Description |
|---|---|---|
| `upstream_kinase` | str | |
| `substrate_kinase` | str | |
| `site_id` | str | |
| `regulatory_sign` | int | +1 / -1 |
| `upstream_NES` | float | From MEA; NaN if not measured |
| `substrate_NES` | float | From MEA; NaN if not measured |
| `upstream_sig` | bool | FDR < threshold |
| `substrate_sig` | bool | FDR < threshold |
| `observed_direction_product` | float | `sign(upstream_NES) * sign(substrate_NES)` (+1 if same direction) |
| `expected_direction_product` | int | `regulatory_sign` (+1 = substrate should follow upstream; -1 = oppose) |
| `corroboration` | str | See below |
| `cohort` | str | `song`, `mukesh`, `tcell`, `5xfad` |
| `contrast` | str | |
| `track` | str | `st` or `py` |
| `coexpressed` | bool | Co-expression gate passed for this cohort/cell-type |

**`corroboration` values:**

- `CORROBORATED` — both sig; `observed_direction_product == expected_direction_product`
- `CONTRADICTED` — both sig; `observed_direction_product != expected_direction_product` (the biologically interesting case: expected cascade broken)
- `PARTIAL_UPSTREAM_ONLY` — upstream sig, substrate not sig
- `PARTIAL_SUBSTRATE_ONLY` — substrate sig, upstream not sig
- `NEITHER_SIG` — neither reaches FDR threshold
- `NOT_MEASURED` — one or both kinases absent from this cohort's MEA (e.g., not in the phosphoproteome)

**FDR threshold:** Use `alz.shared.config.MEA_FDR_THRESH` (currently 0.25, per analysis charter).

**Output file:** `data/derived/kinase_regulatory_corroboration_{cohort}.csv` (one per cohort, regenerable).

**Producer script:** `alz/shared/build_regulatory_corroboration.py`
- Takes `--cohort {song,mukesh,tcell,5xfad}` and `--mea-path`
- Loads `data/derived/kinase_regulatory_edges.csv`
- Loads cohort MEA file(s)
- Applies co-expression gate for the cohort
- Emits the corroboration CSV

Pixi tasks: `regulatory-corroboration-song`, `regulatory-corroboration-human`, `regulatory-corroboration-tcell`, `regulatory-corroboration-5xfad`.

---

## 5. Visualization

### 5a. Standalone artifact (primary deliverable)

A self-contained HTML network viewer:
- `outputs/reports/kinase_regulatory_network/index.html` (built by `alz/shared/build_regulatory_network_viewer.py`)
- Uses D3.js force-directed graph or a hierarchical (Sugiyama-layout) view via Dagre.js
- Nodes: kinases, colored by MEA direction (red = up, blue = down, grey = not sig), sized by |NES|
- Edges: directed arrows colored by `corroboration` (green = CORROBORATED, orange = CONTRADICTED, grey = PARTIAL/NEITHER_SIG)
- Edge tooltip: site_id, motif, motif_score_percentile, PSP PMIDs, regulatory_sign, upstream_NES, substrate_NES
- Node tooltip: kinase name, gene_symbol, family, top cell type (WMB/SEA-AD), confidence_tier from unified_attribution
- Controls: cohort selector, contrast selector, track selector (ST/pY), FDR threshold slider, co-expression filter toggle

**Why standalone:** The regulatory network crosses cohorts and is not naturally a per-context tab in the existing unified viewer (which is cohort+context-aware). The existing viewer already has complexity budget pressures (todo8 audit). A standalone artifact avoids touching the viewer payload contract.

### 5b. Optional viewer integration (deferred)

If the viewer integration is later desired, the corroboration table can be added as a sidecar `edge_slices/kinase_regulatory/` directory mirroring the `edge_slices/human_perdonor/` pattern. The `kinase_explorer.js` tab's node-detail panel could link to the regulatory context. This is out of scope for this plan.

### 5c. Summary table

`outputs/reports/kinase_regulatory_network/corroboration_summary.csv`:
- Per-kinase pair: n_contrasts_corroborated, n_contrasts_contradicted, n_cohorts_with_signal, max_|NES|_upstream, max_|NES|_substrate
- Sort: CONTRADICTED first (most biologically interpretable), then CORROBORATED

---

## 6. Literature Grounding

Before finalizing any biological interpretation in the plan output or downstream reports:

**Step:** Run PubMed / bioRxiv / Scholar Gateway MCP queries for all CONTRADICTED edges that appear in ≥2 cohorts. Specific queries to run at interpretation time:
- `"{upstream_kinase} phosphorylates {substrate_kinase}"` — check if PSP annotation is well-supported
- `"{upstream_kinase} {substrate_kinase} Alzheimer"` — any AD-specific regulatory evidence
- `"{substrate_kinase} activity inhibition {upstream_kinase}"` — corroborate the regulatory sign

Do not contact collaborators at any stage. Literature grounding is a pre-publication step, not a blocker for building the analysis infrastructure.

---

## 7. Implementation Phases

**Phase 0 — External data acquisition (manual, blocked until PSP download)**
- Download `Regulatory_sites.gz` from PhosphoSitePlus (requires account registration)
- Place at `data/external/phosphosite/Regulatory_sites.gz`
- Document provenance in `data/external/phosphosite/MANIFEST.json` (date, version, download URL)
- Add to `conf/data_sources.yaml` as manual dependency

**Phase 1 — Edge table builder** (`alz/shared/build_kinase_regulatory_edges.py`)
- Load scored phosphoproteome → filter to kinase-substrate sites
- Load PSP Regulatory_sites → parse `ON_FUNCTION` for kinase-activity consequences
- Join on gene + residue+position; map to motif percentile score
- Apply quality gates (`motif_score_percentile > 75`, `regulatory_sign` not null)
- Write `data/derived/kinase_regulatory_edges.csv`
- Report: n_edges total, n_activating, n_inhibiting, n_unique_upstream_kinases, n_unique_substrate_kinases

**Phase 2 — Corroboration builder** (`alz/shared/build_regulatory_corroboration.py`)
- Parametric by cohort; load MEA file(s) and co-expression reference
- Join edges against MEA on both endpoints
- Compute `corroboration` call per edge per contrast
- Apply co-expression gate
- Write `data/derived/kinase_regulatory_corroboration_{cohort}.csv`
- Sanity check: CONTRADICTED edges should be few (check that n_CONTRADICTED is not 0 and not >50% — either extreme indicates a join or sign-convention bug)

**Phase 3 — Network viewer** (`alz/shared/build_regulatory_network_viewer.py`)
- Inline all corroboration data as JSON payload
- Build D3/Dagre HTML with cohort/contrast/track selectors
- Output `outputs/reports/kinase_regulatory_network/index.html`

**Phase 4 — Literature grounding**
- For CONTRADICTED edges in ≥2 cohorts, run PubMed MCP queries
- Document in `docs/vignettes/kinase_regulatory_network_biology.md`

**Verification:** After Phase 2, run `pixi run python alz/shared/build_regulatory_corroboration.py --cohort song` and confirm:
- The Song App_2mo contrast produces ≥1 CORROBORATED edge (sanity that the pipeline ran correctly)
- No kinase appears as its own upstream (self-loop guard)
- All `upstream_NES` values match the corresponding row in `mea_stoichiometry.csv` to floating-point precision

---

## 8. Files To Create

```
alz/shared/build_kinase_regulatory_edges.py        (Phase 1)
alz/shared/build_regulatory_corroboration.py       (Phase 2)
alz/shared/build_regulatory_network_viewer.py      (Phase 3)
data/external/phosphosite/MANIFEST.json            (Phase 0, manual)
data/derived/kinase_regulatory_edges.csv           (Phase 1 output)
data/derived/kinase_regulatory_corroboration_*.csv (Phase 2 output)
outputs/reports/kinase_regulatory_network/index.html  (Phase 3 output)
outputs/reports/kinase_regulatory_network/corroboration_summary.csv (Phase 3 output)
```

No changes to existing MEA runner, adapters, or viewer.

---

## 9. Known Gaps and Flags

1. **PSP access is blocked** until manual download. The kinase-library's own substrate data (`Kinase_Substrate_Dataset_count_07_2021.txt`) is binary-encoded and not directly parseable as UTF-8; it is not a substitute for the regulatory consequence annotations in `Regulatory_sites.gz`.

2. **5xFAD cohort is on hold** per memory note (Cortex-IMAC & Hippo-Total exist only as .sne files). The 5xFAD MEA files (`cortex_st_mea_stoichiometry.csv` etc.) exist and can be joined, but cell-type co-expression evidence is limited to the 5xFAD snRNA integration (if available). Include 5xFAD in Phase 2 only if the snRNA integration is complete.

3. **T-cell regulatory interpretation** is mechanistically distinct: the "disease" direction convention (NES > 0 = up in disease) maps to "up at late exhaustion timepoints" in the T-cell context. The corroboration logic is identical but the biological question shifts to "does the upstream kinase activity increase precede a change in the downstream kinase as exhaustion deepens?" This is valid but should be clearly labeled in the viewer.

4. **Motif percentile score is predicted, not validated.** The edge from kinase-library scored phosphoproteome represents computationally predicted substrate preference. For the CONTRADICTED calls, the PSP annotation anchors the regulatory sign but the upstream-kinase assignment is still probabilistic. High-confidence calls require `motif_score_percentile > 90` or independent literature corroboration (Phase 4).

5. **Decomposition per-cluster MEA** (`mea_per_cluster.parquet`) adds cluster resolution for Song. Query via DuckDB to avoid loading 33MB in memory: `duckdb -c "SELECT cluster, kinase, NES, FDR, contrast FROM read_parquet('...') WHERE FDR < 0.25"`. This enables cell-type-resolved corroboration for the Song cohort.
