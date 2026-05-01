# Song Bulk Deconvolution → Per-Cell-Type Kinase Enrichment

Plan for a chief-scientist deliverable that takes the pre-computed Song proportional deconvolution at 46 Yuyu clusters, runs kinase MEA per cell type for both ser/thr and tyr tracks, and accompanies each row with a transcript-level reliability indicator drawn from matched snRNA-seq.

## 1. Goal and audience

**Goal.** Produce a per-cell-type kinase enrichment table (NES + FDR per kinase × cell type × contrast, ser/thr and tyr tracks) from proportionally deconvoluted Song bulk phosphoproteomics, with a defensible per-row reliability indicator.

**Audience.** Chief scientist. Argument leads with the deliverable; uncertainty surfaces as per-row confidence rather than a global feasibility verdict; the closed direct-deconvolution path is not reopened.

**Framing.** This is a **proportion-proxy deconvolution**, not the direct deconvolution closed by the analysis charter. Proportional decomposition redistributes bulk signal across cell types using Song's per-gene scRNA expression as a within-cohort prior — distinct from inferring cell-type-specific effects from `A_obs + bulk` alone.

## 2. Cell-type vocabulary

**46 named clusters** from the 2025-07-21 refresh (`deconvolution_with_new_clusters_20250721/`). The chief scientist's reference resolution.

The existing attribution pipeline (live program) stays at 34 WMB classes — this plan does not touch that. The two outputs cohabit at different vocabularies, joined via a curated mapping:

- **`yuyu_46_to_wmb_class.csv`** — first-draft mapping authored from cluster name semantics + WMB class definitions, committed to the repo, surfaces ambiguity in a `notes` column. Residual `cluster-N` (~15 unnamed clusters) → `Unclassified`.
- Each row of the deconvolution output carries a `wmb_class` column.
- A **secondary rolled-up view** at WMB class is produced as a cross-reference table for direct comparability with the attribution output.

## 3. Inputs

All present locally under `data/raw/external/gdrive_shared/integrations/yuyu01/documentation/incytr/deconvolution/deconvolution_with_new_clusters_20250721/`:

| File | Role |
|---|---|
| `ps_yuyu_deconvoluted.csv` (271 MB) | Pre-computed ser/thr deconvolved phospho per site × cell type × sample (24 groups) |
| `py_yuyu_deconvoluted.csv` (19 MB) | Pre-computed tyr deconvolved phospho per site × cell type × sample (24 groups) |
| `pr_yuyu_deconvoluted.csv` (134 MB) | Pre-computed total protein deconvolved per gene × cell type × sample (for stoichiometry, optional) |
| `yuyu_clustersize.csv` | 46 clusters × 24 samples cell counts |
| `kr_cluster_id_key.csv` | 109 fine cluster IDs → 46 named cell types |
| `yuyu_samplekey.csv` | MS_ID ↔ scRNA group key |

snRNA cross-check oracle: re-run of `code/snrna_integration.py --concordance` at the 46-cluster vocabulary (currently at 34 WMB classes), output to `outputs/reports/snrna_integration/song_concordance_46clusters.csv`. Produces per-gene LFC per cluster per contrast under males-only factorial OLS.

## 4. Stage-by-stage pipeline

### Stage 1 — Adopt pre-computed deconvolution outputs

No recomputation. Load `ps_yuyu_deconvoluted.csv` and `py_yuyu_deconvoluted.csv` as the per-site × cell-type × sample matrices. Track-by-track from here.

### Stage 2 — Males-only factorial OLS per cell type

For each track (S/T, Y) and each cell type (46), filter to the 12 male sample groups, fit the same factorial OLS used in the live pipeline (`const, App, Tau, Int, time_4mo, time_6mo, App×time4, App×time6, Tau×time4, Tau×time6`), and extract per-site β + p-value for the 9 contrasts.

Cohort filter happens here, not at deconvolution time — pre-computed outputs remain at 24 groups; OLS uses only the 12 males.

### Stage 3 — Kinase MEA per cell type, per track

For each cell type × contrast × track:

- **Ser/thr track**: rank phosphosites from Stage 2 by β; run kinase library MEA against the **ser/thr kinase substrate set**.
- **Tyr track**: rank phosphosites from Stage 2 by β; run kinase library MEA against the **tyrosine kinase substrate set**.

No mixing of motif spaces. Two independent MEA runs feed into one merged output table.

Output: NES + FDR per kinase × cell type × contrast × track. Significance threshold: FDR < 0.25 (kinase library / GSEA standard, matches live pipeline).

### Stage 4 — snRNA cross-check (kinase gene LFC concordance)

Re-run `snrna_integration.py --concordance` at 46-cluster vocabulary (males-only factorial OLS on Song pseudobulk).

For each row of the Stage 3 output, look up the **kinase's own gene LFC** in the snRNA at the same cell type and contrast. Annotate three columns:

- `kinase_gene_LFC_snRNA` — signed magnitude
- `kinase_gene_FDR_snRNA` — significance
- `direction_match` — `match` / `opposite` / `flat` / `n/a`

This is transcript-level corroboration, not parallel phospho measurement. The snRNA is a collaborator to the deconvolution, not a replacement.

### Stage 5 — Confidence calibration

Per-row confidence label, computed deterministically:

| Confidence | Condition |
|---|---|
| **High** | Deconv FDR < 0.25 + snRNA gene FDR < 0.10 + direction_match = `match` |
| **Moderate** | Deconv FDR < 0.25 + snRNA flat or n/a (post-translational regulation plausible, not corroborated) |
| **Low** | Deconv FDR < 0.25 + snRNA significant in opposite direction |
| **Insufficient** | Cluster has < 20 cells in any sample group of this contrast (power floor breached) |

Thresholds documented and tunable in code; defaults set in advance to avoid post-hoc tuning.

### Stage 6 — Roll-up to WMB classes (secondary view)

Aggregate Stage 5 output to 34 WMB classes via `yuyu_46_to_wmb_class.csv`. Aggregation rule: take the **strongest-evidence row** per kinase × WMB class × contrast × track (highest |NES| at deconv FDR < 0.25; ties broken by FDR), with cluster-of-origin annotated. `Unclassified` clusters excluded from the rolled-up view.

This produces a parallel table at WMB-class resolution for direct cross-referencing with the live attribution pipeline.

## 5. Deliverables

### Primary table
`outputs/reports/deconvolution/kinase_enrichment_46clusters.csv`

| kinase | cell_type | wmb_class | contrast | track | NES | FDR | kinase_gene_LFC_snRNA | kinase_gene_FDR_snRNA | direction_match | confidence | n_cells_min |

### Secondary table (rolled-up view)
`outputs/reports/deconvolution/kinase_enrichment_wmb_rollup.csv`

Same schema, aggregated to 34 WMB classes.

### Cluster mapping artifact
`code/deconvolution/yuyu_46_to_wmb_class.csv`

Curated mapping with `cluster_name`, `wmb_class`, `confidence_in_mapping` (clean / ambiguous / unmapped), `notes`.

### Methods write-up
`docs/deconvolution_kinase_enrichment.md` — walks the chief scientist through:

1. The proportion-proxy method (legacy-faithful, at 46 clusters)
2. Two-track MEA (S/T and Y separated, with substrate-set rationale)
3. snRNA cross-check as transcript-level corroboration (with honest framing of what it does and doesn't validate)
4. How to read the confidence column
5. Navigation between the 46-cluster primary and the WMB-class roll-up

## 6. Repo layout

New module `code/deconvolution/` (chief-scientist deliverable, not wired into `pixi run live` or `pixi run dual`):

```
code/deconvolution/
├── README.md                          # charter footnote: proportion-proxy, not direct deconvolution
├── yuyu_46_to_wmb_class.csv           # curated mapping
├── load_deconvoluted.py               # readers for ps/py CSVs
├── factorial_ols.py                   # males-only OLS per site per cell type
├── mea_per_celltype.py                # two-track MEA driver
├── snrna_concordance.py               # join Stage 3 output with snRNA gene LFCs
├── confidence.py                      # Stage 5 calibration
├── rollup_wmb.py                      # Stage 6 aggregation
└── run.py                             # orchestrator (Stages 1-6)
```

Charter footnote in `README.md`: "This code path uses proportional redistribution as a proxy. It does not reopen the direct cell-type deconvolution path closed by `docs/foundation/analysis_charter.md`."

## 7. Time budget

| Stage | Estimate |
|---|---|
| 1: Load pre-computed | 0.5 day |
| 2: Males-only factorial OLS at 46 clusters × 2 tracks | 1 day |
| 3: Two-track MEA per cell type per contrast | 1 day |
| 4: snRNA at 46 clusters + join | 1 day |
| 5: Confidence calibration | 0.5 day |
| 6: WMB rollup + mapping curation | 0.5 day |
| Methods write-up | 1 day |
| **Total** | **~5 days** |

## 8. Open risk flags

These do not block kickoff but are worth knowing about:

1. **Prior-findings benchmark.** Whether the chief scientist has specific kinase × cell-type expectations from the legacy iteration that should reproduce. If so, those become an implicit validation target. Worth asking before delivery.
2. **Kinase library coverage variance.** Cell types receiving little signal from the deconvolution may have too few sites with kinase-substrate matches to produce reliable NES — some entries will be `n/a` per kinase library defaults. Honest but needs explanation in the write-up.
3. **`aggexp` per-sample noise inheritance.** The proportional formula uses per-sample, per-cell-type expression weights, so per-cell-type deconvolved signal inherits sample-level scRNA noise. Most pronounced at rare clusters.
4. **Mapping ambiguity.** Some 46→WMB mappings will be subjective (e.g., where does `Erbb4-VIP-inhibitory-neurons` go). The mapping CSV will surface these in a `notes` column rather than hiding them.

## 9. What this plan does not do

- Does not modify the live pipeline or the existing 34-WMB attribution.
- Does not re-run the deconvolution from `aggexp.csv` + bulk. Adopts pre-computed outputs.
- Does not run direct A_obs-based OLS deconvolution. The infeasibility analysis (`docs/deconvolution_infeasibility.md`) remains available as appendix material if asked.
- Does not produce female or full-cohort outputs. Males-only matches the live pipeline default.
- Does not gate or drop clusters by power. All 46 retained; uncertainty surfaces in the confidence column.
