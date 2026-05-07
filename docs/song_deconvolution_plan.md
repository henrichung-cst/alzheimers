# OUT OF DATE — IGNORE

Branch-only plan. The bulk-median inputs it describes were deleted from `data/incytr_collections/song/proteomics/source/` on 2026-05-07 and must be re-pulled via `pixi run ingest-gdrive-shared` before the plan can be executed. Live pipeline scope is in `CLAUDE.md`; the WMB-class spine is now established by the live stoichiometry pipeline, not this decomposition.

---

# Song bulk decomposition → per-cell-type kinase enrichment (CTM-native)

Plan-of-record for the Song bulk → per-cell-type kinase MEA pipeline.
Single-axis: WMB class throughout, fed by a CTM-native proportional
decomposition computed directly from the snRNA-seq h5ad. The legacy
46-cluster + soft-mass-projection plan is preserved for provenance at
[`../archive/song_deconvolution_plan_46cluster.md`](../archive/song_deconvolution_plan_46cluster.md).

## 1. Goal and audience

**Goal.** Per-cell-type kinase enrichment table (NES + FDR per kinase × WMB
class × contrast, ser/thr and tyr tracks) from CTM-native proportional
decomposition of Song bulk phosphoproteomics, with a defensible per-row
reliability indicator drawn from matched snRNA-seq.

**Audience.** Chief scientist. Argument leads with the deliverable; uncertainty
surfaces as per-row evidence columns; the closed direct-deconvolution path is not
reopened.

**Framing.** This is a **proportion-proxy decomposition** with snRNA pseudobulk
as the per-(group, WMB class, gene) prior — distinct from inferring
cell-type-specific effects from `A_obs + bulk` alone, which the analysis
charter closes.

## 2. Cell-type vocabulary

The Allen WMB-class spine (`config.WMB_CLASSES`, 34 classes). Of those, 24
are detectably present in the Song h5ad after the `class_prob ≥ 0.9` and
WMB-mappability filters; the remaining 10 are biological-sampling gaps (no
nuclei from those classes were captured) and appear nowhere in the
decomposition output:

```
15 HY Gnrh1 Glut, 16 HY MM Glut, 21 MB Dopa, 22 MB-HB Sero, 23 P Glut,
24 MY Glut, 25 Pineal Glut, 26 P GABA, 27 MY GABA, 28 CB GABA
```

The live attribution program (`code/kinase_attribution.py`) operates on the
same 34-class spine; the deconvolution branch and the live program share a
single cell-type vocabulary end-to-end.

## 3. Inputs

| File | Role |
|---|---|
| `data/incytr_collections/song/transcriptomics/170_gex_celltypes_00.h5ad` | snRNA-seq with CTM `class_name` per nucleus (63,695 nuclei × 30,567 genes) |
| `data/incytr_collections/song/proteomics/source/imac_median.csv` | Ser/Thr per-(site, group) bulk medians (24 groups) |
| `data/incytr_collections/song/proteomics/source/py_median.csv` | Tyr per-(site, group) bulk medians (24 groups) |
| `data/incytr_collections/song/proteomics/source/pr_median.csv` | Total proteome per-(gene, group) bulk medians (24 groups; for stoichiometry, optional downstream) |
| `data/incytr_collections/song/proteomics/source/yuyu_samplekey.csv` | MS\_ID ↔ SCRNA\_ID/Group bridge |
| `data/external/allen_abc/wmb_class_manifest.csv` | CTM `class_name` → prefixed WMB class label |

snRNA cross-check oracle: `outputs/reports/snrna_integration/song_concordance.csv`
and `song_expression_specificity.csv`, both produced by
`code/snrna_integration.py` on the same WMB-class axis.

## 4. Stage-by-stage pipeline

### Stage 0 — CTM-native proportional decomposition

`code/deconvolution/build_wmb_decomposition.py`. For each gene, group, and
WMB class:

```
deconv[gene, group, w] =
    bulk_median[gene, group]
  · (raw_count[gene, group, w] / Σ_w' raw_count[gene, group, w'])
  · size_factor[group, w]
```

with `size_factor[group, w] = Σ_w' n_cells[group, w'] / n_cells[group, w]`.
Raw counts are aggregated directly from the h5ad on `(group, wmb_class)`.
Genes with zero raw count in any (group, w) cell receive a small floor
(`min_nonzero / 10000`) before the share normalization (matches Yuyu's
original `protein-ms-by-cell-type.py:34-44, 87-92`).

Outputs: `outputs/reports/deconvolution/wmb_decomposition/{ps,py,pr}_wmb_decomposition.csv`
plus `wmb_class_size.csv` (WMB-class × group nucleus count matrix used by
the Stage 5 `n_cells_min` floor).

### Stage 1 — Load decomposition tracks

`load_track("st" | "py")` returns per-site metadata + a (n_sites × (n_groups
× n_classes)) matrix with a tidy MultiIndex of `(sample, wmb_class)`.

### Stage 2 — Males-only factorial OLS per WMB class

`run_per_animal_track`. For each track and each WMB class, expand the
per-(group, w, site) decomposition to per-animal via the
`phospho_group_id` column in `outputs/reports/data_ingest/sample_mapping.csv`,
multiply by per-animal bulk phospho intensity, and fit the same factorial
OLS used in the live pipeline (10 parameters: const, App, Tau, Int,
time\_4mo, time\_6mo, App×time\_4, App×time\_6, Tau×time\_4, Tau×time\_6).
Extract per-site β + p-value for the 9 contrasts.

Sample filtering: males-only after the live outlier-exclusion list.

### Stage 3 — Kinase MEA per WMB class, per track

For each WMB class × contrast × track:

- **Ser/thr track**: rank phosphosites from Stage 2 by β; run kinase library
  MEA against the Ser/Thr substrate set.
- **Tyr track**: rank by β; run against the tyrosine substrate set.

Output: NES + FDR per kinase × WMB class × contrast × track.
Significance threshold: FDR < `DECON_FDR_THRESH` (default 0.25).

### Stage 4 — snRNA cross-check (kinase gene LFC concordance)

For each row of the Stage 3 output, look up the kinase's own gene LFC in
`song_concordance.csv` at the same WMB class and contrast. Annotate three
columns:

- `kinase_gene_LFC_snRNA`
- `kinase_gene_FDR_snRNA`
- `direction_match` — `match` / `opposite` / `flat` / `n/a`

Cohort concordance: per (WMB class, contrast) stratum, binomial test on
`sign(NES) == sign(kinase_gene_LFC_snRNA)` over kinase rows in that stratum,
BH across strata.

### Stage 5 — Attach per-row evidence

Per-row evidence columns are joined onto the Stage 4 MEA table — no
categorical confidence label is assigned. Downstream readers gate on the
underlying columns directly:

- `FDR` (bulk MEA, threshold `DECON_FDR_THRESH`)
- `n_cells_min` (smallest group nucleus count for this (wmb_class, contrast); compare against `MIN_CELLS_PER_GROUP`)
- `cohort_concordant` / `frac_match` / `cohort_fdr` (stratum-level binomial)
- `expressed` (kinase mRNA above `EXPR_PRESENCE_FLOOR` in this WMB class)
- `kinase_gene_LFC_snRNA` / `direction_match` (per-row sign agreement with snRNA)

Thresholds in `paths.py`:

- `MIN_CELLS_PER_GROUP = 20`
- `DECON_FDR_THRESH = 0.25`
- `COHORT_FDR_THRESH` (calibrated via `cohort_concordance_audit.py`)
- `EXPR_PRESENCE_FLOOR` (same audit)

Stoichiometry is **not** applied per cell type: under this proportional
decomposition the share/size_factor terms are gene-level (not phospho-vs-
protein-specific), so `log2(ps_decomp[w]) − log2(pr_decomp[w])` reduces to
bulk stoichiometry and the wmb_class axis cancels. Parent-protein
confounding is handled in the live pipeline, not here.

## 5. What this plan supersedes

The 46-cluster predecessor plan is preserved at
[`../archive/song_deconvolution_plan_46cluster.md`](../archive/song_deconvolution_plan_46cluster.md).
It built decomposition outputs at Yuyu's hand-clustered 46-name axis and
projected onto the WMB axis post hoc via a soft-mass crosswalk derived from
CTM. Three problems retired by this plan:

1. **No defensible reason to keep Yuyu's 46-cluster axis.** It is not NNLS
   — `protein-ms-by-cell-type.py` is a proportional redistribution. Re-running
   the formula on the 24 CTM-reachable WMB classes is mechanically identical
   and removes a layer of interpretation.
2. **The hand crosswalk has known errors.** `Erbb4-VIP-inhibitory-neurons`
   maps to `06 CTX-CGE GABA` while CTM places 99.9% in `05 OB-IMN GABA`;
   `Ptprz1-protoplasmic-astrocytes` maps to `30 Astro-Epen` while CTM places
   99.8% in `32 OEC`. Soft-mass projection overrides these, so the hand
   layer cannot be claimed as a defense.
3. **Two cell-type axes upstream and downstream were confusing.** Single
   axis end-to-end eliminates the mismatch.

## 6. Out of scope

- Re-running `snrna_integration.py --pseudobulk`. Stage 0 reads the h5ad
  directly; pseudobulk artifacts on disk are independent.
- The 10 absent WMB classes — biological sampling gap, not addressed here.
- Live pipeline (`kinase_attribution.py`, `attribution_recovery.py`) —
  already on WMB axis; no changes.
- Incytr integration — separate axis, separate plan.
