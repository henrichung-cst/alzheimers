# Song bulk decomposition → per-cell-type kinase enrichment

Chief-scientist deliverable. Generates a per-(WMB-class × kinase) NES + FDR
table from a CTM-native proportional decomposition of Song bulk
phosphoproteomics (Ser/Thr and Tyr tracks), with a transcript-level
reliability column drawn from matched snRNA-seq.

## Charter footnote

This code path uses **proportional redistribution as a proxy** for cell-type-
specific decomposition (with the snRNA-seq pseudobulk acting as the
per-(group, WMB-class, gene) prior). It does **not** reopen the direct
cell-type deconvolution path closed by
[`docs/foundation/analysis_charter.md`](../../docs/foundation/analysis_charter.md).
The full design rationale, audience framing, confidence model, and explicit
non-goals are in [`docs/song_deconvolution_plan.md`](../../docs/song_deconvolution_plan.md).

The pipeline does **not** modify the live attribution program (34 WMB
classes) and is **not** wired into `pixi run live` or `pixi run dual`.

## Module layout

```
alz/deconvolution/
├── README.md                          # this file
├── paths.py                           # input/output paths + factorial coding
├── build_wmb_decomposition.py         # Stage 0: CTM-native proportional decomposition
│                                      #          (h5ad → ps/py/pr_wmb_decomposition.csv)
├── load_deconvoluted.py               # Stage 1: readers for *_wmb_decomposition.csv
├── factorial_ols.py                   # Stage 2: 10-parameter design + contrast logic
├── per_animal_extension.py            # Stage 2: per-animal proxy + OLS driver
├── mea_per_celltype.py                # Stage 3: two-track MEA driver
├── snrna_concordance.py               # Stage 4: snRNA gene LFC join
├── cohort_concordance.py              # Stage 4: cohort binomial + presence
├── confidence.py                      # Stage 5: attach per-row evidence colion
├── cohort_concordance_audit.py        # one-shot threshold calibration helper
├── run_per_animal.py                  # orchestrator (per-animal grain)
└── _archive/                          # retired 46-cluster artifacts (provenance)
```

## Inputs

All consumed read-only. The four bulk-median + samplekey files under
`data/datasets/song/proteomics/source/` were deleted on
2026-05-07; re-pull via `pixi run ingest-gdrive-shared` and copy from
`data/raw/external/gdrive_shared/integrations/yuyu01/documentation/incytr/deconvolution/`
into that directory before running this branch.

| File | Source |
|---|---|
| `data/datasets/song/transcriptomics/170_gex_celltypes_00.h5ad` | snRNA-seq with Allen Cell Type Mapper `class_name` per nucleus |
| `data/datasets/song/proteomics/source/imac_median.csv` | Ser/Thr per-(site, group) bulk medians (re-pull) |
| `data/datasets/song/proteomics/source/py_median.csv` | Tyr per-(site, group) bulk medians (re-pull) |
| `data/datasets/song/proteomics/source/pr_median.csv` | Total proteome per-(gene, group) bulk medians (re-pull) |
| `data/datasets/song/proteomics/source/yuyu_samplekey.csv` | MS\_ID ↔ SCRNA\_ID/Group mapping (re-pull) |
| `outputs/reports/kinase_attribution/raw_phospho_normalized.csv` (+ `_pY`) | live IRS-normalized per-animal phospho |
| `outputs/reports/data_ingest/sample_mapping.csv` (+ `_exclusions`) | live per-animal sample-to-channel mapping |
| `outputs/reports/snrna_integration/song_concordance.csv` | live snRNA pipeline (WMB-class) |
| `outputs/reports/snrna_integration/song_expression_specificity.csv` | live snRNA pipeline (per-(class, gene) mean expression) |
| `data/datasets/song/analysis_cache/kinase_to_gene_mapping.csv` | live kinase-gene cache |

The cell-type axis throughout is the WMB-class spine
(`config.WMB_CLASSES`, 34 total). Of those, 24 are detectably present in the
Song h5ad after the class-prob and mappability filter; the remaining 10 are
biological-sampling gaps (no nuclei from those classes were captured) and
appear nowhere in the decomposition output. The snRNA confidence/animal-count
gates that further filter `pseudobulk_cpm.csv` for differential-expression
analysis are not applied here — Stage 0 uses the full 24-class coverage so
the proportional shares are computed against the complete cell-type universe.

## Outputs

```
outputs/reports/deconvolution/wmb_decomposition/
├── ps_wmb_decomposition.csv          # Ser/Thr: site × (group, wmb_class)
├── py_wmb_decomposition.csv          # Tyr:    site × (group, wmb_class)
├── pr_wmb_decomposition.csv          # Total proteome: gene × (group, wmb_class)
└── wmb_class_size.csv                # WMB-class × group nucleus counts

outputs/reports/deconvolution/per_animal/
├── site_level_ols.parquet            # per-site β/SE/p (wmb_class × contrast × track)
├── kinase_enrichment_raw.csv         # raw MEA before snRNA / evidence join
├── kinase_enrichment_wmb.csv         # PRIMARY: per-row evidence table on WMB axis
├── cohort_concordance.csv            # per (wmb_class, contrast) binomial test
├── group_class_counts.csv            # snapshot of wmb_class_size used by Stage 5
└── summary.json                      # row counts, deconv-sig counts, runtime
```

## Running

```bash
# Stage 0: build the WMB-class decomposition (one-time per data refresh)
python alz/deconvolution/build_wmb_decomposition.py

# Full per-animal pipeline (24 WMB classes × 9 contrasts × 2 tracks × 1000 perms)
python -m code.deconvolution.run_per_animal --run

# Quick smoke test (subset of WMB classes, 200 permutations, ser/thr only)
python -m code.deconvolution.run_per_animal --run \
    --cell-types "30 Astro-Epen" "31 OPC-Oligo" "34 Immune" \
    --tracks st --permutations 200

# Stop after OLS (no MEA)
python -m code.deconvolution.run_per_animal --ols-only

# Re-run Stages 4–5 only against existing kinase_enrichment_raw.csv
python -m code.deconvolution.run_per_animal --relabel-only

# Inspect cached outputs
python -m code.deconvolution.run_per_animal --summary
```

## Decomposition formula

For each gene, group, and WMB class:

```
deconv[gene, group, w] =
    bulk_median[gene, group]
  · (raw_count[gene, group, w] / Σ_w' raw_count[gene, group, w'])
  · size_factor[group, w]
```

with `size_factor[group, w] = Σ_w' n_cells[group, w'] / n_cells[group, w]`.
Raw counts come from summing the Song h5ad on `(group, wmb_class)` after
the `class_prob ≥ SONG_MIN_SUBCLASS_PROB` and WMB-mappability filters. Genes
with zero raw count in any (group, w) cell receive a small floor before the
share normalization (matches Yuyu's original
`protein-ms-by-cell-type.py:34-44, 87-92`).

The size_factor cancels in `compute_site_fractions` (which renormalizes by
`proportion[w] = n_cells[group, w] / Σ_w' n_cells[group, w']`), so the
formula is post-proportion mass-conserving:
`Σ_w (deconv[w] · proportion[w]) ≈ bulk_median`.

## Per-row evidence columns

The primary table emits per-(kinase × wmb_class × contrast × track) row:

| Column | Meaning |
|---|---|
| `FDR` | Bulk MEA FDR for this stratum |
| `n_cells_min` | Smallest group nucleus count for this (wmb_class, contrast); compare against `MIN_CELLS_PER_GROUP` |
| `kinase_gene_LFC_snRNA` / `kinase_gene_FDR_snRNA` / `direction_match` | snRNA gene LFC at the same (wmb_class, contrast); per-row FDR is saturated at n≈15 males, so use `direction_match` rather than the FDR |
| `cohort_concordant` / `frac_match` / `cohort_fdr` | Stratum-level binomial: of the bulk-significant rows in this (wmb_class, contrast), what fraction sign-match snRNA, and is that above chance |
| `expressed` | True iff kinase mRNA `mean_expression` ≥ `EXPR_PRESENCE_FLOOR` in this WMB class |

No categorical confidence label is assigned. Downstream readers gate on
these columns directly. Thresholds live in `paths.py`:

- `MIN_CELLS_PER_GROUP` (default 20)
- `DECON_FDR_THRESH` (default 0.25)
- `EXPR_PRESENCE_FLOOR` (calibrated via `cohort_concordance_audit.py`)
- `COHORT_FDR_THRESH` (calibrated via `cohort_concordance_audit.py`)

## Stoichiometry note

The branch ranks **fraction-weighted raw phospho**, not stoichiometry. A
naive per-cell-type stoichiometry would compute
`log2(ps_decomp[w]) − log2(pr_decomp[w])`, but since both decompositions
share the same `share[gene, group, w] · size_factor[group, w]` factor,
the wmb_class axis cancels and the result reduces to bulk stoichiometry —
every cell type produces identical β. Cell-type-specific
stoichiometry would require cell-type-specific shares (i.e. a different
prior for phospho vs. protein) and is not in scope here. Parent-protein
confounding is addressed in the live pipeline, not in this branch.
