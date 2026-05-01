# Song bulk deconvolution → per-cell-type kinase enrichment

Chief-scientist deliverable. Generates a per-(cell-type × kinase) NES + FDR
table from the **pre-computed Yuyu 46-cluster proportional decomposition** of
Song bulk phosphoproteomics (Ser/Thr and Tyr tracks), with a transcript-level
reliability column drawn from matched snRNA-seq.

## Charter footnote

This code path uses **proportional redistribution as a proxy** for cell-type-
specific decomposition. It does **not** reopen the direct cell-type
deconvolution path closed by [`docs/foundation/analysis_charter.md`](../../docs/foundation/analysis_charter.md).
The full design rationale, audience framing, confidence model, and explicit
non-goals are in [`docs/song_deconvolution_plan.md`](../../docs/song_deconvolution_plan.md).

The pipeline does **not** modify the live attribution program (34 WMB
classes) and is **not** wired into `pixi run live` or `pixi run dual`.

## Module layout

```
code/deconvolution/
├── README.md                    # this file
├── yuyu_46_to_wmb_class.csv     # curated 46-cluster → 34-WMB mapping
├── paths.py                     # input/output paths + factorial coding
├── load_deconvoluted.py         # Stage 1: readers for ps/py CSVs
├── factorial_ols.py             # Stage 2: males-only OLS per cluster
├── mea_per_celltype.py          # Stage 3: two-track MEA driver
├── snrna_concordance.py         # Stage 4: snRNA gene LFC join
├── confidence.py                # Stage 5: per-row confidence calibration
├── rollup_wmb.py                # Stage 6: aggregate to 34 WMB classes
└── run.py                       # orchestrator
```

## Inputs

All consumed read-only:

| File | Source |
|---|---|
| `ps_yuyu_deconvoluted.csv` (271 MB) | `data/raw/external/gdrive_shared/.../deconvolution_with_new_clusters_20250721/` |
| `py_yuyu_deconvoluted.csv` (19 MB) | same dir |
| `yuyu_clustersize.csv` | same dir (46 clusters × 24 samples) |
| `outputs/reports/snrna_integration/song_concordance.csv` | live snRNA pipeline (34 WMB classes) |
| `data/incytr_collections/song/analysis_cache/kinase_to_gene_mapping.csv` | live kinase-gene cache |

The snRNA cross-check is performed at the **WMB-class level** (the live
`song_concordance.csv` resolution) by joining each 46-cluster row through
the curated mapping. `Unclassified` clusters get NaN snRNA values.

## Outputs

```
outputs/reports/deconvolution/
├── site_level_ols.parquet            # per-site β/SE/p (cluster × contrast × track)
├── kinase_enrichment_raw.csv         # raw MEA before snRNA / confidence join
├── kinase_enrichment_46clusters.csv  # PRIMARY: per-row confidence table
├── kinase_enrichment_wmb_rollup.csv  # SECONDARY: rolled-up to 34 WMB classes
└── summary.json                      # row counts, confidence breakdown, runtime
```

## Running

```bash
# Full pipeline (~46 clusters × 9 contrasts × 2 tracks × 1000 perms)
python -m code.deconvolution.run --run

# Quick smoke test (3 clusters, 200 permutations, ser/thr only)
python -m code.deconvolution.run --run \
    --clusters Astrocytes Microglia Oligodendrocytes \
    --tracks st --permutations 200

# Stop after OLS (no MEA)
python -m code.deconvolution.run --ols-only

# Inspect cached outputs
python -m code.deconvolution.run --summary
```

## Confidence model

Computed deterministically per (kinase × cell_type × contrast × track) row:

| Confidence | Condition |
|---|---|
| **High** | Deconv FDR < 0.25 + snRNA gene FDR < 0.10 + direction match |
| **Moderate** | Deconv FDR < 0.25 + snRNA flat or n/a |
| **Low** | Deconv FDR < 0.25 + snRNA significant in opposite direction |
| **Insufficient** | Cluster has < 20 cells in any sample group of the contrast |
| **NotSig** | Deconv FDR ≥ 0.25 (not a finding) |

Thresholds live in `paths.py` and are tunable. Defaults are set in advance.
