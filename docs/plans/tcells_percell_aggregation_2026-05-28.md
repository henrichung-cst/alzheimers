# T-cell substrate: aggregate by ProjecTILs state, not Seurat cluster

**Date:** 2026-05-28
**Status:** approved, in-flight
**Supersedes:** the post-recluster cluster-annotate path (drops 44.5% of donor1) and
the recluster pivot (`~/.claude/plans/joyful-finding-peacock.md`).

## Problem

ProjecTILs per-cell projection resolves **87.2%** of donor1 (median conf 0.76) and
92.8% of donor2. But the downstream substrate is keyed on `seurat_clusters`, so we
have to collapse each Seurat cluster's per-cell calls into one label. When a
cluster's per-cell calls split across states (e.g. donor1 c0 at res=2.0 is 47%
NaiveLike + 22% Treg + 18% CTL_Exh + 13% other — all CD4-pure), there is no honest
single-state label. The two-tier rule drops the cluster as `unresolved`, taking
2238 valid per-cell labels with it.

The 44.5% drop is a **partition mismatch**, not a labeling failure. Seurat
clusters group cells by PCA similarity. ProjecTILs assigns state by projection
onto a curated reference manifold. The two partitions don't have to agree, and
re-clustering at res=2.0 confirms they don't (the dominant CD4 sub-cluster stays
mixed at any reasonable resolution).

## Decision

**Stop using `seurat_clusters` as the aggregation key. Aggregate by the per-cell
ProjecTILs `functional.cluster` directly.** The "cluster" in the substrate becomes
the reference state (e.g. `CD8Tex`, `CD4Th17`, `Treg`), not the Seurat partition.
Cells without a ProjecTILs call (~13% donor1, ~7% donor2 — the `none`-gate and
doublet fractions) are dropped honestly; everything else flows through.

This is an anti-shim pivot. The cluster-keyed path is gone — no flag, no
fallback. The re-clustering work (`tcells_recluster.R`) is also gone: there is
no value in re-partitioning Seurat clusters that the pipeline no longer reads.

## Files

### Delete (anti-shim)
- `alz/ingest/tcells_annotate_clusters.py` — cluster→label collapse no longer exists.
- `alz/ingest/tcells_recluster.R`
- `alz/runners/supporting/tcells_recluster.sh`
- `alz/runners/supporting/tcells_recluster_pipeline.sh`
- pixi tasks: `tcells-recluster`, `tcells-annotate`
- on-disk: `data/derived/tcells/<donor>/Tcells.reclustered.rds`, `recluster_diagnostics.json`
- on-disk: `cluster_annotations.csv`, `annotation_audit.json`

### Rewrite: `alz/ingest/tcells_scrna_extract.R`
- Read raw RDS (`data/datasets/tcells/<donor>/scrna/...rds`) — repoint away from reclustered.
- Read `projectils_predictions.csv` and join `functional.cluster` per barcode.
- Sanitize state names via the existing 14-entry `_LABEL_MAP` (CD8.TEX → CD8Tex, CD4.Treg → Treg, etc.) — inlined into the R script. Unknown values raise.
- Drop cells with NA state (no ProjecTILs call). Stamp count emitted.
- Assert alphanumeric-only state names (Incytr constraint).
- Set `obj$state` and `Idents(obj) <- "state"`.
- Group key becomes `<state>__d<day>` (replaces `c<cluster>__d<day>`).
- Emit:
  - `cell_counts.csv` keyed on `(state, day)` instead of `(cluster, day)`
  - `aggexp_data.csv` with `<state>__d<day>` columns
  - `allmarkers.csv` from `FindAllMarkers` with state as ident
  - `extract_manifest.json` lists `states` instead of `clusters`
  - `state_audit.json` (new): per-state cell count, per-day count, dropped count by reason

### Rewrite: `alz/ingest/tcells_projectils_map.R`
- Repoint input from reclustered to raw RDS.
- No other change.

### Rewrite: `alz/ingest/tcells_decompose.py`
- Drop the `cluster_annotations.csv` read.
- Drop the `cluster2label` filter (column headers in `aggexp_data.csv` already carry sanitized state names).
- `_load_counts` reads `state, day, n_cells`.
- Mass identity becomes `Σ_s P_s × N_s/N_total ≈ bulk` (s = state). Same algebra.
- Output column naming `d{day}_{state}` unchanged in form.

### Pipeline order (new)
```
ingest-tcells-scrna           # download raw RDS
tcells-projectils-map         # per-cell projection — MUST run before extract now
tcells-scrna-extract          # state-keyed aggexp/counts/markers
tcells-decompose              # state-keyed substrate
```

### `pixi.toml`
- Remove `tcells-recluster`, `tcells-annotate`.
- Keep `tcells-projectils-map`, `tcells-scrna-extract`, `tcells-decompose`.

## Verification

1. `tcells_decompose.py` mass-identity assertion ≤ 1e-15 across all donor × channel × day.
2. donor1 dropped cells ≤ 13% (just the scGate `none`-gate + doublets — same as per-cell projection rate, no cluster-level loss).
3. donor2 dropped cells ≤ 8% (per-cell rate).
4. State alphanumeric check (Incytr constraint): no `_` or `.` in any state name.
5. donor1 substrate gains: `CD4NaiveLike`-equivalent label (`CD4Naive` singleton) and `Treg` appear as named lineages — both were dropped as `unresolved` in the cluster-keyed path.
6. donor1 spine grows from 4 lineages to ~6–8 (all CD4/CD8 states with >100 cells across the time course).
7. Sanity: each substrate column `d{day}_{state}` has N_cells ≥ small threshold (skip empty states gracefully).

## Known limitations

- States with very few cells per day produce noisy per-(state, day) aggexp columns. We don't gate on a minimum N — that's the caller's responsibility (Incytr already filters on `SigProb > 0.1` and `|PDS| > 0.2`).
- The `none`-gated cells (donor1 12.7%, donor2 7.1%) are not T cells per scGate and are correctly dropped.
- This is symmetric to how the existing mouse pipeline keys substrate on `cluster_labels` (the 31-spine), not raw Leiden IDs. Path A brings the T-cell pipeline into the same shape.
