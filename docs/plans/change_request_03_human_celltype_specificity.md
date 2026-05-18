# Change request 04 — Human cell-type specificity reference (SEA-AD MTG + Allen HBCA)

## Goal

Give the Human (Mukesh) tab a cell-type specificity dimension matching
what WMB provides on the mouse side. For each kinase the viewer should
be able to say "this kinase's transcript is specific to <cell type X>
in human brain reference data," parallel to the existing WMB
specificity column on the mouse tab.

## Locked decisions

- Two references, side-by-side:
  - **SEA-AD MTG snRNA-seq** — cortical (MTG only). Same supertype
    axis used for `seaad_human_agreement`. Pulled fresh because the
    h5ad on disk today (`effect_sizes.h5ad`) is *effect sizes*, not
    log-expression — we need a per-supertype log-mean expression file.
  - **Allen Human Brain Cell Atlas (HBCA)** — whole brain coverage.
    The closest human analog to WMB. Different taxonomy from SEA-AD,
    so the two references are shown as parallel columns, not merged.
- Specificity metric: same construction as WMB —
  `log2(<celltype mean> / <brain-wide mean>)` per gene, ranked into a
  per-celltype quantile. Keeps the viewer's cross-reference language
  consistent.

## Surface map

- `alz/atlas_reference.py` — today downloads SEA-AD effect-size h5ads
  + WMB 10Xv3 expression matrices. Will gain two new fetchers.
- `alz/wmb_expression.py` — reference implementation of the
  specificity recipe (per-class expression → kinase matrix → JSON for
  viewer). Use as the template.
- `alz/seaad_human_agreement.py` — existing human-side helper; will
  become one of three sibling modules (`*_specificity.py` is new).
- `alz/build_unified_viewer.py` — packs `PAYLOAD.human`. New blocks
  needed.
- `alz/viewer/template/js/tabs/kinase_human.js` — adds a "Cell type
  specificity" sub-panel in the kinase detail view.
- `alz/config.py` — add paths and `HUMAN_REFERENCE_CLASSES` constants.

## Data acquisition

### SEA-AD MTG expression

- File: SEA-AD MTG snRNA-seq donor-level h5ad (publicly hosted on the
  same AWS bucket as the effect-size files;
  `s3://sea-ad-single-cell-profiling/MTG/RNAseq/` — confirm exact
  object name). Per-cell `X` + obs `Supertype`.
- Pull the per-supertype mean expression server-side (anndata `to_df`
  groupby) — full single-cell file is ~50 GB; we want
  `genes × supertypes` log-mean only.
- Output: `data/external/sea_ad/expression_by_supertype.csv`
  (rows=gene, cols=139 supertypes).
- `atlas_reference.py --sea-ad-expression` flag.

### Allen HBCA

- File: Allen Brain Cell Atlas Human (HBCA) — released via the same
  `abc_atlas_access` Python package we use for WMB. Inspect
  `abc_atlas_access` to find the Human equivalent matrix list (likely
  `HBCA-10Xv3` or similar; verify before downloading).
- Mirror the WMB ingest: stream per-region log-expression and
  groupby `class` (whatever HBCA's top-level taxonomy is called).
- Output: `data/external/allen_hbca/expression_by_class.csv`.
- Cache to the same allen_abc-style `.parquet` layout
  (`data/external/allen_hbca/`).
- `atlas_reference.py --hbca-download` flag.

## New modules

### `alz/human_reference_expression.py`

- Mirror of `alz/wmb_expression.py` but consumes the two human
  references. Reused recipe:
  1. Filter to kinase + phosphatase gene list (already exported by
     `atlas_reference.py`).
  2. For each celltype c, compute log2(mean(c) / mean(brain)).
  3. Write `seaad_kinase_specificity.csv` (kinase × 139 supertypes)
     and `hbca_kinase_specificity.csv` (kinase × N classes).
- `--ref {seaad, hbca, both}` flag.

### `alz/human_celltype_attribution.py`

- Per kinase, derive a "top-N specific cell types" list from each
  reference. Output:
  `outputs/reports/kinase_attribution_human/celltype_specificity.csv`
  with columns:
  `kinase, reference, celltype, specificity_score, rank`.

## Viewer payload (`alz/build_unified_viewer.py`)

- New payload block:
  ```js
  PAYLOAD.human.celltype_specificity = {
    references: ["seaad_mtg", "allen_hbca"],
    seaad_mtg: { celltypes: [...], by_kinase: { kid → [score per celltype] } },
    allen_hbca: { celltypes: [...], by_kinase: { kid → [score per celltype] } },
  }
  ```
- Keep the per-kinase top-N list as a precomputed convenience for
  the table-glance view.

## Viewer JS (`alz/viewer/template/js/tabs/kinase_human.js`)

- New sub-tab in the kinase detail panel: **"Cell-type specificity"**.
  Two stacked mini-tables (SEA-AD MTG | Allen HBCA), each showing the
  top 8 cell types by specificity for the selected kinase.
- Add a `top_celltype_seaad` and `top_celltype_hbca` column to the
  main kinase table (collapsible), sortable.

## Open questions

- **Hippocampus**: SEA-AD MTG is cortex-only. If hippocampal kinase
  specificity matters for the AD story, we either pull in
  Allen Hippocampus SMART-seq separately or rely on HBCA's
  hippocampal classes (assuming HBCA covers HPF — needs confirmation).
  Will add HPF as a follow-up if HBCA doesn't cover it.
- **Matching gene symbols**: SEA-AD obs uses gene symbols; HBCA may
  use Ensembl IDs. Add a mapping step in `human_reference_expression.py`
  that normalizes to HGNC symbols before joining to the kinase list.

## Runtime / cost

- SEA-AD MTG h5ad download: ~50 GB raw. Compute per-supertype mean
  with `chunk_size` streaming (mirrors `wmb_expression.py --proteome`).
  ~1 h on the lab box.
- HBCA download: comparable in size to WMB (~95 GB). Multi-hour. Use
  `abc_atlas_access` cache so this is one-time.
- Specificity compute: minutes once expression is in place.

## Risks

- Allen HBCA may not have the same `class` granularity as WMB. The
  viewer column header text needs to be reference-specific (no
  shared "celltype" axis assumption).
- Disk: HBCA alone needs ~100 GB. The atlas cache is already
  compressed (`MANIFEST.json`); add HBCA to the compression script
  before merging.
- SEA-AD MTG donor cohort overlap with the Mukesh / NBB cohort is
  unclear. Worth flagging in the viewer copy: SEA-AD MTG is the
  reference *population*, not the AD donor population — specificity
  is a transcript-level prior, not a co-measurement.

## Independence

This change is fully independent of changes 01, 02, and 04.
Different data, different files. Touches the human tab payload
alongside change 01 — coordinate at payload-block merge.
