# Plan: rewrite human "Cell-type specificity" → "Attribution" (mirror of mouse)

**Goal.** Restructure the human per-kinase Cell-type specificity sub-tab so it
mirrors the mouse Attribution sub-tab in title, column order, row layout, and
formatting. Only the data sources differ — the UX surface must be identical
so a viewer of either tab sees the same information shape.

## Current state (the divergence)

| Aspect | Mouse Attribution | Human (currently) |
|---|---|---|
| Tab title | "Attribution" | "Cell-type specificity" |
| Table | Single, with Conf, WMB enrich/tier, log2 expr, % cells, SEA-AD LFC, Song LFC, Score, Decomp NES, Decomp FDR, vs Bulk | Two separate mini-tables (SEA-AD MTG + HBCA), Rank / Cell type / log₂-spec only |
| Rows | All celltypes (31 levy_t5 clusters) per (kinase, contrast) | Top-N (8) celltypes per kinase, no contrast filter |
| Contrast-varying signal | Decomp NES per cluster, vs Bulk badge | *(none — tab is static per kinase)* |

## Target state (mirror)

| Column (group) | Mouse source | Human source |
|---|---|---|
| Cell type (id) | levy_t5 cluster name | SEA-AD supertype name OR HBCA supercluster name |
| Reference (id) | *(implicit: levy_t5)* | "SEA-AD MTG" or "HBCA" badge — new column to distinguish refs |
| Conf (attr) | cross_rank | tier derived from specificity score (≥10× / ≥5× / ≥2× / ≥1× of uniform; matches mouse WMB-tier logic) |
| Specificity (attr) | wmb_specificity (share of total log2 expression) | log₂-ratio (cell-type mean / brain-wide mean) — already in payload |
| Tier (attr) | wmb_tier | same bucketing applied to human score |
| log2 expr (attr) | wmb_mean_log2_expression | **needs payload extension** — pull from SEA-AD / HBCA expression matrices |
| % cells (attr) | wmb_fraction_cells_expressing | **needs payload extension** — fraction of cells with non-zero counts |
| SEA-AD LFC (attr) | sea_ad_lfc (median LFC over SEA-AD supertypes mapped to cluster) | direct per-supertype SEA-AD LFC (no crosswalk needed — rows ARE SEA-AD supertypes); empty for HBCA rows |
| Song LFC (attr) | song_lfc | *(N/A — Song is mouse only; column hidden on human tab)* |
| Score (attr) | combined_score | combined: effective specificity × (0.5 + tier weight) — same formula |
| Kinase NES @ contrast (contrast-varying) | decomp_nes | **per-donor kinase NES** from `perdonor_index` at the selected donor — broadcast across all rows of that kinase (constant per row group, varies as donor changes) |
| Kinase FDR @ contrast | decomp_fdr | per-donor kinase FDR from `perdonor_index` |
| vs Bulk | bulk_match (sign agreement with bulk MEA) | sign agreement between selected donor's kinase NES and the cohort-aggregated NES |

## Work breakdown

### Payload extension (`alz/human_reference_expression.py` + `alz/human_celltype_attribution.py`)

Currently the payload only carries `celltype`, `score`, `rank` per kinase per
reference. The mirror needs three additional per-(kinase, celltype) fields:

- `mean_log2_expression` — raw mean log2 expression of the kinase gene in that
  cell type (already computed upstream when specificity is calculated; just
  needs to be carried through to the payload).
- `fraction_cells_expressing` — fraction of cells in that cell type with
  non-zero counts. Available from the same upstream expression matrix.
- `sea_ad_lfc` (SEA-AD reference only) — per-supertype LFC from
  `effect_sizes.h5ad`. Already loaded in `kinase_attribute.py` for the
  mouse-side join; reuse that loader.

Extend `build_celltype_specificity_payload()` to include these fields in each
`top_n_by_kinase` entry. Bump `top_n` default if the mirror should show
*all* celltypes (mouse shows 31, full spine) rather than just top-N — the
mouse table is not truncated; the human one shouldn't be either. Recommend
emitting the full ranked list, not just top-8.

### Per-donor kinase NES (already available)

`PAYLOAD.human.perdonor_index` already carries per-donor kinase NES/FDR. No
new payload work needed — the JS just needs to join the currently-selected
donor's value onto every row of the Attribution table.

### Viewer JS rewrite (`alz/viewer/template/js/tabs/kinase_human.js`)

1. Rename the sub-tab from "Cell-type specificity" → "Attribution"
   (button label at line 61; route handler at line 462; helpers prefixed
   `_khRenderAttribution` instead of `_khRenderCelltypeSpecificity`).
2. Replace `_khRenderCelltypeSpecificity` with `_khRenderAttribution` that
   renders a single combined table with the column set from the mirror table
   above. Use the same DOM scaffolding the mouse Attribution tab uses
   (`audit-panel`, `data-table`, tier chip classes, color-coded numeric cells).
3. Reuse the mouse Attribution tab's tier-chip render logic (extract to a
   shared helper if needed — `_attrTierChip`).
4. Join per-donor kinase NES from `perdonor_index` onto every row at render
   time; re-render the table when the donor selector fires.
5. Drop the "phase-2 data unavailable" fallback only if the underlying
   payload is missing — keep the same gating logic the current code has.

## Out of scope

- The mouse-only Song LFC column. Human has no Song equivalent.
- Confidence tier "very high" upgrade based on decomp agreement. Human has
  no decomp layer; the highest tier on human is "high" from attribution-only
  evidence.
- Sortable columns beyond what the mouse Attribution tab already supports.

## Execution order

1. Extend `build_celltype_specificity_payload()` and its upstream expression
   helper to carry `mean_log2_expression`, `fraction_cells_expressing`, and
   `sea_ad_lfc` per (kinase, celltype) row. Drop the top-N truncation OR
   raise it well above the cell-type count.
2. Confirm `perdonor_index` schema in the existing payload (no payload change
   anticipated).
3. Rewrite `_khRenderCelltypeSpecificity` → `_khRenderAttribution` mirroring
   mouse `_renderAttributionVerdict`.
4. Rebuild the viewer (`pixi run python alz/build_unified_viewer.py`).
5. Hard-refresh and confirm the human Attribution tab is structurally
   indistinguishable from the mouse Attribution tab.
