# Substrate Conservation viewer tab

A dedicated unified-viewer tab for the D1 cross-cohort substrate-motif comparison
(human NBB/Mukesh AD vs 5xFAD mouse), modeled after the Kinase Explorer tab.
Replaces the standalone HTML report as the durable surface for this analysis.

## Prerequisite (analysis code, not viewer)

The significance metric the tab ranks on — the label-permutation null enrichment
(`enrich_z`, `p_emp`, `enrich`) — is **not** in the production C5 output today. It
exists only in the scratchpad prototype `null_enrich.py`. Before any viewer work:

- Fold the label-permutation enrichment into `alz/cross_reference/c5_mukesh_5xfad.py`
  as a first-class computation. Statistic `S(A,B) = Σ_{a∈A} max_{b∈B} sim(a,b)`
  (similarity-weighted, threshold-free, BLOSUM62 center-aligned + central-type gated);
  per kinase×context: `observed = S(human_K, mouse_K)`, `null = {S(human_K, mouse_j) : j≠K}`,
  `z = (obs−mean)/sd`, `p_emp = (1+#{null≥obs})/(1+n_null)`, `enrich = obs/mean`.
- Emit `enrich_z`, `p_emp`, `enrich`, `n_null` as columns of `kinase_summary.csv`.
- **Drop `jaccard`** from `kinase_summary.csv` (anti-shim — the pivot replaces it).
  Keep the descriptive raw counts `n_shared`, `n_human_only`, `n_mouse_only`.
- Precompute a per-(kinase, context) BLOSUM-similarity **histogram** (fixed ~10 bins
  over [0.5, 1.0]) from the shared pairs and emit it (e.g.
  `similarity_histogram_<context>.csv` or a `hist_bins` column) so the viewer detail
  histogram needs no per-pair shard fetch to draw.

## Decisions (locked)

- **Direction columns:** two glyph columns = kinase NES direction in human vs in mouse.
  Substrate direction-agreement (`direction_agree_frac`) is a separate column.
- **Row granularity:** 60-kinase pool rows (`overlap_AD8_sus_clean`), one per kinase,
  with a 2×4 (tissue × age) enrichment mini-heatmap — analog of the Kinase Explorer
  NES profile. Detail pane drills into one selected context.
- **Track:** ST only (matches current C5 scope). pY is out of scope for this tab.
- **Jaccard:** dropped as headline scalar; raw counts retained.
- **Placement:** cross-cohort — surfaced alongside the crosstable, not the mouse-only
  Explorer group. Requires both `human` and `supporting_5xfad` present.

## Data flow

### Inline (small) — `PAYLOAD.substrate_compare`, top-level key
New builder in `alz/viewer/cohorts/` (cross-reference slice). Columnar, per
(kinase × context):
- `kinase_id[]`, `context[]` (tissue_age)
- `enrich_z[]`, `p_emp[]`, `enrich[]`
- `n_shared[]`, `n_human_only[]`, `n_mouse_only[]`
- `direction_agree_frac[]`, `direction_corr[]`
- `human_dir[]`, `mouse_dir[]` (kinase NES sign per cohort)
- `sim_hist[]` — precomputed histogram bin counts per (kinase, context)

Add `substrate_compare` to `TOP_LEVEL_ORDER` in `alz/viewer/shared/compose.py`.
Add a `substrate_compare` capability flag in `build_unified_viewer.py` meta.

### Lazy shards — full per-pair substrate lists
`kinase_pairs_<context>.csv` total ~19 MB — too large to inline. Follow the existing
`EdgeSliceContribution` lazy-shard pattern (as `decomp_ols` / `song_concordance` do):
per-kinase shards under `edge_slices/substrate_pairs/`, fetched on kinase selection to
fill the detail table. Shard columns: `gene_a, site_a, motif_a, gene_b, site_b,
motif_b, similarity, match_class, direction_a, direction_b, direction_agree,
support_a, support_b, context`.

## Viewer wiring

1. **TAB_MANIFEST** entry in `alz/viewer/template/js/02_ui_chrome.js`:
   `group: "reference"` (cross-cohort), `label: "Substrate Conservation"`,
   `modes:` whichever surface both cohorts (match crosstable),
   `requires: [{type:"payload", key:"substrate_compare", ...}]`,
   `wire: () => wireSubstrateCompare()`, `render: () => { renderSubstrateCompare(); ... }`,
   `rerenderOn: { selection: ["kinase"] }`.
2. **Panel** `<div id="tab-substrate" class="tab-panel">` in `alz/viewer/template/body.html`.
3. **New JS tab** `alz/viewer/template/js/tabs/substrate_compare.js`:
   - Row model built from `PAYLOAD.substrate_compare` (aggregate to 60 rows; per-kinase
     the 8 contexts fold into a `_z[]`/`_p[]` array parallel to a fixed context order).
   - `renderSubstrateCompare()` — filter + sort + string-concat table.
   - Mini-heatmap helper `_renderEnrichProfile(r)` — 2×4 grid, cell color scaled by
     `enrich_z`, sig cells (p_emp<0.05) outlined. Mirror `_renderNesProfile`.
   - Detail `renderSubstrateDetail(kid)`: header + context `<select>` + BLOSUM histogram
     (new `_simHistogram(bins)` canvas/SVG helper — none exists today) + `AuditTable`
     over the fetched per-pair shard (search/sort/pagination/CSV for free).
   - Filter store mirrors `KinaseFilter` (search, disease/timepoint scope, sort).

### Master table columns
Kinase (+ST/Y) · Gene · Family · Human dir (▲/▼) · Mouse dir (▲/▼) ·
Overlap significance (2×4 enrich_z mini-heatmap) · Peak z (sortable) ·
Substrate dir agreement (fraction pill) · n shared

## Honest-surfacing notes
- Substrate direction agreement reads ~0.5 for most kinases (uncoupled direction) —
  surface the number, do not hide it. It is a genuine finding.
- Coverage limitation (the two phosphoproteomes share ~12–25% of detected motifs) is
  why the null-referenced enrichment replaces raw overlap; note it in the tab header
  copy, framed standalone (no reference to a prior metric).
