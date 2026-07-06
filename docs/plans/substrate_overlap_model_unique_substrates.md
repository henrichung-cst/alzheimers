# Substrate-overlap redesign — surface model-unique substrates (replace the motif-z)

## Goal

Replace the Substrate Conservation tab's headline metric. The current metric —
BLOSUM62 motif similarity aggregated into a **label-permutation null-enrichment z**
(`compute_null_enrichment`) — does not answer the question we want, and its results
are near-degenerate (median z ≈ 2.2, **60/60 kinases "significant"**). Two reasons:

1. **Circularity.** A site enters kinase K's leading edge because its motif scores
   high against K's position-weight matrix (the kinase library) *and* it moved in
   disease. So human_K and mouse_K leading edges both cluster around K's library
   consensus motif — high self-similarity is guaranteed by construction, not by
   cross-species biology. The z mostly measures "is K motif-distinctive within the
   60-kinase pool," a library property.
2. **Wrong question for uniqueness.** Matching substrates by motif similarity *hides*
   model-unique substrates: a substrate engaged only in human (gene X) still finds a
   high-BLOSUM match to some mouse site on gene Y (same kinase motif) and is
   miscounted as **shared**. Motif-keyed matching and "surface model-unique
   substrates" are mutually exclusive.

**What we want instead:** for each kinase, the substrate **set decomposition** —
shared / human-only / mouse-only — computed on **substrate identity (ortholog gene,
then site)**, with the actual substrate lists surfaced and exportable, plus direction
concordance on the shared set. BLOSUM keeps a role, but a subordinate one: within a
shared gene, it distinguishes *same-site* (motif conserved) from *different-residue-
same-protein*, and it supplies a graded descriptive similarity — it is **not** the
partition key.

This is a **replacement** (anti-shim): the null-enrichment path is deleted, not kept
behind a flag.

## Definitions

For kinase K, per context (tissue × age), with human profile H and mouse profile M
(each a `dict[motif_upper -> ProfileEntry]` carrying `gene`, `site_position`,
`direction`, `support`):

### Ortholog gene key
`gene_key(entry) = entry.gene.upper()`. Human HGNC symbols and mouse MGI symbols are
identical up to case for the ~1:1 orthologs of these substrates (APP↔App, MAPT↔Mapt,
…), so uppercased-symbol equality is the v1 ortholog match. (A curated ortholog
override map is a future refinement; note the limitation in the manifest, do not build
it now.)

### Gene-level partition
- `shared_genes`   = gene_keys present in **both** H and M.
- `human_only_genes` = gene_keys in H not in M.
- `mouse_only_genes` = gene_keys in M not in H.

### Site-level refinement (within `shared_genes` only)
For a gene g in both, take H's motif(s) on g and M's motif(s) on g. A human site
(g, m_h) is **same-site** if some mouse site (g, m_m) has
`motif_similarity(m_h, m_m).score >= SIM_FLOOR` (central-residue-type gated, as today);
otherwise it is **diff-site** (same protein, different residue engaged across species).
Report `n_shared_site` and `n_diffsite`.

### Coverage-aware uniqueness (the honesty guard — REQUIRED)
A human-only gene is only *biologically* model-unique if the mouse cohort could have
detected it. Split each unique set using the **measured-gene universe** of the other
cohort (distinct `gene_symbol` in that cohort's ST stoichiometry matrix for the
context):
- `human_only_engaged`    = human-only gene **is** in the mouse measured universe →
  detectable in mouse but not engaged by K → real model difference.
- `human_only_unmeasured` = human-only gene **not** in the mouse measured universe →
  assay coverage gap, **not** biology.
- symmetric for `mouse_only_engaged` / `mouse_only_unmeasured`.

### Direction concordance
On `shared_site` pairs only: fraction with `direction_h == direction_m`
(both +1 or both −1). NaN when no shared sites.

**Direction estimator must be symmetric across cohorts (option a).** Concordance is
only meaningful if human and mouse direction are estimated by the *same* rule and both
are independent of the leading-edge selection statistic. The human path already does
this (`_build_human_perdonor_profile`, lines 778–800): direction = majority sign of
per-donor `(donor − ctrl_mean)` polled over **all** AD donors at the site, not the
donors whose leading edge selected it — polling only the selectors is circular and
collapses to +1. The mouse path (`_build_fivexfad_profile`) does **not**: it sets
`direction = sign(stoich_lfc_{age})`, the same group statistic the leading edge is
ranked by, so mouse direction has near-zero within-kinase variance and re-reads the
selection criterion. Fix: recompute mouse direction with the identical estimator the
human path uses — per-replicate majority vote — so both sides use replicate-count
sign, distinct from the ranking statistic. The 5xFAD stoichiometry matrix carries
per-replicate columns named `{tissue}_{age}mo_{TG|WT}_{rep}` (verified: e.g.
`cortex_6mo_TG_16`, `cortex_6mo_WT_12`), so this is a direct mirror of the human path.
`stoich_lfc` is no longer read for direction (it may remain a magnitude descriptor if
wanted, but direction comes from the replicate vote).

### Graded similarity (secondary, descriptive — keep, do not headline)
Symmetric mean best-match BLOSUM similarity over the motif sets:
`sim(K) = ½·[ mean_a max_b blosum(a,b) + mean_b max_a blosum(a,b) ]` in [0,1].
Retain the existing best-match histogram (`sim_hist`) as a secondary descriptor.

## Metric outputs (per kinase × context, new `kinase_summary.csv` columns)

Drop: `enrich_z`, `p_emp`, `enrich`, `n_null`.
Keep: `kinase, tissue, age, context, direction_agree_frac, direction_corr,
human_support_min, human_support_median, sim_hist`.
Add:
- `n_shared_gene, n_human_only_gene, n_mouse_only_gene`
- `n_human_only_engaged, n_human_only_unmeasured`
- `n_mouse_only_engaged, n_mouse_only_unmeasured`
- `n_shared_site, n_diffsite`
- `overlap_frac_gene` = `n_shared_gene / (n_shared_gene + n_human_only_gene + n_mouse_only_gene)` (0 when denom 0)
- `blosum_similarity`  = the graded `sim(K)` above

## File-by-file changes

### 1. `alz/cross_reference/substrate_motif_compare.py`
- **Delete** `compute_null_enrichment` and the vectorized null helpers it exclusively
  serves (`_encode_motif_set`, `_sim_matrix`, and the `_ALPHA/_BMAT` block **only if**
  nothing else uses them — `_encode_motif`/`_BMAT` may be reused by the new graded
  similarity; keep what the new code needs, delete the rest). Grep for remaining
  references before deleting.
- **Add** `substrate_overlap(profile_a, profile_b, *, universe_a, universe_b,
  sim_floor=SIM_FLOOR) -> SubstrateOverlap` where `universe_a/b` are `set[str]` of
  uppercased measured gene symbols for each cohort/context. Returns a new dataclass:
  ```
  @dataclass
  class SubstrateOverlap:
      shared_sites: list[SharedSite]      # (gene, human motif/site/dir, mouse motif/site/dir, similarity, direction_agree)
      diff_site:    list[SharedSite]      # same gene, no motif-matched residue (per-side entries)
      human_only:   list[UniqueSub]       # gene, site, motif, direction, support, coverage ∈ {engaged, unmeasured}
      mouse_only:   list[UniqueSub]
      # + convenience counts and direction_agree_frac as properties
  ```
  Partition on `gene_key`, refine shared genes with `motif_similarity`, tag each unique
  entry's `coverage` via `universe_b`/`universe_a`. Reuse `motif_similarity` verbatim.
- **Add** `substrate_similarity(profile_a, profile_b) -> float` = the symmetric mean
  best-match in [0,1] (the graded descriptor). May reuse the existing per-position
  BLOSUM matrix path; keep it simple and correct over clever.
- **Add** cohort measured-universe loaders (DuckDB-streamed, `SELECT DISTINCT
  upper(gene_symbol)` from the ST stoichiometry matrix — human:
  `kinase_attribution_human/stoichiometry_matrix.csv`; 5xFAD:
  `kinase_attribution_5xfad/{tissue}_st_stoichiometry_matrix.csv`). Return `set[str]`.
  **Never** `read_csv`/pandas the matrix whole — DISTINCT in DuckDB only.
- Keep `motif_similarity`, `build_profile`, and the human/Song profile builders
  unchanged.
- **Rewrite the mouse direction estimator in `_build_fivexfad_profile` (option a).**
  Replace `direction=_sign(lfc)` (from `stoich_lfc_{age}` in the `ols` table) with a
  per-replicate majority vote, mirroring `_build_human_perdonor_profile` lines 785–800:
  for each leading-edge site, read the contrast age's TG and WT replicate columns from
  the `matrix` (`{tissue}_{age}mo_{TG|WT}_{rep}`), compute `wt_mean = nanmean(WT cols)`,
  then `direction = majority sign of (TG_i − wt_mean)` over the TG replicate columns
  (`+1` if `n_up > n_down`, `−1` if `n_down > n_up`, else `0`). Select the age's columns
  by parsing `age_contrast` (`TG_vs_WT_6mo` → `6mo`) and filtering matrix columns whose
  name contains `_{age}_TG_`/`_{age}_WT_`. The `ols`/`stoich_lfc` read is dropped from
  the direction path. This makes human and mouse direction the identical replicate-count
  estimator and removes the circularity where mouse direction re-read the enrichment
  ranking statistic.
- `compare()` (motif best-match decomposition) is **no longer used for partitioning**.
  If nothing else imports it after these changes, delete it and its CSV emit helpers;
  grep first (it may be used by other cross_reference scripts — if so, leave it).

### 2. `alz/cross_reference/c5_mukesh_5xfad.py`
- Replace the `compute_null_enrichment` import/call with `substrate_overlap` +
  `substrate_similarity` + the universe loaders.
- Load `universe_human` once; `universe_mouse[tissue]` once per tissue (age-invariant —
  the measured matrix is per tissue).
- `_summary_row` emits the new columns (above); drop the enrich_* columns. Direction
  concordance comes from the `SubstrateOverlap` shared-site set.
- `_emit_pairs_csv`: add a `partition` column per row ∈ {`shared_site`,
  `shared_gene_diffsite`, `human_only_engaged`, `human_only_unmeasured`,
  `mouse_only_engaged`, `mouse_only_unmeasured`} and a `coverage` column for unique
  rows. Keep gene/site/motif/direction/support/similarity fields. Replace the old
  `match_class` semantics with `partition`.
- MANIFEST: remove the `overlap_significance` (null-enrichment) block. Add an
  `overlap_definition` block documenting: gene-identity partition, site-level BLOSUM
  refinement (floor = SIM_FLOOR), coverage-aware uniqueness via measured-gene universe,
  the uppercase-symbol ortholog caveat, and direction concordance. Update the module
  docstring accordingly (no tombstones — rewrite, don't annotate).
- Headline sort: by `overlap_frac_gene` (desc), not `enrich_z`.

### 3. `alz/viewer/cohorts/substrate_compare.py`
- `_build_summary_block`: SELECT the new columns; drop enrich_*. Emit them in the block
  dict. Keep `contexts/tissues/ages/hist_*/tier*` scaffolding.
- `_write_pair_shards`: SELECT `partition, coverage` instead of `match_class`; keep
  gene/site/motif/direction/similarity/support. Shard schema bumps — set
  `schema_version` = 2 in `index.json`.
- Update the block's own `schema_version` to 2.

### 4. `alz/viewer/template/js/tabs/substrate_compare.js`
- **Master table:** replace the "Cortex z / Hippocampus z" enrichment heatmaps with
  **gene-overlap-fraction** mini-heatmaps (`overlap_frac_gene`, shaded 0→1, one cell
  per age, tissue tooltip = `shared / human-only / mouse-only` gene counts). Keep the
  Human/Mouse direction glyph columns as-is. Peak/sort keys switch from `peakZ` to
  peak `overlap_frac_gene`. Remove all `enrich_z / p_emp / enrich` references.
- **Detail — replace the "Overlap z (8 contexts)" sub-tab** with an "Overlap"
  sub-tab: per-context table of `overlap_frac_gene`, `n_shared_gene`,
  `n_human_only_gene` (engaged / unmeasured split), `n_mouse_only_gene` (engaged /
  unmeasured split), `n_shared_site`, `n_diffsite`, `direction_agree_frac`,
  `blosum_similarity`. No z/p columns.
- **Detail — "Substrate motifs" sub-tab** already renders Shared / Human-only /
  Mouse-only sections. Rewire them to the new `partition` field:
  - Shared = `partition == 'shared_site'` (add a `diff-site` sub-note or column for
    `shared_gene_diffsite`).
  - Human-only = `partition in {human_only_engaged, human_only_unmeasured}`, with a
    `coverage` badge (engaged = real model-unique; unmeasured = coverage gap, visually
    de-emphasized).
  - Mouse-only symmetric.
  - Section headers: "Human-only (engaged in human, not in mouse)" etc. — drop the old
    "no 5xFAD match ≥ 0.50" motif wording.
- Keep the best-match BLOSUM histogram as a secondary descriptor; relabel it "motif
  best-match distribution (descriptive)".
- Update `exportSubstrateCsv` columns to the new set (overlap_frac_gene, unique counts,
  direction) — drop peak_z.
- Update the tab preamble/help copy in `01_state.js` (or wherever the substrate tab
  guide lives) to describe overlap + model-unique substrates, not null-enrichment z.

### 5. `alz/viewer/template/body.html`
- Update the substrate toolbar: the "Min peak z" filter becomes "Min overlap"
  (`overlap_frac_gene` threshold) or is removed; the "Significant only" checkbox (based
  on p_emp) is removed (no p-value now). Update the two mini-heatmap column headers
  ("Cortex z" → "Cortex overlap", etc.) and their tooltips.

### 6. `alz/viewer/template/styles.css`
- Reuse the `.sub-zc` / `.sub-profile` / `.sub-age-ticks` machinery (already fixed to
  15px cells, legible ticks). The overlap-fraction heatmap uses the same green
  `--op` scale (0→1 fraction instead of |z|/4). Add a muted style for `coverage ==
  unmeasured` rows in the detail tables. No structural CSS rewrite needed.

## Re-run + rebuild (order matters)

1. **Regenerate the C5 run** (writes the new columns):
   ```
   systemd-run --user --scope -p MemoryMax=16G -p MemorySwapMax=0 \
     pixi run c5-mukesh-5xfad
   ```
   DuckDB-streamed; the universe DISTINCT queries and profile builds stay well under
   the cap. Confirm a new `outputs/reports/substrate_compare/c5_mukesh_5xfad_<ts>/`
   with the new `kinase_summary.csv` columns and the `partition`/`coverage` columns in
   `kinase_pairs_*.csv`.
2. **Rebuild the viewer payload + html** (the slice reads the new run; payload schema
   changes, so `--html` alone is insufficient — a full build is required):
   ```
   systemd-run --user --scope -p MemoryMax=24G -p MemorySwapMax=0 \
     pixi run viewer
   ```
   (or `pixi run python alz/build_unified_viewer.py --payload --html` under the same
   cap). Hard-refresh the viewer.

## Verification

- **Column sanity:** `n_shared_gene + n_human_only_gene + n_mouse_only_gene` equals the
  union gene count for a spot-checked kinase×context (DuckDB, not pandas whole-read).
- **Coverage split reconciles:** `n_human_only_engaged + n_human_only_unmeasured ==
  n_human_only_gene` (and mouse side).
- **A known shared substrate** (same gene both cohorts, e.g. a well-detected App/Mapt
  site for a relevant kinase) lands in `shared_site`, not human/mouse-only.
- **A human-only gene** absent from the mouse ST matrix is flagged `unmeasured`, not
  `engaged`.
- **Direction concordance** is computed over shared sites only (spot-check one kinase).
- **Direction estimator swap took effect (option a):** mouse direction is now a
  per-replicate vote, so it must no longer be one-signed within a kinase. Confirm ≥1
  kinase×context has **both** `+1` and `−1` mouse directions across its leading-edge
  sites (under the old `sign(stoich_lfc)` path they were near-uniformly one sign). If
  every kinase is still single-signed on the mouse side, the swap did not take —
  reproduce before reporting done.
- **JS:** `node --check` the edited tab + `01_state.js`; build exits 0; the payload's
  `substrate_compare` block carries the new keys and no `enrich_z`.
- Report counts (kinases with ≥1 engaged model-unique substrate; distribution of
  `overlap_frac_gene`) — do not tune anything to hit a number; report straight.

## Anti-shim removals (must be gone, not toggled)

- `compute_null_enrichment` and its exclusive helpers.
- `enrich_z / p_emp / enrich / n_null` in the runner, the summary CSV, the slice
  builder, the payload block, the tab, `body.html`, and the help copy.
- The manifest `overlap_significance` block and all "null enrichment" / "label-
  permutation" prose. Rewrite the docstrings; leave no "formerly z" pointers.

## Out of scope (do not build now)

- Curated ortholog table (uppercase-symbol match is v1; note the limitation).
- Restricting the coverage universe to K's candidate library substrates (v1 uses the
  cohort's full measured-gene universe; note this is a conservative "detectable at all"
  denominator).
- Per-donor human axis / any change to the human profile builder or the b-donorset.
- CSS structural rework beyond the coverage-muted style.
