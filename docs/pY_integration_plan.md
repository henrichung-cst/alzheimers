# Tyrosine Phosphoproteomics (pY) Integration Plan

## Goal

Plumb the Song pY phosphoproteomics through the live pipeline as a parallel track alongside the existing IMAC Ser/Thr phospho data, producing a Tyr-kinase activity layer and feeding it into unified cell-type attribution.

This is **net-new capability**, not reopening a closed path. The current pipeline silently excludes the entire Tyr kinome (Src family, RTKs, JAK, FAK, Abl, etc.) because IMAC enrichment yields ~0.5–2% pY recovery. Adding a dedicated pY track recovers that signal.

## Inputs (already on disk)

- `data/incytr_collections/song/primary/proteomics/song_pY_sitequant_merged_labeled (2).xlsx` — 1,471 pY sites × 6 plexes
- `data/incytr_collections/song/primary/proteomics/song_pY_compositeSites_merged_labeled (2).xlsx` — composite pY sites
- Parent-protein denominator: same `song2024_tmttotal_protein_quant_merged_labeled (2).xlsx` already loaded for the Ser/Thr track.
- Sample mapping: same `Sample_list_72mice (1).xlsx` (TMT channels are identical across IMAC and pY workbooks).

## Schema differences vs. IMAC sitequant

| Field | IMAC (S/T) | pY |
|---|---|---|
| `site_id` | present | absent — synthesize from `protein_id` + `site_position` |
| TMT channel cols | `p{N}_{channel}_sn_sum` | `plex{N}_{channel}_sn_sum` — rename on load |
| Sites | 16,114 | 1,471 |
| Extra cols | — | per-plex sequences, transcript_id (drop) |

The math, sample mapping, IRS normalization, and stoichiometry definition are identical.

## Design Decisions

1. **Two parallel MEA runs, not one unified call.** Tyr and Ser/Thr substrate libraries differ; ranking distributions and FDR control must be computed independently. The kinase-library `kinase_type='Y'` query handles this.
2. **Stoichiometry math unchanged**: `log2(pY_site) − log2(parent_protein)`. The parent denominator is residue-agnostic.
3. **OLS design unchanged**: same 9 factorial contrasts, same `ANALYSIS_MODE` (males-only primary, full-cohort sensitivity), same outlier exclusion list.
4. **Attribution evidence weights unchanged**: SEA-AD / WMB / Song concordance are gene-level and residue-agnostic; a Tyr kinase gets attributed the same way as a Ser/Thr kinase.
5. **Outputs are parallel files**, not new columns in existing files. This keeps the Ser/Thr-only deliverable stable for downstream consumers and makes residue-stratified diagnostics easy.
6. **Composite sites deferred**: implement sitequant first; composite pY can follow the same adapter pattern once sitequant is verified.

## Implementation Phases

### Phase 1 — Adapter + ingestion (data_ingest.py)

- Add `PY_SITEQUANT_FILE`, `PY_COMPOSITE_FILE` constants alongside `IMAC_SITEQUANT_FILE`.
- Add a small loader helper that:
  - reads the pY xlsx with `header=1`,
  - renames `plex{N}_*` → `p{N}_*`,
  - synthesizes `site_id = f"{protein_id}_{site_position}"`,
  - drops sequence/transcript-only columns.
- Extend `--phospho-match` to report match rate for pY against the bulk proteome (expect lower than 91.7% given fewer sites and pY-biased proteins).
- Add a residue-type sanity check: confirm all `motif` central residues are Y for the pY file and S/T for the IMAC file.

### Phase 2 — Normalization + stoichiometry (kinase_attribution.py Stage 1)

- Mirror the IMAC sitequant IRS normalization on the pY matrix, using the same plex bridge channels.
- Compute `stoichiometry_matrix_pY.csv` and `raw_phospho_normalized_pY.csv` alongside the existing files.
- Verify: log-transform behavior on a sparser matrix (pY has more missingness); decide whether to tighten `MIN_NONZERO_PER_PLEX` or accept higher dropout.

### Phase 3 — Factorial OLS (Stage 2)

- Run the same factorial OLS on the pY stoichiometry matrix.
- Output `site_level_ols_pY.csv` with the 9 contrast LFC/FDR columns.
- Apply the same outlier-exclusion sample list and `ANALYSIS_MODE` filter.

### Phase 4 — MEA on Tyr kinome

- Pull Tyr kinome from kinase-library: `kinase_type='Y'` (or equivalent).
- Median-center + winsorize (1st/99th) the pY β values per contrast, identical to the S/T track.
- Run GSEA pre-ranked against Tyr substrate sets.
- Output `mea_stoichiometry_pY.csv`, `mea_global_shift_pY.csv`, `winsorized_sites_pY.csv`.
- FDR threshold: same 0.25 as S/T track unless ranking distribution diagnostics suggest otherwise.

### Phase 5 — Unified attribution

- Concatenate Tyr and Ser/Thr MEA outputs into the unified attribution input (kinase symbol is the join key; residue type tracked as a column).
- Re-run `kinase_attribution.py --attribute` against the combined kinase set.
- Verify subclass coverage: WMB and Song expression matrices are gene-level, so Tyr kinases are already covered.

### Phase 6 — Recovery / hypothesis tables

- Extend `attribution_recovery.py` to include Tyr kinases in the kinase activity matrix and hypothesis table.
- Add a `residue_type` column (`ST` / `Y`) to `kinase_hypothesis_table.csv`.
- Verify `bubble_plots/` rendering handles the larger kinase set; consider faceting by residue type.

### Phase 7 — Viewer + supplementary

- `build_unified_viewer.py`: surface pY kinases with a residue-type filter or facet.
- Update supplementary diagnostics (`fdr_stringent`, `threshold_sensitivity`, etc.) to run residue-stratified.

## Testing & Validation

- **Match-rate floor**: pY parent-protein match rate ≥ 70% (sanity floor; IMAC achieves 91.7% on more abundant S/T sites).
- **Residue purity**: `motif` column central residue is Y for ≥99% of pY rows.
- **Top-kinase plausibility**: expect canonical Tyr kinases to surface in at least one disease contrast — Src family (Src, Fyn, Lyn), RTKs (EGFR, INSR, MET), focal adhesion (FAK/PTK2), JAK/STAT axis. If none surface, investigate ranking or substrate-set coverage before accepting results.
- **Stoichiometry sanity**: per-site stoichiometry distribution should be roughly centered after IRS; large global shifts are a flag.
- **Cross-track consistency**: Tyr kinases that share substrates with S/T-promiscuous kinases (e.g., dual-specificity DYRKs, GSK3) should show coherent direction across tracks.

## Risks & Open Questions

- **Sparser matrix**: 1,471 pY sites vs. 16,114 IMAC sites means MEA has fewer ranked features. Substrate sets may have low overlap with detected sites for some Tyr kinases → low statistical power. Document per-kinase site-coverage in MEA output.
- **Parent-protein denominator coverage**: Tyr-phospho proteins (RTKs, signaling adapters) may be lower-abundance and missing from the bulk TMT proteome more often. Sites without a parent measurement fall back to raw phospho or are dropped — decide policy and document.
- **Kinase-library Tyr coverage**: confirm the installed kinase-library version (1.7.0) ships Tyr substrate sets with sufficient depth. If not, this is a hard blocker — escalate before Phase 4.
- **Composite pY**: skipped in Phase 1; revisit if sitequant alone underpowers MEA.
- **Closed-paths check**: this plan does not reopen direct deconvolution, factor model, two-compartment, or transcript-only rescue. It extends the existing stoichiometry+MEA+attribution architecture to a new substrate set.

## Out of Scope

- Other PTMs (ubiquitin, acetyl, glyco) — no data, no plan.
- Phosphatase activity inference — substrate libraries are not symmetric; not enabled for Tyr or S/T tracks.
- Re-running closed paths.

## Acceptance Criteria

A successful pY integration produces, end-to-end via `pixi run dual`:

1. `stoichiometry_matrix_pY.csv` and `site_level_ols_pY.csv` under `outputs/reports/kinase_attribution/`.
2. `mea_stoichiometry_pY.csv` with at least one Tyr kinase passing FDR 0.25 in at least one contrast.
3. Unified attribution and `kinase_hypothesis_table.csv` containing both `ST` and `Y` rows with a `residue_type` column.
4. Viewer renders Tyr kinases with attribution and is filterable by residue type.
5. Supplementary diagnostics run cleanly on the combined kinase set.

## Estimated Scope

- Phases 1–4: ~1 working session (adapter + ingestion + stoichiometry + OLS + Tyr MEA).
- Phases 5–6: ~half a session (attribution, recovery, hypothesis tables — mostly parameter extension).
- Phase 7: ~half a session (viewer + diagnostics).

Total: ~2 working sessions assuming no kinase-library Tyr coverage surprises.
