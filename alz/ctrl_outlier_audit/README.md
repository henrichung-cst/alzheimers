# `alz/ctrl_outlier_audit/` — human CTRL-outlier investigation

Self-contained investigation (2026-05-25) establishing that three human controls —
**CTRL-07, CTRL-08, CTRL-10** — carry a genuine pre-symptomatic-AD-like phospho-omic
signature and must be excluded from the clean-control baseline. **Clean controls =
CTRL-01/02/03/04.** Split out of `alz/cross_reference/` on 2026-05-29.

Plan / findings: `docs/plans/human_ctrl_outlier_audit_2026-05-25.md` (+ `_findings`).
Investigation outputs: `outputs/reports/kinase_attribution_human/ctrl_audit/`.

## File inventory

| File | Role | Driver |
|---|---|---|
| `ctrl_outlier_audit.py` | Phases A–C: sample structure, artifact controls, site-level attribution. Read-only on derived matrices. | one-off |
| `ctrl_outlier_audit_kinases.py` | Phase D: per-kinase leading-edge proof — the AD-like NES traces to substrate-site LFC, identical in suspect controls and AD, opposite in clean controls. | one-off |
| `ctrl_outlier_audit_report_figs.py` | Meeting figure: raw-phosphosite heatmap. Driver sites chosen from AD + clean controls only; suspect controls held out of selection (genuine validation, not built in). | one-off |
| `ctrl_outlier_suspect_lfc_table.py` | Per-site LFC table (suspect-vs-clean alongside AD-vs-clean), annotated with HPA secretome location. Generates `suspect_vs_ad_lfc_*.csv`. | one-off |
| `concordance_overlap_AD_excl_01_03.py` | Single, self-contained generator for the audit's operational output: the base-gated boolean overlap of kinases where AD (excl. AD-01/AD-03 from the per-donor vote) and the suspect controls **CTRL-08/CTRL-10** agree and the clean controls oppose. Computes the group AD-vs-CLEAN / suspect-vs-CLEAN GSEA inline from the stoichiometry matrices, then applies the per-donor agreement vote. Writes one folder, `ctrl_audit/concordance_AD8_excl01_03/`, with `overlap_AD8_sus_clean.csv`, `substrates_leading_edge.csv`, and `MANIFEST.md`. | `pixi run concordance-overlap-ad8-excl01-03` |

## Notes

- **Sign convention** is pipeline-wide `+ = up in disease/suspect direction`.
- `SUSPECT = {CTRL-08, CTRL-10}`, `CLEAN_CTRL = {CTRL-01..04}` (the fixed clean baseline), and
  `EXCLUDED_AD = {AD-01, AD-03}` are defined in `concordance_overlap_AD_excl_01_03.py`.
  **CTRL-07 is excluded ENTIRELY** — it is NOT a suspect and, because CLEAN is fixed (NOT "all CTRL
  minus suspect"), it is never reclassified into the clean baseline (the audit verdict is that
  CTRL-07 is genuinely AD-like). The MANIFEST stamps it in an "EXCLUDED (in neither pool)" row.
- **Provenance rule:** the boolean-overlap result stamps its full sample membership (which AD /
  suspect / clean samples filled each role) into its `MANIFEST.md` and per row. The group MEA
  runs over all 10 AD samples (direction + significance base gate); AD-01/AD-03 are excluded
  ONLY from the per-donor agreement vote. Never read an overlap result without its MANIFEST.
- **`substrates_leading_edge.csv`** records, per overlap kinase, the GSEA **leading-edge**
  substrate sites it shares in **both** group contrasts — the sites in the kinase's leading edge
  in the AD-vs-clean AND the suspect-vs-clean enrichment, i.e. the substrates concordantly driving
  it in AD and in the suspect controls. (The leading edge is the subset of a kinase's
  motif-matched substrates lying at/before the running-sum peak — the sites that drive the NES; it
  differs between the two contrasts because each ranks sites by its own LFC, so this file is the
  intersection.) Each shared site carries its `lfc_AD` and `lfc_suspect`; `kl_percentile` is single
  (motif-vs-kinase match strength, contrast-independent). Substrate identity is the motif (±7 aa
  window) with the central phospho-residue **lowercased** (kinase-library convention, e.g.
  `IRANRADsEEEGTVE`) so the acceptor site is explicit; one motif can map to several phosphosites,
  and every matching site is emitted. (Internally the join back to the matrix is on the uppercased
  motif, since the matrix stores the window all-uppercase.) Substrate-set membership is gated on the
  kinase library's per-site rank (`kl_thresh` = 15 for `st`, 7 for `py`; `percentile_rank` method,
  lower rank = better), NOT on the `kl_percentile` column (a 0–100 match-strength for inspection).
- The former multi-contrast `human_group_mea_reanalysis.py` + `ctrl_outlier_concordance_tiers.py`
  pipeline and its `reanalysis_mea/` outputs were removed on 2026-06-12 — the 4-contrast set and
  the 64-combo sweep had no consumer; the single generator above computes the one needed overlap
  directly from inputs. The standalone `human_group_mea.py` was archived to
  `archive/human_group_mea_2026-05-29/` on 2026-05-29.
- Cross-reference attribution against external atlases (SEA-AD/WMB/Song) lives in the
  sibling [`alz/cross_reference/`](../cross_reference/README.md).
