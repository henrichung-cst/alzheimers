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
| `human_group_mea_reanalysis.py` | Clean-baseline human group MEA: four labeled contrasts on the same 17 brains. `mea_AD_vs_cleanCTRL.csv` (AD vs clean CTRL-01..04) is the canonical clean-baseline human group MEA. | `pixi run mea-suspect-reanalysis` |

## Notes

- **Sign convention** is pipeline-wide `+ = up in disease/suspect direction`.
- `AD_LIKE_CTRL = {CTRL-07, CTRL-08, CTRL-10}` is defined in
  `human_group_mea_reanalysis.py`; it is the operational output of this audit.
- The former standalone `human_group_mea.py` (whose "production" clean-CTRL CSV was never
  run by a task or read by any consumer) was archived to
  `archive/human_group_mea_2026-05-29/` on 2026-05-29; the reanalysis's
  `mea_AD_vs_cleanCTRL.csv` is the live clean-baseline output, produced by the same method.
- Cross-reference attribution against external atlases (SEA-AD/WMB/Song) lives in the
  sibling [`alz/cross_reference/`](../cross_reference/README.md).
