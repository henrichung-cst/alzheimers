"""Calibration audit for cohort-level snRNA cross-check.

Reads the existing per-animal MEA output and the snRNA specificity table,
computes the (wmb_class, contrast) binomial sign-concordance distribution
and the kinase mean_expression distribution, and writes a one-page
markdown report with proposed thresholds for COHORT_FDR_THRESH and
EXPR_PRESENCE_FLOOR.

Usage:
    python -m code.deconvolution.cohort_concordance_audit
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(CODE_DIR)
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, REPO_ROOT)

import numpy as np
import pandas as pd

from alz.shared import config
from deconvolution import paths
from deconvolution.cohort_concordance import compute_cohort_concordance


def _df_to_md(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    lines = ["| " + " | ".join(df.columns) + " |",
             "|" + "|".join(["---"] * len(df.columns)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for v in row.values:
            if isinstance(v, (float, np.floating)):
                cells.append("nan" if not np.isfinite(v) else float_fmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)

PA_DIR = os.path.join(paths.OUTPUT_DIR, "per_animal")
PA_PRIMARY = os.path.join(PA_DIR, "kinase_enrichment_wmb.csv")
REPORT_PATH = os.path.join(PA_DIR, "cohort_concordance_calibration.md")


def main():
    mea = pd.read_csv(PA_PRIMARY)
    spec = pd.read_csv(config.SONG_EXPRESSION_FILE)
    print(f"MEA rows: {len(mea):,}")
    print(f"Specificity rows: {len(spec):,}")

    sig = mea[(mea["FDR"] < paths.DECON_FDR_THRESH)
              & mea["kinase_gene_LFC_snRNA"].notna()
              & mea["NES"].notna()].copy()
    print(f"Bulk-sig rows with snRNA LFC: {len(sig):,}")

    cohort = compute_cohort_concordance(mea)

    n_strata = len(cohort)
    n_above_half = int((cohort["frac_match"] > 0.5).sum())
    n_pass_05 = int((cohort["cohort_fdr"] < 0.05).sum())
    n_pass_10 = int((cohort["cohort_fdr"] < 0.10).sum())
    n_pass_25 = int((cohort["cohort_fdr"] < 0.25).sum())
    median_frac = float(cohort["frac_match"].median())

    spec_lookup = spec.set_index(["cell_type", "gene_symbol"])["mean_expression"]
    sig_pres = sig.copy()
    sig_pres["mean_expression"] = sig_pres.set_index(
        ["wmb_class", "gene_symbol"]).index.map(spec_lookup)

    n_with_spec = int(sig_pres["mean_expression"].notna().sum())
    n_no_spec = int(sig_pres["mean_expression"].isna().sum())
    nz_expr = sig_pres.loc[sig_pres["mean_expression"] > 0, "mean_expression"]
    quantiles = nz_expr.quantile([0.05, 0.10, 0.25, 0.50, 0.75]).to_dict() \
        if len(nz_expr) else {}

    floors_to_test = [0.05, 0.10, 0.25, 0.50, 1.00]
    floor_rows = []
    for f in floors_to_test:
        n_below = int(((sig_pres["mean_expression"].fillna(0)) < f).sum())
        floor_rows.append({
            "floor_log2cpm": f,
            "n_below_floor": n_below,
            "frac_of_sig": n_below / len(sig_pres) if len(sig_pres) else np.nan,
        })
    floor_tbl = pd.DataFrame(floor_rows)

    md = ["# Cohort-concordance + presence calibration audit\n",
          "**Source:** `outputs/reports/deconvolution/per_animal/"
          "kinase_enrichment_wmb.csv`\n",
          f"- Bulk-significant rows (FDR<{paths.DECON_FDR_THRESH}) with "
          f"finite snRNA LFC: **{len(sig):,}**\n",
          "## 1. Cohort sign-concordance (binomial vs 0.5 null)\n",
          f"- Strata (wmb_class × contrast) with ≥5 rows: **{n_strata}**",
          f"- Strata with frac_match > 0.5: **{n_above_half} / {n_strata}**",
          f"- Strata passing cohort_fdr < 0.05: **{n_pass_05}**",
          f"- Strata passing cohort_fdr < 0.10: **{n_pass_10}**",
          f"- Strata passing cohort_fdr < 0.25: **{n_pass_25}**",
          f"- Median frac_match across strata: **{median_frac:.3f}**\n",
          "### Per-stratum table (top 20 by frac_match)\n",
          _df_to_md(cohort.sort_values("frac_match", ascending=False).head(20)),
          "\n## 2. Expression presence (log2(CPM+1) in snRNA pseudobulk)\n",
          f"- Sig-rows with a (wmb_class, gene) match in specificity: "
          f"**{n_with_spec:,}**",
          f"- Sig-rows missing from specificity (`expressed=False`): "
          f"**{n_no_spec:,}**\n",
          "### Quantiles of nonzero mean_expression (kinase × wmb_class)\n",
          "| q | mean_expression |", "|---|---|"]
    for q, v in quantiles.items():
        md.append(f"| {q:.2f} | {v:.3f} |")
    md.append("\n### Sig-rows below floor at candidate `EXPR_PRESENCE_FLOOR` values\n")
    md.append(_df_to_md(floor_tbl))

    if n_pass_05 >= max(3, n_strata // 10):
        cohort_thresh = "0.05"
        cohort_note = "FDR<0.05 yields enough concordant strata to be useful."
    elif n_pass_10 >= max(3, n_strata // 10):
        cohort_thresh = "0.10"
        cohort_note = ("FDR<0.05 too strict (insufficient strata); use 0.10 "
                       "as the cohort gate.")
    else:
        cohort_thresh = "0.25"
        cohort_note = ("Even FDR<0.10 fails. Use 0.25 as the cohort gate; "
                       "treat cohort_concordant rows with caution.")

    if len(nz_expr) and nz_expr.quantile(0.10) >= 0.10:
        floor_choice = "0.10"
    else:
        q05 = float(nz_expr.quantile(0.05)) if len(nz_expr) else 0.0
        floor_choice = f"{max(0.05, q05):.2f}"

    md.append("\n## Recommended thresholds\n")
    md.append(f"- `COHORT_FDR_THRESH = {cohort_thresh}` — {cohort_note}")
    md.append(f"- `EXPR_PRESENCE_FLOOR = {floor_choice}` "
              "(log2(CPM+1); rows below this have `expressed=False`).")
    md.append("\nReview the floor table above; pick the most conservative "
              "value that retains a meaningful number of expressed rows.")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(str(x) for x in md))
    print(f"Wrote {REPORT_PATH}")
    print(f"  cohort: {n_pass_05}/{n_strata} pass FDR<0.05, "
          f"{n_pass_10} pass FDR<0.10")
    print(f"  presence: {n_no_spec} sig rows missing from specificity")


if __name__ == "__main__":
    main()
